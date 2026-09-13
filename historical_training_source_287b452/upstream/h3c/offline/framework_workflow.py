"""Microsoft Agent Framework workflow for offline onboarding and real HITL pauses."""

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

from h3c.causal.workflow import confirm_graph
from h3c.offline.artifacts import OfflineWorkspace, read_jsonl
from h3c.offline.contracts import (
    OnboardingSpec,
    bind_edge_provenance,
    load_onboarding_spec,
    mapping_warnings,
    merge_profile_template,
    object_identity,
    read_text_document,
    standard_variables,
    validate_causal_proposal,
    validate_mapping,
)
from h3c.offline.model import OfflineAgentPair
from h3c.offline.prompts import (
    CAUSAL_SYSTEM_PROMPT,
    MAPPING_SYSTEM_PROMPT,
    render_causal_user_prompt,
    render_mapping_user_prompt,
)

WORKFLOW_NAME = "h3c-offline-onboarding-v1"


@dataclass(frozen=True)
class StartOnboarding:
    workspace_path: str


@dataclass(frozen=True)
class MappingRevision:
    workspace_path: str
    feedback: str


@dataclass(frozen=True)
class MappingDraft:
    workspace_path: str
    proposal_identity: str
    round: int


@dataclass(frozen=True)
class MappingApproved:
    workspace_path: str
    mapping_identity: str


@dataclass(frozen=True)
class CausalRevision:
    workspace_path: str
    feedback: str


@dataclass(frozen=True)
class CausalDraft:
    workspace_path: str
    proposal_identity: str
    round: int


@dataclass(frozen=True)
class CausalApproved:
    workspace_path: str
    proposal_identity: str
    reviewer: str
    round: int


@dataclass(frozen=True)
class ReviewRequest:
    stage: str
    round: int
    proposal_identity: str
    display: str


@dataclass(frozen=True)
class ReviewResponse:
    reviewer: str
    command: str


@dataclass(frozen=True)
class WorkflowFinished:
    status: str
    workspace_path: str
    message: str


CHECKPOINT_TYPES = [
    f"{__name__}:{name}"
    for name in (
        "StartOnboarding",
        "MappingRevision",
        "MappingDraft",
        "MappingApproved",
        "CausalRevision",
        "CausalDraft",
        "CausalApproved",
        "ReviewRequest",
        "ReviewResponse",
        "WorkflowFinished",
    )
]


def _root(workspace: OfflineWorkspace) -> Path:
    try:
        return workspace.path.parents[3]
    except IndexError as error:
        raise RuntimeError("offline workspace is not below outputs/offline/<case>") from error


def _spec(workspace: OfflineWorkspace) -> OnboardingSpec:
    return load_onboarding_spec(workspace.path / "resolved_spec.json", _root(workspace))


def _rows(workspace: OfflineWorkspace, name: str) -> list[dict[str, Any]]:
    path = workspace.path / name
    return [] if not path.exists() else read_jsonl(path)


def _inventory(spec: OnboardingSpec) -> list[str] | None:
    if spec.point_inventory is None:
        return None
    raw = json.loads(spec.point_inventory.read_text(encoding="utf-8"))
    return list(raw["points"] if isinstance(raw, dict) else raw)


def _call_sequence(workspace: OfflineWorkspace) -> int:
    return len(_rows(workspace, "model_calls.jsonl")) + 1


def _role_call_count(workspace: OfflineWorkspace, role: str) -> int:
    return sum(row.get("role") == role for row in _rows(workspace, "model_calls.jsonl"))


def _parse_response(response: ReviewResponse) -> tuple[str, str | None]:
    if not response.reviewer.strip() or response.reviewer != response.reviewer.strip():
        raise ValueError("reviewer must be a non-empty trimmed name")
    command = response.command.strip()
    lowered = command.lower()
    if lowered == "approve":
        return "approve", None
    if lowered == "abort":
        return "abort", None
    if lowered.startswith("revise ") and command[7:].strip():
        return "revise", command[7:].strip()
    raise ValueError("review reply must be `approve`, `revise <feedback>`, or `abort`")


def _review_event(
    *, stage: str, draft: MappingDraft | CausalDraft, response: ReviewResponse
) -> dict[str, Any]:
    decision, feedback = _parse_response(response)
    return {
        "artifact_schema": "h3c_offline_review_event",
        "schema_version": 1,
        "stage": stage,
        "round": draft.round,
        "proposal_identity": draft.proposal_identity,
        "reviewer": response.reviewer,
        "decision": decision,
        "feedback": feedback,
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
    }


def _mapping_display(row: dict[str, Any]) -> str:
    warnings = row["warnings"]
    warning_text = "none" if not warnings else "\n".join(f"- {item}" for item in warnings)
    return (
        "MAPPING PROPOSAL\n"
        f"Identity: {row['proposal_identity']}\n"
        f"Heuristic warnings:\n{warning_text}\n\n"
        "Structured mapping:\n"
        f"{json.dumps(row['proposal'], ensure_ascii=False, sort_keys=True, indent=2)}\n\n"
        "Merged case-profile candidate:\n"
        f"{json.dumps(row['profile_candidate'], ensure_ascii=False, sort_keys=True, indent=2)}"
    )


def _causal_display(row: dict[str, Any]) -> str:
    arrows = []
    for edge in row["proposal"]["edges"]:
        tags = ", ".join(edge["tags"])
        sources = ", ".join(edge["evidence_source_ids"])
        arrows.append(
            f"- {edge['source']} --[{edge['relation']}]--> {edge['target']} "
            f"[{tags}] | evidence: {sources}"
        )
    return (
        "CAUSAL GRAPH PROPOSAL\n"
        f"Identity: {row['proposal_identity']}\n\n"
        "Readable arrows:\n"
        + "\n".join(arrows)
        + "\n\nStructured edge table and raw JSON:\n"
        + json.dumps(row["proposal"], ensure_ascii=False, sort_keys=True, indent=2)
    )


def _proposal_row(
    workspace: OfflineWorkspace, name: str, proposal_identity: str, round_number: int
) -> dict[str, Any]:
    matching = [
        row
        for row in _rows(workspace, name)
        if row.get("proposal_identity") == proposal_identity and row.get("round") == round_number
    ]
    if len(matching) != 1:
        raise RuntimeError("review request does not resolve to exactly one stored proposal")
    return matching[0]


def _prompt_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_workflow(workspace: OfflineWorkspace, agents: OfflineAgentPair) -> tuple[Any, Any]:
    """Build the typed Agent Framework workflow and file checkpoint storage lazily."""

    try:
        from agent_framework import (
            Executor,
            FileCheckpointStorage,
            WorkflowBuilder,
            WorkflowContext,
            handler,
            response_handler,
        )
    except ImportError as error:
        raise RuntimeError(
            "offline dependencies are missing; run `uv sync --extra offline`"
        ) from error
    # Handler annotations are resolved by Agent Framework against module globals.
    # The import remains lazy so base H3C installations never import the optional package.
    globals()["WorkflowContext"] = WorkflowContext

    class MappingExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="semantic_mapping_agent")

        async def _run(
            self,
            workspace_path: str,
            feedback: str | None,
            ctx: Any,
        ) -> None:
            current = OfflineWorkspace.open(Path(workspace_path))
            spec = _spec(current)
            prior_rows = _rows(current, "mapping_proposals.jsonl")
            round_number = len(prior_rows) + 1
            if _role_call_count(current, "semantic_mapping") >= spec.mapping_model_calls:
                raise RuntimeError("Mapping model generation limit was reached")
            previous = prior_rows[-1]["proposal"] if prior_rows else None
            document = read_text_document(spec.building_document)
            user = render_mapping_user_prompt(
                spec,
                document,
                _inventory(spec),
                previous_proposal=previous,
                reviewer_feedback=feedback,
            )
            generation = await agents.generate_mapping(
                MAPPING_SYSTEM_PROMPT, user, _call_sequence(current), current
            )
            proposal = validate_mapping(generation.parsed, spec)
            profile = merge_profile_template(spec, proposal)
            identity = object_identity(proposal)
            current.append(
                "mapping_proposals.jsonl",
                {
                    "artifact_schema": "h3c_mapping_proposal",
                    "schema_version": 1,
                    "round": round_number,
                    "proposal_identity": identity,
                    "proposal": proposal,
                    "profile_candidate": profile,
                    "warnings": mapping_warnings(proposal),
                    "system_prompt_sha256": _prompt_hash(MAPPING_SYSTEM_PROMPT),
                    "user_prompt_sha256": _prompt_hash(user),
                },
            )
            current.update_state(stage="mapping_review", mapping_round=round_number)
            await ctx.send_message(MappingDraft(workspace_path, identity, round_number))

        @handler
        async def start(self, message: StartOnboarding, ctx: WorkflowContext[MappingDraft]) -> None:
            await self._run(message.workspace_path, None, ctx)

        @handler
        async def revise(
            self, message: MappingRevision, ctx: WorkflowContext[MappingDraft]
        ) -> None:
            await self._run(message.workspace_path, message.feedback, ctx)

    class MappingReviewExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="mapping_human_review")

        @handler
        async def review(
            self,
            draft: MappingDraft,
            ctx: WorkflowContext[MappingApproved | MappingRevision, WorkflowFinished],
        ) -> None:
            current = OfflineWorkspace.open(Path(draft.workspace_path))
            row = _proposal_row(
                current, "mapping_proposals.jsonl", draft.proposal_identity, draft.round
            )
            await ctx.request_info(
                ReviewRequest(
                    "mapping", draft.round, draft.proposal_identity, _mapping_display(row)
                ),
                ReviewResponse,
                request_id=f"mapping-{draft.round}-{draft.proposal_identity[:12]}",
            )

        @response_handler
        async def respond(
            self,
            request: ReviewRequest,
            response: ReviewResponse,
            ctx: WorkflowContext[MappingApproved | MappingRevision, WorkflowFinished],
        ) -> None:
            if request.stage != "mapping":
                raise ValueError("mapping review received a request for another stage")
            current = workspace
            draft = MappingDraft(str(current.path), request.proposal_identity, request.round)
            event = _review_event(stage="mapping", draft=draft, response=response)
            current.append("mapping_review_events.jsonl", event)
            row = _proposal_row(
                current, "mapping_proposals.jsonl", request.proposal_identity, request.round
            )
            if event["decision"] == "approve":
                current.publish("confirmed_mapping.json", row["proposal"])
                current.publish("case_profile_candidate.json", row["profile_candidate"])
                current.update_state(stage="causal_generation")
                await ctx.send_message(
                    MappingApproved(str(current.path), request.proposal_identity)
                )
            elif event["decision"] == "revise":
                current.update_state(stage="mapping_revision")
                await ctx.send_message(MappingRevision(str(current.path), event["feedback"]))
            else:
                current.update_state(
                    stage="aborted", aborted=True, abort_reason="human aborted Mapping review"
                )
                await ctx.yield_output(
                    WorkflowFinished("aborted", str(current.path), "Mapping review aborted")
                )

    class CausalExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="causal_discovery_agent")

        async def _run(self, workspace_path: str, feedback: str | None, ctx: Any) -> None:
            current = OfflineWorkspace.open(Path(workspace_path))
            spec = _spec(current)
            mapping = current.path.joinpath("confirmed_mapping.json")
            confirmed_mapping = json.loads(mapping.read_text(encoding="utf-8"))
            variables = standard_variables(confirmed_mapping)
            prior_rows = _rows(current, "causal_proposals.jsonl")
            round_number = len(prior_rows) + 1
            if _role_call_count(current, "causal_discovery") >= spec.causal_model_calls:
                raise RuntimeError("Causal Discovery model generation limit was reached")
            evidence = {
                source.identifier: read_text_document(source.path)
                for source in spec.evidence_sources
            }
            previous = prior_rows[-1]["proposal"] if prior_rows else None
            user = render_causal_user_prompt(
                spec,
                variables,
                evidence,
                previous_proposal=previous,
                reviewer_feedback=feedback,
            )
            generation = await agents.generate_causal(
                CAUSAL_SYSTEM_PROMPT, user, _call_sequence(current), current
            )
            validated = validate_causal_proposal(generation.parsed, spec, confirmed_mapping)
            identity = object_identity(validated["raw"])
            current.append(
                "causal_proposals.jsonl",
                {
                    "artifact_schema": "h3c_causal_proposal",
                    "schema_version": 1,
                    "round": round_number,
                    "proposal_identity": identity,
                    "proposal": validated["raw"],
                    "graph_proposal": validated["graph_proposal"],
                    "edge_provenance": validated["edge_provenance"],
                    "standard_variables": variables,
                    "system_prompt_sha256": _prompt_hash(CAUSAL_SYSTEM_PROMPT),
                    "user_prompt_sha256": _prompt_hash(user),
                },
            )
            current.update_state(stage="causal_review", causal_round=round_number)
            await ctx.send_message(CausalDraft(workspace_path, identity, round_number))

        @handler
        async def start(self, message: MappingApproved, ctx: WorkflowContext[CausalDraft]) -> None:
            await self._run(message.workspace_path, None, ctx)

        @handler
        async def revise(self, message: CausalRevision, ctx: WorkflowContext[CausalDraft]) -> None:
            await self._run(message.workspace_path, message.feedback, ctx)

    class CausalReviewExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="causal_human_review")

        @handler
        async def review(
            self,
            draft: CausalDraft,
            ctx: WorkflowContext[CausalApproved | CausalRevision, WorkflowFinished],
        ) -> None:
            current = OfflineWorkspace.open(Path(draft.workspace_path))
            row = _proposal_row(
                current, "causal_proposals.jsonl", draft.proposal_identity, draft.round
            )
            await ctx.request_info(
                ReviewRequest("causal", draft.round, draft.proposal_identity, _causal_display(row)),
                ReviewResponse,
                request_id=f"causal-{draft.round}-{draft.proposal_identity[:12]}",
            )

        @response_handler
        async def respond(
            self,
            request: ReviewRequest,
            response: ReviewResponse,
            ctx: WorkflowContext[CausalApproved | CausalRevision, WorkflowFinished],
        ) -> None:
            if request.stage != "causal":
                raise ValueError("causal review received a request for another stage")
            current = workspace
            draft = CausalDraft(str(current.path), request.proposal_identity, request.round)
            event = _review_event(stage="causal", draft=draft, response=response)
            current.append("causal_review_events.jsonl", event)
            if event["decision"] == "approve":
                current.update_state(stage="confirmation")
                await ctx.send_message(
                    CausalApproved(
                        str(current.path),
                        request.proposal_identity,
                        response.reviewer,
                        request.round,
                    )
                )
            elif event["decision"] == "revise":
                current.update_state(stage="causal_revision")
                await ctx.send_message(CausalRevision(str(current.path), event["feedback"]))
            else:
                current.update_state(
                    stage="aborted", aborted=True, abort_reason="human aborted causal review"
                )
                await ctx.yield_output(
                    WorkflowFinished("aborted", str(current.path), "Causal review aborted")
                )

    class ConfirmationExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="causal_graph_confirmation")

        @handler
        async def confirm(
            self,
            message: CausalApproved,
            ctx: WorkflowContext[WorkflowFinished, WorkflowFinished],
        ) -> None:
            current = OfflineWorkspace.open(Path(message.workspace_path))
            row = _proposal_row(
                current, "causal_proposals.jsonl", message.proposal_identity, message.round
            )
            graph = confirm_graph(
                row["graph_proposal"], reviewer=message.reviewer, date=date.today().isoformat()
            ).resolved()
            current.publish("confirmed_causal_graph.json", graph)
            feedback_events = _rows(current, "causal_review_events.jsonl")
            provenance_edges = bind_edge_provenance(
                graph["edges"],
                row["edge_provenance"],
                confirmed_round=message.round,
                human_feedback=[
                    event["feedback"] for event in feedback_events if event["feedback"] is not None
                ],
            )
            current.publish(
                "causal_provenance.json",
                {
                    "provenance_schema": "h3c_causal_provenance",
                    "schema_version": 1,
                    "case_id": graph["profile"],
                    "graph_identity": object_identity(graph),
                    "reviewer": message.reviewer,
                    "confirmation_date": graph["confirmation"]["date"],
                    "sources": [
                        source["id"] for source in _spec(current).public()["evidence_sources"]
                    ],
                    "edges": provenance_edges,
                },
            )
            current.update_state(stage="approved")
            await ctx.yield_output(
                WorkflowFinished("approved", str(current.path), "Both human reviews approved")
            )

    mapping = MappingExecutor()
    mapping_review = MappingReviewExecutor()
    causal = CausalExecutor()
    causal_review = CausalReviewExecutor()
    confirmation = ConfirmationExecutor()
    storage = FileCheckpointStorage(
        workspace.checkpoints / "framework",
        allowed_checkpoint_types=CHECKPOINT_TYPES,
    )
    builder = WorkflowBuilder(
        start_executor=mapping,
        checkpoint_storage=storage,
        max_iterations=100,
        name=WORKFLOW_NAME,
        output_from=[mapping_review, causal_review, confirmation],
    )
    builder.add_edge(mapping, mapping_review)
    builder.add_edge(
        mapping_review, mapping, condition=lambda item: isinstance(item, MappingRevision)
    )
    builder.add_edge(
        mapping_review, causal, condition=lambda item: isinstance(item, MappingApproved)
    )
    builder.add_edge(causal, causal_review)
    builder.add_edge(causal_review, causal, condition=lambda item: isinstance(item, CausalRevision))
    builder.add_edge(
        causal_review, confirmation, condition=lambda item: isinstance(item, CausalApproved)
    )
    return builder.build(), storage
