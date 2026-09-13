"""Independent verification, completion publication, and safe export."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from h3c.causal.graph import validate_graph
from h3c.causal.workflow import confirm_graph
from h3c.experiments.profiles import validate_profile
from h3c.offline.artifacts import (
    OfflineArtifactError,
    OfflineWorkspace,
    read_json,
    read_jsonl,
    secret_occurrences,
    source_file_records,
    write_atomic_json,
    write_new_json,
)
from h3c.offline.contracts import (
    OfflineContractError,
    OnboardingSpec,
    bind_edge_provenance,
    file_identity,
    load_onboarding_spec,
    mapping_warnings,
    merge_profile_template,
    object_identity,
    standard_variables,
    validate_causal_proposal,
    validate_mapping,
)
from h3c.offline.model import validate_wire_request

REVIEW_FIELDS = {
    "artifact_schema",
    "schema_version",
    "stage",
    "round",
    "proposal_identity",
    "reviewer",
    "decision",
    "feedback",
    "reviewed_at_utc",
}
RAW_MODEL_FIELDS = {
    "artifact_schema",
    "schema_version",
    "role",
    "sequence",
    "wire_request",
    "wire_request_identity",
    "response_text",
    "response_identity",
}
MODEL_CALL_COMMON_FIELDS = {
    "artifact_schema",
    "schema_version",
    "role",
    "sequence",
    "model",
    "endpoint_identity",
    "framework_versions",
    "system_prompt_sha256",
    "user_prompt_sha256",
    "reasoning_effort",
    "temperature_absent",
    "top_p_absent",
    "wire_request_identity",
    "latency_seconds",
    "state_checkpoint_before",
    "state_checkpoint_after",
    "status",
}
CONFIRMED_MODEL_CALL_FIELDS = MODEL_CALL_COMMON_FIELDS | {"response_identity", "usage"}
FAILED_MODEL_CALL_FIELDS = MODEL_CALL_COMMON_FIELDS | {
    "wire_request",
    "failure_category",
    "resumable",
    "error_type",
    "error",
}


def _root(workspace: OfflineWorkspace) -> Path:
    return workspace.path.parents[3]


def _spec(workspace: OfflineWorkspace) -> OnboardingSpec:
    return load_onboarding_spec(workspace.path / "resolved_spec.json", _root(workspace))


def _rows(workspace: OfflineWorkspace, name: str) -> list[dict[str, Any]]:
    path = workspace.path / name
    return [] if not path.exists() else read_jsonl(path)


def _check_review_events(
    rows: list[dict[str, Any]], *, stage: str, proposals: list[dict[str, Any]]
) -> tuple[bool, dict[str, Any] | None]:
    if not rows or len(rows) > len(proposals):
        return False, None
    approved: dict[str, Any] | None = None
    for index, row in enumerate(rows):
        if set(row) != REVIEW_FIELDS:
            return False, None
        if (
            row["artifact_schema"] != "h3c_offline_review_event"
            or row["schema_version"] != 1
            or row["stage"] != stage
            or row["round"] != index + 1
            or row["proposal_identity"] != proposals[index]["proposal_identity"]
            or not isinstance(row["reviewer"], str)
            or not row["reviewer"].strip()
            or row["reviewer"] != row["reviewer"].strip()
            or row["decision"] not in {"approve", "revise", "abort"}
        ):
            return False, None
        if row["decision"] == "revise":
            if not isinstance(row["feedback"], str) or not row["feedback"].strip():
                return False, None
        elif row["feedback"] is not None:
            return False, None
        if row["decision"] in {"approve", "abort"}:
            if index != len(rows) - 1:
                return False, None
            approved = row if row["decision"] == "approve" else None
    return True, approved


def _verify_sources(spec: OnboardingSpec, manifest: dict[str, Any]) -> bool:
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list):
        return False
    root = spec.repository_root.resolve()
    try:
        onboarding_rows = [
            row
            for row in source_files
            if isinstance(row, dict) and row.get("role") == "onboarding_spec"
        ]
        if len(onboarding_rows) != 1:
            return False
        onboarding = onboarding_rows[0]
        if set(onboarding) != {"role", "path", "sha256"}:
            return False
        source_path = (root / onboarding["path"]).resolve()
        if (
            not source_path.is_relative_to(root)
            or not source_path.is_file()
            or file_identity(source_path) != onboarding["sha256"]
            or load_onboarding_spec(source_path, root).identity != spec.identity
        ):
            return False
        expected = [onboarding, *source_file_records(spec, include_onboarding_spec=False)]
        return source_files == expected
    except (KeyError, OSError, TypeError):
        return False


def _verify_model_rows(workspace: OfflineWorkspace, manifest: dict[str, Any]) -> bool:
    calls = _rows(workspace, "model_calls.jsonl")
    raw_rows = _rows(workspace, "raw_model_io.jsonl")
    if not calls or [row.get("sequence") for row in calls] != list(range(1, len(calls) + 1)):
        return False
    if any(set(row) != RAW_MODEL_FIELDS for row in raw_rows):
        return False
    raw_by_sequence: dict[int, dict[str, Any]] = {}
    raw_sequence_order: list[int] = []
    for row in raw_rows:
        raw_sequence = row.get("sequence")
        if (
            isinstance(raw_sequence, bool)
            or not isinstance(raw_sequence, int)
            or raw_sequence in raw_by_sequence
        ):
            return False
        raw_by_sequence[raw_sequence] = row
        raw_sequence_order.append(raw_sequence)
    if raw_sequence_order != sorted(raw_sequence_order):
        return False
    provider = manifest.get("provider", {})
    role_counts = {"semantic_mapping": 0, "causal_discovery": 0}
    confirmed_count = 0
    for sequence, call in enumerate(calls, start=1):
        role = call.get("role")
        status = call.get("status")
        expected_fields = (
            CONFIRMED_MODEL_CALL_FIELDS
            if status == "confirmed_response"
            else FAILED_MODEL_CALL_FIELDS
        )
        if (
            role not in role_counts
            or status not in {"confirmed_response", "failed"}
            or set(call) != expected_fields
            or call.get("artifact_schema") != "h3c_offline_model_call"
            or call.get("schema_version") != 1
            or call.get("sequence") != sequence
            or call.get("model") != provider.get("model")
            or call.get("endpoint_identity") != provider.get("endpoint_identity")
            or call.get("framework_versions") != manifest.get("framework_versions")
            or call.get("reasoning_effort") != "low"
            or call.get("temperature_absent") is not True
            or call.get("top_p_absent") is not True
            or not isinstance(call.get("system_prompt_sha256"), str)
            or len(call["system_prompt_sha256"]) != 64
            or not isinstance(call.get("user_prompt_sha256"), str)
            or len(call["user_prompt_sha256"]) != 64
            or isinstance(call.get("latency_seconds"), bool)
            or not isinstance(call.get("latency_seconds"), (int, float))
            or call["latency_seconds"] < 0
            or not isinstance(call.get("state_checkpoint_before"), str)
        ):
            return False
        role_counts[role] += 1
        raw = raw_by_sequence.get(sequence)
        if status == "confirmed_response":
            confirmed_count += 1
            if raw is None or call.get("state_checkpoint_after") is None:
                return False
            wire = raw.get("wire_request")
        else:
            if (
                raw is not None
                or call.get("failure_category") != "network_interruption"
                or call.get("resumable") is not True
                or call.get("state_checkpoint_after") is not None
                or not isinstance(call.get("error_type"), str)
                or not call["error_type"]
                or not isinstance(call.get("error"), str)
                or not call["error"]
            ):
                return False
            wire = call.get("wire_request")
        try:
            wire = validate_wire_request(wire)
        except (OfflineContractError, TypeError):
            return False
        messages = wire.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            return False
        system_messages = [row for row in messages if row.get("role") == "system"]
        user_messages = [row for row in messages if row.get("role") == "user"]
        if len(system_messages) != 1 or len(user_messages) != 1:
            return False
        system_content = system_messages[0].get("content")
        user_content = user_messages[0].get("content")
        if not isinstance(system_content, str) or not isinstance(user_content, str):
            return False
        if (
            wire.get("model") != provider.get("model")
            or call.get("wire_request_identity") != object_identity(wire)
            or call.get("system_prompt_sha256")
            != hashlib.sha256(system_content.encode("utf-8")).hexdigest()
            or call.get("user_prompt_sha256")
            != hashlib.sha256(user_content.encode("utf-8")).hexdigest()
        ):
            return False
        if status == "confirmed_response" and (
            raw is None
            or raw.get("artifact_schema") != "h3c_offline_raw_model_io"
            or raw.get("schema_version") != 1
            or raw.get("sequence") != sequence
            or raw.get("role") != role
            or raw.get("wire_request_identity") != call.get("wire_request_identity")
            or call.get("response_identity") != raw.get("response_identity")
            or object_identity(json.loads(raw["response_text"])) != raw.get("response_identity")
        ):
            return False
    if confirmed_count != len(raw_rows):
        return False
    limits = manifest.get("model_call_limits", {})
    mapping_limit = limits.get("mapping")
    causal_limit = limits.get("causal_discovery")
    return (
        isinstance(mapping_limit, int)
        and isinstance(causal_limit, int)
        and role_counts["semantic_mapping"] <= mapping_limit
        and role_counts["causal_discovery"] <= causal_limit
    )


def _expected_provenance(
    graph: dict[str, Any],
    causal_row: dict[str, Any],
    review_rows: list[dict[str, Any]],
    reviewer: str,
) -> dict[str, Any]:
    return {
        "provenance_schema": "h3c_causal_provenance",
        "schema_version": 1,
        "case_id": graph["profile"],
        "graph_identity": object_identity(graph),
        "reviewer": reviewer,
        "confirmation_date": graph["confirmation"]["date"],
        "sources": list(graph["sources"]),
        "edges": bind_edge_provenance(
            graph["edges"],
            causal_row["edge_provenance"],
            confirmed_round=causal_row["round"],
            human_feedback=[
                event["feedback"] for event in review_rows if event["feedback"] is not None
            ],
        ),
    }


def verify_workspace(workspace_path: Path, *, require_completion: bool = True) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        workspace = OfflineWorkspace.open(workspace_path)
        spec = _spec(workspace)
        manifest = workspace.manifest()
        state = workspace.state()
        checks["spec_identity"] = manifest.get("spec_identity") == spec.identity
        checks["workflow_identity"] = state.get("workflow_identity") == manifest.get(
            "workflow_identity"
        )
        checks["source_files"] = _verify_sources(spec, manifest)
        provider = manifest.get("provider", {})
        checks["thinking_contract"] = (
            provider.get("reasoning_effort") == "low"
            and provider.get("temperature_absent") is True
            and provider.get("top_p_absent") is True
            and provider.get("automatic_retries") == 0
        )
        checks["secret_scan"] = (
            manifest.get("secret_scan_status") == "completed"
            and manifest.get("secret_exposure_count") == 0
        )
        checks["model_calls"] = _verify_model_rows(workspace, manifest)

        mapping_rows = _rows(workspace, "mapping_proposals.jsonl")
        mapping_valid = bool(mapping_rows)
        for index, row in enumerate(mapping_rows, start=1):
            try:
                proposal = validate_mapping(row["proposal"], spec)
                candidate = merge_profile_template(spec, proposal)
                mapping_valid = mapping_valid and (
                    row.get("artifact_schema") == "h3c_mapping_proposal"
                    and row.get("schema_version") == 1
                    and row.get("round") == index
                    and row.get("proposal_identity") == object_identity(proposal)
                    and row.get("profile_candidate") == candidate
                    and row.get("warnings") == mapping_warnings(proposal)
                )
            except (KeyError, ValueError, TypeError):
                mapping_valid = False
        checks["mapping_proposals"] = mapping_valid
        mapping_reviews = _rows(workspace, "mapping_review_events.jsonl")
        mapping_review_valid, mapping_approval = _check_review_events(
            mapping_reviews, stage="mapping", proposals=mapping_rows
        )
        checks["mapping_review_events"] = mapping_review_valid and mapping_approval is not None
        if mapping_approval is None:
            raise OfflineArtifactError("Mapping has not been approved")
        approved_mapping_row = mapping_rows[mapping_approval["round"] - 1]
        checks["mapping_approval_binding"] = (
            approved_mapping_row["proposal_identity"] == mapping_approval["proposal_identity"]
        )
        checks["confirmed_mapping"] = (
            read_json(workspace.path / "confirmed_mapping.json") == approved_mapping_row["proposal"]
        )
        checks["profile_candidate"] = (
            read_json(workspace.path / "case_profile_candidate.json")
            == approved_mapping_row["profile_candidate"]
        )

        confirmed_mapping = approved_mapping_row["proposal"]
        causal_rows = _rows(workspace, "causal_proposals.jsonl")
        causal_valid = bool(causal_rows)
        validated_rows: list[dict[str, Any]] = []
        for index, row in enumerate(causal_rows, start=1):
            try:
                validated = validate_causal_proposal(row["proposal"], spec, confirmed_mapping)
                validated_rows.append(validated)
                causal_valid = causal_valid and (
                    row.get("artifact_schema") == "h3c_causal_proposal"
                    and row.get("schema_version") == 1
                    and row.get("round") == index
                    and row.get("proposal_identity") == object_identity(validated["raw"])
                    and row.get("graph_proposal") == validated["graph_proposal"]
                    and row.get("edge_provenance") == validated["edge_provenance"]
                    and row.get("standard_variables") == standard_variables(confirmed_mapping)
                )
            except (KeyError, ValueError, TypeError):
                causal_valid = False
        checks["causal_proposals"] = causal_valid
        causal_reviews = _rows(workspace, "causal_review_events.jsonl")
        causal_review_valid, causal_approval = _check_review_events(
            causal_reviews, stage="causal", proposals=causal_rows
        )
        checks["causal_review_events"] = causal_review_valid and causal_approval is not None
        if causal_approval is None:
            raise OfflineArtifactError("causal graph has not been approved")
        approved_index = causal_approval["round"] - 1
        approved_causal_row = causal_rows[approved_index]
        checks["causal_approval_binding"] = (
            approved_causal_row["proposal_identity"] == causal_approval["proposal_identity"]
        )
        graph = confirm_graph(
            validated_rows[approved_index]["graph_proposal"],
            reviewer=causal_approval["reviewer"],
            date=read_json(workspace.path / "confirmed_causal_graph.json")["confirmation"]["date"],
        ).resolved()
        recorded_graph = read_json(workspace.path / "confirmed_causal_graph.json")
        checks["confirmed_graph"] = graph == recorded_graph
        recorded_provenance = read_json(workspace.path / "causal_provenance.json")
        checks["causal_provenance"] = recorded_provenance == _expected_provenance(
            graph, approved_causal_row, causal_reviews, causal_approval["reviewer"]
        )
        checks["application_state"] = (
            state.get("stage") == "approved"
            and state.get("aborted") is False
            and state.get("mapping_round") == len(mapping_rows)
            and state.get("causal_round") == len(causal_rows)
        )
        checks["checkpoint_storage"] = any(
            path.is_file() for path in (workspace.checkpoints / "framework").rglob("*")
        ) and isinstance(state.get("latest_framework_checkpoint"), str)
        if require_completion:
            completion = read_json(workspace.path / "completion.json")
            recorded_verification = read_json(workspace.path / "verification.json")
            checks["recorded_verification"] = recorded_verification.get("passed") is True and all(
                recorded_verification.get("checks", {}).values()
            )
            checks["completion"] = (
                completion.get("artifact_schema") == "h3c_offline_completion"
                and completion.get("schema_version") == 1
                and completion.get("status") == "complete"
                and completion.get("workflow_identity") == manifest.get("workflow_identity")
                and completion.get("verification_identity")
                == object_identity(recorded_verification)
                and (workspace.path / "completion.json").stat().st_mtime_ns
                >= max(
                    path.stat().st_mtime_ns
                    for path in workspace.path.rglob("*")
                    if path.is_file() and path.name != "completion.json"
                )
            )
    except Exception as error:
        errors.append(str(error))
    passed = not errors and bool(checks) and all(checks.values())
    return {
        "verification_schema": "h3c_offline_verification",
        "schema_version": 1,
        "passed": passed,
        "checks": checks,
        "errors": errors,
    }


def finalize_workspace(workspace_path: Path, *, api_key: str) -> dict[str, Any]:
    workspace = OfflineWorkspace.open(workspace_path)
    manifest = workspace.manifest()
    first_scan = secret_occurrences(workspace.path, api_key)
    manifest["secret_scan_status"] = "completed"
    manifest["secret_exposure_count"] = first_scan
    workspace.replace_manifest(manifest)
    if first_scan:
        raise OfflineArtifactError("secret scan found API key material in the workspace")
    verification = verify_workspace(workspace.path, require_completion=False)
    if not verification["passed"]:
        raise OfflineArtifactError(
            f"offline workspace verification failed: {verification['errors']}"
        )
    workspace.publish("verification.json", verification)
    if secret_occurrences(workspace.path, api_key):
        raise OfflineArtifactError("final secret scan found API key material in the workspace")
    completion = {
        "artifact_schema": "h3c_offline_completion",
        "schema_version": 1,
        "status": "complete",
        "workflow_identity": manifest["workflow_identity"],
        "verification_identity": object_identity(verification),
    }
    write_atomic_json(workspace.path / "completion.json", completion)
    final = verify_workspace(workspace.path, require_completion=True)
    if not final["passed"]:
        raise OfflineArtifactError("published offline completion does not independently verify")
    return final


def export_workspace(
    workspace_path: Path,
    *,
    case_profile: Path,
    graph: Path,
    provenance: Path,
) -> dict[str, str]:
    verification = verify_workspace(workspace_path, require_completion=True)
    if not verification["passed"]:
        raise OfflineArtifactError("only a complete verified workspace can be exported")
    workspace = OfflineWorkspace.open(workspace_path)
    root = _root(workspace).resolve()
    targets = {
        "case_profile": case_profile.resolve(),
        "graph": graph.resolve(),
        "provenance": provenance.resolve(),
    }
    if len(set(targets.values())) != 3:
        raise OfflineArtifactError("export targets must be distinct")
    for name, target in targets.items():
        if not target.is_relative_to(root):
            raise OfflineArtifactError(f"{name} export target is outside the repository")
        if target.exists():
            raise OfflineArtifactError(f"refusing to overwrite export target: {target}")
    graph_value = read_json(workspace.path / "confirmed_causal_graph.json")
    validate_graph(graph_value)
    provenance_value = read_json(workspace.path / "causal_provenance.json")
    profile = copy.deepcopy(read_json(workspace.path / "case_profile_candidate.json"))
    profile["graph"] = targets["graph"].relative_to(root).as_posix()
    validate_profile(profile, root=root, allow_missing_graph=True)
    program = (root / profile["program"]).resolve()
    if not program.is_relative_to(root) or not program.is_file():
        raise OfflineArtifactError("exported profile program reference is invalid")
    write_new_json(targets["graph"], graph_value)
    write_new_json(targets["provenance"], provenance_value)
    write_new_json(targets["case_profile"], profile)
    return {name: str(target) for name, target in targets.items()}
