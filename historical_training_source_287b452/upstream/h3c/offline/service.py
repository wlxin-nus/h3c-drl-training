"""Public dry-plan, execution, resume, verification, and export services."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from h3c.offline.artifacts import (
    OfflineArtifactError,
    OfflineWorkspace,
    endpoint_identity,
    read_jsonl,
)
from h3c.offline.contracts import OfflineContractError, OnboardingSpec, load_onboarding_spec
from h3c.offline.framework_workflow import (
    ReviewRequest,
    ReviewResponse,
    StartOnboarding,
    WorkflowFinished,
    build_workflow,
)
from h3c.offline.model import MicrosoftOfflineAgents, framework_versions, secret_identity
from h3c.offline.verification import export_workspace, finalize_workspace, verify_workspace

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


def resolved_plan(spec: OnboardingSpec) -> dict[str, Any]:
    return {
        "mode": "dry_plan",
        "workflow": "offline_mapping_and_hitl_causal_discovery",
        "case_id": spec.case_id,
        "spec_identity": spec.identity,
        "workspace_root": str(spec.repository_root / "outputs" / "offline" / spec.case_id),
        "stages": [
            "semantic_mapping_agent",
            "mapping_human_review",
            "deterministic_profile_merge",
            "causal_discovery_agent",
            "causal_human_review",
            "confirm_graph",
        ],
        "provider": {
            "kind": "openai_compatible",
            "endpoint_env": spec.provider.endpoint_env,
            "api_key_env": spec.provider.api_key_env,
            "model_env": spec.provider.model_env,
            "reasoning_effort": "low",
            "temperature_absent": True,
            "top_p_absent": True,
            "automatic_retries": 0,
        },
        "maximum_model_calls": {
            "semantic_mapping": spec.mapping_model_calls,
            "causal_discovery": spec.causal_model_calls,
            "total": spec.mapping_model_calls + spec.causal_model_calls,
        },
        "will_create_workspace": False,
        "will_call_model": False,
    }


def _valid_review_command(command: str) -> bool:
    normalized = command.strip().lower()
    return normalized in {"approve", "abort"} or (
        normalized.startswith("revise ") and bool(command.strip()[7:].strip())
    )


async def _checkpoint_identity(storage: Any) -> str:
    checkpoint = await storage.get_latest(workflow_name="h3c-offline-onboarding-v1")
    if checkpoint is None:
        raise OfflineArtifactError("Agent Framework did not persist a workflow checkpoint")
    return str(checkpoint.checkpoint_id)


async def _run_stream(
    *,
    workflow: Any,
    storage: Any,
    workspace: OfflineWorkspace,
    reviewer: str,
    initial_message: StartOnboarding | None,
    checkpoint_id: str | None,
    input_function: InputFunction,
    output_function: OutputFunction,
) -> WorkflowFinished:
    next_message: StartOnboarding | None = initial_message
    next_checkpoint = checkpoint_id
    responses: dict[str, ReviewResponse] | None = None
    while True:
        stream = workflow.run(
            next_message,
            stream=True,
            checkpoint_id=next_checkpoint,
            responses=responses,
        )
        pending: tuple[str, ReviewRequest] | None = None
        finished: WorkflowFinished | None = None
        try:
            async for event in stream:
                event_type = str(event.type)
                if event_type == "request_info":
                    if not isinstance(event.data, ReviewRequest) or not isinstance(
                        event.request_id, str
                    ):
                        raise OfflineArtifactError(
                            "Agent Framework emitted an invalid review request"
                        )
                    if pending is not None:
                        raise OfflineArtifactError(
                            "offline workflow emitted concurrent human requests"
                        )
                    pending = (event.request_id, event.data)
                elif event_type == "output" and isinstance(event.data, WorkflowFinished):
                    finished = event.data
        except Exception:
            latest_checkpoint = await storage.get_latest(workflow_name="h3c-offline-onboarding-v1")
            if latest_checkpoint is not None:
                workspace.update_state(latest_framework_checkpoint=latest_checkpoint.checkpoint_id)
            raise
        latest = await _checkpoint_identity(storage)
        workspace.update_state(latest_framework_checkpoint=latest)
        if finished is not None:
            return finished
        if pending is None:
            raise OfflineArtifactError("offline workflow stopped without output or a human request")
        request_id, request = pending
        output_function(request.display)
        while True:
            command = input_function(
                f"[{request.stage} round {request.round}] approve | revise <feedback> | abort: "
            )
            if _valid_review_command(command):
                break
            output_function("Invalid reply. Use `approve`, `revise <feedback>`, or `abort`.")
        next_message = None
        next_checkpoint = None
        responses = {request_id: ReviewResponse(reviewer=reviewer, command=command)}


def _run_async(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise OfflineArtifactError("offline CLI cannot be nested inside an active asyncio event loop")


def discover(
    spec_path: Path,
    repository_root: Path,
    *,
    reviewer: str,
    input_function: InputFunction = input,
    output_function: OutputFunction = print,
) -> dict[str, Any]:
    if not reviewer.strip() or reviewer != reviewer.strip():
        raise OfflineContractError("--reviewer must be a non-empty trimmed name")
    spec = load_onboarding_spec(spec_path, repository_root)
    agents, endpoint, model, api_key, key_identity = MicrosoftOfflineAgents.from_environment(spec)
    try:
        workspace = OfflineWorkspace.create(
            spec,
            framework_versions=framework_versions(),
            endpoint=endpoint,
            model=model,
            secret_identity=key_identity,
        )
        output_function(f"Offline workspace: {workspace.path}")
        workflow, storage = build_workflow(workspace, agents)
        result = _run_async(
            _run_stream(
                workflow=workflow,
                storage=storage,
                workspace=workspace,
                reviewer=reviewer,
                initial_message=StartOnboarding(str(workspace.path)),
                checkpoint_id=None,
                input_function=input_function,
                output_function=output_function,
            )
        )
        if result.status == "approved":
            verification = finalize_workspace(workspace.path, api_key=api_key)
            return {
                "status": "complete",
                "workspace": str(workspace.path),
                "verification": verification,
            }
        return {"status": "aborted", "workspace": str(workspace.path), "message": result.message}
    finally:
        _run_async(agents.close())


def _resume_identity(
    workspace: OfflineWorkspace, spec: OnboardingSpec, endpoint: str, model: str, api_key: str
) -> None:
    manifest = workspace.manifest()
    provider = manifest.get("provider", {})
    if (
        manifest.get("spec_identity") != spec.identity
        or provider.get("endpoint_identity") != endpoint_identity(endpoint)
        or provider.get("model") != model
        or provider.get("secret_identity") != secret_identity(api_key)
    ):
        raise OfflineArtifactError(
            "resume provider or workflow identity differs from the checkpoint"
        )
    state = workspace.state()
    if (
        state.get("stage") in {"approved", "aborted"}
        or (workspace.path / "completion.json").exists()
    ):
        raise OfflineArtifactError("completed or aborted offline workspaces cannot be resumed")
    calls_path = workspace.path / "model_calls.jsonl"
    if calls_path.exists():
        calls = read_jsonl(calls_path)
        if calls and calls[-1].get("status") == "failed" and calls[-1].get("resumable") is not True:
            raise OfflineArtifactError(
                "the last model failure is not an approved network interruption"
            )


def resume(
    workspace_path: Path,
    *,
    reviewer: str,
    input_function: InputFunction = input,
    output_function: OutputFunction = print,
) -> dict[str, Any]:
    if not reviewer.strip() or reviewer != reviewer.strip():
        raise OfflineContractError("--reviewer must be a non-empty trimmed name")
    workspace = OfflineWorkspace.open(workspace_path)
    spec = load_onboarding_spec(workspace.path / "resolved_spec.json", workspace.path.parents[3])
    agents, endpoint, model, api_key, _ = MicrosoftOfflineAgents.from_environment(spec)
    try:
        _resume_identity(workspace, spec, endpoint, model, api_key)
        workflow, storage = build_workflow(workspace, agents)
        checkpoint = workspace.state().get("latest_framework_checkpoint")
        if not isinstance(checkpoint, str):
            latest = _run_async(storage.get_latest(workflow_name="h3c-offline-onboarding-v1"))
            if latest is None:
                raise OfflineArtifactError("offline workspace has no resumable checkpoint")
            checkpoint = latest.checkpoint_id
        result = _run_async(
            _run_stream(
                workflow=workflow,
                storage=storage,
                workspace=workspace,
                reviewer=reviewer,
                initial_message=None,
                checkpoint_id=checkpoint,
                input_function=input_function,
                output_function=output_function,
            )
        )
        if result.status == "approved":
            verification = finalize_workspace(workspace.path, api_key=api_key)
            return {
                "status": "complete",
                "workspace": str(workspace.path),
                "verification": verification,
            }
        return {"status": "aborted", "workspace": str(workspace.path), "message": result.message}
    finally:
        _run_async(agents.close())


__all__ = [
    "discover",
    "export_workspace",
    "load_onboarding_spec",
    "resolved_plan",
    "resume",
    "verify_workspace",
]
