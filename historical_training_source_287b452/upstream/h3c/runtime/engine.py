"""Strictly serial H3C physical execution engine."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

from h3c.agents.contracts import DEFAULT_PER_ZONE_RESERVED_CAP_C
from h3c.agents.roles import (
    Executor,
    ModelCallContext,
    ModelClient,
    ModelContractError,
    Orchestrator,
    Reflector,
)
from h3c.causal.graph import ConfirmedGraph, derive_variant, load_graph
from h3c.control.budget import (
    BudgetLedger,
    allocation_fallback_audit,
    site_cap_max,
    validated_fallback_allocation,
)
from h3c.control.program import load_program, program_hash
from h3c.control.program_execution import build_program_observations, execute_zone_programs
from h3c.control.validation import validate_candidate
from h3c.experiments.matrix import RunPlan
from h3c.experiments.profiles import load_profile, repository_root
from h3c.experiments.settings import load_model_provider_contract, load_runtime_contract
from h3c.memory.caol import (
    ReflectorResolution,
    active_experiences,
    agent_visible_number,
    apply_memory_operations,
    attach_hourly_lessons,
    build_hourly_cao,
    empty_regime_store,
    reflector_slot_view,
    select_caol_working_memory,
    validate_memory_refs,
)
from h3c.memory.ledger import ProgramLedger
from h3c.memory.working import (
    completed_executor_records,
    completed_summary_frame,
)
from h3c.outputs.artifacts import RunArtifacts
from h3c.outputs.metrics import compute_run_metrics
from h3c.outputs.verification import verify_run
from h3c.runtime.clients import (
    BoptestHttpClient,
    OpenAICompatibleModelClient,
    TransportError,
)
from h3c.runtime.comfort import MetricsAccumulator, comfort_headroom, step_reward_breakdown
from h3c.runtime.execution_lock import physical_execution_lock
from h3c.runtime.occupancy import effective_count, hourly_route
from h3c.runtime.protocol import (
    EvaluationBoundaryState,
    PhysicalClient,
    build_forecast_evidence,
    control_input,
    forecast_points,
    initialize_evaluation_boundary,
    require_time,
    resolve_forecast_missing_occupancy,
    site_power,
    zone_temperature_c,
)
from h3c.runtime.resume import ResumePrefix, load_resume_prefix
from h3c.runtime.source_identity import committed_source_identity
from h3c.runtime.weather import weather_condition_inputs, weather_view

PhysicalFactory = Callable[[str], PhysicalClient]
ModelFactory = Callable[[RunArtifacts, str], ModelClient]


class RunAcceptanceFailure(RuntimeError):
    def __init__(self, run_dir: Path, verification: Mapping[str, Any]) -> None:
        super().__init__(f"run failed pre-completion verification: {verification['errors']}")
        self.run_dir = run_dir
        self.verification = dict(verification)


class ExecutorBatchTransportError(TransportError):
    def __init__(self, failures: Sequence[tuple[str, TransportError]]) -> None:
        if not failures:
            raise ValueError("executor transport batch requires at least one failure")
        primary_zone, primary = failures[0]
        super().__init__(
            f"Executor transport batch failed; primary zone {primary_zone}: {primary}",
            retryable=primary.retryable,
            error_type=primary.error_type,
            provider_response_received=primary.provider_response_received,
            retry_after_seconds=primary.retry_after_seconds,
        )
        self.primary_zone = primary_zone
        self.failed_zones = tuple(zone for zone, _ in failures)
        self.failure_count = len(failures)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _replay_equal(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _replay_equal(left[key], right[key], tolerance=tolerance) for key in left
        )
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(
            _replay_equal(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _source_commit() -> str:
    return committed_source_identity()


def _resolved_plan(plan: RunPlan) -> tuple[dict[str, Any], ConfirmedGraph | None]:
    root = repository_root()
    profile = load_profile(plan.profile)
    method = plan.method_config()
    resolved: dict[str, Any] = {"case_profile": profile, "method": method}
    graph: ConfirmedGraph | None = None
    if plan.causal_enabled:
        graph = load_graph(root / profile["graph"])
        if graph.profile != profile["profile"] or graph.zones != tuple(profile["zones"]):
            raise ValueError("confirmed graph identity does not match the case profile")
        if plan.graph_mutation is not None:
            graph = derive_variant(graph, plan.graph_mutation)
        resolved["resolved_graph"] = graph.resolved()
        if plan.graph_mutation is not None:
            resolved["graph_mutation"] = copy.deepcopy(plan.graph_mutation)
    if plan.controller == "h3c_agent":
        resolved["model_provider"] = plan.effective_model_provider()
    return resolved, graph


def _endpoint_identity(endpoint: str) -> str:
    return hashlib.sha256(endpoint.rstrip("/").encode("utf-8")).hexdigest()


def _secret_occurrences(directory: Path, secret: str) -> int:
    if not secret:
        return 0
    needle = secret.encode("utf-8")
    return sum(
        path.read_bytes().count(needle)
        for path in directory.iterdir()
        if path.is_file() and path.name != ".execution.lock"
    )


def _real_physical_factory(endpoint: str) -> PhysicalClient:
    return BoptestHttpClient(endpoint)


def _call_ordinal(
    plan: RunPlan,
    zones: Sequence[str],
    *,
    hour: int,
    role: str,
    zone: str | None = None,
) -> int:
    calls_per_hour = len(zones) + 1 + int(plan.coordination_enabled)
    ordinal = hour * calls_per_hour
    if role == "orchestrator":
        if not plan.coordination_enabled or zone is not None:
            raise ValueError("Orchestrator call context is invalid")
        return ordinal
    ordinal += int(plan.coordination_enabled)
    if role == "executor":
        if zone not in zones:
            raise ValueError("Executor call context has an unknown zone")
        return ordinal + list(zones).index(str(zone))
    if role == "reflector" and zone is None:
        return ordinal + len(zones)
    raise ValueError("model call context role is invalid")


def _model_context(
    plan: RunPlan,
    zones: Sequence[str],
    *,
    hour: int,
    step: int,
    role: str,
    zone: str | None = None,
) -> ModelCallContext:
    return ModelCallContext(
        hour=hour,
        step=step,
        call_ordinal=_call_ordinal(plan, zones, hour=hour, role=role, zone=zone),
        zone=zone,
    )


def _recorded_retry_count(run_dir: Path) -> int:
    attempts = run_dir / "model_request_attempts.jsonl"
    if not attempts.is_file():
        return 0
    count = 0
    for line in attempts.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        count += int(row.get("will_retry") is True)
    return count


def _graph_edges(graph: ConfirmedGraph | None) -> list[dict[str, Any]] | None:
    return None if graph is None else [edge.as_object() for edge in graph.edges]


def _site_graph_edges(graph: ConfirmedGraph | None) -> list[dict[str, Any]] | None:
    if graph is None:
        return None
    site_nodes = {str(node["id"]) for node in graph.nodes if node["scope"] == "site"}
    return [edge.as_object() for edge in graph.edges if edge.target in site_nodes]


def _shared_power_edge_ids(graph: ConfirmedGraph | None) -> set[str] | None:
    if graph is None:
        return None
    return {edge.identifier for edge in graph.edges if edge.target == "power_meters"}


def _forecast_slice(
    forecast: Mapping[str, Sequence[float]], point: str, step: int, count: int = 5
) -> list[float]:
    values = list(forecast[point][step : step + count])
    if len(values) != count:
        raise ValueError("evaluation forecast slice is incomplete")
    return [float(value) for value in values]


def _hour_observations(
    *,
    profile: Mapping[str, Any],
    forecast: Mapping[str, Sequence[float]],
    step: int,
    time_seconds: int,
    state: Mapping[str, Any],
    last_setpoint: Mapping[str, float],
    last_pmv: Mapping[str, float],
    last_occupancy: Mapping[str, float],
    pmv_of_temperature: Callable[[float], float],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, float], dict[str, float]]:
    global_inputs = profile["global_inputs"]
    weather = weather_view(
        _forecast_slice(forecast, global_inputs["outdoor_temperature"], step),
        _forecast_slice(forecast, global_inputs["solar_irradiance"], step),
    )
    weather_conditions = weather_condition_inputs(weather)
    current_occupancy: dict[str, float] = {}
    next_hour_occupancy: dict[str, float] = {}
    future_occupancy: dict[str, list[float]] = {}
    temperatures: dict[str, float] = {}
    for zone, mapping in profile["zones"].items():
        raw = _forecast_slice(forecast, mapping["occupancy_forecast"], step)
        effective = [
            effective_count(profile["occupancy"], time_seconds + offset * 900, value)
            for offset, value in enumerate(raw)
        ]
        current_occupancy[zone] = effective[0]
        next_hour_occupancy[zone] = effective[4]
        future_occupancy[zone] = effective[1:5]
        temperatures[zone] = zone_temperature_c(profile, state, zone)
    program_observations = build_program_observations(
        tuple(profile["zones"]),
        current_occupancy=current_occupancy,
        future_occupancy=future_occupancy,
        last_setpoints_c=last_setpoint,
        last_pmv=last_pmv,
        last_occupancy=last_occupancy,
    )
    observations: dict[str, dict[str, Any]] = {}
    for zone in profile["zones"]:
        temperature = temperatures[zone]
        observations[zone] = {
            "zone_temperature_c": temperature,
            **program_observations[zone],
            "next_hour_occupancy": next_hour_occupancy[zone],
            "electricity_price": float(forecast[global_inputs["electricity_price"]][step]),
            "outdoor_temp_c": round(
                float(forecast[global_inputs["outdoor_temperature"]][step]) - 273.15, 2
            ),
            "solar_irr": round(float(forecast[global_inputs["solar_irradiance"]][step]), 1),
            "comfort_headroom_c": comfort_headroom(pmv_of_temperature, temperature),
            **weather,
            **weather_conditions,
        }
    site_state = {
        "outdoor_temp_c": round(
            float(forecast[global_inputs["outdoor_temperature"]][step]) - 273.15, 2
        ),
        "solar_irr": round(float(forecast[global_inputs["solar_irradiance"]][step]), 1),
        "price_now": round(float(forecast[global_inputs["electricity_price"]][step]), 5),
        "price_next_hour": round(float(forecast[global_inputs["electricity_price"]][step + 4]), 5),
        **weather,
    }
    return observations, site_state, current_occupancy, next_hour_occupancy


def _zone_coupling_view(
    zones: Sequence[str],
    observations: Mapping[str, Mapping[str, Any]],
    programs: Mapping[str, ProgramLedger],
    executor_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    coupling: dict[str, Any] = {}
    for zone in zones:
        observation = observations[zone]
        headroom = observation.get("comfort_headroom_c")
        if headroom is not None and not isinstance(headroom, Mapping):
            raise ValueError("comfort headroom must be an object when available")
        precool_offset = float(programs[zone].current_program["params"]["precool_residual_c"])
        row: dict[str, Any] = {
            "zone_temperature_c": agent_visible_number(float(observation["zone_temperature_c"])),
            "pmv": agent_visible_number(float(observation["last_pmv"])),
            "occupancy": round(float(observation["current_occupancy"]), 1),
            "setpoint_c": agent_visible_number(float(observation["last_setpoint"])),
            "occupancy_at_interval_end": round(float(observation["next_hour_occupancy"]), 1),
            "precool_offset_from_unoccupied_base_c": round(precool_offset, 4),
            "resulting_precool_setpoint_c": round(30.0 + precool_offset, 4),
        }
        if isinstance(headroom, Mapping):
            row["temp_rise_to_warm_pmv_edge_c"] = headroom.get("warmer_c")
            row["temp_drop_to_cool_pmv_edge_c"] = headroom.get("cooler_c")
        occupied_events = [
            record
            for record in executor_records
            if record.get("scope", {}).get("zone") == zone
            and float(record.get("observation", {}).get("current_occupancy", 0.0)) > 0
        ][-4:]
        if occupied_events:
            more = sum(
                float(event["outcome"]["final_setpoint"]) <= 20.0 + 1e-6
                for event in occupied_events
            )
            less = sum(
                float(event["outcome"]["final_setpoint"]) >= 30.0 - 1e-6
                for event in occupied_events
            )
            if more:
                row["occupied_share_with_no_room_to_ask_for_more"] = round(
                    more / len(occupied_events), 3
                )
            if less:
                row["occupied_share_with_no_room_to_ask_for_less"] = round(
                    less / len(occupied_events), 3
                )
        coupling[zone] = row
    return coupling


async def _agent_hour(
    *,
    plan: RunPlan,
    hour: int,
    step: int,
    zones: Sequence[str],
    observations: Mapping[str, Mapping[str, Any]],
    site_state: Mapping[str, Any],
    route: Mapping[str, Any],
    graph: ConfirmedGraph | None,
    programs: Mapping[str, ProgramLedger],
    caol_records: Sequence[Mapping[str, Any]],
    executor_records: Sequence[Mapping[str, Any]],
    long_term_store: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
    client: ModelClient,
    artifacts: RunArtifacts,
    previous_allocation: Mapping[str, Any] | None,
    previous_utilisation: Mapping[str, Any] | None,
    last_rejection_by_zone: dict[str, Mapping[str, Any] | None],
    decision_time_seconds: int | None = None,
    previous_ledger: BudgetLedger | None = None,
) -> tuple[
    BudgetLedger | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[dict[str, Any]],
    bool,
]:
    del previous_ledger
    resolved_decision_time = hour * 3600 if decision_time_seconds is None else decision_time_seconds
    thinking_mode = (
        "disabled" if plan.thinking_policy == "all_roles_disabled" else str(route["thinking_mode"])
    )
    edges = _graph_edges(graph) if plan.causal_enabled else None
    site_edges = _site_graph_edges(graph) if plan.causal_enabled else None
    allowed_edge_ids = None if graph is None else set(graph.by_id)
    shared_power_edge_ids = _shared_power_edge_ids(graph)
    allocation: dict[str, Any] | None = None
    ledger: BudgetLedger | None = None
    raw_rejection: dict[str, Any] | None = None
    fallback_used = False
    fallback_source: str | None = None
    orchestrator_rationale_telemetry: dict[str, Any] | None = None
    resolved_site_cap = site_cap_max(zones)
    resolved_per_zone_reserved_cap = DEFAULT_PER_ZONE_RESERVED_CAP_C
    if plan.coordination_enabled:
        orchestrator = Orchestrator(client)
        user = orchestrator.build_user(
            hour=hour,
            decision_time_seconds=resolved_decision_time,
            zones=zones,
            site_state=site_state,
            zone_coupling=_zone_coupling_view(zones, observations, programs, executor_records),
            causal_edges=site_edges,
            previous_allocation=previous_allocation,
            previous_utilisation=previous_utilisation,
            working_memory=select_caol_working_memory(
                caol_records,
                current_hour=hour,
                zone=None,
                working_memory_hours=plan.working_memory_hours,
                zones=zones,
            ),
            allocation_limits={
                "zones": list(zones),
                "site_cap_c": resolved_site_cap,
                "per_zone_reserved_cap_c": resolved_per_zone_reserved_cap,
            },
        )
        try:
            allocation = await orchestrator.allocate(
                context=_model_context(
                    plan,
                    zones,
                    hour=hour,
                    step=step,
                    role="orchestrator",
                ),
                zones=zones,
                user=user,
                causal_enabled=plan.causal_enabled,
                thinking_mode=thinking_mode,
                allowed_causal_edge_ids=allowed_edge_ids,
                site_causal_edge_ids=shared_power_edge_ids,
                expected_site_cap_c=resolved_site_cap,
                expected_per_zone_reserved_cap_c=resolved_per_zone_reserved_cap,
            )
            orchestrator_rationale_telemetry = orchestrator.last_rationale_telemetry
        except ModelContractError as error:
            raw_rejection = {
                "code": "orchestrator_model_contract_rejected",
                "message": str(error),
                "raw_output": error.raw_output,
            }
            allocation, fallback_source = validated_fallback_allocation(
                zones,
                previous_allocation,
                site_cap_c=resolved_site_cap,
                causal_enabled=plan.causal_enabled,
                causal_edge_ids=(
                    [edge["id"] for edge in site_edges or ()] if plan.causal_enabled else None
                ),
                allowed_causal_edge_ids=allowed_edge_ids,
                site_causal_edge_ids=shared_power_edge_ids,
                per_zone_reserved_cap_c=resolved_per_zone_reserved_cap,
            )
            fallback_used = True
        ledger = BudgetLedger(
            allocation,
            zones,
            per_zone_reserved_cap_c=resolved_per_zone_reserved_cap,
        )

    proposals: dict[str, dict[str, Any] | ModelContractError] = {}
    proposal_rationale_telemetry: dict[str, dict[str, Any] | None] = {}
    proposal_memory_audit: dict[str, dict[str, Any]] = {}

    async def propose_for_zone(
        zone: str,
    ) -> tuple[
        dict[str, Any] | ModelContractError,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        executor = Executor(client)
        allowance = None if ledger is None else ledger.snapshot(zone)
        exposed = active_experiences(long_term_store, zone) if plan.long_term_memory else []
        user = executor.build_user(
            hour=hour,
            decision_time_seconds=resolved_decision_time,
            zone=zone,
            observation=observations[zone],
            current_executable_program=programs[zone].prompt_view(),
            causal_edges=edges,
            allowance=allowance,
            working_memory=select_caol_working_memory(
                caol_records,
                current_hour=hour,
                zone=zone,
                working_memory_hours=plan.working_memory_hours,
                zones=zones,
            ),
            long_term_experiences=exposed if plan.long_term_memory else None,
            rejection_feedback=last_rejection_by_zone[zone],
        )
        try:
            proposal = await executor.propose(
                context=_model_context(
                    plan,
                    zones,
                    hour=hour,
                    step=step,
                    role="executor",
                    zone=zone,
                ),
                user=user,
                causal_enabled=plan.causal_enabled,
                coordination_enabled=plan.coordination_enabled,
                thinking_mode=thinking_mode,
                long_term_memory=plan.long_term_memory,
            )
            if plan.long_term_memory:
                valid_refs, invalid_refs = validate_memory_refs(
                    executor.last_memory_refs,
                    exposed,
                )
                memory_audit: dict[str, Any] | None = {
                    "available": True,
                    "exposed": exposed,
                    "reported": executor.last_memory_refs,
                    "valid": valid_refs,
                    "invalid": invalid_refs,
                }
            else:
                memory_audit = None
            return proposal, executor.last_rationale_telemetry, memory_audit
        except ModelContractError as error:
            return (
                error,
                None,
                (
                    {
                        "available": False,
                        "exposed": exposed,
                        "reported": [],
                        "valid": [],
                        "invalid": [],
                    }
                    if plan.long_term_memory
                    else None
                ),
            )

    # Every Executor receives an uncharged snapshot. All issued requests finish before any
    # proposal is settled, so response timing cannot influence budget or program state.
    proposal_results = await asyncio.gather(
        *(propose_for_zone(zone) for zone in zones),
        return_exceptions=True,
    )
    transport_failures: list[tuple[str, TransportError]] = []
    other_failures: list[BaseException] = []
    for zone, result in zip(zones, proposal_results, strict=True):
        if isinstance(result, TransportError):
            transport_failures.append((zone, result))
        elif isinstance(result, BaseException):
            other_failures.append(result)
        else:
            (
                proposals[zone],
                proposal_rationale_telemetry[zone],
                memory_audit,
            ) = result
            if memory_audit is not None:
                proposal_memory_audit[zone] = memory_audit
    if transport_failures:
        artifacts.append_jsonl(
            "timing.jsonl",
            {
                "phase": "executor_batch",
                "event": "terminal_transport_failure",
                "hour": hour,
                "step": step,
                "issued_zones": list(zones),
                "failed_zones": [zone for zone, _ in transport_failures],
                "primary_failure_zone": transport_failures[0][0],
                "settlement_performed": False,
                "physical_advance_performed": False,
            },
        )
        raise ExecutorBatchTransportError(transport_failures)
    if other_failures:
        raise other_failures[0]

    updates: list[dict[str, Any]] = []
    settlement_order = list(ledger.priority) if ledger is not None else list(zones)
    if set(settlement_order) != set(zones) or len(settlement_order) != len(zones):
        raise ValueError("settlement order must cover exactly the configured zones")
    for zone in settlement_order:
        proposal = proposals[zone]
        version_before = programs[zone].version
        hash_before = program_hash(programs[zone].current_program)
        if isinstance(proposal, ModelContractError):
            rejected_row: dict[str, Any] = {
                "hour": hour,
                "step": step,
                "zone": zone,
                "status": "model_output_rejected",
                "patch": {
                    "op": "no_change",
                    "rationale": "tool-generated no_change after model contract rejection",
                },
                "rationale_telemetry": None,
                "completed_validation_stages": [],
                "rejection": {
                    "stage": "program_validation",
                    "code": "model_output_schema_rejected",
                    "message": str(proposal),
                    "raw_output": proposal.raw_output,
                },
                "program_version_before": version_before,
                "program_version_after": version_before,
                "program_hash_before": hash_before,
                "program_hash_after": hash_before,
                "current_program_version": version_before,
                "current_program_hash": hash_before,
                "replay_verified": bool(programs[zone].replay()),
            }
            if plan.long_term_memory:
                rejected_row["memory_refs"] = proposal_memory_audit[zone]
            artifacts.append_jsonl("program_updates.jsonl", rejected_row)
            updates.append(rejected_row)
            last_rejection_by_zone[zone] = copy.deepcopy(rejected_row["rejection"])
            continue
        validation = validate_candidate(
            proposal,
            programs[zone].current_program,
            graph=graph,
            ledger=ledger,
            zone=zone,
            step=step,
            causal_enabled=plan.causal_enabled,
            coordination_enabled=plan.coordination_enabled,
        )
        settled_row: dict[str, Any] = {
            "hour": hour,
            "step": step,
            "zone": zone,
            "status": "accepted" if validation.accepted else "rejected",
            "patch": copy.deepcopy(dict(validation.patch)),
            "rationale_telemetry": proposal_rationale_telemetry[zone],
            "completed_validation_stages": list(validation.completed_stages),
            "rejection": None if validation.rejection is None else validation.rejection.as_dict(),
            "program_version_before": version_before,
            "program_hash_before": hash_before,
        }
        if plan.long_term_memory:
            settled_row["memory_refs"] = proposal_memory_audit[zone]
        if validation.accepted and validation.patch["op"] != "no_change":
            update = programs[zone].commit(validation.patch, step=step, hour=hour)
            settled_row["accepted_update"] = update.as_dict()
        settled_row["current_program_version"] = programs[zone].version
        settled_row["current_program_hash"] = program_hash(programs[zone].current_program)
        settled_row["program_version_after"] = settled_row["current_program_version"]
        settled_row["program_hash_after"] = settled_row["current_program_hash"]
        settled_row["replay_verified"] = bool(programs[zone].replay())
        artifacts.append_jsonl("program_updates.jsonl", settled_row)
        updates.append(settled_row)
        last_rejection_by_zone[zone] = (
            copy.deepcopy(settled_row["rejection"]) if settled_row["status"] == "rejected" else None
        )

    coordination_audit: dict[str, Any] | None = None
    if plan.coordination_enabled:
        coordination_audit = {
            "status": "fallback" if fallback_used else "accepted",
            "raw_contract": {
                "status": "rejected" if fallback_used else "accepted",
                "rejection": raw_rejection,
            },
            "rationale_telemetry": orchestrator_rationale_telemetry,
            "fallback": allocation_fallback_audit(
                used=fallback_used,
                reason=(
                    raw_rejection["message"]
                    if fallback_used and raw_rejection is not None
                    else None
                ),
                source=fallback_source if fallback_used else None,
            ),
            "allocation_audit": allocation,
            "settlement_order": settlement_order,
        }
    return ledger, allocation, coordination_audit, updates, fallback_used


async def _reflect_hour(
    *,
    plan: RunPlan,
    hour: int,
    step: int,
    interval_start_time_seconds: int,
    zones: Sequence[str],
    route: Mapping[str, Any],
    current_cao: Sequence[Mapping[str, Any]],
    long_term_store: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
    client: ModelClient,
) -> tuple[ReflectorResolution, dict[str, Any]]:
    thinking_mode = (
        "disabled" if plan.thinking_policy == "all_roles_disabled" else str(route["thinking_mode"])
    )
    reflector = Reflector(client)
    observed_by_zone: dict[str, list[str]] = {}
    for record in current_cao:
        zone = str(record.get("zone"))
        context = record.get("context")
        coverage = context.get("regime_step_coverage") if isinstance(context, Mapping) else None
        if zone not in zones or not isinstance(coverage, Mapping):
            raise ValueError("completed-hour regime coverage is incomplete")
        ordered: list[tuple[int, str]] = []
        for regime, raw_steps in coverage.items():
            if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
                raise ValueError("completed-hour regime coverage must contain step sequences")
            steps = [int(value) for value in raw_steps]
            if not steps:
                continue
            ordered.append((min(steps), str(regime)))
        ordered.sort(key=lambda item: item[0])
        observed_by_zone[zone] = [regime for _, regime in ordered]
    if set(observed_by_zone) != set(zones):
        raise ValueError("completed-hour evidence must cover every configured zone")
    user = reflector.build_user(
        current_hour_cao=current_cao,
        interval_start_time_seconds=interval_start_time_seconds,
        long_term_slots=(
            {
                zone: reflector_slot_view(long_term_store, zone, observed_by_zone[zone])
                for zone in zones
            }
            if plan.long_term_memory
            else None
        ),
    )
    try:
        resolution = await reflector.summarize(
            context=_model_context(
                plan,
                zones,
                hour=hour,
                step=step,
                role="reflector",
            ),
            user=user,
            causal_enabled=plan.causal_enabled,
            thinking_mode=thinking_mode,
            zones=zones,
            long_term_memory=plan.long_term_memory,
        )
        return resolution, {
            "status": "accepted" if resolution.clean else "model_contract_degraded",
            "issues": list(resolution.issues),
            "rejection": None,
        }
    except ModelContractError as error:
        resolution = ReflectorResolution(
            lessons={},
            operations={},
            issues=tuple({"zone": zone, "code": "root_contract_rejected"} for zone in zones),
        )
        return resolution, {
            "status": "model_output_rejected",
            "issues": list(resolution.issues),
            "rejection": {
                "code": "reflector_model_contract_rejected",
                "message": str(error),
                "raw_output": error.raw_output,
            },
        }


async def _evaluate(
    *,
    plan: RunPlan,
    profile: Mapping[str, Any],
    graph: ConfirmedGraph | None,
    physical: PhysicalClient,
    model: ModelClient | None,
    artifacts: RunArtifacts,
    boundary: EvaluationBoundaryState,
    run_identity: str,
    resume_prefix: ResumePrefix | None = None,
) -> tuple[dict[str, Any], bool, int, int]:
    step_seconds = int(profile["control_step_seconds"])
    evaluation_steps = plan.evaluation_hours * 4
    points = forecast_points(profile)
    source_forecast = physical.forecast(
        points, (evaluation_steps + 96) * step_seconds, step_seconds
    )
    evaluation_start = plan.evaluation_start_seconds(dict(profile))
    forecast, resolution_events = resolve_forecast_missing_occupancy(
        profile,
        source_forecast,
        points,
        evaluation_steps + 97,
        forecast_phase="evaluation",
        start_time_seconds=evaluation_start,
        step_seconds=step_seconds,
    )
    forecast_evidence = build_forecast_evidence(
        points,
        source_forecast,
        forecast,
        start_time_seconds=evaluation_start,
        step_seconds=step_seconds,
    )
    if resume_prefix is not None and forecast_evidence != resume_prefix.forecast_inputs:
        raise ValueError("resume forecast evidence differs from the failed run")
    artifacts.write_forecast_inputs(forecast_evidence)
    for event in resolution_events:
        artifacts.append_jsonl("timing.jsonl", event)
    zones = tuple(profile["zones"])
    programs = {
        zone: ProgramLedger(
            load_program(repository_root() / profile["program"], zone),
            causal_enabled=plan.causal_enabled,
        )
        for zone in zones
    }
    state = boundary.state
    last_setpoint = dict(boundary.last_setpoint_c)
    last_pmv = dict(boundary.last_pmv)
    last_occupancy = dict(boundary.last_occupancy)
    metrics = MetricsAccumulator()
    frames: list[dict[str, Any]] = []
    executor_records: list[dict[str, Any]] = []
    caol_records: list[dict[str, Any]] = []
    long_term_store = empty_regime_store(zones)
    previous_allocation: Mapping[str, Any] | None = None
    previous_utilisation: Mapping[str, Any] | None = None
    hour_results: list[dict[str, Any]] = []
    route: dict[str, Any] = {}
    ledger: BudgetLedger | None = None
    coordination_audit: dict[str, Any] | None = None
    hour_program_decisions: list[dict[str, Any]] = []
    last_rejection_by_zone: dict[str, Mapping[str, Any] | None] = {zone: None for zone in zones}
    fallback_count = 0
    replay_until_step = 0 if resume_prefix is None else resume_prefix.next_step
    if resume_prefix is not None:
        if resume_prefix.plan != plan:
            raise ValueError("resume plan differs from the failed run")
        if plan.identity(dict(profile)) != resume_prefix.source_plan_identity:
            raise ValueError("resume profile or method identity differs from the failed run")
        for stream_name in (
            "agent_calls.jsonl",
            "raw_model_io.jsonl",
            "model_request_attempts.jsonl",
        ):
            for imported in resume_prefix.calls_for_import(stream_name):
                artifacts.append_jsonl(stream_name, imported)
        artifacts.append_jsonl(
            "timing.jsonl",
            {
                "phase": "resume_replay",
                "event": "started",
                "source_run_identity": resume_prefix.source_run_identity,
                "source_test_id": resume_prefix.source_test_id,
                "source_prefix_identity": resume_prefix.prefix_identity,
                "replay_step_count": replay_until_step,
            },
        )

    for step in range(evaluation_steps):
        action_time = evaluation_start + step * step_seconds
        require_time(state, action_time)
        observations, site_state, current_occ, next_occ = _hour_observations(
            profile=profile,
            forecast=forecast,
            step=step,
            time_seconds=action_time,
            state=state,
            last_setpoint=last_setpoint,
            last_pmv=last_pmv,
            last_occupancy=last_occupancy,
            pmv_of_temperature=boundary.comfort.pmv,
        )
        hour = step // 4
        if step % 4 == 0:
            route = hourly_route(hour, current_occ, next_occ)
            hour_results = []
            if step < replay_until_step:
                assert resume_prefix is not None
                source_decision = resume_prefix.decision_for_hour(hour)
                if not _replay_equal(route, source_decision["route"]):
                    raise ValueError("resume hourly route differs from source evidence")
                hour_program_decisions = [dict(row) for row in resume_prefix.updates_for_hour(hour)]
                if len(hour_program_decisions) != len(zones):
                    raise ValueError("resume hour does not cover every zone program decision")
                for source_update in hour_program_decisions:
                    zone = str(source_update["zone"])
                    program = programs[zone]
                    if (
                        program.version != int(source_update["program_version_before"])
                        or program_hash(program.current_program)
                        != source_update["program_hash_before"]
                    ):
                        raise ValueError("resume program before-state differs")
                    patch = source_update["patch"]
                    if (
                        source_update.get("status") == "accepted"
                        and isinstance(patch, Mapping)
                        and patch.get("op") != "no_change"
                    ):
                        accepted = program.commit(
                            patch,
                            step=int(source_update["step"]),
                            hour=int(source_update["hour"]),
                        ).as_dict()
                        if accepted != source_update.get("accepted_update"):
                            raise ValueError("resume accepted program update differs")
                    if (
                        program.version != int(source_update["program_version_after"])
                        or program_hash(program.current_program)
                        != source_update["program_hash_after"]
                    ):
                        raise ValueError("resume program after-state differs")
                    artifacts.append_jsonl("program_updates.jsonl", source_update)
                    rejection = source_update.get("rejection")
                    last_rejection_by_zone[zone] = (
                        copy.deepcopy(rejection)
                        if source_update.get("status") in {"rejected", "model_output_rejected"}
                        and isinstance(rejection, Mapping)
                        else None
                    )
                coordination_audit = source_decision.get("orchestration")
                if plan.coordination_enabled:
                    if not isinstance(coordination_audit, Mapping):
                        raise ValueError("resume orchestration evidence is missing")
                    previous_allocation = coordination_audit.get("allocation_audit")
                    fallback = coordination_audit.get("fallback")
                    if isinstance(fallback, Mapping):
                        fallback_count += int(fallback.get("used") is True)
                else:
                    previous_allocation = None
                ledger = None
            elif plan.controller == "h3c_agent":
                if model is None:
                    raise ValueError("Agent execution requires a model client")
                (
                    ledger,
                    previous_allocation,
                    coordination_audit,
                    hour_program_decisions,
                    fallback_used,
                ) = await _agent_hour(
                    plan=plan,
                    hour=hour,
                    step=step,
                    decision_time_seconds=action_time,
                    zones=zones,
                    observations=observations,
                    site_state=site_state,
                    route=route,
                    graph=graph,
                    programs=programs,
                    caol_records=caol_records,
                    executor_records=executor_records,
                    long_term_store=long_term_store,
                    client=model,
                    artifacts=artifacts,
                    previous_allocation=previous_allocation,
                    previous_utilisation=previous_utilisation,
                    last_rejection_by_zone=last_rejection_by_zone,
                )
                fallback_count += int(fallback_used)

        proposed, assured, assurance_audit = execute_zone_programs(
            {zone: programs[zone].current_program for zone in zones}, observations
        )
        if step < replay_until_step:
            assert resume_prefix is not None
            source_by_zone = resume_prefix.zone_rows_for_step(step)
            if set(source_by_zone) != set(zones):
                raise ValueError("resume step does not cover every zone")
            for zone in zones:
                source_row = source_by_zone[zone]
                if not (
                    _replay_equal(observations[zone], source_row["observation"])
                    and _replay_equal(proposed[zone], source_row["interpreter"])
                    and _replay_equal(assurance_audit[zone], source_row["action_assurance"])
                    and _replay_equal(assured[zone], source_row["final_setpoint_c"])
                ):
                    raise ValueError("resume observation/program/action replay differs")
        next_state = physical.advance(control_input(profile, assured))
        require_time(next_state, action_time + step_seconds)
        if physical.test_id != boundary.test_id:
            raise ValueError("test id changed during formal evaluation")

        outdoor_point = profile["global_inputs"]["outdoor_temperature"]
        daily = forecast[outdoor_point][step : step + 97 : 4]
        boundary.comfort.update_clothing(
            action_time, sum(float(value) - 273.15 for value in daily) / len(daily)
        )
        temperatures = {zone: zone_temperature_c(profile, next_state, zone) for zone in zones}
        pmv = {zone: boundary.comfort.pmv(temperatures[zone]) for zone in zones}
        power = site_power(profile, next_state)
        price = float(forecast[profile["global_inputs"]["electricity_price"]][step])
        cost = power * step_seconds / 3.6e6 * price
        reward_breakdown = step_reward_breakdown(
            cost=cost,
            pmv=[pmv[zone] for zone in zones],
            occupancy=[current_occ[zone] for zone in zones],
            setpoints_c=[assured[zone] for zone in zones],
            previous_setpoints_c=[last_setpoint[zone] for zone in zones],
            objective=profile["objective"],
            zone_names=zones,
        )
        reward = float(reward_breakdown["reward"])
        metrics.add(
            cost=cost,
            power_w=power,
            reward=reward,
            pmv=[pmv[zone] for zone in zones],
            occupancy=[current_occ[zone] for zone in zones],
        )
        for zone in zones:
            row = {
                "hour": hour,
                "step": step,
                "zone": zone,
                "test_id": boundary.test_id,
                "action_time_seconds": action_time,
                "outcome_time_seconds": action_time + step_seconds,
                "observation": observations[zone],
                "interpreter": proposed[zone],
                "action_assurance": assurance_audit[zone],
                "final_setpoint_c": assured[zone],
                "outcome": {
                    "zone_temperature_c": temperatures[zone],
                    "pmv": pmv[zone],
                    "effective_occupancy": current_occ[zone],
                    "power_w": power,
                    "cost": cost,
                    **(
                        {
                            "objective_feedback": {
                                "site_step_reward": reward,
                                "site_energy_penalty": reward_breakdown["site_energy_penalty"],
                                "site_comfort_penalty": reward_breakdown["site_comfort_penalty"],
                                "site_smoothness_penalty": reward_breakdown[
                                    "site_smoothness_penalty"
                                ],
                                "zone_comfort_penalty_contribution": reward_breakdown[
                                    "zone_comfort_penalty_contributions"
                                ][zone],
                                "zone_smoothness_penalty_contribution": reward_breakdown[
                                    "zone_smoothness_penalty_contributions"
                                ][zone],
                            }
                        }
                        if plan.controller == "h3c_agent"
                        else {}
                    ),
                },
            }
            if step < replay_until_step:
                assert resume_prefix is not None
                source_row = resume_prefix.zone_rows_for_step(step)[zone]
                comparable_source = {
                    key: value for key, value in source_row.items() if key != "test_id"
                }
                comparable_current = {key: value for key, value in row.items() if key != "test_id"}
                if not _replay_equal(comparable_current, comparable_source):
                    raise ValueError("resume physical outcome differs from source evidence")
            artifacts.append_jsonl("zone_steps.jsonl", row)
            hour_results.append(row)
        performance_row = (
            action_time,
            hour,
            step,
            power,
            cost,
            reward,
            _canonical([temperatures[zone] for zone in zones]),
            _canonical([assured[zone] for zone in zones]),
            _canonical([pmv[zone] for zone in zones]),
            _canonical([current_occ[zone] for zone in zones]),
        )
        if step < replay_until_step:
            assert resume_prefix is not None
            source_performance = resume_prefix.performance[step]
            replayed_performance = {
                "time_seconds": str(action_time),
                "hour": str(hour),
                "step": str(step),
                "total_power_w": str(power),
                "step_cost": str(cost),
                "step_reward": str(reward),
                "zone_temperatures_c": performance_row[6],
                "zone_setpoints_c": performance_row[7],
                "zone_pmv": performance_row[8],
                "zone_occupancy": performance_row[9],
            }
            scalar_fields = (
                "time_seconds",
                "hour",
                "step",
                "total_power_w",
                "step_cost",
                "step_reward",
            )
            if not all(
                _replay_equal(
                    float(replayed_performance[field]),
                    float(source_performance[field]),
                )
                for field in scalar_fields
            ) or not all(
                _replay_equal(
                    json.loads(str(replayed_performance[field])),
                    json.loads(source_performance[field]),
                )
                for field in (
                    "zone_temperatures_c",
                    "zone_setpoints_c",
                    "zone_pmv",
                    "zone_occupancy",
                )
            ):
                raise ValueError("resume performance/KPI replay differs")
        artifacts.append_performance(performance_row)
        state = next_state
        last_setpoint = assured
        last_pmv = pmv
        last_occupancy = current_occ

        if step % 4 == 3:
            lessons: list[dict[str, str]] = []
            reflector_contract: dict[str, Any] | None = None
            new_frames: list[dict[str, Any]] = []
            new_executor_records: list[dict[str, Any]] = []
            new_cao: list[dict[str, Any]] = []
            for zone in zones:
                zone_rows = [row for row in hour_results if row["zone"] == zone]
                decision = next(
                    (row for row in hour_program_decisions if row.get("zone") == zone),
                    {
                        "hour": hour,
                        "step": hour * 4,
                        "zone": zone,
                        "status": "not_called",
                        "patch": {"op": "no_change", "rationale": "controller has no model"},
                        "current_program_version": programs[zone].version,
                        "current_program_hash": program_hash(programs[zone].current_program),
                        "rejection": None,
                    },
                )
                new_frames.append(
                    completed_summary_frame(
                        hour=hour,
                        zone=zone,
                        step_rows=zone_rows,
                        program_decision=decision,
                    )
                )
                new_executor_records.extend(
                    completed_executor_records(
                        hour=hour,
                        zone=zone,
                        step_rows=zone_rows,
                        program_decision=decision,
                    )
                )
                new_cao.append(
                    build_hourly_cao(
                        hour=hour,
                        zone=zone,
                        step_rows=zone_rows,
                        program_decision=decision,
                    )
                )
            if step < replay_until_step:
                assert resume_prefix is not None
                source_caol = resume_prefix.caol_for_hour(hour)
                if len(source_caol) != len(zones):
                    raise ValueError("resume completed hour lacks zone working memory")
                source_caol_by_zone = {str(row["zone"]): row for row in source_caol}
                for deterministic_cao in new_cao:
                    source_record = source_caol_by_zone[str(deterministic_cao["zone"])]
                    without_lesson = {
                        key: value for key, value in source_record.items() if key != "lesson"
                    }
                    if not _replay_equal(deterministic_cao, without_lesson):
                        raise ValueError("resume working-memory evidence differs")
                completed_caol = [dict(row) for row in source_caol]
                source_hourly = resume_prefix.decision_for_hour(hour)
                previous_utilisation = (
                    source_hourly.get("energy_budget") if plan.coordination_enabled else None
                )
                artifacts.append_jsonl("hourly_decisions.jsonl", source_hourly)
            elif plan.controller == "h3c_agent":
                assert model is not None
                resolution, reflector_contract = await _reflect_hour(
                    plan=plan,
                    hour=hour,
                    step=step,
                    interval_start_time_seconds=int(hour_results[0]["action_time_seconds"]),
                    zones=zones,
                    route=route,
                    current_cao=new_cao,
                    long_term_store=long_term_store,
                    client=model,
                )
                lessons = [
                    {"zone": zone, "lesson": resolution.lessons[zone]}
                    for zone in zones
                    if zone in resolution.lessons
                ]
                if plan.long_term_memory:
                    observed_regimes = {
                        str(row["zone"]): list(row["context"]["regime_step_coverage"])
                        for row in new_cao
                    }
                    long_term_store, memory_audits = apply_memory_operations(
                        long_term_store,
                        resolution.operations,
                        zones=zones,
                        hour=hour,
                        observed_regimes=observed_regimes,
                    )
                    for audit in memory_audits:
                        artifacts.append_jsonl("long_term_memory_crud.jsonl", audit)
                completed_caol = attach_hourly_lessons(new_cao, resolution.lessons)
            else:
                completed_caol = new_cao
            for record in completed_caol:
                artifacts.append_jsonl("caol_records.jsonl", record)
            caol_records.extend(completed_caol)
            frames.extend(new_frames)
            executor_records.extend(new_executor_records)
            hourly: dict[str, Any] = {
                "hour": hour,
                "route": route,
                "reflector_summary": lessons,
                "reflector_contract": reflector_contract,
                "program_replay": {
                    zone: {
                        "verified": bool(programs[zone].replay()),
                        "version": programs[zone].version,
                        "hash": program_hash(programs[zone].current_program),
                    }
                    for zone in zones
                },
            }
            if plan.coordination_enabled:
                if step < replay_until_step:
                    assert resume_prefix is not None
                    source_hourly = resume_prefix.decision_for_hour(hour)
                    hourly["orchestration"] = source_hourly["orchestration"]
                    hourly["energy_budget"] = source_hourly["energy_budget"]
                else:
                    hourly["orchestration"] = coordination_audit
                    if ledger is None:
                        raise ValueError("completed Agent hour lacks its Budget ledger")
                    hourly["energy_budget"] = ledger.utilisation()
                previous_utilisation = hourly["energy_budget"]
            if step >= replay_until_step:
                artifacts.append_jsonl("hourly_decisions.jsonl", hourly)
            artifacts.replace_completed_hour_checkpoint(
                {
                    "artifact_schema": "h3c_completed_hour_checkpoint",
                    "schema_version": 1,
                    "run_identity": run_identity,
                    "test_id": boundary.test_id,
                    "completed_hour": hour,
                    "completed_step": step,
                    "next_step": step + 1,
                    "program_versions": {zone: programs[zone].version for zone in zones},
                }
            )
            if step + 1 == replay_until_step:
                assert resume_prefix is not None
                if {zone: programs[zone].version for zone in zones} != dict(
                    resume_prefix.program_versions
                ):
                    raise ValueError("resume restored program versions differ at checkpoint")
                artifacts.append_jsonl(
                    "timing.jsonl",
                    {
                        "phase": "resume_replay",
                        "event": "completed",
                        "source_run_identity": resume_prefix.source_run_identity,
                        "source_test_id": resume_prefix.source_test_id,
                        "source_prefix_identity": resume_prefix.prefix_identity,
                        "replay_step_count": replay_until_step,
                        "next_step": replay_until_step,
                    },
                )
    replay_verified = all(bool(program.replay()) for program in programs.values())
    return metrics.resolved(), replay_verified, len(resolution_events), fallback_count


async def _execute_one(
    plan: RunPlan,
    *,
    suite: str,
    output_root: Path,
    physical_factory: PhysicalFactory,
    model_factory: ModelFactory | None,
    resume_prefix: ResumePrefix | None = None,
) -> dict[str, Any]:
    resolved, graph = _resolved_plan(plan)
    profile = resolved["case_profile"]
    runtime = load_runtime_contract()
    source_commit = _source_commit()
    physical_environment = runtime["physical_service"]["endpoint_environment_variable"]
    physical_endpoint = os.environ.get(physical_environment, "").rstrip("/")
    if not physical_endpoint:
        raise ValueError(f"{physical_environment} is required for physical execution")
    model_endpoint = ""
    provider_id: str | None = None
    provider_contract: dict[str, Any] | None = None
    key_environment = ""
    if plan.controller == "h3c_agent":
        provider_id = plan.effective_model_provider()
        if provider_id is None:
            raise ValueError("Agent execution requires a model provider")
        provider_contract = load_model_provider_contract(provider_id)
        model_environment = provider_contract["endpoint_environment_variable"]
        key_environment = str(provider_contract["api_key_environment_variable"])
        fixed_endpoint = provider_contract["fixed_endpoint"]
        model_endpoint = (
            str(fixed_endpoint)
            if fixed_endpoint is not None
            else os.environ.get(str(model_environment), "")
        ).rstrip("/")
        if not model_endpoint or not os.environ.get(key_environment):
            raise ValueError("model endpoint and API key are required for Agent execution")
    model_identity_fields: dict[str, Any] = {}
    if plan.controller == "h3c_agent":
        assert provider_id is not None and provider_contract is not None
        model_identity_fields = {
            "model_provider": provider_id,
            "model_name": provider_contract["model"],
            "model_endpoint_identity": _endpoint_identity(model_endpoint),
            "objective_feedback_contract": "completed_interval_reward_breakdown_v1",
        }
    execution_identity = {
        "plan_identity": plan.identity(profile),
        "source_commit": source_commit,
        "runtime_contract": runtime,
        "physical_endpoint_identity": _endpoint_identity(physical_endpoint),
        "dispatch_mode": "auto",
        **model_identity_fields,
    }
    if resume_prefix is not None:
        lineage = resume_prefix.lineage()
        execution_identity["resume_replay_identity"] = _identity(lineage)
        resolved["resume_replay"] = lineage
    run_identity = _identity(execution_identity)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + run_identity[:12]
    artifacts = RunArtifacts(output_root, suite, plan.profile, run_id)
    resolved["runtime_contract"] = runtime
    resolved["execution_identity"] = execution_identity
    manifest: dict[str, Any] = {
        "manifest_schema": "h3c_run_manifest",
        "schema_version": 5,
        "run_identity": run_identity,
        "source_commit": source_commit,
        "controller": plan.controller,
        "dispatch_mode": "auto",
        "expected_agent_calls": plan.expected_agent_calls(len(profile["zones"])),
        "retry_count": 0,
        "transport_error_count": 0,
        "fallback_count": 0,
        "secret_exposure_count": 0,
        "objective_feedback_contract": (
            "completed_interval_reward_breakdown_v1"
            if plan.controller == "h3c_agent"
            else "not_applicable"
        ),
        "secret_scan_status": ("pending" if plan.controller == "h3c_agent" else "not_applicable"),
        "occupancy_forecast_missing_value_resolution_count": 0,
        "program_replay_verified": False,
        "conditioning_prefix_identity": None,
        "evaluation_boundary_identity": None,
        "lifecycle": {
            "initialize_count": 0,
            "stop_count": 0,
            "test_id_changes": 0,
            "conditioning_advance_count": 0,
        },
    }
    artifacts.create(resolved, manifest)
    artifacts.replace_dispatch_state(
        {
            "artifact_schema": "h3c_dispatch_state",
            "schema_version": 1,
            "run_identity": run_identity,
            "dispatch_mode": "auto",
            "status": "SELECT_PENDING",
            "test_id": None,
            "testcase": profile["testcase"],
        }
    )
    physical = physical_factory(physical_endpoint)
    model: ModelClient | None = None
    if plan.controller == "h3c_agent":
        assert provider_id is not None and provider_contract is not None
        session_header = provider_contract["session_affinity_header"]
        extra_headers = (
            {str(session_header): run_identity}
            if isinstance(session_header, str) and session_header
            else None
        )
        model = (
            model_factory(artifacts, str(provider_contract["model"]))
            if model_factory is not None
            else OpenAICompatibleModelClient(
                endpoint=model_endpoint,
                api_key=os.environ[key_environment],
                model=str(provider_contract["model"]),
                sink=lambda name, row: artifacts.append_jsonl(name, row),
                retry_count_limit=runtime["model"]["retry_count"],
                retry_backoff_seconds=tuple(runtime["model"]["retry_backoff_seconds"]),
                provider_id=provider_id,
                extra_headers=extra_headers,
                retryable_status_codes=tuple(provider_contract["retryable_status_codes"]),
                response_format=str(provider_contract["response_format"]),
            )
        )
    started = time.perf_counter()
    initialized = False
    stop_attempted = False
    terminal_error: Exception | None = None
    terminal_failure_type = "terminal_runtime_error"

    def record_dispatch(event: Mapping[str, Any]) -> None:
        status = event.get("status")
        if event.get("event") == "selected" and status is None:
            status = "SELECTED"
        elif event.get("event") == "stopped":
            status = "STOPPED"
        artifacts.replace_dispatch_state(
            {
                "artifact_schema": "h3c_dispatch_state",
                "schema_version": 1,
                "run_identity": run_identity,
                "dispatch_mode": "auto",
                "status": status,
                "test_id": event.get("test_id"),
                "testcase": event.get("testcase", profile["testcase"]),
            }
        )
        artifacts.append_jsonl("timing.jsonl", dict(event))

    lifecycle_setter = getattr(physical, "set_lifecycle_sink", None)
    if callable(lifecycle_setter):
        lifecycle_setter(record_dispatch)

    def record_initialization(test_id: str) -> None:
        nonlocal initialized
        initialized = True
        manifest["lifecycle"]["initialize_count"] = 1
        artifacts.replace_manifest(manifest)
        artifacts.replace_dispatch_state(
            {
                "artifact_schema": "h3c_dispatch_state",
                "schema_version": 1,
                "run_identity": run_identity,
                "dispatch_mode": "auto",
                "status": "Running",
                "test_id": test_id,
                "testcase": profile["testcase"],
            }
        )

    try:
        boundary = initialize_evaluation_boundary(
            physical,
            profile,
            artifacts,
            evaluation_start_seconds=plan.evaluation_start_seconds(profile),
            on_initialized=record_initialization,
        )
        if resume_prefix is not None and boundary.test_id == resume_prefix.source_test_id:
            raise ValueError("resume requires a fresh physical test identity")
        manifest["occupancy_forecast_missing_value_resolution_count"] = (
            boundary.occupancy_missing_value_resolution_count
        )
        manifest["conditioning_prefix_identity"] = boundary.conditioning_prefix_identity
        manifest["evaluation_boundary_identity"] = boundary.evaluation_boundary_identity
        manifest["lifecycle"]["conditioning_advance_count"] = 0
        artifacts.append_jsonl(
            "timing.jsonl",
            {"phase": "initialization", "elapsed_seconds": time.perf_counter() - started},
        )
        evaluation_start_seconds = plan.evaluation_start_seconds(profile)
        artifacts.append_jsonl(
            "timing.jsonl",
            {
                "phase": "physical_lifecycle",
                "event": "evaluation_started",
                "time_seconds": evaluation_start_seconds,
                "test_id": boundary.test_id,
            },
        )
        evaluation_started = time.perf_counter()
        _, replay_verified, evaluation_resolution_count, fallback_count = await _evaluate(
            plan=plan,
            profile=profile,
            graph=graph,
            physical=physical,
            model=model,
            artifacts=artifacts,
            boundary=boundary,
            run_identity=run_identity,
            resume_prefix=resume_prefix,
        )
        manifest["occupancy_forecast_missing_value_resolution_count"] += evaluation_resolution_count
        artifacts.append_jsonl(
            "timing.jsonl",
            {"phase": "evaluation", "elapsed_seconds": time.perf_counter() - evaluation_started},
        )
        evaluation_end_seconds = evaluation_start_seconds + plan.evaluation_hours * 3600
        artifacts.append_jsonl(
            "timing.jsonl",
            {
                "phase": "physical_lifecycle",
                "event": "evaluation_completed",
                "time_seconds": evaluation_end_seconds,
                "test_id": boundary.test_id,
            },
        )
        stop_attempted = True
        physical.stop()
        artifacts.append_jsonl(
            "timing.jsonl",
            {
                "phase": "physical_lifecycle",
                "event": "stopped",
                "time_seconds": evaluation_end_seconds,
                "test_id": boundary.test_id,
            },
        )
        initialized = False
        artifacts.replace_dispatch_state(
            {
                "artifact_schema": "h3c_dispatch_state",
                "schema_version": 1,
                "run_identity": run_identity,
                "dispatch_mode": "auto",
                "status": "STOPPED",
                "test_id": boundary.test_id,
                "testcase": profile["testcase"],
            }
        )
        manifest["lifecycle"]["stop_count"] = 1
        manifest["program_replay_verified"] = replay_verified
        manifest["fallback_count"] = fallback_count
        manifest["retry_count"] = _recorded_retry_count(artifacts.run_dir)
        if plan.controller == "h3c_agent":
            secret_name = key_environment
            manifest["secret_exposure_count"] = _secret_occurrences(
                artifacts.run_dir, os.environ[secret_name]
            )
            manifest["secret_scan_status"] = "completed"
        artifacts.replace_manifest(manifest)
        metrics = compute_run_metrics(artifacts.run_dir)
        artifacts.write_metrics(metrics)
        verification = verify_run(artifacts.run_dir, require_completion=False)
        artifacts.write_verification(verification)
        if not verification["completion_eligible"]:
            raise RunAcceptanceFailure(artifacts.run_dir, verification)
        if verify_run(artifacts.run_dir, require_completion=False) != verification:
            raise ValueError("pre-completion verification changed after result publication")
        completion = {
            "status": "complete",
            "classification": verification["classification"],
            "run_identity": run_identity,
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        completion_path = artifacts.publish_completion(completion)
        return {
            "profile": plan.profile,
            "status": "complete",
            "classification": verification["classification"],
            "run_identity": run_identity,
            "completion": str(completion_path),
            "metrics": metrics,
            **({"resume_replay": resume_prefix.lineage()} if resume_prefix is not None else {}),
        }
    except TransportError as error:
        manifest["retry_count"] = _recorded_retry_count(artifacts.run_dir)
        manifest["transport_error_count"] += int(getattr(error, "failure_count", 1))
        terminal_error = error
        terminal_failure_type = "terminal_transport_error"
    except Exception as error:
        terminal_error = error
    finally:
        try:
            if (initialized or physical.test_id is not None) and not stop_attempted:
                stop_attempted = True
                stopping_test_id = physical.test_id
                physical.stop()
                manifest["lifecycle"]["stop_count"] += 1
                artifacts.replace_dispatch_state(
                    {
                        "artifact_schema": "h3c_dispatch_state",
                        "schema_version": 1,
                        "run_identity": run_identity,
                        "dispatch_mode": "auto",
                        "status": "STOPPED",
                        "test_id": stopping_test_id,
                        "testcase": profile["testcase"],
                    }
                )
        finally:
            if (
                plan.controller == "h3c_agent"
                and artifacts.run_dir.is_dir()
                and not (artifacts.run_dir / "completion.json").exists()
            ):
                secret_name = key_environment
                manifest["secret_exposure_count"] = _secret_occurrences(
                    artifacts.run_dir, os.environ[secret_name]
                )
                manifest["secret_scan_status"] = "completed"
            if artifacts.run_dir.is_dir() and not (artifacts.run_dir / "completion.json").exists():
                artifacts.replace_manifest(manifest)
    if terminal_error is not None:
        metrics_path = artifacts.run_dir / "metrics.json"
        verification_path = artifacts.run_dir / "verification.json"
        finalization_error_type: str | None = None
        failure_verification: Mapping[str, Any] | None = None
        try:
            if not metrics_path.is_file():
                artifacts.write_metrics(compute_run_metrics(artifacts.run_dir))
            if verification_path.is_file():
                failure_verification = json.loads(verification_path.read_text(encoding="utf-8"))
            else:
                failure_verification = verify_run(artifacts.run_dir, require_completion=False)
                artifacts.write_verification(failure_verification)
        except Exception as finalization_error:
            finalization_error_type = type(finalization_error).__name__
        artifacts.publish_failure(
            {
                "status": "failed",
                "classification": (
                    failure_verification.get("classification", "RUN-INVALID")
                    if failure_verification is not None
                    else "RUN-INVALID"
                ),
                "run_identity": run_identity,
                "finished_at": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "failure_type": terminal_failure_type,
                "error_type": getattr(terminal_error, "error_type", type(terminal_error).__name__),
                "retryable": bool(getattr(terminal_error, "retryable", False)),
                "provider_response_received": bool(
                    getattr(terminal_error, "provider_response_received", False)
                ),
                "failure_count": int(getattr(terminal_error, "failure_count", 1)),
                **(
                    {"finalization_error_type": finalization_error_type}
                    if finalization_error_type is not None
                    else {}
                ),
            }
        )
        raise terminal_error
    raise AssertionError("execution exited without a result or terminal error")


def execute_serial(
    plans: Sequence[RunPlan],
    *,
    suite: str,
    output_root: Path | None = None,
    physical_factory: PhysicalFactory | None = None,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    if not plans:
        raise ValueError("serial execution requires at least one run")
    root = (output_root or repository_root() / "outputs" / "runs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with physical_execution_lock(root):
        for plan in plans:
            try:
                result = asyncio.run(
                    _execute_one(
                        plan,
                        suite=suite,
                        output_root=root,
                        physical_factory=physical_factory or _real_physical_factory,
                        model_factory=model_factory,
                    )
                )
            except RunAcceptanceFailure as error:
                result = {
                    "profile": plan.profile,
                    "status": "verification_failed",
                    "run_dir": str(error.run_dir),
                    "verification": error.verification,
                }
            results.append(result)
    return {"execution": "serial", "completed_runs": results}


def resume_plan(source_run: Path) -> dict[str, Any]:
    prefix = load_resume_prefix(source_run)
    return {
        "mode": "resume_dry_plan",
        "source_run": str(prefix.source_run),
        "profile": prefix.plan.profile,
        "method": prefix.plan.method_config(),
        "model_provider": prefix.plan.effective_model_provider(),
        "completed_hour": prefix.completed_hour,
        "completed_step": prefix.completed_step,
        "next_step": prefix.next_step,
        "remaining_steps": prefix.plan.evaluation_hours * 4 - prefix.next_step,
        "replay_step_count": prefix.next_step,
        "source_commit": prefix.source_commit,
        "source_prefix_identity": prefix.prefix_identity,
        "fresh_run_and_test_required": True,
    }


def execute_resume(
    source_run: Path,
    *,
    output_root: Path | None = None,
    physical_factory: PhysicalFactory | None = None,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    prefix = load_resume_prefix(source_run)
    root = (output_root or repository_root() / "outputs" / "runs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    with physical_execution_lock(root):
        try:
            result = asyncio.run(
                _execute_one(
                    prefix.plan,
                    suite="resume-run",
                    output_root=root,
                    physical_factory=physical_factory or _real_physical_factory,
                    model_factory=model_factory,
                    resume_prefix=prefix,
                )
            )
        except RunAcceptanceFailure as error:
            result = {
                "profile": prefix.plan.profile,
                "status": "verification_failed",
                "run_dir": str(error.run_dir),
                "verification": error.verification,
                "resume_replay": prefix.lineage(),
            }
    return {"execution": "resume_replay", "completed_runs": [result]}
