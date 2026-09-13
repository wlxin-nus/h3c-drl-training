"""Fail-closed run artifact verification."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

from h3c.agents.contracts import (
    DEFAULT_PER_ZONE_RESERVED_CAP_C,
    executor_response_schema,
    orchestrator_response_schema,
    rationale_length_telemetry,
    reflector_response_schema,
)
from h3c.agents.roles import (
    ModelContractError,
    resolve_executor_memory_model_output,
    resolve_executor_model_output,
    resolve_orchestrator_model_output,
)
from h3c.assurance.action import ACTION_ASSURANCE_ORDER, action_assurance
from h3c.causal.graph import ConfirmedGraph, derive_variant, load_graph, validate_graph
from h3c.control.budget import (
    BUDGET_ABS_TOLERANCE,
    BudgetLedger,
    allocation_fallback_audit,
    site_cap_max,
    validate_allocation,
    validated_fallback_allocation,
)
from h3c.control.program import load_program, program_hash
from h3c.control.validation import validate_candidate
from h3c.experiments.matrix import RunPlan
from h3c.experiments.profiles import load_profile, repository_root
from h3c.experiments.settings import evaluation_start_seconds, load_runtime_contract
from h3c.memory.caol import (
    active_experiences,
    apply_memory_operations,
    build_hourly_cao,
    empty_regime_store,
    resolve_reflector_payload,
    validate_memory_refs,
)
from h3c.memory.ledger import ProgramLedger
from h3c.outputs.artifacts import PERFORMANCE_COLUMNS, STREAM_FILES
from h3c.outputs.metrics import compute_run_metrics
from h3c.runtime.clients import (
    model_logical_call_identity,
    model_request_body,
    model_request_contract,
    model_request_identity,
    normalized_usage,
    provider_neutral_request_identity,
)
from h3c.runtime.comfort import step_reward, step_reward_breakdown
from h3c.runtime.occupancy import (
    hourly_route,
    verify_missing_occupancy_resolution_evidence,
)
from h3c.runtime.protocol import reconstruct_forecast_evidence
from h3c.runtime.resume import recompute_resume_prefix_identity

_SETPOINT_EFFECT_ACTION_FIELDS = {
    "regime_base_setpoints_c",
    "setpoint_offsets_from_regime_base_c",
    "cooling_effects_relative_to_regime_base",
}


def _align_expected_caol_with_persisted_schema(
    expected: dict[str, Any],
    persisted: dict[str, Any],
) -> dict[str, Any]:
    """Project newly derived action facts away only for an intact legacy record."""
    persisted_action = persisted.get("action")
    expected_action = expected.get("action")
    if not isinstance(persisted_action, dict) or not isinstance(expected_action, dict):
        raise ValueError("CAOL action evidence must be an object")
    present = _SETPOINT_EFFECT_ACTION_FIELDS.intersection(persisted_action)
    if present and present != _SETPOINT_EFFECT_ACTION_FIELDS:
        raise ValueError("setpoint-effect action facts must be present as one complete group")
    if not present:
        for field in _SETPOINT_EFFECT_ACTION_FIELDS:
            expected_action.pop(field, None)
    return expected


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number} must contain an object")
        rows.append(value)
    return rows


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _nonnegative_finite_number(value: Any) -> bool:
    return _finite_number(value) and float(value) >= 0


def _same_number(left: Any, right: Any) -> bool:
    return (
        _finite_number(left)
        and _finite_number(right)
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    )


def _same_budget_number(left: Any, right: Any) -> bool:
    return (
        _finite_number(left)
        and _finite_number(right)
        and math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=BUDGET_ABS_TOLERANCE,
        )
    )


def _finite_exact_map(value: Any, keys: set[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == keys
        and all(_finite_number(item) for item in value.values())
    )


def _occupancy_resolution_evidence(
    profile: dict[str, Any],
    method: dict[str, Any],
    manifest: dict[str, Any],
    streams: dict[str, list[dict[str, Any]]],
    forecast_evidence: dict[str, Any],
) -> bool:
    events = [
        row
        for row in streams["timing.jsonl"]
        if row.get("phase") == "occupancy_forecast_missing_value_resolution"
    ]
    hours = int(method["evaluation_hours"])
    evaluation_start = evaluation_start_seconds(profile, method.get("diagnostic_window"))
    try:
        _, expected_events = reconstruct_forecast_evidence(
            profile,
            forecast_evidence,
            hours * 4 + 97,
            forecast_phase="evaluation",
            start_time_seconds=evaluation_start,
            step_seconds=int(profile["control_step_seconds"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return expected_events == events and verify_missing_occupancy_resolution_evidence(
        profile,
        hours,
        manifest.get("occupancy_forecast_missing_value_resolution_count"),
        events,
        evaluation_start_seconds=evaluation_start,
    )


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


def _file_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _performance_rows(path: Path) -> tuple[list[dict[str, str]], bool]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        header_valid = tuple(reader.fieldnames or ()) == PERFORMANCE_COLUMNS
        rows = [dict(row) for row in reader]
    row_schema_valid = all(
        set(row) == set(PERFORMANCE_COLUMNS)
        and all(isinstance(value, str) for value in row.values())
        for row in rows
    )
    return rows, header_valid and row_schema_valid


def _raw_contract(
    row: dict[str, Any],
    *,
    zones: list[str],
    causal_enabled: bool,
    allowed_edge_ids: set[str] | None,
    shared_power_edge_ids: set[str] | None,
    long_term_memory: bool,
    expected_site_cap_c: float,
    expected_per_zone_reserved_cap_c: float,
) -> bool:
    try:
        value = json.loads(row["output"])
        if not isinstance(value, dict):
            return False
        role = row["role"]
        if role == "orchestrator":
            resolve_orchestrator_model_output(
                row["output"],
                zones,
                causal_enabled=causal_enabled,
                allowed_causal_edge_ids=allowed_edge_ids,
                site_causal_edge_ids=shared_power_edge_ids,
                expected_site_cap_c=expected_site_cap_c,
                expected_per_zone_reserved_cap_c=expected_per_zone_reserved_cap_c,
            )
            return True
        if role == "executor":
            if long_term_memory:
                resolve_executor_memory_model_output(row["output"], causal_enabled=causal_enabled)
            else:
                resolve_executor_model_output(row["output"], causal_enabled=causal_enabled)
            return True
        if role == "reflector":
            return resolve_reflector_payload(
                value,
                zones=zones,
                long_term_memory=long_term_memory,
            ).clean
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return False


def _program_replay(
    profile: dict[str, Any],
    method: dict[str, Any],
    updates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    graph: ConfirmedGraph | None,
) -> tuple[bool, bool]:
    zones = list(profile["zones"])
    program_ledgers = {
        zone: ProgramLedger(
            load_program(repository_root() / profile["program"], zone),
            causal_enabled=bool(method["causal_enabled"]),
        )
        for zone in zones
    }
    replay_ok = len(updates) == int(method["evaluation_hours"]) * len(zones)
    settlement_ok = replay_ok
    decision_by_hour = {int(decision.get("hour", -1)): decision for decision in decisions}
    if len(decision_by_hour) != int(method["evaluation_hours"]):
        return False, False
    budget_ledgers: dict[int, BudgetLedger | None] = {}
    expected_order: list[tuple[int, str]] = []
    for hour in range(int(method["evaluation_hours"])):
        if method["coordination_enabled"]:
            try:
                allocation = decision_by_hour[hour]["orchestration"]["allocation_audit"]
                budget_ledgers[hour] = BudgetLedger(allocation, zones)
                order = list(allocation["priority"])
            except (KeyError, TypeError, ValueError):
                return False, False
        else:
            budget_ledgers[hour] = None
            order = zones
        expected_order.extend((hour, zone) for zone in order)
    observed_order = [(row.get("hour"), row.get("zone")) for row in updates]
    if observed_order != expected_order:
        replay_ok = False
        settlement_ok = False

    seen: set[tuple[int, str]] = set()
    for row in updates:
        try:
            zone = str(row["zone"])
            hour = int(row["hour"])
            step = int(row["step"])
            ledger = program_ledgers[zone]
            identity = (hour, zone)
            if identity in seen or step != hour * 4:
                raise ValueError("duplicate or misaligned program decision")
            seen.add(identity)
            base_fields = {
                "hour",
                "step",
                "zone",
                "status",
                "patch",
                "rationale_telemetry",
                "completed_validation_stages",
                "rejection",
                "program_version_before",
                "program_version_after",
                "program_hash_before",
                "program_hash_after",
                "current_program_version",
                "current_program_hash",
                "replay_verified",
            }
            patch = row["patch"]
            if not isinstance(patch, dict):
                raise ValueError("program update patch must be an object")
            expected_fields = set(base_fields)
            if bool(method.get("long_term_memory")):
                expected_fields.add("memory_refs")
            if row.get("status") == "accepted" and patch.get("op") != "no_change":
                expected_fields.add("accepted_update")
            if set(row) != expected_fields:
                raise ValueError("program update fields do not match the exact contract")
            before_version = ledger.version
            before_hash = program_hash(ledger.current_program)
            replay_ok = replay_ok and (
                row.get("program_version_before") == before_version
                and row.get("program_hash_before") == before_hash
            )
            status = row.get("status")
            completed = row.get("completed_validation_stages")
            rejection = row.get("rejection")
            if status == "model_output_rejected":
                settlement_ok = settlement_ok and (
                    row.get("rationale_telemetry") is None
                    and completed == []
                    and isinstance(rejection, dict)
                    and set(rejection) == {"stage", "code", "message", "raw_output"}
                    and rejection.get("stage") == "program_validation"
                    and rejection.get("code") == "model_output_schema_rejected"
                    and patch
                    == {
                        "op": "no_change",
                        "rationale": "tool-generated no_change after model contract rejection",
                    }
                )
            else:
                validation = validate_candidate(
                    patch,
                    ledger.current_program,
                    graph=graph,
                    ledger=budget_ledgers[hour],
                    zone=zone,
                    step=step,
                    causal_enabled=bool(method["causal_enabled"]),
                    coordination_enabled=bool(method["coordination_enabled"]),
                )
                expected_status = "accepted" if validation.accepted else "rejected"
                expected_rejection = (
                    None if validation.rejection is None else validation.rejection.as_dict()
                )
                settlement_ok = settlement_ok and (
                    status == expected_status
                    and patch == validation.patch
                    and row.get("rationale_telemetry")
                    == rationale_length_telemetry(
                        "executor", {"operation": str(patch["rationale"])}
                    )
                    and completed == list(validation.completed_stages)
                    and rejection == expected_rejection
                )
                if validation.accepted and validation.patch["op"] != "no_change":
                    update = ledger.commit(validation.patch, step=step, hour=hour)
                    replay_ok = replay_ok and row.get("accepted_update") == update.as_dict()
            after_hash = program_hash(ledger.current_program)
            replay_ok = replay_ok and (
                row.get("program_version_after") == ledger.version
                and row.get("current_program_version") == ledger.version
                and row.get("program_hash_after") == after_hash
                and row.get("current_program_hash") == after_hash
                and row.get("replay_verified") is True
                and bool(ledger.replay())
            )
        except (KeyError, TypeError, ValueError):
            replay_ok = False
            settlement_ok = False
    expected_identities = {
        (hour, zone) for hour in range(int(method["evaluation_hours"])) for zone in zones
    }
    if method["coordination_enabled"]:
        for hour, budget_ledger in budget_ledgers.items():
            if budget_ledger is None:
                settlement_ok = False
                continue
            settlement_ok = settlement_ok and (
                decision_by_hour[hour].get("energy_budget") == budget_ledger.utilisation()
            )
    return replay_ok and seen == expected_identities, settlement_ok


def _rationale_persistence(
    *,
    method: dict[str, Any],
    raw_calls: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    zones: list[str],
    allowed_edge_ids: set[str] | None,
    shared_power_edge_ids: set[str] | None,
    expected_site_cap_c: float,
    expected_per_zone_reserved_cap_c: float,
) -> bool:
    """Tie every raw Orchestrator/Executor rationale to its parsed artifact."""
    if method["controller"] == "deterministic_baseline":
        return not raw_calls and not updates
    causal_enabled = bool(method["causal_enabled"])
    update_by_scope = {(row.get("hour"), row.get("zone")): row for row in updates}
    decision_by_hour = {row.get("hour"): row for row in decisions}
    if len(update_by_scope) != len(updates) or len(decision_by_hour) != len(decisions):
        return False
    checked_executors = 0
    checked_orchestrators = 0
    for raw in raw_calls:
        role = raw.get("role")
        if role == "executor":
            checked_executors += 1
            parsed = update_by_scope.get((raw.get("hour"), raw.get("zone")))
            if parsed is None:
                return False
            try:
                if bool(method.get("long_term_memory")):
                    patch, telemetry, memory_refs = resolve_executor_memory_model_output(
                        raw["output"], causal_enabled=causal_enabled
                    )
                else:
                    patch, telemetry = resolve_executor_model_output(
                        raw["output"], causal_enabled=causal_enabled
                    )
                    memory_refs = []
            except (KeyError, ModelContractError):
                rejection = parsed.get("rejection")
                if not (
                    parsed.get("status") == "model_output_rejected"
                    and parsed.get("rationale_telemetry") is None
                    and isinstance(rejection, dict)
                    and rejection.get("raw_output") == raw.get("output")
                ):
                    return False
            else:
                stored_patch = parsed.get("patch")
                public_patch_matches = False
                if isinstance(stored_patch, dict):
                    derived_fields = {
                        "expected_effects",
                        "consistent_program_direction_proof",
                    }
                    raw_public = {
                        key: value
                        for key, value in patch.items()
                        if key not in {"causal_edge_ids", *derived_fields}
                    }
                    stored_public = {
                        key: value
                        for key, value in stored_patch.items()
                        if key not in {"causal_edge_ids", *derived_fields}
                    }
                    raw_ids = patch.get("causal_edge_ids", [])
                    stored_ids = stored_patch.get("causal_edge_ids", [])
                    causal_completion_ok = raw_ids == stored_ids
                    if causal_enabled and patch.get("op") != "no_change":
                        causal_completion_ok = (
                            isinstance(raw_ids, list)
                            and isinstance(stored_ids, list)
                            and stored_ids[: len(raw_ids)] == raw_ids
                            and len(stored_ids) - len(raw_ids) in {0, 1}
                            and set(stored_ids[len(raw_ids) :]) <= set(shared_power_edge_ids or ())
                        )
                    public_patch_matches = (
                        raw_public == stored_public
                        and causal_completion_ok
                        and set(stored_patch) - set(patch) <= derived_fields
                    )
                if not (
                    parsed.get("status") != "model_output_rejected"
                    and public_patch_matches
                    and parsed.get("rationale_telemetry") == telemetry
                    and (
                        not bool(method.get("long_term_memory"))
                        or parsed.get("memory_refs", {}).get("reported") == memory_refs
                    )
                ):
                    return False
        elif role == "orchestrator":
            checked_orchestrators += 1
            decision = decision_by_hour.get(raw.get("hour"))
            if decision is None or not isinstance(decision.get("orchestration"), dict):
                return False
            audit = decision["orchestration"]
            try:
                allocation, telemetry = resolve_orchestrator_model_output(
                    raw["output"],
                    zones,
                    causal_enabled=causal_enabled,
                    allowed_causal_edge_ids=allowed_edge_ids,
                    site_causal_edge_ids=shared_power_edge_ids,
                    expected_site_cap_c=expected_site_cap_c,
                    expected_per_zone_reserved_cap_c=expected_per_zone_reserved_cap_c,
                )
            except (KeyError, ModelContractError):
                rejection = audit.get("raw_contract", {}).get("rejection")
                if not (
                    audit.get("status") == "fallback"
                    and audit.get("rationale_telemetry") is None
                    and isinstance(rejection, dict)
                    and rejection.get("raw_output") == raw.get("output")
                ):
                    return False
            else:
                if not (
                    audit.get("status") == "accepted"
                    and audit.get("allocation_audit") == allocation
                    and audit.get("rationale_telemetry") == telemetry
                ):
                    return False
    hours = int(method["evaluation_hours"])
    return checked_executors == hours * len(zones) and checked_orchestrators == (
        hours if method["coordination_enabled"] else 0
    )


_CLOCK_VALUE = r"(?:next day )?\d{2}:\d{2}"
_INTERNAL_AGENT_COORDINATES = (
    "decision_hour",
    "latest_completed_step",
    "target_action_steps",
    "sample_index",
    "physical_step",
    '"step":',
    '"step_ahead":',
    "time_seconds",
)


def _clock_minute(value: str) -> int:
    match = re.fullmatch(r"(next day )?(\d{2}):(\d{2})", value)
    if match is None:
        raise ValueError(f"invalid clock value: {value}")
    hour = int(match.group(2))
    minute = int(match.group(3))
    if hour >= 24 or minute >= 60:
        raise ValueError(f"invalid clock value: {value}")
    return (1440 if match.group(1) else 0) + hour * 60 + minute


def _clock_field(text: str, name: str) -> str:
    match = re.search(rf"(?m)^{re.escape(name)}: ({_CLOCK_VALUE})$", text)
    if match is None:
        raise ValueError(f"missing clock field: {name}")
    return match.group(1)


def _clock_list_field(text: str, name: str) -> list[str]:
    match = re.search(rf"(?m)^{re.escape(name)}: (\[[^\r\n]+\])$", text)
    if match is None:
        raise ValueError(f"missing clock list: {name}")
    values = json.loads(match.group(1))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"invalid clock list: {name}")
    return values


def _interval_field(text: str, name: str) -> tuple[str, str]:
    match = re.search(
        rf"(?m)^{re.escape(name)}: \[({_CLOCK_VALUE}), ({_CLOCK_VALUE})\)$",
        text,
    )
    if match is None:
        raise ValueError(f"missing interval field: {name}")
    return match.group(1), match.group(2)


def _clock_sequence_is_valid(
    *,
    interval: tuple[str, str],
    action_times: list[str],
    outcome_times: list[str],
) -> bool:
    if len(action_times) != 4 or len(outcome_times) != 4:
        return False
    try:
        start, end = map(_clock_minute, interval)
        actions = [_clock_minute(value) for value in action_times]
        outcomes = [_clock_minute(value) for value in outcome_times]
    except ValueError:
        return False
    return (
        end - start == 60
        and actions == [start + offset for offset in (0, 15, 30, 45)]
        and outcomes == [action + 15 for action in actions]
        and outcomes[-1] == end
    )


def _agent_surface_has_internal_coordinates(text: str) -> bool:
    return any(token in text for token in _INTERNAL_AGENT_COORDINATES) or bool(
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    )


def _decision_clock_surface_is_valid(text: str, *, expect_working_memory: bool) -> bool:
    try:
        current_time = _clock_field(text, "current_time")
        interval = _interval_field(text, "control_interval")
        action_times = _clock_list_field(text, "action_times")
        outcome_times = _clock_list_field(text, "forecast_outcome_times")
        if (
            "### DECISION WINDOW" not in text
            or "control_period: 15 min" not in text
            or "coordination_period: 60 min" not in text
            or current_time != interval[0]
            or not _clock_sequence_is_valid(
                interval=interval,
                action_times=action_times,
                outcome_times=outcome_times,
            )
            or _agent_surface_has_internal_coordinates(text)
        ):
            return False
        if not expect_working_memory:
            return True
        completed_interval = _interval_field(text, "completed_interval")
        state_times = _clock_list_field(text, "state_times")
        if len(state_times) != 4:
            return False
        completed_start, completed_end = map(_clock_minute, completed_interval)
        rendered_state_times = [_clock_minute(value) for value in state_times]
        return (
            completed_end % 1440 == _clock_minute(current_time) % 1440
            and completed_end - completed_start == 60
            and rendered_state_times == [completed_start + offset for offset in (0, 15, 30, 45)]
            and "control_action_history:" in text
            and "previous_decision_forecast:" in text
            and "derived_features:" in text
            and "RECENT OUTCOME SUMMARY" not in text
        )
    except (json.JSONDecodeError, ValueError):
        return False


def _completed_interval_clock_surface_is_valid(text: str) -> bool:
    try:
        interval = _interval_field(text, "interval")
        action_times = _clock_list_field(text, "action_times")
        outcome_times = _clock_list_field(text, "outcome_times")
    except (json.JSONDecodeError, ValueError):
        return False
    return (
        "### COMPLETED CONTROL INTERVAL" in text
        and "control_period: 15 min" in text
        and _clock_sequence_is_valid(
            interval=interval,
            action_times=action_times,
            outcome_times=outcome_times,
        )
        and not _agent_surface_has_internal_coordinates(text)
    )


def _caol_memory_checks(
    *,
    method: dict[str, Any],
    zones: list[str],
    zone_steps: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    caol_rows: list[dict[str, Any]],
    crud_rows: list[dict[str, Any]],
    raw_calls: list[dict[str, Any]],
) -> dict[str, bool]:
    hours = int(method["evaluation_hours"])
    controller_is_agent = method["controller"] == "h3c_agent"
    long_term_memory = bool(method.get("long_term_memory", False))
    update_by_scope = {(int(row["hour"]), str(row["zone"])): row for row in updates}
    persisted_by_scope = {(int(row["hour"]), str(row["zone"])): row for row in caol_rows}
    lesson_by_scope: dict[tuple[int, str], str] = {}
    for decision in decisions:
        for row in decision.get("reflector_summary", []):
            if isinstance(row, dict) and set(row) == {"zone", "lesson"}:
                lesson_by_scope[(int(decision["hour"]), str(row["zone"]))] = str(row["lesson"])

    expected_cao: dict[tuple[int, str], dict[str, Any]] = {}
    caol_replayed = (
        len(caol_rows) == hours * len(zones)
        and len(persisted_by_scope) == len(caol_rows)
        and len(decisions) == hours
    )
    try:
        for hour in range(hours):
            for zone in zones:
                step_rows = sorted(
                    (
                        row
                        for row in zone_steps
                        if int(row["hour"]) == hour and str(row["zone"]) == zone
                    ),
                    key=lambda row: int(row["step"]),
                )
                if controller_is_agent:
                    decision = update_by_scope[(hour, zone)]
                else:
                    decision = {
                        "hour": hour,
                        "step": hour * 4,
                        "zone": zone,
                        "status": "not_called",
                        "patch": {"op": "no_change", "rationale": "controller has no model"},
                        "current_program_version": 0,
                        "rejection": None,
                    }
                expected = build_hourly_cao(
                    hour=hour,
                    zone=zone,
                    step_rows=step_rows,
                    program_decision=decision,
                )
                expected_cao[(hour, zone)] = expected
                persisted = persisted_by_scope[(hour, zone)]
                persisted_cao = {key: value for key, value in persisted.items() if key != "lesson"}
                expected = _align_expected_caol_with_persisted_schema(expected, persisted_cao)
                expected_lesson = lesson_by_scope.get((hour, zone))
                caol_replayed = caol_replayed and persisted_cao == expected
                caol_replayed = caol_replayed and (
                    (expected_lesson is None and "lesson" not in persisted)
                    or persisted.get("lesson") == expected_lesson
                )
    except (KeyError, TypeError, ValueError):
        caol_replayed = False

    caol_prompt_surface = True
    if controller_is_agent:
        memory_hours = int(method["working_memory_hours"])
        for raw in raw_calls:
            role = raw.get("role")
            user = str(raw.get("user", ""))
            hour = int(raw.get("hour", -1))
            if role in {"orchestrator", "executor"}:
                expected_block = hour >= memory_hours
                caol_prompt_surface = caol_prompt_surface and (
                    ("### WORKING MEMORY" in user) == expected_block
                )
                caol_prompt_surface = caol_prompt_surface and _decision_clock_surface_is_valid(
                    user,
                    expect_working_memory=expected_block,
                )
            elif role == "reflector":
                caol_prompt_surface = (
                    caol_prompt_surface
                    and "### WORKING MEMORY" not in user
                    and _completed_interval_clock_surface_is_valid(user)
                )

    off_tokens = (
        "ACTIVE LONG-TERM EXPERIENCES",
        "ACTIVE LONG-TERM EXPERIENCE SLOTS",
        "EMPTY LONG-TERM EXPERIENCE SLOTS",
        "ELIGIBLE LONG-TERM EXPERIENCE SLOTS",
        "memory_refs",
        "memory_operations",
        "expected_revision",
    )
    off_surface = _canonical(
        {"raw_calls": raw_calls, "updates": updates, "decisions": decisions, "crud": crud_rows}
    )
    memory_off_isolation = (
        True
        if long_term_memory
        else not crud_rows and all(token not in off_surface for token in off_tokens)
    )

    memory_store_replayed = True
    memory_refs_valid = True
    memory_crud_clean = True
    if long_term_memory:
        store = empty_regime_store(zones)
        audit_by_hour: dict[int, list[dict[str, Any]]] = {}
        for row in crud_rows:
            audit_by_hour.setdefault(int(row.get("hour", -1)), []).append(row)
        try:
            for hour in range(hours):
                for zone in zones:
                    reference_audit = update_by_scope[(hour, zone)]["memory_refs"]
                    exposed = active_experiences(store, zone)
                    valid, invalid = validate_memory_refs(reference_audit["reported"], exposed)
                    memory_refs_valid = memory_refs_valid and (
                        reference_audit["available"] is True
                        and reference_audit["exposed"] == exposed
                        and reference_audit["valid"] == valid
                        and reference_audit["invalid"] == invalid == []
                    )
                rows = audit_by_hour.get(hour, [])
                operations = {
                    str(row["zone"]): row["requested_operation"]
                    for row in rows
                    if isinstance(row.get("requested_operation"), dict)
                }
                observed = {
                    zone: list(expected_cao[(hour, zone)]["context"]["regime_step_coverage"])
                    for zone in zones
                }
                store, expected_audits = apply_memory_operations(
                    store,
                    operations,
                    zones=zones,
                    hour=hour,
                    observed_regimes=observed,
                )
                ordered_rows = sorted(rows, key=lambda row: zones.index(str(row["zone"])))
                memory_store_replayed = memory_store_replayed and ordered_rows == expected_audits
                memory_crud_clean = memory_crud_clean and all(
                    row.get("status") == "accepted" for row in rows
                )
            memory_store_replayed = memory_store_replayed and len(crud_rows) == hours * len(zones)
        except (KeyError, TypeError, ValueError):
            memory_store_replayed = False
            memory_refs_valid = False
            memory_crud_clean = False
        for raw in raw_calls:
            role = raw.get("role")
            if role == "orchestrator":
                memory_store_replayed = memory_store_replayed and all(
                    token not in str(raw.get("system", "")) + str(raw.get("user", ""))
                    for token in off_tokens
                )
            elif role == "reflector":
                reflector_user = str(raw.get("user", ""))
                memory_store_replayed = memory_store_replayed and (
                    (
                        "ACTIVE LONG-TERM EXPERIENCE SLOTS" in reflector_user
                        or "EMPTY LONG-TERM EXPERIENCE SLOTS" in reflector_user
                    )
                    and "ELIGIBLE LONG-TERM EXPERIENCE SLOTS" not in reflector_user
                    and "memory_operations" in str(raw.get("system", ""))
                )
            elif role == "executor":
                memory_store_replayed = memory_store_replayed and "memory_refs" in str(
                    raw.get("system", "")
                )
    else:
        memory_store_replayed = not crud_rows

    return {
        "caol_replayed": caol_replayed,
        "caol_working_memory_surface": caol_prompt_surface,
        "memory_off_isolation": memory_off_isolation,
        "memory_store_replayed": memory_store_replayed,
        "memory_refs_valid": memory_refs_valid,
        "memory_crud_clean": memory_crud_clean,
    }


EXECUTION_CHECKS = {
    "required_artifacts",
    "physical_protocol",
    "test_id_continuity",
    "timeline_and_stream_alignment",
    "occupancy_forecast_missing_value_resolution",
    "action_assurance_recomputed",
    "program_replay_recomputed",
    "agent_call_counts",
    "agent_call_alignment",
    "model_identity",
    "model_request_identity",
    "manifest_identity",
    "conditioning_prefix_identity",
    "evaluation_boundary_identity",
    "metrics_recomputed",
    "objective_feedback_replayed",
    "model_transport_retry_accounting",
    "secret_exposure_count_zero",
    "orchestration_resolution_recomputed",
    "deterministic_settlement",
    "thinking_route",
    "causal_surface",
    "caol_replayed",
    "caol_working_memory_surface",
    "memory_off_isolation",
    "memory_store_replayed",
    "resume_replay_integrity",
}

MODEL_CHECKS = {
    "json_schema",
    "usage_contract",
    "transport_error_count_zero",
    "coordination_surface",
    "fallback_count_zero",
    "role_contract_audit",
    "model_finish_clean",
    "rationale_persistence",
    "memory_refs_valid",
    "memory_crud_clean",
}


def _classified(checks: dict[str, bool], *, performance_pass: bool = False) -> dict[str, Any]:
    execution_integrity = all(checks.get(name, False) for name in EXECUTION_CHECKS)
    model_contract_clean = all(checks.get(name, False) for name in MODEL_CHECKS)
    if not execution_integrity:
        classification = "RUN-INVALID"
    elif model_contract_clean:
        classification = "RELEASE-PASS"
    else:
        classification = "EXECUTION-HEALTHY-MODEL-CONTRACT-DEGRADED"
    failed_execution = sorted(name for name in EXECUTION_CHECKS if not checks.get(name, False))
    failed_model = sorted(name for name in MODEL_CHECKS if not checks.get(name, False))
    errors: list[str] = []
    if failed_execution:
        errors.append(f"execution integrity failed checks: {failed_execution}")
    if failed_model:
        errors.append(f"model contract failed checks: {failed_model}")
    return {
        "passed": classification == "RELEASE-PASS",
        "completion_eligible": classification != "RUN-INVALID",
        "execution_integrity": execution_integrity,
        "model_contract_clean": model_contract_clean,
        "trajectory_status": ("EXECUTION-HEALTHY" if execution_integrity else "EXECUTION-INVALID"),
        "model_contract_status": "CLEAN" if model_contract_clean else "DEGRADED",
        "performance_status": "REWARD-PMV-PASS" if performance_pass else "METHOD-DEGRADED",
        "classification": classification,
        "checks": checks,
        "errors": errors,
    }


def _performance_evaluation(
    profile: dict[str, Any], method: dict[str, Any], metrics: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    criteria_document = _object(
        repository_root() / "configs" / "evaluation" / "reward_pmv_release_criteria.json"
    )
    if set(criteria_document) != {"criteria_schema", "schema_version", "application", "cases"}:
        raise ValueError("reward/PMV release criteria have an invalid root contract")
    if (
        criteria_document["criteria_schema"] != "h3c_reward_pmv_release_criteria"
        or criteria_document["schema_version"] != 1
        or criteria_document["application"] != "terminal_evaluation_only_never_runtime_admission"
    ):
        raise ValueError("reward/PMV release criteria identity is invalid")
    profile_name = str(profile["profile"])
    criteria = criteria_document["cases"].get(profile_name)
    if not isinstance(criteria, dict) or set(criteria) != {
        "formal_evaluation_hours",
        "reward_strictly_greater_than",
        "occupied_peak_absolute_pmv_at_most",
    }:
        raise ValueError("case reward/PMV release criteria are missing or malformed")
    physical = metrics["physical"]
    reward = float(physical["reward"])
    peak = float(physical["occupied_peak_absolute_pmv"])
    formal_hours = int(criteria["formal_evaluation_hours"])
    is_formal_window = int(method["evaluation_hours"]) == formal_hours
    reward_pass = reward > float(criteria["reward_strictly_greater_than"])
    peak_pass = peak <= float(criteria["occupied_peak_absolute_pmv_at_most"])
    passed = is_formal_window and reward_pass and peak_pass
    return passed, {
        "criteria_identity": _identity(criteria_document),
        "application": criteria_document["application"],
        "profile": profile_name,
        "formal_window": is_formal_window,
        "observed": {
            "evaluation_hours": int(method["evaluation_hours"]),
            "reward": reward,
            "occupied_peak_absolute_pmv": peak,
        },
        "criteria": criteria,
        "checks": {
            "formal_evaluation_hours": is_formal_window,
            "reward_strictly_greater_than": reward_pass,
            "occupied_peak_absolute_pmv_at_most": peak_pass,
        },
        "passed": passed,
    }


def verify_run(
    run_dir: Path,
    *,
    require_completion: bool = True,
    historical_runtime_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = run_dir.resolve()
    checks: dict[str, bool] = {}
    required = {
        "resolved_config.yaml",
        "manifest.json",
        "metrics.json",
        "forecast_inputs.json",
        *STREAM_FILES,
        "performance.csv",
    }
    missing = sorted(name for name in required if not (directory / name).is_file())
    checks["required_artifacts"] = not missing
    if missing:
        result = _classified(checks)
        result["errors"] = [f"missing artifacts: {missing}", *result["errors"]]
        return result

    try:
        resolved = _object(directory / "resolved_config.yaml")
        manifest = _object(directory / "manifest.json")
        recorded_metrics = _object(directory / "metrics.json")
        forecast_evidence = _object(directory / "forecast_inputs.json")
        profile = resolved["case_profile"]
        method = resolved["method"]
        zones = list(profile["zones"])
        zone_set = set(zones)
        resolved_site_cap = site_cap_max(zones)
        resolved_per_zone_reserved_cap = DEFAULT_PER_ZONE_RESERVED_CAP_C
        hours = int(method["evaluation_hours"])
        expected_steps = hours * 4
        streams = {name: _rows(directory / name) for name in STREAM_FILES}
        performance, header_valid = _performance_rows(directory / "performance.csv")
        conditioning = streams["physical_conditioning.jsonl"]
        zone_steps = streams["zone_steps.jsonl"]
        decisions = streams["hourly_decisions.jsonl"]
        updates = streams["program_updates.jsonl"]
        calls = streams["agent_calls.jsonl"]
        raw_calls = streams["raw_model_io.jsonl"]
        model_attempts = streams["model_request_attempts.jsonl"]
        caol_rows = streams["caol_records.jsonl"]
        crud_rows = streams["long_term_memory_crud.jsonl"]

        protocol = profile["protocol"]
        conditioning_count = 0
        lifecycle = manifest["lifecycle"]
        conditioning_test_ids = {row.get("test_id") for row in conditioning}
        checks["physical_protocol"] = (
            lifecycle
            == {
                "initialize_count": 1,
                "stop_count": 1,
                "test_id_changes": 0,
                "conditioning_advance_count": conditioning_count,
            }
            and len(conditioning) == conditioning_count
            and not conditioning_test_ids
            and protocol["initialization_mode"] == "evaluation_start_internal_warmup"
            and protocol["internal_warmup_days"] == 7
        )

        evaluation_start = evaluation_start_seconds(profile, method.get("diagnostic_window"))
        conditioning_start = evaluation_start - conditioning_count * 900
        evaluation_end = evaluation_start + expected_steps * 900
        lifecycle_events = [
            row for row in streams["timing.jsonl"] if row.get("phase") == "physical_lifecycle"
        ]
        lifecycle_contract = [
            {
                "phase": "physical_lifecycle",
                "event": "initialized",
                "time_seconds": conditioning_start,
                "warmup_period_seconds": int(protocol["internal_warmup_days"]) * 86400,
            },
            {
                "phase": "physical_lifecycle",
                "event": "evaluation_started",
                "time_seconds": evaluation_start,
            },
            {
                "phase": "physical_lifecycle",
                "event": "evaluation_completed",
                "time_seconds": evaluation_end,
            },
            {
                "phase": "physical_lifecycle",
                "event": "stopped",
                "time_seconds": evaluation_end,
            },
        ]
        boundary_events = [
            row for row in streams["timing.jsonl"] if row.get("phase") == "evaluation_boundary"
        ]
        evaluation_test_ids = {row.get("test_id") for row in zone_steps}
        lifecycle_test_ids = {row.get("test_id") for row in lifecycle_events}
        boundary_test_ids = {row.get("test_id") for row in boundary_events}
        all_test_id_sets = (evaluation_test_ids, lifecycle_test_ids, boundary_test_ids)
        checks["test_id_continuity"] = (
            all(len(values) == 1 for values in all_test_id_sets)
            and len(set().union(*all_test_id_sets)) == 1
            and all(
                {key: value for key, value in row.items() if key != "test_id"} == expected
                and isinstance(row.get("test_id"), str)
                and bool(row["test_id"])
                for row, expected in zip(lifecycle_events, lifecycle_contract, strict=True)
            )
            and len(lifecycle_events) == len(lifecycle_contract)
        )
        conditioning_timeline = conditioning == []
        conditioning_contract = conditioning == []
        step_groups: dict[int, list[dict[str, Any]]] = {}
        for row in zone_steps:
            step_groups.setdefault(int(row.get("step", -1)), []).append(row)
        timeline_ok = header_valid and len(performance) == expected_steps
        timeline_ok = timeline_ok and len(zone_steps) == expected_steps * len(zones)
        timeline_ok = timeline_ok and len(decisions) == hours
        timeline_ok = timeline_ok and [row.get("hour") for row in decisions] == list(range(hours))
        zone_step_fields = {
            "hour",
            "step",
            "zone",
            "test_id",
            "action_time_seconds",
            "outcome_time_seconds",
            "observation",
            "interpreter",
            "action_assurance",
            "final_setpoint_c",
            "outcome",
        }
        outcome_fields = {
            "zone_temperature_c",
            "pmv",
            "effective_occupancy",
            "power_w",
            "cost",
        }
        reward_feedback_expected = (
            method["controller"] == "h3c_agent"
            and manifest.get("schema_version") == 5
            and manifest.get("objective_feedback_contract")
            == "completed_interval_reward_breakdown_v1"
        )
        expected_outcome_fields = set(outcome_fields)
        if reward_feedback_expected:
            expected_outcome_fields.add("objective_feedback")
        objective_feedback_ok = True
        for step in range(expected_steps):
            rows = step_groups.get(step, [])
            if {row.get("zone") for row in rows} != zone_set or len(rows) != len(zones):
                timeline_ok = False
                continue
            by_zone = {str(row["zone"]): row for row in rows}
            performance_row = performance[step]
            try:
                temperatures = json.loads(performance_row["zone_temperatures_c"])
                setpoints = json.loads(performance_row["zone_setpoints_c"])
                pmv_values = json.loads(performance_row["zone_pmv"])
                occupancy_values = json.loads(performance_row["zone_occupancy"])
                power_w = float(performance_row["total_power_w"])
                step_cost = float(performance_row["step_cost"])
                step_reward_value = float(performance_row["step_reward"])
                price_values = [
                    float(by_zone[zone]["observation"]["electricity_price"]) for zone in zones
                ]
                expected_reward = step_reward(
                    cost=step_cost,
                    pmv=[float(value) for value in pmv_values],
                    occupancy=[float(value) for value in occupancy_values],
                    setpoints_c=[float(value) for value in setpoints],
                    previous_setpoints_c=[
                        float(by_zone[zone]["observation"]["last_setpoint"]) for zone in zones
                    ],
                    objective=profile["objective"],
                )
                expected_breakdown = step_reward_breakdown(
                    cost=step_cost,
                    pmv=[float(value) for value in pmv_values],
                    occupancy=[float(value) for value in occupancy_values],
                    setpoints_c=[float(value) for value in setpoints],
                    previous_setpoints_c=[
                        float(by_zone[zone]["observation"]["last_setpoint"]) for zone in zones
                    ],
                    objective=profile["objective"],
                    zone_names=zones,
                )
                if reward_feedback_expected:
                    expected_feedback_fields = {
                        "site_step_reward",
                        "site_energy_penalty",
                        "site_comfort_penalty",
                        "site_smoothness_penalty",
                        "zone_comfort_penalty_contribution",
                        "zone_smoothness_penalty_contribution",
                    }
                    for zone in zones:
                        feedback = by_zone[zone]["outcome"].get("objective_feedback")
                        objective_feedback_ok = objective_feedback_ok and (
                            isinstance(feedback, dict)
                            and set(feedback) == expected_feedback_fields
                            and _same_number(feedback["site_step_reward"], expected_reward)
                            and _same_number(
                                feedback["site_energy_penalty"],
                                expected_breakdown["site_energy_penalty"],
                            )
                            and _same_number(
                                feedback["site_comfort_penalty"],
                                expected_breakdown["site_comfort_penalty"],
                            )
                            and _same_number(
                                feedback["site_smoothness_penalty"],
                                expected_breakdown["site_smoothness_penalty"],
                            )
                            and _same_number(
                                feedback["zone_comfort_penalty_contribution"],
                                expected_breakdown["zone_comfort_penalty_contributions"][zone],
                            )
                            and _same_number(
                                feedback["zone_smoothness_penalty_contribution"],
                                expected_breakdown["zone_smoothness_penalty_contributions"][zone],
                            )
                        )
                timeline_ok = timeline_ok and (
                    performance_row["time_seconds"] == str(evaluation_start + step * 900)
                    and int(performance_row["step"]) == step
                    and int(performance_row["hour"]) == step // 4
                    and all(
                        set(row) == zone_step_fields
                        and isinstance(row["observation"], dict)
                        and isinstance(row["interpreter"], dict)
                        and isinstance(row["action_assurance"], dict)
                        and isinstance(row["outcome"], dict)
                        and set(row["outcome"]) == expected_outcome_fields
                        and all(_finite_number(row["outcome"][field]) for field in outcome_fields)
                        and _finite_number(row["final_setpoint_c"])
                        for row in rows
                    )
                    and all(
                        row["hour"] == step // 4
                        and row["action_time_seconds"] == evaluation_start + step * 900
                        and row["outcome_time_seconds"] == evaluation_start + (step + 1) * 900
                        for row in rows
                    )
                    and temperatures
                    == [by_zone[zone]["outcome"]["zone_temperature_c"] for zone in zones]
                    and setpoints == [by_zone[zone]["final_setpoint_c"] for zone in zones]
                    and pmv_values == [by_zone[zone]["outcome"]["pmv"] for zone in zones]
                    and occupancy_values
                    == [by_zone[zone]["outcome"]["effective_occupancy"] for zone in zones]
                    and all(_same_number(value, price_values[0]) for value in price_values)
                    and _same_number(step_cost, power_w * 0.25 / 1000.0 * price_values[0])
                    and _same_number(step_reward_value, expected_reward)
                    and all(
                        _same_number(row["outcome"]["power_w"], power_w)
                        and _same_number(row["outcome"]["cost"], step_cost)
                        and _same_number(
                            row["outcome"]["effective_occupancy"],
                            row["observation"]["current_occupancy"],
                        )
                        for row in rows
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                timeline_ok = False
        checks["timeline_and_stream_alignment"] = (
            conditioning_timeline and conditioning_contract and timeline_ok
        )
        checks["objective_feedback_replayed"] = objective_feedback_ok
        checks["occupancy_forecast_missing_value_resolution"] = _occupancy_resolution_evidence(
            profile, method, manifest, streams, forecast_evidence
        )

        causal_enabled = bool(method["causal_enabled"])
        graph: ConfirmedGraph | None = None
        allowed_edge_ids: set[str] | None = None
        shared_power_edge_ids: set[str] | None = None
        if causal_enabled:
            try:
                graph_object = resolved["resolved_graph"]
                graph = validate_graph(graph_object)
                canonical = load_graph(repository_root() / profile["graph"])
                mutation = method.get("graph_mutation")
                expected_graph = (
                    canonical if mutation is None else derive_variant(canonical, mutation)
                )
                checks["causal_surface"] = (
                    graph.resolved() == expected_graph.resolved()
                    and graph.profile == profile["profile"]
                    and graph.zones == tuple(zones)
                )
                allowed_edge_ids = set(graph.by_id)
                shared_power_edge_ids = {
                    edge.identifier for edge in graph.edges if edge.target == "power_meters"
                }
            except (KeyError, TypeError, ValueError):
                checks["causal_surface"] = False
        else:
            agent_surface = _canonical(
                {
                    "calls": calls,
                    "raw": raw_calls,
                    "updates": updates,
                    "decisions": decisions,
                }
            ).lower()
            checks["causal_surface"] = (
                "resolved_graph" not in resolved
                and "graph_mutation" not in resolved
                and "causal" not in agent_surface
                and re.search(r"\bce_[0-9a-f]{8}\b", agent_surface) is None
            )

        assurance_ok = True
        for row in zone_steps:
            try:
                final_setpoint, audit = action_assurance(row["interpreter"], row["observation"])
                assurance_ok = assurance_ok and (
                    audit["order"] == list(ACTION_ASSURANCE_ORDER)
                    and audit == row["action_assurance"]
                    and final_setpoint == row["final_setpoint_c"]
                )
            except (KeyError, TypeError, ValueError):
                assurance_ok = False
        checks["action_assurance_recomputed"] = assurance_ok

        if method["controller"] == "h3c_agent":
            replay_ok, settlement_ok = _program_replay(profile, method, updates, decisions, graph)
        else:
            replay_ok, settlement_ok = not updates, not updates
        checks["program_replay_recomputed"] = replay_ok
        checks["deterministic_settlement"] = settlement_ok
        checks.update(
            _caol_memory_checks(
                method=method,
                zones=zones,
                zone_steps=zone_steps,
                updates=updates,
                decisions=decisions,
                caol_rows=caol_rows,
                crud_rows=crud_rows,
                raw_calls=raw_calls,
            )
        )

        expected_calls = (
            0
            if method["controller"] == "deterministic_baseline"
            else hours * (len(zones) + 1 + int(bool(method["coordination_enabled"])))
        )
        checks["agent_call_counts"] = (
            manifest.get("expected_agent_calls") == expected_calls
            and len(calls) == expected_calls
            and len(raw_calls) == expected_calls
        )
        alignment_keys = (
            "role",
            "hour",
            "step",
            "zone",
            "call_ordinal",
            "thinking_mode",
            "logical_call_identity",
            "request_identity",
            "provider_neutral_request_identity",
            "attempt_count",
            "transport_retry_count",
            "request_model",
            "response_model",
            "model_provider",
            "finish_reason",
            "usage",
            "elapsed_seconds",
        )
        call_fields = {
            "hour",
            "step",
            "call_ordinal",
            "role",
            "thinking_mode",
            "logical_call_identity",
            "request_identity",
            "provider_neutral_request_identity",
            "attempt_count",
            "transport_retry_count",
            "request_model",
            "response_model",
            "model_provider",
            "finish_reason",
            "usage",
            "provider_usage",
            "elapsed_seconds",
            "status",
        }
        raw_fields = (call_fields - {"status"}) | {
            "system",
            "user",
            "output",
            "request_parameters",
        }
        call_by_identity = {str(row.get("logical_call_identity")): row for row in calls}
        raw_by_identity = {str(row.get("logical_call_identity")): row for row in raw_calls}
        unique_call_identities = (
            len(call_by_identity) == len(calls)
            and len(raw_by_identity) == len(raw_calls)
            and set(call_by_identity) == set(raw_by_identity)
            and None not in {row.get("logical_call_identity") for row in calls}
            and None not in {row.get("logical_call_identity") for row in raw_calls}
        )
        call_alignment = unique_call_identities and all(
            set(call) == call_fields | ({"zone"} if call.get("role") == "executor" else set())
            and set(raw) == raw_fields | ({"zone"} if raw.get("role") == "executor" else set())
            and all(call.get(key) == raw.get(key) for key in alignment_keys)
            for identity, call in call_by_identity.items()
            for raw in (raw_by_identity[identity],)
        )
        expected_call_surface: list[tuple[int, str, int, int, str | None]] = []
        if method["controller"] == "h3c_agent":
            call_ordinal = 0
            for hour in range(hours):
                if method["coordination_enabled"]:
                    expected_call_surface.append(
                        (call_ordinal, "orchestrator", hour, hour * 4, None)
                    )
                    call_ordinal += 1
                for zone in zones:
                    expected_call_surface.append((call_ordinal, "executor", hour, hour * 4, zone))
                    call_ordinal += 1
                expected_call_surface.append((call_ordinal, "reflector", hour, hour * 4 + 3, None))
                call_ordinal += 1
        observed_surface = sorted(
            (
                row.get("call_ordinal"),
                row.get("role"),
                row.get("hour"),
                row.get("step"),
                row.get("zone"),
            )
            for row in calls
        )
        checks["agent_call_alignment"] = (
            call_alignment and observed_surface == expected_call_surface
        )
        runtime = (
            historical_runtime_contract
            if historical_runtime_contract is not None
            else load_runtime_contract()
        )
        selected_provider = resolved.get("model_provider")
        provider_contract = (
            runtime["model"]["providers"].get(selected_provider)
            if isinstance(selected_provider, str)
            else None
        )
        raw_model_name = None if provider_contract is None else provider_contract["model"]
        model_name = raw_model_name if isinstance(raw_model_name, str) else ""
        provider_response_format = (
            str(provider_contract["response_format"])
            if provider_contract is not None
            else "json_object"
        )
        provider_retryable_http_errors = (
            {f"http_{code}" for code in provider_contract["retryable_status_codes"]}
            if provider_contract is not None
            else set()
        )
        checks["model_identity"] = (
            all(
                row.get("status") == "received"
                and row.get("request_model") == model_name
                and row.get("response_model") == model_name
                and row.get("model_provider") == selected_provider
                and _finite_number(row.get("elapsed_seconds"))
                and float(row["elapsed_seconds"]) >= 0
                for row in calls
            )
            if calls
            else method["controller"] == "deterministic_baseline"
        )
        checks["model_finish_clean"] = (
            all(row.get("finish_reason") not in (None, "length") for row in calls)
            if calls
            else method["controller"] == "deterministic_baseline"
        )

        identity_ok = False
        try:
            registered_profile = load_profile(str(profile["profile"]))
            plan = RunPlan(
                profile=str(profile["profile"]),
                controller=str(method["controller"]),
                working_memory_hours=method["working_memory_hours"],
                causal_enabled=method["causal_enabled"],
                coordination_enabled=method["coordination_enabled"],
                thinking_policy=str(method["thinking_policy"]),
                graph_mutation=method.get("graph_mutation"),
                evaluation_hours=method["evaluation_hours"],
                long_term_memory=bool(method.get("long_term_memory", False)),
                model_provider=(str(selected_provider) if selected_provider is not None else None),
                diagnostic_window=method.get("diagnostic_window"),
            )
            execution_identity = resolved["execution_identity"]
            resume_lineage = resolved.get("resume_replay")
            expected_execution_fields = {
                "plan_identity",
                "source_commit",
                "runtime_contract",
                "physical_endpoint_identity",
                "dispatch_mode",
            }
            if plan.controller == "h3c_agent":
                expected_execution_fields.update(
                    {"model_provider", "model_name", "model_endpoint_identity"}
                )
                if manifest.get("schema_version") == 5:
                    expected_execution_fields.add("objective_feedback_contract")
            if resume_lineage is not None:
                expected_execution_fields.add("resume_replay_identity")
            expected_resolved_fields = {
                "case_profile",
                "method",
                "runtime_contract",
                "execution_identity",
            }
            if plan.controller == "h3c_agent":
                expected_resolved_fields.add("model_provider")
            if resume_lineage is not None:
                expected_resolved_fields.add("resume_replay")
            if plan.causal_enabled:
                expected_resolved_fields.add("resolved_graph")
            if plan.graph_mutation is not None:
                expected_resolved_fields.add("graph_mutation")
            expected_manifest_fields = {
                "manifest_schema",
                "schema_version",
                "run_identity",
                "source_commit",
                "controller",
                "dispatch_mode",
                "expected_agent_calls",
                "retry_count",
                "transport_error_count",
                "fallback_count",
                "secret_exposure_count",
                "secret_scan_status",
                "occupancy_forecast_missing_value_resolution_count",
                "program_replay_verified",
                "conditioning_prefix_identity",
                "evaluation_boundary_identity",
                "lifecycle",
            }
            if manifest.get("schema_version") == 5:
                expected_manifest_fields.add("objective_feedback_contract")
            source_commit = execution_identity["source_commit"]
            endpoint_fields = {"physical_endpoint_identity"}
            if plan.controller == "h3c_agent":
                endpoint_fields.add("model_endpoint_identity")
            resume_replay_ok = False
            resume_events = [
                row for row in streams["timing.jsonl"] if row.get("phase") == "resume_replay"
            ]
            if resume_lineage is None:
                resume_replay_ok = (
                    not resume_events and "resume_replay_identity" not in execution_identity
                )
            elif isinstance(resume_lineage, dict):
                lineage_fields = {
                    "source_commit",
                    "source_run_identity",
                    "source_plan_identity",
                    "source_test_id",
                    "source_completed_hour",
                    "source_completed_step",
                    "source_next_step",
                    "source_program_versions",
                    "source_prefix_identity",
                }
                source_completed_hour = resume_lineage.get("source_completed_hour")
                source_completed_step = resume_lineage.get("source_completed_step")
                source_next_step = resume_lineage.get("source_next_step")
                source_versions = resume_lineage.get("source_program_versions")
                expected_started = {
                    "phase": "resume_replay",
                    "event": "started",
                    "source_run_identity": resume_lineage.get("source_run_identity"),
                    "source_test_id": resume_lineage.get("source_test_id"),
                    "source_prefix_identity": resume_lineage.get("source_prefix_identity"),
                    "replay_step_count": source_next_step,
                }
                expected_completed = {
                    **expected_started,
                    "event": "completed",
                    "next_step": source_next_step,
                }
                resume_replay_ok = (
                    set(resume_lineage) == lineage_fields
                    and isinstance(source_completed_hour, int)
                    and not isinstance(source_completed_hour, bool)
                    and source_completed_hour >= 0
                    and isinstance(source_completed_step, int)
                    and not isinstance(source_completed_step, bool)
                    and source_completed_step == source_completed_hour * 4 + 3
                    and isinstance(source_next_step, int)
                    and not isinstance(source_next_step, bool)
                    and source_next_step == source_completed_step + 1
                    and source_next_step < expected_steps
                    and isinstance(source_versions, dict)
                    and set(source_versions) == zone_set
                    and all(
                        isinstance(value, int) and not isinstance(value, bool) and value >= 0
                        for value in source_versions.values()
                    )
                    and isinstance(resume_lineage.get("source_run_identity"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", str(resume_lineage["source_run_identity"]))
                    is not None
                    and isinstance(resume_lineage.get("source_commit"), str)
                    and re.fullmatch(r"[0-9a-f]{40}", str(resume_lineage["source_commit"]))
                    is not None
                    and resume_lineage.get("source_plan_identity") == plan.identity(profile)
                    and isinstance(resume_lineage.get("source_test_id"), str)
                    and bool(resume_lineage["source_test_id"])
                    and resume_lineage.get("source_test_id") not in boundary_test_ids
                    and isinstance(resume_lineage.get("source_prefix_identity"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", str(resume_lineage["source_prefix_identity"]))
                    is not None
                    and execution_identity.get("resume_replay_identity")
                    == _identity(resume_lineage)
                    and resume_events == [expected_started, expected_completed]
                    and recompute_resume_prefix_identity(directory, resume_lineage)
                    == resume_lineage.get("source_prefix_identity")
                )
            checks["resume_replay_integrity"] = resume_replay_ok
            identity_ok = (
                set(resolved) == expected_resolved_fields
                and registered_profile == profile
                and method == plan.method_config()
                and resolved["runtime_contract"] == runtime
                and (
                    "graph_mutation" not in resolved
                    or resolved["graph_mutation"] == plan.graph_mutation
                )
                and isinstance(execution_identity, dict)
                and set(execution_identity) == expected_execution_fields
                and execution_identity["plan_identity"] == plan.identity(profile)
                and execution_identity["runtime_contract"] == runtime
                and execution_identity["dispatch_mode"] == "auto"
                and (
                    plan.controller == "deterministic_baseline"
                    or (
                        resolved["model_provider"] == selected_provider
                        and execution_identity["model_provider"] == selected_provider
                        and execution_identity["model_name"] == model_name
                    )
                )
                and isinstance(source_commit, str)
                and re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None
                and all(
                    isinstance(execution_identity[field], str)
                    and re.fullmatch(r"[0-9a-f]{64}", execution_identity[field]) is not None
                    for field in endpoint_fields
                )
                and set(manifest) == expected_manifest_fields
                and manifest["manifest_schema"] == "h3c_run_manifest"
                and manifest["schema_version"] in {4, 5}
                and (
                    manifest["schema_version"] == 4
                    or manifest["objective_feedback_contract"]
                    == (
                        "completed_interval_reward_breakdown_v1"
                        if plan.controller == "h3c_agent"
                        else "not_applicable"
                    )
                )
                and (
                    plan.controller != "h3c_agent"
                    or manifest["schema_version"] == 4
                    or execution_identity["objective_feedback_contract"]
                    == "completed_interval_reward_breakdown_v1"
                )
                and manifest["source_commit"] == source_commit
                and manifest["run_identity"] == _identity(execution_identity)
                and manifest["controller"] == plan.controller
                and manifest["dispatch_mode"] == "auto"
                and manifest["expected_agent_calls"] == plan.expected_agent_calls(len(zones))
                and isinstance(manifest["retry_count"], int)
                and not isinstance(manifest["retry_count"], bool)
                and 0 <= manifest["retry_count"] <= expected_calls * runtime["model"]["retry_count"]
                and manifest["program_replay_verified"] is replay_ok
                and re.fullmatch(r"[0-9a-f]{64}", str(manifest["conditioning_prefix_identity"]))
                is not None
                and re.fullmatch(r"[0-9a-f]{64}", str(manifest["evaluation_boundary_identity"]))
                is not None
            )
            dispatch_state = _object(directory / "dispatch_state.json")
            completed_hour = _object(directory / "completed_hour_checkpoint.json")
            identity_ok = identity_ok and (
                dispatch_state
                == {
                    "artifact_schema": "h3c_dispatch_state",
                    "schema_version": 1,
                    "run_identity": manifest["run_identity"],
                    "dispatch_mode": "auto",
                    "status": "STOPPED",
                    "test_id": boundary_events[0]["test_id"],
                    "testcase": profile["testcase"],
                }
                and completed_hour.get("artifact_schema") == "h3c_completed_hour_checkpoint"
                and completed_hour.get("schema_version") == 1
                and completed_hour.get("run_identity") == manifest["run_identity"]
                and completed_hour.get("test_id") == boundary_events[0]["test_id"]
                and completed_hour.get("completed_hour") == hours - 1
                and completed_hour.get("completed_step") == hours * 4 - 1
                and completed_hour.get("next_step") == hours * 4
                and completed_hour.get("program_versions")
                == {zone: decisions[-1]["program_replay"][zone]["version"] for zone in zones}
            )
        except (IndexError, KeyError, OSError, TypeError, ValueError):
            identity_ok = False
        checks["manifest_identity"] = identity_ok

        prefix_hash = hashlib.sha256()
        for row in conditioning:
            prefix_hash.update(
                (
                    _canonical({key: value for key, value in row.items() if key != "test_id"})
                    + "\n"
                ).encode("utf-8")
            )
        checks["conditioning_prefix_identity"] = (
            manifest.get("conditioning_prefix_identity") == prefix_hash.hexdigest()
        )
        boundary_contract = False
        if len(boundary_events) == 1 and not conditioning:
            try:
                event = boundary_events[0]
                boundary = event["boundary"]
                physical_state = boundary["physical_state"]
                sensor_points = {
                    str(mapping["temperature_sensor"]) for mapping in profile["zones"].values()
                }
                power_points = {str(point) for point in profile["global_inputs"]["power_meters"]}
                boundary_contract = (
                    set(event) == {"phase", "boundary", "evaluation_boundary_identity", "test_id"}
                    and set(boundary)
                    == {
                        "physical_state",
                        "last_setpoint_c",
                        "last_pmv",
                        "last_occupancy",
                        "clothing_insulation",
                    }
                    and isinstance(physical_state, dict)
                    and set(physical_state) >= {"time", *sensor_points, *power_points}
                    and all(
                        _finite_number(physical_state[point])
                        for point in {"time", *sensor_points, *power_points}
                    )
                    and _same_number(physical_state["time"], evaluation_start)
                    and _finite_exact_map(boundary["last_setpoint_c"], zone_set)
                    and _finite_exact_map(boundary["last_pmv"], zone_set)
                    and _finite_exact_map(boundary["last_occupancy"], zone_set)
                    and _finite_number(boundary["clothing_insulation"])
                    and all(
                        _same_number(value, protocol["initial_setpoint_c"])
                        for value in boundary["last_setpoint_c"].values()
                    )
                    and all(_same_number(value, 0.0) for value in boundary["last_pmv"].values())
                    and all(
                        _same_number(value, 0.0) for value in boundary["last_occupancy"].values()
                    )
                )
            except (KeyError, TypeError, ValueError):
                boundary_contract = False
        checks["evaluation_boundary_identity"] = boundary_contract and boundary_events[0].get(
            "evaluation_boundary_identity"
        ) == _identity(boundary_events[0].get("boundary")) == manifest.get(
            "evaluation_boundary_identity"
        )
        checks["metrics_recomputed"] = compute_run_metrics(directory) == recorded_metrics
        retry_accounting_ok = True
        recomputed_retry_count = sum(
            int(attempt.get("will_retry") is True) for attempt in model_attempts
        )
        attempt_fields = {
            "hour",
            "step",
            "call_ordinal",
            "role",
            "thinking_mode",
            "request_model",
            "model_provider",
            "logical_call_identity",
            "request_identity",
            "provider_neutral_request_identity",
            "request_body",
            "attempt_number",
            "maximum_attempts",
            "outcome",
            "retryable",
            "will_retry",
            "error_type",
            "provider_charge_status",
            "elapsed_seconds",
            "provider_retry_after_seconds",
            "retry_delay_seconds",
        }
        retryable_connection_error_types = {
            "ConnectionResetError",
            "ConnectionAbortedError",
            "BrokenPipeError",
            "TimeoutError",
            "gaierror",
            "IncompleteRead",
            "SSLEOFError",
            "SSLZeroReturnError",
        }
        attempts_by_identity: dict[str, list[dict[str, Any]]] = {}
        for attempt in model_attempts:
            identity = attempt.get("logical_call_identity")
            if not isinstance(identity, str) or not identity:
                retry_accounting_ok = False
                continue
            attempts_by_identity.setdefault(identity, []).append(attempt)

        expected_by_ordinal = {surface[0]: surface for surface in expected_call_surface}
        issued_surfaces: list[tuple[int, str, int, int, str | None]] = []
        terminal_identities: set[str] = set(attempts_by_identity) - set(call_by_identity)
        seen_ordinals: set[int] = set()
        maximum_attempts = runtime["model"]["retry_count"] + 1
        for identity, unsorted_group in attempts_by_identity.items():
            try:
                group = sorted(unsorted_group, key=lambda row: int(row["attempt_number"]))
                first = group[0]
                role = str(first["role"])
                context: dict[str, Any] = {
                    "hour": first["hour"],
                    "step": first["step"],
                    "call_ordinal": first["call_ordinal"],
                }
                if role == "executor":
                    context["zone"] = first["zone"]
                ordinal = context["call_ordinal"]
                if not isinstance(ordinal, int) or isinstance(ordinal, bool):
                    raise ValueError("call ordinal is not an integer")
                surface = (
                    ordinal,
                    role,
                    int(context["hour"]),
                    int(context["step"]),
                    context.get("zone"),
                )
                issued_surfaces.append(surface)
                retry_accounting_ok = retry_accounting_ok and (
                    ordinal not in seen_ordinals
                    and expected_by_ordinal.get(ordinal) == surface
                    and len(group) <= maximum_attempts
                    and [row.get("attempt_number") for row in group]
                    == list(range(1, len(group) + 1))
                )
                seen_ordinals.add(ordinal)

                body_text = str(first["request_body"])
                body = json.loads(body_text)
                messages = body["messages"]
                wire_response_schema: dict[str, Any] | None = None
                if provider_response_format == "json_schema":
                    raw_wire_schema = body["response_format"]["json_schema"]["schema"]
                    if not isinstance(raw_wire_schema, dict):
                        raise ValueError("wire response schema is not an object")
                    wire_response_schema = raw_wire_schema
                contract = model_request_contract(
                    model=str(body["model"]),
                    system=str(messages[0]["content"]),
                    user=str(messages[1]["content"]),
                    thinking_mode=str(first["thinking_mode"]),
                    response_format=provider_response_format,
                    response_schema=wire_response_schema,
                )
                request_identity = model_request_identity(contract)
                neutral_request_identity = provider_neutral_request_identity(contract)
                logical_identity = model_logical_call_identity(
                    context,
                    role,
                    str(first["thinking_mode"]),
                    request_identity,
                )
                retry_accounting_ok = retry_accounting_ok and (
                    identity == logical_identity
                    and body_text == model_request_body(contract)
                    and body["model"] == model_name
                    and first["model_provider"] == selected_provider
                    and first["provider_neutral_request_identity"] == neutral_request_identity
                )

                successful = identity in call_by_identity
                if successful:
                    call = call_by_identity[identity]
                    raw = raw_by_identity[identity]
                    retry_accounting_ok = retry_accounting_ok and (
                        call["request_identity"] == raw["request_identity"] == request_identity
                        and call["provider_neutral_request_identity"]
                        == raw["provider_neutral_request_identity"]
                        == neutral_request_identity
                        and all(call.get(key) == value for key, value in context.items())
                        and all(raw.get(key) == value for key, value in context.items())
                        and call["attempt_count"] == raw["attempt_count"] == len(group)
                        and call["transport_retry_count"]
                        == raw["transport_retry_count"]
                        == len(group) - 1
                    )
                for index, attempt in enumerate(group, 1):
                    is_final = index == len(group)
                    expected_fields = attempt_fields | ({"zone"} if role == "executor" else set())
                    retryable = attempt.get("retryable") is True
                    provider_retry_after = attempt.get("provider_retry_after_seconds")
                    retry_delay = attempt.get("retry_delay_seconds")
                    common_ok = (
                        set(attempt) == expected_fields
                        and all(attempt.get(key) == value for key, value in context.items())
                        and attempt.get("role") == role
                        and attempt.get("thinking_mode") == first["thinking_mode"]
                        and attempt.get("request_model") == model_name
                        and attempt.get("model_provider") == selected_provider
                        and attempt.get("logical_call_identity") == logical_identity
                        and attempt.get("request_identity") == request_identity
                        and attempt.get("provider_neutral_request_identity")
                        == neutral_request_identity
                        and attempt.get("request_body") == body_text
                        and attempt.get("maximum_attempts") == maximum_attempts
                        and _nonnegative_finite_number(attempt.get("elapsed_seconds"))
                    )
                    if attempt.get("outcome") == "request_failed":
                        provider_status = attempt.get("provider_charge_status")
                        failure_ok = (
                            isinstance(attempt.get("error_type"), str)
                            and bool(attempt["error_type"])
                            and (
                                (
                                    retryable
                                    and attempt.get("error_type") in provider_retryable_http_errors
                                    and provider_status == "response_received_usage_unavailable"
                                )
                                or (
                                    retryable
                                    and attempt.get("error_type")
                                    in retryable_connection_error_types
                                    and provider_status == "unknown_after_request_failure"
                                )
                                or (
                                    not retryable
                                    and provider_status
                                    in {
                                        "unknown_after_request_failure",
                                        "response_received_usage_unavailable",
                                    }
                                )
                            )
                        )
                        if attempt.get("will_retry") is True:
                            delay_ok = (
                                not is_final
                                and retryable
                                and _nonnegative_finite_number(retry_delay)
                                and (
                                    (
                                        _finite_number(provider_retry_after)
                                        and _same_number(retry_delay, provider_retry_after)
                                    )
                                    or (
                                        provider_retry_after is None
                                        and _same_number(
                                            retry_delay,
                                            runtime["model"]["retry_backoff_seconds"][index - 1],
                                        )
                                    )
                                )
                            )
                        else:
                            delay_ok = retry_delay is None
                        retry_accounting_ok = (
                            retry_accounting_ok and common_ok and failure_ok and delay_ok
                        )
                    else:
                        retry_accounting_ok = retry_accounting_ok and (
                            common_ok
                            and successful
                            and is_final
                            and attempt.get("outcome") == "response_received"
                            and attempt.get("retryable") is False
                            and attempt.get("will_retry") is False
                            and attempt.get("error_type") is None
                            and attempt.get("provider_charge_status")
                            == "confirmed_response_usage_recorded"
                            and provider_retry_after is None
                            and retry_delay is None
                        )
                final = group[-1]
                if successful:
                    retry_accounting_ok = retry_accounting_ok and (
                        final.get("outcome") == "response_received"
                    )
                else:
                    retry_accounting_ok = retry_accounting_ok and (
                        final.get("outcome") == "request_failed"
                        and final.get("will_retry") is False
                        and (final.get("retryable") is not True or len(group) == maximum_attempts)
                    )
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                retry_accounting_ok = False

        terminal_attempts = [
            attempt
            for identity in terminal_identities
            for attempt in attempts_by_identity[identity]
        ]
        issued_surfaces.sort()
        expected_prefix_ok = False
        failed_executor_hour: int | None = None
        if not terminal_identities:
            expected_prefix_ok = issued_surfaces == expected_call_surface
        else:
            terminal_surfaces = [
                surface
                for surface in issued_surfaces
                if any(
                    row.get("logical_call_identity") in terminal_identities
                    and row.get("call_ordinal") == surface[0]
                    for row in model_attempts
                )
            ]
            terminal_roles = {surface[1] for surface in terminal_surfaces}
            terminal_hours = {surface[2] for surface in terminal_surfaces}
            if terminal_roles == {"executor"} and len(terminal_hours) == 1:
                failed_executor_hour = next(iter(terminal_hours))
                expected_issued = [
                    surface
                    for surface in expected_call_surface
                    if surface[0]
                    <= max(
                        item[0]
                        for item in expected_call_surface
                        if item[1] == "executor" and item[2] == failed_executor_hour
                    )
                ]
                expected_prefix_ok = issued_surfaces == expected_issued
            elif len(terminal_surfaces) == 1:
                cutoff = terminal_surfaces[0][0]
                expected_prefix_ok = issued_surfaces == [
                    surface for surface in expected_call_surface if surface[0] <= cutoff
                ]

        failed_batch_ok = True
        if failed_executor_hour is not None:
            batch_events = [
                row
                for row in streams["timing.jsonl"]
                if row.get("phase") == "executor_batch" and row.get("hour") == failed_executor_hour
            ]
            failed_zones = [
                zone
                for zone in zones
                if any(
                    attempt.get("logical_call_identity") in terminal_identities
                    and attempt.get("zone") == zone
                    for attempt in model_attempts
                )
            ]
            failed_batch_ok = (
                len(batch_events) == 1
                and batch_events[0].get("event") == "terminal_transport_failure"
                and batch_events[0].get("issued_zones") == zones
                and batch_events[0].get("failed_zones") == failed_zones
                and batch_events[0].get("primary_failure_zone") == failed_zones[0]
                and batch_events[0].get("settlement_performed") is False
                and batch_events[0].get("physical_advance_performed") is False
                and not any(row.get("hour") == failed_executor_hour for row in updates)
                and not any(row.get("hour") == failed_executor_hour for row in zone_steps)
            )

        checks["model_transport_retry_accounting"] = (
            retry_accounting_ok
            and set(attempts_by_identity) == set(call_by_identity) | terminal_identities
            and expected_prefix_ok
            and failed_batch_ok
            and recomputed_retry_count == manifest.get("retry_count")
            and manifest.get("transport_error_count") == len(terminal_identities)
            and (
                bool(calls)
                or bool(terminal_attempts)
                or (not model_attempts and manifest.get("retry_count") == 0)
            )
        )
        checks["transport_error_count_zero"] = manifest.get("transport_error_count") == 0
        checks["secret_exposure_count_zero"] = manifest.get(
            "secret_exposure_count"
        ) == 0 and manifest.get("secret_scan_status") == (
            "completed" if method["controller"] == "h3c_agent" else "not_applicable"
        )

        checks["rationale_persistence"] = _rationale_persistence(
            method=method,
            raw_calls=raw_calls,
            updates=updates,
            decisions=decisions,
            zones=zones,
            allowed_edge_ids=allowed_edge_ids,
            shared_power_edge_ids=shared_power_edge_ids,
            expected_site_cap_c=resolved_site_cap,
            expected_per_zone_reserved_cap_c=resolved_per_zone_reserved_cap,
        )

        raw_schema = all(
            _raw_contract(
                row,
                zones=zones,
                causal_enabled=causal_enabled,
                allowed_edge_ids=allowed_edge_ids,
                shared_power_edge_ids=shared_power_edge_ids,
                long_term_memory=bool(method.get("long_term_memory")),
                expected_site_cap_c=resolved_site_cap,
                expected_per_zone_reserved_cap_c=resolved_per_zone_reserved_cap,
            )
            for row in raw_calls
        )
        checks["json_schema"] = (
            (raw_schema and all(row.get("status") != "model_output_rejected" for row in updates))
            if method["controller"] == "h3c_agent"
            else True
        )
        usage_fields = {
            "available",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
        }
        token_fields = usage_fields - {"available"}
        request_contract_ok = True
        for row in raw_calls:
            try:
                response_format = runtime["model"]["providers"][selected_provider][
                    "response_format"
                ]
                response_schema: dict[str, Any] | None = None
                if response_format == "json_schema":
                    role = row["role"]
                    if role == "orchestrator":
                        response_schema = orchestrator_response_schema(
                            zones=tuple(zones), causal_enabled=causal_enabled
                        )
                    elif role == "executor":
                        response_schema = executor_response_schema(
                            causal_enabled=causal_enabled,
                            long_term_memory=bool(method.get("long_term_memory")),
                        )
                    elif role == "reflector":
                        response_schema = reflector_response_schema(
                            zones=tuple(zones),
                            long_term_memory=bool(method.get("long_term_memory")),
                        )
                    else:
                        raise ValueError("raw call role is invalid")
                expected_contract = model_request_contract(
                    model=model_name,
                    system=row["system"],
                    user=row["user"],
                    thinking_mode=row["thinking_mode"],
                    response_format=response_format,
                    response_schema=response_schema,
                )
                expected_request = {
                    key: value for key, value in expected_contract.items() if key != "messages"
                }
                if row["thinking_mode"] == "low":
                    expected_request["reasoning_effort"] = runtime["model"][
                        "thinking_reasoning_effort"
                    ]
                else:
                    expected_request["temperature"] = runtime["model"]["no_thinking_temperature"]
                    expected_request["top_p"] = runtime["model"]["no_thinking_top_p"]
                request_contract_ok = request_contract_ok and (
                    row["request_parameters"] == expected_request
                    and isinstance(row["system"], str)
                    and bool(row["system"])
                    and isinstance(row["user"], str)
                    and bool(row["user"])
                    and isinstance(row["output"], str)
                )
            except (KeyError, TypeError):
                request_contract_ok = False
        checks["model_request_identity"] = (
            request_contract_ok if calls else method["controller"] == "deterministic_baseline"
        )
        checks["usage_contract"] = (
            all(
                isinstance(row.get("usage"), dict)
                and set(row["usage"]) == usage_fields
                and row["usage"]["available"] is True
                and all(
                    isinstance(row["usage"][field], int)
                    and not isinstance(row["usage"][field], bool)
                    and row["usage"][field] >= 0
                    for field in token_fields
                )
                and row["usage"]["cache_hit_tokens"] + row["usage"]["cache_miss_tokens"]
                == row["usage"]["prompt_tokens"]
                and row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"]
                == row["usage"]["total_tokens"]
                and row["usage"]["reasoning_tokens"] <= row["usage"]["completion_tokens"]
                and normalized_usage(row.get("provider_usage")) == row["usage"]
                for row in calls
            )
            if calls
            else method["controller"] == "deterministic_baseline"
        )

        route_by_hour: dict[int, str] = {}
        route_valid = True
        for decision in decisions:
            hour = int(decision["hour"])
            first_rows = step_groups.get(hour * 4, [])
            current = {
                str(row["zone"]): float(row["observation"]["current_occupancy"])
                for row in first_rows
            }
            future = {
                str(row["zone"]): float(row["observation"]["next_hour_occupancy"])
                for row in first_rows
            }
            expected_route = hourly_route(hour, current, future)
            route_valid = route_valid and decision.get("route") == expected_route
            route_by_hour[hour] = str(expected_route["thinking_mode"])
        checks["thinking_route"] = route_valid and all(
            row.get("thinking_mode")
            == (
                "disabled"
                if method["thinking_policy"] == "all_roles_disabled"
                else route_by_hour[int(row["hour"])]
            )
            for row in calls
        )

        fallback_count = sum(
            row.get("orchestration", {}).get("fallback", {}).get("used") is True
            for row in decisions
        )
        checks["fallback_count_zero"] = manifest.get("fallback_count") == fallback_count == 0
        if method["coordination_enabled"]:
            coordination_ok = True
            orchestration_resolution_ok = True
            previous_expected_allocation: dict[str, Any] | None = None
            fallback_causal_edge_ids: list[str] | None = None
            if causal_enabled and graph is not None:
                site_node_ids = {str(node["id"]) for node in graph.nodes if node["scope"] == "site"}
                fallback_causal_edge_ids = [
                    edge.identifier for edge in graph.edges if edge.target in site_node_ids
                ]
            for decision in decisions:
                try:
                    audit = decision["orchestration"]
                    allocation = audit["allocation_audit"]
                    validate_allocation(
                        allocation,
                        zones,
                        causal_enabled=causal_enabled,
                        allowed_causal_edge_ids=allowed_edge_ids,
                        site_causal_edge_ids=shared_power_edge_ids,
                        expected_site_cap_c=resolved_site_cap,
                        expected_per_zone_reserved_cap_c=resolved_per_zone_reserved_cap,
                    )
                    budget = decision["energy_budget"]
                    granted = sum(float(value) for value in allocation["zone_budgets_c"].values())
                    initial = float(allocation["site_cap_c"]) - granted
                    matching_raw = [
                        row
                        for row in raw_calls
                        if row.get("role") == "orchestrator"
                        and row.get("hour") == decision.get("hour")
                    ]
                    if len(matching_raw) != 1:
                        raise ValueError("one raw Orchestrator call is required per hour")
                    raw_output = matching_raw[0]["output"]
                    expected_rejection: dict[str, str] | None
                    try:
                        expected_allocation, expected_telemetry = resolve_orchestrator_model_output(
                            raw_output,
                            zones,
                            causal_enabled=causal_enabled,
                            allowed_causal_edge_ids=allowed_edge_ids,
                            site_causal_edge_ids=shared_power_edge_ids,
                            expected_site_cap_c=resolved_site_cap,
                            expected_per_zone_reserved_cap_c=resolved_per_zone_reserved_cap,
                        )
                    except ModelContractError as error:
                        expected_rejection = {
                            "code": "orchestrator_model_contract_rejected",
                            "message": str(error),
                            "raw_output": error.raw_output,
                        }
                        expected_allocation, expected_source = validated_fallback_allocation(
                            zones,
                            previous_expected_allocation,
                            site_cap_c=resolved_site_cap,
                            causal_enabled=causal_enabled,
                            causal_edge_ids=fallback_causal_edge_ids,
                            allowed_causal_edge_ids=allowed_edge_ids,
                            site_causal_edge_ids=shared_power_edge_ids,
                            per_zone_reserved_cap_c=resolved_per_zone_reserved_cap,
                        )
                        expected_status = "fallback"
                        expected_telemetry = None
                        expected_fallback = allocation_fallback_audit(
                            used=True,
                            reason=str(error),
                            source=expected_source,
                        )
                    else:
                        expected_status = "accepted"
                        expected_rejection = None
                        expected_fallback = allocation_fallback_audit(used=False)
                    expected_raw_contract = {
                        "status": "rejected" if expected_rejection is not None else "accepted",
                        "rejection": expected_rejection,
                    }
                    rationale_resolution_ok = (
                        audit.get("status") == expected_status
                        and audit.get("raw_contract") == expected_raw_contract
                        and audit.get("rationale_telemetry") == expected_telemetry
                        and audit.get("fallback") == expected_fallback
                        and allocation == expected_allocation
                    )
                    orchestration_resolution_ok = (
                        orchestration_resolution_ok and rationale_resolution_ok
                    )
                    previous_expected_allocation = expected_allocation
                    coordination_ok = coordination_ok and (
                        audit["settlement_order"] == allocation["priority"]
                        and _same_budget_number(budget["site_cap_c"], allocation["site_cap_c"])
                        and _same_budget_number(budget["granted_c"], granted)
                        and _same_budget_number(budget["residual_initial_c"], initial)
                        and float(budget["used_c"]) <= granted + initial + BUDGET_ABS_TOLERANCE
                        and audit["raw_contract"]["status"]
                        == ("rejected" if audit["fallback"]["used"] else "accepted")
                    )
                except (KeyError, TypeError, ValueError):
                    coordination_ok = False
                    orchestration_resolution_ok = False
            checks["coordination_surface"] = coordination_ok
            checks["orchestration_resolution_recomputed"] = orchestration_resolution_ok
        else:
            no_coordination_surface = _canonical(
                {"raw": raw_calls, "decisions": decisions, "updates": updates}
            ).lower()
            checks["coordination_surface"] = all(
                row.get("role") != "orchestrator" for row in calls
            ) and all(
                token not in no_coordination_surface
                for token in ("allocation", "allowance", "energy_budget", "ledger")
            )
            checks["orchestration_resolution_recomputed"] = True
        checks["role_contract_audit"] = (
            all(
                (
                    decision.get("reflector_contract", {}).get("status") == "accepted"
                    and (
                        not method["coordination_enabled"]
                        or decision.get("orchestration", {}).get("raw_contract", {}).get("status")
                        == "accepted"
                    )
                )
                for decision in decisions
            )
            if method["controller"] == "h3c_agent"
            else True
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as error:
        result = _classified(checks)
        result["errors"] = [str(error), *result["errors"]]
        return result

    try:
        performance_pass, performance_evaluation = _performance_evaluation(
            profile, method, recorded_metrics
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        performance_pass = False
        performance_evaluation = {"passed": False, "error": str(error)}
    base_result = _classified(checks, performance_pass=performance_pass)
    base_result["performance_evaluation"] = performance_evaluation
    if not require_completion:
        return base_result

    completion_checks = dict(checks)
    try:
        recorded = _object(directory / "verification.json")
        completion = _object(directory / "completion.json")
        completion_checks["recorded_verification"] = recorded == base_result
        completion_checks["completion"] = (
            completion.get("status") == "complete"
            and completion.get("run_identity") == manifest.get("run_identity")
            and completion.get("classification") == base_result["classification"]
            and (directory / "completion.json").stat().st_mtime_ns
            >= max(
                path.stat().st_mtime_ns
                for path in directory.iterdir()
                if path.is_file() and path.name != "completion.json"
            )
        )
    except (OSError, ValueError, json.JSONDecodeError):
        completion_checks["recorded_verification"] = False
        completion_checks["completion"] = False
    final = _classified(completion_checks, performance_pass=performance_pass)
    final["performance_evaluation"] = performance_evaluation
    if not completion_checks.get("recorded_verification") or not completion_checks.get(
        "completion"
    ):
        final["execution_integrity"] = False
        final["model_contract_clean"] = base_result["model_contract_clean"]
        final["trajectory_status"] = "EXECUTION-INVALID"
        final["model_contract_status"] = base_result["model_contract_status"]
        final["performance_status"] = base_result["performance_status"]
        final["classification"] = "RUN-INVALID"
        final["passed"] = False
        final["completion_eligible"] = False
        final["errors"] = [
            "completion publication failed checks",
            *base_result["errors"],
        ]
    return final


def recertify_run(
    run_dir: Path,
    *,
    output_path: Path,
    recertifier_source_commit: str,
) -> dict[str, Any]:
    """Append a zero-call re-verification without changing historical evidence."""
    directory = run_dir.resolve()
    destination = output_path.resolve()
    if destination.parent != directory:
        raise ValueError("recertification artifact must be created inside the source run")
    if not re.fullmatch(r"[0-9a-f]{40}", recertifier_source_commit):
        raise ValueError("recertifier source commit must be a lowercase 40-character git hash")
    resolved = _object(directory / "resolved_config.yaml")
    runtime_contract = resolved.get("runtime_contract")
    if not isinstance(runtime_contract, dict):
        raise ValueError("historical run does not embed its runtime contract")
    manifest = _object(directory / "manifest.json")
    terminal_artifacts = {
        name: _file_identity(directory / name)
        for name in ("completion.json", "failure.json", "verification.json")
        if (directory / name).is_file()
    }
    result = verify_run(
        directory,
        require_completion=False,
        historical_runtime_contract=runtime_contract,
    )
    artifact = {
        "recertification_schema": "h3c_zero_call_run_recertification",
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_run": {
            "directory": str(directory),
            "run_identity": manifest.get("run_identity"),
            "source_commit": manifest.get("source_commit"),
            "original_terminal_artifacts_sha256": terminal_artifacts,
        },
        "recertifier": {
            "source_commit": recertifier_source_commit,
            "external_calls": 0,
            "historical_runtime_contract_identity": _identity(runtime_contract),
        },
        "result": result,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2))
        file.write("\n")
    return artifact
