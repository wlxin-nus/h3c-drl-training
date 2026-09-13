"""Fail-closed verification for non-Agent baseline evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from h3c.experiments.profiles import repository_root
from h3c.runtime.comfort import step_reward
from h3c.runtime.occupancy import effective_count, verify_missing_occupancy_resolution_evidence
from h3c.runtime.protocol import (
    physical_evidence_identity,
    reconstruct_forecast_evidence,
)
from h3c_baselines.configuration import BaselineRunPlan
from h3c_baselines.controllers.enhanced_rbc import EnhancedRbcController
from h3c_baselines.models import model_entry, verify_checkpoint
from h3c_baselines.mpc.registry import verify_frozen_mpc_suite
from h3c_baselines.mpc.training import verify_frozen_mpc_model
from h3c_baselines.outputs.artifacts import PERFORMANCE_COLUMNS
from h3c_baselines.outputs.metrics import compute_baseline_metrics
from h3c_baselines.policies.observation_contracts import PolicyObservationBuilder

FORBIDDEN_AGENT_FILES = {
    "agent_calls.jsonl",
    "raw_model_io.jsonl",
    "hourly_decisions.jsonl",
    "program_updates.jsonl",
    "zone_steps.jsonl",
    "model_request_attempts.jsonl",
}


def _load(path: Path) -> dict[str, Any]:
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(candidate, dict):
        raise ValueError(f"{path.name} is not an object")
    return candidate


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        candidate = json.loads(line)
        if not isinstance(candidate, dict):
            raise ValueError(f"{path.name} contains a non-object row")
        rows.append(candidate)
    return rows


def _performance(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != PERFORMANCE_COLUMNS:
            raise ValueError("baseline performance header is invalid")
        rows = [dict(row) for row in reader]
    if any(any(value is None for value in row.values()) for row in rows):
        raise ValueError("baseline performance row is incomplete")
    return rows


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


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _same(left: Any, right: Any) -> bool:
    return (
        _finite(left)
        and _finite(right)
        and math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    )


def verify_concurrent_suite_evidence(path: Path) -> dict[str, Any]:
    """Verify suite binding, arm order, and fresh cross-arm physical identities."""

    try:
        evidence = _load(path.resolve())
        expected_fields = {
            "suite_evidence_schema",
            "schema_version",
            "suite",
            "suite_identity",
            "dispatch_mode",
            "source_commit",
            "plan_identities",
            "mpc_freeze_identity",
            "arms",
        }
        plans = evidence.get("plan_identities")
        arms = evidence.get("arms")
        source_commit = evidence.get("source_commit")
        suite = evidence.get("suite")
        expected_identity = _identity(
            {
                "dispatch_mode": "auto",
                "plans": plans,
                "source_commit": source_commit,
                "suite": suite,
                "mpc_freeze_identity": evidence.get("mpc_freeze_identity"),
            }
        )
        test_ids = [row.get("test_id") for row in arms] if isinstance(arms, list) else []
        run_ids = [row.get("run_identity") for row in arms] if isinstance(arms, list) else []
        recorded_freeze_identity = evidence.get("mpc_freeze_identity")
        freeze_binding = recorded_freeze_identity is None
        if isinstance(recorded_freeze_identity, str):
            frozen = verify_frozen_mpc_suite()
            if frozen.get("valid") is True:
                freeze_manifest = _load(Path(str(frozen["target"])) / "freeze_manifest.json")
                freeze_binding = freeze_manifest.get("freeze_identity") == recorded_freeze_identity
        actual_arm_bindings: list[bool] = []
        if isinstance(arms, list):
            suite_root = path.resolve().parent.parent
            for arm in arms:
                if not isinstance(arm, dict) or not isinstance(arm.get("run_dir"), str):
                    actual_arm_bindings.append(False)
                    continue
                try:
                    run_dir = Path(arm["run_dir"]).resolve()
                    run_dir.relative_to(suite_root)
                    resolved = _load(run_dir / "resolved_config.json")
                    manifest = _load(run_dir / "manifest.json")
                    completion = _load(run_dir / "completion.json")
                    execution = resolved["execution_identity"]
                    timing = _rows(run_dir / "timing.jsonl")
                    actual_test_ids = {
                        row.get("test_id")
                        for row in timing
                        if row.get("phase")
                        in {"physical_dispatch", "physical_lifecycle", "evaluation_boundary"}
                        and isinstance(row.get("test_id"), str)
                    }
                    verification = verify_baseline_run(run_dir)
                    actual_arm_bindings.append(
                        verification.get("execution_integrity") is True
                        and resolved.get("plan_identity") == arm.get("plan_identity")
                        and execution.get("plan_identity") == arm.get("plan_identity")
                        and execution.get("suite_identity") == evidence.get("suite_identity")
                        and execution.get("mpc_freeze_identity") == recorded_freeze_identity
                        and execution.get("dispatch_mode") == "auto"
                        and execution.get("source_commit") == source_commit
                        and manifest.get("run_identity")
                        == arm.get("run_identity")
                        == _identity(execution)
                        and completion.get("run_identity") == arm.get("run_identity")
                        and completion.get("classification") == arm.get("classification")
                        and manifest.get("case") == arm.get("case")
                        and manifest.get("controller") == arm.get("controller")
                        and actual_test_ids == {arm.get("test_id")}
                    )
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    actual_arm_bindings.append(False)
        checks = {
            "schema": set(evidence) == expected_fields
            and evidence.get("suite_evidence_schema") == "h3c_concurrent_baseline_suite"
            and evidence.get("schema_version") == 1,
            "dispatch_mode": evidence.get("dispatch_mode") == "auto",
            "source_identity": isinstance(source_commit, str)
            and re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
            "suite_identity": evidence.get("suite_identity") == expected_identity,
            "mpc_freeze_identity": evidence.get("mpc_freeze_identity") is None
            or (
                isinstance(evidence.get("mpc_freeze_identity"), str)
                and re.fullmatch(r"[0-9a-f]{64}", evidence["mpc_freeze_identity"]) is not None
            ),
            "mpc_freeze_registry_binding": freeze_binding,
            "arm_identity": isinstance(plans, list)
            and isinstance(arms, list)
            and len(plans) == len(arms)
            and [row.get("plan_identity") for row in arms] == plans
            and all(
                isinstance(row, dict)
                and isinstance(row.get("run_identity"), str)
                and isinstance(row.get("case"), str)
                for row in arms
            ),
            "cross_arm_test_identity": bool(test_ids)
            and all(isinstance(value, str) and value for value in test_ids)
            and len(test_ids) == len(set(test_ids)),
            "cross_arm_run_identity": bool(run_ids) and len(run_ids) == len(set(run_ids)),
            "actual_run_evidence_binding": bool(actual_arm_bindings) and all(actual_arm_bindings),
        }
        errors = sorted(name for name, value in checks.items() if not value)
        return {"checks": checks, "errors": errors, "valid": not errors}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {"checks": {}, "errors": [type(error).__name__], "valid": False}


def _exact_numeric_map(value: Any, keys: set[str]) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == keys
        and all(_finite(item) for item in value.values())
    )


def _numeric_sequence(value: Any, length: int) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == length
        and all(_finite(item) for item in value)
    )


def _exact_dynamic_dispatch(
    dispatch: Sequence[Mapping[str, Any]], lifecycle_test_ids: set[Any]
) -> bool:
    events = [row.get("event") for row in dispatch]
    configured_index = events.index("configured") if "configured" in events else -1
    initialized_index = events.index("initialized") if "initialized" in events else -1
    admission = dispatch[1:configured_index] if configured_index > 0 else []
    initialization_admission = (
        dispatch[configured_index + 1 : initialized_index]
        if initialized_index > configured_index >= 0
        else []
    )
    return (
        len(dispatch) >= 5
        and events[0] == "selected"
        and events[-1] == "stopped"
        and events.count("selected") == 1
        and events.count("configured") == 1
        and events.count("initialized") == 1
        and events.count("stopped") == 1
        and configured_index > 1
        and initialized_index > configured_index
        and events[initialized_index + 1 :] == ["stopped"]
        and all(row.get("event") == "status_changed" for row in admission)
        and [row.get("status") for row in admission[:-1]] == ["Queued"] * (len(admission) - 1)
        and admission[-1].get("status") == "Running"
        and dispatch[configured_index].get("status") == "Running"
        and bool(initialization_admission)
        and all(
            row.get("event") == "status_changed" and row.get("status") == "Running"
            for row in initialization_admission
        )
        and dispatch[initialized_index].get("status") == "Running"
        and all(row.get("dispatch_mode") == "auto" for row in dispatch)
        and {row.get("test_id") for row in dispatch} == lifecycle_test_ids
        and len({row.get("testcase") for row in dispatch}) == 1
    )


def _invalid_result(error: str) -> dict[str, Any]:
    return {
        "verification_schema": "h3c_baseline_verification",
        "schema_version": 1,
        "checks": {"evidence_parse": False},
        "errors": [f"evidence_parse:{error}"],
        "execution_integrity": False,
        "completion_eligible": False,
        "classification": "RUN-INVALID",
        "metrics_identity": None,
    }


def verify_baseline_run(run_dir: Path, *, require_completion: bool = True) -> dict[str, Any]:
    try:
        directory = run_dir.resolve()
        manifest = _load(directory / "manifest.json")
        resolved = _load(directory / "resolved_config.json")
        recorded_metrics = _load(directory / "metrics.json")
        metrics = compute_baseline_metrics(directory)
        performance = _performance(directory / "performance.csv")
        actions = _rows(directory / "actions.jsonl")
        diagnostics = _rows(directory / "controller_diagnostics.jsonl")
        conditioning = _rows(directory / "physical_conditioning.jsonl")
        timing = _rows(directory / "timing.jsonl")
        forecast_evidence = _load(directory / "forecast_inputs.json")

        case = manifest["case"]
        controller = manifest["controller"]
        evaluation_hours = resolved["evaluation_hours"]
        if (
            not isinstance(case, str)
            or not isinstance(controller, str)
            or isinstance(evaluation_hours, bool)
            or not isinstance(evaluation_hours, int)
        ):
            raise ValueError("baseline plan identity fields are invalid")
        overrides = resolved.get("profile_overrides")
        if not isinstance(overrides, dict):
            raise ValueError("profile overrides are invalid")
        plan = BaselineRunPlan(case, controller, evaluation_hours, overrides)
        expected_plan = plan.resolved()
        execution_identity = resolved.get("execution_identity")
        if not isinstance(execution_identity, dict):
            raise ValueError("execution identity is missing")

        profile = expected_plan["case_profile"]
        protocol = profile["protocol"]
        zones = tuple(profile["zones"])
        zone_set = set(zones)
        expected_steps = evaluation_hours * 4
        evaluation_start = int(profile["evaluation_start_day"]) * 86400
        evaluation_end = evaluation_start + expected_steps * 900
        source_commit = execution_identity.get("source_commit")

        legacy_execution_fields = {
            "plan_identity",
            "source_commit",
            "physical_endpoint_identity",
            "mpc_model_identity",
        }
        current_execution_fields = {
            "plan_identity",
            "source_commit",
            "physical_endpoint_identity",
            "mpc_model_identity",
            "dispatch_mode",
            "suite_identity",
            "mpc_freeze_identity",
        }
        execution_fields = set(execution_identity)
        legacy_execution_identity = execution_fields == legacy_execution_fields
        current_execution_identity = execution_fields == current_execution_fields
        expected_manifest_fields = {
            "manifest_schema",
            "schema_version",
            "run_identity",
            "source_commit",
            "case",
            "controller",
            "conditioning_prefix_identity",
            "evaluation_boundary_identity",
            "mpc_model_identity",
            "secret_scan_status",
            "secret_exposure_count",
            "occupancy_forecast_missing_value_resolution_count",
            "lifecycle",
        }
        expected_lifecycle = {
            "initialize_count": 1,
            "conditioning_advance_count": 0,
            "evaluation_advance_count": expected_steps,
            "stop_count": 1,
            "test_id_changes": 0,
        }
        resolution_events = [
            row
            for row in timing
            if row.get("phase") == "occupancy_forecast_missing_value_resolution"
        ]
        expected_forecast: dict[str, list[float]] = {}
        expected_resolution_events: list[dict[str, Any]] = []
        forecast_contract = True
        try:
            expected_forecast, expected_resolution_events = reconstruct_forecast_evidence(
                profile,
                forecast_evidence,
                expected_steps + 97,
                forecast_phase="evaluation",
                start_time_seconds=evaluation_start,
                step_seconds=900,
            )
        except (KeyError, TypeError, ValueError):
            forecast_contract = False
        checks: dict[str, bool] = {
            "evidence_parse": True,
            "resolved_plan_identity": set(resolved) == set(expected_plan) | {"execution_identity"}
            and all(resolved.get(key) == value for key, value in expected_plan.items()),
            "execution_identity_schema": legacy_execution_identity or current_execution_identity,
            "execution_identity": (legacy_execution_identity or current_execution_identity)
            and execution_identity.get("plan_identity") == expected_plan["plan_identity"]
            and isinstance(source_commit, str)
            and re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None
            and isinstance(execution_identity.get("physical_endpoint_identity"), str)
            and re.fullmatch(r"[0-9a-f]{64}", execution_identity["physical_endpoint_identity"])
            is not None
            and (
                execution_identity.get("mpc_model_identity") is None
                if controller != "hierarchical-mpc"
                else isinstance(execution_identity.get("mpc_model_identity"), str)
            )
            and (
                legacy_execution_identity
                or (
                    execution_identity.get("dispatch_mode") in {"auto", "strictly_serial"}
                    and isinstance(execution_identity.get("suite_identity"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", execution_identity["suite_identity"])
                    is not None
                    and (
                        execution_identity.get("mpc_freeze_identity") is None
                        if controller != "hierarchical-mpc"
                        else isinstance(execution_identity.get("mpc_freeze_identity"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", execution_identity["mpc_freeze_identity"])
                        is not None
                    )
                )
            ),
            "manifest_identity": set(manifest) == expected_manifest_fields
            and manifest.get("manifest_schema") == "h3c_baseline_manifest"
            and manifest.get("schema_version") == 1
            and manifest.get("source_commit") == source_commit
            and manifest.get("run_identity") == _identity(execution_identity)
            and manifest.get("case") == case
            and manifest.get("controller") == controller
            and manifest.get("mpc_model_identity") == execution_identity.get("mpc_model_identity"),
            "physical_lifecycle_counts": manifest.get("lifecycle") == expected_lifecycle,
            "no_explicit_conditioning": conditioning == []
            and manifest.get("conditioning_prefix_identity") == hashlib.sha256(b"").hexdigest(),
            "metrics_recomputed": recorded_metrics == metrics,
            "native_kpis_present": (directory / "native_boptest_kpis.json").is_file(),
            "secret_scan_complete": manifest.get("secret_scan_status") == "completed",
            "secret_absent": manifest.get("secret_exposure_count") == 0,
            "no_agent_evidence": not any(
                (directory / name).exists() for name in FORBIDDEN_AGENT_FILES
            ),
            "forecast_input_evidence": forecast_contract,
            "occupancy_resolution_evidence": forecast_contract
            and expected_resolution_events == resolution_events
            and verify_missing_occupancy_resolution_evidence(
                profile,
                evaluation_hours,
                manifest.get("occupancy_forecast_missing_value_resolution_count"),
                resolution_events,
            ),
        }

        lifecycle = [row for row in timing if row.get("phase") == "physical_lifecycle"]
        lifecycle_contract = [
            {
                "phase": "physical_lifecycle",
                "event": "initialized",
                "time_seconds": evaluation_start,
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
        lifecycle_test_ids = {row.get("test_id") for row in lifecycle}
        checks["lifecycle_timeline"] = (
            protocol.get("initialization_mode") == "evaluation_start_internal_warmup"
            and protocol.get("internal_warmup_days") == 7
            and len(lifecycle) == len(lifecycle_contract)
            and all(
                {key: value for key, value in row.items() if key != "test_id"} == expected
                and isinstance(row.get("test_id"), str)
                and bool(row["test_id"])
                for row, expected in zip(lifecycle, lifecycle_contract, strict=True)
            )
            and len(lifecycle_test_ids) == 1
        )
        dispatch = [row for row in timing if row.get("phase") == "physical_dispatch"]
        if execution_identity.get("dispatch_mode") == "auto":
            checks["dynamic_dispatch_lifecycle"] = _exact_dynamic_dispatch(
                dispatch, lifecycle_test_ids
            )
        elif legacy_execution_identity:
            checks["dynamic_dispatch_lifecycle"] = not dispatch or _exact_dynamic_dispatch(
                dispatch, lifecycle_test_ids
            )
        else:
            checks["dynamic_dispatch_lifecycle"] = not dispatch or all(
                row.get("dispatch_mode") == "auto" for row in dispatch
            )

        boundary_rows = [row for row in timing if row.get("phase") == "evaluation_boundary"]
        boundary_ok = False
        boundary_test_id: Any = None
        boundary_physical_state: dict[str, Any] | None = None
        if len(boundary_rows) == 1:
            event = boundary_rows[0]
            boundary = event.get("boundary")
            boundary_test_id = event.get("test_id")
            if isinstance(boundary, dict):
                state = boundary.get("physical_state")
                if isinstance(state, dict):
                    boundary_physical_state = state
                temperature_points = {
                    mapping["temperature_sensor"] for mapping in profile["zones"].values()
                }
                power_points = set(profile["global_inputs"]["power_meters"])
                event_identity = physical_evidence_identity(boundary)
                boundary_ok = (
                    set(event)
                    == {
                        "phase",
                        "boundary",
                        "evaluation_boundary_identity",
                        "test_id",
                    }
                    and set(boundary)
                    == {
                        "physical_state",
                        "last_setpoint_c",
                        "last_pmv",
                        "last_occupancy",
                        "clothing_insulation",
                    }
                    and isinstance(state, dict)
                    and _same(state.get("time"), evaluation_start)
                    and all(
                        _finite(state.get(point)) for point in temperature_points | power_points
                    )
                    and _exact_numeric_map(boundary.get("last_setpoint_c"), zone_set)
                    and _exact_numeric_map(boundary.get("last_pmv"), zone_set)
                    and _exact_numeric_map(boundary.get("last_occupancy"), zone_set)
                    and all(
                        _same(value, protocol["initial_setpoint_c"])
                        for value in boundary["last_setpoint_c"].values()
                    )
                    and all(_same(value, 0.0) for value in boundary["last_pmv"].values())
                    and all(_same(value, 0.0) for value in boundary["last_occupancy"].values())
                    and _finite(boundary.get("clothing_insulation"))
                    and event.get("evaluation_boundary_identity") == event_identity
                    and manifest.get("evaluation_boundary_identity") == event_identity
                )
        checks["evaluation_boundary_identity"] = boundary_ok

        action_fields = {
            "step",
            "zone",
            "test_id",
            "action_time_seconds",
            "outcome_time_seconds",
            "final_setpoint_c",
            "outcome",
        }
        outcome_fields = {
            "zone_temperature_c",
            "pmv",
            "effective_occupancy",
            "power_w",
            "electricity_price",
            "cost",
        }
        by_step: dict[int, list[dict[str, Any]]] = {}
        for row in actions:
            step = row.get("step")
            if isinstance(step, int) and not isinstance(step, bool):
                by_step.setdefault(step, []).append(row)
        trajectory_ok = (
            len(performance) == expected_steps
            and len(actions) == expected_steps * len(zones)
            and len(diagnostics) == expected_steps
            and [row.get("step") for row in diagnostics] == list(range(expected_steps))
        )
        previous_setpoints = [float(protocol["initial_setpoint_c"])] * len(zones)
        action_test_ids: set[Any] = set()
        for step in range(expected_steps):
            step_actions = by_step.get(step, [])
            if (
                len(step_actions) != len(zones)
                or {row.get("zone") for row in step_actions} != zone_set
            ):
                trajectory_ok = False
                continue
            by_zone = {row["zone"]: row for row in step_actions}
            row = performance[step]
            try:
                temperatures = json.loads(row["zone_temperatures_c"])
                setpoints = json.loads(row["zone_setpoints_c"])
                pmv = json.loads(row["zone_pmv"])
                occupancy = json.loads(row["zone_occupancy"])
                power = float(row["total_power_w"])
                cost = float(row["step_cost"])
                reward = float(row["step_reward"])
                ordered = [by_zone[zone] for zone in zones]
                prices = [item["outcome"]["electricity_price"] for item in ordered]
                expected_occupancy = [
                    effective_count(
                        profile["occupancy"],
                        evaluation_start + step * 900,
                        float(
                            expected_forecast[profile["zones"][zone]["occupancy_forecast"]][step]
                        ),
                    )
                    for zone in zones
                ]
                expected_reward = step_reward(
                    cost=cost,
                    pmv=[float(value) for value in pmv],
                    occupancy=[float(value) for value in occupancy],
                    setpoints_c=[float(value) for value in setpoints],
                    previous_setpoints_c=previous_setpoints,
                    objective=profile["objective"],
                )
                trajectory_ok = trajectory_ok and (
                    int(row["step"]) == step
                    and row["time_seconds"] == str(evaluation_start + step * 900)
                    and len(temperatures) == len(zones)
                    and len(setpoints) == len(zones)
                    and len(pmv) == len(zones)
                    and len(occupancy) == len(zones)
                    and all(
                        set(item) == action_fields
                        and set(item.get("outcome", {})) == outcome_fields
                        and item["action_time_seconds"] == evaluation_start + step * 900
                        and item["outcome_time_seconds"] == evaluation_start + (step + 1) * 900
                        and all(_finite(value) for value in item["outcome"].values())
                        and _finite(item["final_setpoint_c"])
                        for item in ordered
                    )
                    and all(
                        _same(ordered[index]["outcome"]["zone_temperature_c"], temperatures[index])
                        and _same(ordered[index]["final_setpoint_c"], setpoints[index])
                        and _same(ordered[index]["outcome"]["pmv"], pmv[index])
                        and _same(
                            ordered[index]["outcome"]["effective_occupancy"], occupancy[index]
                        )
                        and _same(occupancy[index], expected_occupancy[index])
                        and _same(ordered[index]["outcome"]["power_w"], power)
                        and _same(ordered[index]["outcome"]["cost"], cost)
                        for index in range(len(zones))
                    )
                    and all(_same(price, prices[0]) for price in prices)
                    and _same(cost, power * 0.25 / 1000.0 * float(prices[0]))
                    and _same(reward, expected_reward)
                )
                previous_setpoints = [float(value) for value in setpoints]
                action_test_ids.update(item.get("test_id") for item in ordered)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                trajectory_ok = False
        checks["trajectory_recomputed"] = trajectory_ok
        checks["test_id_continuity"] = (
            len(lifecycle_test_ids) == 1
            and action_test_ids == lifecycle_test_ids
            and boundary_test_id in lifecycle_test_ids
        )

        if controller == "basic-rbc":
            controller_contract = True
            for step, diagnostic in enumerate(diagnostics):
                action_by_zone = {row["zone"]: row for row in by_step.get(step, [])}
                controller_contract = controller_contract and (
                    diagnostic == {"step": step, "status": "scheduled", "method_degraded": False}
                    and set(action_by_zone) == zone_set
                    and all(
                        _same(
                            action_by_zone[zone]["final_setpoint_c"],
                            25.0
                            if float(action_by_zone[zone]["outcome"]["effective_occupancy"]) > 0
                            else 30.0,
                        )
                        for zone in zones
                    )
                )
            checks["controller_contract"] = controller_contract
        elif controller == "enhanced-rbc":
            controller_contract = True
            for step, diagnostic in enumerate(diagnostics):
                zone_diagnostics = diagnostic.get("zones")
                action_by_zone = {row["zone"]: row for row in by_step.get(step, [])}
                controller_contract = controller_contract and (
                    set(diagnostic) == {"step", "status", "method_degraded", "zones"}
                    and diagnostic.get("step") == step
                    and diagnostic.get("status") == "canonical_program"
                    and diagnostic.get("method_degraded") is False
                    and isinstance(zone_diagnostics, Mapping)
                    and set(zone_diagnostics) == zone_set
                    and set(action_by_zone) == zone_set
                )
                if not controller_contract:
                    continue
                assert isinstance(zone_diagnostics, Mapping)
                controller_contract = all(
                    isinstance(zone_diagnostics[zone], Mapping)
                    and set(zone_diagnostics[zone]) == {"interpreter", "action_assurance"}
                    and isinstance(zone_diagnostics[zone]["interpreter"], Mapping)
                    and isinstance(zone_diagnostics[zone]["action_assurance"], Mapping)
                    and _same(
                        zone_diagnostics[zone]["action_assurance"].get("final_setpoint"),
                        action_by_zone[zone]["final_setpoint_c"],
                    )
                    for zone in zones
                )
            checks["controller_contract"] = controller_contract
        elif controller in {"c-drl", "h-drl"}:
            entry = model_entry(case, controller)
            expected_model = verify_checkpoint(entry)
            recorded_model = _load(directory / "model_identity.json")
            observations = _rows(directory / "observations.jsonl")
            inferences = _rows(directory / "policy_inference.jsonl")
            checks["model_identity"] = all(
                recorded_model.get(name) == expected_model[name] for name in ("bytes", "sha256")
            )
            policy_contract = (
                len(observations) == expected_steps
                and len(inferences) == expected_steps
                and [row.get("step") for row in observations] == list(range(expected_steps))
                and [row.get("step") for row in inferences] == list(range(expected_steps))
            )
            observation_dimension = int(entry["observation_dimension"])
            local_dimension = int(entry.get("local_observation_dimension", 0))
            policy_zones = tuple(str(zone) for zone in entry["policy_zone_order"])
            base_mode = str(entry["residual_base"])
            observation_builder: PolicyObservationBuilder | None = None
            try:
                if boundary_physical_state is None or not boundary_ok or not forecast_contract:
                    raise ValueError("policy reconstruction inputs are unavailable")
                observation_builder = PolicyObservationBuilder(profile, entry)
                observation_builder.reset(boundary_physical_state)
            except (KeyError, TypeError, ValueError):
                policy_contract = False
            for step in range(expected_steps):
                if not policy_contract:
                    break
                observation = observations[step]
                inference = inferences[step]
                diagnostic = diagnostics[step]
                local = observation.get("local_normalized")
                columns = observation.get("columns")
                raw_action = inference.get("raw_action")
                inferred_setpoints = inference.get("setpoints_c")
                action_by_zone = {row["zone"]: row for row in by_step.get(step, [])}
                policy_contract = (
                    set(observation)
                    == {
                        "step",
                        "time_seconds",
                        "columns",
                        "raw",
                        "normalized",
                        "local_normalized",
                    }
                    and observation.get("time_seconds") == evaluation_start + step * 900
                    and isinstance(columns, list)
                    and len(columns) == observation_dimension
                    and len(set(columns)) == observation_dimension
                    and all(isinstance(column, str) and column for column in columns)
                    and _numeric_sequence(observation.get("raw"), observation_dimension)
                    and _numeric_sequence(observation.get("normalized"), observation_dimension)
                    and isinstance(local, Mapping)
                    and (
                        set(local) == zone_set
                        and all(_numeric_sequence(local[zone], local_dimension) for zone in zones)
                        if controller == "h-drl"
                        else local == {}
                    )
                    and set(inference)
                    == {
                        "step",
                        "policy_zone_order",
                        "raw_action",
                        "setpoints_c",
                        "observation_dimension",
                    }
                    and inference.get("step") == step
                    and inference.get("policy_zone_order") == list(policy_zones)
                    and inference.get("observation_dimension") == observation_dimension
                    and _numeric_sequence(raw_action, len(policy_zones))
                    and isinstance(raw_action, Sequence)
                    and all(-1.0 <= float(value) <= 1.0 for value in raw_action)
                    and _exact_numeric_map(inferred_setpoints, zone_set)
                    and isinstance(inferred_setpoints, Mapping)
                    and set(action_by_zone) == zone_set
                    and set(diagnostic)
                    == {
                        "step",
                        "status",
                        "method_degraded",
                        "policy_zone_order",
                        "raw_action",
                        "setpoints_c",
                        "observation_dimension",
                        "policy_input_clothing_insulation",
                        "policy_input_pmv",
                    }
                    and diagnostic.get("status") == "policy_inference"
                    and diagnostic.get("method_degraded") is False
                    and all(diagnostic.get(key) == inference.get(key) for key in inference)
                    and _finite(diagnostic.get("policy_input_clothing_insulation"))
                    and _exact_numeric_map(diagnostic.get("policy_input_pmv"), zone_set)
                )
                if not policy_contract:
                    continue
                assert observation_builder is not None
                assert isinstance(raw_action, Sequence)
                assert isinstance(inferred_setpoints, Mapping)
                assert isinstance(local, Mapping)
                try:
                    packet = observation_builder.build(
                        expected_forecast,
                        step=step,
                        action_time_seconds=evaluation_start + step * 900,
                        step_seconds=900,
                    )
                    expected_local = {
                        zone: values.tolist() for zone, values in packet.local_normalized.items()
                    }
                    policy_contract = (
                        observation.get("columns") == list(packet.columns)
                        and all(
                            _same(actual, expected)
                            for actual, expected in zip(
                                observation["raw"], packet.raw.tolist(), strict=True
                            )
                        )
                        and all(
                            _same(actual, expected)
                            for actual, expected in zip(
                                observation["normalized"],
                                packet.normalized.tolist(),
                                strict=True,
                            )
                        )
                        and set(local) == set(expected_local)
                        and all(
                            all(
                                _same(actual, expected)
                                for actual, expected in zip(
                                    local[zone], expected_local[zone], strict=True
                                )
                            )
                            for zone in expected_local
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    policy_contract = False
                if not policy_contract:
                    continue
                for index, zone in enumerate(policy_zones):
                    occupancy = float(action_by_zone[zone]["outcome"]["effective_occupancy"])
                    base = 25.0 if base_mode == "fixed_25" or occupancy > 0 else 30.0
                    expected_setpoint = max(20.0, min(30.0, base + 5.0 * float(raw_action[index])))
                    policy_contract = (
                        policy_contract
                        and _same(inferred_setpoints[zone], expected_setpoint)
                        and _same(action_by_zone[zone]["final_setpoint_c"], expected_setpoint)
                    )
                if not policy_contract:
                    continue
                policy_pmv = diagnostic["policy_input_pmv"]
                assert isinstance(policy_pmv, Mapping)
                next_state = {
                    profile["zones"][zone]["temperature_sensor"]: float(
                        action_by_zone[zone]["outcome"]["zone_temperature_c"]
                    )
                    + 273.15
                    for zone in policy_zones
                }
                observation_builder.update(
                    next_state,
                    {zone: float(inferred_setpoints[zone]) for zone in policy_zones},
                    {zone: float(policy_pmv[zone]) for zone in policy_zones},
                    float(action_by_zone[policy_zones[0]]["outcome"]["power_w"]),
                )
            checks["policy_contract"] = policy_contract
        elif controller == "hierarchical-mpc":
            recorded_model = _load(directory / "mpc_model_identity.json")
            current_model = verify_frozen_mpc_model(case)
            predictions = _rows(directory / "predictions.jsonl")
            solver_rows = _rows(directory / "solver_trace.jsonl")
            checks["mpc_model_identity"] = (
                current_model.get("valid") is True
                and recorded_model == current_model
                and manifest.get("mpc_model_identity") == current_model.get("model_identity")
            )
            mpc_contract = (
                len(predictions) == expected_steps
                and len(solver_rows) == expected_steps
                and [row.get("step") for row in predictions] == list(range(expected_steps))
                and [row.get("step") for row in solver_rows] == list(range(expected_steps))
            )
            enhanced = EnhancedRbcController(
                zones, repository_root() / expected_plan["case_profile"]["program"]
            )
            last_setpoints = {zone: float(protocol["initial_setpoint_c"]) for zone in zones}
            last_pmv = {zone: 0.0 for zone in zones}
            last_occupancy = {zone: 0.0 for zone in zones}
            for step in range(expected_steps):
                if not mpc_contract:
                    break
                diagnostic = diagnostics[step]
                prediction = predictions[step]
                solver = solver_rows[step]
                action_by_zone = {row["zone"]: row for row in by_step.get(step, [])}
                action_time = evaluation_start + step * 900
                current_occupancy = {
                    zone: effective_count(
                        profile["occupancy"],
                        action_time,
                        float(
                            expected_forecast[profile["zones"][zone]["occupancy_forecast"]][step]
                        ),
                    )
                    for zone in zones
                }
                future_occupancy = {
                    zone: [
                        effective_count(
                            profile["occupancy"],
                            action_time + offset * 900,
                            float(
                                expected_forecast[profile["zones"][zone]["occupancy_forecast"]][
                                    step + offset
                                ]
                            ),
                        )
                        for offset in range(1, 5)
                    ]
                    for zone in zones
                }
                fallback_setpoints, _ = enhanced.decide(
                    occupancy=current_occupancy,
                    future_occupancy=future_occupancy,
                    last_setpoints_c=last_setpoints,
                    last_pmv=last_pmv,
                    last_occupancy=last_occupancy,
                )
                status = diagnostic.get("status")
                if status == "optimized":
                    planned_setpoints = diagnostic.get("planned_setpoints_c")
                    predicted_outputs = diagnostic.get("predicted_outputs")
                    mpc_contract = (
                        diagnostic.get("method_degraded") is False
                        and diagnostic.get("history_initialization") == "repeat_boundary_state"
                        and diagnostic.get("coordinator_updated") is (step % 4 == 0)
                        and diagnostic.get("feedback_iterations") in {0, 1}
                        and isinstance(planned_setpoints, list)
                        and len(planned_setpoints) == 4
                        and all(
                            isinstance(row, list)
                            and len(row) == len(zones)
                            and all(_finite(value) and 20 <= float(value) <= 30 for value in row)
                            for row in planned_setpoints
                        )
                        and isinstance(predicted_outputs, list)
                        and len(predicted_outputs) == 4
                        and all(
                            isinstance(row, list)
                            and len(row) == len(zones) + 1
                            and all(_finite(value) for value in row)
                            for row in predicted_outputs
                        )
                        and all(
                            _same(
                                action_by_zone[zone]["final_setpoint_c"],
                                planned_setpoints[0][index],
                            )
                            for index, zone in enumerate(zones)
                        )
                    )
                elif status == "fallback":
                    mpc_contract = (
                        diagnostic.get("method_degraded") is True
                        and diagnostic.get("history_initialization") == "repeat_boundary_state"
                        and isinstance(diagnostic.get("reason"), str)
                        and all(
                            _same(
                                action_by_zone[zone]["final_setpoint_c"],
                                fallback_setpoints[zone],
                            )
                            for zone in zones
                        )
                    )
                else:
                    mpc_contract = False
                mpc_contract = mpc_contract and solver == {"step": step, **diagnostic}
                mpc_contract = mpc_contract and prediction == {
                    "step": step,
                    "predicted_outputs": diagnostic.get("predicted_outputs"),
                    "negative_power_prediction_count": 0,
                }
                if action_by_zone:
                    last_setpoints = {
                        zone: float(action_by_zone[zone]["final_setpoint_c"]) for zone in zones
                    }
                    last_pmv = {
                        zone: float(action_by_zone[zone]["outcome"]["pmv"]) for zone in zones
                    }
                    last_occupancy = current_occupancy
            checks["mpc_controller_contract"] = mpc_contract

        errors = sorted(name for name, passed in checks.items() if not passed)
        execution_integrity = not errors
        classification = (
            "RUN-INVALID"
            if not execution_integrity
            else "METHOD-DEGRADED"
            if metrics["controller"]["method_degraded"]
            else "BASELINE-PASS"
        )
        if require_completion:
            completion = _load(directory / "completion.json")
            checks["completion_identity"] = (
                set(completion)
                == {
                    "completion_schema",
                    "schema_version",
                    "classification",
                    "run_identity",
                    "elapsed_seconds",
                }
                and completion.get("completion_schema") == "h3c_baseline_completion"
                and completion.get("schema_version") == 1
                and completion.get("classification") == classification
                and completion.get("run_identity") == manifest.get("run_identity")
                and _finite(completion.get("elapsed_seconds"))
                and float(completion["elapsed_seconds"]) >= 0
            )
            if not checks["completion_identity"]:
                errors.append("completion_identity")
                execution_integrity = False
                classification = "RUN-INVALID"
        return {
            "verification_schema": "h3c_baseline_verification",
            "schema_version": 1,
            "checks": checks,
            "errors": sorted(set(errors)),
            "execution_integrity": execution_integrity,
            "completion_eligible": execution_integrity,
            "classification": classification,
            "metrics_identity": _identity(metrics),
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _invalid_result(type(error).__name__)
