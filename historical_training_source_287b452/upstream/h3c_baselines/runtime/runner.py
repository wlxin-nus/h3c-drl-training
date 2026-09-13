"""Fresh conditioned BOPTEST runner shared by independent baseline controllers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from h3c.experiments.profiles import repository_root
from h3c.experiments.settings import load_runtime_contract
from h3c.runtime.clients import BoptestHttpClient
from h3c.runtime.comfort import ComfortModel, step_reward
from h3c.runtime.execution_lock import physical_execution_lock
from h3c.runtime.occupancy import effective_count
from h3c.runtime.protocol import (
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
from h3c.runtime.source_identity import committed_source_identity
from h3c_baselines.configuration import (
    BaselineRunPlan,
    formal_evaluation_plans,
    load_hierarchical_mpc_config,
    mpc_formal_evaluation_plans,
)
from h3c_baselines.controllers.basic_rbc import basic_rbc_setpoints
from h3c_baselines.controllers.drl import FrozenDrlController
from h3c_baselines.controllers.enhanced_rbc import EnhancedRbcController
from h3c_baselines.models import model_entry, verify_checkpoint
from h3c_baselines.mpc.optimizer import HierarchicalMpcController
from h3c_baselines.mpc.registry import verify_frozen_mpc_suite
from h3c_baselines.mpc.training import verify_frozen_mpc_model
from h3c_baselines.mpc.vector_arx import FittedArxModel
from h3c_baselines.outputs.artifacts import BaselineArtifacts
from h3c_baselines.outputs.integrity import secret_occurrences
from h3c_baselines.outputs.metrics import compute_baseline_metrics
from h3c_baselines.outputs.verification import (
    verify_baseline_run,
    verify_concurrent_suite_evidence,
)
from h3c_baselines.policies.observation_contracts import PolicyObservationBuilder

PhysicalFactory = Callable[[str], PhysicalClient]


@dataclass(frozen=True)
class _PreparedBaselineRun:
    plan: BaselineRunPlan
    resolved: dict[str, Any]
    drl: FrozenDrlController | None
    observation_builder: PolicyObservationBuilder | None
    drl_identity: dict[str, Any] | None
    mpc_model: FittedArxModel | None
    mpc_identity: dict[str, Any] | None
    enhanced_controller: EnhancedRbcController | None
    mpc_controller: HierarchicalMpcController | None
    source_commit: str
    execution_identity: dict[str, Any]
    run_identity: str
    run_id: str


class ConcurrentBaselineExecutionError(RuntimeError):
    """Raised only after every submitted independent baseline arm has finished."""

    def __init__(self, summary: dict[str, Any]) -> None:
        self.summary = summary
        failed = summary.get("failed_runs", [])
        super().__init__(f"{len(failed)} concurrent baseline arm(s) failed")


class _SelectedTestIdentityRegistry:
    """Claim each physical identity at select time, before initialization."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owners: dict[str, str] = {}

    def claim(self, test_id: str, run_identity: str) -> None:
        if not test_id:
            raise ValueError("selected physical test identity is empty")
        with self._lock:
            owner = self._owners.get(test_id)
            if owner is not None:
                raise ValueError(f"selected physical test identity is already owned by run {owner}")
            self._owners[test_id] = run_identity


def _source_commit() -> str:
    return committed_source_identity()


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _future_occupancy(
    profile: Mapping[str, Any],
    forecast: Mapping[str, Sequence[float]],
    zones: tuple[str, ...],
    *,
    step: int,
    action_time: int,
) -> dict[str, list[float]]:
    return {
        zone: [
            effective_count(
                profile["occupancy"],
                action_time + offset * 900,
                float(forecast[profile["zones"][zone]["occupancy_forecast"]][step + offset]),
            )
            for offset in range(1, 5)
        ]
        for zone in zones
    }


def _legacy_policy_daily_outdoor_mean_c(
    profile: Mapping[str, Any],
    forecast: Mapping[str, Sequence[float]],
    *,
    step: int,
) -> float:
    """Return the original DRL policy's 96 consecutive 15-minute samples."""
    outdoor_point = str(profile["global_inputs"]["outdoor_temperature"])
    values = forecast[outdoor_point][step : step + 96]
    if len(values) != 96:
        raise ValueError("policy comfort forecast lacks 96 consecutive samples")
    return sum(float(value) - 273.15 for value in values) / 96.0


def _legacy_policy_pmv(
    comfort: ComfortModel,
    profile: Mapping[str, Any],
    forecast: Mapping[str, Sequence[float]],
    temperatures_c: Mapping[str, float],
    *,
    step: int,
    action_time: int,
) -> dict[str, float]:
    comfort.update_clothing(
        action_time,
        _legacy_policy_daily_outdoor_mean_c(profile, forecast, step=step),
    )
    return {zone: comfort.pmv(value) for zone, value in temperatures_c.items()}


def _mpc_disturbances(
    profile: Mapping[str, Any],
    forecast: Mapping[str, Sequence[float]],
    zones: tuple[str, ...],
    *,
    step: int,
    action_time: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    list[int],
    list[float],
]:
    disturbances: list[list[float]] = []
    occupancy_rows: list[list[float]] = []
    prices: list[float] = []
    action_times: list[int] = []
    daily_means: list[float] = []
    outdoor_point = profile["global_inputs"]["outdoor_temperature"]
    for horizon in range(4):
        time_seconds = action_time + horizon * 900
        fraction = (time_seconds % 86400) / 86400.0
        occupancy = [
            effective_count(
                profile["occupancy"],
                time_seconds,
                float(forecast[profile["zones"][zone]["occupancy_forecast"]][step + horizon]),
            )
            for zone in zones
        ]
        disturbances.append(
            [
                float(forecast[outdoor_point][step + horizon]) - 273.15,
                float(forecast[profile["global_inputs"]["solar_irradiance"]][step + horizon]),
                *occupancy,
                math.sin(2.0 * math.pi * fraction),
                math.cos(2.0 * math.pi * fraction),
            ]
        )
        occupancy_rows.append(occupancy)
        prices.append(
            float(forecast[profile["global_inputs"]["electricity_price"]][step + horizon])
        )
        action_times.append(time_seconds)
        daily = forecast[outdoor_point][step + horizon : step + horizon + 97 : 4]
        daily_means.append(sum(float(value) - 273.15 for value in daily) / len(daily))
    return (
        np.asarray(disturbances, dtype=np.float64),
        np.asarray(prices, dtype=np.float64),
        np.asarray(occupancy_rows, dtype=np.float64),
        action_times,
        daily_means,
    )


def _native_kpis(physical: PhysicalClient) -> dict[str, Any]:
    method = getattr(physical, "get_kpis", None)
    if not callable(method):
        raise ValueError("physical client does not implement native KPI retrieval")
    value = method()
    if not isinstance(value, dict):
        raise ValueError("native BOPTEST KPI response is invalid")
    return value


def _publish_finalized_failure(
    artifacts: BaselineArtifacts,
    manifest: dict[str, Any],
    failure_details: Mapping[str, Any],
) -> None:
    """Finalize inspectable evidence before publishing the terminal sentinel."""

    artifacts.write_new_json("failure_details.json", failure_details)
    manifest["secret_exposure_count"] = secret_occurrences(artifacts.run_dir)
    manifest["secret_scan_status"] = "completed"
    artifacts.replace_json("manifest.json", manifest)
    artifacts.publish_failure(
        {
            "failure_schema": "h3c_baseline_failure",
            "schema_version": 2,
            "classification": "RUN-INVALID",
            "run_identity": failure_details["run_identity"],
            "failure_details_identity": _identity(failure_details),
        }
    )


def _verified_mpc_freeze_identity() -> str:
    verification = verify_frozen_mpc_suite()
    if verification.get("valid") is not True:
        raise ValueError("frozen MPC suite verification failed")
    manifest_path = Path(str(verification["target"])) / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze_identity = manifest.get("freeze_identity")
    if not isinstance(freeze_identity, str) or not freeze_identity:
        raise ValueError("frozen MPC suite identity is missing")
    return freeze_identity


def _prepare_baseline_run(
    plan: BaselineRunPlan,
    *,
    endpoint: str,
    source_commit: str,
    dispatch_mode: str = "strictly_serial",
    suite_identity: str | None = None,
    mpc_freeze_identity: str | None = None,
) -> _PreparedBaselineRun:
    """Resolve one arm and load all optional dependencies before physical work."""

    resolved = plan.resolved()
    profile = resolved["case_profile"]
    zones = tuple(profile["zones"])
    drl: FrozenDrlController | None = None
    observation_builder: PolicyObservationBuilder | None = None
    drl_identity: dict[str, Any] | None = None
    if plan.controller in {"c-drl", "h-drl"}:
        entry = model_entry(plan.case, plan.controller)
        drl_identity = verify_checkpoint(entry)
        drl = FrozenDrlController(plan.case, plan.controller)
        observation_builder = PolicyObservationBuilder(profile, entry)

    mpc_model: FittedArxModel | None = None
    mpc_identity: dict[str, Any] | None = None
    enhanced_controller = (
        EnhancedRbcController(zones, repository_root() / profile["program"])
        if plan.controller in {"enhanced-rbc", "hierarchical-mpc"}
        else None
    )
    mpc_controller: HierarchicalMpcController | None = None
    if plan.controller == "hierarchical-mpc":
        mpc_freeze_identity = mpc_freeze_identity or _verified_mpc_freeze_identity()
        mpc_identity = verify_frozen_mpc_model(plan.case)
        if mpc_identity["valid"] is not True:
            raise ValueError("frozen hierarchical MPC model verification failed")
        mpc_model = FittedArxModel.load(
            repository_root() / "models" / "mpc" / plan.case / "model_coefficients.npz"
        )
        if mpc_model.layout.zones != zones:
            raise ValueError("MPC model zone layout is invalid")
        mpc_controller = HierarchicalMpcController(
            mpc_model,
            profile["objective"],
            load_hierarchical_mpc_config()["excitation"],
        )

    resolved_suite_identity = suite_identity or _identity(
        {
            "dispatch_mode": dispatch_mode,
            "plans": [resolved["plan_identity"]],
            "source_commit": source_commit,
        }
    )

    execution_identity = {
        "plan_identity": resolved["plan_identity"],
        "source_commit": source_commit,
        "physical_endpoint_identity": _identity(endpoint.rstrip("/")),
        "mpc_model_identity": None if mpc_model is None else mpc_model.identity,
        "dispatch_mode": dispatch_mode,
        "suite_identity": resolved_suite_identity,
        "mpc_freeze_identity": mpc_freeze_identity,
    }
    run_identity = _identity(execution_identity)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + run_identity[:12]
    resolved["execution_identity"] = execution_identity
    return _PreparedBaselineRun(
        plan=plan,
        resolved=resolved,
        drl=drl,
        observation_builder=observation_builder,
        drl_identity=drl_identity,
        mpc_model=mpc_model,
        mpc_identity=mpc_identity,
        enhanced_controller=enhanced_controller,
        mpc_controller=mpc_controller,
        source_commit=source_commit,
        execution_identity=execution_identity,
        run_identity=run_identity,
        run_id=run_id,
    )


def _execute_one(
    plan: BaselineRunPlan,
    *,
    suite: str,
    endpoint: str,
    output_root: Path,
    physical_factory: PhysicalFactory,
    prepared: _PreparedBaselineRun | None = None,
    reserved_run_directory: bool = False,
    selected_test_registry: _SelectedTestIdentityRegistry | None = None,
) -> dict[str, Any]:
    prepared_run = prepared or _prepare_baseline_run(
        plan,
        endpoint=endpoint,
        source_commit=_source_commit(),
    )
    if prepared_run.plan != plan:
        raise ValueError("prepared baseline arm does not match the requested plan")
    resolved = prepared_run.resolved
    profile = resolved["case_profile"]
    drl = prepared_run.drl
    observation_builder = prepared_run.observation_builder
    drl_identity = prepared_run.drl_identity
    mpc_model = prepared_run.mpc_model
    mpc_identity = prepared_run.mpc_identity
    enhanced = prepared_run.enhanced_controller
    mpc = prepared_run.mpc_controller
    execution_identity = prepared_run.execution_identity
    run_identity = prepared_run.run_identity
    artifacts = BaselineArtifacts(
        output_root,
        suite,
        plan.case,
        prepared_run.run_id,
        plan.controller,
    )
    manifest: dict[str, Any] = {
        "manifest_schema": "h3c_baseline_manifest",
        "schema_version": 1,
        "run_identity": run_identity,
        "source_commit": prepared_run.source_commit,
        "case": plan.case,
        "controller": plan.controller,
        "conditioning_prefix_identity": None,
        "evaluation_boundary_identity": None,
        "mpc_model_identity": execution_identity["mpc_model_identity"],
        "secret_scan_status": "pending",
        "secret_exposure_count": None,
        "occupancy_forecast_missing_value_resolution_count": 0,
        "lifecycle": {
            "initialize_count": 0,
            "conditioning_advance_count": 0,
            "evaluation_advance_count": 0,
            "stop_count": 0,
            "test_id_changes": 0,
        },
    }
    try:
        artifacts.create(
            resolved,
            manifest,
            reserved_by_execution_lock=reserved_run_directory,
        )
    except Exception as error:
        # Artifact creation is intentionally before any physical client exists. A
        # manifest-backed partial directory can be closed as terminal evidence;
        # failure before manifest creation remains non-terminal and is owned by
        # the suite-level failure evidence.
        if (artifacts.run_dir / "manifest.json").is_file():
            _publish_finalized_failure(
                artifacts,
                manifest,
                {
                    "failure_details_schema": "h3c_baseline_failure_details",
                    "schema_version": 1,
                    "classification": "RUN-INVALID",
                    "run_identity": run_identity,
                    "primary_failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "stop_failure": None,
                    "elapsed_seconds": 0.0,
                },
            )
        raise
    try:
        if drl_identity is not None:
            artifacts.write_new_json("model_identity.json", drl_identity)
        if mpc_identity is not None:
            artifacts.write_new_json("mpc_model_identity.json", mpc_identity)
    except Exception as error:
        _publish_finalized_failure(
            artifacts,
            manifest,
            {
                "failure_details_schema": "h3c_baseline_failure_details",
                "schema_version": 1,
                "classification": "RUN-INVALID",
                "run_identity": run_identity,
                "primary_failure": {"type": type(error).__name__, "message": str(error)},
                "stop_failure": None,
                "elapsed_seconds": 0.0,
            },
        )
        raise
    physical: PhysicalClient | None = None
    initialized = False
    stop_attempted = False
    frozen_test_id: str | None = None
    evaluation_start = int(profile["evaluation_start_day"]) * 86400
    evaluation_end = evaluation_start + plan.evaluation_hours * 3600
    last_physical_time = evaluation_start
    started = time.perf_counter()
    primary_error: Exception | None = None
    stop_error: Exception | None = None
    try:
        physical = physical_factory(endpoint)
        lifecycle_sink_setter = getattr(physical, "set_lifecycle_sink", None)
        if selected_test_registry is not None and not callable(lifecycle_sink_setter):
            raise ValueError("concurrent physical client lacks lifecycle identity evidence")
        if callable(lifecycle_sink_setter):

            def record_lifecycle(row: Mapping[str, Any]) -> None:
                if row.get("event") == "selected" and selected_test_registry is not None:
                    test_id = row.get("test_id")
                    if not isinstance(test_id, str):
                        raise ValueError("selected physical test identity is invalid")
                    selected_test_registry.claim(test_id, run_identity)
                artifacts.append_jsonl("timing.jsonl", dict(row))

            lifecycle_sink_setter(record_lifecycle)
        boundary = initialize_evaluation_boundary(
            physical,
            profile,
            artifacts,
            on_initialized=lambda _test_id: manifest["lifecycle"].update({"initialize_count": 1}),
        )
        initialized = True
        frozen_test_id = boundary.test_id
        manifest["conditioning_prefix_identity"] = boundary.conditioning_prefix_identity
        manifest["evaluation_boundary_identity"] = boundary.evaluation_boundary_identity
        manifest["lifecycle"]["conditioning_advance_count"] = 0
        zones = tuple(profile["zones"])
        steps = plan.evaluation_hours * 4
        artifacts.append_jsonl(
            "timing.jsonl",
            {
                "phase": "physical_lifecycle",
                "event": "evaluation_started",
                "time_seconds": evaluation_start,
                "test_id": boundary.test_id,
            },
        )
        points = forecast_points(profile)
        source_forecast = physical.forecast(points, (steps + 96) * 900, 900)
        forecast, resolution_events = resolve_forecast_missing_occupancy(
            profile,
            source_forecast,
            points,
            steps + 97,
            forecast_phase="evaluation",
            start_time_seconds=evaluation_start,
            step_seconds=900,
        )
        artifacts.write_new_json(
            "forecast_inputs.json",
            build_forecast_evidence(
                points,
                source_forecast,
                forecast,
                start_time_seconds=evaluation_start,
                step_seconds=900,
            ),
        )
        for event in resolution_events:
            artifacts.append_jsonl("timing.jsonl", event)
        manifest["occupancy_forecast_missing_value_resolution_count"] = len(resolution_events)
        state = boundary.state
        last_setpoints = dict(boundary.last_setpoint_c)
        last_pmv = dict(boundary.last_pmv)
        last_occupancy = dict(boundary.last_occupancy)
        policy_comfort = ComfortModel(profile["comfort"]) if drl is not None else None
        if observation_builder is not None:
            observation_builder.reset(state)
        output_history: NDArray[np.float64] | None = None
        control_history: NDArray[np.float64] | None = None
        if plan.controller == "hierarchical-mpc":
            assert mpc_model is not None
            if (
                mpc_model.identity != manifest["mpc_model_identity"]
                or mpc_model.layout.zones != zones
            ):
                raise ValueError("MPC model identity or zone layout is invalid")
            if mpc is None:
                raise ValueError("MPC controller was not constructed during preflight")
            boundary_output = np.asarray(
                [
                    *[zone_temperature_c(profile, state, zone) for zone in zones],
                    site_power(profile, state),
                ],
                dtype=np.float64,
            )
            output_history = np.vstack([boundary_output] * 4)
            control_history = np.vstack(
                [[float(profile["protocol"]["initial_setpoint_c"])] * len(zones)] * 4
            )

        for step in range(steps):
            action_time = evaluation_start + step * 900
            require_time(state, action_time)
            current_occupancy = {
                zone: effective_count(
                    profile["occupancy"],
                    action_time,
                    float(forecast[profile["zones"][zone]["occupancy_forecast"]][step]),
                )
                for zone in zones
            }
            future = _future_occupancy(profile, forecast, zones, step=step, action_time=action_time)
            enhanced_setpoints: dict[str, float] | None = None
            enhanced_diagnostics: dict[str, Any] | None = None
            if enhanced is not None:
                enhanced_setpoints, enhanced_diagnostics = enhanced.decide(
                    occupancy=current_occupancy,
                    future_occupancy=future,
                    last_setpoints_c=last_setpoints,
                    last_pmv=last_pmv,
                    last_occupancy=last_occupancy,
                )
            diagnostics: dict[str, Any]
            if plan.controller == "basic-rbc":
                setpoints = basic_rbc_setpoints(zones, current_occupancy)
                diagnostics = {"status": "scheduled", "method_degraded": False}
            elif plan.controller == "enhanced-rbc":
                assert enhanced_setpoints is not None and enhanced_diagnostics is not None
                setpoints = enhanced_setpoints
                diagnostics = {
                    "status": "canonical_program",
                    "method_degraded": False,
                    "zones": enhanced_diagnostics,
                }
            elif plan.controller in {"c-drl", "h-drl"}:
                assert drl is not None and observation_builder is not None
                packet = observation_builder.build(
                    forecast, step=step, action_time_seconds=action_time, step_seconds=900
                )
                setpoints, policy_diagnostics = drl.decide(packet, current_occupancy)
                diagnostics = {
                    "status": "policy_inference",
                    "method_degraded": False,
                    **policy_diagnostics,
                }
                artifacts.append_jsonl(
                    "observations.jsonl",
                    {
                        "step": step,
                        "time_seconds": action_time,
                        "columns": list(packet.columns),
                        "raw": packet.raw.tolist(),
                        "normalized": packet.normalized.tolist(),
                        "local_normalized": {
                            zone: value.tolist() for zone, value in packet.local_normalized.items()
                        },
                    },
                )
                artifacts.append_jsonl(
                    "policy_inference.jsonl", {"step": step, **policy_diagnostics}
                )
            else:
                assert (
                    mpc is not None and output_history is not None and control_history is not None
                )
                assert enhanced_setpoints is not None
                disturbances, prices, occupancy_horizon, times, daily_means = _mpc_disturbances(
                    profile, forecast, zones, step=step, action_time=action_time
                )
                decision = mpc.decide(
                    step=step,
                    output_history=output_history,
                    control_history=control_history,
                    disturbances=disturbances,
                    prices=prices,
                    occupancy=occupancy_horizon,
                    terminal_occupancy={zone: future[zone][3] for zone in zones},
                    action_times=times,
                    daily_outdoor_means_c=daily_means,
                    comfort=boundary.comfort,
                    previous_setpoints_c=last_setpoints,
                    enhanced_rbc_warm_start=enhanced_setpoints,
                )
                setpoints = decision.setpoints_c
                diagnostics = decision.diagnostics
                artifacts.append_jsonl("solver_trace.jsonl", {"step": step, **diagnostics})
                artifacts.append_jsonl(
                    "predictions.jsonl",
                    {
                        "step": step,
                        "predicted_outputs": diagnostics.get("predicted_outputs"),
                        "negative_power_prediction_count": diagnostics.get(
                            "negative_power_prediction_count", 0
                        ),
                    },
                )
            next_state = physical.advance(control_input(profile, setpoints))
            manifest["lifecycle"]["evaluation_advance_count"] += 1
            require_time(next_state, action_time + 900)
            last_physical_time = action_time + 900
            if physical.test_id != boundary.test_id:
                manifest["lifecycle"]["test_id_changes"] += 1
                raise ValueError("test id changed during baseline evaluation")
            outdoor = forecast[profile["global_inputs"]["outdoor_temperature"]][
                step : step + 97 : 4
            ]
            boundary.comfort.update_clothing(
                action_time, sum(float(value) - 273.15 for value in outdoor) / len(outdoor)
            )
            temperatures = {zone: zone_temperature_c(profile, next_state, zone) for zone in zones}
            pmv = {zone: boundary.comfort.pmv(temperatures[zone]) for zone in zones}
            policy_pmv = pmv
            if policy_comfort is not None:
                policy_pmv = _legacy_policy_pmv(
                    policy_comfort,
                    profile,
                    forecast,
                    temperatures,
                    step=step,
                    action_time=action_time,
                )
                diagnostics["policy_input_clothing_insulation"] = policy_comfort.clothing_insulation
                diagnostics["policy_input_pmv"] = policy_pmv
            power = site_power(profile, next_state)
            price = float(forecast[profile["global_inputs"]["electricity_price"]][step])
            cost = power * 0.25 / 1000.0 * price
            reward = step_reward(
                cost=cost,
                pmv=[pmv[zone] for zone in zones],
                occupancy=[current_occupancy[zone] for zone in zones],
                setpoints_c=[setpoints[zone] for zone in zones],
                previous_setpoints_c=[last_setpoints[zone] for zone in zones],
                objective=profile["objective"],
            )
            for zone in zones:
                artifacts.append_jsonl(
                    "actions.jsonl",
                    {
                        "step": step,
                        "zone": zone,
                        "test_id": boundary.test_id,
                        "action_time_seconds": action_time,
                        "outcome_time_seconds": action_time + 900,
                        "final_setpoint_c": setpoints[zone],
                        "outcome": {
                            "zone_temperature_c": temperatures[zone],
                            "pmv": pmv[zone],
                            "effective_occupancy": current_occupancy[zone],
                            "power_w": power,
                            "electricity_price": price,
                            "cost": cost,
                        },
                    },
                )
            artifacts.append_jsonl("controller_diagnostics.jsonl", {"step": step, **diagnostics})
            artifacts.append_performance(
                (
                    action_time,
                    step,
                    power,
                    cost,
                    reward,
                    _canonical([temperatures[zone] for zone in zones]),
                    _canonical([setpoints[zone] for zone in zones]),
                    _canonical([pmv[zone] for zone in zones]),
                    _canonical([current_occupancy[zone] for zone in zones]),
                )
            )
            if observation_builder is not None:
                observation_builder.update(next_state, setpoints, policy_pmv, power)
            if output_history is not None and control_history is not None:
                output_history = np.vstack(
                    ([*[temperatures[zone] for zone in zones], power], output_history[:-1])
                )
                control_history = np.vstack(
                    ([*[setpoints[zone] for zone in zones]], control_history[:-1])
                )
            state = next_state
            last_setpoints = setpoints
            last_pmv = pmv
            last_occupancy = current_occupancy
        artifacts.append_jsonl(
            "timing.jsonl",
            {
                "phase": "physical_lifecycle",
                "event": "evaluation_completed",
                "time_seconds": evaluation_end,
                "test_id": boundary.test_id,
            },
        )
        artifacts.write_new_json("native_boptest_kpis.json", _native_kpis(physical))
    except Exception as error:
        primary_error = error
    finally:
        if physical is not None and (initialized or physical.test_id is not None):
            stop_attempted = True
            stopped_test_id = frozen_test_id or physical.test_id
            try:
                physical.stop()
                manifest["lifecycle"]["stop_count"] += 1
                artifacts.append_jsonl(
                    "timing.jsonl",
                    {
                        "phase": "physical_lifecycle",
                        "event": "stopped",
                        "time_seconds": last_physical_time,
                        "test_id": stopped_test_id,
                    },
                )
            except Exception as error:
                stop_error = error
    if primary_error is not None or stop_error is not None:
        _publish_finalized_failure(
            artifacts,
            manifest,
            {
                "failure_details_schema": "h3c_baseline_failure_details",
                "schema_version": 1,
                "classification": "RUN-INVALID",
                "run_identity": run_identity,
                "primary_failure": (
                    None
                    if primary_error is None
                    else {"type": type(primary_error).__name__, "message": str(primary_error)}
                ),
                "stop_failure": (
                    None
                    if stop_error is None
                    else {"type": type(stop_error).__name__, "message": str(stop_error)}
                ),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        if primary_error is not None:
            if stop_error is not None:
                raise primary_error from stop_error
            raise primary_error
        assert stop_error is not None
        raise stop_error
    manifest["secret_exposure_count"] = secret_occurrences(artifacts.run_dir)
    manifest["secret_scan_status"] = "completed"
    artifacts.replace_json("manifest.json", manifest)
    if not stop_attempted:
        raise AssertionError("baseline execution exited without stopping the physical test")
    try:
        metrics = compute_baseline_metrics(artifacts.run_dir)
        artifacts.write_new_json("metrics.json", metrics)
        verification = verify_baseline_run(artifacts.run_dir, require_completion=False)
        artifacts.write_new_json("verification.json", verification)
        if verification["completion_eligible"] is not True:
            raise ValueError(f"baseline verification failed: {verification['errors']}")
    except Exception as error:
        _publish_finalized_failure(
            artifacts,
            manifest,
            {
                "failure_details_schema": "h3c_baseline_failure_details",
                "schema_version": 1,
                "classification": "RUN-INVALID",
                "run_identity": run_identity,
                "primary_failure": {"type": type(error).__name__, "message": str(error)},
                "stop_failure": None,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
    completion = {
        "completion_schema": "h3c_baseline_completion",
        "schema_version": 1,
        "classification": verification["classification"],
        "run_identity": run_identity,
        "elapsed_seconds": time.perf_counter() - started,
    }
    artifacts.publish_completion(completion)
    final = verify_baseline_run(artifacts.run_dir)
    if final["execution_integrity"] is not True:
        raise ValueError(f"baseline completion verification failed: {final['errors']}")
    return {
        "case": plan.case,
        "controller": plan.controller,
        "classification": final["classification"],
        "run_dir": str(artifacts.run_dir),
        "test_id": frozen_test_id,
        "run_identity": run_identity,
        "plan_identity": resolved["plan_identity"],
    }


def execute_baseline_plans(
    plans: Sequence[BaselineRunPlan],
    *,
    suite: str,
    output_root: Path | None = None,
    physical_factory: PhysicalFactory | None = None,
    lock_root: Path | None = None,
) -> dict[str, Any]:
    if not plans:
        raise ValueError("baseline execution requires at least one plan")
    runtime = load_runtime_contract()
    endpoint_name = runtime["physical_service"]["endpoint_environment_variable"]
    endpoint = os.environ.get(endpoint_name, "").rstrip("/")
    if not endpoint:
        raise ValueError(f"{endpoint_name} is required for baseline execution")
    root = (output_root or repository_root() / "outputs" / "baselines" / "runs").resolve()
    resolved_lock_root = (lock_root or repository_root() / "outputs" / "runs").resolve()
    results: list[dict[str, Any]] = []
    with physical_execution_lock(resolved_lock_root):
        for plan in plans:
            results.append(
                _execute_one(
                    plan,
                    suite=suite,
                    endpoint=endpoint,
                    output_root=root,
                    physical_factory=physical_factory or BoptestHttpClient,
                )
            )
    return {"execution": "strictly_serial", "completed_runs": results}


def _execute_prepared_concurrent_arm(
    prepared: _PreparedBaselineRun,
    *,
    suite: str,
    endpoint: str,
    output_root: Path,
    physical_factory: PhysicalFactory,
    selected_test_registry: _SelectedTestIdentityRegistry,
) -> dict[str, Any]:
    artifacts = BaselineArtifacts(
        output_root,
        suite,
        prepared.plan.case,
        prepared.run_id,
        prepared.plan.controller,
    )
    with physical_execution_lock(artifacts.run_dir):
        return _execute_one(
            prepared.plan,
            suite=suite,
            endpoint=endpoint,
            output_root=output_root,
            physical_factory=physical_factory,
            prepared=prepared,
            reserved_run_directory=True,
            selected_test_registry=selected_test_registry,
        )


def _concurrent_failure(
    prepared: _PreparedBaselineRun,
    error: Exception,
    *,
    suite: str,
    output_root: Path,
) -> dict[str, Any]:
    artifacts = BaselineArtifacts(
        output_root,
        suite,
        prepared.plan.case,
        prepared.run_id,
        prepared.plan.controller,
    )
    test_ids: set[str] = set()
    timing_path = artifacts.run_dir / "timing.jsonl"
    if timing_path.is_file():
        for line in timing_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            test_id = row.get("test_id")
            if isinstance(test_id, str) and test_id:
                test_ids.add(test_id)
    return {
        "case": prepared.plan.case,
        "controller": prepared.plan.controller,
        "classification": "RUN-INVALID",
        "run_dir": str(artifacts.run_dir),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "failure_evidence_present": (artifacts.run_dir / "failure.json").is_file(),
        "run_identity": prepared.run_identity,
        "test_id": next(iter(test_ids)) if len(test_ids) == 1 else None,
        "plan_identity": prepared.resolved["plan_identity"],
    }


def execute_baseline_plans_concurrently(
    plans: Sequence[BaselineRunPlan],
    *,
    suite: str,
    output_root: Path | None = None,
    physical_factory: PhysicalFactory | None = None,
    lock_root: Path | None = None,
) -> dict[str, Any]:
    """Execute independent arms once each with BOPTEST-owned dynamic admission."""

    if not plans:
        raise ValueError("baseline execution requires at least one plan")
    runtime = load_runtime_contract()
    endpoint_name = runtime["physical_service"]["endpoint_environment_variable"]
    endpoint = os.environ.get(endpoint_name, "").rstrip("/")
    if not endpoint:
        raise ValueError(f"{endpoint_name} is required for baseline execution")
    root = (output_root or repository_root() / "outputs" / "baselines" / "runs").resolve()
    resolved_lock_root = (lock_root or repository_root() / "outputs" / "runs").resolve()

    # Source identity and every optional dependency are resolved on the caller thread.
    # No artifacts, workers, test ids, or physical requests exist before this completes.
    source_commit = _source_commit()
    plan_identities = [plan.resolved()["plan_identity"] for plan in plans]
    mpc_freeze_identity = (
        _verified_mpc_freeze_identity()
        if any(plan.controller == "hierarchical-mpc" for plan in plans)
        else None
    )
    suite_identity = _identity(
        {
            "dispatch_mode": "auto",
            "plans": plan_identities,
            "source_commit": source_commit,
            "suite": suite,
            "mpc_freeze_identity": mpc_freeze_identity,
        }
    )
    prepared = [
        _prepare_baseline_run(
            plan,
            endpoint=endpoint,
            source_commit=source_commit,
            dispatch_mode="auto",
            suite_identity=suite_identity,
            mpc_freeze_identity=mpc_freeze_identity,
        )
        for plan in plans
    ]
    for item in prepared:
        artifacts = BaselineArtifacts(
            root,
            suite,
            item.plan.case,
            item.run_id,
            item.plan.controller,
        )
        if artifacts.run_dir.exists():
            raise ValueError("fresh baseline run directory already exists")

    suite_lock_identity = suite_identity
    suite_lock_root = resolved_lock_root / "concurrent-suites" / suite_lock_identity
    suite_evidence_dir = root / suite / "suite-evidence"
    suite_evidence_path = suite_evidence_dir / f"{suite_identity}.json"
    suite_claim_path = suite_evidence_dir / f".{suite_identity}.claim"
    factory = physical_factory or BoptestHttpClient
    selected_test_registry = _SelectedTestIdentityRegistry()
    completed: list[dict[str, Any] | None] = [None] * len(prepared)
    failures: list[dict[str, Any] | None] = [None] * len(prepared)
    with physical_execution_lock(suite_lock_root):
        if suite_evidence_path.exists() or suite_claim_path.exists():
            raise ValueError("concurrent suite evidence already exists")
        suite_evidence_dir.mkdir(parents=True, exist_ok=True)
        with suite_claim_path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(suite_identity + "\n")
            file.flush()
            os.fsync(file.fileno())
        # This is one local task per registered arm, not a physical worker limit.
        # BOPTEST alone admits each selected test as Running or Queued.
        with ThreadPoolExecutor(
            max_workers=len(prepared),
            thread_name_prefix="h3c-baseline-arm",
        ) as executor:
            futures: dict[Future[dict[str, Any]], int] = {
                executor.submit(
                    _execute_prepared_concurrent_arm,
                    item,
                    suite=suite,
                    endpoint=endpoint,
                    output_root=root,
                    physical_factory=factory,
                    selected_test_registry=selected_test_registry,
                ): index
                for index, item in enumerate(prepared)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    completed[index] = future.result()
                except Exception as error:
                    failures[index] = _concurrent_failure(
                        prepared[index],
                        error,
                        suite=suite,
                        output_root=root,
                    )

    ordered_runs = [
        completed[index] if completed[index] is not None else failures[index]
        for index in range(len(prepared))
    ]
    summary = {
        "execution": "dynamic_concurrent",
        "dispatch_mode": "auto",
        "source_commit": source_commit,
        "suite_identity": suite_identity,
        "submitted_runs": len(prepared),
        "runs": ordered_runs,
        "completed_runs": [row for row in completed if row is not None],
        "failed_runs": [row for row in failures if row is not None],
    }
    suite_arms: list[dict[str, Any]] = []
    for index, optional_row in enumerate(ordered_runs):
        if optional_row is None:
            raise AssertionError("concurrent arm completed without a result")
        suite_arms.append(
            {
                **{
                    key: optional_row.get(key)
                    for key in (
                        "case",
                        "controller",
                        "classification",
                        "run_dir",
                        "run_identity",
                        "test_id",
                        "failure_evidence_present",
                    )
                    if key in optional_row
                },
                "plan_identity": plan_identities[index],
            }
        )
    suite_evidence = {
        "suite_evidence_schema": "h3c_concurrent_baseline_suite",
        "schema_version": 1,
        "suite": suite,
        "suite_identity": suite_identity,
        "dispatch_mode": "auto",
        "source_commit": source_commit,
        "plan_identities": plan_identities,
        "mpc_freeze_identity": mpc_freeze_identity,
        "arms": suite_arms,
    }
    pending_suite_evidence = suite_evidence_dir / f".{suite_identity}.pending"
    with pending_suite_evidence.open("x", encoding="utf-8", newline="\n") as file:
        file.write(_canonical(suite_evidence) + "\n")
        file.flush()
        os.fsync(file.fileno())
    pending_suite_evidence.replace(suite_evidence_path)
    suite_claim_path.unlink()
    summary["suite_evidence"] = str(suite_evidence_path)
    suite_verification = verify_concurrent_suite_evidence(suite_evidence_path)
    summary["suite_evidence_valid"] = suite_verification["valid"]
    if suite_verification["valid"] is not True:
        summary["suite_evidence_errors"] = suite_verification["errors"]
        raise ConcurrentBaselineExecutionError(summary)
    if summary["failed_runs"]:
        raise ConcurrentBaselineExecutionError(summary)
    return summary


def execute_formal_suite() -> dict[str, Any]:
    return execute_baseline_plans(formal_evaluation_plans(), suite="formal")


def execute_mpc_formal_suite() -> dict[str, Any]:
    return execute_baseline_plans_concurrently(
        mpc_formal_evaluation_plans(),
        suite="mpc-formal",
    )
