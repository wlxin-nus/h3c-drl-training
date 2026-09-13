"""Fresh concurrent closed-loop validation for offline-refit MPC candidates."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from h3c.experiments.profiles import load_profile, repository_root
from h3c.runtime.clients import BoptestHttpClient
from h3c.runtime.comfort import ComfortModel, step_reward
from h3c.runtime.source_identity import committed_source_identity
from h3c_baselines.configuration import load_hierarchical_mpc_config
from h3c_baselines.mpc.optimizer import PMV_LIMIT, comfort_target_met
from h3c_baselines.mpc.refit import verify_refit_workspace
from h3c_baselines.mpc.training import (
    STEPS_PER_WEEK,
    _collect_episode,
    _Lane,
)
from h3c_baselines.mpc.vector_arx import FittedArxModel, expected_model_identity
from h3c_baselines.outputs.integrity import secret_occurrences

STEP_SECONDS = 900
VALIDATION_STEPS = STEPS_PER_WEEK - 4
WARMUP_SECONDS = 7 * 86400


class ValidationPhysicalClient(Protocol):
    test_id: str | None

    def set_lifecycle_sink(self, sink: Callable[[Mapping[str, Any]], None]) -> None: ...

    def select_testcase(self, testcase: str) -> str: ...

    def initialize_selected(
        self, start_time_seconds: int, warmup_period_seconds: int
    ) -> dict[str, Any]: ...

    def forecast(
        self, points: Sequence[str], horizon_seconds: int, interval_seconds: int
    ) -> dict[str, list[float | None]]: ...

    def advance(self, controls: Mapping[str, float]) -> dict[str, Any]: ...

    def stop(self) -> None: ...


PhysicalFactory = Callable[[str], ValidationPhysicalClient]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def _write_atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise ValueError(f"terminal evidence already exists: {path.name}")
    pending = path.with_name(f".{path.name}.pending")
    with pending.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(pending, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, sort_keys=True, allow_nan=False)
        file.write("\n")
        file.flush()


def _json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _path_present(path: Path) -> bool:
    return os.path.lexists(path)


def _artifact_hashes(arm_dir: Path) -> dict[str, str]:
    relative_paths = (
        "execution_owner.json",
        "resolved_arm.json",
        "step_diagnostics.jsonl",
        "lifecycle.jsonl",
        "episodes/validation-000-lane-0/manifest.json",
        "episodes/validation-000-lane-0/trajectory.npz",
    )
    return {name: _sha256(arm_dir / name) for name in relative_paths}


def _artifact_hashes_valid(arm_dir: Path, value: Any) -> bool:
    try:
        return isinstance(value, Mapping) and dict(value) == _artifact_hashes(arm_dir)
    except OSError:
        return False


@contextmanager
def _validation_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("another fresh MPC validation owns the suite lock") from error
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def _resolve_refit_workspace(workspace: Path) -> tuple[Path, dict[str, Any]]:
    root = repository_root().resolve()
    target = workspace if workspace.is_absolute() else root / workspace
    target = target.resolve()
    try:
        target.relative_to((root / "outputs" / "baselines" / "mpc" / "refit").resolve())
    except ValueError as error:
        raise ValueError("MPC validation requires a registered refit workspace") from error
    verification = verify_refit_workspace(target)
    if verification.get("valid") is not True:
        raise ValueError("MPC validation refit preflight failed")
    return target, verification


def _candidate_path(refit_workspace: Path, case: str) -> Path:
    return refit_workspace / case / "candidate_model"


def _finite_model(model: FittedArxModel) -> bool:
    arrays = (
        model.intercept,
        model.coefficients,
        model.scaling.feature_mean,
        model.scaling.feature_scale,
        model.scaling.output_mean,
        model.scaling.output_scale,
    )
    return bool(
        all(np.all(np.isfinite(value)) for value in arrays)
        and np.isfinite(model.ridge_alpha)
        and np.isfinite(model.pmv_robust_margin)
        and model.identity == expected_model_identity(model)
    )


def _refit_case_gate(
    refit_workspace: Path,
    refit_verification: Mapping[str, Any],
    case: str,
) -> dict[str, Any]:
    candidate = _candidate_path(refit_workspace, case)
    model = FittedArxModel.load(candidate / "model_coefficients.npz")
    report = _read_json(candidate / "candidate_report.json")
    case_rows = refit_verification.get("cases")
    verified_rows = (
        {str(row.get("case")): row for row in case_rows if isinstance(row, Mapping)}
        if isinstance(case_rows, list)
        else {}
    )
    verified = verified_rows.get(case, {})
    verified_checks = verified.get("checks")
    verified_checks = verified_checks if isinstance(verified_checks, Mapping) else {}
    quality = report.get("prediction_quality")
    quality = quality if isinstance(quality, Mapping) else {}
    checks = {
        "refit_candidate_verified": verified.get("valid") is True,
        "recomputed_persistence_gate": verified_checks.get("source_persistence_gate") is True,
        "refit_model_identity": verified.get("model_identity") == model.identity,
        "candidate_report_identity": report.get("case") == case
        and report.get("model_identity") == model.identity
        and report.get("eligible") is True,
        "finite_model": _finite_model(model),
        "source_prediction_finite": quality.get("finite") is True,
        "source_beats_persistence": quality.get("beats_persistence") is True,
        "fresh_validation_pending": report.get("physical_validation") == "pending_fresh_validation",
    }
    return {
        "case": case,
        "model_identity": model.identity,
        "prediction_quality": dict(quality),
        "checks": checks,
        "valid": all(checks.values()),
    }


def resolved_validation_plan(refit_workspace: Path) -> dict[str, Any]:
    source, verification = _resolve_refit_workspace(refit_workspace)
    root = repository_root().resolve()
    target = root / "models" / "mpc"
    if _path_present(target):
        raise ValueError("registered MPC model target already exists")
    config = load_hierarchical_mpc_config()
    case_order = list(config["case_order"])
    if len(case_order) != 3 or len(set(case_order)) != 3:
        raise ValueError("fresh MPC validation requires exactly three unique registered cases")
    gates = [_refit_case_gate(source, verification, case) for case in case_order]
    if not all(value["valid"] for value in gates):
        raise ValueError("MPC validation candidate preflight failed")
    value: dict[str, Any] = {
        "execution": False,
        "schema": "h3c_hierarchical_mpc_fresh_validation_plan",
        "schema_version": 1,
        "refit_workspace": source.relative_to(root).as_posix(),
        "refit_verification_identity": _identity(verification),
        "case_order": case_order,
        "dispatch_mode": "auto",
        "natural_case_futures": len(case_order),
        "client_worker_limit": None,
        "fresh_test_identity_per_case": True,
        "training_week_days_before_evaluation": 7,
        "warmup_days": 7,
        "step_seconds": STEP_SECONDS,
        "closed_loop_steps": VALIDATION_STEPS,
        "physical_gate": {
            "fallback_count": 0,
        },
        "comfort_target": {"occupied_peak_absolute_pmv_max": PMV_LIMIT},
        "candidate_gates": gates,
        "promotion": "all_three_or_none_transactional_directory_replace",
        "model_api_calls": 0,
    }
    value["plan_identity"] = _identity(value)
    return value


def _install_lifecycle_sink(
    client: ValidationPhysicalClient,
    sink: Callable[[Mapping[str, Any]], None],
) -> None:
    setter = getattr(client, "set_lifecycle_sink", None)
    if not callable(setter):
        raise ValueError("dynamic BOPTEST lifecycle client is required for fresh MPC validation")
    cast(Callable[[Callable[[Mapping[str, Any]], None]], None], setter)(sink)


def _run_closed_loop(
    *,
    case: str,
    client: ValidationPhysicalClient,
    test_id: str,
    refit_workspace: Path,
    diagnostics_path: Path,
) -> None:
    profile = load_profile(case)
    candidate = _candidate_path(refit_workspace, case)
    model = FittedArxModel.load(candidate / "model_coefficients.npz")
    lane = _Lane(0, client, test_id)

    def record(row: Mapping[str, Any]) -> None:
        _append_jsonl(
            diagnostics_path,
            {"case": case, **dict(row)},
        )

    result = _collect_episode(
        lane=lane,
        profile=profile,
        role="validation",
        episode=0,
        excitation_config=load_hierarchical_mpc_config()["excitation"],
        output_dir=diagnostics_path.parent,
        model=model,
        diagnostic_sink=record,
        forecast_phase="mpc_fresh_validation",
    )
    if (
        lane.initialize_count != 1
        or result.test_id != test_id
        or result.role != "validation"
        or result.episode != 0
        or len(result.times) != VALIDATION_STEPS + 1
    ):
        raise ValueError("fresh MPC validation episode lifecycle is invalid")


def _numeric_map(value: Any, keys: set[str]) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == keys
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value.values()
        )
    )


def _setpoints_in_bounds(value: Any, zones: tuple[str, ...]) -> bool:
    return _numeric_map(value, set(zones)) and all(
        20.0 <= float(cast(Mapping[str, Any], value)[zone]) <= 30.0 for zone in zones
    )


def _episode_artifacts_valid(
    *,
    case: str,
    arm_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    test_id: str,
    model_identity: str,
    reward: float,
    fallback_count: int,
    peak_pmv: float,
) -> bool:
    try:
        profile = load_profile(case)
        zones = tuple(profile["zones"])
        zone_count = len(zones)
        episode = arm_dir / "episodes" / "validation-000-lane-0"
        manifest = _read_json(episode / "manifest.json")
        with np.load(episode / "trajectory.npz", allow_pickle=False) as source:
            times = np.asarray(source["times"], dtype=np.int64)
            outputs = np.asarray(source["outputs"], dtype=np.float64)
            controls = np.asarray(source["controls"], dtype=np.float64)
            disturbances = np.asarray(source["disturbances"], dtype=np.float64)
        start = (int(profile["evaluation_start_day"]) - 7) * 86400
        if not (
            times.shape == (VALIDATION_STEPS + 1,)
            and np.array_equal(
                times,
                np.arange(VALIDATION_STEPS + 1, dtype=np.int64) * STEP_SECONDS + start,
            )
            and outputs.shape == (VALIDATION_STEPS + 1, zone_count + 1)
            and controls.shape == (VALIDATION_STEPS + 1, zone_count)
            and disturbances.shape == (VALIDATION_STEPS + 1, zone_count + 4)
            and all(np.all(np.isfinite(value)) for value in (outputs, controls, disturbances))
            and len(rows) == VALIDATION_STEPS
            and np.array_equal(controls[-1], controls[-2])
            and np.array_equal(disturbances[-1], disturbances[-2])
        ):
            return False
        for step, row in enumerate(rows):
            action_temperatures = row.get("action_zone_temperature_c")
            outcome_temperatures = row.get("outcome_zone_temperature_c")
            occupancy = row.get("occupancy")
            setpoints = row.get("setpoints_c")
            if not all(
                _numeric_map(value, set(zones))
                for value in (
                    action_temperatures,
                    outcome_temperatures,
                    occupancy,
                    setpoints,
                )
            ):
                return False
            assert isinstance(action_temperatures, Mapping)
            assert isinstance(outcome_temperatures, Mapping)
            assert isinstance(occupancy, Mapping)
            assert isinstance(setpoints, Mapping)
            action_time = start + step * STEP_SECONDS
            day_fraction = (action_time % 86400) / 86400.0
            if not (
                np.allclose(
                    outputs[step, :zone_count],
                    [float(action_temperatures[zone]) for zone in zones],
                    rtol=0.0,
                    atol=1e-12,
                )
                and np.allclose(
                    outputs[step + 1, :zone_count],
                    [float(outcome_temperatures[zone]) for zone in zones],
                    rtol=0.0,
                    atol=1e-12,
                )
                and math.isclose(
                    float(outputs[step + 1, -1]),
                    float(row["outcome_site_power_w"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and np.allclose(
                    controls[step],
                    [float(setpoints[zone]) for zone in zones],
                    rtol=0.0,
                    atol=1e-12,
                )
                and np.allclose(
                    disturbances[step, 2 : 2 + zone_count],
                    [float(occupancy[zone]) for zone in zones],
                    rtol=0.0,
                    atol=1e-12,
                )
                and math.isclose(
                    float(disturbances[step, 2 + zone_count]),
                    math.sin(2.0 * math.pi * day_fraction),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(disturbances[step, 3 + zone_count]),
                    math.cos(2.0 * math.pi * day_fraction),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                return False
        complete_days = VALIDATION_STEPS // 96
        for day in range(complete_days):
            day_slice = disturbances[day * 96 : (day + 1) * 96, 0]
            recorded = {
                float(rows[step]["daily_outdoor_mean_c"])
                for step in range(day * 96, (day + 1) * 96)
            }
            if len(recorded) != 1 or not math.isclose(
                recorded.pop(),
                float(np.mean(day_slice)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False
        return bool(
            manifest.get("role") == "validation"
            and manifest.get("episode") == 0
            and manifest.get("lane") == 0
            and manifest.get("test_id") == test_id
            and manifest.get("model_identity") == model_identity
            and manifest.get("start_time_seconds") == start
            and manifest.get("warmup_period_seconds") == WARMUP_SECONDS
            and manifest.get("steps") == VALIDATION_STEPS
            and manifest.get("fallback_count") == fallback_count
            and manifest.get("recovery_step_count") == 0
            and manifest.get("forecast_phase") == "mpc_fresh_validation"
            and math.isclose(float(manifest.get("reward", math.nan)), reward, abs_tol=1e-12)
            and math.isclose(
                float(manifest.get("peak_occupied_absolute_pmv", math.nan)),
                peak_pmv,
                abs_tol=1e-12,
            )
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _recompute_arm(
    *,
    case: str,
    arm_dir: Path,
    model_identity: str,
    expected_test_id: str,
) -> dict[str, Any]:
    profile = load_profile(case)
    zones = tuple(profile["zones"])
    zone_set = set(zones)
    configuration = load_hierarchical_mpc_config()
    occupied_bounds = tuple(
        float(value) for value in configuration["excitation"]["occupied_bounds_c"]
    )
    unoccupied_bounds = tuple(
        float(value) for value in configuration["excitation"]["unoccupied_bounds_c"]
    )
    start_time = (int(profile["evaluation_start_day"]) - 7) * 86400
    rows = _json_rows(arm_dir / "step_diagnostics.jsonl")
    lifecycle = _json_rows(arm_dir / "lifecycle.jsonl")
    previous = {zone: float(profile["protocol"]["initial_setpoint_c"]) for zone in zones}
    reward_total = 0.0
    peak_pmv = 0.0
    fallback_count = 0
    row_contract = len(rows) == VALIDATION_STEPS
    comfort = ComfortModel(profile["comfort"])
    daily_means: dict[int, float] = {}
    for step, row in enumerate(rows):
        occupancy = row.get("occupancy")
        setpoints = row.get("setpoints_c")
        outcome_pmv = row.get("outcome_pmv")
        controller = row.get("controller_diagnostics")
        maps_valid = all(
            _numeric_map(value, zone_set)
            for value in (
                occupancy,
                setpoints,
                outcome_pmv,
                row.get("action_zone_temperature_c"),
                row.get("action_pmv"),
                row.get("outcome_zone_temperature_c"),
            )
        )
        scalars = (
            row.get("daily_outdoor_mean_c"),
            row.get("clothing_insulation"),
            row.get("electricity_price"),
            row.get("outcome_site_power_w"),
            row.get("step_cost"),
            row.get("step_reward"),
        )
        scalar_valid = all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in scalars
        )
        basic = (
            row.get("schema") == "h3c_hierarchical_mpc_validation_step"
            and row.get("case") == case
            and row.get("step") == step
            and row.get("test_id") == expected_test_id
            and row.get("model_identity") == model_identity
            and row.get("action_time_seconds") == start_time + step * STEP_SECONDS
            and row.get("outcome_time_seconds") == start_time + (step + 1) * STEP_SECONDS
            and maps_valid
            and scalar_valid
            and isinstance(controller, Mapping)
            and controller.get("status") in {"optimized", "fallback"}
            and isinstance(controller.get("method_degraded"), bool)
            and (
                (controller.get("status") == "fallback")
                is (controller.get("method_degraded") is True)
            )
            and _setpoints_in_bounds(setpoints, zones)
        )
        if not basic:
            row_contract = False
            continue
        assert isinstance(occupancy, Mapping)
        assert isinstance(setpoints, Mapping)
        assert isinstance(outcome_pmv, Mapping)
        assert isinstance(controller, Mapping)
        action_pmv = row["action_pmv"]
        action_temperatures = row["action_zone_temperature_c"]
        outcome_temperatures = row["outcome_zone_temperature_c"]
        assert isinstance(action_pmv, Mapping)
        assert isinstance(action_temperatures, Mapping)
        assert isinstance(outcome_temperatures, Mapping)
        action_time = start_time + step * STEP_SECONDS
        daily_mean = float(row["daily_outdoor_mean_c"])
        day = action_time // 86400
        if day in daily_means and not math.isclose(
            daily_means[day], daily_mean, rel_tol=0.0, abs_tol=1e-12
        ):
            row_contract = False
        daily_means[day] = daily_mean
        comfort.update_clothing(action_time, daily_mean)
        expected_action_pmv = {
            zone: comfort.pmv(float(action_temperatures[zone])) for zone in zones
        }
        expected_outcome_pmv = {
            zone: comfort.pmv(float(outcome_temperatures[zone])) for zone in zones
        }
        support_valid = True
        for zone in zones:
            count = float(occupancy[zone])
            if count < 0.0:
                support_valid = False
                continue
            if controller["status"] == "optimized":
                lower, upper = occupied_bounds if count > 0.0 else unoccupied_bounds
                if not lower <= float(setpoints[zone]) <= upper:
                    support_valid = False
        if not (
            support_valid
            and math.isclose(
                float(row["clothing_insulation"]),
                comfort.clothing_insulation,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and all(
                math.isclose(
                    float(action_pmv[zone]),
                    expected_action_pmv[zone],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(outcome_pmv[zone]),
                    expected_outcome_pmv[zone],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for zone in zones
            )
        ):
            row_contract = False
        expected_cost = (
            float(row["outcome_site_power_w"]) * 0.25 / 1000.0 * float(row["electricity_price"])
        )
        expected_reward = step_reward(
            cost=expected_cost,
            pmv=[expected_outcome_pmv[zone] for zone in zones],
            occupancy=[float(occupancy[zone]) for zone in zones],
            setpoints_c=[float(setpoints[zone]) for zone in zones],
            previous_setpoints_c=[previous[zone] for zone in zones],
            objective=profile["objective"],
        )
        if not (
            math.isclose(float(row["step_cost"]), expected_cost, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(row["step_reward"]), expected_reward, rel_tol=0.0, abs_tol=1e-12)
        ):
            row_contract = False
        reward_total += expected_reward
        fallback_count += int(controller["method_degraded"] is True)
        occupied = [
            abs(expected_outcome_pmv[zone]) for zone in zones if float(occupancy[zone]) > 0.0
        ]
        peak_pmv = max([peak_pmv, *occupied])
        previous = {zone: float(setpoints[zone]) for zone in zones}
    sequences = [row.get("sequence") for row in lifecycle]
    events = [row.get("event") for row in lifecycle]
    selected = [index for index, event in enumerate(events) if event == "selected"]
    configured = [index for index, event in enumerate(events) if event == "configured"]
    initialized = [index for index, event in enumerate(events) if event == "initialized"]
    stopped = [index for index, event in enumerate(events) if event == "stopped"]
    running = [
        index
        for index, row in enumerate(lifecycle)
        if row.get("event") == "status_changed" and row.get("status") == "Running"
    ]
    lifecycle_contract = (
        bool(lifecycle)
        and sequences == list(range(len(lifecycle)))
        and len(selected) == len(configured) == len(initialized) == len(stopped) == 1
        and selected[0] < configured[0] < initialized[0] < stopped[0]
        and any(selected[0] < index < configured[0] for index in running)
        and any(configured[0] < index < initialized[0] for index in running)
        and all(row.get("test_id") == expected_test_id for row in lifecycle)
        and all(row.get("dispatch_mode") == "auto" for row in lifecycle)
        and all(row.get("case") == case for row in lifecycle)
        and all(row.get("testcase") == profile["testcase"] for row in lifecycle)
        and all(row.get("phase") == "physical_dispatch" for row in lifecycle)
        and all(
            row.get("event")
            in {"selected", "status_changed", "configured", "initialized", "stopped"}
            for row in lifecycle
        )
        and all(
            row.get("status") in {"Running", "Queued"}
            for row in lifecycle
            if row.get("event") == "status_changed"
        )
    )
    checks = {
        "diagnostic_rows": len(rows) == VALIDATION_STEPS,
        "timeline_reward_and_identity": row_contract,
        "lifecycle": lifecycle_contract,
        "finite_reward": math.isfinite(reward_total),
        "finite_peak_pmv": math.isfinite(peak_pmv),
        "episode_artifacts": _episode_artifacts_valid(
            case=case,
            arm_dir=arm_dir,
            rows=rows,
            test_id=expected_test_id,
            model_identity=model_identity,
            reward=reward_total,
            fallback_count=fallback_count,
            peak_pmv=peak_pmv,
        ),
    }
    return {
        "case": case,
        "test_id": expected_test_id,
        "model_identity": model_identity,
        "steps": len(rows),
        "reward": reward_total,
        "fallback_count": fallback_count,
        "occupied_peak_absolute_pmv": peak_pmv,
        "comfort_target_met": comfort_target_met(peak_pmv),
        "checks": checks,
        "evidence_valid": all(checks.values()),
        "eligible": all(checks.values()) and fallback_count == 0,
    }


def _execute_case_validation(
    *,
    case: str,
    endpoint: str,
    refit_workspace: Path,
    run_dir: Path,
    physical_factory: PhysicalFactory,
    source_commit: str,
) -> dict[str, Any]:
    arm_dir = run_dir / case
    arm_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        arm_dir / "execution_owner.json",
        {
            "schema": "h3c_hierarchical_mpc_validation_execution_owner",
            "schema_version": 1,
            "case": case,
            "validation_source_commit": source_commit,
            "pid": os.getpid(),
            "lock_scope": "independent_case_arm",
        },
    )
    with _validation_lock(arm_dir / ".execution.lock"):
        return _execute_case_validation_owned(
            case=case,
            endpoint=endpoint,
            refit_workspace=refit_workspace,
            arm_dir=arm_dir,
            physical_factory=physical_factory,
            source_commit=source_commit,
        )


def _execute_case_validation_owned(
    *,
    case: str,
    endpoint: str,
    refit_workspace: Path,
    arm_dir: Path,
    physical_factory: PhysicalFactory,
    source_commit: str,
) -> dict[str, Any]:
    candidate = _candidate_path(refit_workspace, case)
    model = FittedArxModel.load(candidate / "model_coefficients.npz")
    profile = load_profile(case)
    _write_json(
        arm_dir / "resolved_arm.json",
        {
            "schema": "h3c_hierarchical_mpc_validation_arm",
            "schema_version": 1,
            "case": case,
            "testcase": profile["testcase"],
            "model_identity": model.identity,
            "validation_source_commit": source_commit,
            "refit_workspace": refit_workspace.relative_to(repository_root()).as_posix(),
            "warmup_period_seconds": WARMUP_SECONDS,
            "steps": VALIDATION_STEPS,
            "dispatch_mode": "auto",
        },
    )
    client = physical_factory(endpoint)
    lifecycle_path = arm_dir / "lifecycle.jsonl"
    sequence = 0

    def lifecycle_sink(raw: Mapping[str, Any]) -> None:
        nonlocal sequence
        row = {"sequence": sequence, "case": case, **dict(raw)}
        _append_jsonl(lifecycle_path, row)
        sequence += 1

    primary_error: Exception | None = None
    stop_error: Exception | None = None
    test_id = ""
    owned_test_id: str | None = None
    try:
        _install_lifecycle_sink(client, lifecycle_sink)
        selected = client.select_testcase(profile["testcase"])
        if not selected or client.test_id != selected:
            raise ValueError("fresh MPC validation select identity is invalid")
        test_id = selected
        _run_closed_loop(
            case=case,
            client=client,
            test_id=test_id,
            refit_workspace=refit_workspace,
            diagnostics_path=arm_dir / "step_diagnostics.jsonl",
        )
    except Exception as error:
        primary_error = error
    finally:
        if client.test_id is not None:
            owned_test_id = client.test_id
            try:
                client.stop()
            except Exception as error:
                stop_error = error
    if primary_error is not None or stop_error is not None:
        failure = {
            "schema": "h3c_hierarchical_mpc_validation_arm_failure",
            "schema_version": 1,
            "case": case,
            "test_id": test_id or owned_test_id,
            "model_identity": model.identity,
            "error_type": type(primary_error).__name__ if primary_error else None,
            "error": str(primary_error) if primary_error else None,
            "stop_error_type": type(stop_error).__name__ if stop_error else None,
            "stop_error": str(stop_error) if stop_error else None,
            "secret_exposure_count": secret_occurrences(arm_dir),
        }
        _write_atomic_json(arm_dir / "failure.json", failure)
        return {"case": case, "status": "failed", **failure}
    result = _recompute_arm(
        case=case,
        arm_dir=arm_dir,
        model_identity=model.identity,
        expected_test_id=test_id,
    )
    completion = {
        "schema": "h3c_hierarchical_mpc_validation_arm_completion",
        "schema_version": 1,
        **result,
        "secret_exposure_count": secret_occurrences(arm_dir),
        "artifact_sha256": _artifact_hashes(arm_dir),
    }
    if completion["secret_exposure_count"] != 0:
        raise ValueError(f"secret exposure detected in {case} MPC validation")
    _write_atomic_json(arm_dir / "completion.json", completion)
    return {
        "status": "completed",
        "arm_completion_sha256": _sha256(arm_dir / "completion.json"),
        **completion,
    }


def _suite_evidence_checks(
    run_dir: Path,
    evidence: Mapping[str, Any],
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    root = repository_root().resolve()
    refit_workspace = (root / str(evidence.get("refit_workspace", ""))).resolve()
    try:
        refit_workspace.relative_to((root / "outputs" / "baselines" / "mpc" / "refit").resolve())
    except ValueError:
        return {"refit_path": False}, []
    refit_verification = verify_refit_workspace(refit_workspace)
    config = load_hierarchical_mpc_config()
    cases = evidence.get("cases")
    by_case = (
        {str(row.get("case")): row for row in cases if isinstance(row, Mapping)}
        if isinstance(cases, list)
        else {}
    )
    recomputed: list[dict[str, Any]] = []
    arm_terminal_checks: dict[str, bool] = {}
    for case in config["case_order"]:
        row = by_case.get(case)
        if row is None or row.get("status") != "completed":
            arm_terminal_checks[case] = False
            continue
        arm_dir = run_dir / case
        arm_completion = _read_json(arm_dir / "completion.json")
        resolved_arm = _read_json(arm_dir / "resolved_arm.json")
        execution_owner = _read_json(arm_dir / "execution_owner.json")
        expected_result = {
            "status": "completed",
            "arm_completion_sha256": _sha256(arm_dir / "completion.json"),
            **arm_completion,
        }
        terminal_valid = (
            dict(row) == expected_result
            and arm_completion.get("schema") == "h3c_hierarchical_mpc_validation_arm_completion"
            and arm_completion.get("case") == case
            and arm_completion.get("secret_exposure_count") == 0
            and _artifact_hashes_valid(arm_dir, arm_completion.get("artifact_sha256"))
            and resolved_arm
            == {
                "schema": "h3c_hierarchical_mpc_validation_arm",
                "schema_version": 1,
                "case": case,
                "testcase": load_profile(case)["testcase"],
                "model_identity": row.get("model_identity"),
                "validation_source_commit": evidence.get("validation_source_commit"),
                "refit_workspace": evidence.get("refit_workspace"),
                "warmup_period_seconds": WARMUP_SECONDS,
                "steps": VALIDATION_STEPS,
                "dispatch_mode": "auto",
            }
            and execution_owner.get("schema") == "h3c_hierarchical_mpc_validation_execution_owner"
            and execution_owner.get("case") == case
            and execution_owner.get("validation_source_commit")
            == evidence.get("validation_source_commit")
            and execution_owner.get("lock_scope") == "independent_case_arm"
            and isinstance(execution_owner.get("pid"), int)
            and int(execution_owner["pid"]) > 0
        )
        arm_terminal_checks[case] = terminal_valid
        recomputed.append(
            _recompute_arm(
                case=case,
                arm_dir=arm_dir,
                model_identity=str(arm_completion.get("model_identity", "")),
                expected_test_id=str(arm_completion.get("test_id", "")),
            )
        )
    refit_gates = (
        [
            _refit_case_gate(refit_workspace, refit_verification, case)
            for case in config["case_order"]
        ]
        if refit_verification.get("valid") is True
        else []
    )
    test_ids = [row["test_id"] for row in recomputed]
    refit_by_case = {row["case"]: row for row in refit_gates}
    checks = {
        "schema": evidence.get("schema") == "h3c_hierarchical_mpc_validation_evidence",
        "validation_source_commit": isinstance(evidence.get("validation_source_commit"), str)
        and len(str(evidence["validation_source_commit"])) == 40,
        "case_order": evidence.get("case_order") == list(config["case_order"]),
        "refit_verified": refit_verification.get("valid") is True,
        "refit_verification_identity": evidence.get("refit_verification_identity")
        == _identity(refit_verification),
        "all_refit_gates": len(refit_gates) == len(config["case_order"])
        and all(row["valid"] for row in refit_gates)
        and all(
            refit_by_case[row["case"]]["model_identity"] == row["model_identity"]
            for row in recomputed
        ),
        "all_arms_recomputed": len(recomputed) == len(config["case_order"])
        and all(row["eligible"] for row in recomputed),
        "arm_terminal_artifacts": set(arm_terminal_checks) == set(config["case_order"])
        and all(arm_terminal_checks.values()),
        "recorded_arm_results": len(recomputed) == len(by_case)
        and all(
            by_case[row["case"]].get("eligible") is row["eligible"]
            and by_case[row["case"]].get("reward") == row["reward"]
            and by_case[row["case"]].get("fallback_count") == row["fallback_count"]
            and by_case[row["case"]].get("occupied_peak_absolute_pmv")
            == row["occupied_peak_absolute_pmv"]
            for row in recomputed
        ),
        "fresh_unique_test_ids": len(test_ids) == len(set(test_ids)) == len(config["case_order"])
        and all(test_ids),
        "all_futures_awaited": evidence.get("submitted_case_count") == len(config["case_order"])
        and evidence.get("awaited_case_count") == len(config["case_order"])
        and evidence.get("future_cancel_count") == 0,
        "no_model_calls": evidence.get("model_api_calls") == 0,
        "secret_scan": evidence.get("secret_exposure_count") == 0
        and secret_occurrences(run_dir) == 0,
    }
    return checks, recomputed


def execute_fresh_validation(
    refit_workspace: Path,
    *,
    endpoint: str,
    physical_factory: PhysicalFactory | None = None,
) -> dict[str, Any]:
    plan = resolved_validation_plan(refit_workspace)
    source_commit = committed_source_identity()
    root = repository_root().resolve()
    source = (root / str(plan["refit_workspace"])).resolve()
    output_root = root / "outputs" / "baselines" / "mpc" / "validation"
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + source_commit[:8]
    run_dir = output_root / run_id
    case_order = list(load_hierarchical_mpc_config()["case_order"])
    factory = physical_factory or cast(PhysicalFactory, BoptestHttpClient)
    results: dict[str, dict[str, Any]] = {}
    completion_order: list[str] = []
    with (
        _validation_lock(output_root / ".validation.lock"),
        ThreadPoolExecutor() as executor,
    ):
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(
            run_dir / "resolved_plan.json",
            {**plan, "execution": True, "validation_source_commit": source_commit},
        )
        futures: dict[Future[dict[str, Any]], str] = {
            executor.submit(
                _execute_case_validation,
                case=case,
                endpoint=endpoint,
                refit_workspace=source,
                run_dir=run_dir,
                physical_factory=factory,
                source_commit=source_commit,
            ): case
            for case in case_order
        }
        for future in as_completed(futures):
            case = futures[future]
            completion_order.append(case)
            try:
                results[case] = future.result()
            except Exception as error:
                results[case] = {
                    "case": case,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
    ordered = [results[case] for case in case_order]
    refit_verification = verify_refit_workspace(source)
    evidence = {
        "schema": "h3c_hierarchical_mpc_validation_evidence",
        "schema_version": 1,
        "validation_source_commit": source_commit,
        "refit_workspace": source.relative_to(root).as_posix(),
        "refit_verification_identity": _identity(refit_verification),
        "case_order": case_order,
        "submitted_case_count": len(case_order),
        "awaited_case_count": len(results),
        "future_cancel_count": 0,
        "future_completion_order": completion_order,
        "cases": ordered,
        "model_api_calls": 0,
        "secret_exposure_count": secret_occurrences(run_dir),
    }
    _write_json(run_dir / "suite_evidence.json", evidence)
    checks, recomputed = _suite_evidence_checks(run_dir, evidence)
    if not all(checks.values()):
        failure = {
            "schema": "h3c_hierarchical_mpc_validation_failure",
            "schema_version": 1,
            "validation_source_commit": source_commit,
            "checks": checks,
            "cases": recomputed,
            "candidate_promotion": False,
            "secret_exposure_count": secret_occurrences(run_dir),
        }
        _write_atomic_json(run_dir / "failure.json", failure)
        raise ValueError("fresh MPC validation suite failed; no model was promoted")
    completion = {
        "schema": "h3c_hierarchical_mpc_validation_completion",
        "schema_version": 1,
        "validation_source_commit": source_commit,
        "plan_identity": plan["plan_identity"],
        "refit_workspace": source.relative_to(root).as_posix(),
        "refit_verification_identity": _identity(refit_verification),
        "case_order": case_order,
        "cases": recomputed,
        "validation_evidence_sha256": _sha256(run_dir / "suite_evidence.json"),
        "candidate_promotion": False,
        "secret_exposure_count": secret_occurrences(run_dir),
    }
    _write_atomic_json(run_dir / "completion.json", completion)
    return {**completion, "run_dir": str(run_dir)}


def verify_validation_workspace(workspace: Path) -> dict[str, Any]:
    try:
        root = repository_root().resolve()
        target = workspace if workspace.is_absolute() else root / workspace
        target = target.resolve()
        target.relative_to((root / "outputs" / "baselines" / "mpc" / "validation").resolve())
        if (target / "failure.json").exists():
            raise ValueError("MPC validation workspace contains failure evidence")
        completion = _read_json(target / "completion.json")
        plan = _read_json(target / "resolved_plan.json")
        evidence = _read_json(target / "suite_evidence.json")
        checks, recomputed = _suite_evidence_checks(target, evidence)
        config = load_hierarchical_mpc_config()
        plan_payload = {
            key: value
            for key, value in plan.items()
            if key not in {"validation_source_commit", "plan_identity"}
        }
        plan_payload["execution"] = False
        completion_cases = completion.get("cases")
        recorded = (
            {str(row.get("case")): row for row in completion_cases if isinstance(row, Mapping)}
            if isinstance(completion_cases, list)
            else {}
        )
        checks.update(
            {
                "completion_schema": completion.get("schema")
                == "h3c_hierarchical_mpc_validation_completion",
                "completion_case_order": completion.get("case_order") == list(config["case_order"]),
                "plan_identity": plan.get("schema") == "h3c_hierarchical_mpc_fresh_validation_plan"
                and plan.get("execution") is True
                and _identity(plan_payload) == plan.get("plan_identity")
                and completion.get("plan_identity") == plan.get("plan_identity")
                and completion.get("validation_source_commit")
                == plan.get("validation_source_commit")
                and isinstance(completion.get("validation_source_commit"), str)
                and target.name.endswith(f"-{str(completion.get('validation_source_commit'))[:8]}"),
                "refit_identity_chain": plan.get("refit_workspace")
                == evidence.get("refit_workspace")
                == completion.get("refit_workspace")
                and plan.get("refit_verification_identity")
                == evidence.get("refit_verification_identity")
                == completion.get("refit_verification_identity"),
                "evidence_identity": completion.get("validation_evidence_sha256")
                == _sha256(target / "suite_evidence.json"),
                "completion_results": set(recorded) == set(config["case_order"])
                and all(recorded[row["case"]] == row for row in recomputed),
                "not_promoted_by_validation": completion.get("candidate_promotion") is False,
                "completion_secret_scan": completion.get("secret_exposure_count") == 0,
            }
        )
        return {
            "schema": "h3c_hierarchical_mpc_validation_verification",
            "schema_version": 1,
            "workspace": str(target),
            "checks": checks,
            "cases": recomputed,
            "valid": all(checks.values()),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "schema": "h3c_hierarchical_mpc_validation_verification",
            "schema_version": 1,
            "workspace": str(workspace),
            "error": f"{type(error).__name__}: {error}",
            "valid": False,
        }
