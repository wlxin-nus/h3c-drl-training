"""Pure-offline recovery of hierarchical MPC candidates from preserved episodes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

import numpy as np

from h3c.experiments.profiles import load_profile, repository_root
from h3c.runtime.comfort import COMFORT_BAND, ComfortModel
from h3c.runtime.source_identity import committed_source_identity
from h3c_baselines.configuration import load_hierarchical_mpc_config
from h3c_baselines.mpc.training import (
    EpisodeData,
    _arx_layout,
    _episode_dataset,
    _training_lock,
    _write_completion,
    _write_json,
    open_loop_prediction_quality,
)
from h3c_baselines.mpc.vector_arx import (
    FittedArxModel,
    expected_model_identity,
    fit_vector_arx,
    with_pmv_robust_margin,
)
from h3c_baselines.outputs.integrity import secret_occurrences

STEP_SECONDS = 900
ROBUST_QUANTILE = 0.95


@dataclass(frozen=True)
class _SourceEpisode:
    relative_path: str
    manifest: dict[str, Any]
    data: EpisodeData


@dataclass(frozen=True)
class _LaneIdentity:
    lane: int
    test_id: str
    episode_count: int
    frozen_manifest_cross_checked: bool


@dataclass(frozen=True)
class _EpisodeRoles:
    fit: tuple[_SourceEpisode, ...]
    adaptive_validation: tuple[_SourceEpisode, ...]
    calibration: _SourceEpisode
    holdout: tuple[_SourceEpisode, ...]
    excluded_fallback: tuple[_SourceEpisode, ...]
    unused_validation: tuple[_SourceEpisode, ...]
    lane_identities: tuple[_LaneIdentity, ...]

    def evidence(self) -> dict[str, list[str] | str]:
        return {
            "fit": [episode.relative_path for episode in self.fit],
            "adaptive_validation": [episode.relative_path for episode in self.adaptive_validation],
            "calibration": self.calibration.relative_path,
            "holdout": [episode.relative_path for episode in self.holdout],
            "excluded_fallback": [episode.relative_path for episode in self.excluded_fallback],
            "unused_validation": [episode.relative_path for episode in self.unused_validation],
        }


@dataclass(frozen=True)
class _RebuiltCandidate:
    model: FittedArxModel
    evidence: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve_source_run(source_run: Path) -> Path:
    root = repository_root().resolve()
    source = source_run if source_run.is_absolute() else root / source_run
    source = source.resolve()
    training_root = (root / "outputs" / "baselines" / "mpc" / "training").resolve()
    try:
        source.relative_to(training_root)
    except ValueError as error:
        raise ValueError("MPC refit source must be a registered training workspace") from error
    if not source.is_dir():
        raise ValueError("MPC refit source workspace does not exist")
    failure = source / "failure.json"
    if not failure.is_file() or (source / "completion.json").exists():
        raise ValueError("MPC refit requires one preserved failed training workspace")
    failure_value = _read_json(failure)
    plan_path = source / "resolved_plan.json"
    if not plan_path.is_file():
        raise ValueError("MPC refit source resolved plan is missing")
    plan_value = _read_json(plan_path)
    config = load_hierarchical_mpc_config()
    if failure_value.get("schema") != "h3c_hierarchical_mpc_training_failure":
        raise ValueError("MPC refit source failure schema is invalid")
    if (
        plan_value.get("schema") != "h3c_hierarchical_mpc_training_plan"
        or plan_value.get("execution") is not True
        or plan_value.get("source_commit") != failure_value.get("source_commit")
        or plan_value.get("case_order") != list(config["case_order"])
        or plan_value.get("fit_checkpoints") != [int(value) for value in config["fit_checkpoints"]]
        or int(plan_value.get("holdout_episodes", -1)) != int(config["holdout_episodes"])
        or int(plan_value.get("workers", -1)) != 4
        or int(plan_value.get("warmup_days_per_episode", -1)) != 7
        or int(plan_value.get("training_week_days_before_evaluation", -1)) != 7
        or int(failure_value.get("secret_exposure_count", -1)) != 0
    ):
        raise ValueError("MPC refit source plan/failure identity is inconsistent")
    return source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_file_manifest(source: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("MPC refit source must not contain symbolic links")
        if path.is_file():
            result.append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if not result:
        raise ValueError("MPC refit source workspace is empty")
    return result


def _load_episode(case: str, case_dir: Path, episode_dir: Path) -> _SourceEpisode:
    profile = load_profile(case)
    layout = _arx_layout(profile)
    manifest = _read_json(episode_dir / "manifest.json")
    role = str(manifest.get("role"))
    episode = int(manifest.get("episode", -1))
    lane = int(manifest.get("lane", -1))
    expected_name = f"{role}-{episode:03d}-lane-{lane}"
    if episode_dir.name != expected_name:
        raise ValueError(f"MPC source episode identity mismatch: {episode_dir.name}")
    if role not in {"fit", "holdout", "basic_reference", "validation"}:
        raise ValueError(f"MPC source episode role is invalid: {role}")
    registered_steps = 668 if role == "validation" else 672
    if (
        int(manifest.get("warmup_period_seconds", -1)) != 7 * 86400
        or int(manifest.get("start_time_seconds", -1))
        != (int(profile["evaluation_start_day"]) - 7) * 86400
        or int(manifest.get("steps", -1)) != registered_steps
    ):
        raise ValueError(f"MPC source episode protocol is invalid: {episode_dir.name}")
    trajectory_path = episode_dir / "trajectory.npz"
    with np.load(trajectory_path, allow_pickle=False) as source:
        times = np.asarray(source["times"], dtype=np.int64)
        outputs = np.asarray(source["outputs"], dtype=np.float64)
        controls = np.asarray(source["controls"], dtype=np.float64)
        disturbances = np.asarray(source["disturbances"], dtype=np.float64)
    rows = len(times)
    expected_steps = int(manifest.get("steps", -1))
    expected_start = (int(profile["evaluation_start_day"]) - 7) * 86400
    if (
        rows != expected_steps + 1
        or rows != registered_steps + 1
        or int(times[0]) != expected_start
        or int(times[-1]) != expected_start + registered_steps * STEP_SECONDS
        or outputs.shape != (rows, layout.output_dimension)
        or controls.shape != (rows, layout.control_dimension)
        or disturbances.shape != (rows, len(layout.disturbance_names))
        or np.any(np.diff(times) != STEP_SECONDS)
        or any(np.any(~np.isfinite(value)) for value in (outputs, controls, disturbances))
    ):
        raise ValueError(f"MPC source trajectory is invalid: {episode_dir.name}")
    test_id = str(manifest.get("test_id", ""))
    if not test_id:
        raise ValueError(f"MPC source test identity is missing: {episode_dir.name}")
    reward = float(manifest.get("reward", np.nan))
    peak_pmv = float(manifest.get("peak_occupied_absolute_pmv", np.nan))
    fallback_count = int(manifest.get("fallback_count", -1))
    recovery_step_count = int(manifest.get("recovery_step_count", -1))
    if (
        not np.isfinite(reward)
        or not np.isfinite(peak_pmv)
        or peak_pmv < 0.0
        or fallback_count < 0
        or recovery_step_count < 0
        or fallback_count > expected_steps
        or recovery_step_count > expected_steps
    ):
        raise ValueError(f"MPC source episode metrics are invalid: {episode_dir.name}")
    data = EpisodeData(
        role=role,
        episode=episode,
        lane=lane,
        test_id=test_id,
        times=times,
        outputs=outputs,
        controls=controls,
        disturbances=disturbances,
        reward=reward,
        peak_occupied_absolute_pmv=peak_pmv,
        fallback_count=fallback_count,
        recovery_step_count=recovery_step_count,
    )
    return _SourceEpisode(
        relative_path=episode_dir.relative_to(case_dir).as_posix(),
        manifest=manifest,
        data=data,
    )


def _case_roles(case: str, source: Path) -> _EpisodeRoles:
    config = load_hierarchical_mpc_config()
    source_plan = _read_json(source / "resolved_plan.json")
    worker_count = int(source_plan.get("workers", -1))
    if worker_count != 4:
        raise ValueError(f"MPC refit source must contain exactly four lanes for {case}")
    case_dir = source / case
    episode_root = case_dir / "episodes"
    if not episode_root.is_dir():
        raise ValueError(f"MPC source has no episode bank for {case}")
    episodes = [
        _load_episode(case, case_dir, path)
        for path in sorted(episode_root.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    ]
    identities = [(episode.data.role, episode.data.episode) for episode in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError(f"MPC source contains duplicate episode identity for {case}")
    fit_episode_ids = sorted(
        episode.data.episode for episode in episodes if episode.data.role == "fit"
    )
    if fit_episode_ids != list(range(len(fit_episode_ids))):
        raise ValueError(f"MPC source fit episode IDs are not contiguous for {case}")
    registered_fit_counts = {int(value) for value in config["fit_checkpoints"]}
    if len(fit_episode_ids) not in registered_fit_counts:
        raise ValueError(f"MPC source fit episode count is not registered for {case}")
    for episode in episodes:
        expected_lane = (
            episode.data.episode % worker_count if episode.data.role in {"fit", "holdout"} else 0
        )
        if episode.data.lane != expected_lane:
            raise ValueError(f"MPC source episode lane assignment is invalid for {case}")
    lane_test_ids: dict[int, set[str]] = {lane: set() for lane in range(worker_count)}
    lane_episode_counts = {lane: 0 for lane in range(worker_count)}
    for episode in episodes:
        lane = episode.data.lane
        if lane not in lane_test_ids:
            raise ValueError(f"MPC source episode has an invalid lane for {case}")
        lane_test_ids[lane].add(episode.data.test_id)
        lane_episode_counts[lane] += 1
    if any(len(values) != 1 for values in lane_test_ids.values()):
        raise ValueError(f"MPC source lane test identity is unstable for {case}")
    derived_test_ids = {lane: next(iter(values)) for lane, values in lane_test_ids.items()}
    if len(set(derived_test_ids.values())) != worker_count:
        raise ValueError(f"MPC source lane test identities are not unique for {case}")
    frozen_manifest_path = case_dir / "frozen_model" / "training_manifest.json"
    frozen_cross_checked = frozen_manifest_path.is_file()
    if frozen_cross_checked:
        frozen_manifest = _read_json(frozen_manifest_path)
        lifecycle = frozen_manifest.get("lane_lifecycle")
        rows = (
            {int(row.get("lane", -1)): row for row in lifecycle}
            if isinstance(lifecycle, list) and all(isinstance(row, dict) for row in lifecycle)
            else {}
        )
        expected_training_output = case_dir.relative_to(repository_root()).as_posix()
        if (
            frozen_manifest.get("schema") != "h3c_hierarchical_mpc_training_manifest"
            or frozen_manifest.get("case") != case
            or frozen_manifest.get("source_commit") != source_plan.get("source_commit")
            or frozen_manifest.get("training_output") != expected_training_output
            or set(rows) != set(range(worker_count))
            or any(
                row.get("test_id") != derived_test_ids[lane]
                or int(row.get("select_count", -1)) != 1
                or int(row.get("initialize_count", -1)) != lane_episode_counts[lane]
                or int(row.get("stop_count", -1)) != 1
                or int(row.get("warmup_days_per_initialize", -1)) != 7
                for lane, row in rows.items()
            )
        ):
            raise ValueError(f"MPC frozen lane lifecycle is inconsistent for {case}")
    lane_identities = tuple(
        _LaneIdentity(
            lane=lane,
            test_id=derived_test_ids[lane],
            episode_count=lane_episode_counts[lane],
            frozen_manifest_cross_checked=frozen_cross_checked,
        )
        for lane in range(worker_count)
    )
    fit = sorted(
        (episode for episode in episodes if episode.data.role == "fit"),
        key=lambda item: item.data.episode,
    )
    basic = [episode for episode in episodes if episode.data.role == "basic_reference"]
    holdout = sorted(
        (episode for episode in episodes if episode.data.role == "holdout"),
        key=lambda item: item.data.episode,
    )
    validations = {
        episode.data.episode: episode for episode in episodes if episode.data.role == "validation"
    }
    if (
        not fit
        or len(basic) != 1
        or len(holdout) != int(config["holdout_episodes"])
        or [episode.data.episode for episode in holdout]
        != list(range(int(config["holdout_episodes"])))
    ):
        raise ValueError(f"MPC source episode budget is invalid for {case}")
    expected_validation_ids = tuple(
        int(value) for value in config["fit_checkpoints"] if int(value) <= len(fit_episode_ids)
    )
    if tuple(sorted(validations)) != expected_validation_ids:
        raise ValueError(
            f"MPC source validation IDs do not exactly match eligible checkpoints for {case}"
        )
    checkpoint_root = case_dir / "checkpoints"
    expected_checkpoint_names = {
        f"checkpoint-{checkpoint:03d}.json" for checkpoint in expected_validation_ids
    }
    actual_checkpoint_names = (
        {path.name for path in checkpoint_root.iterdir() if path.is_file()}
        if checkpoint_root.is_dir()
        else set()
    )
    if actual_checkpoint_names != expected_checkpoint_names:
        raise ValueError(
            f"MPC source checkpoint reports do not exactly match fit coverage for {case}"
        )
    ordered_validations: list[_SourceEpisode] = []
    for checkpoint in expected_validation_ids:
        validation_episode = validations[checkpoint]
        checkpoint_path = case_dir / "checkpoints" / f"checkpoint-{checkpoint:03d}.json"
        report = _read_json(checkpoint_path)
        closed_loop = report.get("closed_loop_validation")
        if not isinstance(closed_loop, dict) or (
            int(report.get("checkpoint_fit_episodes", -1)) != checkpoint
            or int(closed_loop.get("fallback_count", -1)) != validation_episode.data.fallback_count
            or not np.isclose(
                float(closed_loop.get("peak_occupied_absolute_pmv", np.nan)),
                validation_episode.data.peak_occupied_absolute_pmv,
            )
            or not np.isclose(
                float(closed_loop.get("reward", np.nan)), validation_episode.data.reward
            )
        ):
            raise ValueError(f"MPC validation/checkpoint evidence mismatch for {case}")
        ordered_validations.append(validation_episode)
    zero_fallback = [episode for episode in ordered_validations if episode.data.fallback_count == 0]
    if len(zero_fallback) < 3:
        raise ValueError(f"MPC refit requires three zero-fallback validations for {case}")
    adaptive = tuple(zero_fallback[:2])
    calibration = zero_fallback[-1]
    unused = tuple(zero_fallback[2:-1])
    excluded = tuple(episode for episode in ordered_validations if episode.data.fallback_count != 0)
    fit_roles = tuple([*fit, basic[0], *adaptive])
    role_paths = [
        {episode.relative_path for episode in fit_roles},
        {episode.relative_path for episode in holdout},
        {calibration.relative_path},
        {episode.relative_path for episode in excluded},
        {episode.relative_path for episode in unused},
    ]
    for index, first in enumerate(role_paths):
        if any(first & second for second in role_paths[index + 1 :]):
            raise ValueError(f"MPC refit episode roles overlap for {case}")
    return _EpisodeRoles(
        fit=fit_roles,
        adaptive_validation=adaptive,
        calibration=calibration,
        holdout=tuple(holdout),
        excluded_fallback=excluded,
        unused_validation=unused,
        lane_identities=lane_identities,
    )


def _role_summary(roles: _EpisodeRoles) -> dict[str, Any]:
    return {
        "fit_episode_count": len(roles.fit),
        "prbs_fit_episode_count": sum(episode.data.role == "fit" for episode in roles.fit),
        "basic_reference_fit_episode_count": sum(
            episode.data.role == "basic_reference" for episode in roles.fit
        ),
        "adaptive_validation_episode_count": len(roles.adaptive_validation),
        "calibration_episode_count": 1,
        "holdout_episode_count": len(roles.holdout),
        "excluded_fallback_validation_count": len(roles.excluded_fallback),
        "unused_zero_fallback_validation_count": len(roles.unused_validation),
        "lane_lifecycle": [
            {
                "lane": value.lane,
                "test_id": value.test_id,
                "episode_count": value.episode_count,
                "test_identity_count": 1,
                "frozen_manifest_cross_checked": value.frozen_manifest_cross_checked,
            }
            for value in roles.lane_identities
        ],
        "episodes": roles.evidence(),
    }


def resolved_refit_plan(source_run: Path) -> dict[str, Any]:
    source = _resolve_source_run(source_run)
    config = load_hierarchical_mpc_config()
    source_failure = _read_json(source / "failure.json")
    return {
        "execution": False,
        "schema": "h3c_hierarchical_mpc_refit_plan",
        "schema_version": 1,
        "source_run": source.relative_to(repository_root()).as_posix(),
        "source_training_commit": source_failure["source_commit"],
        "case_order": list(config["case_order"]),
        "episode_roles": {
            case: _role_summary(_case_roles(case, source)) for case in config["case_order"]
        },
        "arx_structure": "vector_arx_four_lag_four_step",
        "robust_margin": {
            "scope": "case_specific_estimate_common_formula",
            "metric": "absolute_occupied_pmv_prediction_residual",
            "quantile": ROBUST_QUANTILE,
            "order_statistic": "higher",
            "application": "internal_soft_comfort_band_only",
        },
        "holdout_use": "alpha_selection_and_persistence_gate_only",
        "final_fit_includes_holdout": False,
        "terminal_reference_runtime_owner": "effective_occupancy_forecast(step+4)",
        "physical_calls": 0,
        "model_api_calls": 0,
        "candidate_promotion": False,
    }


def _executed_refit_plan(source_run: Path, refit_source_commit: str) -> dict[str, Any]:
    return {
        **resolved_refit_plan(source_run),
        "execution": True,
        "refit_source_commit": refit_source_commit,
    }


def _daily_outdoor_means(reference: EpisodeData) -> dict[int, float]:
    by_day: dict[int, list[float]] = {}
    for time_seconds, outdoor_c in zip(
        reference.times[:-1], reference.disturbances[:-1, 0], strict=True
    ):
        by_day.setdefault(int(time_seconds) // 86400, []).append(float(outdoor_c))
    if not by_day or any(len(values) != 96 for values in by_day.values()):
        raise ValueError("MPC reference episode lacks complete daily outdoor observations")
    return {day: float(np.mean(values)) for day, values in by_day.items()}


def _require_calibration_disturbance_alignment(
    calibration: EpisodeData,
    basic_reference: EpisodeData,
) -> None:
    reference_indices = {
        int(time_seconds): index for index, time_seconds in enumerate(basic_reference.times[:-1])
    }
    if len(reference_indices) != len(basic_reference.times) - 1:
        raise ValueError("MPC Basic-RBC reference timeline contains duplicate timestamps")
    for calibration_index, time_seconds in enumerate(calibration.times[:-1]):
        reference_index = reference_indices.get(int(time_seconds))
        if reference_index is None or not np.array_equal(
            calibration.disturbances[calibration_index],
            basic_reference.disturbances[reference_index],
        ):
            raise ValueError("MPC calibration exogenous disturbances do not align with Basic RBC")


def _calibration_report(
    model: FittedArxModel,
    calibration: EpisodeData,
    occupancy_reference: EpisodeData,
    profile: Mapping[str, Any],
    daily_outdoor_means: Mapping[int, float],
) -> dict[str, Any]:
    layout = model.layout
    _require_calibration_disturbance_alignment(calibration, occupancy_reference)
    residuals: list[float] = []
    temperature_residuals: list[float] = []
    terminal_residual_count = 0
    terminal_occupied_count = 0
    comfort = ComfortModel(profile["comfort"])
    reference_indices = {
        int(time_seconds): index
        for index, time_seconds in enumerate(occupancy_reference.times[:-1])
    }
    if len(reference_indices) != len(occupancy_reference.times) - 1:
        raise ValueError("MPC occupancy reference timeline contains duplicate timestamps")
    last_origin = len(calibration.times) - layout.horizon_steps - 1
    for origin in range(layout.lag_count - 1, last_origin + 1):
        output_history = np.vstack(
            [calibration.outputs[origin - lag] for lag in range(layout.lag_count)]
        )
        control_history = np.vstack(
            [calibration.controls[origin - lag] for lag in range(layout.lag_count)]
        )
        predictions = model.rollout_unclipped(
            output_history,
            control_history,
            calibration.controls[origin : origin + layout.horizon_steps],
            calibration.disturbances[origin : origin + layout.horizon_steps],
        )
        for horizon_step in range(layout.horizon_steps):
            target_index = origin + horizon_step + 1
            target_time = int(calibration.times[target_index])
            reference_index = reference_indices.get(target_time)
            if reference_index is None:
                raise ValueError("MPC calibration target lacks real reference occupancy")
            day = target_time // 86400
            if day not in daily_outdoor_means:
                raise ValueError("MPC calibration target lacks a real daily outdoor mean")
            comfort.update_clothing(target_time, float(daily_outdoor_means[day]))
            for zone_index in range(layout.control_dimension):
                occupied = (
                    float(occupancy_reference.disturbances[reference_index, 2 + zone_index]) > 0.0
                )
                if horizon_step == layout.horizon_steps - 1:
                    terminal_residual_count += 1
                    terminal_occupied_count += int(occupied)
                if not occupied:
                    continue
                predicted_temperature = float(predictions[horizon_step, zone_index])
                observed_temperature = float(calibration.outputs[target_index, zone_index])
                predicted_pmv = comfort.pmv(predicted_temperature)
                observed_pmv = comfort.pmv(observed_temperature)
                residuals.append(abs(predicted_pmv - observed_pmv))
                temperature_residuals.append(abs(predicted_temperature - observed_temperature))
    values = np.asarray(residuals, dtype=np.float64)
    temperature_values = np.asarray(temperature_residuals, dtype=np.float64)
    if len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("MPC calibration has no finite occupied PMV residuals")
    margin = float(np.quantile(values, ROBUST_QUANTILE, method="higher"))
    if not 0.0 <= margin < COMFORT_BAND:
        raise ValueError("MPC calibrated PMV robust margin is outside [0, 0.5)")
    return {
        "schema": "h3c_hierarchical_mpc_residual_calibration",
        "schema_version": 1,
        "residual_metric": "absolute_occupied_pmv_prediction_residual",
        "scope": "case_specific_estimate_common_formula",
        "residual_count": len(values),
        "pmv_absolute_residual_mean": float(np.mean(values)),
        "pmv_absolute_residual_median": float(np.median(values)),
        "pmv_absolute_residual_p95_higher": margin,
        "pmv_absolute_residual_max": float(np.max(values)),
        "temperature_absolute_residual_mean_c": float(np.mean(temperature_values)),
        "temperature_absolute_residual_p95_c": float(
            np.quantile(temperature_values, ROBUST_QUANTILE, method="higher")
        ),
        "pmv_robust_margin": margin,
        "internal_comfort_band": COMFORT_BAND - margin,
        "terminal_reference": {
            "target_offset_steps": layout.horizon_steps,
            "calibration_occupancy_source": ("basic_reference.disturbances[timestamp=origin+4]"),
            "runtime_owner": "effective_occupancy_forecast(step+4)",
            "terminal_zone_targets": terminal_residual_count,
            "terminal_occupied_zone_targets": terminal_occupied_count,
        },
    }


def _finite_model(model: FittedArxModel) -> bool:
    return bool(
        all(
            np.all(np.isfinite(value))
            for value in (
                model.intercept,
                model.coefficients,
                model.scaling.feature_mean,
                model.scaling.feature_scale,
                model.scaling.output_mean,
                model.scaling.output_scale,
            )
        )
        and np.isfinite(model.ridge_alpha)
        and np.isfinite(model.pmv_robust_margin)
        and model.identity == expected_model_identity(model)
    )


def _fit_candidate(
    case: str,
    source: Path,
    target: Path,
) -> dict[str, Any]:
    config = load_hierarchical_mpc_config()
    profile = load_profile(case)
    roles = _case_roles(case, source)
    layout = _arx_layout(profile)
    fit_features, fit_targets = _episode_dataset(layout, [episode.data for episode in roles.fit])
    holdout_features, holdout_targets = _episode_dataset(
        layout, [episode.data for episode in roles.holdout]
    )
    base_model, fit_report = fit_vector_arx(
        layout,
        fit_features,
        fit_targets,
        holdout_features=holdout_features,
        holdout_outputs=holdout_targets,
        alpha_candidates=tuple(float(value) for value in config["ridge_alpha_candidates"]),
    )
    quality = open_loop_prediction_quality(base_model, [episode.data for episode in roles.holdout])
    basic_reference = next(
        episode.data for episode in roles.fit if episode.data.role == "basic_reference"
    )
    calibration = _calibration_report(
        base_model,
        roles.calibration.data,
        basic_reference,
        profile,
        _daily_outdoor_means(basic_reference),
    )
    candidate = with_pmv_robust_margin(base_model, float(calibration["pmv_robust_margin"]))
    checks = {
        "finite": _finite_model(candidate) and quality["finite"] is True,
        "beats_persistence": quality["beats_persistence"] is True,
        "whole_episode_holdout_isolated": not (
            {episode.relative_path for episode in roles.holdout}
            & {
                *[episode.relative_path for episode in roles.fit],
                roles.calibration.relative_path,
            }
        ),
        "calibration_isolated": roles.calibration.relative_path
        not in {episode.relative_path for episode in roles.fit},
        "adaptive_validation_count": len(roles.adaptive_validation) == 2,
        "terminal_occupancy_k_plus_4": calibration["terminal_reference"][
            "calibration_occupancy_source"
        ]
        == "basic_reference.disturbances[timestamp=origin+4]",
        "robust_margin": 0.0 <= candidate.pmv_robust_margin < COMFORT_BAND,
    }
    eligible = all(checks.values())
    report = {
        "schema": "h3c_hierarchical_mpc_refit_candidate",
        "schema_version": 1,
        "case": case,
        "model_identity": candidate.identity,
        "eligible": eligible,
        "checks": checks,
        "episode_roles": _role_summary(roles),
        "fit_rows": len(fit_features),
        "holdout_rows": len(holdout_features),
        "calibration_one_step_rows": len(roles.calibration.data.times) - layout.lag_count,
        "fit_report": fit_report,
        "holdout_use": "alpha_selection_and_persistence_gate_only",
        "final_fit_includes_holdout": False,
        "prediction_quality": quality,
        "calibration": calibration,
        "physical_validation": "pending_fresh_validation",
    }
    if not eligible:
        raise ValueError(f"{case} offline MPC refit candidate is not eligible")
    target.mkdir(parents=True, exist_ok=False)
    candidate.save(target / "model_coefficients.npz")
    _write_json(target / "candidate_report.json", report)
    _write_json(
        target / "model_card.json",
        {
            "schema": "h3c_hierarchical_mpc_refit_model_card",
            "schema_version": 1,
            "case": case,
            "controller": "hierarchical-mpc",
            "model_identity": candidate.identity,
            "model_structure": "vector_arx_four_lag_four_step",
            "hierarchy": "hourly_building_coordinator_and_15_minute_zone_mpc",
            "holdout_use": "alpha_selection_and_persistence_gate_only",
            "final_fit_includes_holdout": False,
            "robust_margin": {
                "scope": "case_specific_estimate_common_formula",
                "estimator": "p95_absolute_occupied_pmv_prediction_residual",
                "quantile": ROBUST_QUANTILE,
                "order_statistic": "higher",
                "calibration_episode": roles.calibration.relative_path,
                "sample_count": calibration["residual_count"],
                "pmv_margin": calibration["pmv_robust_margin"],
                "internal_comfort_band": calibration["internal_comfort_band"],
                "application": "internal_soft_comfort_band_only",
            },
            "unchanged_contract": {
                "public_reward_comfort_band": COMFORT_BAND,
                "objective_weights": dict(profile["objective"]),
                "action_support": {
                    key: config["excitation"][key]
                    for key in ("occupied_bounds_c", "unoccupied_bounds_c")
                },
                "terminal_reference_runtime_owner": ("effective_occupancy_forecast(step+4)"),
            },
            "physical_validation": "pending_fresh_validation",
        },
    )
    loaded = FittedArxModel.load(target / "model_coefficients.npz")
    if (
        loaded.identity != candidate.identity
        or loaded.pmv_robust_margin != candidate.pmv_robust_margin
    ):
        raise ValueError(f"{case} staged MPC refit candidate identity is invalid")
    return report


def _rebuild_candidate_from_source(
    case: str,
    source_run: Path,
) -> _RebuiltCandidate:
    """Rebuild the exact candidate from immutable episodes, never candidate JSON."""
    source = _resolve_source_run(source_run)
    config = load_hierarchical_mpc_config()
    roles = _case_roles(case, source)
    profile = load_profile(case)
    layout = _arx_layout(profile)
    fit_features, fit_targets = _episode_dataset(layout, [episode.data for episode in roles.fit])
    holdout_features, holdout_targets = _episode_dataset(
        layout, [episode.data for episode in roles.holdout]
    )
    base_model, fit_report = fit_vector_arx(
        layout,
        fit_features,
        fit_targets,
        holdout_features=holdout_features,
        holdout_outputs=holdout_targets,
        alpha_candidates=tuple(float(value) for value in config["ridge_alpha_candidates"]),
    )
    basic_reference = next(
        episode.data for episode in roles.fit if episode.data.role == "basic_reference"
    )
    calibration = _calibration_report(
        base_model,
        roles.calibration.data,
        basic_reference,
        profile,
        _daily_outdoor_means(basic_reference),
    )
    candidate = with_pmv_robust_margin(base_model, float(calibration["pmv_robust_margin"]))
    evidence = {
        "episode_roles": _role_summary(roles),
        "fit_rows": len(fit_features),
        "holdout_rows": len(holdout_features),
        "calibration_one_step_rows": len(roles.calibration.data.times) - layout.lag_count,
        "fit_report": fit_report,
        "holdout_use": "alpha_selection_and_persistence_gate_only",
        "final_fit_includes_holdout": False,
        "prediction_quality": open_loop_prediction_quality(
            candidate, [episode.data for episode in roles.holdout]
        ),
        "calibration": calibration,
        "candidate_model_identity": candidate.identity,
    }
    return _RebuiltCandidate(candidate, evidence)


def recompute_candidate_source_evidence(
    case: str,
    source_run: Path,
) -> dict[str, Any]:
    """Expose immutable source evidence for later physical-launch validation."""
    return _rebuild_candidate_from_source(case, source_run).evidence


def _source_model_checks(
    stored: FittedArxModel,
    rebuilt: FittedArxModel | None,
) -> dict[str, bool]:
    if rebuilt is None:
        return {"source_model_exact": True}
    checks = {
        "source_layout": stored.layout == rebuilt.layout,
        "source_ridge_alpha": stored.ridge_alpha == rebuilt.ridge_alpha,
        "source_pmv_robust_margin": stored.pmv_robust_margin == rebuilt.pmv_robust_margin,
        "source_intercept": np.array_equal(stored.intercept, rebuilt.intercept),
        "source_coefficients": np.array_equal(stored.coefficients, rebuilt.coefficients),
        "source_feature_mean": np.array_equal(
            stored.scaling.feature_mean, rebuilt.scaling.feature_mean
        ),
        "source_feature_scale": np.array_equal(
            stored.scaling.feature_scale, rebuilt.scaling.feature_scale
        ),
        "source_output_mean": np.array_equal(
            stored.scaling.output_mean, rebuilt.scaling.output_mean
        ),
        "source_output_scale": np.array_equal(
            stored.scaling.output_scale, rebuilt.scaling.output_scale
        ),
        "stored_expected_identity": stored.identity == expected_model_identity(stored),
        "rebuilt_expected_identity": rebuilt.identity == expected_model_identity(rebuilt),
        "source_model_identity": stored.identity == rebuilt.identity,
    }
    return {**checks, "source_model_exact": all(checks.values())}


def _verify_candidate(
    case: str,
    target: Path,
    *,
    source_evidence: Mapping[str, Any] | None = None,
    source_model: FittedArxModel | None = None,
) -> dict[str, Any]:
    report = _read_json(target / "candidate_report.json")
    card = _read_json(target / "model_card.json")
    model = FittedArxModel.load(target / "model_coefficients.npz")
    role_value = report.get("episode_roles")
    episode_value = role_value.get("episodes") if isinstance(role_value, dict) else None
    fit_paths = set(episode_value.get("fit", [])) if isinstance(episode_value, dict) else set()
    adaptive_paths = (
        set(episode_value.get("adaptive_validation", []))
        if isinstance(episode_value, dict)
        else set()
    )
    holdout_paths = (
        set(episode_value.get("holdout", [])) if isinstance(episode_value, dict) else set()
    )
    calibration_path = (
        str(episode_value.get("calibration", "")) if isinstance(episode_value, dict) else ""
    )
    calibration = report.get("calibration")
    calibration = calibration if isinstance(calibration, dict) else {}
    robust = card.get("robust_margin")
    robust = robust if isinstance(robust, dict) else {}
    reported_checks = report.get("checks")
    fit_report = report.get("fit_report")
    source_model_checks = _source_model_checks(model, source_model)
    checks = {
        "report_schema": report.get("schema") == "h3c_hierarchical_mpc_refit_candidate",
        "case": report.get("case") == case,
        "model_identity": report.get("model_identity") == model.identity,
        "candidate_eligible": report.get("eligible") is True,
        "finite_model": _finite_model(model),
        "layout": model.layout == _arx_layout(load_profile(case)),
        "margin_identity": np.isclose(
            float(calibration.get("pmv_robust_margin", np.nan)),
            model.pmv_robust_margin,
        ),
        "reported_checks": isinstance(reported_checks, dict)
        and bool(reported_checks)
        and all(value is True for value in reported_checks.values()),
        "holdout_isolation": not holdout_paths & (fit_paths | {calibration_path}),
        "holdout_contract": report.get("holdout_use") == "alpha_selection_and_persistence_gate_only"
        and report.get("final_fit_includes_holdout") is False
        and isinstance(fit_report, dict)
        and fit_report.get("final_fit_includes_holdout") is False,
        "adaptive_is_fit": adaptive_paths <= fit_paths,
        "calibration_isolation": bool(calibration_path) and calibration_path not in fit_paths,
        "source_roles": source_evidence is None
        or role_value == source_evidence.get("episode_roles"),
        "source_rows": source_evidence is None
        or (
            report.get("fit_rows") == source_evidence.get("fit_rows")
            and report.get("holdout_rows") == source_evidence.get("holdout_rows")
            and report.get("calibration_one_step_rows")
            == source_evidence.get("calibration_one_step_rows")
        ),
        "source_holdout_contract": source_evidence is None
        or (
            source_evidence.get("holdout_use") == "alpha_selection_and_persistence_gate_only"
            and source_evidence.get("final_fit_includes_holdout") is False
        ),
        "source_persistence_gate": source_evidence is None
        or (
            source_evidence.get("prediction_quality") == report.get("prediction_quality")
            and isinstance(source_evidence.get("prediction_quality"), dict)
            and source_evidence["prediction_quality"].get("finite") is True
            and source_evidence["prediction_quality"].get("beats_persistence") is True
        ),
        "source_calibration": source_evidence is None
        or source_evidence.get("calibration") == calibration,
        "source_fit_report": source_evidence is None
        or source_evidence.get("fit_report") == fit_report,
        "source_candidate_identity": source_evidence is None
        or source_evidence.get("candidate_model_identity") == model.identity,
        **source_model_checks,
        "model_card": card.get("schema") == "h3c_hierarchical_mpc_refit_model_card"
        and card.get("case") == case
        and card.get("model_identity") == model.identity
        and card.get("holdout_use") == "alpha_selection_and_persistence_gate_only"
        and card.get("final_fit_includes_holdout") is False
        and robust.get("estimator") == "p95_absolute_occupied_pmv_prediction_residual"
        and robust.get("scope") == "case_specific_estimate_common_formula"
        and robust.get("quantile") == ROBUST_QUANTILE
        and robust.get("order_statistic") == "higher"
        and robust.get("application") == "internal_soft_comfort_band_only"
        and robust.get("calibration_episode") == calibration_path
        and robust.get("sample_count") == calibration.get("residual_count")
        and np.isclose(float(robust.get("pmv_margin", np.nan)), model.pmv_robust_margin)
        and np.isclose(
            float(robust.get("internal_comfort_band", np.nan)),
            COMFORT_BAND - model.pmv_robust_margin,
        ),
        "fresh_physical_validation_pending": report.get("physical_validation")
        == "pending_fresh_validation"
        and card.get("physical_validation") == "pending_fresh_validation",
    }
    normalized_checks = {name: bool(value) for name, value in checks.items()}
    return {
        "case": case,
        "model_identity": model.identity,
        "pmv_robust_margin": model.pmv_robust_margin,
        "checks": normalized_checks,
        "valid": all(normalized_checks.values()),
    }


def refit_hierarchical_mpc(source_run: Path) -> dict[str, Any]:
    source = _resolve_source_run(source_run)
    config = load_hierarchical_mpc_config()
    source_commit = committed_source_identity()
    root = repository_root() / "outputs" / "baselines" / "mpc" / "refit"
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + source_commit[:8]
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "resolved_plan.json",
        _executed_refit_plan(source, source_commit),
    )
    reports: list[dict[str, Any]] = []
    source_manifest_identity = ""
    try:
        with _training_lock(root / ".refit.lock"):
            file_manifest = _source_file_manifest(source)
            _write_json(
                run_dir / "source_manifest.json",
                {
                    "schema": "h3c_hierarchical_mpc_refit_source_manifest",
                    "schema_version": 1,
                    "source_run": source.relative_to(repository_root()).as_posix(),
                    "source_training_commit": _read_json(source / "failure.json")["source_commit"],
                    "source_plan_sha256": _sha256(source / "resolved_plan.json"),
                    "source_failure_sha256": _sha256(source / "failure.json"),
                    "files": file_manifest,
                },
            )
            source_manifest_identity = _sha256(run_dir / "source_manifest.json")
            for case in config["case_order"]:
                report = _fit_candidate(case, source, run_dir / case / "candidate_model")
                rebuilt = _rebuild_candidate_from_source(case, source)
                verification = _verify_candidate(
                    case,
                    run_dir / case / "candidate_model",
                    source_evidence=rebuilt.evidence,
                    source_model=rebuilt.model,
                )
                if verification["valid"] is not True:
                    raise ValueError(f"{case} staged MPC refit verification failed")
                _write_json(run_dir / case / "verification.json", verification)
                reports.append(report)
        secret_count = secret_occurrences(run_dir)
        if secret_count:
            raise ValueError("secret exposure detected in MPC refit outputs")
    except Exception as error:
        _write_json(
            run_dir / "failure.json",
            {
                "schema": "h3c_hierarchical_mpc_refit_failure",
                "schema_version": 1,
                "refit_source_commit": source_commit,
                "source_run": source.relative_to(repository_root()).as_posix(),
                "source_manifest_sha256": source_manifest_identity,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_cases": [report["case"] for report in reports],
                "secret_exposure_count": secret_occurrences(run_dir),
            },
        )
        raise
    completion = {
        "schema": "h3c_hierarchical_mpc_refit_completion",
        "schema_version": 1,
        "refit_source_commit": source_commit,
        "source_run": source.relative_to(repository_root()).as_posix(),
        "source_manifest_sha256": source_manifest_identity,
        "case_order": list(config["case_order"]),
        "cases": [
            {
                "case": report["case"],
                "model_identity": report["model_identity"],
                "pmv_robust_margin": report["calibration"]["pmv_robust_margin"],
                "prediction_quality": report["prediction_quality"],
            }
            for report in reports
        ],
        "robust_margin_scope": "case_specific_estimate_common_formula",
        "secret_exposure_count": secret_count,
        "physical_calls": 0,
        "model_api_calls": 0,
        "candidate_promotion": False,
    }
    _write_completion(run_dir / "completion.json", completion)
    return {**completion, "run_dir": str(run_dir)}


def verify_refit_workspace(workspace: Path) -> dict[str, Any]:
    try:
        root = repository_root().resolve()
        target = workspace if workspace.is_absolute() else root / workspace
        target = target.resolve()
        target.relative_to((root / "outputs" / "baselines" / "mpc" / "refit").resolve())
        completion = _read_json(target / "completion.json")
        resolved_plan = _read_json(target / "resolved_plan.json")
        if (target / "failure.json").exists():
            raise ValueError("MPC refit workspace contains failure evidence")
        source_manifest = _read_json(target / "source_manifest.json")
        source = _resolve_source_run(root / str(source_manifest["source_run"]))
        source_failure = _read_json(source / "failure.json")
        refit_source_commit = completion.get("refit_source_commit")
        expected_plan = (
            _executed_refit_plan(source, refit_source_commit)
            if isinstance(refit_source_commit, str)
            else {}
        )
        refit_commit_suffix = (
            refit_source_commit[:8] if isinstance(refit_source_commit, str) else ""
        )
        expected_files = source_manifest.get("files")
        if not isinstance(expected_files, list) or expected_files != _source_file_manifest(source):
            raise ValueError("MPC refit source file identity changed")
        config = load_hierarchical_mpc_config()
        case_results: list[dict[str, Any]] = []
        for case in config["case_order"]:
            rebuilt = _rebuild_candidate_from_source(case, source)
            case_results.append(
                _verify_candidate(
                    case,
                    target / case / "candidate_model",
                    source_evidence=rebuilt.evidence,
                    source_model=rebuilt.model,
                )
            )
        completion_cases = completion.get("cases")
        completion_by_case = (
            {str(value.get("case")): value for value in completion_cases}
            if isinstance(completion_cases, list)
            and all(isinstance(value, dict) for value in completion_cases)
            else {}
        )
        checks = {
            "completion_schema": completion.get("schema")
            == "h3c_hierarchical_mpc_refit_completion",
            "robust_margin_scope": completion.get("robust_margin_scope")
            == "case_specific_estimate_common_formula",
            "case_order": completion.get("case_order") == list(config["case_order"]),
            "plan_identity": resolved_plan == expected_plan
            and isinstance(refit_source_commit, str)
            and len(refit_source_commit) == 40
            and target.name.endswith(f"-{refit_commit_suffix}")
            and resolved_plan.get("source_run") == completion.get("source_run"),
            "source_identity": source_manifest.get("schema")
            == "h3c_hierarchical_mpc_refit_source_manifest"
            and source_manifest.get("source_run") == completion.get("source_run")
            and source_manifest.get("source_training_commit") == source_failure.get("source_commit")
            and source_manifest.get("source_plan_sha256") == _sha256(source / "resolved_plan.json")
            and source_manifest.get("source_failure_sha256") == _sha256(source / "failure.json")
            and completion.get("source_manifest_sha256")
            == _sha256(target / "source_manifest.json"),
            "all_candidates": all(result["valid"] for result in case_results),
            "completion_candidates": set(completion_by_case) == set(config["case_order"])
            and all(
                completion_by_case[result["case"]].get("model_identity") == result["model_identity"]
                and np.isclose(
                    float(completion_by_case[result["case"]].get("pmv_robust_margin", np.nan)),
                    float(result["pmv_robust_margin"]),
                )
                for result in case_results
            ),
            "no_physical_or_model_calls": completion.get("physical_calls") == 0
            and completion.get("model_api_calls") == 0,
            "no_promotion": completion.get("candidate_promotion") is False,
            "secret_scan": completion.get("secret_exposure_count") == 0
            and secret_occurrences(target) == 0,
        }
        return {
            "schema": "h3c_hierarchical_mpc_refit_verification",
            "schema_version": 1,
            "workspace": str(target),
            "checks": checks,
            "cases": case_results,
            "valid": all(checks.values()),
        }
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {
            "schema": "h3c_hierarchical_mpc_refit_verification",
            "schema_version": 1,
            "workspace": str(workspace),
            "error": f"{type(error).__name__}: {error}",
            "valid": False,
        }
