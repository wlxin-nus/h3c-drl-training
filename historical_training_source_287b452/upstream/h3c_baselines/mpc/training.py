"""Parallel, episode-safe identification for the hierarchical MPC baseline."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from threading import Event
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from h3c.experiments.profiles import load_profile, repository_root
from h3c.runtime.clients import BoptestHttpClient
from h3c.runtime.comfort import ComfortModel, step_reward
from h3c.runtime.occupancy import effective_count
from h3c.runtime.protocol import (
    control_input,
    forecast_points,
    require_time,
    resolve_forecast_missing_occupancy,
    site_power,
    zone_temperature_c,
)
from h3c.runtime.source_identity import committed_source_identity
from h3c_baselines.configuration import load_hierarchical_mpc_config
from h3c_baselines.controllers.basic_rbc import basic_rbc_setpoints
from h3c_baselines.controllers.enhanced_rbc import EnhancedRbcController
from h3c_baselines.mpc.optimizer import HierarchicalMpcController, comfort_target_met
from h3c_baselines.mpc.vector_arx import (
    ArxLayout,
    FittedArxModel,
    build_dataset,
    expected_model_identity,
    fit_vector_arx,
)
from h3c_baselines.outputs.integrity import secret_occurrences

STEP_SECONDS = 900
STEPS_PER_WEEK = 7 * 24 * 4


class TrainingPhysicalClient(Protocol):
    test_id: str | None

    def select_testcase(self, testcase: str) -> str: ...

    def initialize_selected(
        self, start_time_seconds: int, warmup_period_seconds: int
    ) -> dict[str, Any]: ...

    def forecast(
        self, points: Sequence[str], horizon_seconds: int, interval_seconds: int
    ) -> dict[str, list[float | None]]: ...

    def advance(self, controls: Mapping[str, float]) -> dict[str, Any]: ...

    def stop(self) -> None: ...


PhysicalFactory = Callable[[str], TrainingPhysicalClient]
DiagnosticSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class EpisodeData:
    role: str
    episode: int
    lane: int
    test_id: str
    times: NDArray[np.int64]
    outputs: NDArray[np.float64]
    controls: NDArray[np.float64]
    disturbances: NDArray[np.float64]
    reward: float
    peak_occupied_absolute_pmv: float
    fallback_count: int
    recovery_step_count: int


@dataclass
class _Lane:
    index: int
    client: TrainingPhysicalClient
    test_id: str
    initialize_count: int = 0
    stop_count: int = 0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def _write_completion(path: Path, value: Any) -> None:
    if path.exists():
        raise ValueError("MPC training completion already exists")
    pending = path.with_name(f".{path.name}.pending")
    with pending.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(pending, path)


@contextmanager
def _training_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("another hierarchical MPC training task owns the lock") from error
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def _disturbance(
    profile: Mapping[str, Any],
    forecast: Mapping[str, Sequence[float]],
    zones: tuple[str, ...],
    step: int,
    time_seconds: int,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    occupancy = {
        zone: effective_count(
            profile["occupancy"],
            time_seconds,
            float(forecast[profile["zones"][zone]["occupancy_forecast"]][step]),
        )
        for zone in zones
    }
    day_fraction = (time_seconds % 86400) / 86400.0
    return (
        np.asarray(
            [
                float(forecast[profile["global_inputs"]["outdoor_temperature"]][step]) - 273.15,
                float(forecast[profile["global_inputs"]["solar_irradiance"]][step]),
                *[occupancy[zone] for zone in zones],
                math.sin(2 * math.pi * day_fraction),
                math.cos(2 * math.pi * day_fraction),
            ],
            dtype=np.float64,
        ),
        occupancy,
    )


def _daily_outdoor_mean(
    profile: Mapping[str, Any], forecast: Mapping[str, Sequence[float]], step: int
) -> float:
    start = (step // 96) * 96
    values = forecast[profile["global_inputs"]["outdoor_temperature"]][start : start + 96]
    if len(values) != 96:
        raise ValueError("training forecast lacks a complete in-week day")
    return sum(float(value) - 273.15 for value in values) / 96


def _future_occupancy(
    profile: Mapping[str, Any],
    forecast: Mapping[str, Sequence[float]],
    zones: tuple[str, ...],
    step: int,
    time_seconds: int,
    last_index: int,
) -> dict[str, list[float]]:
    return {
        zone: [
            effective_count(
                profile["occupancy"],
                time_seconds + offset * STEP_SECONDS,
                float(
                    forecast[profile["zones"][zone]["occupancy_forecast"]][
                        min(step + offset, last_index)
                    ]
                ),
            )
            for offset in range(1, 5)
        ]
        for zone in zones
    }


class ActiveExcitationController:
    """Deterministic PRBS/GBN excitation with canonical comfort recovery."""

    def __init__(
        self,
        zones: Sequence[str],
        config: Mapping[str, Any],
        seed: int,
    ) -> None:
        self.zones = tuple(zones)
        self.config = dict(config)
        self.random = np.random.default_rng(seed)
        self.remaining = {zone: 0 for zone in self.zones}
        self.targets = {zone: 25.0 for zone in self.zones}
        self.recovery = {zone: False for zone in self.zones}

    def decide(
        self,
        *,
        occupancy: Mapping[str, float],
        pmv: Mapping[str, float],
        recovery_setpoints: Mapping[str, float],
    ) -> tuple[dict[str, float], int]:
        result: dict[str, float] = {}
        recovery_steps = 0
        trigger = float(self.config["comfort_recovery_trigger_absolute_pmv"])
        release = float(self.config["comfort_recovery_release_absolute_pmv"])
        dwell_steps = tuple(int(value) // 15 for value in self.config["dwell_minutes"])
        for zone in self.zones:
            occupied = float(occupancy[zone]) > 0
            if occupied and abs(float(pmv[zone])) > trigger:
                self.recovery[zone] = True
            elif self.recovery[zone] and abs(float(pmv[zone])) <= release:
                self.recovery[zone] = False
            if self.recovery[zone]:
                result[zone] = float(recovery_setpoints[zone])
                recovery_steps += 1
                continue
            if self.remaining[zone] <= 0:
                bounds = (
                    self.config["occupied_bounds_c"]
                    if occupied
                    else self.config["unoccupied_bounds_c"]
                )
                self.targets[zone] = float(self.random.uniform(float(bounds[0]), float(bounds[1])))
                self.remaining[zone] = int(self.random.choice(dwell_steps))
            result[zone] = self.targets[zone]
            self.remaining[zone] -= 1
        return result, recovery_steps


def _mpc_horizon(
    profile: Mapping[str, Any],
    forecast: Mapping[str, Sequence[float]],
    zones: tuple[str, ...],
    step: int,
    time_seconds: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[int], list[float]]:
    disturbances: list[NDArray[np.float64]] = []
    prices: list[float] = []
    occupancy: list[list[float]] = []
    times: list[int] = []
    daily_means: list[float] = []
    price_point = profile["global_inputs"]["electricity_price"]
    for offset in range(4):
        disturbance, counts = _disturbance(
            profile, forecast, zones, step + offset, time_seconds + offset * STEP_SECONDS
        )
        disturbances.append(disturbance)
        prices.append(float(forecast[price_point][step + offset]))
        occupancy.append([counts[zone] for zone in zones])
        times.append(time_seconds + offset * STEP_SECONDS)
        daily_means.append(_daily_outdoor_mean(profile, forecast, step + offset))
    return (
        np.vstack(disturbances),
        np.asarray(prices, dtype=np.float64),
        np.asarray(occupancy, dtype=np.float64),
        times,
        daily_means,
    )


def _collect_episode(
    *,
    lane: _Lane,
    profile: Mapping[str, Any],
    role: str,
    episode: int,
    excitation_config: Mapping[str, Any],
    output_dir: Path,
    model: FittedArxModel | None = None,
    cancel_event: Event | None = None,
    diagnostic_sink: DiagnosticSink | None = None,
    forecast_phase: str = "mpc_training",
) -> EpisodeData:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("parallel MPC episode batch was cancelled")
    if diagnostic_sink is not None and (role != "validation" or model is None):
        raise ValueError("MPC diagnostic sink is only valid for a modeled validation episode")
    zones = tuple(profile["zones"])
    start_time = (int(profile["evaluation_start_day"]) - 7) * 86400
    state = lane.client.initialize_selected(start_time, 7 * 86400)
    lane.initialize_count += 1
    if lane.client.test_id != lane.test_id:
        raise ValueError("BOPTEST test identity changed during MPC episode initialize")
    require_time(state, start_time)
    steps = STEPS_PER_WEEK - (4 if role == "validation" else 0)
    points = forecast_points(profile)
    raw_forecast = lane.client.forecast(points, STEPS_PER_WEEK * STEP_SECONDS, STEP_SECONDS)
    forecast, _ = resolve_forecast_missing_occupancy(
        profile,
        raw_forecast,
        points,
        STEPS_PER_WEEK + 1,
        forecast_phase=forecast_phase,
        start_time_seconds=start_time,
        step_seconds=STEP_SECONDS,
    )
    comfort = ComfortModel(profile["comfort"])
    enhanced = EnhancedRbcController(zones, repository_root() / profile["program"])
    excitation = ActiveExcitationController(
        zones,
        excitation_config,
        seed=10_000 * (lane.index + 1) + 101 * episode + sum(ord(value) for value in role),
    )
    hierarchical = (
        HierarchicalMpcController(
            model,
            profile["objective"],
            load_hierarchical_mpc_config()["excitation"],
        )
        if role == "validation" and model is not None
        else None
    )
    last_setpoints = {zone: float(profile["protocol"]["initial_setpoint_c"]) for zone in zones}
    last_pmv = {zone: 0.0 for zone in zones}
    last_occupancy = {zone: 0.0 for zone in zones}
    initial_output = np.asarray(
        [*[zone_temperature_c(profile, state, zone) for zone in zones], site_power(profile, state)],
        dtype=np.float64,
    )
    output_history = np.vstack([initial_output] * 4)
    control_history = np.vstack([[last_setpoints[zone] for zone in zones]] * 4)
    times: list[int] = []
    outputs: list[NDArray[np.float64]] = []
    controls: list[NDArray[np.float64]] = []
    disturbances: list[NDArray[np.float64]] = []
    reward_total = 0.0
    peak_pmv = 0.0
    fallback_count = 0
    recovery_count = 0
    for step in range(steps):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("parallel MPC episode batch was cancelled")
        action_time = start_time + step * STEP_SECONDS
        require_time(state, action_time)
        disturbance, occupancy = _disturbance(profile, forecast, zones, step, action_time)
        daily_outdoor_mean = _daily_outdoor_mean(profile, forecast, step)
        comfort.update_clothing(action_time, daily_outdoor_mean)
        current_temperatures = {zone: zone_temperature_c(profile, state, zone) for zone in zones}
        current_pmv = {zone: comfort.pmv(current_temperatures[zone]) for zone in zones}
        future = _future_occupancy(profile, forecast, zones, step, action_time, STEPS_PER_WEEK)
        recovery_setpoints, _ = enhanced.decide(
            occupancy=occupancy,
            future_occupancy=future,
            last_setpoints_c=last_setpoints,
            last_pmv=last_pmv,
            last_occupancy=last_occupancy,
        )
        decision_diagnostics: Mapping[str, Any] | None = None
        if role in {"fit", "holdout"}:
            setpoints, recovery_steps = excitation.decide(
                occupancy=occupancy,
                pmv=current_pmv,
                recovery_setpoints=recovery_setpoints,
            )
            recovery_count += recovery_steps
        elif role == "basic_reference":
            setpoints = basic_rbc_setpoints(zones, occupancy)
        elif role == "validation" and hierarchical is not None:
            horizon = _mpc_horizon(profile, forecast, zones, step, action_time)
            decision = hierarchical.decide(
                step=step,
                output_history=output_history,
                control_history=control_history,
                disturbances=horizon[0],
                prices=horizon[1],
                occupancy=horizon[2],
                terminal_occupancy={zone: future[zone][3] for zone in zones},
                action_times=horizon[3],
                daily_outdoor_means_c=horizon[4],
                comfort=comfort,
                previous_setpoints_c=last_setpoints,
                enhanced_rbc_warm_start=recovery_setpoints,
            )
            setpoints = decision.setpoints_c
            decision_diagnostics = decision.diagnostics
            fallback_count += int(decision.diagnostics["method_degraded"] is True)
        else:
            raise ValueError("unknown MPC training episode role")
        current_output = np.asarray(
            [*[current_temperatures[zone] for zone in zones], site_power(profile, state)],
            dtype=np.float64,
        )
        times.append(action_time)
        outputs.append(current_output)
        controls.append(np.asarray([setpoints[zone] for zone in zones], dtype=np.float64))
        disturbances.append(disturbance)
        next_state = lane.client.advance(control_input(profile, setpoints))
        if lane.client.test_id != lane.test_id:
            raise ValueError("BOPTEST test identity changed after MPC episode advance")
        require_time(next_state, action_time + STEP_SECONDS)
        next_temperatures = {zone: zone_temperature_c(profile, next_state, zone) for zone in zones}
        pmv = {zone: comfort.pmv(next_temperatures[zone]) for zone in zones}
        power = site_power(profile, next_state)
        price = float(forecast[profile["global_inputs"]["electricity_price"]][step])
        cost = power * 0.25 / 1000 * price
        reward = step_reward(
            cost=cost,
            pmv=[pmv[zone] for zone in zones],
            occupancy=[occupancy[zone] for zone in zones],
            setpoints_c=[setpoints[zone] for zone in zones],
            previous_setpoints_c=[last_setpoints[zone] for zone in zones],
            objective=profile["objective"],
        )
        reward_total += reward
        if diagnostic_sink is not None:
            assert (
                model is not None
                and role == "validation"
                and hierarchical is not None
                and decision_diagnostics is not None
            )
            diagnostic_sink(
                {
                    "schema": "h3c_hierarchical_mpc_validation_step",
                    "schema_version": 1,
                    "step": step,
                    "test_id": lane.test_id,
                    "model_identity": model.identity,
                    "action_time_seconds": action_time,
                    "outcome_time_seconds": action_time + STEP_SECONDS,
                    "occupancy": occupancy,
                    "daily_outdoor_mean_c": daily_outdoor_mean,
                    "clothing_insulation": comfort.clothing_insulation,
                    "action_zone_temperature_c": current_temperatures,
                    "action_pmv": current_pmv,
                    "setpoints_c": setpoints,
                    "electricity_price": price,
                    "outcome_site_power_w": power,
                    "step_cost": cost,
                    "outcome_zone_temperature_c": next_temperatures,
                    "outcome_pmv": pmv,
                    "step_reward": reward,
                    "controller_diagnostics": decision_diagnostics,
                }
            )
        occupied_values = [abs(pmv[zone]) for zone in zones if occupancy[zone] > 0]
        peak_pmv = max([peak_pmv, *occupied_values])
        output_history = np.vstack(
            ([*[next_temperatures[zone] for zone in zones], power], output_history[:-1])
        )
        control_history = np.vstack(([*[setpoints[zone] for zone in zones]], control_history[:-1]))
        state = next_state
        last_setpoints = setpoints
        last_pmv = pmv
        last_occupancy = occupancy
    terminal_time = start_time + steps * STEP_SECONDS
    times.append(terminal_time)
    outputs.append(
        np.asarray(
            [
                *[zone_temperature_c(profile, state, zone) for zone in zones],
                site_power(profile, state),
            ],
            dtype=np.float64,
        )
    )
    controls.append(controls[-1].copy())
    disturbances.append(disturbances[-1].copy())
    result = EpisodeData(
        role,
        episode,
        lane.index,
        lane.test_id,
        np.asarray(times, dtype=np.int64),
        np.vstack(outputs),
        np.vstack(controls),
        np.vstack(disturbances),
        reward_total,
        peak_pmv,
        fallback_count,
        recovery_count,
    )
    path = output_dir / "episodes" / f"{role}-{episode:03d}-lane-{lane.index}"
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        path / "trajectory.npz",
        times=result.times,
        outputs=result.outputs,
        controls=result.controls,
        disturbances=result.disturbances,
    )
    manifest = {
        "role": role,
        "episode": episode,
        "lane": lane.index,
        "test_id": lane.test_id,
        "start_time_seconds": start_time,
        "warmup_period_seconds": 7 * 86400,
        "steps": steps,
        "reward": reward_total,
        "peak_occupied_absolute_pmv": peak_pmv,
        "fallback_count": fallback_count,
        "recovery_step_count": recovery_count,
        "forecast_phase": forecast_phase,
    }
    if model is not None:
        manifest["model_identity"] = model.identity
    _write_json(path / "manifest.json", manifest)
    return result


def _episode_dataset(
    layout: ArxLayout, episodes: Sequence[EpisodeData]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    feature_parts: list[NDArray[np.float64]] = []
    target_parts: list[NDArray[np.float64]] = []
    for episode in episodes:
        features, targets, _ = build_dataset(
            layout,
            episode.times,
            episode.outputs,
            episode.controls,
            episode.disturbances,
        )
        feature_parts.append(features)
        target_parts.append(targets)
    return np.vstack(feature_parts), np.vstack(target_parts)


def _arx_layout(profile: Mapping[str, Any]) -> ArxLayout:
    zones = tuple(profile["zones"])
    return ArxLayout(
        zones,
        (
            "outdoor_temperature_c",
            "solar_irradiance_w_m2",
            *[f"effective_occupancy_{zone}" for zone in zones],
            "time_sine",
            "time_cosine",
        ),
        lag_count=4,
        horizon_steps=4,
    )


def open_loop_prediction_quality(
    model: FittedArxModel, episodes: Sequence[EpisodeData]
) -> dict[str, Any]:
    predicted: list[NDArray[np.float64]] = []
    persistent: list[NDArray[np.float64]] = []
    truth: list[NDArray[np.float64]] = []
    layout = model.layout
    for episode in episodes:
        last_origin = len(episode.times) - layout.horizon_steps - 1
        for origin in range(layout.lag_count - 1, last_origin + 1):
            output_history = np.vstack(
                [episode.outputs[origin - lag] for lag in range(layout.lag_count)]
            )
            control_history = np.vstack(
                [episode.controls[origin - lag] for lag in range(layout.lag_count)]
            )
            candidate = model.rollout_unclipped(
                output_history,
                control_history,
                episode.controls[origin : origin + layout.horizon_steps],
                episode.disturbances[origin : origin + layout.horizon_steps],
            )
            actual = episode.outputs[origin + 1 : origin + layout.horizon_steps + 1]
            predicted.append(candidate)
            persistent.append(np.vstack([episode.outputs[origin]] * layout.horizon_steps))
            truth.append(actual)
    prediction = np.vstack(predicted)
    persistence = np.vstack(persistent)
    observed = np.vstack(truth)

    def rmse(error: NDArray[np.float64]) -> float:
        return float(np.sqrt(np.mean(np.square(error))))

    scale = model.scaling.output_scale
    model_metrics = {
        "standardized_rmse": rmse((prediction - observed) / scale),
        "zone_temperature_rmse_c": rmse(prediction[:, :-1] - observed[:, :-1]),
        "site_power_rmse_w": rmse(prediction[:, -1] - observed[:, -1]),
    }
    persistence_metrics = {
        "standardized_rmse": rmse((persistence - observed) / scale),
        "zone_temperature_rmse_c": rmse(persistence[:, :-1] - observed[:, :-1]),
        "site_power_rmse_w": rmse(persistence[:, -1] - observed[:, -1]),
    }
    finite = all(
        np.isfinite(value) for value in [*model_metrics.values(), *persistence_metrics.values()]
    )
    beats = finite and all(
        model_metrics[name] < persistence_metrics[name] for name in model_metrics
    )
    return {
        "model": model_metrics,
        "persistence": persistence_metrics,
        "finite": finite,
        "beats_persistence": beats,
        "origin_count": len(predicted),
    }


def resolved_training_plan(*, workers: int, max_fit_episodes: int) -> dict[str, Any]:
    config = load_hierarchical_mpc_config()
    capacity = min(
        int(config["maximum_workers"]),
        int(config["service_workers"]) - int(config["reserved_workers"]),
    )
    if workers < 1 or workers > capacity:
        raise ValueError(f"MPC worker count must be within 1..{capacity}")
    checkpoints = [
        int(value) for value in config["fit_checkpoints"] if int(value) <= max_fit_episodes
    ]
    if max_fit_episodes < 8 or not checkpoints or checkpoints[-1] != max_fit_episodes:
        raise ValueError("maximum fit episodes must be one of 8, 16, 32, or 64")
    return {
        "execution": False,
        "schema": "h3c_hierarchical_mpc_training_plan",
        "schema_version": 1,
        "case_order": config["case_order"],
        "workers": workers,
        "service_workers": config["service_workers"],
        "reserved_workers": config["reserved_workers"],
        "fit_checkpoints": checkpoints,
        "holdout_episodes": config["holdout_episodes"],
        "training_week_days_before_evaluation": 7,
        "warmup_days_per_episode": 7,
        "wall_clock_limit_hours": config["wall_clock_limit_hours"],
    }


def _fit_case(
    *,
    case: str,
    lanes: Sequence[_Lane],
    config: Mapping[str, Any],
    output_dir: Path,
    maximum_fit_episodes: int,
    deadline: float,
) -> tuple[FittedArxModel, dict[str, Any]]:
    profile = load_profile(case)
    layout = _arx_layout(profile)
    if layout.lag_count != int(config["model"]["lag_count"]) or layout.horizon_steps != int(
        config["model"]["horizon_steps"]
    ):
        raise ValueError("hierarchical MPC ARX configuration is invalid")
    worker_count = len(lanes)

    def collect_batch(
        role: str, first: int, count: int, model: FittedArxModel | None = None
    ) -> list[EpisodeData]:
        if time.monotonic() >= deadline:
            raise TimeoutError("hierarchical MPC wall-clock limit reached")
        cancel_event = Event()
        ordered: list[EpisodeData | None] = [None] * count

        def collect(index: int) -> EpisodeData:
            try:
                return _collect_episode(
                    lane=lanes[index % worker_count],
                    profile=profile,
                    role=role,
                    episode=first + index,
                    excitation_config=config["excitation"],
                    output_dir=output_dir,
                    model=model,
                    cancel_event=cancel_event,
                )
            except Exception:
                cancel_event.set()
                raise

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: dict[Future[EpisodeData], int] = {
                executor.submit(collect, index): index for index in range(count)
            }
            try:
                for future in as_completed(futures):
                    ordered[futures[future]] = future.result()
            except Exception:
                cancel_event.set()
                for future in futures:
                    future.cancel()
                raise
        if any(value is None for value in ordered):
            raise AssertionError("parallel MPC episode batch ended without all results")
        return [value for value in ordered if value is not None]

    holdouts = collect_batch("holdout", 0, int(config["holdout_episodes"]))
    reference = collect_batch("basic_reference", 0, 1)[0]
    fits: list[EpisodeData] = []
    candidates: list[tuple[FittedArxModel, dict[str, Any]]] = []
    low_improvement_count = 0
    previous_eligible_reward: float | None = None
    for checkpoint in config["fit_checkpoints"]:
        checkpoint = int(checkpoint)
        if checkpoint > maximum_fit_episodes:
            break
        while len(fits) < checkpoint:
            count = min(worker_count, checkpoint - len(fits))
            fits.extend(collect_batch("fit", len(fits), count))
        fit_features, fit_targets = _episode_dataset(layout, fits)
        holdout_features, holdout_targets = _episode_dataset(layout, holdouts)
        model, fit_report = fit_vector_arx(
            layout,
            fit_features,
            fit_targets,
            holdout_features=holdout_features,
            holdout_outputs=holdout_targets,
            alpha_candidates=tuple(float(value) for value in config["ridge_alpha_candidates"]),
        )
        quality = open_loop_prediction_quality(model, holdouts)
        validation = collect_batch("validation", checkpoint, 1, model=model)[0]
        eligible = bool(
            quality["beats_persistence"]
            and validation.fallback_count == 0
            and validation.peak_occupied_absolute_pmv <= 0.70
        )
        report = {
            "checkpoint_fit_episodes": checkpoint,
            "fit_report": fit_report,
            "prediction_quality": quality,
            "closed_loop_validation": {
                "reward": validation.reward,
                "peak_occupied_absolute_pmv": validation.peak_occupied_absolute_pmv,
                "fallback_count": validation.fallback_count,
            },
            "eligible": eligible,
        }
        _write_json(output_dir / "checkpoints" / f"checkpoint-{checkpoint:03d}.json", report)
        if eligible:
            candidates.append((model, report))
            if previous_eligible_reward is not None:
                improvement = (validation.reward - previous_eligible_reward) / max(
                    abs(previous_eligible_reward), 1e-12
                )
                low_improvement_count = low_improvement_count + 1 if improvement < 0.01 else 0
            previous_eligible_reward = validation.reward
            if low_improvement_count >= 2:
                break
    if not candidates:
        raise ValueError(f"{case} produced no eligible hierarchical MPC checkpoint")
    selected_model, selected_report = max(
        candidates,
        key=lambda item: float(item[1]["closed_loop_validation"]["reward"]),
    )
    summary = {
        "case": case,
        "selected_checkpoint_fit_episodes": selected_report["checkpoint_fit_episodes"],
        "selected_model_identity": selected_model.identity,
        "selected_validation": selected_report["closed_loop_validation"],
        "selected_prediction_quality": selected_report["prediction_quality"],
        "basic_rbc_training_week_reward": reference.reward,
        "fit_episodes_collected": len(fits),
        "holdout_episodes": len(holdouts),
        "eligible_checkpoint_count": len(candidates),
    }
    return selected_model, summary


def _freeze_model(
    *,
    case: str,
    model: FittedArxModel,
    summary: Mapping[str, Any],
    source_commit: str,
    training_run: Path,
    lanes: Sequence[_Lane],
    target: Path | None = None,
) -> Path:
    target = target or repository_root() / "models" / "mpc" / case
    if target.exists():
        raise ValueError(f"frozen MPC model already exists for {case}")
    target.mkdir(parents=True)
    model.save(target / "model_coefficients.npz")
    profile = load_profile(case)
    _write_json(
        target / "model_card.json",
        {
            "schema": "h3c_hierarchical_mpc_model_card",
            "schema_version": 1,
            "case": case,
            "controller": "hierarchical-mpc",
            "model_identity": model.identity,
            "pmv_robust_margin": model.pmv_robust_margin,
            "source_commit": source_commit,
            "training_week_start_day": int(profile["evaluation_start_day"]) - 7,
            "training_week_days": 7,
            "model_structure": "vector_arx_four_lag_four_step",
            "hierarchy": "hourly_building_coordinator_and_15_minute_zone_mpc",
            "control_support_bounds_c": {
                key: load_hierarchical_mpc_config()["excitation"][key]
                for key in ("occupied_bounds_c", "unoccupied_bounds_c")
            },
            "summary": dict(summary),
        },
    )
    _write_json(
        target / "training_manifest.json",
        {
            "schema": "h3c_hierarchical_mpc_training_manifest",
            "schema_version": 1,
            "case": case,
            "source_commit": source_commit,
            "model_identity": model.identity,
            "training_output": str(training_run.relative_to(repository_root())).replace("\\", "/"),
            "lane_lifecycle": [
                {
                    "lane": lane.index,
                    "test_id": lane.test_id,
                    "select_count": 1,
                    "initialize_count": lane.initialize_count,
                    "stop_count": lane.stop_count,
                    "warmup_days_per_initialize": 7,
                }
                for lane in lanes
            ],
        },
    )
    return target


def _verify_mpc_model_directory(case: str, target: Path) -> dict[str, Any]:
    profile = load_profile(case)
    configuration = load_hierarchical_mpc_config()
    expected_support = {
        key: configuration["excitation"][key]
        for key in ("occupied_bounds_c", "unoccupied_bounds_c")
    }
    card = json.loads((target / "model_card.json").read_text(encoding="utf-8"))
    manifest = json.loads((target / "training_manifest.json").read_text(encoding="utf-8"))
    model = FittedArxModel.load(target / "model_coefficients.npz")
    zones = tuple(profile["zones"])
    schema_version = manifest.get("schema_version")
    legacy_lanes = manifest.get("lane_lifecycle")
    fresh_validation = manifest.get("fresh_validation")
    admission_mode = manifest.get("admission_mode", "strict_zero_fallback")
    degraded_admission = admission_mode == "post_result_method_degraded"
    fresh_peak = (
        fresh_validation.get("occupied_peak_absolute_pmv")
        if isinstance(fresh_validation, Mapping)
        else None
    )
    fresh_comfort_target_valid = (
        not isinstance(fresh_peak, bool)
        and isinstance(fresh_peak, (int, float))
        and math.isfinite(float(fresh_peak))
        and isinstance(fresh_validation.get("comfort_target_met"), bool)
        and fresh_validation["comfort_target_met"] is comfort_target_met(float(fresh_peak))
        if isinstance(fresh_validation, Mapping)
        else False
    )
    lifecycle_valid = (
        schema_version == 1
        and isinstance(legacy_lanes, list)
        and len(legacy_lanes) == 4
        and all(
            row.get("select_count") == 1
            and row.get("stop_count") == 1
            and row.get("initialize_count", 0) > 0
            and row.get("warmup_days_per_initialize") == 7
            for row in legacy_lanes
        )
    ) or (
        schema_version == 2
        and card.get("schema_version") == 2
        and card.get("physical_validation")
        == ("fresh_validation_method_degraded" if degraded_admission else "fresh_validation_passed")
        and card.get("admission_mode", "strict_zero_fallback") == admission_mode
        and card.get("freeze_identity") == manifest.get("freeze_identity")
        and isinstance(fresh_validation, dict)
        and isinstance(fresh_validation.get("test_id"), str)
        and bool(fresh_validation.get("test_id"))
        and fresh_validation.get("select_count") == 1
        and fresh_validation.get("initialize_count") == 1
        and fresh_validation.get("stop_count") == 1
        and fresh_validation.get("warmup_days") == 7
        and fresh_validation.get("steps") == STEPS_PER_WEEK - 4
        and isinstance(fresh_validation.get("fallback_count"), int)
        and int(fresh_validation["fallback_count"]) >= 0
        and (degraded_admission or fresh_validation.get("fallback_count") == 0)
        and fresh_comfort_target_valid
        and isinstance(card.get("validation"), dict)
        and card["validation"].get("eligible") is (int(fresh_validation["fallback_count"]) == 0)
        and card["validation"].get("comfort_target_met")
        is fresh_validation.get("comfort_target_met")
        and card["validation"].get("model_identity") == model.identity
        and card["validation"].get("test_id") == fresh_validation.get("test_id")
        and isinstance(card.get("robust_calibration_attestation"), dict)
        and card.get("robust_calibration_attestation")
        == manifest.get("robust_calibration_attestation")
    )
    checks = {
        "schemas": card.get("schema") == "h3c_hierarchical_mpc_model_card"
        and manifest.get("schema") == "h3c_hierarchical_mpc_training_manifest"
        and schema_version in {1, 2},
        "case_identity": card.get("case") == case == manifest.get("case"),
        "model_identity": card.get("model_identity")
        == model.identity
        == manifest.get("model_identity")
        and (schema_version == 1 or model.identity == expected_model_identity(model)),
        "source_identity": isinstance(card.get("source_commit"), str)
        and bool(card["source_commit"])
        and card.get("source_commit") == manifest.get("source_commit")
        and (schema_version == 1 or len(card["source_commit"]) == 40),
        "layout": model.layout.zones == zones
        and model.layout.lag_count == 4
        and model.layout.horizon_steps == 4,
        "training_window": card.get("training_week_start_day")
        == int(profile["evaluation_start_day"]) - 7
        and card.get("training_week_days") == 7,
        "control_support": card.get("control_support_bounds_c") == expected_support,
        "pmv_robust_margin": card.get("pmv_robust_margin", 0.0) == model.pmv_robust_margin,
        "lifecycle": lifecycle_valid,
    }
    return {
        "case": case,
        "model_identity": model.identity,
        "checks": checks,
        "valid": all(checks.values()),
    }


def verify_frozen_mpc_model(case: str) -> dict[str, Any]:
    suite_target = repository_root() / "models" / "mpc"
    try:
        result = _verify_mpc_model_directory(case, suite_target / case)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "case": case,
            "checks": {},
            "error": f"{type(error).__name__}: {error}",
            "valid": False,
        }
    try:
        manifest = json.loads(
            (suite_target / case / "training_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return result
    if manifest.get("schema_version") != 2:
        return result

    # Schema 2 is an atomic three-case registry, not a standalone case bundle.
    # Import locally to avoid a module cycle: registry uses the private directory
    # verifier while this public runtime entry point binds the case to its parent.
    from h3c_baselines.mpc.registry import verify_frozen_mpc_suite

    suite = verify_frozen_mpc_suite(suite_target)
    suite_case_checks = suite.get("case_checks")
    suite_case_valid = bool(
        isinstance(suite_case_checks, Mapping) and suite_case_checks.get(case) is True
    )
    checks = {
        **result["checks"],
        "transactional_suite_registry": suite.get("valid") is True and suite_case_valid,
    }
    return {
        **result,
        "admission_mode": manifest.get("admission_mode", "strict_zero_fallback"),
        "validation_classification": manifest.get("validation_classification", "BASELINE-READY"),
        "freeze_identity": manifest.get("freeze_identity"),
        "checks": checks,
        "valid": all(checks.values()),
    }


def train_hierarchical_mpc(
    *,
    endpoint: str,
    workers: int,
    maximum_fit_episodes: int,
    physical_factory: PhysicalFactory = BoptestHttpClient,
) -> dict[str, Any]:
    plan = resolved_training_plan(workers=workers, max_fit_episodes=maximum_fit_episodes)
    config = load_hierarchical_mpc_config()
    source_commit = committed_source_identity()
    root = repository_root() / "outputs" / "baselines" / "mpc"
    final_root = repository_root() / "models" / "mpc"
    existing_targets = [case for case in config["case_order"] if (final_root / case).exists()]
    if existing_targets:
        raise ValueError(f"frozen MPC model targets already exist: {existing_targets}")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + source_commit[:8]
    run_dir = root / "training" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "resolved_plan.json", {**plan, "execution": True, "source_commit": source_commit}
    )
    started = time.monotonic()
    deadline = started + float(config["wall_clock_limit_hours"]) * 3600
    summaries: list[dict[str, Any]] = []
    staged_models: list[tuple[str, Path]] = []
    try:
        with _training_lock(root / ".training.lock"):
            for case in config["case_order"]:
                case_dir = run_dir / case
                case_dir.mkdir()
                lanes: list[_Lane] = []
                selected_model: FittedArxModel | None = None
                summary: dict[str, Any] | None = None
                try:
                    for index in range(workers):
                        client = physical_factory(endpoint)
                        test_id = client.select_testcase(load_profile(case)["testcase"])
                        lanes.append(_Lane(index, client, test_id))
                    selected_model, summary = _fit_case(
                        case=case,
                        lanes=lanes,
                        config=config,
                        output_dir=case_dir,
                        maximum_fit_episodes=maximum_fit_episodes,
                        deadline=deadline,
                    )
                finally:
                    stop_errors: list[str] = []
                    for lane in lanes:
                        try:
                            lane.client.stop()
                            lane.stop_count += 1
                        except Exception as error:
                            stop_errors.append(f"lane{lane.index}:{type(error).__name__}")
                    if stop_errors:
                        raise ValueError(f"BOPTEST lane stop failed: {stop_errors}")
                assert selected_model is not None and summary is not None
                staged = _freeze_model(
                    case=case,
                    model=selected_model,
                    summary=summary,
                    source_commit=source_commit,
                    training_run=case_dir,
                    lanes=lanes,
                    target=case_dir / "frozen_model",
                )
                verification = _verify_mpc_model_directory(case, staged)
                if not verification["valid"]:
                    raise ValueError(f"staged MPC model verification failed for {case}")
                _write_json(case_dir / "model_verification.json", verification)
                summaries.append(summary)
                staged_models.append((case, staged))
        secret_count = secret_occurrences(run_dir)
        if secret_count:
            raise ValueError("secret exposure detected in MPC training outputs")
        final_root.mkdir(parents=True, exist_ok=True)
        for case, staged in staged_models:
            staged.replace(final_root / case)
    except Exception as error:
        secret_count = secret_occurrences(run_dir)
        _write_json(
            run_dir / "failure.json",
            {
                "schema": "h3c_hierarchical_mpc_training_failure",
                "schema_version": 1,
                "source_commit": source_commit,
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "secret_exposure_count": secret_count,
                "completed_cases": summaries,
            },
        )
        raise
    completion = {
        "schema": "h3c_hierarchical_mpc_training_completion",
        "schema_version": 1,
        "source_commit": source_commit,
        "elapsed_seconds": time.monotonic() - started,
        "secret_exposure_count": secret_count,
        "cases": summaries,
    }
    _write_completion(run_dir / "completion.json", completion)
    return {**completion, "run_dir": str(run_dir)}
