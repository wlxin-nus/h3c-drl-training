from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import portalocker

from .boptest import BoptestHttpClient
from .comfort import ComfortModel
from .config import TaskSpec
from .io import append_jsonl, atomic_json, utc_now
from .observation_contract import refined_model_entry
from .observations import PolicyObservationBuilder
from .occupancy import effective_count
from .profiles import load_profile
from .protocol import (
    control_input,
    forecast_points,
    resolve_forecast_missing_occupancy,
    setpoint_from_action,
    site_power,
    zone_temperature_c,
)


class LifecycleRecorder:
    """Persist a selected ID before initialization and clear it only after stop."""

    def __init__(self, run_dir: Path, owner: str, task: str, seed: int):
        self.run_dir = Path(run_dir)
        self.owner = owner
        self.task = task
        self.seed = int(seed)
        self.directory = self.run_dir / "lifecycle"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events = self.directory / f"{owner}.jsonl"
        self.active = self.directory / f"{owner}.active.json"
        self.lock = self.directory / f"{owner}.lock"

    def __call__(self, event: Mapping[str, Any]) -> None:
        row = {
            "timestamp": utc_now(),
            "hostname": os.environ.get("COMPUTERNAME", "unknown"),
            "pid": os.getpid(),
            "owner": self.owner,
            "task": self.task,
            "seed": self.seed,
            **dict(event),
        }
        with portalocker.Lock(str(self.lock), mode="a+", timeout=30):
            append_jsonl(self.events, row)
            test_id = row.get("test_id")
            if row.get("event") == "selected" and test_id:
                atomic_json(self.active, {"test_id": test_id, **row})
            elif row.get("event") == "stopped":
                self.active.unlink(missing_ok=True)


def cleanup_run_testids(run_dir: Path, endpoint: str) -> list[dict[str, Any]]:
    """Stop only IDs owned by this run directory; never touch a concurrent run."""

    from concurrent.futures import ThreadPoolExecutor

    import requests

    directory = Path(run_dir) / "lifecycle"
    active_paths = list(directory.glob("*.active.json")) if directory.exists() else []

    def stop_one(active: Path) -> dict[str, Any]:
        try:
            record = json.loads(active.read_text(encoding="utf-8"))
            test_id = str(record["test_id"])
            response = requests.put(f"{endpoint.rstrip('/')}/stop/{test_id}", timeout=30)
            ok = response.status_code in {200, 400, 404}
            if ok:
                active.unlink(missing_ok=True)
            return {"test_id": test_id, "status_code": response.status_code, "stopped": ok}
        except Exception as exc:
            return {"path": str(active), "stopped": False, "error": repr(exc)}

    if not active_paths:
        return []
    with ThreadPoolExecutor(max_workers=min(10, len(active_paths))) as pool:
        results = list(pool.map(stop_one, active_paths))
    if results:
        for row in results:
            append_jsonl(directory / "startup_cleanup.jsonl", {"timestamp": utc_now(), **row})
    return results


def _training_objective(profile: Mapping[str, Any], spec: TaskSpec) -> dict[str, float]:
    objective = {key: float(value) for key, value in profile["objective"].items()}
    # The MZ-Air archive expressed the identical smoothness coefficient as
    # 0.2 * 0.193466; the packaged case profile stores 0.1 * 0.386932.
    return objective


def training_reward(
    *,
    cost: float,
    pmv: Sequence[float],
    occupancy: Sequence[float],
    setpoints: Sequence[float],
    previous: Sequence[float],
    objective: Mapping[str, float],
    comfort_threshold: float,
) -> tuple[float, dict[str, float]]:
    zones = len(pmv)
    comfort = sum(
        max(0.0, abs(float(value)) - comfort_threshold) ** 2 if float(count) > 0 else 0.0
        for value, count in zip(pmv, occupancy, strict=True)
    )
    smooth = sum(abs(float(now) - float(old)) for now, old in zip(setpoints, previous, strict=True))
    energy_penalty = objective["energy_weight"] * objective["energy_scale"] * cost / zones
    comfort_penalty = objective["comfort_weight"] * objective["comfort_scale"] * comfort / zones
    smooth_penalty = objective["smoothness_weight"] * objective["smoothness_scale"] * smooth / zones
    return -(energy_penalty + comfort_penalty + smooth_penalty), {
        "rew_energy": energy_penalty,
        "rew_comfort": comfort_penalty,
        "rew_smooth": smooth_penalty,
    }


def prefetch_forecast(
    spec: TaskSpec, *, endpoint: str, run_dir: Path, seed: int, evaluation: bool = False
) -> dict[str, list[float]]:
    profile = load_profile(spec.case_key)
    owner = "formal_forecast" if evaluation else "training_forecast"
    recorder = LifecycleRecorder(run_dir, owner, spec.key, seed)
    client = BoptestHttpClient(endpoint)
    client.set_lifecycle_sink(recorder)
    start_day = spec.test_day if evaluation else spec.start_day
    required = spec.episode_steps + 97
    try:
        client.initialize(spec.case_name, start_day * 86400, 0)
        source = client.forecast(forecast_points(profile), required * 900, 900)
        resolved, _events = resolve_forecast_missing_occupancy(
            profile,
            source,
            forecast_points(profile),
            required,
            forecast_phase="evaluation" if evaluation else "training",
            start_time_seconds=start_day * 86400,
            step_seconds=900,
        )
        return {
            key: [float(value) for value in values[:required]] for key, values in resolved.items()
        }
    finally:
        if client.test_id is not None:
            client.stop()


class BoptestTrainingEnv(gym.Env[np.ndarray, np.ndarray]):
    """BOPTEST environment shared by training and deterministic evaluation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        spec: TaskSpec,
        seed: int,
        forecast: Mapping[str, Sequence[float]],
        endpoint: str,
        run_dir: Path,
        rank: int | str,
        phase: str,
        *,
        evaluation: bool = False,
    ):
        super().__init__()
        self.spec = spec
        self.seed_value = int(seed)
        self.forecast = forecast
        self.endpoint = endpoint
        self.run_dir = Path(run_dir)
        self.rank = rank
        self.phase = phase
        self.evaluation = evaluation
        self.profile = load_profile(spec.case_key)
        self.model_contract = refined_model_entry(spec.key)
        self.builder = PolicyObservationBuilder(self.profile, self.model_contract)
        self.comfort = ComfortModel(self.profile["comfort"])
        self.objective = _training_objective(self.profile, spec)
        self.client = BoptestHttpClient(endpoint)
        self.client.set_lifecycle_sink(
            LifecycleRecorder(self.run_dir, f"{phase}_{rank}", spec.key, seed)
        )
        self.zone_order = tuple(str(value) for value in self.model_contract["policy_zone_order"])
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(spec.observation_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(spec.action_dim,), dtype=np.float32
        )
        self.step_index = 0
        self.state: dict[str, Any] = {}
        self.previous_setpoints = dict.fromkeys(self.zone_order, 25.0)
        self.current_policy_pmv = dict.fromkeys(self.zone_order, 0.0)
        self.last_packet = None

    @property
    def local_observations(self) -> dict[str, np.ndarray]:
        if self.last_packet is None:
            raise RuntimeError("Environment has not been reset")
        return self.last_packet.local_normalized

    def _start_seconds(self) -> int:
        return (self.spec.test_day if self.evaluation else self.spec.start_day) * 86400

    def _current_occupancy(self, step: int) -> dict[str, float]:
        action_time = self._start_seconds() + step * 900
        return {
            zone: effective_count(
                self.profile["occupancy"],
                action_time,
                float(self.forecast[self.profile["zones"][zone]["occupancy_forecast"]][step]),
            )
            for zone in self.zone_order
        }

    def _policy_pmv(self, temperatures: Mapping[str, float], step: int) -> dict[str, float]:
        outdoor_name = self.profile["global_inputs"]["outdoor_temperature"]
        values = self.forecast[outdoor_name][step : step + 96]
        mean_c = sum(float(value) - 273.15 for value in values) / len(values)
        self.comfort.update_clothing(self._start_seconds() + step * 900, mean_c)
        return {zone: self.comfort.pmv(temperatures[zone]) for zone in self.zone_order}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.client.test_id is None:
            self.state = self.client.initialize(
                self.spec.case_name, self._start_seconds(), 7 * 86400
            )
        else:
            self.state = self.client.initialize_selected(self._start_seconds(), 7 * 86400)
        self.step_index = 0
        self.previous_setpoints = dict.fromkeys(self.zone_order, 25.0)
        self.current_policy_pmv = dict.fromkeys(self.zone_order, 0.0)
        self.comfort = ComfortModel(self.profile["comfort"])
        self.builder.reset(self.state)
        self.last_packet = self.builder.build(
            self.forecast, step=0, action_time_seconds=self._start_seconds(), step_seconds=900
        )
        return self.last_packet.normalized.copy(), {}

    def step(self, action: np.ndarray):
        if self.last_packet is None:
            raise RuntimeError("Environment must be reset before step")
        policy_packet = self.last_packet
        actions = np.asarray(action, dtype=np.float64).reshape(-1)
        if actions.shape != (self.spec.action_dim,) or np.any(~np.isfinite(actions)):
            raise ValueError("Policy action has the wrong shape or contains non-finite values")
        actions = np.clip(actions, -1.0, 1.0)
        occupancy = self._current_occupancy(self.step_index)
        setpoints: dict[str, float] = {}
        for index, zone in enumerate(self.zone_order):
            setpoints[zone] = setpoint_from_action(
                str(self.model_contract["residual_base"]),
                occupancy[zone],
                float(actions[index]),
            )
        next_state = self.client.advance(control_input(self.profile, setpoints))
        temperatures = {
            zone: zone_temperature_c(self.profile, next_state, zone) for zone in self.zone_order
        }
        pmv = self._policy_pmv(temperatures, self.step_index)
        power = site_power(self.profile, next_state)
        price_name = self.profile["global_inputs"]["electricity_price"]
        price = float(self.forecast[price_name][self.step_index])
        cost = power * 0.25 / 1000.0 * price
        comfort_threshold = self.spec.comfort_threshold
        reward, components = training_reward(
            cost=cost,
            pmv=[pmv[z] for z in self.zone_order],
            occupancy=[occupancy[z] for z in self.zone_order],
            setpoints=[setpoints[z] for z in self.zone_order],
            previous=[self.previous_setpoints[z] for z in self.zone_order],
            objective=self.objective,
            comfort_threshold=comfort_threshold,
        )
        self.builder.update(next_state, setpoints, pmv, power)
        self.step_index += 1
        truncated = self.step_index >= self.spec.episode_steps
        next_step = min(self.step_index, self.spec.episode_steps - 1)
        self.last_packet = self.builder.build(
            self.forecast,
            step=next_step,
            action_time_seconds=self._start_seconds() + next_step * 900,
            step_seconds=900,
        )
        self.state = next_state
        self.previous_setpoints = setpoints
        info = {
            "time": self._start_seconds() + (self.step_index - 1) * 900,
            "actions": actions.tolist(),
            "setpoints": [setpoints[z] for z in self.zone_order],
            "occupancies": [occupancy[z] for z in self.zone_order],
            "temperatures": [temperatures[z] for z in self.zone_order],
            "pmvs": [pmv[z] for z in self.zone_order],
            "power_total": power,
            "phys_cost": cost,
            "price": price,
            **components,
            "raw_observation": policy_packet.raw.tolist(),
            "normalized_observation": policy_packet.normalized.tolist(),
            "observation_columns": list(policy_packet.columns),
        }
        return self.last_packet.normalized.copy(), float(reward), False, truncated, info

    def close(self) -> None:
        try:
            if self.client.test_id is not None:
                self.client.stop()
        finally:
            super().close()


def make_env(
    spec: TaskSpec,
    seed: int,
    forecast: Mapping[str, Sequence[float]],
    endpoint: str,
    run_dir: Path,
    rank: int | str,
    phase: str,
    *,
    evaluation: bool = False,
) -> BoptestTrainingEnv:
    return BoptestTrainingEnv(
        spec, seed, forecast, endpoint, run_dir, rank, phase, evaluation=evaluation
    )
