"""Phase 1.5f reproducible PPO retraining infrastructure.

This module is intentionally shared by the MZ hydronic and MZ air notebooks.
Scientific settings live in :class:`CaseConfig`; runtime services here provide:

* BOPTEST test-id lifecycle persistence and stale-id recovery;
* the approved residual-control environments;
* post-PPO-update epoch commits and train-week model selection;
* atomic, retryable checkpoint bundles with exact resume state;
* local-first diagnostics with fail-open Weights & Biases logging.

The legacy notebooks and their result directories are never modified.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import shutil
import signal
import sys
import time
import traceback
import uuid
import zipfile
from collections import deque
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
import requests
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv


UTC = timezone.utc
HYDRONIC_ZONES = ("nz", "sz")
AIR_ZONES = ("cor", "nor", "sou", "eas", "wes")
AUXILIARY_HYDRONIC_KEYS = frozenset(
    {
        "bms_oveTProHea_u",
        "bms_oveTProCoo_u",
        "bms_oveTSupEmiHeaNz_u",
        "bms_oveTSupEmiHeaSz_u",
        "bms_oveTSupEmiCooNz_u",
        "bms_oveTSupEmiCooSz_u",
    }
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError):
        return float(default)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_ready(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            json_ready(payload),
            handle,
            indent=2,
            sort_keys=True,
            default=json_default,
            allow_nan=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a run-owned CSV atomically.

    The caller must source ``rows`` from checkpoint state. CSV is never used to
    infer model progress.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame = pd.DataFrame(list(rows))
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _hydronic_observation_columns() -> tuple[str, ...]:
    return (
        "obs_sin_time",
        "obs_cos_time",
        "obs_T_nz",
        "obs_T_nz_p1",
        "obs_T_nz_p2",
        "obs_T_nz_p3",
        "obs_T_nz_p4",
        "obs_T_sz",
        "obs_T_sz_p1",
        "obs_T_sz_p2",
        "obs_T_sz_p3",
        "obs_T_sz_p4",
        "obs_pmv_nz",
        "obs_pmv_sz",
        "obs_act_nz_p1",
        "obs_act_sz_p1",
        "obs_pow_p1",
        "obs_pow_p2",
        "obs_pow_p3",
        "obs_pow_p4",
        "obs_TDryBul_0",
        "obs_TDryBul_f1",
        "obs_TDryBul_f2",
        "obs_TDryBul_f3",
        "obs_TDryBul_f4",
        "obs_HGloHor_0",
        "obs_HGloHor_f1",
        "obs_HGloHor_f2",
        "obs_HGloHor_f3",
        "obs_HGloHor_f4",
        "obs_Price_0",
        "obs_Price_f1",
        "obs_Price_f2",
        "obs_Price_f3",
        "obs_Price_f4",
        "obs_Occ_nz_0",
        "obs_Occ_nz_f1",
        "obs_Occ_nz_f2",
        "obs_Occ_nz_f3",
        "obs_Occ_nz_f4",
        "obs_Occ_sz_0",
        "obs_Occ_sz_f1",
        "obs_Occ_sz_f2",
        "obs_Occ_sz_f3",
        "obs_Occ_sz_f4",
    )


def _air_observation_columns() -> tuple[str, ...]:
    columns: list[str] = ["obs_sin_time", "obs_cos_time"]
    for zone in AIR_ZONES:
        columns.extend([f"obs_T_{zone}", *(f"obs_T_{zone}_p{i}" for i in range(1, 5))])
    columns.extend(f"obs_pmv_{zone}" for zone in AIR_ZONES)
    columns.extend(f"obs_act_{zone}_p1" for zone in AIR_ZONES)
    columns.extend(f"obs_power_norm_p{i}" for i in range(1, 5))
    for name in ("TDryBul", "HGloHor", "Price"):
        columns.extend(f"obs_{name}_{i}" for i in range(5))
    for zone in AIR_ZONES:
        columns.extend(f"obs_OccMask_{zone}_{i}" for i in range(5))
    return tuple(columns)


@dataclasses.dataclass
class CaseConfig:
    case_key: str
    case_name: str
    project_root: Path
    case_dir: Path
    output_root: Path
    legacy_model_path: Path
    validation_filename: str
    run_mode: str = "smoke"
    resume: bool = True
    wandb_mode: str = "online"
    run_tag: str = "seed42"
    boptest_url: str = "http://127.0.0.1:80"
    start_day: int = 0
    simulation_days: int = 7
    control_period: int = 900
    rbc_collect_days: int = 14
    n_zones: int = 1
    zones: tuple[str, ...] = ()
    max_system_power: float = 1.0
    observation_columns: tuple[str, ...] = ()
    forecast_points: tuple[str, ...] = ()
    observations_config: dict[str, tuple[float, float]] = dataclasses.field(default_factory=dict)
    reward_weights: dict[str, float] = dataclasses.field(default_factory=dict)
    physical_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    learning_rate: float = 3e-4
    final_lr_fraction: float = 0.1
    n_steps: int = 1
    batch_size: int = 1
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    policy_net: tuple[int, int] = (256, 256)
    seed: int = 42
    residual_scale: float = 5.0
    occupied_base_k: float = 298.15
    unoccupied_base_k: float = 303.15
    action_min_k: float = 293.15
    action_max_k: float = 303.15
    comfort_threshold: float = 0.5
    num_envs: int = 4
    max_full_epochs: int = 500
    smoke_epochs: int = 2
    min_early_stop_epoch: int = 120
    train_eval_interval: int = 25
    wandb_project: str = "drl-phase1-5f-retraining"
    wandb_group: str = "phase1_5f_v2_seed42"

    def __post_init__(self) -> None:
        for field_name in ("project_root", "case_dir", "output_root", "legacy_model_path"):
            setattr(self, field_name, Path(getattr(self, field_name)).resolve())

    @property
    def max_epochs(self) -> int:
        return self.smoke_epochs if self.run_mode == "smoke" else self.max_full_epochs

    @property
    def eval_interval(self) -> int:
        return 1 if self.run_mode == "smoke" else self.train_eval_interval

    @property
    def steps_per_epoch(self) -> int:
        return self.n_steps * self.num_envs

    @property
    def total_cap_steps(self) -> int:
        return self.max_epochs * self.steps_per_epoch

    @property
    def episode_steps(self) -> int:
        return int(self.simulation_days * 24 * 3600 / self.control_period)

    @property
    def run_dir(self) -> Path:
        return self.output_root / f"{self.run_mode}-{self.run_tag}"

    @property
    def scientific_payload(self) -> dict[str, Any]:
        return {
            "schema": "phase1.5f-v2",
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
            "pmv_implementation": "pythermalcomfort-3.8.0-pmv_ppd_iso-scalar-equivalent",
            "case_key": self.case_key,
            "case_name": self.case_name,
            "start_day": self.start_day,
            "simulation_days": self.simulation_days,
            "control_period": self.control_period,
            "n_zones": self.n_zones,
            "zones": self.zones,
            "max_system_power": self.max_system_power,
            "observation_columns": self.observation_columns,
            "forecast_points": self.forecast_points,
            "observations_config": self.observations_config,
            "reward_weights": self.reward_weights,
            "physical_config": self.physical_config,
            "learning_rate": self.learning_rate,
            "final_lr_fraction": self.final_lr_fraction,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "policy_net": self.policy_net,
            "seed": self.seed,
            "residual_scale": self.residual_scale,
            "occupied_base_k": self.occupied_base_k,
            "unoccupied_base_k": self.unoccupied_base_k,
            "action_bounds_k": (self.action_min_k, self.action_max_k),
            "comfort_threshold": self.comfort_threshold,
            "num_envs": self.num_envs,
            "max_epochs": self.max_epochs,
        }

    @property
    def config_hash(self) -> str:
        return sha256_payload(self.scientific_payload)


def build_hydronic_config(
    project_root: str | Path,
    run_mode: str = "smoke",
    resume: bool = True,
    wandb_mode: str = "online",
    run_tag: str = "seed42",
) -> CaseConfig:
    root = Path(project_root).resolve()
    case_dir = root / "CASE_TEST" / "MZ_OFFICE_HYDRONIC"
    return CaseConfig(
        case_key="mz_hydro",
        case_name="multizone_office_simple_hydronic",
        project_root=root,
        case_dir=case_dir,
        output_root=case_dir / "phase1_5f_ppo_v2",
        legacy_model_path=case_dir
        / "multizone_office_simple_hydronic_residual_opt"
        / "models_drl"
        / "best_model_ppo.zip",
        validation_filename="ppo_validation_hydronic_2zone.csv",
        run_mode=run_mode,
        resume=resume,
        wandb_mode=wandb_mode,
        run_tag=run_tag,
        start_day=213,
        simulation_days=5,
        n_zones=2,
        zones=HYDRONIC_ZONES,
        max_system_power=35000.0,
        observation_columns=_hydronic_observation_columns(),
        forecast_points=(
            "Occupancy[nZ]",
            "Occupancy[sZ]",
            "PriceElectricPowerDynamic",
            "TDryBul",
            "HGloHor",
        ),
        observations_config={
            "T": (288.15, 308.15),
            "pow": (0.0, 1.0),
            "pmv": (-3.0, 3.0),
            "TDryBul": (263.15, 313.15),
            "HGloHor": (0.0, 1200.0),
            "Price": (0.0, 0.2),
            "Occ": (0.0, 200.0),
        },
        reward_weights={
            "w_energy": 1.0,
            "w_comfort": 20.0,
            "w_smooth": 0.1,
            "scaler_energy": 3.0,
            "scaler_comfort": 10.0,
            "scaler_smooth": 0.1755995,
        },
        physical_config={
            "metabolic_rate": 1.1,
            "relative_humidity": 50.0,
            "air_velocity": 0.1,
            "clo_dynamic_params": {
                "winter_clo": 1.0,
                "summer_clo": 0.5,
                "temp_low": 10.0,
                "temp_high": 26.0,
            },
        },
        n_steps=480,
        batch_size=120,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
    )


def build_air_config(
    project_root: str | Path,
    run_mode: str = "smoke",
    resume: bool = True,
    wandb_mode: str = "online",
    run_tag: str = "seed42",
) -> CaseConfig:
    root = Path(project_root).resolve()
    case_dir = root / "CASE_TEST" / "MZ_OFFICE_AIR"
    return CaseConfig(
        case_key="mz_air",
        case_name="multizone_office_simple_air",
        project_root=root,
        case_dir=case_dir,
        output_root=case_dir / "phase1_5f_ppo_v2",
        legacy_model_path=case_dir
        / "multizone_office_simple_air_residual_transfer"
        / "models_drl"
        / "bake"
        / "best_model_ppo.zip",
        validation_filename="ppo_validation_air_5zone.csv",
        run_mode=run_mode,
        resume=resume,
        wandb_mode=wandb_mode,
        run_tag=run_tag,
        start_day=192,
        simulation_days=7,
        n_zones=5,
        zones=AIR_ZONES,
        max_system_power=40000.0,
        observation_columns=_air_observation_columns(),
        forecast_points=(
            "Occupancy[cor]",
            "Occupancy[nor]",
            "Occupancy[sou]",
            "Occupancy[eas]",
            "Occupancy[wes]",
            "PriceElectricPowerDynamic",
            "TDryBul",
            "HGloHor",
        ),
        observations_config={
            "T": (288.15, 308.15),
            "pow": (0.0, 1.0),
            "pmv": (-3.0, 3.0),
            "TDryBul": (263.15, 313.15),
            "HGloHor": (0.0, 1200.0),
            "Price": (0.0, 0.2),
            "Occ": (0.0, 50.0),
            "act": (293.15, 303.15),
        },
        reward_weights={
            "w_energy": 1.0,
            "w_comfort": 20.0,
            "w_smooth": 0.2,
            "scaler_energy": 18.532822,
            "scaler_comfort": 10.0,
            "scaler_smooth": 0.193466,
        },
        physical_config={
            "metabolic_rate": 1.1,
            "relative_humidity": 50.0,
            "air_velocity": 0.1,
            "clo_dynamic_params": {
                "winter_clo": 1.0,
                "summer_clo": 0.5,
                "temp_low": 10.0,
                "temp_high": 26.0,
            },
        },
        n_steps=672,
        batch_size=168,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
    )


def residual_setpoints(
    actions: Sequence[float],
    occupancies: Sequence[float],
    config: CaseConfig,
    legacy_fixed_baseline: bool = False,
) -> np.ndarray:
    action_array = np.asarray(actions, dtype=float)
    occupancy_array = np.asarray(occupancies, dtype=float)
    if action_array.shape != occupancy_array.shape:
        raise ValueError("Actions and occupancies must have identical shapes")
    if legacy_fixed_baseline:
        bases = np.full_like(action_array, config.occupied_base_k)
    else:
        bases = np.where(occupancy_array > 0.0, config.occupied_base_k, config.unoccupied_base_k)
    return np.clip(
        bases + action_array * config.residual_scale,
        config.action_min_k,
        config.action_max_k,
    )


def comfort_excess_squared(pmv: float, occupied: float, threshold: float = 0.5) -> float:
    return max(0.0, abs(safe_float(pmv)) - threshold) ** 2 if occupied > 0 else 0.0


def get_dynamic_clo(daily_mean_tout_c: float, params: Mapping[str, float]) -> float:
    temperature = safe_float(daily_mean_tout_c, 20.0)
    if temperature < params["temp_low"]:
        return float(params["winter_clo"])
    if temperature > params["temp_high"]:
        return float(params["summer_clo"])
    ratio = (temperature - params["temp_low"]) / (params["temp_high"] - params["temp_low"])
    return float(params["winter_clo"] + ratio * (params["summer_clo"] - params["winter_clo"]))


def calculate_pmv_state(t_air_k: float, physical: Mapping[str, Any], clo: float) -> float:
    """Scalar equivalent of pythermalcomfort 3.8 ``pmv_ppd_iso`` defaults.

    The upstream package's ``models`` import can block on a Numba cache lock
    when four Windows workers start together. This preserves the same Fanger
    equation, ISO input limits, PMV validity limit and two-decimal output while
    avoiding any runtime import or JIT compilation.
    """

    temperature_c = safe_float(t_air_k, 298.15) - 273.15
    radiant_c = temperature_c
    velocity = safe_float(physical.get("air_velocity"), 0.1)
    humidity = safe_float(physical.get("relative_humidity"), 50.0)
    metabolic = safe_float(physical.get("metabolic_rate"), 1.1)
    clothing = safe_float(clo, 0.5)
    if not (
        10.0 <= temperature_c <= 30.0
        and 10.0 <= radiant_c <= 40.0
        and 0.0 <= velocity <= 1.0
        and 0.8 <= metabolic <= 4.0
        and 0.0 <= clothing <= 2.0
    ):
        return 0.0
    try:
        vapor_pressure = humidity * 10.0 * math.exp(
            16.6536 - 4030.183 / (temperature_c + 235.0)
        )
        clothing_insulation = 0.155 * clothing
        metabolic_w = metabolic * 58.15
        internal_heat = metabolic_w
        clothing_area = (
            1.0 + 1.29 * clothing_insulation
            if clothing_insulation <= 0.078
            else 1.05 + 0.645 * clothing_insulation
        )
        forced_convection = 12.1 * math.sqrt(velocity)
        air_absolute = temperature_c + 273.0
        radiant_absolute = radiant_c + 273.0
        clothing_surface = air_absolute + (35.5 - temperature_c) / (
            3.5 * clothing_insulation + 0.1
        )
        p1 = clothing_insulation * clothing_area
        p2 = p1 * 3.96
        p3 = p1 * 100.0
        p4 = p1 * air_absolute
        p5 = (308.7 - 0.028 * internal_heat) + p2 * (radiant_absolute / 100.0) ** 4
        xn = clothing_surface / 100.0
        xf = clothing_surface / 50.0
        heat_transfer = forced_convection
        iterations = 0
        while abs(xn - xf) > 0.00015:
            xf = (xf + xn) / 2.0
            natural_convection = 2.38 * abs(100.0 * xf - air_absolute) ** 0.25
            heat_transfer = max(natural_convection, forced_convection)
            xn = (p5 + p4 * heat_transfer - p2 * xf**4) / (
                100.0 + p3 * heat_transfer
            )
            iterations += 1
            if iterations > 150:
                return 0.0
        clothing_temperature = 100.0 * xn - 273.0
        heat_loss_skin = 3.05 * 0.001 * (
            5733.0 - 6.99 * internal_heat - vapor_pressure
        )
        heat_loss_sweat = (
            0.42 * (internal_heat - 58.15) if internal_heat > 58.15 else 0.0
        )
        heat_loss_latent = (
            1.7 * 0.00001 * metabolic_w * (5867.0 - vapor_pressure)
        )
        heat_loss_dry = 0.0014 * metabolic_w * (34.0 - temperature_c)
        heat_loss_radiation = 3.96 * clothing_area * (
            xn**4 - (radiant_absolute / 100.0) ** 4
        )
        heat_loss_convection = (
            clothing_area * heat_transfer * (clothing_temperature - temperature_c)
        )
        transfer_coefficient = 0.303 * math.exp(-0.036 * metabolic_w) + 0.028
        pmv = transfer_coefficient * (
            internal_heat
            - heat_loss_skin
            - heat_loss_sweat
            - heat_loss_latent
            - heat_loss_dry
            - heat_loss_radiation
            - heat_loss_convection
        )
        if not -2.0 <= pmv <= 2.0:
            return 0.0
        return float(np.round(pmv, 2))
    except Exception:
        return 0.0


def request_json(
    method: str,
    url: str,
    *,
    max_retries: int = 5,
    timeout: float = 90.0,
    allow_stopped: bool = False,
    session: Any = requests,
    **kwargs: Any,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            if response.status_code == 200:
                if not response.content:
                    return {"status": "success"}
                return response.json()
            if allow_stopped and response.status_code in (400, 404):
                return {"status": "already_stopped", "http_status": response.status_code}
            raise RuntimeError(f"HTTP {response.status_code} from {url}: {response.text[:500]}")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Request failed after {max_retries} attempts: {method} {url}") from last_error


def _safe_owner(owner: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in owner)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class TestIDLifecycle:
    """Persistent lifecycle for one test-id owner.

    Each worker owns separate files, avoiding cross-process writes on Windows.
    """

    __test__ = False

    def __init__(
        self,
        config: CaseConfig,
        owner: str,
        algorithm: str,
        phase: str,
        rank: int | str,
        session: Any = requests,
    ):
        self.config = config
        self.owner = _safe_owner(owner)
        self.algorithm = algorithm
        self.phase = phase
        self.rank = rank
        self.session = session
        self.testid: str | None = None
        self.epoch: int | None = None
        self.parts_dir = config.run_dir / "lifecycle_parts"
        self.runtime_dir = config.run_dir / "runtime"
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.parts_dir / f"{self.owner}.jsonl"
        self.active_path = self.runtime_dir / f"active_{self.owner}.json"

    def _event(self, event: str, **extra: Any) -> None:
        row = {
            "timestamp": utc_now(),
            "algorithm": self.algorithm,
            "rank": str(self.rank),
            "testid": self.testid,
            "event": event,
            "phase": self.phase,
            "epoch": self.epoch,
            "owner": self.owner,
            **extra,
        }
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, default=json_default, sort_keys=True) + "\n")

    def _write_active(self) -> None:
        if self.testid is None:
            return
        atomic_write_json(
            self.active_path,
            {
                "testid": self.testid,
                "pid": os.getpid(),
                "owner": self.owner,
                "algorithm": self.algorithm,
                "phase": self.phase,
                "rank": str(self.rank),
                "epoch": self.epoch,
                "heartbeat": utc_now(),
                "boptest_url": self.config.boptest_url,
            },
        )

    def select(self) -> str:
        if self.testid:
            return self.testid
        response = request_json(
            "post",
            f"{self.config.boptest_url}/testcases/{self.config.case_name}/select",
            session=self.session,
        )
        self.testid = str(response["testid"])
        self._event("selected")
        self._write_active()
        return self.testid

    def configure(self) -> None:
        testid = self.select()
        request_json(
            "put",
            f"{self.config.boptest_url}/step/{testid}",
            json={"step": self.config.control_period},
            session=self.session,
        )
        request_json(
            "put",
            f"{self.config.boptest_url}/scenario/{testid}",
            json={"electricity_price": "dynamic"},
            session=self.session,
        )

    def initialize(self, start_time: int, warmup_period: int) -> dict[str, Any]:
        testid = self.select()
        # Air uses a seven-day simulator warm-up.  Four such initializations
        # may be serialized by the BOPTEST host and legitimately exceed the
        # generic request timeout.  Initialization is stateful, so never
        # blindly retry a timed-out PUT: allow one bounded long request and
        # leave the persisted ID available for deterministic cleanup on error.
        response = request_json(
            "put",
            f"{self.config.boptest_url}/initialize/{testid}",
            json={"start_time": int(start_time), "warmup_period": int(warmup_period)},
            max_retries=1,
            timeout=1200.0,
            session=self.session,
        )
        self._event("initialized", start_time=int(start_time), warmup_period=int(warmup_period))
        self._write_active()
        return response

    def heartbeat(self, epoch: int | None = None) -> None:
        self.epoch = epoch
        self._write_active()

    def stop(self, reason: str = "normal") -> None:
        if not self.testid:
            return
        testid = self.testid
        try:
            response = request_json(
                "put",
                f"{self.config.boptest_url}/stop/{testid}",
                allow_stopped=True,
                max_retries=3,
                timeout=30,
                session=self.session,
            )
            event = "already_stopped" if response.get("status") == "already_stopped" else "stopped"
            self._event(event, reason=reason)
            try:
                self.active_path.unlink(missing_ok=True)
            finally:
                self.testid = None
        except Exception as exc:
            self._event("stop_failed", reason=reason, error=repr(exc))
            raise


def cleanup_stale_testids(config: CaseConfig, force: bool = False, session: Any = requests) -> list[dict[str, Any]]:
    """Stop dead-owner ids from this v2 run only.

    A live local PID is never stopped unless ``force=True``.
    """

    runtime = config.run_dir / "runtime"
    results: list[dict[str, Any]] = []
    if not runtime.exists():
        return results
    for active_path in sorted(runtime.glob("active_*.json")):
        payload: dict[str, Any] | None = None
        try:
            payload = json.loads(active_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", -1))
            if not force and _pid_is_alive(pid):
                results.append({"path": str(active_path), "status": "live_owner_skipped", "pid": pid})
                continue
            testid = str(payload["testid"])
            base_url = str(payload.get("boptest_url", config.boptest_url))
            response = request_json(
                "put",
                f"{base_url}/stop/{testid}",
                allow_stopped=True,
                max_retries=3,
                timeout=30,
                session=session,
            )
            recovery_event = {
                "timestamp": utc_now(),
                "algorithm": payload.get("algorithm"),
                "rank": str(payload.get("rank", "")),
                "testid": testid,
                "event": "startup_cleanup_stopped",
                "phase": payload.get("phase"),
                "epoch": payload.get("epoch"),
                "owner": payload.get("owner"),
                "reason": "dead_or_forced_owner_recovery",
                "stop_status": response.get("status", "stopped"),
                "previous_pid": pid,
            }
            recovery_log = (
                config.run_dir
                / "lifecycle_parts"
                / f"{_safe_owner(str(payload.get('owner', 'recovered')))}.jsonl"
            )
            recovery_log.parent.mkdir(parents=True, exist_ok=True)
            with recovery_log.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(recovery_event, default=json_default, sort_keys=True)
                    + "\n"
                )
            active_path.unlink(missing_ok=True)
            results.append({"testid": testid, "status": response.get("status", "stopped")})
        except Exception as exc:
            if payload:
                try:
                    recovery_log = (
                        config.run_dir
                        / "lifecycle_parts"
                        / f"{_safe_owner(str(payload.get('owner', 'recovered')))}.jsonl"
                    )
                    recovery_log.parent.mkdir(parents=True, exist_ok=True)
                    recovery_event = {
                        "timestamp": utc_now(),
                        "algorithm": payload.get("algorithm"),
                        "rank": str(payload.get("rank", "")),
                        "testid": payload.get("testid"),
                        "event": "startup_cleanup_failed",
                        "phase": payload.get("phase"),
                        "epoch": payload.get("epoch"),
                        "owner": payload.get("owner"),
                        "reason": "dead_or_forced_owner_recovery",
                        "error": repr(exc),
                    }
                    with recovery_log.open(
                        "a", encoding="utf-8", newline="\n"
                    ) as handle:
                        handle.write(
                            json.dumps(
                                recovery_event,
                                default=json_default,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                except Exception:
                    pass
            results.append({"path": str(active_path), "status": "cleanup_failed", "error": repr(exc)})
    return results


def combine_lifecycle_logs(run_dir: Path) -> Path:
    output = run_dir / "testid_lifecycle.jsonl"
    events: list[dict[str, Any]] = []
    for path in sorted((run_dir / "lifecycle_parts").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    events.sort(key=lambda row: (row.get("timestamp", ""), row.get("owner", "")))
    temp = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, default=json_default) + "\n")
    os.replace(temp, output)
    return output


class ForecastProvider:
    def __init__(self, config: CaseConfig, data: Mapping[str, Sequence[float]] | None = None):
        self.config = config
        self.data = {key: np.asarray(value, dtype=float) for key, value in (data or {}).items()}
        self.start_time = config.start_day * 24 * 3600
        self.total_horizon = (config.rbc_collect_days + 1) * 24 * 3600

    def prefetch_all(self) -> None:
        lifecycle = TestIDLifecycle(
            self.config,
            owner=f"{self.config.case_key}_forecast",
            algorithm=f"{self.config.case_key}_ppo_v2",
            phase="forecast",
            rank="forecast",
        )
        try:
            lifecycle.configure()
            response = lifecycle.initialize(self.start_time, 0)
            if "payload" not in response:
                raise RuntimeError("BOPTEST initialize returned no payload")
            forecast = request_json(
                "put",
                f"{self.config.boptest_url}/forecast/{lifecycle.testid}",
                json={
                    "point_names": list(self.config.forecast_points),
                    "horizon": self.total_horizon,
                    "interval": self.config.control_period,
                },
                timeout=120,
            )
            payload = forecast.get("payload")
            if not isinstance(payload, Mapping):
                raise RuntimeError("BOPTEST forecast returned no payload")
            self.data = {
                key: np.asarray([0.0 if item is None else item for item in values], dtype=float)
                for key, values in payload.items()
            }
            missing = sorted(set(self.config.forecast_points) - set(self.data))
            if missing:
                raise RuntimeError(f"Forecast missing points: {missing}")
        finally:
            try:
                lifecycle.stop("forecast_complete")
            except Exception:
                pass

    def get_local_forecast(self, current_time: float, point_name: str, horizon_steps: int = 5) -> np.ndarray:
        index = int(max(0, (current_time - self.start_time) // self.config.control_period))
        values = self.data.get(point_name)
        if values is None:
            return np.zeros(horizon_steps, dtype=float)
        result = values[index : index + horizon_steps]
        if result.size < horizon_steps:
            fill = result[-1] if result.size else 0.0
            result = np.pad(result, (0, horizon_steps - result.size), constant_values=fill)
        return np.asarray(result, dtype=float)


class NormalizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.low_b = np.asarray(env.observation_space.low, dtype=np.float32)
        self.high_b = np.asarray(env.observation_space.high, dtype=np.float32)
        self.range_b = self.high_b - self.low_b
        self.range_b[self.range_b == 0] = 1.0
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(observation, nan=0.0)
        return np.nan_to_num(2.0 * (clean - self.low_b) / self.range_b - 1.0, nan=0.0).astype(np.float32)


def hydronic_action_payload(setpoints_k: Sequence[float]) -> dict[str, float | int]:
    """The approved aux-overwrite-disabled hydronic payload."""

    north, south = (float(value) for value in setpoints_k)
    return {
        "bms_oveTZonSetMaxNz_u": north,
        "bms_oveTZonSetMaxNz_activate": 1,
        "bms_oveTZonSetMaxSz_u": south,
        "bms_oveTZonSetMaxSz_activate": 1,
        "bms_oveTZonSetMinNz_u": 288.15,
        "bms_oveTZonSetMinNz_activate": 1,
        "bms_oveTZonSetMinSz_u": 288.15,
        "bms_oveTZonSetMinSz_activate": 1,
    }


def air_action_payload(zones: Sequence[str], setpoints_k: Sequence[float]) -> dict[str, float | int]:
    payload: dict[str, float | int] = {
        "hvac_oveAhu_TSupSet_u": 290.15,
        "hvac_oveAhu_TSupSet_activate": 1,
    }
    for zone, setpoint in zip(zones, setpoints_k):
        title = zone.capitalize()
        payload[f"hvac_oveZonSup{title}_TZonCooSet_u"] = float(setpoint)
        payload[f"hvac_oveZonSup{title}_TZonCooSet_activate"] = 1
        payload[f"hvac_oveZonSup{title}_TZonHeaSet_u"] = 288.15
        payload[f"hvac_oveZonSup{title}_TZonHeaSet_activate"] = 1
    return payload


class HydronicResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: CaseConfig,
        forecast_data: Mapping[str, Sequence[float]],
        *,
        is_validation: bool = False,
        legacy_fixed_baseline: bool = False,
        rank: int | str = 0,
        phase: str = "training",
        capture_series: bool = False,
    ):
        super().__init__()
        self.config = config
        self.fp = ForecastProvider(config, forecast_data)
        self.is_validation = is_validation
        self.legacy_fixed_baseline = legacy_fixed_baseline
        self.rank = rank
        self.phase = phase
        self.capture_series = capture_series
        self.start_time = (config.start_day + (7 if is_validation else 0)) * 24 * 3600
        self.max_episode_steps = config.episode_steps
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        low, high = self._observation_bounds()
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.lifecycle = TestIDLifecycle(
            config,
            owner=f"{config.case_key}_{phase}_{rank}",
            algorithm=f"{config.case_key}_ppo_v2",
            phase=phase,
            rank=rank,
        )
        self.history_temp_nz: deque[float] = deque(maxlen=5)
        self.history_temp_sz: deque[float] = deque(maxlen=5)
        self.history_act_nz: deque[float] = deque(maxlen=5)
        self.history_act_sz: deque[float] = deque(maxlen=5)
        self.history_power: deque[float] = deque(maxlen=5)
        self.current_pmv_nz = 0.0
        self.current_pmv_sz = 0.0
        self.current_clo = 0.5
        self.current_step = 0
        self.episode_index = 0
        self.last_raw_observation = np.zeros(len(config.observation_columns), dtype=np.float32)

    def _observation_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lows: list[float] = []
        highs: list[float] = []
        for column in self.config.observation_columns:
            base = column.replace("obs_", "")
            if "Price" in base:
                key = "Price"
            elif "Occ" in base:
                key = "Occ"
            elif "HGloHor" in base:
                key = "HGloHor"
            elif "TDryBul" in base:
                key = "TDryBul"
            elif "pmv" in base:
                key = "pmv"
            elif "pow" in base:
                key = "pow"
            elif "T_" in base or "act_" in base:
                key = "T"
            else:
                key = base.split("_past")[0].split("_future")[0].split("_current")[0]
            bounds = self.config.observations_config.get(key, (0.0, 1.0))
            lows.append(bounds[0])
            highs.append(bounds[1])
        return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)

    @staticmethod
    def _history(buffer: deque[float], steps_back: int, default: float = 298.15) -> float:
        index = -(1 + steps_back)
        return float(buffer[index]) if len(buffer) >= 1 + steps_back else float(default)

    def _get_observations(self, payload: Mapping[str, Any]) -> np.ndarray:
        current_time = safe_float(payload.get("time"), self.start_time)
        raw: dict[str, float] = {
            "obs_sin_time": math.sin(2 * math.pi * (current_time % 86400) / 86400),
            "obs_cos_time": math.cos(2 * math.pi * (current_time % 86400) / 86400),
            "obs_T_nz": safe_float(payload.get("structure_reaTZonNz_y"), 298.15),
            "obs_T_nz_p1": self._history(self.history_temp_nz, 1),
            "obs_T_nz_p2": self._history(self.history_temp_nz, 2),
            "obs_T_nz_p3": self._history(self.history_temp_nz, 3),
            "obs_T_nz_p4": self._history(self.history_temp_nz, 4),
            "obs_T_sz": safe_float(payload.get("structure_reaTZonSz_y"), 298.15),
            "obs_T_sz_p1": self._history(self.history_temp_sz, 1),
            "obs_T_sz_p2": self._history(self.history_temp_sz, 2),
            "obs_T_sz_p3": self._history(self.history_temp_sz, 3),
            "obs_T_sz_p4": self._history(self.history_temp_sz, 4),
            "obs_pmv_nz": self.current_pmv_nz,
            "obs_pmv_sz": self.current_pmv_sz,
            "obs_act_nz_p1": self._history(self.history_act_nz, 1),
            "obs_act_sz_p1": self._history(self.history_act_sz, 1),
            "obs_pow_p1": self._history(self.history_power, 1, 0.0) / self.config.max_system_power,
            "obs_pow_p2": self._history(self.history_power, 2, 0.0) / self.config.max_system_power,
            "obs_pow_p3": self._history(self.history_power, 3, 0.0) / self.config.max_system_power,
            "obs_pow_p4": self._history(self.history_power, 4, 0.0) / self.config.max_system_power,
        }
        forecast_map = {
            "Occupancy[nZ]": "obs_Occ_nz",
            "Occupancy[sZ]": "obs_Occ_sz",
            "PriceElectricPowerDynamic": "obs_Price",
            "TDryBul": "obs_TDryBul",
            "HGloHor": "obs_HGloHor",
        }
        for point, observation_name in forecast_map.items():
            values = self.fp.get_local_forecast(current_time, point, 5)
            if point == "TDryBul" and float(np.mean(values)) < 100.0:
                values = values + 273.15
            raw[f"{observation_name}_0"] = float(values[0])
            for index in range(1, 5):
                raw[f"{observation_name}_f{index}"] = float(values[index])
        return np.asarray([raw.get(column, 0.0) for column in self.config.observation_columns], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.lifecycle.testid is None:
            self.lifecycle.configure()
        self.lifecycle.epoch = self.episode_index + 1
        response = self.lifecycle.initialize(self.start_time, 7 * 24 * 3600)
        payload = response["payload"]
        self.current_step = 0
        self.current_clo = 0.5  # Preserve the executed v1 hydronic clo convention.
        self.current_pmv_nz = 0.0
        self.current_pmv_sz = 0.0
        self.history_temp_nz.clear()
        self.history_temp_sz.clear()
        self.history_act_nz.clear()
        self.history_act_sz.clear()
        self.history_power.clear()
        self.history_temp_nz.append(safe_float(payload.get("structure_reaTZonNz_y"), 298.15))
        self.history_temp_sz.append(safe_float(payload.get("structure_reaTZonSz_y"), 298.15))
        self.history_act_nz.append(298.15)
        self.history_act_sz.append(298.15)
        self.history_power.append(0.0)
        self.episode_index += 1
        observation = self._get_observations(payload)
        self.last_raw_observation = observation.copy()
        return observation, {}

    def step(self, action: np.ndarray):
        current_time = self.start_time + self.current_step * self.config.control_period
        occupancies = np.asarray(
            [
                self.fp.get_local_forecast(current_time, "Occupancy[nZ]", 1)[0],
                self.fp.get_local_forecast(current_time, "Occupancy[sZ]", 1)[0],
            ],
            dtype=float,
        )
        actions = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        setpoints = residual_setpoints(
            actions,
            occupancies,
            self.config,
            legacy_fixed_baseline=self.legacy_fixed_baseline,
        )
        control_payload = hydronic_action_payload(setpoints)
        if AUXILIARY_HYDRONIC_KEYS.intersection(control_payload):
            raise AssertionError("Approved hydronic payload must not contain auxiliary overwrites")
        response = request_json(
            "post",
            f"{self.config.boptest_url}/advance/{self.lifecycle.testid}",
            json=control_payload,
        )
        next_payload = response["payload"]
        total_power = sum(
            safe_float(next_payload.get(name))
            for name in (
                "heating_cooling_reaPProCoo_y",
                "heating_cooling_reaPProHea_y",
                "heating_cooling_reaPFcuNz_y",
                "heating_cooling_reaPFcuSz_y",
            )
        )
        price = safe_float(self.fp.get_local_forecast(current_time, "PriceElectricPowerDynamic", 1)[0])
        cost = total_power * self.config.control_period / 3_600_000.0 * price
        temperature_nz = safe_float(next_payload.get("structure_reaTZonNz_y"), 298.15)
        temperature_sz = safe_float(next_payload.get("structure_reaTZonSz_y"), 298.15)
        self.current_pmv_nz = calculate_pmv_state(
            temperature_nz, self.config.physical_config, self.current_clo
        )
        self.current_pmv_sz = calculate_pmv_state(
            temperature_sz, self.config.physical_config, self.current_clo
        )
        penalties = np.asarray(
            [
                comfort_excess_squared(
                    self.current_pmv_nz, occupancies[0], self.config.comfort_threshold
                ),
                comfort_excess_squared(
                    self.current_pmv_sz, occupancies[1], self.config.comfort_threshold
                ),
            ]
        )
        action_changes = np.asarray(
            [
                abs(setpoints[0] - self.history_act_nz[-1]),
                abs(setpoints[1] - self.history_act_sz[-1]),
            ]
        )
        weights = self.config.reward_weights
        reward_energy = weights["w_energy"] * (cost / self.config.n_zones) * weights["scaler_energy"]
        reward_comfort = (
            weights["w_comfort"]
            * float(np.sum(penalties) / self.config.n_zones)
            * weights["scaler_comfort"]
        )
        reward_smooth = (
            weights["w_smooth"]
            * float(np.sum(action_changes) / self.config.n_zones)
            * weights["scaler_smooth"]
        )
        reward = -(reward_energy + reward_comfort + reward_smooth)
        previous_observation = self.last_raw_observation.copy()
        self.history_temp_nz.append(temperature_nz)
        self.history_temp_sz.append(temperature_sz)
        self.history_act_nz.append(float(setpoints[0]))
        self.history_act_sz.append(float(setpoints[1]))
        self.history_power.append(total_power)
        self.current_step += 1
        if self.current_step % 24 == 0:
            self.lifecycle.heartbeat(self.episode_index)
        next_observation = self._get_observations(next_payload)
        self.last_raw_observation = next_observation.copy()
        outside = safe_float(self.fp.get_local_forecast(current_time, "TDryBul", 1)[0])
        outside_c = outside - 273.15 if outside > 100 else outside
        info: dict[str, Any] = {
            "time": current_time,
            "rew_energy": reward_energy,
            "rew_comfort": reward_comfort,
            "rew_smooth": reward_smooth,
            "phys_cost": cost,
            "phys_pmv_abs": float(np.mean(np.abs([self.current_pmv_nz, self.current_pmv_sz]))),
            "phys_act_diff": float(np.mean(action_changes)),
            "power_total": total_power,
            "temp_out": outside_c,
            "temperatures": [temperature_nz - 273.15, temperature_sz - 273.15],
            "pmvs": [self.current_pmv_nz, self.current_pmv_sz],
            "setpoints": (setpoints - 273.15).tolist(),
            "occupancies": occupancies.tolist(),
            "actions": actions.tolist(),
            "price": price,
        }
        if self.capture_series:
            info["raw_observation"] = previous_observation.tolist()
        truncated = self.current_step >= self.max_episode_steps
        return next_observation, float(reward), False, truncated, info

    def close(self):
        try:
            self.lifecycle.stop("environment_close")
        except Exception:
            pass
        super().close()


class Air5ZoneResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: CaseConfig,
        forecast_data: Mapping[str, Sequence[float]],
        *,
        is_validation: bool = False,
        legacy_fixed_baseline: bool = False,
        rank: int | str = 0,
        phase: str = "training",
        capture_series: bool = False,
    ):
        super().__init__()
        self.config = config
        self.fp = ForecastProvider(config, forecast_data)
        self.is_validation = is_validation
        self.legacy_fixed_baseline = legacy_fixed_baseline
        self.rank = rank
        self.phase = phase
        self.capture_series = capture_series
        self.zones = AIR_ZONES
        self.start_time = (config.start_day + (7 if is_validation else 0)) * 24 * 3600
        self.max_episode_steps = config.episode_steps
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        low, high = self._observation_bounds()
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.lifecycle = TestIDLifecycle(
            config,
            owner=f"{config.case_key}_{phase}_{rank}",
            algorithm=f"{config.case_key}_ppo_v2",
            phase=phase,
            rank=rank,
        )
        self.history_temp = {zone: deque(maxlen=5) for zone in self.zones}
        self.history_act = {zone: deque(maxlen=5) for zone in self.zones}
        self.history_power: deque[float] = deque(maxlen=5)
        self.current_pmv = {zone: 0.0 for zone in self.zones}
        self.current_clo = 0.5
        self.current_step = 0
        self.episode_index = 0
        self.last_raw_observation = np.zeros(len(config.observation_columns), dtype=np.float32)

    def _observation_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lows: list[float] = []
        highs: list[float] = []
        for column in self.config.observation_columns:
            if "_T_" in column:
                key = "T"
            elif "Price" in column:
                key = "Price"
            elif "OccMask" in column:
                key = "Occ"
            elif "HGloHor" in column:
                key = "HGloHor"
            elif "TDryBul" in column:
                key = "TDryBul"
            elif "power_norm" in column:
                key = "pow"
            elif "pmv" in column:
                key = "pmv"
            elif "act" in column:
                key = "act"
            else:
                key = column.replace("obs_", "")
            bounds = self.config.observations_config.get(key, (0.0, 1.0))
            lows.append(bounds[0])
            highs.append(bounds[1])
        return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)

    def _get_observations(self, payload: Mapping[str, Any]) -> np.ndarray:
        current_time = safe_float(payload.get("time"), self.start_time)
        raw: dict[str, float] = {
            "obs_sin_time": math.sin(2 * math.pi * (current_time % 86400) / 86400),
            "obs_cos_time": math.cos(2 * math.pi * (current_time % 86400) / 86400),
        }
        for zone in self.zones:
            title = zone.capitalize()
            raw[f"obs_T_{zone}"] = safe_float(
                payload.get(f"hvac_reaZon{title}_TZon_y"), 298.15
            )
            for index in range(1, 5):
                raw[f"obs_T_{zone}_p{index}"] = (
                    float(self.history_temp[zone][-index])
                    if len(self.history_temp[zone]) >= index
                    else 298.15
                )
            raw[f"obs_pmv_{zone}"] = self.current_pmv[zone]
            raw[f"obs_act_{zone}_p1"] = (
                float(self.history_act[zone][-1]) if self.history_act[zone] else 298.15
            )
            occupancy = self.fp.get_local_forecast(current_time, f"Occupancy[{zone}]", 5)
            for index in range(5):
                raw[f"obs_OccMask_{zone}_{index}"] = float(occupancy[index] > 0)
        for index in range(1, 5):
            raw[f"obs_power_norm_p{index}"] = (
                float(self.history_power[-index]) / self.config.max_system_power
                if len(self.history_power) >= index
                else 0.0
            )
        for point in ("TDryBul", "HGloHor", "PriceElectricPowerDynamic"):
            values = self.fp.get_local_forecast(current_time, point, 5)
            name = "Price" if point == "PriceElectricPowerDynamic" else point
            for index in range(5):
                raw[f"obs_{name}_{index}"] = float(values[index])
        return np.asarray([raw.get(column, 0.0) for column in self.config.observation_columns], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.lifecycle.testid is None:
            self.lifecycle.configure()
        self.lifecycle.epoch = self.episode_index + 1
        response = self.lifecycle.initialize(self.start_time, 7 * 24 * 3600)
        payload = response["payload"]
        self.current_step = 0
        self.current_clo = 0.5
        self.history_power.clear()
        self.history_power.append(0.0)
        for zone in self.zones:
            title = zone.capitalize()
            self.history_temp[zone].clear()
            self.history_act[zone].clear()
            self.history_temp[zone].append(
                safe_float(payload.get(f"hvac_reaZon{title}_TZon_y"), 298.15)
            )
            self.history_act[zone].append(298.15)
            self.current_pmv[zone] = 0.0
        self.episode_index += 1
        observation = self._get_observations(payload)
        self.last_raw_observation = observation.copy()
        return observation, {}

    def step(self, action: np.ndarray):
        current_time = self.start_time + self.current_step * self.config.control_period
        if self.current_step % 96 == 0:
            outside = self.fp.get_local_forecast(current_time, "TDryBul", 96)
            self.current_clo = get_dynamic_clo(
                float(np.mean(outside)) - 273.15,
                self.config.physical_config["clo_dynamic_params"],
            )
        occupancies = np.asarray(
            [
                self.fp.get_local_forecast(current_time, f"Occupancy[{zone}]", 1)[0]
                for zone in self.zones
            ],
            dtype=float,
        )
        actions = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        setpoints = residual_setpoints(
            actions,
            occupancies,
            self.config,
            legacy_fixed_baseline=self.legacy_fixed_baseline,
        )
        response = request_json(
            "post",
            f"{self.config.boptest_url}/advance/{self.lifecycle.testid}",
            json=air_action_payload(self.zones, setpoints),
        )
        next_payload = response["payload"]
        total_power = sum(
            safe_float(next_payload.get(name))
            for name in (
                "chi_reaPChi_y",
                "chi_reaPPumDis_y",
                "heaPum_reaPHeaPum_y",
                "heaPum_reaPPumDis_y",
                "hvac_reaAhu_PFanSup_y",
                "hvac_reaAhu_PPumCoo_y",
                "hvac_reaAhu_PPumHea_y",
            )
        )
        price = safe_float(self.fp.get_local_forecast(current_time, "PriceElectricPowerDynamic", 1)[0])
        cost = total_power * self.config.control_period / 3_600_000.0 * price
        temperatures: list[float] = []
        pmvs: list[float] = []
        comfort_penalty = 0.0
        smooth_penalty = 0.0
        for index, zone in enumerate(self.zones):
            title = zone.capitalize()
            temperature_k = safe_float(next_payload.get(f"hvac_reaZon{title}_TZon_y"), 298.15)
            pmv = calculate_pmv_state(
                temperature_k, self.config.physical_config, self.current_clo
            )
            self.current_pmv[zone] = pmv
            comfort_penalty += comfort_excess_squared(
                pmv, occupancies[index], self.config.comfort_threshold
            )
            smooth_penalty += abs(setpoints[index] - self.history_act[zone][-1])
            temperatures.append(temperature_k - 273.15)
            pmvs.append(pmv)
        weights = self.config.reward_weights
        reward_energy = weights["w_energy"] * (cost / self.config.n_zones) * weights["scaler_energy"]
        reward_comfort = (
            weights["w_comfort"]
            * (comfort_penalty / self.config.n_zones)
            * weights["scaler_comfort"]
        )
        reward_smooth = (
            weights["w_smooth"]
            * (smooth_penalty / self.config.n_zones)
            * weights["scaler_smooth"]
        )
        reward = -(reward_energy + reward_comfort + reward_smooth)
        previous_observation = self.last_raw_observation.copy()
        for index, zone in enumerate(self.zones):
            self.history_temp[zone].append(temperatures[index] + 273.15)
            self.history_act[zone].append(float(setpoints[index]))
        self.history_power.append(total_power)
        self.current_step += 1
        if self.current_step % 24 == 0:
            self.lifecycle.heartbeat(self.episode_index)
        next_observation = self._get_observations(next_payload)
        self.last_raw_observation = next_observation.copy()
        outside = safe_float(self.fp.get_local_forecast(current_time, "TDryBul", 1)[0])
        outside_c = outside - 273.15 if outside > 100 else outside
        info: dict[str, Any] = {
            "time": current_time,
            "rew_energy": reward_energy,
            "rew_comfort": reward_comfort,
            "rew_smooth": reward_smooth,
            "phys_cost": cost,
            "phys_pmv_abs": float(np.mean(np.abs(pmvs))),
            "phys_act_diff": smooth_penalty / self.config.n_zones,
            "power_total": total_power,
            "temp_out": outside_c,
            "temperatures": temperatures,
            "pmvs": pmvs,
            "setpoints": (setpoints - 273.15).tolist(),
            "occupancies": occupancies.tolist(),
            "actions": actions.tolist(),
            "price": price,
        }
        if self.capture_series:
            info["raw_observation"] = previous_observation.tolist()
        truncated = self.current_step >= self.max_episode_steps
        return next_observation, float(reward), False, truncated, info

    def close(self):
        try:
            self.lifecycle.stop("environment_close")
        except Exception:
            pass
        super().close()


def build_environment(
    config: CaseConfig,
    forecast_data: Mapping[str, Sequence[float]],
    *,
    is_validation: bool = False,
    legacy_fixed_baseline: bool = False,
    rank: int | str = 0,
    phase: str = "training",
    capture_series: bool = False,
    monitored: bool = False,
) -> gym.Env:
    environment_class = HydronicResidualEnv if config.case_key == "mz_hydro" else Air5ZoneResidualEnv
    environment: gym.Env = environment_class(
        config,
        forecast_data,
        is_validation=is_validation,
        legacy_fixed_baseline=legacy_fixed_baseline,
        rank=rank,
        phase=phase,
        capture_series=capture_series,
    )
    environment = NormalizedObservationWrapper(environment)
    return Monitor(environment) if monitored else environment


def _subprocess_environment(
    config: CaseConfig,
    forecast_data: Mapping[str, Sequence[float]],
    rank: int,
) -> gym.Env:
    return build_environment(
        config,
        forecast_data,
        rank=rank,
        phase="training",
        monitored=True,
    )


def global_linear_schedule(initial_value: float, final_fraction: float = 0.1) -> Callable[[float], float]:
    """Preserve the executed v1 schedule: linear decay to 10% of initial LR."""

    initial = float(initial_value)
    final = float(final_fraction)

    def schedule(progress_remaining: float) -> float:
        return initial * (final + (1.0 - final) * float(progress_remaining))

    return schedule


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _write_json_file(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            json_ready(payload),
            handle,
            indent=2,
            sort_keys=True,
            default=json_default,
            allow_nan=False,
        )
        handle.flush()
        os.fsync(handle.fileno())


class AtomicCheckpointManager:
    def __init__(
        self,
        run_dir: Path,
        config_hash: str,
        *,
        save_retries: int = 5,
        replace_func: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None] = os.replace,
    ):
        self.run_dir = Path(run_dir)
        self.config_hash = config_hash
        self.save_retries = save_retries
        self.replace_func = replace_func
        self.directory = self.run_dir / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / "latest.json"
        self.best_path = self.directory / "best.json"

    def _replace(self, source: Path, destination: Path) -> None:
        self.replace_func(source, destination)

    def _verify_model_zip(self, path: Path) -> None:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Empty model checkpoint: {path}")
        with zipfile.ZipFile(path, "r") as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise RuntimeError(f"Corrupt checkpoint member: {bad_member}")

    def _bundle_names(self, epoch: int, steps: int) -> dict[str, str]:
        prefix = f"epoch_{epoch:04d}_steps_{steps:09d}"
        return {
            "model": f"{prefix}.model.zip",
            "state": f"{prefix}.state.json",
            "rng": f"{prefix}.rng.pt",
            "manifest": f"{prefix}.manifest.json",
        }

    def save(self, model: PPO, state: Mapping[str, Any], epoch: int) -> dict[str, Any]:
        steps = int(model.num_timesteps)
        names = self._bundle_names(epoch, steps)
        last_error: Exception | None = None
        for attempt in range(self.save_retries):
            token = uuid.uuid4().hex
            temporary = {
                "model": self.directory / f".model.{token}.tmp.zip",
                "state": self.directory / f".state.{token}.tmp.json",
                "rng": self.directory / f".rng.{token}.tmp.pt",
                "manifest": self.directory / f".manifest.{token}.tmp.json",
            }
            final = {key: self.directory / name for key, name in names.items()}
            try:
                model.save(temporary["model"], exclude=["_phase15f_hook"])
                self._verify_model_zip(temporary["model"])
                _write_json_file(temporary["state"], state)
                with temporary["rng"].open("wb") as handle:
                    torch.save(capture_rng_state(), handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                manifest = {
                    "schema": "phase1.5f-checkpoint-v2",
                    "created_at": utc_now(),
                    "epoch": int(epoch),
                    "num_timesteps": steps,
                    "config_hash": self.config_hash,
                    "files": {
                        "model": {"name": names["model"], "sha256": sha256_file(temporary["model"])},
                        "state": {"name": names["state"], "sha256": sha256_file(temporary["state"])},
                        "rng": {"name": names["rng"], "sha256": sha256_file(temporary["rng"])},
                    },
                }
                _write_json_file(temporary["manifest"], manifest)
                for key in ("model", "state", "rng", "manifest"):
                    self._replace(temporary[key], final[key])
                self._verify_model_zip(final["model"])
                for key in ("model", "state", "rng"):
                    expected = manifest["files"][key]["sha256"]
                    if sha256_file(final[key]) != expected:
                        raise RuntimeError(f"Checksum mismatch after commit: {final[key]}")
                pointer = {
                    "schema": "phase1.5f-pointer-v2",
                    "manifest": names["manifest"],
                    "manifest_sha256": sha256_file(final["manifest"]),
                    "epoch": int(epoch),
                    "num_timesteps": steps,
                    "config_hash": self.config_hash,
                    "committed_at": utc_now(),
                }
                pointer_temp = self.directory / f".latest.{token}.tmp"
                _write_json_file(pointer_temp, pointer)
                self._replace(pointer_temp, self.latest_path)
                self._prune()
                return {**pointer, "model_path": str(final["model"])}
            except Exception as exc:
                last_error = exc
                for path in temporary.values():
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
                if attempt + 1 < self.save_retries:
                    time.sleep(0.25 * (2**attempt))
        raise RuntimeError(
            f"Atomic checkpoint failed after {self.save_retries} attempts; "
            "the previous latest pointer is unchanged"
        ) from last_error

    def mark_best(self, pointer: Mapping[str, Any], score: float) -> None:
        manifest_name = str(pointer["manifest"])
        payload = {
            "schema": "phase1.5f-best-v2",
            "manifest": manifest_name,
            "manifest_sha256": sha256_file(self.directory / manifest_name),
            "epoch": int(pointer["epoch"]),
            "num_timesteps": int(pointer["num_timesteps"]),
            "score": float(score),
            "config_hash": self.config_hash,
            "committed_at": utc_now(),
        }
        atomic_write_json(self.best_path, payload)
        self._prune()

    def _read_pointer(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        pointer = json.loads(path.read_text(encoding="utf-8"))
        if pointer.get("config_hash") != self.config_hash:
            raise RuntimeError(
                "Checkpoint scientific configuration hash mismatch; refusing silent resume"
            )
        manifest_path = self.directory / pointer["manifest"]
        if sha256_file(manifest_path) != pointer["manifest_sha256"]:
            raise RuntimeError(f"Checkpoint manifest checksum mismatch: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != self.config_hash:
            raise RuntimeError("Checkpoint manifest configuration hash mismatch")
        for key in ("model", "state", "rng"):
            metadata = manifest["files"][key]
            path_to_file = self.directory / metadata["name"]
            if sha256_file(path_to_file) != metadata["sha256"]:
                raise RuntimeError(f"Checkpoint file checksum mismatch: {path_to_file}")
        return pointer, manifest

    def load_latest_metadata(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        pointer, manifest = self._read_pointer(self.latest_path)
        state_path = self.directory / manifest["files"]["state"]["name"]
        rng_path = self.directory / manifest["files"]["rng"]["name"]
        model_path = self.directory / manifest["files"]["model"]["name"]
        state = json.loads(state_path.read_text(encoding="utf-8"))
        with rng_path.open("rb") as handle:
            rng = torch.load(handle, map_location="cpu", weights_only=False)
        if int(state.get("committed_epoch", -1)) != int(pointer["epoch"]):
            raise RuntimeError("Checkpoint state epoch does not match latest pointer")
        return {
            "pointer": pointer,
            "manifest": manifest,
            "state": state,
            "rng": rng,
            "model_path": model_path,
        }

    def load_best_metadata(self) -> dict[str, Any]:
        if not self.best_path.exists():
            raise FileNotFoundError("No train-week best checkpoint exists yet")
        pointer, manifest = self._read_pointer(self.best_path)
        return {
            "pointer": pointer,
            "manifest": manifest,
            "model_path": self.directory / manifest["files"]["model"]["name"],
            "state_path": self.directory / manifest["files"]["state"]["name"],
        }

    def _pinned_manifests(self) -> set[str]:
        pinned: set[str] = set()
        for pointer_path in (self.latest_path, self.best_path):
            if pointer_path.exists():
                try:
                    pinned.add(str(json.loads(pointer_path.read_text(encoding="utf-8"))["manifest"]))
                except Exception:
                    pass
        manifests = sorted(
            self.directory.glob("epoch_*.manifest.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        pinned.update(path.name for path in manifests[:2])
        return pinned

    def _prune(self) -> None:
        pinned = self._pinned_manifests()
        for manifest_path in self.directory.glob("epoch_*.manifest.json"):
            if manifest_path.name in pinned:
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for metadata in manifest.get("files", {}).values():
                    (self.directory / metadata["name"]).unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
            except Exception:
                # Pruning is best-effort and can never invalidate a committed pointer.
                pass


def get_or_create_run_identity(config: CaseConfig, preferred_id: str | None = None) -> dict[str, Any]:
    path = config.run_dir / "run_identity.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if preferred_id and existing.get("wandb_run_id") != preferred_id:
            existing["superseded_wandb_run_id"] = existing.get("wandb_run_id")
            existing["wandb_run_id"] = preferred_id
            existing["reconciled_from_checkpoint_at"] = utc_now()
            atomic_write_json(path, existing)
        return existing
    run_id = preferred_id or f"{config.case_key}-{config.run_mode}-{uuid.uuid4().hex[:12]}"
    payload = {
        "wandb_run_id": run_id,
        "case_key": config.case_key,
        "run_mode": config.run_mode,
        "run_tag": config.run_tag,
        "created_at": utc_now(),
    }
    atomic_write_json(path, payload)
    return payload


class SafeWandb:
    """Scalar-only W&B adapter. Any W&B failure degrades to local logging."""

    def __init__(self, config: CaseConfig, run_id: str):
        self.config = config
        self.run_id = run_id
        self.run: Any = None
        self.mode = config.wandb_mode
        self.error: str | None = None
        self.disabled = False

    def start(self) -> None:
        wandb_root = self.config.run_dir / "wandb"
        wandb_root.mkdir(parents=True, exist_ok=True)
        # Keep every W&B write inside the experiment directory.  This avoids
        # user-profile temp/cache failures and makes the online-to-offline
        # fallback self-contained on managed Windows hosts.
        for environment_name, child in (
            ("WANDB_DATA_DIR", "data"),
            ("WANDB_CACHE_DIR", "cache"),
            ("WANDB_CONFIG_DIR", "config"),
            ("WANDB_ARTIFACT_DIR", "artifacts"),
        ):
            destination = wandb_root / child
            destination.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault(environment_name, str(destination))
        try:
            import wandb
        except Exception as exc:
            self.error = f"wandb import failed: {exc!r}"
            self.disabled = True
            return
        modes = [self.mode]
        if self.mode == "online":
            modes.append("offline")
        for mode in modes:
            try:
                # The default 90-second W&B init timeout is too long for a
                # training dependency that is explicitly non-fatal.  A
                # bounded online attempt is followed by a local offline run.
                settings = wandb.Settings(
                    init_timeout=15.0,
                    login_timeout=10.0,
                    x_graphql_timeout_seconds=10.0,
                )
                self.run = wandb.init(
                    project=self.config.wandb_project,
                    group=self.config.wandb_group,
                    name=f"{self.config.case_key}-ppo-v2-{self.config.run_mode}-{self.config.run_tag}",
                    id=self.run_id,
                    resume="allow",
                    mode=mode,
                    dir=str(wandb_root),
                    config=self.config.scientific_payload,
                    reinit=True,
                    settings=settings,
                )
                self.mode = mode
                return
            except Exception as exc:
                self.error = repr(exc)
                self.run = None
                try:
                    wandb.finish()
                except Exception:
                    pass
        self.disabled = True

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        if self.disabled or self.run is None:
            return
        scalar_metrics: dict[str, float | int | bool] = {}
        for key, value in metrics.items():
            if isinstance(value, (bool, np.bool_)):
                scalar_metrics[key] = bool(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                number = safe_float(value, float("nan"))
                if math.isfinite(number):
                    scalar_metrics[key] = number
        try:
            self.run.log(scalar_metrics, step=int(step))
        except Exception as exc:
            self.error = repr(exc)
            self.disabled = True

    def finish(self, lifecycle_path: Path | None = None, summary: Mapping[str, Any] | None = None) -> None:
        if self.run is None:
            return
        try:
            if summary:
                for key, value in summary.items():
                    if isinstance(value, (str, bool, int, float)) or value is None:
                        self.run.summary[key] = value
            if lifecycle_path and lifecycle_path.exists() and not self.disabled:
                try:
                    import wandb

                    artifact = wandb.Artifact(
                        f"{self.config.case_key}-{self.run_id}-lifecycle",
                        type="testid-lifecycle",
                    )
                    artifact.add_file(str(lifecycle_path))
                    self.run.log_artifact(artifact)
                except Exception:
                    pass
            self.run.finish()
        except Exception:
            pass


class Phase15fPPO(PPO):
    """Unmodified PPO updates plus an explicit post-update commit hook."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._phase15f_hook: Callable[[PPO], bool] | None = None

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "_phase15f_hook"]

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        tb_log_name: str = "Phase15fPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())
        assert self.env is not None
        try:
            while self.num_timesteps < total_timesteps:
                continue_training = self.collect_rollouts(
                    self.env,
                    callback,
                    self.rollout_buffer,
                    n_rollout_steps=self.n_steps,
                )
                if not continue_training:
                    break
                iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
                if log_interval is not None and iteration % log_interval == 0:
                    self.dump_logs(iteration)
                self.train()
                hook = getattr(self, "_phase15f_hook", None)
                if hook is not None and not hook(self):
                    break
        finally:
            callback.on_training_end()
        return self


def _median(values: Iterable[float], default: float = float("nan")) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else default


def evaluate_convergence(
    training_rows: Sequence[Mapping[str, Any]],
    sb3_rows: Sequence[Mapping[str, Any]],
    train_eval_rows: Sequence[Mapping[str, Any]],
    *,
    best_epoch: int | None,
    min_epoch: int = 120,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "converged": False,
        "eligible": False,
        "curve_plateau": False,
        "curve_band": False,
        "best_not_late": False,
        "reward_healthy": False,
        "action_healthy": False,
        "sb3_healthy": False,
        "reasons": [],
    }
    if not training_rows:
        result["reasons"].append("no_training_rows")
        return result
    epoch = int(training_rows[-1]["epoch"])
    if epoch < min_epoch or len(training_rows) < 90:
        result["reasons"].append("minimum_epoch_not_reached")
        return result
    result["eligible"] = True
    rewards = np.asarray([safe_float(row["reward_mean"]) for row in training_rows], dtype=float)
    rolling = pd.Series(rewards).rolling(30, min_periods=30).mean().to_numpy()
    current = rolling[-1]
    previous = rolling[-61] if len(rolling) >= 61 else float("nan")
    denominator = max(abs(previous), 1e-9)
    gain_60 = abs(current - previous) / denominator if np.isfinite(previous) else float("inf")
    last_60 = rolling[-60:]
    finite_60 = last_60[np.isfinite(last_60)]
    if finite_60.size >= 30:
        slope = float(np.polyfit(np.arange(finite_60.size), finite_60, 1)[0])
        projected = abs(slope * 60.0) / max(abs(current), 1e-9)
    else:
        projected = float("inf")
    result["gain_60"] = gain_60
    result["projected_trend_60"] = projected
    result["curve_plateau"] = bool(gain_60 < 0.01 and projected < 0.01)
    tail_length = max(1, int(math.ceil(0.2 * len(rolling))))
    tail = rolling[-tail_length:]
    tail = tail[np.isfinite(tail)]
    result["curve_band"] = bool(
        tail.size
        and np.all(np.abs(tail - current) <= 0.02 * max(abs(current), 1e-9))
    )
    result["best_not_late"] = bool(best_epoch is not None and best_epoch <= 0.9 * epoch)

    latest_training = training_rows[-1]
    latest_eval = train_eval_rows[-1] if train_eval_rows else {}
    energy = abs(safe_float(latest_training.get("reward_energy_mean")))
    comfort = abs(safe_float(latest_training.get("reward_comfort_mean")))
    discomfort = safe_float(latest_eval.get("pmv_hours"))
    reward_healthy = discomfort <= 0.0 or comfort >= 0.05 * max(energy, 1e-12)
    result["reward_healthy"] = bool(reward_healthy)
    result["action_healthy"] = bool(
        safe_float(latest_training.get("occupied_saturation"), 1.0) < 0.5
        and safe_float(latest_training.get("occupancy_action_gap"), 0.0) >= 0.1
    )

    recent_sb3 = list(sb3_rows[-10:])
    explained = _median(row.get("explained_variance", float("nan")) for row in recent_sb3)
    clip_fraction = _median(row.get("clip_fraction", float("nan")) for row in recent_sb3)
    policy_std = safe_float(recent_sb3[-1].get("policy_std")) if recent_sb3 else float("nan")
    learning_rates = np.asarray(
        [safe_float(row.get("learning_rate"), float("nan")) for row in sb3_rows], dtype=float
    )
    learning_rates = learning_rates[np.isfinite(learning_rates)]
    lr_monotonic = bool(
        learning_rates.size
        and np.all(np.diff(learning_rates) <= np.maximum(1e-12, 1e-9 * learning_rates[:-1]))
    )
    result.update(
        {
            "explained_variance_median_10": explained,
            "clip_fraction_median_10": clip_fraction,
            "policy_std": policy_std,
            "learning_rate_monotonic": lr_monotonic,
        }
    )
    result["sb3_healthy"] = bool(
        explained > 0.7
        and 0.05 <= clip_fraction <= 0.30
        and 0.1 <= policy_std <= 0.4
        and lr_monotonic
    )
    checks = (
        "curve_plateau",
        "curve_band",
        "best_not_late",
        "reward_healthy",
        "action_healthy",
        "sb3_healthy",
    )
    result["reasons"] = [name for name in checks if not result[name]]
    result["converged"] = not result["reasons"]
    return result


def _rollout_row(config: CaseConfig, info: Mapping[str, Any], reward: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "time": safe_float(info.get("time")),
        "temp_out": safe_float(info.get("temp_out")),
        "power_total": safe_float(info.get("power_total")),
        "cost": safe_float(info.get("phys_cost")),
        "reward": safe_float(reward),
    }
    for index, zone in enumerate(config.zones):
        row[f"temp_{zone}"] = safe_float(info["temperatures"][index])
        row[f"pmv_{zone}"] = safe_float(info["pmvs"][index])
        row[f"setpoint_{zone}"] = safe_float(info["setpoints"][index])
        row[f"occ_{zone}"] = safe_float(info["occupancies"][index])
    if config.case_key == "mz_hydro":
        raw_observation = info.get("raw_observation", [])
        for index, column in enumerate(config.observation_columns):
            row[column] = safe_float(raw_observation[index]) if index < len(raw_observation) else 0.0
    row["_actions"] = list(info.get("actions", []))
    row["_price"] = safe_float(info.get("price"))
    return row


def validation_columns(config: CaseConfig) -> list[str]:
    columns = ["time", "temp_out", "power_total", "cost", "reward"]
    for zone in config.zones:
        columns.extend([f"temp_{zone}", f"pmv_{zone}", f"setpoint_{zone}", f"occ_{zone}"])
    if config.case_key == "mz_hydro":
        columns.extend(config.observation_columns)
    return columns


def calculate_rollout_metrics(
    frame: pd.DataFrame,
    actions: Sequence[Sequence[float]],
    config: CaseConfig,
) -> dict[str, Any]:
    step_hours = config.control_period / 3600.0
    zone_hours = 0.0
    pmv_hours = 0.0
    occupied_violations = 0
    occupied_samples = 0
    all_violations = 0
    all_samples = len(frame) * config.n_zones
    occupied_actions: list[float] = []
    unoccupied_actions: list[float] = []
    gaps: list[float] = []
    action_array = np.asarray(actions, dtype=float)
    for zone_index, zone in enumerate(config.zones):
        pmv = frame[f"pmv_{zone}"].abs().to_numpy(dtype=float)
        occupancy = frame[f"occ_{zone}"].to_numpy(dtype=float) > 0
        violation = pmv > config.comfort_threshold
        zone_hours += float(np.sum(violation & occupancy) * step_hours)
        pmv_hours += float(
            np.sum(np.maximum(0.0, pmv[occupancy] - config.comfort_threshold)) * step_hours
        )
        occupied_violations += int(np.sum(violation & occupancy))
        occupied_samples += int(np.sum(occupancy))
        all_violations += int(np.sum(violation))
        if action_array.ndim == 2 and action_array.shape[0] == len(frame):
            zone_actions = action_array[:, zone_index]
            occupied_actions.extend(zone_actions[occupancy].tolist())
            unoccupied_actions.extend(zone_actions[~occupancy].tolist())
            if np.any(occupancy) and np.any(~occupancy):
                gaps.append(abs(float(np.mean(zone_actions[occupancy]) - np.mean(zone_actions[~occupancy]))))
    any_occupied = np.zeros(len(frame), dtype=bool)
    for zone in config.zones:
        any_occupied |= frame[f"occ_{zone}"].to_numpy(dtype=float) > 0
    cost = frame["cost"].to_numpy(dtype=float)
    setpoint_return: list[float] = []
    for zone in config.zones:
        unoccupied = frame[f"occ_{zone}"].to_numpy(dtype=float) <= 0
        if np.any(unoccupied):
            setpoint_return.extend(
                (
                    np.abs(
                        frame.loc[unoccupied, f"setpoint_{zone}"].to_numpy(dtype=float) - 30.0
                    )
                    <= 0.5
                ).astype(float)
            )
    metrics: dict[str, Any] = {
        "return": float(frame["reward"].sum()),
        "cost": float(frame["cost"].sum()),
        "zone_hours": zone_hours,
        "pmv_hours": pmv_hours,
        "occupied_pmv_violation_rate": occupied_violations / max(occupied_samples, 1),
        "all_time_pmv_violation_rate": all_violations / max(all_samples, 1),
        "occupied_cost": float(np.sum(cost[any_occupied])),
        "unoccupied_cost": float(np.sum(cost[~any_occupied])),
        "occupied_saturation": float(
            np.mean(np.abs(occupied_actions) > 0.95) if occupied_actions else float("nan")
        ),
        "unoccupied_saturation": float(
            np.mean(np.abs(unoccupied_actions) > 0.95) if unoccupied_actions else float("nan")
        ),
        "occupancy_action_gap": float(np.mean(gaps) if gaps else float("nan")),
        "unoccupied_return_rate": float(np.mean(setpoint_return) if setpoint_return else float("nan")),
    }
    if config.case_key == "mz_hydro":
        lead_times: list[float] = []
        for zone in config.zones:
            occupancy = frame[f"occ_{zone}"].to_numpy(dtype=float) > 0
            setpoint = frame[f"setpoint_{zone}"].to_numpy(dtype=float)
            starts = np.where(occupancy & ~np.r_[False, occupancy[:-1]])[0]
            for start in starts:
                begin = max(0, start - int(2 * 3600 / config.control_period))
                lowered = np.where(setpoint[begin:start] <= 29.5)[0]
                if lowered.size:
                    lead_times.append((start - (begin + lowered[0])) * step_hours)
        metrics["median_precooling_lead_hours"] = (
            float(np.median(lead_times)) if lead_times else 0.0
        )
    return metrics


def rollout_policy(
    config: CaseConfig,
    forecast_data: Mapping[str, Sequence[float]],
    *,
    model: Any | None,
    is_validation: bool,
    phase: str,
    legacy_fixed_baseline: bool = False,
    zero_policy: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    environment = build_environment(
        config,
        forecast_data,
        is_validation=is_validation,
        legacy_fixed_baseline=legacy_fixed_baseline,
        rank=-1,
        phase=phase,
        capture_series=True,
        monitored=False,
    )
    rows: list[dict[str, Any]] = []
    action_rows: list[list[float]] = []
    try:
        observation, _ = environment.reset(seed=config.seed)
        for _ in range(config.episode_steps):
            if zero_policy:
                action = np.zeros(config.n_zones, dtype=np.float32)
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = environment.step(action)
            rows.append(_rollout_row(config, info, reward))
            action_rows.append(list(np.asarray(action, dtype=float)))
            if terminated or truncated:
                break
    finally:
        environment.close()
    frame = pd.DataFrame(rows)
    metrics = calculate_rollout_metrics(frame, action_rows, config)
    frame["_actions"] = [json.dumps(action) for action in action_rows]
    return frame, metrics


class TrainWeekEvaluator:
    def __init__(self, config: CaseConfig, forecast_data: Mapping[str, Sequence[float]]):
        self.config = config
        self.forecast_data = forecast_data
        self.directory = config.run_dir / "diagnostics" / "train_week"
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, model: PPO, epoch: int) -> dict[str, Any]:
        frame, metrics = rollout_policy(
            self.config,
            self.forecast_data,
            model=model,
            is_validation=False,
            phase=f"train_week_eval_epoch_{epoch:04d}",
        )
        output = self.directory / f"train_week_epoch_{epoch:04d}.csv"
        export = frame[validation_columns(self.config)]
        export.to_csv(output, index=False)
        return {
            "epoch": int(epoch),
            "global_step": int(model.num_timesteps),
            **metrics,
            "trajectory": str(output),
            "timestamp": utc_now(),
        }


def _summarize_action_health(samples: Sequence[tuple[int, float, float]], n_zones: int) -> dict[str, float]:
    occupied = [action for _, action, occupancy in samples if occupancy > 0]
    unoccupied = [action for _, action, occupancy in samples if occupancy <= 0]
    gaps: list[float] = []
    for zone in range(n_zones):
        zone_occupied = [action for index, action, occupancy in samples if index == zone and occupancy > 0]
        zone_unoccupied = [
            action for index, action, occupancy in samples if index == zone and occupancy <= 0
        ]
        if zone_occupied and zone_unoccupied:
            gaps.append(abs(float(np.mean(zone_occupied) - np.mean(zone_unoccupied))))
    return {
        "occupied_saturation": float(
            np.mean(np.abs(occupied) > 0.95) if occupied else float("nan")
        ),
        "unoccupied_saturation": float(
            np.mean(np.abs(unoccupied) > 0.95) if unoccupied else float("nan")
        ),
        "occupancy_action_gap": float(np.mean(gaps) if gaps else float("nan")),
        "occupied_action_mean": float(np.mean(occupied) if occupied else float("nan")),
        "unoccupied_action_mean": float(np.mean(unoccupied) if unoccupied else float("nan")),
    }


class Phase15fCallback(BaseCallback):
    def __init__(
        self,
        config: CaseConfig,
        checkpoint: AtomicCheckpointManager,
        train_week_evaluator: TrainWeekEvaluator,
        wandb_logger: SafeWandb,
        *,
        environment_fingerprint: str,
        restored_state: Mapping[str, Any] | None = None,
    ):
        super().__init__(verbose=0)
        self.config = config
        self.checkpoint = checkpoint
        self.train_week_evaluator = train_week_evaluator
        self.wandb = wandb_logger
        self.environment_fingerprint = environment_fingerprint
        restored = dict(restored_state or {})
        self.training_rows: list[dict[str, Any]] = list(restored.get("training_rows", []))
        self.sb3_rows: list[dict[str, Any]] = list(restored.get("sb3_rows", []))
        self.train_eval_rows: list[dict[str, Any]] = list(restored.get("train_eval_rows", []))
        self.committed_epoch = int(restored.get("committed_epoch", 0))
        self.best_score = safe_float(restored.get("best_score"), -float("inf"))
        self.best_epoch = (
            int(restored["best_epoch"]) if restored.get("best_epoch") is not None else None
        )
        self.resume_count = int(restored.get("resume_count", 0)) + int(bool(restored_state))
        self.convergence = dict(restored.get("convergence", {"converged": False}))
        self.started_at = time.perf_counter()
        self._reset_epoch_accumulators()

    def _reset_epoch_accumulators(self) -> None:
        self.component_energy: list[float] = []
        self.component_comfort: list[float] = []
        self.component_smooth: list[float] = []
        self.action_samples: list[tuple[int, float, float]] = []
        self.episode_returns: list[float] = []
        self.running_returns = np.zeros(self.config.num_envs, dtype=float)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        if rewards.size == self.running_returns.size:
            self.running_returns += rewards
        for env_index, info in enumerate(infos):
            self.component_energy.append(safe_float(info.get("rew_energy")))
            self.component_comfort.append(safe_float(info.get("rew_comfort")))
            self.component_smooth.append(safe_float(info.get("rew_smooth")))
            actions = info.get("actions", [])
            occupancies = info.get("occupancies", [])
            for zone_index, (action, occupancy) in enumerate(zip(actions, occupancies)):
                self.action_samples.append(
                    (zone_index, safe_float(action), safe_float(occupancy))
                )
            if env_index < dones.size and dones[env_index]:
                episode = info.get("episode", {})
                episode_return = safe_float(
                    episode.get("r"),
                    self.running_returns[env_index] if env_index < self.running_returns.size else 0.0,
                )
                self.episode_returns.append(episode_return)
                if env_index < self.running_returns.size:
                    self.running_returns[env_index] = 0.0
        return True

    def _sb3_row(self, model: PPO, epoch: int) -> dict[str, Any]:
        logger_values = getattr(model.logger, "name_to_value", {})
        try:
            policy_std = float(torch.exp(model.policy.log_std.detach()).mean().cpu().item())
        except Exception:
            policy_std = safe_float(logger_values.get("train/std"), float("nan"))
        learning_rate = safe_float(model.policy.optimizer.param_groups[0]["lr"])
        return {
            "epoch": epoch,
            "global_step": int(model.num_timesteps),
            "explained_variance": safe_float(
                logger_values.get("train/explained_variance"), float("nan")
            ),
            "clip_fraction": safe_float(
                logger_values.get("train/clip_fraction"), float("nan")
            ),
            "approx_kl": safe_float(logger_values.get("train/approx_kl"), float("nan")),
            "entropy_loss": safe_float(
                logger_values.get("train/entropy_loss"), float("nan")
            ),
            "value_loss": safe_float(logger_values.get("train/value_loss"), float("nan")),
            "policy_gradient_loss": safe_float(
                logger_values.get("train/policy_gradient_loss"), float("nan")
            ),
            "policy_std": policy_std,
            "learning_rate": learning_rate,
            "timestamp": utc_now(),
        }

    def _training_row(self, model: PPO, epoch: int) -> dict[str, Any]:
        episode_returns = self.episode_returns or self.running_returns.tolist()
        action_health = _summarize_action_health(
            self.action_samples, self.config.n_zones
        )
        prior_rewards = [safe_float(row["reward_mean"]) for row in self.training_rows]
        current_reward = float(np.mean(episode_returns))
        rolling_values = [*prior_rewards, current_reward][-30:]
        return {
            "epoch": epoch,
            "global_step": int(model.num_timesteps),
            "reward_mean": current_reward,
            "reward_std": float(np.std(episode_returns)),
            "rolling_30": float(np.mean(rolling_values)),
            "reward_energy_mean": float(np.mean(self.component_energy)),
            "reward_comfort_mean": float(np.mean(self.component_comfort)),
            "reward_smooth_mean": float(np.mean(self.component_smooth)),
            **action_health,
            "epoch_wall_seconds": float(time.perf_counter() - self.started_at),
            "timestamp": utc_now(),
        }

    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema": "phase1.5f-state-v2",
            "config_hash": self.config.config_hash,
            "committed_epoch": self.committed_epoch,
            "training_rows": self.training_rows,
            "sb3_rows": self.sb3_rows,
            "train_eval_rows": self.train_eval_rows,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "resume_count": self.resume_count,
            "convergence": self.convergence,
            "wandb_run_id": self.wandb.run_id,
            "environment_fingerprint": self.environment_fingerprint,
            "updated_at": utc_now(),
        }

    def on_post_update(self, model: PPO) -> bool:
        epoch = int(model.num_timesteps // self.config.steps_per_epoch)
        if epoch <= self.committed_epoch:
            self._reset_epoch_accumulators()
            return True
        if model.num_timesteps % self.config.steps_per_epoch:
            raise RuntimeError("Post-update checkpoint is not on an epoch boundary")
        training_row = self._training_row(model, epoch)
        sb3_row = self._sb3_row(model, epoch)
        self.training_rows.append(training_row)
        self.sb3_rows.append(sb3_row)
        new_best = False
        if epoch % self.config.eval_interval == 0:
            evaluation = self.train_week_evaluator(model, epoch)
            self.train_eval_rows.append(evaluation)
            if safe_float(evaluation["return"]) > self.best_score:
                self.best_score = safe_float(evaluation["return"])
                self.best_epoch = epoch
                new_best = True
        self.committed_epoch = epoch
        self.convergence = evaluate_convergence(
            self.training_rows,
            self.sb3_rows,
            self.train_eval_rows,
            best_epoch=self.best_epoch,
            min_epoch=self.config.min_early_stop_epoch,
        )
        pointer = self.checkpoint.save(model, self._state_payload(), epoch)
        if new_best:
            self.checkpoint.mark_best(pointer, self.best_score)
        write_csv(self.config.run_dir / "training_metrics.csv", self.training_rows)
        write_csv(self.config.run_dir / "sb3_updates.csv", self.sb3_rows)
        write_csv(self.config.run_dir / "train_week_eval.csv", self.train_eval_rows)
        metrics = {
            "epoch": epoch,
            "train/reward_mean": training_row["reward_mean"],
            "train/reward_std": training_row["reward_std"],
            "train/rolling_30": training_row["rolling_30"],
            "reward/energy": training_row["reward_energy_mean"],
            "reward/comfort": training_row["reward_comfort_mean"],
            "reward/smooth": training_row["reward_smooth_mean"],
            "action/occupied_saturation": training_row["occupied_saturation"],
            "action/unoccupied_saturation": training_row["unoccupied_saturation"],
            "action/occupancy_gap": training_row["occupancy_action_gap"],
            "ppo/explained_variance": sb3_row["explained_variance"],
            "ppo/clip_fraction": sb3_row["clip_fraction"],
            "ppo/approx_kl": sb3_row["approx_kl"],
            "ppo/policy_std": sb3_row["policy_std"],
            "ppo/learning_rate": sb3_row["learning_rate"],
            "convergence/converged": self.convergence["converged"],
        }
        if self.train_eval_rows and self.train_eval_rows[-1]["epoch"] == epoch:
            metrics.update(
                {
                    "train_week/return": self.train_eval_rows[-1]["return"],
                    "train_week/cost": self.train_eval_rows[-1]["cost"],
                    "train_week/zone_hours": self.train_eval_rows[-1]["zone_hours"],
                    "train_week/pmv_hours": self.train_eval_rows[-1]["pmv_hours"],
                }
            )
        self.wandb.log(metrics, int(model.num_timesteps))
        print(
            f"[{self.config.case_key.upper()} Epoch {epoch:03d}] "
            f"reward={training_row['reward_mean']:.3f} "
            f"rolling30={training_row['rolling_30']:.3f} "
            f"lr={sb3_row['learning_rate']:.3e} "
            f"best_train_week={self.best_score:.3f}"
        )
        should_continue = not bool(self.convergence.get("converged"))
        self.started_at = time.perf_counter()
        self._reset_epoch_accumulators()
        return should_continue


def legacy_mappo_status(project_root: Path) -> dict[str, Any]:
    base = (
        project_root
        / "CASE_TEST"
        / "MZ_OFFICE_HYDRONIC"
        / "multizone_office_simple_hydronic_convergence"
    )
    log_path = base / "mappo" / "logs" / "training.csv"
    if not log_path.exists():
        return {"status": "not_found", "path": str(log_path)}
    try:
        frame = pd.read_csv(log_path)
        epoch = int(frame.iloc[-1]["epoch"])
        age_seconds = max(0.0, time.time() - log_path.stat().st_mtime)
        run_id = None
        identity_path = base / "mappo" / "run_identity.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            run_id = identity.get("wandb_run_id")
        last_events: dict[str, dict[str, Any]] = {}
        for lifecycle_path in (base / "lifecycle").glob("mappo_*.jsonl"):
            for line in lifecycle_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if run_id and event.get("run_id") != run_id:
                    continue
                testid = event.get("testid")
                if testid:
                    last_events[str(testid)] = event
        stopped_events = {"stopped", "already_stopped"}
        active_testids = [
            testid
            for testid, event in last_events.items()
            if event.get("event") not in stopped_events
        ]
        if active_testids:
            status = "finalizing" if epoch >= 500 else "active"
        else:
            status = "complete" if epoch >= 500 else "incomplete_stale"
        return {
            "status": status,
            "epoch": epoch,
            "rows": len(frame),
            "age_seconds": age_seconds,
            "run_id": run_id,
            "active_testids": active_testids,
            "path": str(log_path),
        }
    except Exception as exc:
        return {"status": "unreadable", "path": str(log_path), "error": repr(exc)}


def hydro_full_predecessor_status(project_root: Path, run_tag: str) -> dict[str, Any]:
    path = (
        project_root
        / "CASE_TEST"
        / "MZ_OFFICE_HYDRONIC"
        / "phase1_5f_ppo_v2"
        / f"full-{run_tag}"
        / "run_manifest.json"
    )
    if not path.exists():
        return {"status": "not_found", "path": str(path)}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return {
            "status": manifest.get("status", "unknown"),
            "actual_epochs": manifest.get("actual_epochs", 0),
            "converged": manifest.get("converged", False),
            "path": str(path),
        }
    except Exception as exc:
        return {"status": "unreadable", "path": str(path), "error": repr(exc)}


def query_boptest_version(config: CaseConfig) -> dict[str, Any]:
    for endpoint in ("/version", "/name"):
        try:
            response = requests.get(f"{config.boptest_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                try:
                    payload: Any = response.json()
                except Exception:
                    payload = response.text
                return {"endpoint": endpoint, "payload": payload}
        except Exception:
            continue
    return {"status": "unavailable"}


def runtime_versions() -> dict[str, Any]:
    import stable_baselines3

    versions: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gym.__version__,
        "module_sha256": sha256_file(Path(__file__).resolve()),
    }
    try:
        import pythermalcomfort

        versions["pythermalcomfort"] = pythermalcomfort.__version__
    except Exception:
        versions["pythermalcomfort"] = "unavailable"
    try:
        import wandb

        versions["wandb"] = wandb.__version__
    except Exception:
        versions["wandb"] = "unavailable"
    return versions


def run_preflight(config: CaseConfig, *, online: bool = False) -> dict[str, Any]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    if config.run_mode not in {"smoke", "full"}:
        errors.append("RUN_MODE must be 'smoke' or 'full'")
    if config.num_envs != 4:
        errors.append("Exactly four training environments are approved")
    if config.steps_per_epoch % config.batch_size:
        errors.append("Rollout buffer size must be divisible by batch_size")
    if config.learning_rate != 3e-4:
        errors.append("Initial learning rate changed from the approved 3e-4")
    if config.gamma != 0.99 or config.gae_lambda != 0.95 or config.clip_range != 0.2:
        errors.append("Frozen PPO gamma/GAE/clip configuration changed")
    if config.ent_coef != 0.02:
        errors.append("Approved entropy coefficient must be 0.02")
    if config.comfort_threshold != 0.5:
        errors.append("Training comfort threshold must be 0.5")
    if config.policy_net != (256, 256):
        errors.append("Frozen policy/value network must be [256, 256]")
    pmv_reference = calculate_pmv_state(
        298.15,
        {
            "air_velocity": 0.1,
            "relative_humidity": 50.0,
            "metabolic_rate": 1.1,
        },
        0.5,
    )
    if pmv_reference != -0.13:
        errors.append(f"ISO PMV scalar equivalence check failed: {pmv_reference}")
    if config.case_key == "mz_hydro":
        if config.reward_weights.get("scaler_comfort") != 10.0:
            errors.append("Hydronic comfort scaler must be 10.0")
        if config.n_steps != 480 or config.batch_size != 120:
            errors.append("Hydronic n_steps/batch_size changed")
        payload = hydronic_action_payload((298.15, 298.15))
        if AUXILIARY_HYDRONIC_KEYS.intersection(payload):
            errors.append("Hydronic action payload contains forbidden auxiliary overwrites")
    elif config.case_key == "mz_air":
        if config.reward_weights.get("scaler_comfort") != 10.0:
            errors.append("MZ Air comfort scaler must remain 10.0")
        if config.n_steps != 672 or config.batch_size != 168:
            errors.append("MZ Air n_steps/batch_size changed")
    else:
        errors.append(f"Unknown case key: {config.case_key}")
    cleanup = cleanup_stale_testids(config)
    failed_cleanup = [row for row in cleanup if row.get("status") == "cleanup_failed"]
    if failed_cleanup:
        warnings.append("One or more stale v2 test IDs could not be stopped")
    mappo = legacy_mappo_status(config.project_root)
    if mappo.get("status") in {"active", "finalizing"}:
        warnings.append(
            f"Legacy MAPPO appears {mappo.get('status')} at epoch "
            f"{mappo.get('epoch')}; online PPO is blocked"
        )
    predecessor = hydro_full_predecessor_status(config.project_root, config.run_tag)
    if (
        config.case_key == "mz_air"
        and config.run_mode == "full"
        and (
            predecessor.get("status") != "complete"
            or int(predecessor.get("actual_epochs", 0)) <= 0
        )
    ):
        errors.append("Hydronic Phase 1.5f full training must complete before MZ_Air full")
    report = {
        "schema": "phase1.5f-preflight-v2",
        "timestamp": utc_now(),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "case": config.case_key,
        "run_mode": config.run_mode,
        "run_dir": str(config.run_dir),
        "config_hash": config.config_hash,
        "scientific_config": config.scientific_payload,
        "runtime_versions": runtime_versions(),
        "legacy_mappo": mappo,
        "hydronic_full_predecessor": predecessor,
        "stale_cleanup": cleanup,
        "boptest": query_boptest_version(config) if online else {"status": "not_queried"},
    }
    atomic_write_json(config.run_dir / "preflight.json", report)
    if errors:
        raise RuntimeError("Phase 1.5f preflight failed: " + "; ".join(errors))
    return report


def audit_forecast_composition(
    config: CaseConfig, forecast_data: Mapping[str, Sequence[float]]
) -> dict[str, Any]:
    start_index = 0
    validation_index = int(7 * 24 * 3600 / config.control_period)
    steps = config.episode_steps
    zones: dict[str, Any] = {}
    for zone in config.zones:
        point = f"Occupancy[{('nZ' if zone == 'nz' else 'sZ') if config.case_key == 'mz_hydro' else zone}]"
        values = np.asarray(forecast_data[point], dtype=float)
        train = values[start_index : start_index + steps] > 0
        validation = values[validation_index : validation_index + steps] > 0
        zones[zone] = {
            "train_occupied_steps": int(np.sum(train)),
            "validation_occupied_steps": int(np.sum(validation)),
            "train_occupied_hours": float(np.sum(train) * config.control_period / 3600),
            "validation_occupied_hours": float(
                np.sum(validation) * config.control_period / 3600
            ),
        }
    report = {
        "timestamp": utc_now(),
        "start_day": config.start_day,
        "validation_start_day": config.start_day + 7,
        "episode_steps": steps,
        "zones": zones,
    }
    atomic_write_json(config.run_dir / "forecast_composition.json", report)
    return report


def environment_fingerprint(
    config: CaseConfig,
    boptest_version: Mapping[str, Any],
    forecast_data: Mapping[str, Sequence[float]],
) -> str:
    forecast_hashes = {
        point: hashlib.sha256(np.asarray(forecast_data[point], dtype=np.float64).tobytes()).hexdigest()
        for point in sorted(config.forecast_points)
    }
    return sha256_payload(
        {
            "config_hash": config.config_hash,
            "boptest": boptest_version,
            "forecast_hashes": forecast_hashes,
        }
    )


class RunLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self):
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if _pid_is_alive(int(payload.get("pid", -1))):
                    raise RuntimeError(
                        f"Another training process owns {self.path}: PID {payload.get('pid')}"
                    )
            except json.JSONDecodeError:
                pass
        atomic_write_json(self.path, {"pid": os.getpid(), "created_at": utc_now()})
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.acquired:
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if int(payload.get("pid", -1)) == os.getpid():
                    self.path.unlink(missing_ok=True)
            except Exception:
                pass


def _install_termination_handler() -> tuple[Any, Any]:
    old_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_termination(signum, frame):
        raise KeyboardInterrupt(f"Received termination signal {signum}")

    signal.signal(signal.SIGTERM, handle_termination)
    return old_sigterm, handle_termination


def _restore_termination_handler(old_sigterm: Any) -> None:
    signal.signal(signal.SIGTERM, old_sigterm)


def _initial_manifest(config: CaseConfig, identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "phase1.5f-run-v2",
        "case": config.case_key,
        "status": "starting",
        "started_at": utc_now(),
        "completed_at": None,
        "config_hash": config.config_hash,
        "scientific_config": config.scientific_payload,
        "wandb_run_id": identity["wandb_run_id"],
        "runtime_versions": runtime_versions(),
        "boptest": {"status": "not_queried"},
        "actual_epochs": 0,
        "global_step": 0,
        "early_stopped": False,
        "resume_count": 0,
        "error": None,
    }


def generate_training_diagnostics(
    config: CaseConfig,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics_dir = config.run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    training = pd.DataFrame(state.get("training_rows", []))
    sb3 = pd.DataFrame(state.get("sb3_rows", []))
    train_eval = pd.DataFrame(state.get("train_eval_rows", []))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not training.empty:
            figure, axis = plt.subplots(figsize=(10, 5))
            axis.plot(training["epoch"], training["reward_mean"], alpha=0.35, label="epoch mean")
            axis.fill_between(
                training["epoch"].to_numpy(dtype=float),
                (training["reward_mean"] - training["reward_std"]).to_numpy(dtype=float),
                (training["reward_mean"] + training["reward_std"]).to_numpy(dtype=float),
                alpha=0.15,
                label="± std across 4 envs",
            )
            axis.plot(training["epoch"], training["rolling_30"], label="rolling 30", linewidth=2)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Training return")
            axis.legend()
            axis.grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(diagnostics_dir / "learning_curve.png", dpi=160)
            plt.close(figure)

            figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for key, label in (
                ("reward_energy_mean", "energy"),
                ("reward_comfort_mean", "comfort"),
                ("reward_smooth_mean", "smooth"),
            ):
                axes[0].plot(training["epoch"], training[key], label=label)
            axes[0].legend()
            axes[0].set_ylabel("Penalty component")
            axes[1].plot(
                training["epoch"], training["occupied_saturation"], label="occupied saturation"
            )
            axes[1].plot(
                training["epoch"], training["unoccupied_saturation"], label="unoccupied saturation"
            )
            axes[1].plot(
                training["epoch"], training["occupancy_action_gap"], label="occupancy action gap"
            )
            axes[1].axhline(0.5, color="red", linestyle=":", alpha=0.5)
            axes[1].axhline(0.1, color="green", linestyle=":", alpha=0.5)
            axes[1].legend()
            axes[1].set_xlabel("Epoch")
            axes[1].grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(diagnostics_dir / "reward_action_health.png", dpi=160)
            plt.close(figure)
        if not sb3.empty:
            figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            axes[0, 0].plot(sb3["epoch"], sb3["explained_variance"])
            axes[0, 0].axhline(0.7, color="green", linestyle=":")
            axes[0, 0].set_title("Explained variance")
            axes[0, 1].plot(sb3["epoch"], sb3["clip_fraction"])
            axes[0, 1].axhspan(0.05, 0.30, alpha=0.12, color="green")
            axes[0, 1].set_title("Clip fraction")
            axes[1, 0].plot(sb3["epoch"], sb3["policy_std"])
            axes[1, 0].axhspan(0.1, 0.4, alpha=0.12, color="green")
            axes[1, 0].set_title("Policy std")
            axes[1, 1].plot(sb3["epoch"], sb3["learning_rate"])
            axes[1, 1].set_title("Learning rate")
            for axis in axes.flat:
                axis.grid(alpha=0.25)
                axis.set_xlabel("Epoch")
            figure.tight_layout()
            figure.savefig(diagnostics_dir / "sb3_health.png", dpi=160)
            plt.close(figure)
    except Exception as exc:
        (diagnostics_dir / "plot_error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )

    report = {
        "case": config.case_key,
        "generated_at": utc_now(),
        "committed_epoch": int(state.get("committed_epoch", 0)),
        "global_step": (
            int(training.iloc[-1]["global_step"]) if not training.empty else 0
        ),
        "best_epoch": state.get("best_epoch"),
        "best_train_week_return": state.get("best_score"),
        "resume_count": int(state.get("resume_count", 0)),
        "convergence": state.get("convergence", {}),
        "train_week_evaluations": len(train_eval),
    }
    atomic_write_json(diagnostics_dir / "training_report.json", report)
    lines = [
        f"# Phase 1.5f training report — {config.case_key}",
        "",
        f"- Committed epoch: {report['committed_epoch']}",
        f"- Global step: {report['global_step']}",
        f"- Best train-week epoch: {report['best_epoch']}",
        f"- Best train-week return: {report['best_train_week_return']}",
        f"- Resume count: {report['resume_count']}",
        f"- Converged: {bool(report['convergence'].get('converged', False))}",
        f"- Remaining convergence checks: {report['convergence'].get('reasons', [])}",
        "",
        "The next-week validation was not used for checkpoint selection.",
    ]
    (diagnostics_dir / "training_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


class Phase15fExperiment:
    def __init__(self, config: CaseConfig):
        self.config = config

    def train(self) -> dict[str, Any]:
        preflight = run_preflight(self.config, online=False)
        if preflight["legacy_mappo"].get("status") in {"active", "finalizing"}:
            raise RuntimeError(
                "Legacy MAPPO is still active. Let it finish before starting an online PPO run."
            )
        checkpoint = AtomicCheckpointManager(
            self.config.run_dir, self.config.config_hash
        )
        latest = checkpoint.load_latest_metadata()
        if latest and not self.config.resume:
            raise RuntimeError(
                "A committed checkpoint already exists and RESUME=False. "
                "Choose a new RUN_TAG for a cold start."
            )
        if latest and (
            latest["state"].get("convergence", {}).get("converged")
            or int(latest["pointer"]["num_timesteps"]) >= self.config.total_cap_steps
        ):
            write_csv(
                self.config.run_dir / "training_metrics.csv",
                latest["state"].get("training_rows", []),
            )
            write_csv(
                self.config.run_dir / "sb3_updates.csv",
                latest["state"].get("sb3_rows", []),
            )
            write_csv(
                self.config.run_dir / "train_week_eval.csv",
                latest["state"].get("train_eval_rows", []),
            )
            generate_training_diagnostics(self.config, latest["state"])
            return latest["state"]

        preferred_run_id = latest["state"].get("wandb_run_id") if latest else None
        identity = get_or_create_run_identity(self.config, preferred_run_id)
        manifest = _initial_manifest(self.config, identity)
        atomic_write_json(self.config.run_dir / "run_manifest.json", manifest)
        wandb_logger = SafeWandb(self.config, identity["wandb_run_id"])
        wandb_logger.start()
        vector_environment: VecEnv | None = None
        callback: Phase15fCallback | None = None
        old_sigterm: Any = None
        lock_path = self.config.run_dir / "runtime" / "training.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        start_wall = time.perf_counter()
        interrupted = False
        online_session_started = False
        try:
            with RunLock(lock_path):
                online_session_started = True
                old_sigterm, _ = _install_termination_handler()
                forecast = ForecastProvider(self.config)
                forecast.prefetch_all()
                audit_forecast_composition(self.config, forecast.data)
                manifest["boptest"] = query_boptest_version(self.config)
                current_environment_fingerprint = environment_fingerprint(
                    self.config,
                    manifest["boptest"],
                    forecast.data,
                )
                manifest["environment_fingerprint"] = current_environment_fingerprint
                if (
                    latest
                    and latest["state"].get("environment_fingerprint")
                    and latest["state"]["environment_fingerprint"]
                    != current_environment_fingerprint
                ):
                    raise RuntimeError(
                        "BOPTEST/forecast environment fingerprint changed; refusing silent resume"
                    )
                atomic_write_json(self.config.run_dir / "run_manifest.json", manifest)
                environment_factories = [
                    partial(_subprocess_environment, self.config, forecast.data, rank)
                    for rank in range(self.config.num_envs)
                ]
                vector_environment = SubprocVecEnv(
                    environment_factories,
                    start_method="spawn",
                )
                if latest:
                    model = Phase15fPPO.load(
                        latest["model_path"],
                        env=vector_environment,
                        device="auto",
                    )
                    if int(model.num_timesteps) != int(latest["pointer"]["num_timesteps"]):
                        raise RuntimeError("Loaded model timestep does not match checkpoint pointer")
                    restored_state = latest["state"]
                else:
                    model = Phase15fPPO(
                        "MlpPolicy",
                        vector_environment,
                        learning_rate=global_linear_schedule(
                            self.config.learning_rate,
                            self.config.final_lr_fraction,
                        ),
                        n_steps=self.config.n_steps,
                        batch_size=self.config.batch_size,
                        n_epochs=self.config.n_epochs,
                        gamma=self.config.gamma,
                        gae_lambda=self.config.gae_lambda,
                        clip_range=self.config.clip_range,
                        ent_coef=self.config.ent_coef,
                        policy_kwargs={
                            "net_arch": {
                                "pi": list(self.config.policy_net),
                                "vf": list(self.config.policy_net),
                            }
                        },
                        tensorboard_log=str(self.config.run_dir / "tensorboard"),
                        seed=self.config.seed,
                        verbose=1,
                        device="auto",
                    )
                    restored_state = None
                callback = Phase15fCallback(
                    self.config,
                    checkpoint,
                    TrainWeekEvaluator(self.config, forecast.data),
                    wandb_logger,
                    environment_fingerprint=current_environment_fingerprint,
                    restored_state=restored_state,
                )
                if latest:
                    restore_rng_state(latest["rng"])
                model._phase15f_hook = callback.on_post_update
                remaining = self.config.total_cap_steps - int(model.num_timesteps)
                if remaining > 0:
                    model.learn(
                        total_timesteps=remaining,
                        reset_num_timesteps=False,
                        callback=callback,
                        tb_log_name=f"{self.config.case_key}_phase1_5f",
                    )
                model._phase15f_hook = None
        except KeyboardInterrupt:
            interrupted = True
            manifest["status"] = "interrupted"
            manifest["error"] = "KeyboardInterrupt"
        except Exception:
            manifest["status"] = "failed"
            manifest["error"] = traceback.format_exc()
            raise
        finally:
            if old_sigterm is not None:
                _restore_termination_handler(old_sigterm)
            if vector_environment is not None:
                try:
                    vector_environment.close()
                except Exception:
                    pass
            # Child workers are now dead; any remaining owner files are stale.
            cleanup_stale_testids(self.config, force=online_session_started)
            lifecycle_path = combine_lifecycle_logs(self.config.run_dir)
            final_metadata = checkpoint.load_latest_metadata()
            final_state = final_metadata["state"] if final_metadata else {}
            if manifest["status"] not in {"failed", "interrupted"}:
                manifest["status"] = "complete"
            manifest["completed_at"] = utc_now()
            manifest["wall_seconds"] = time.perf_counter() - start_wall
            manifest["actual_epochs"] = int(final_state.get("committed_epoch", 0))
            training_rows = final_state.get("training_rows", [])
            manifest["global_step"] = (
                int(training_rows[-1]["global_step"]) if training_rows else 0
            )
            manifest["resume_count"] = int(final_state.get("resume_count", 0))
            manifest["early_stopped"] = bool(
                final_state.get("convergence", {}).get("converged", False)
                and manifest["actual_epochs"] < self.config.max_epochs
            )
            manifest["interrupted"] = interrupted
            manifest["converged"] = bool(
                final_state.get("convergence", {}).get("converged", False)
            )
            manifest["training_outcome"] = (
                "converged"
                if manifest["converged"]
                else (
                    "cap_reached_not_fully_converged"
                    if manifest["actual_epochs"] >= self.config.max_epochs
                    else manifest["status"]
                )
            )
            manifest["cumulative_training_seconds"] = float(
                sum(
                    safe_float(row.get("epoch_wall_seconds"))
                    for row in final_state.get("training_rows", [])
                )
            )
            atomic_write_json(self.config.run_dir / "run_manifest.json", manifest)
            if final_state:
                generate_training_diagnostics(self.config, final_state)
            wandb_logger.finish(
                lifecycle_path,
                {
                    "status": manifest["status"],
                    "actual_epochs": manifest["actual_epochs"],
                    "global_step": manifest["global_step"],
                    "converged": bool(
                        final_state.get("convergence", {}).get("converged", False)
                    ),
                    "interrupted": interrupted,
                },
            )
        final_metadata = checkpoint.load_latest_metadata()
        if not final_metadata:
            raise RuntimeError("Training ended before the first epoch checkpoint was committed")
        return final_metadata["state"]


def _validation_companion_name(config: CaseConfig, prefix: str) -> str:
    if config.case_key == "mz_hydro":
        return f"{prefix}_validation_hydronic_2zone.csv"
    return f"{prefix}_validation_air_5zone.csv"


def _plot_final_validation(
    config: CaseConfig,
    trajectories: Mapping[str, pd.DataFrame],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for label, frame in trajectories.items():
        hours = np.arange(len(frame)) * config.control_period / 3600.0
        axes[0].plot(hours, frame["cost"].cumsum(), label=label)
    axes[0].set_ylabel("Cumulative cost")
    axes[0].legend()
    v2 = trajectories["v2"]
    hours = np.arange(len(v2)) * config.control_period / 3600.0
    for zone in config.zones:
        axes[1].plot(hours, v2[f"pmv_{zone}"], label=f"PMV {zone}")
    axes[1].axhspan(-0.5, 0.5, color="green", alpha=0.12)
    axes[1].set_ylabel("PMV")
    axes[1].legend(ncol=max(1, len(config.zones)))
    for zone in config.zones:
        axes[2].plot(hours, v2[f"setpoint_{zone}"], label=f"SP {zone}")
        occupied = v2[f"occ_{zone}"].to_numpy(dtype=float) > 0
        axes[2].fill_between(
            hours,
            19.5,
            20.0,
            where=occupied,
            alpha=0.15,
            step="post",
        )
    axes[2].set_ylabel("Cooling setpoint [°C]")
    axes[2].set_xlabel("Validation time [h]")
    axes[2].legend(ncol=max(1, len(config.zones)))
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def evaluate_best(config: CaseConfig, *, force: bool = False) -> dict[str, Any]:
    """Run the held-out next week exactly once for RBC, v1 and v2 best."""

    preflight = run_preflight(config, online=False)
    if preflight["legacy_mappo"].get("status") in {"active", "finalizing"}:
        raise RuntimeError("Legacy MAPPO is still active; held-out PPO validation is blocked")
    checkpoint = AtomicCheckpointManager(config.run_dir, config.config_hash)
    best = checkpoint.load_best_metadata()
    results_dir = config.run_dir / "final_validation"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "final_validation_report.json"
    best_manifest_hash = sha256_file(
        checkpoint.directory / best["pointer"]["manifest"]
    )
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("config_hash") == config.config_hash
            and existing.get("best_manifest_sha256") == best_manifest_hash
        ):
            return existing
        raise RuntimeError(
            "A final validation exists for a different best checkpoint. "
            "Use force=True only if an intentional rerun is required."
        )
    cleanup_results = cleanup_stale_testids(config)
    live = [row for row in cleanup_results if row.get("status") == "live_owner_skipped"]
    if live:
        raise RuntimeError("Training IDs are still owned by a live process; final validation is blocked")

    forecast = ForecastProvider(config)
    forecast.prefetch_all()
    v2_model = Phase15fPPO.load(best["model_path"], device="auto")
    trajectories: dict[str, pd.DataFrame] = {}
    metrics: dict[str, Any] = {}

    rbc_frame, rbc_metrics = rollout_policy(
        config,
        forecast.data,
        model=None,
        is_validation=True,
        phase="final_rbc",
        zero_policy=True,
    )
    trajectories["rbc"] = rbc_frame
    metrics["rbc"] = rbc_metrics
    rbc_frame[validation_columns(config)].to_csv(
        results_dir / _validation_companion_name(config, "rbc"), index=False
    )

    legacy_error: str | None = None
    if config.legacy_model_path.exists():
        try:
            legacy_model = PPO.load(config.legacy_model_path, device="auto")
            legacy_frame, legacy_metrics = rollout_policy(
                config,
                forecast.data,
                model=legacy_model,
                is_validation=True,
                phase="final_v1",
                legacy_fixed_baseline=True,
            )
            trajectories["v1"] = legacy_frame
            metrics["v1"] = legacy_metrics
            legacy_frame[validation_columns(config)].to_csv(
                results_dir / _validation_companion_name(config, "ppo_v1"),
                index=False,
            )
        except Exception:
            legacy_error = traceback.format_exc()
    else:
        legacy_error = f"Legacy model not found: {config.legacy_model_path}"

    v2_frame, v2_metrics = rollout_policy(
        config,
        forecast.data,
        model=v2_model,
        is_validation=True,
        phase="final_v2",
    )
    trajectories["v2"] = v2_frame
    metrics["v2"] = v2_metrics
    v2_output = results_dir / config.validation_filename
    v2_frame[validation_columns(config)].to_csv(v2_output, index=False)
    if len(v2_frame) != config.episode_steps:
        raise RuntimeError(
            f"Validation row count {len(v2_frame)} != required {config.episode_steps}"
        )
    if list(pd.read_csv(v2_output, nrows=1).columns) != validation_columns(config):
        raise RuntimeError("Validation CSV schema does not match the approved schema")

    _plot_final_validation(
        config,
        {key: value for key, value in trajectories.items() if key in {"rbc", "v1", "v2"}},
        results_dir / "final_validation.png",
    )
    comparison_rows = []
    for policy, policy_metrics in metrics.items():
        comparison_rows.append({"policy": policy, **policy_metrics})
    write_csv(results_dir / "kpi_comparison.csv", comparison_rows)
    report = {
        "schema": "phase1.5f-final-validation-v2",
        "generated_at": utc_now(),
        "case": config.case_key,
        "config_hash": config.config_hash,
        "best_manifest_sha256": best_manifest_hash,
        "best_epoch": int(best["pointer"]["epoch"]),
        "best_global_step": int(best["pointer"]["num_timesteps"]),
        "validation_start_day": config.start_day + 7,
        "validation_steps": config.episode_steps,
        "deterministic": True,
        "checkpoint_selection": "deterministic training-week return only",
        "metrics": metrics,
        "legacy_error": legacy_error,
        "reward_comparison_warning": (
            "Hydronic v1/v2 training returns are not directly comparable because "
            "the approved comfort scaler changed; use cost and comfort KPIs."
            if config.case_key == "mz_hydro"
            else None
        ),
        "reference_values_are_non_authoritative": True,
        "boptest": query_boptest_version(config),
        "validation_csv": str(v2_output),
    }
    atomic_write_json(report_path, report)
    lines = [
        f"# Phase 1.5f final validation — {config.case_key}",
        "",
        f"- Best training-week checkpoint: epoch {report['best_epoch']}",
        f"- Held-out start day: {report['validation_start_day']}",
        f"- Deterministic rollout rows: {report['validation_steps']}",
        "",
        "| Policy | Cost | Zone-h | PMV·h | Occupied PMV violation |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, values in metrics.items():
        lines.append(
            f"| {policy} | {values['cost']:.6f} | {values['zone_hours']:.3f} | "
            f"{values['pmv_hours']:.3f} | {100*values['occupied_pmv_violation_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "The held-out week was never used for checkpoint selection.",
            "Local same-image/same-week RBC measurements are authoritative; historical absolute "
            "numbers are retained as references only.",
        ]
    )
    if legacy_error:
        lines.extend(["", "## Legacy evaluation warning", "", "The v1 policy could not be evaluated; see JSON report."])
    (results_dir / "final_validation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    combine_lifecycle_logs(config.run_dir)
    return report


def train_ppo(config: CaseConfig) -> dict[str, Any]:
    return Phase15fExperiment(config).train()


def audit_frozen_sz_air(project_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Evidence-only audit; never modifies or retrains SZ_Air."""

    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    notebook_path = root / "CASE_TEST" / "BESTEST_AIR" / "FinalSZAIR.ipynb"
    log_path = (
        root
        / "CASE_TEST"
        / "BESTEST_AIR"
        / "bestest_air_residual_opt"
        / "models_drl4"
        / "ppo_training_log.csv"
    )
    variants: list[dict[str, Any]] = []
    current_model_dir: str | None = None
    notebook_error: str | None = None
    try:
        import nbformat

        notebook = nbformat.read(notebook_path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            source = cell.source
            if "models_drl4/" in source:
                current_model_dir = "models_drl4"
            if "rbc_base = 24.0 if occ > 0 else 30.0" in source:
                variants.append(
                    {
                        "cell": index,
                        "execution_count": cell.execution_count,
                        "baseline": "24C occupied / 30C unoccupied",
                    }
                )
            if "rbc_base = 25.0" in source:
                variants.append(
                    {
                        "cell": index,
                        "execution_count": cell.execution_count,
                        "baseline": "fixed 25C",
                    }
                )
    except Exception:
        notebook_error = traceback.format_exc()
    training_summary: dict[str, Any] = {"status": "missing"}
    if log_path.exists():
        frame = pd.read_csv(log_path)
        training_summary = {
            "status": "available",
            "path": str(log_path),
            "rows": len(frame),
            "first_epoch": int(frame.iloc[0]["Epoch"]),
            "last_epoch": int(frame.iloc[-1]["Epoch"]),
            "first_reward": safe_float(frame.iloc[0]["MeanReward"]),
            "last_reward": safe_float(frame.iloc[-1]["MeanReward"]),
            "columns": list(frame.columns),
            "sha256": sha256_file(log_path),
            "sb3_update_metrics_available": False,
        }
    report = {
        "schema": "phase1.5f-sz-air-evidence-v2",
        "generated_at": utc_now(),
        "retrained": False,
        "notebook": str(notebook_path),
        "notebook_sha256": sha256_file(notebook_path) if notebook_path.exists() else None,
        "notebook_error": notebook_error,
        "current_config_model_dir": current_model_dir,
        "discovered_variants": variants,
        "training_log": training_summary,
        "published_variant": "unresolved_pending_author_confirmation",
        "integrity_note": (
            "The repository contains multiple executed environment variants. Existing evidence "
            "does not uniquely map the published number to 24/30 or fixed-25; no variant is "
            "silently designated and no missing SB3 scalar is fabricated."
        ),
    }
    atomic_write_json(output / "sz_air_audit.json", report)
    lines = [
        "# SZ_Air frozen evidence audit",
        "",
        "- Retrained: no",
        f"- Current Config model directory: {current_model_dir}",
        f"- Published environment variant: {report['published_variant']}",
        f"- Archived training rows: {training_summary.get('rows')}",
        "- Historical SB3 update scalars: unavailable",
        "",
        "Multiple executed variants exist, including 24/30 and fixed-25 baselines. "
        "The available repository evidence does not uniquely identify the published variant, "
        "so it remains explicitly unresolved.",
    ]
    (output / "sz_air_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def archive_legacy_mappo(project_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create a content-addressed manifest after the legacy run reaches epoch 500."""

    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = legacy_mappo_status(root)
    base = (
        root
        / "CASE_TEST"
        / "MZ_OFFICE_HYDRONIC"
        / "multizone_office_simple_hydronic_convergence"
        / "mappo"
    )
    files: list[dict[str, Any]] = []
    if status.get("status") == "complete":
        candidates = [
            base / "run_identity.json",
            base / "logs" / "training.csv",
            base / "logs" / "validation.csv",
        ]
        checkpoint_dir = base / "checkpoints"
        if checkpoint_dir.exists():
            candidates.extend(
                path
                for path in checkpoint_dir.iterdir()
                if path.is_file()
                and not path.name.endswith(".tmp")
                and (
                    path.name.startswith("latest")
                    or path.name.startswith("best")
                    or path.suffix in {".json", ".pt", ".zip"}
                )
            )
        run_id = status.get("run_id")
        lifecycle_dir = base.parent / "lifecycle"
        if run_id and lifecycle_dir.exists():
            candidates.extend(lifecycle_dir.glob(f"mappo_*{run_id}*.jsonl"))
        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)
            files.append(
                {
                    "path": str(resolved.relative_to(root)),
                    "bytes": resolved.stat().st_size,
                    "sha256": sha256_file(resolved),
                }
            )
    report = {
        "schema": "phase1.5f-legacy-mappo-archive-v2",
        "generated_at": utc_now(),
        "classification": "legacy_v1_excluded_from_unified_reward_claim",
        "status": status,
        "files": files,
        "archive_complete": status.get("status") == "complete" and bool(files),
    }
    atomic_write_json(output / "legacy_mappo_manifest.json", report)
    return report
