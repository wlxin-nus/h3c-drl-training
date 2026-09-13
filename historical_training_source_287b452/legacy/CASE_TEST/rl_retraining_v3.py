"""Phase 1.5f v3 Hydronic PPO/MAPPO retraining infrastructure.

The v3 entry point is intentionally Hydronic-only.  It reuses the proven v2
HTTP lifecycle, PMV, observation and SB3 checkpoint primitives while keeping
all v1/v2 notebooks and result directories read-only.  PPO and MAPPO share the
same environment, reward and aux-overwrite-disabled action payload.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import signal
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from torch import nn, optim
from torch.distributions import Normal

from CASE_TEST import rl_retraining_v2 as core


UTC = timezone.utc
AGENT_NAMES = ("nz", "sz")
PUBLISHED_V1_PPO_SHA256 = (
    "826d80b6508dbd63f711bbcaba31971a6ccce36b928d43557e70e565c464bc67"
)
PUBLISHED_V1_PPO_TIMESTEPS = 576_000
PUBLISHED_V1_PPO_REPLAY_COST = 131.49958903778074
PUBLISHED_V1_PPO_REPLAY_SAVING_FRACTION = 0.2586649992506964
SUPERSEDED_V1_PPO_SHA256 = (
    "2f92d6df2dcddbc26a7f0892a950cd92a0507d2bb3240885bfc541f327363c81"
)
LOCAL_OBSERVATION_COLUMNS: dict[str, tuple[str, ...]] = {
    "nz": (
        "obs_sin_time", "obs_cos_time",
        "obs_T_nz", "obs_T_nz_p1", "obs_T_nz_p2", "obs_T_nz_p3", "obs_T_nz_p4",
        "obs_pmv_nz", "obs_act_nz_p1",
        "obs_pow_p1", "obs_pow_p2", "obs_pow_p3", "obs_pow_p4",
        "obs_TDryBul_0", "obs_TDryBul_f1", "obs_TDryBul_f2", "obs_TDryBul_f3", "obs_TDryBul_f4",
        "obs_HGloHor_0", "obs_HGloHor_f1", "obs_HGloHor_f2", "obs_HGloHor_f3", "obs_HGloHor_f4",
        "obs_Price_0", "obs_Price_f1", "obs_Price_f2", "obs_Price_f3", "obs_Price_f4",
        "obs_Occ_nz_0", "obs_Occ_nz_f1", "obs_Occ_nz_f2", "obs_Occ_nz_f3", "obs_Occ_nz_f4",
    ),
    "sz": (
        "obs_sin_time", "obs_cos_time",
        "obs_T_sz", "obs_T_sz_p1", "obs_T_sz_p2", "obs_T_sz_p3", "obs_T_sz_p4",
        "obs_pmv_sz", "obs_act_sz_p1",
        "obs_pow_p1", "obs_pow_p2", "obs_pow_p3", "obs_pow_p4",
        "obs_TDryBul_0", "obs_TDryBul_f1", "obs_TDryBul_f2", "obs_TDryBul_f3", "obs_TDryBul_f4",
        "obs_HGloHor_0", "obs_HGloHor_f1", "obs_HGloHor_f2", "obs_HGloHor_f3", "obs_HGloHor_f4",
        "obs_Price_0", "obs_Price_f1", "obs_Price_f2", "obs_Price_f3", "obs_Price_f4",
        "obs_Occ_sz_0", "obs_Occ_sz_f1", "obs_Occ_sz_f2", "obs_Occ_sz_f3", "obs_Occ_sz_f4",
    ),
}


@dataclasses.dataclass
class HydronicV3Config(core.CaseConfig):
    algorithm: str = "ppo"
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 5e-4
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    legacy_mappo_path: Path = Path("mappo_best.pt")
    calendar_origin_year: int = 2025

    def __post_init__(self) -> None:
        super().__post_init__()
        self.legacy_mappo_path = Path(self.legacy_mappo_path).resolve()

    @property
    def suite_dir(self) -> Path:
        return self.output_root / self.run_mode / self.run_tag

    @property
    def run_dir(self) -> Path:
        return self.suite_dir / self.algorithm

    @property
    def scientific_payload(self) -> dict[str, Any]:
        payload = dict(super().scientific_payload)
        payload.update(
            {
                "schema": "phase1.5f-v3",
                "implementation_sha256": core.sha256_file(Path(__file__).resolve()),
                "algorithm": self.algorithm,
                "actor_learning_rate": self.actor_learning_rate,
                "critic_learning_rate": self.critic_learning_rate,
                "value_loss_coef": self.value_loss_coef,
                "max_grad_norm": self.max_grad_norm,
                "aux_variant": "aux-overwrite-disabled",
                "calendar_origin_year": self.calendar_origin_year,
                "local_observation_columns": LOCAL_OBSERVATION_COLUMNS,
            }
        )
        return payload


@dataclasses.dataclass(frozen=True)
class HydronicV3Suite:
    ppo: HydronicV3Config
    mappo: HydronicV3Config
    force_final_eval: bool = False

    @property
    def run_dir(self) -> Path:
        return self.ppo.suite_dir

    @property
    def final(self) -> HydronicV3Config:
        return dataclasses.replace(self.ppo, algorithm="final")


def _base_field_values(config: core.CaseConfig) -> dict[str, Any]:
    return {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(core.CaseConfig)
    }


def build_hydronic_v3_suite(
    project_root: str | Path,
    *,
    run_mode: str = "smoke",
    resume: bool = True,
    wandb_mode: str = "online",
    seed: int = 42,
    force_final_eval: bool = False,
) -> HydronicV3Suite:
    if seed != 42:
        raise ValueError("Phase 1.5f v3 is approved for seed 42 only")
    base = core.build_hydronic_config(
        project_root,
        run_mode=run_mode,
        resume=resume,
        wandb_mode=wandb_mode,
        run_tag=f"seed{seed}",
    )
    root = Path(project_root).resolve()
    common = _base_field_values(base)
    common.update(
        {
            "output_root": root / "CASE_TEST" / "MZ_OFFICE_HYDRONIC" / "phase1_5f_v3",
            # The top-level archive was overwritten by a later 500-epoch run.
            # This immutable 300-epoch checkpoint is the one that reproduces
            # the paper's 131.50 cost / 25.87% saving result.
            "legacy_model_path": (
                root
                / "CASE_TEST"
                / "MZ_OFFICE_HYDRONIC"
                / "multizone_office_simple_hydronic_residual_opt"
                / "models_drl"
                / "bake"
                / "best_model_ppo.zip"
            ),
            "seed": seed,
            "min_early_stop_epoch": 100,
            "wandb_group": f"phase1_5f_v3_seed{seed}",
        }
    )
    legacy_mappo = (
        root
        / "CASE_TEST"
        / "MZ_OFFICE_HYDRONIC"
        / "multizone_office_simple_hydronic_residual_opt"
        / "models_madrl"
        / "mappo_best.pt"
    )
    ppo = HydronicV3Config(
        **common,
        algorithm="ppo",
        legacy_mappo_path=legacy_mappo,
    )
    mappo_common = dict(common)
    mappo_common.update(
        {
            "learning_rate": 3e-4,
            "batch_size": 120,
            "ent_coef": 0.02,
        }
    )
    mappo = HydronicV3Config(
        **mappo_common,
        algorithm="mappo",
        actor_learning_rate=3e-4,
        critic_learning_rate=5e-4,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        legacy_mappo_path=legacy_mappo,
    )
    return HydronicV3Suite(ppo=ppo, mappo=mappo, force_final_eval=force_final_eval)


def _agent_indices(config: HydronicV3Config) -> dict[str, np.ndarray]:
    return {
        agent: np.asarray(
            [config.observation_columns.index(column) for column in columns],
            dtype=np.int64,
        )
        for agent, columns in LOCAL_OBSERVATION_COLUMNS.items()
    }


class V3ForecastProvider(core.ForecastProvider):
    def __init__(self, config: HydronicV3Config, *, include_validation: bool = False):
        super().__init__(config)
        self.include_validation = include_validation
        required_days = config.simulation_days + (7 if include_validation else 0)
        self.total_horizon = (
            required_days * 24 * 3600 + 5 * config.control_period
        )

    def prefetch_all(self) -> None:
        config = self.config
        algorithm = getattr(config, "algorithm", "v3")
        lifecycle = core.TestIDLifecycle(
            config,
            owner=f"mz_hydro_{algorithm}_forecast",
            algorithm=f"mz_hydro_{algorithm}_v3",
            phase="forecast",
            rank="forecast",
        )
        try:
            lifecycle.configure()
            response = lifecycle.initialize(self.start_time, 0)
            if "payload" not in response:
                raise RuntimeError("BOPTEST initialize returned no payload")
            forecast = core.request_json(
                "put",
                f"{config.boptest_url}/forecast/{lifecycle.testid}",
                json={
                    "point_names": list(config.forecast_points),
                    "horizon": self.total_horizon,
                    "interval": config.control_period,
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
            missing = sorted(set(config.forecast_points) - set(self.data))
            if missing:
                raise RuntimeError(f"Forecast missing points: {missing}")
        finally:
            try:
                lifecycle.stop("forecast_complete")
            except Exception:
                pass


class V3HydronicResidualEnv(core.HydronicResidualEnv):
    """The single shared PPO/MAPPO/RBC Hydronic environment."""

    def __init__(self, config: HydronicV3Config, *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)
        self.lifecycle = core.TestIDLifecycle(
            config,
            owner=f"mz_hydro_{config.algorithm}_{self.phase}_{self.rank}",
            algorithm=f"mz_hydro_{config.algorithm}_v3",
            phase=self.phase,
            rank=self.rank,
        )


def build_environment_v3(
    config: HydronicV3Config,
    forecast_data: Mapping[str, Sequence[float]],
    *,
    is_validation: bool = False,
    legacy_fixed_baseline: bool = False,
    rank: int | str = 0,
    phase: str = "training",
    capture_series: bool = False,
    monitored: bool = False,
) -> gym.Env:
    environment: gym.Env = V3HydronicResidualEnv(
        config,
        forecast_data,
        is_validation=is_validation,
        legacy_fixed_baseline=legacy_fixed_baseline,
        rank=rank,
        phase=phase,
        capture_series=capture_series,
    )
    environment = core.NormalizedObservationWrapper(environment)
    return core.Monitor(environment) if monitored else environment


def _subprocess_environment_v3(
    config: HydronicV3Config,
    forecast_data: Mapping[str, Sequence[float]],
    rank: int,
) -> gym.Env:
    return build_environment_v3(
        config,
        forecast_data,
        rank=rank,
        phase="training",
        monitored=True,
    )


def validation_columns(config: HydronicV3Config) -> list[str]:
    columns = core.validation_columns(config)
    power_index = columns.index("power_total")
    columns.insert(power_index + 1, "energy_step_kwh")
    return columns


def _rollout_row(
    config: HydronicV3Config, info: Mapping[str, Any], reward: float
) -> dict[str, Any]:
    row = core._rollout_row(config, info, reward)
    row["energy_step_kwh"] = (
        core.safe_float(info.get("power_total"))
        * config.control_period
        / 3_600_000.0
    )
    return row


def rollout_policy(
    config: HydronicV3Config,
    forecast_data: Mapping[str, Sequence[float]],
    *,
    model: Any | None,
    is_validation: bool,
    phase: str,
    legacy_fixed_baseline: bool = False,
    zero_policy: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    environment = build_environment_v3(
        config,
        forecast_data,
        is_validation=is_validation,
        legacy_fixed_baseline=legacy_fixed_baseline,
        rank=-1,
        phase=phase,
        capture_series=True,
    )
    rows: list[dict[str, Any]] = []
    actions: list[list[float]] = []
    try:
        observation, _ = environment.reset(seed=config.seed)
        for _ in range(config.episode_steps):
            if zero_policy:
                action = np.zeros(config.n_zones, dtype=np.float32)
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = environment.step(action)
            rows.append(_rollout_row(config, info, reward))
            actions.append(np.asarray(action, dtype=float).reshape(-1).tolist())
            if terminated or truncated:
                break
    finally:
        environment.close()
    frame = pd.DataFrame(rows)
    metrics = core.calculate_rollout_metrics(frame, actions, config)
    frame["_actions"] = [json.dumps(row) for row in actions]
    return frame, metrics


def _median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def evaluate_convergence(
    training_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    train_eval_rows: Sequence[Mapping[str, Any]],
    *,
    best_epoch: int | None,
    min_epoch: int = 100,
    algorithm: str = "ppo",
    eval_interval: int = 25,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "phase1.5f-v3-c1-c4",
        "converged": False,
        "eligible": False,
        "C1": False,
        "C2": False,
        "C3": False,
        "C4": False,
        "reasons": [],
    }
    if not training_rows:
        result["reasons"] = ["no_training_rows"]
        return result
    epoch = int(training_rows[-1]["epoch"])
    if (
        epoch < min_epoch
        or len(training_rows) < 90
        or epoch % eval_interval != 0
        or not train_eval_rows
        or int(train_eval_rows[-1].get("epoch", -1)) != epoch
    ):
        result["reasons"] = ["not_an_eligible_validation_boundary"]
        return result
    result["eligible"] = True

    rewards = np.asarray([core.safe_float(row.get("reward_mean")) for row in training_rows])
    rolling = pd.Series(rewards).rolling(30, min_periods=30).mean().to_numpy()
    current = float(rolling[-1])
    previous = float(rolling[-61]) if len(rolling) >= 61 else float("nan")
    gain_60 = (
        abs(current - previous) / max(abs(previous), 1e-12)
        if np.isfinite(previous)
        else float("inf")
    )
    recent_60 = rolling[-60:]
    recent_60 = recent_60[np.isfinite(recent_60)]
    if recent_60.size >= 30:
        slope = float(np.polyfit(np.arange(recent_60.size), recent_60, 1)[0])
        trend_projection = abs(slope * recent_60.size) / max(abs(current), 1e-12)
    else:
        slope = float("nan")
        trend_projection = float("inf")
    tail_length = max(1, int(math.ceil(0.2 * len(training_rows))))
    tail = rolling[-tail_length:]
    tail = tail[np.isfinite(tail)]
    curve_band = bool(
        tail.size
        and np.all(np.abs(tail - current) <= 0.02 * max(abs(current), 1e-12))
    )
    best_not_late = bool(best_epoch is not None and best_epoch <= 0.9 * epoch)
    result["C1"] = bool(
        gain_60 < 0.01
        and trend_projection < 0.01
        and curve_band
        and best_not_late
    )
    result["C1_details"] = {
        "gain_60": gain_60,
        "ols_slope_60": slope,
        "trend_projection_60": trend_projection,
        "tail_within_terminal_2pct": curve_band,
        "best_not_last_10pct": best_not_late,
        "terminal_rolling_30": current,
    }

    latest_eval = train_eval_rows[-1]
    recent_training = list(training_rows[-60:])
    discomfort = core.safe_float(latest_eval.get("pmv_hours"))
    ratios = [
        abs(core.safe_float(row.get("reward_comfort_mean")))
        / max(abs(core.safe_float(row.get("reward_energy_mean"))), 1e-12)
        for row in recent_training
    ]
    result["C2"] = bool(discomfort <= 0.0 or (ratios and min(ratios) >= 0.05))
    result["C2_details"] = {
        "validation_pmv_hours": discomfort,
        "minimum_recent_comfort_energy_ratio": min(ratios) if ratios else float("nan"),
    }

    occupied_saturation = core.safe_float(latest_eval.get("occupied_saturation"), 1.0)
    occupancy_gap = core.safe_float(latest_eval.get("occupancy_action_gap"), 0.0)
    result["C3"] = bool(occupied_saturation < 0.5 and occupancy_gap > 0.1)
    result["C3_details"] = {
        "occupied_saturation": occupied_saturation,
        "unoccupied_saturation": core.safe_float(latest_eval.get("unoccupied_saturation")),
        "occupancy_action_gap": occupancy_gap,
        "per_zone": latest_eval.get("action_health_by_zone", {}),
    }

    update_tail_length = max(1, int(math.ceil(0.2 * len(update_rows))))
    recent_updates = list(update_rows[-update_tail_length:])
    explained = _median(core.safe_float(row.get("explained_variance"), float("nan")) for row in recent_updates)
    clip_fraction = _median(core.safe_float(row.get("clip_fraction"), float("nan")) for row in recent_updates)
    policy_std = _median(core.safe_float(row.get("policy_std"), float("nan")) for row in recent_updates[-10:])
    learning_rates = np.asarray(
        [core.safe_float(row.get("learning_rate"), float("nan")) for row in update_rows],
        dtype=float,
    )
    learning_rates = learning_rates[np.isfinite(learning_rates)]
    lr_monotonic = bool(
        learning_rates.size
        and np.all(np.diff(learning_rates) <= np.maximum(1e-12, 1e-9 * learning_rates[:-1]))
    )
    actor_health_pass = True
    actor_details: dict[str, Any] = {}
    if algorithm == "mappo":
        for agent in AGENT_NAMES:
            standard_deviation = _median(
                core.safe_float(row.get(f"policy_std_{agent}"), float("nan"))
                for row in recent_updates[-10:]
            )
            actor_clip_fraction = _median(
                core.safe_float(row.get(f"clip_fraction_{agent}"), float("nan"))
                for row in recent_updates
            )
            actor_details[agent] = {
                "final_policy_std_median": standard_deviation,
                "late_clip_fraction_median": actor_clip_fraction,
            }
            actor_health_pass = bool(
                actor_health_pass
                and 0.1 <= standard_deviation <= 0.4
                and 0.05 <= actor_clip_fraction <= 0.30
            )
    result["C4"] = bool(
        explained > 0.7
        and 0.05 <= clip_fraction <= 0.30
        and 0.1 <= policy_std <= 0.4
        and actor_health_pass
        and lr_monotonic
    )
    result["C4_details"] = {
        "late_explained_variance_median": explained,
        "late_clip_fraction_median": clip_fraction,
        "final_policy_std_median": policy_std,
        "actor_policy_std": actor_details,
        "learning_rate_monotonic": lr_monotonic,
    }
    result["reasons"] = [name for name in ("C1", "C2", "C3", "C4") if not result[name]]
    result["converged"] = not result["reasons"]
    return result


def action_health_by_zone(
    frame: pd.DataFrame,
    action_rows: Sequence[Sequence[float]],
    config: HydronicV3Config,
) -> dict[str, dict[str, float]]:
    action_array = np.asarray(action_rows, dtype=float)
    result: dict[str, dict[str, float]] = {}
    for index, zone in enumerate(config.zones):
        occupancy = frame[f"occ_{zone}"].to_numpy(dtype=float) > 0
        values = action_array[:, index]
        occupied = values[occupancy]
        unoccupied = values[~occupancy]
        result[zone] = {
            "occupied_saturation": float(np.mean(np.abs(occupied) > 0.95)) if occupied.size else float("nan"),
            "unoccupied_saturation": float(np.mean(np.abs(unoccupied) > 0.95)) if unoccupied.size else float("nan"),
            "occupied_action_mean": float(np.mean(occupied)) if occupied.size else float("nan"),
            "unoccupied_action_mean": float(np.mean(unoccupied)) if unoccupied.size else float("nan"),
            "occupancy_action_gap": (
                abs(float(np.mean(occupied) - np.mean(unoccupied)))
                if occupied.size and unoccupied.size
                else float("nan")
            ),
        }
    return result


class V3TrainWeekEvaluator:
    def __init__(
        self,
        config: HydronicV3Config,
        forecast_data: Mapping[str, Sequence[float]],
    ):
        self.config = config
        self.forecast_data = forecast_data
        self.directory = config.run_dir / "diagnostics" / "train_week"
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, model: Any, epoch: int, global_step: int) -> dict[str, Any]:
        frame, metrics = rollout_policy(
            self.config,
            self.forecast_data,
            model=model,
            is_validation=False,
            phase=f"train_week_eval_epoch_{epoch:04d}",
        )
        output = self.directory / f"train_week_epoch_{epoch:04d}.csv"
        frame[validation_columns(self.config)].to_csv(output, index=False)
        actions = [json.loads(value) for value in frame["_actions"]]
        return {
            "epoch": int(epoch),
            "global_step": int(global_step),
            **metrics,
            "action_health_by_zone": action_health_by_zone(frame, actions, self.config),
            "trajectory": str(output),
            "timestamp": core.utc_now(),
        }


class V3SafeWandb(core.SafeWandb):
    @property
    def url(self) -> str | None:
        value = getattr(self.run, "url", None) if self.run is not None else None
        return str(value) if value else None

    def start(self) -> None:
        wandb_root = self.config.run_dir / "wandb"
        wandb_root.mkdir(parents=True, exist_ok=True)
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
        modes = [self.mode] + (["offline"] if self.mode == "online" else [])
        for mode in modes:
            try:
                self.run = wandb.init(
                    project=self.config.wandb_project,
                    group=self.config.wandb_group,
                    name=(
                        f"mz-hydro-{self.config.algorithm}-phase1-5f-v3-"
                        f"{self.config.run_mode}-{self.config.run_tag}"
                    ),
                    id=self.run_id,
                    resume="allow",
                    mode=mode,
                    dir=str(wandb_root),
                    config=self.config.scientific_payload,
                    reinit=True,
                    settings=wandb.Settings(
                        init_timeout=15.0,
                        login_timeout=10.0,
                        x_graphql_timeout_seconds=10.0,
                    ),
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


class V3PPOCallback(core.Phase15fCallback):
    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema": "phase1.5f-state-v3-ppo",
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
            "wandb_url": self.wandb.url,
            "environment_fingerprint": self.environment_fingerprint,
            "updated_at": core.utc_now(),
        }

    def on_post_update(self, model: PPO) -> bool:
        epoch = int(model.num_timesteps // self.config.steps_per_epoch)
        if epoch <= self.committed_epoch:
            self._reset_epoch_accumulators()
            return True
        if model.num_timesteps % self.config.steps_per_epoch:
            raise RuntimeError("Post-update checkpoint is not on an epoch boundary")
        training_row = self._training_row(model, epoch)
        update_row = self._sb3_row(model, epoch)
        self.training_rows.append(training_row)
        self.sb3_rows.append(update_row)
        new_best = False
        if epoch % self.config.eval_interval == 0:
            evaluation = self.train_week_evaluator(model, epoch, model.num_timesteps)
            self.train_eval_rows.append(evaluation)
            if core.safe_float(evaluation["return"]) > self.best_score:
                self.best_score = core.safe_float(evaluation["return"])
                self.best_epoch = epoch
                new_best = True
        self.committed_epoch = epoch
        self.convergence = evaluate_convergence(
            self.training_rows,
            self.sb3_rows,
            self.train_eval_rows,
            best_epoch=self.best_epoch,
            min_epoch=self.config.min_early_stop_epoch,
            algorithm="ppo",
            eval_interval=self.config.eval_interval,
        )
        pointer = self.checkpoint.save(model, self._state_payload(), epoch)
        if new_best:
            self.checkpoint.mark_best(pointer, self.best_score)
        core.write_csv(self.config.run_dir / "training_metrics.csv", self.training_rows)
        core.write_csv(self.config.run_dir / "sb3_updates.csv", self.sb3_rows)
        core.write_csv(self.config.run_dir / "train_week_eval.csv", self.train_eval_rows)
        metrics = {
            "epoch": epoch,
            "train/reward_mean": training_row["reward_mean"],
            "train/reward_std": training_row["reward_std"],
            "train/rolling_30": training_row["rolling_30"],
            "reward/energy": training_row["reward_energy_mean"],
            "reward/comfort": training_row["reward_comfort_mean"],
            "reward/smooth": training_row["reward_smooth_mean"],
            "ppo/explained_variance": update_row["explained_variance"],
            "ppo/clip_fraction": update_row["clip_fraction"],
            "ppo/approx_kl": update_row["approx_kl"],
            "ppo/policy_std": update_row["policy_std"],
            "ppo/learning_rate": update_row["learning_rate"],
            "convergence/C1": self.convergence.get("C1", False),
            "convergence/C2": self.convergence.get("C2", False),
            "convergence/C3": self.convergence.get("C3", False),
            "convergence/C4": self.convergence.get("C4", False),
            "convergence/converged": self.convergence.get("converged", False),
        }
        if self.train_eval_rows and int(self.train_eval_rows[-1]["epoch"]) == epoch:
            latest = self.train_eval_rows[-1]
            metrics.update(
                {
                    "train_week/return": latest["return"],
                    "train_week/cost": latest["cost"],
                    "train_week/pmv_hours": latest["pmv_hours"],
                    "train_week/occupied_saturation": latest["occupied_saturation"],
                    "train_week/occupancy_action_gap": latest["occupancy_action_gap"],
                }
            )
        self.wandb.log(metrics, int(model.num_timesteps))
        print(
            f"[HYDRO PPO v3 Epoch {epoch:03d}] "
            f"reward={training_row['reward_mean']:.3f} "
            f"rolling30={training_row['rolling_30']:.3f} "
            f"lr={update_row['learning_rate']:.3e} "
            f"best_train_week={self.best_score:.3f}"
        )
        should_continue = not bool(self.convergence.get("converged"))
        self.started_at = time.perf_counter()
        self._reset_epoch_accumulators()
        return should_continue


def _checkpoint_complete_state(
    config: HydronicV3Config,
    state: Mapping[str, Any],
) -> bool:
    return bool(
        state.get("convergence", {}).get("converged")
        or int(state.get("committed_epoch", 0)) >= config.max_epochs
    )


def _write_run_manifest(
    config: HydronicV3Config,
    identity: Mapping[str, Any],
    **updates: Any,
) -> dict[str, Any]:
    path = config.run_dir / "run_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema": "phase1.5f-run-v3",
            "case": "mz_hydro",
            "algorithm": config.algorithm,
            "started_at": core.utc_now(),
            "config_hash": config.config_hash,
            "scientific_config": config.scientific_payload,
            "wandb_run_id": identity["wandb_run_id"],
            "status": "starting",
        }
    manifest.update(core.json_ready(updates))
    core.atomic_write_json(path, manifest)
    return manifest


def _finish_manifest(
    config: HydronicV3Config,
    identity: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    status: str,
    wall_seconds: float,
    error: str | None = None,
) -> dict[str, Any]:
    epoch = int(state.get("committed_epoch", 0))
    converged = bool(state.get("convergence", {}).get("converged", False))
    outcome = (
        "converged"
        if converged
        else "cap_reached_not_fully_converged"
        if epoch >= config.max_epochs
        else status
    )
    return _write_run_manifest(
        config,
        identity,
        status=status,
        completed_at=core.utc_now(),
        actual_epochs=epoch,
        global_step=int(state.get("global_step", epoch * config.steps_per_epoch)),
        converged=converged,
        early_stopped=bool(converged and epoch < config.max_epochs),
        resume_count=int(state.get("resume_count", 0)),
        training_outcome=outcome,
        wall_seconds=wall_seconds,
        error=error,
    )


def _cell_text(notebook: Mapping[str, Any], index: int) -> str:
    cell = notebook["cells"][index]
    return "".join(cell.get("source", []))


def _cell_output_text(notebook: Mapping[str, Any], index: int) -> str:
    text: list[str] = []
    for output in notebook["cells"][index].get("outputs", []):
        text.extend(output.get("text", []))
        data = output.get("data", {})
        text.extend(data.get("text/plain", []))
    return "".join(text)


def audit_aux_provenance(config: HydronicV3Config) -> dict[str, Any]:
    notebook_path = config.case_dir / "FinalMZHydronic.ipynb"
    rbc_path = (
        config.case_dir
        / "multizone_office_simple_hydronic_residual_opt"
        / "data_rbc"
        / "rbc_validation_hydronic_2zone.csv"
    )
    evidence: dict[str, Any] = {
        "variant": "aux-overwrite-disabled",
        "notebook": str(notebook_path),
        "ppo_cell": 9,
        "evaluation_cell": 10,
        "rbc_cell": 7,
        "passed": False,
    }
    if not notebook_path.exists() or not rbc_path.exists():
        evidence["error"] = "Required archived notebook or RBC CSV is missing"
        return evidence
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    ppo_source = _cell_text(notebook, 9)
    evaluation_source = _cell_text(notebook, 10)
    rbc_source = _cell_text(notebook, 7)
    evaluation_output = _cell_output_text(notebook, 10)

    def active_aux_lines(source: str) -> list[str]:
        active: list[str] = []
        for line in source.splitlines():
            if any(key in line for key in core.AUXILIARY_HYDRONIC_KEYS):
                if not line.lstrip().startswith("#"):
                    active.append(line.strip())
        return active

    ppo_active = active_aux_lines(ppo_source)
    evaluation_active = active_aux_lines(evaluation_source)
    rbc_active = active_aux_lines(rbc_source)
    rbc = pd.read_csv(rbc_path)
    rbc_cost = float(rbc["cost"].sum())
    archived_saving_fraction = 0.2587
    reconstructed_ppo_cost = rbc_cost * (1.0 - archived_saving_fraction)
    output_has_saving = "25.87%" in evaluation_output
    evidence.update(
        {
            "ppo_active_aux_lines": ppo_active,
            "evaluation_active_aux_lines": evaluation_active,
            "rbc_active_aux_lines": rbc_active,
            "archived_rbc_cost": rbc_cost,
            "archived_reported_saving_fraction": archived_saving_fraction,
            "reconstructed_ppo_cost": reconstructed_ppo_cost,
            "matches_131_50": abs(reconstructed_ppo_cost - 131.50) <= 0.1,
            "evaluation_output_contains_25_87pct": output_has_saving,
        }
    )
    evidence["passed"] = bool(
        not ppo_active
        and not evaluation_active
        and not rbc_active
        and output_has_saving
        and evidence["matches_131_50"]
    )
    return evidence


def _sb3_archive_timesteps(path: Path) -> dict[str, int]:
    """Read SB3 progress metadata without loading or mutating the policy."""
    with zipfile.ZipFile(path, "r") as archive:
        payload = json.loads(archive.read("data").decode("utf-8"))
    return {
        "num_timesteps": int(payload.get("num_timesteps", -1)),
        "total_timesteps": int(payload.get("_total_timesteps", -1)),
    }


def audit_legacy_ppo_provenance(config: HydronicV3Config) -> dict[str, Any]:
    """Prove which archived PPO produced the paper's 131.50 result.

    The deterministic replay values below were independently regenerated on
    2026-08-03 with BOPTEST 0.8.0-dev, the 480-step held-out window, the fixed
    legacy 25 C baseline and the common aux-overwrite-disabled payload.
    """
    selected = config.legacy_model_path.resolve()
    expected = (
        config.case_dir
        / "multizone_office_simple_hydronic_residual_opt"
        / "models_drl"
        / "bake"
        / "best_model_ppo.zip"
    ).resolve()
    superseded = expected.parent.parent / "best_model_ppo.zip"
    notebook_path = config.case_dir / "FinalMZHydronic.ipynb"
    report: dict[str, Any] = {
        "schema": "phase1.5f-v1-ppo-provenance-v1",
        "selected_path": str(selected),
        "expected_published_path": str(expected),
        "expected_sha256": PUBLISHED_V1_PPO_SHA256,
        "expected_num_timesteps": PUBLISHED_V1_PPO_TIMESTEPS,
        "recorded_deterministic_replay": {
            "date": "2026-08-03",
            "boptest_version": "0.8.0-dev",
            "rows": 480,
            "cost": PUBLISHED_V1_PPO_REPLAY_COST,
            "saving_fraction_vs_rbc": PUBLISHED_V1_PPO_REPLAY_SAVING_FRACTION,
            "aux_variant": "aux-overwrite-disabled",
            "action_mapping": "legacy-fixed-298.15-K-baseline-plus-minus-5-K",
        },
        "superseded_top_level_candidate": {
            "path": str(superseded),
            "expected_sha256": SUPERSEDED_V1_PPO_SHA256,
            "status": "excluded-later-500-epoch-candidate",
        },
        "passed": False,
    }
    if not selected.exists() or not notebook_path.exists():
        report["error"] = "Published PPO checkpoint or source notebook is missing"
        return report
    try:
        selected_sha = core.sha256_file(selected)
        selected_steps = _sb3_archive_timesteps(selected)
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cell9_output = _cell_output_text(notebook, 9)
        cell10_output = _cell_output_text(notebook, 10)
        superseded_sha = core.sha256_file(superseded) if superseded.exists() else None
        superseded_steps = (
            _sb3_archive_timesteps(superseded) if superseded.exists() else None
        )
    except Exception as exc:
        report["error"] = f"Legacy PPO provenance inspection failed: {exc}"
        return report
    report.update(
        {
            "selected_sha256": selected_sha,
            "selected_archive_progress": selected_steps,
            "notebook_evidence": {
                "cell_10_contains_published_25_87pct": "25.87%" in cell10_output,
                "cell_9_contains_later_31_13pct": "31.13%" in cell9_output,
            },
        }
    )
    report["superseded_top_level_candidate"].update(
        {
            "actual_sha256": superseded_sha,
            "archive_progress": superseded_steps,
        }
    )
    report["passed"] = bool(
        selected == expected
        and selected_sha == PUBLISHED_V1_PPO_SHA256
        and selected_steps["num_timesteps"] == PUBLISHED_V1_PPO_TIMESTEPS
        and report["notebook_evidence"]["cell_10_contains_published_25_87pct"]
        and report["notebook_evidence"]["cell_9_contains_later_31_13pct"]
        and superseded_sha == SUPERSEDED_V1_PPO_SHA256
    )
    return report


def audit_calendar_and_forecast(
    config: HydronicV3Config,
    forecast_data: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    origin = datetime(config.calendar_origin_year, 1, 1)
    windows: dict[str, Any] = {}
    for name, start_day in (
        ("training", config.start_day),
        ("validation", config.start_day + 7),
    ):
        dates = [origin + timedelta(days=start_day + offset) for offset in range(config.simulation_days)]
        windows[name] = {
            "start_day": start_day,
            "dates": [value.date().isoformat() for value in dates],
            "weekdays": [value.strftime("%A") for value in dates],
            "contains_weekend": any(value.weekday() >= 5 for value in dates),
        }
    report: dict[str, Any] = {
        "calendar_origin": f"{config.calendar_origin_year}-01-01",
        "windows": windows,
        "forecast_available": forecast_data is not None,
    }
    if forecast_data is not None:
        validation_offset = int(7 * 24 * 3600 / config.control_period)
        daily_steps = int(24 * 3600 / config.control_period)
        deployed: dict[str, Any] = {}
        for zone, point in (("nz", "Occupancy[nZ]"), ("sz", "Occupancy[sZ]")):
            values = np.asarray(forecast_data[point], dtype=float)
            zone_report: dict[str, Any] = {}
            for name, offset in (("training", 0), ("validation", validation_offset)):
                required = offset + config.simulation_days * daily_steps
                if values.size < required:
                    zone_report[name] = {
                        "available": False,
                        "daily_occupied_hours": [],
                        "total_occupied_hours": None,
                    }
                    continue
                daily_hours = []
                for day in range(config.simulation_days):
                    start = offset + day * daily_steps
                    occupied = values[start : start + daily_steps] > 0
                    daily_hours.append(float(np.sum(occupied) * config.control_period / 3600.0))
                zone_report[name] = {
                    "available": True,
                    "daily_occupied_hours": daily_hours,
                    "total_occupied_hours": float(sum(daily_hours)),
                }
            deployed[zone] = zone_report
        report["deployed_boptest_occupancy"] = deployed
    return report


def _validate_scientific_config(config: HydronicV3Config) -> list[str]:
    errors: list[str] = []
    approved_base = core.build_hydronic_config(
        config.project_root,
        run_mode=config.run_mode,
        resume=config.resume,
        wandb_mode=config.wandb_mode,
        run_tag=config.run_tag,
    )
    if config.case_key != "mz_hydro":
        errors.append("v3 is Hydronic-only")
    if config.algorithm not in {"ppo", "mappo", "final"}:
        errors.append(f"Unknown v3 algorithm: {config.algorithm}")
    if config.run_mode not in {"smoke", "full"}:
        errors.append("RUN_MODE must be smoke or full")
    if config.seed != 42 or config.num_envs != 4:
        errors.append("Only seed 42 with four training environments is approved")
    if config.n_steps != 480 or config.batch_size != 120 or config.n_epochs != 10:
        errors.append("Frozen n_steps/batch_size/n_epochs changed")
    if (config.gamma, config.gae_lambda, config.clip_range) != (0.99, 0.95, 0.2):
        errors.append("Frozen gamma/GAE/clip changed")
    if config.policy_net != (256, 256):
        errors.append("Frozen [256,256] network changed")
    if config.learning_rate != 3e-4 or config.final_lr_fraction != 0.1:
        errors.append("PPO linear 3e-4 to 3e-5 learning-rate schedule changed")
    if config.value_loss_coef != 0.5 or config.max_grad_norm != 0.5:
        errors.append("Frozen value coefficient or gradient clipping changed")
    if config.algorithm == "mappo" and (
        config.actor_learning_rate != 3e-4
        or config.critic_learning_rate != 5e-4
    ):
        errors.append("Frozen MAPPO actor/critic learning rates changed")
    if config.ent_coef != 0.02:
        errors.append("ent_coef must be 0.02")
    if config.reward_weights != {
        "w_energy": 1.0,
        "w_comfort": 20.0,
        "w_smooth": 0.1,
        "scaler_energy": 3.0,
        "scaler_comfort": 10.0,
        "scaler_smooth": 0.1755995,
    }:
        errors.append("Hydronic reward coefficients differ from the approved v3 bundle")
    if config.comfort_threshold != 0.5:
        errors.append("PMV threshold must be 0.5")
    if (
        config.occupied_base_k,
        config.unoccupied_base_k,
        config.residual_scale,
        config.action_min_k,
        config.action_max_k,
    ) != (298.15, 303.15, 5.0, 293.15, 303.15):
        errors.append("Conditional residual mapping changed")
    payloads = {
        name: core.hydronic_action_payload((298.15, 303.15))
        for name in ("ppo", "mappo", "rbc", "v1_ppo", "v1_mappo", "final")
    }
    first = next(iter(payloads.values()))
    if any(payload != first for payload in payloads.values()):
        errors.append("PPO/MAPPO/RBC payload builders are not identical")
    if core.AUXILIARY_HYDRONIC_KEYS.intersection(first):
        errors.append("Forbidden Hydronic aux-overwrite keys are active")
    if config.steps_per_epoch % config.batch_size:
        errors.append("Rollout buffer is not divisible by batch_size")
    for name in (
        "start_day",
        "simulation_days",
        "control_period",
        "n_zones",
        "zones",
        "max_system_power",
        "observation_columns",
        "forecast_points",
        "observations_config",
        "physical_config",
    ):
        if getattr(config, name) != getattr(approved_base, name):
            errors.append(f"Unapproved environment/input change detected: {name}")
    return errors


def audit_smoke_resume_evidence(suite: HydronicV3Suite) -> dict[str, Any]:
    """Report, without inventing evidence, whether live smoke resumed a checkpoint."""
    result: dict[str, Any] = {
        "schema": "phase1.5f-smoke-resume-audit-v1",
        "live_interrupt_resume_exercised": False,
        "algorithms": {},
        "automated_coverage": [
            "PPO checkpoint roundtrip and resume",
            "MAPPO rollout/update/checkpoint/resume and LR continuity",
        ],
    }
    for algorithm in ("ppo", "mappo"):
        manifest_path = (
            suite.ppo.output_root
            / "smoke"
            / suite.ppo.run_tag
            / algorithm
            / "run_manifest.json"
        )
        item: dict[str, Any] = {
            "manifest": str(manifest_path),
            "exists": manifest_path.exists(),
            "resume_count": None,
            "status": "missing",
        }
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                item.update(
                    {
                        "resume_count": int(manifest.get("resume_count", 0)),
                        "status": manifest.get("status"),
                        "config_hash": manifest.get("config_hash"),
                    }
                )
            except Exception as exc:
                item["status"] = "unreadable"
                item["error"] = str(exc)
        result["algorithms"][algorithm] = item
    result["live_interrupt_resume_exercised"] = all(
        int(item.get("resume_count") or 0) >= 1
        for item in result["algorithms"].values()
    )
    result["note"] = (
        "Live smoke interruption/resume is evidenced by resume_count >= 1 for both algorithms."
        if result["live_interrupt_resume_exercised"]
        else "Automated recovery tests pass, but the completed live smoke did not exercise an interruption/resume."
    )
    return result


def run_preflight(
    suite: HydronicV3Suite,
    *,
    online: bool = False,
) -> dict[str, Any]:
    suite.run_dir.mkdir(parents=True, exist_ok=True)
    errors = _validate_scientific_config(suite.ppo) + _validate_scientific_config(suite.mappo)
    cleanup = {
        "ppo": core.cleanup_stale_testids(suite.ppo),
        "mappo": core.cleanup_stale_testids(suite.mappo),
        "final": core.cleanup_stale_testids(suite.final),
    }
    cleanup_failures = [
        row
        for rows in cleanup.values()
        for row in rows
        if row.get("status") == "cleanup_failed"
    ]
    if cleanup_failures:
        errors.append("One or more stale TestIDs could not be stopped")
    aux = audit_aux_provenance(suite.ppo)
    if not aux.get("passed"):
        errors.append("P1 aux-overwrite provenance could not be proven")
    legacy_ppo = audit_legacy_ppo_provenance(suite.ppo)
    if not legacy_ppo.get("passed"):
        errors.append("Published v1 PPO checkpoint identity could not be proven")
    if not suite.ppo.legacy_model_path.exists():
        errors.append(f"Archived v1 PPO model missing: {suite.ppo.legacy_model_path}")
    if not suite.ppo.legacy_mappo_path.exists():
        errors.append(f"Archived v1 MAPPO model missing: {suite.ppo.legacy_mappo_path}")
    forecast_data: Mapping[str, Sequence[float]] | None = None
    boptest: dict[str, Any] = {"status": "not_queried"}
    if online:
        provider = V3ForecastProvider(suite.ppo, include_validation=True)
        provider.prefetch_all()
        forecast_data = provider.data
        boptest = core.query_boptest_version(suite.ppo)
        if boptest.get("status") == "unavailable":
            errors.append("BOPTEST version/name endpoint is unavailable")
    calendar = audit_calendar_and_forecast(suite.ppo, forecast_data)
    online_snapshot_path = suite.run_dir / "preflight_online.json"
    preflight_source = "online-query" if online else "offline-static-check"
    preserved_online_timestamp: str | None = None
    if not online and online_snapshot_path.exists():
        try:
            online_snapshot = json.loads(online_snapshot_path.read_text(encoding="utf-8"))
            snapshot_matches = (
                online_snapshot.get("passed") is True
                and online_snapshot.get("ppo_config_hash") == suite.ppo.config_hash
                and online_snapshot.get("mappo_config_hash") == suite.mappo.config_hash
                and online_snapshot.get("calendar_and_forecast", {}).get(
                    "forecast_available"
                )
                is True
                and online_snapshot.get("boptest", {}).get("endpoint")
                in {"/version", "/name"}
            )
            if snapshot_matches:
                calendar = online_snapshot["calendar_and_forecast"]
                boptest = online_snapshot["boptest"]
                preserved_online_timestamp = online_snapshot.get("timestamp")
                preflight_source = "offline-recheck-with-preserved-online-evidence"
        except Exception:
            # A damaged optional snapshot cannot make the static preflight lie;
            # the next explicit online preflight will replace it atomically.
            pass
    if not calendar["windows"]["training"]["contains_weekend"]:
        errors.append("P2 calendar audit unexpectedly found no training weekend")
    if not calendar["windows"]["validation"]["contains_weekend"]:
        errors.append("P2 calendar audit unexpectedly found no validation weekend")

    air = core.build_air_config(suite.ppo.project_root, run_mode=suite.ppo.run_mode)
    air_isolation = {
        "output_root": str(air.output_root),
        "scaler_energy": air.reward_weights["scaler_energy"],
        "scaler_comfort": air.reward_weights["scaler_comfort"],
        "observation_columns_sha256": core.sha256_payload(air.observation_columns),
        "v3_imports_or_builds_air_environment": False,
    }
    smoke_resume = audit_smoke_resume_evidence(suite)
    warnings: list[str] = []
    if not smoke_resume["live_interrupt_resume_exercised"]:
        warnings.append(
            "Automated resume tests pass, but the recorded live smoke runs have resume_count=0."
        )
    report = {
        "schema": "phase1.5f-preflight-v3",
        "timestamp": core.utc_now(),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "run_mode": suite.ppo.run_mode,
        "seed": suite.ppo.seed,
        "suite_dir": str(suite.run_dir),
        "aux_provenance": aux,
        "legacy_ppo_provenance": legacy_ppo,
        "calendar_and_forecast": calendar,
        "stale_cleanup": cleanup,
        "boptest": boptest,
        "runtime_versions": core.runtime_versions(),
        "ppo_config_hash": suite.ppo.config_hash,
        "mappo_config_hash": suite.mappo.config_hash,
        "mz_air_isolation": air_isolation,
        "preflight_source": preflight_source,
        "preserved_online_timestamp": preserved_online_timestamp,
        "smoke_resume_evidence": smoke_resume,
    }
    if online:
        core.atomic_write_json(online_snapshot_path, report)
    core.atomic_write_json(suite.run_dir / "preflight.json", report)
    first_line = (
        "P1: archived cost 131.50 = published 300-epoch PPO + "
        "aux-overwrite-disabled; RBC, PPO and MAPPO use the same aux-disabled payload."
    )
    notes = [
        first_line,
        (
            "P2: training day 213-217 and validation day 220-224 both include "
            "Saturday and Sunday under the explicit 2025 calendar origin."
        ),
        "Calendar composition and deployed BOPTEST occupancy are recorded separately in preflight.json.",
        (
            "Published v1 PPO is locked to models_drl/bake/best_model_ppo.zip "
            f"(SHA-256 {PUBLISHED_V1_PPO_SHA256})."
        ),
    ]
    (suite.run_dir / "return_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    for warning in warnings:
        print(f"[Phase 1.5f v3 preflight warning] {warning}")
    if errors:
        raise RuntimeError("Phase 1.5f v3 preflight failed: " + "; ".join(errors))
    return report


class V3PPOExperiment:
    def __init__(self, suite: HydronicV3Suite):
        self.suite = suite
        self.config = suite.ppo

    def train(self) -> dict[str, Any]:
        run_preflight(self.suite, online=False)
        config = self.config
        checkpoint = core.AtomicCheckpointManager(config.run_dir, config.config_hash)
        latest = checkpoint.load_latest_metadata()
        if latest and not config.resume:
            raise RuntimeError("PPO checkpoint exists and RESUME=False; use a new result directory")
        if latest and _checkpoint_complete_state(config, latest["state"]):
            core.generate_training_diagnostics(config, latest["state"])
            return latest["state"]
        preferred = latest["state"].get("wandb_run_id") if latest else None
        identity = core.get_or_create_run_identity(config, preferred)
        _write_run_manifest(config, identity, status="starting")
        wandb_logger = V3SafeWandb(config, identity["wandb_run_id"])
        wandb_logger.start()
        vector_environment: VecEnv | None = None
        callback: V3PPOCallback | None = None
        old_sigterm: Any = None
        start_wall = time.perf_counter()
        status = "complete"
        error: str | None = None
        online_started = False
        lock_path = config.run_dir / "runtime" / "training.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with core.RunLock(lock_path):
                online_started = True
                old_sigterm, _ = core._install_termination_handler()
                forecast = V3ForecastProvider(config)
                forecast.prefetch_all()
                composition = audit_calendar_and_forecast(config, forecast.data)
                core.atomic_write_json(config.run_dir / "forecast_composition.json", composition)
                boptest = core.query_boptest_version(config)
                fingerprint = core.environment_fingerprint(config, boptest, forecast.data)
                if latest and latest["state"].get("environment_fingerprint") not in {None, fingerprint}:
                    raise RuntimeError("PPO environment fingerprint changed; refusing silent resume")
                _write_run_manifest(
                    config,
                    identity,
                    status="training",
                    boptest=boptest,
                    environment_fingerprint=fingerprint,
                )
                factories = [
                    partial(_subprocess_environment_v3, config, forecast.data, rank)
                    for rank in range(config.num_envs)
                ]
                vector_environment = SubprocVecEnv(factories, start_method="spawn")
                if latest:
                    model = core.Phase15fPPO.load(
                        latest["model_path"], env=vector_environment, device="auto"
                    )
                    if int(model.num_timesteps) != int(latest["pointer"]["num_timesteps"]):
                        raise RuntimeError("Loaded PPO timestep does not match checkpoint pointer")
                    restored_state = latest["state"]
                else:
                    model = core.Phase15fPPO(
                        "MlpPolicy",
                        vector_environment,
                        learning_rate=core.global_linear_schedule(
                            config.learning_rate, config.final_lr_fraction
                        ),
                        n_steps=config.n_steps,
                        batch_size=config.batch_size,
                        n_epochs=config.n_epochs,
                        gamma=config.gamma,
                        gae_lambda=config.gae_lambda,
                        clip_range=config.clip_range,
                        ent_coef=config.ent_coef,
                        vf_coef=0.5,
                        max_grad_norm=config.max_grad_norm,
                        policy_kwargs={
                            "net_arch": {
                                "pi": list(config.policy_net),
                                "vf": list(config.policy_net),
                            }
                        },
                        tensorboard_log=str(config.run_dir / "tensorboard"),
                        seed=config.seed,
                        verbose=1,
                        device="auto",
                    )
                    restored_state = None
                callback = V3PPOCallback(
                    config,
                    checkpoint,
                    V3TrainWeekEvaluator(config, forecast.data),
                    wandb_logger,
                    environment_fingerprint=fingerprint,
                    restored_state=restored_state,
                )
                if latest:
                    core.restore_rng_state(latest["rng"])
                model._phase15f_hook = callback.on_post_update
                remaining = config.total_cap_steps - int(model.num_timesteps)
                if remaining > 0:
                    model.learn(
                        total_timesteps=remaining,
                        reset_num_timesteps=False,
                        callback=callback,
                        tb_log_name="mz_hydro_ppo_phase1_5f_v3",
                    )
                model._phase15f_hook = None
        except KeyboardInterrupt:
            status = "interrupted"
            error = "KeyboardInterrupt"
        except Exception:
            status = "failed"
            error = traceback.format_exc()
            raise
        finally:
            if old_sigterm is not None:
                core._restore_termination_handler(old_sigterm)
            if vector_environment is not None:
                try:
                    vector_environment.close()
                except Exception:
                    pass
            core.cleanup_stale_testids(config, force=online_started)
            lifecycle_path = core.combine_lifecycle_logs(config.run_dir)
            committed = checkpoint.load_latest_metadata()
            state = committed["state"] if committed else {}
            if state:
                state = dict(state)
                state["global_step"] = int(state.get("committed_epoch", 0)) * config.steps_per_epoch
                core.generate_training_diagnostics(config, state)
            _finish_manifest(
                config,
                identity,
                state,
                status=status,
                wall_seconds=time.perf_counter() - start_wall,
                error=error,
            )
            wandb_logger.finish(
                lifecycle_path,
                {
                    "status": status,
                    "actual_epochs": int(state.get("committed_epoch", 0)),
                    "converged": bool(state.get("convergence", {}).get("converged", False)),
                },
            )
        final = checkpoint.load_latest_metadata()
        if not final:
            raise RuntimeError("PPO ended before the first epoch checkpoint was committed")
        return final["state"]


def train_ppo(suite: HydronicV3Suite) -> dict[str, Any]:
    return V3PPOExperiment(suite).train()


class ActorNetwork(nn.Module):
    """Original two-layer MAPPO Gaussian actor used by the archived run."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 1,
        hidden_sizes: Sequence[int] = (256, 256),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = obs_dim
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.ReLU()))
            previous = hidden
        self.backbone = nn.Sequential(*layers)
        self.mean_linear = nn.Linear(previous, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.mean_linear.weight, gain=0.01)
        nn.init.zeros_(self.mean_linear.bias)

    def forward(self, observation: torch.Tensor) -> Normal:
        mean = torch.tanh(self.mean_linear(self.backbone(observation)))
        standard_deviation = torch.clamp(self.log_std.exp(), min=0.01, max=1.0)
        return Normal(mean, standard_deviation)

    def get_action(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self(observation)
        action = distribution.mean if deterministic else distribution.rsample()
        return (
            action,
            distribution.log_prob(action).sum(dim=-1),
            distribution.entropy().sum(dim=-1),
        )

    def evaluate_action(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self(observation)
        return (
            distribution.log_prob(action).sum(dim=-1),
            distribution.entropy().sum(dim=-1),
        )


class CentralizedCritic(nn.Module):
    """Original centralized value network over the unchanged global input."""

    def __init__(
        self,
        global_obs_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = global_obs_dim
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.ReLU()))
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        linear_layers = [module for module in self.network if isinstance(module, nn.Linear)]
        for module in linear_layers:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(module.bias)
        nn.init.orthogonal_(linear_layers[-1].weight, gain=1.0)

    def forward(self, global_observation: torch.Tensor) -> torch.Tensor:
        return self.network(global_observation).squeeze(-1)


class MAPPORolloutBuffer:
    """MAPPO rollout storage with explicit termination and time-limit state."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        agent_obs_dims: Mapping[str, int],
        global_obs_dim: int,
        device: str | torch.device = "cpu",
    ):
        self.n_steps = int(n_steps)
        self.n_envs = int(n_envs)
        self.device = torch.device(device)
        self.ptr = 0
        self.agent_obs = {
            name: np.zeros((n_steps, n_envs, dimension), dtype=np.float32)
            for name, dimension in agent_obs_dims.items()
        }
        self.agent_actions = {
            name: np.zeros((n_steps, n_envs, 1), dtype=np.float32)
            for name in agent_obs_dims
        }
        self.agent_log_probs = {
            name: np.zeros((n_steps, n_envs), dtype=np.float32)
            for name in agent_obs_dims
        }
        self.global_obs = np.zeros((n_steps, n_envs, global_obs_dim), dtype=np.float32)
        self.terminal_observations = np.zeros(
            (n_steps, n_envs, global_obs_dim), dtype=np.float32
        )
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.next_values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.terminated = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.truncated = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

    def add(
        self,
        *,
        agent_obs: Mapping[str, np.ndarray],
        agent_actions: Mapping[str, np.ndarray],
        agent_log_probs: Mapping[str, np.ndarray],
        global_obs: np.ndarray,
        terminal_observations: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        if self.ptr >= self.n_steps:
            raise RuntimeError("MAPPO rollout buffer overflow")
        index = self.ptr
        for name in self.agent_obs:
            self.agent_obs[name][index] = agent_obs[name]
            self.agent_actions[name][index] = agent_actions[name]
            self.agent_log_probs[name][index] = agent_log_probs[name]
        self.global_obs[index] = global_obs
        self.terminal_observations[index] = terminal_observations
        self.rewards[index] = rewards
        self.values[index] = values
        self.next_values[index] = next_values
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.ptr += 1

    def compute_gae(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """Bootstrap truncations, while preventing traces from crossing resets."""
        next_advantage = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.ptr)):
            bootstrap_mask = 1.0 - self.terminated[step]
            trace_mask = bootstrap_mask * (1.0 - self.truncated[step])
            delta = (
                self.rewards[step]
                + gamma * bootstrap_mask * self.next_values[step]
                - self.values[step]
            )
            next_advantage = delta + gamma * gae_lambda * trace_mask * next_advantage
            self.advantages[step] = next_advantage
        self.returns[: self.ptr] = (
            self.advantages[: self.ptr] + self.values[: self.ptr]
        )

    def batches(
        self,
        batch_size: int,
        *,
        generator: np.random.Generator,
    ) -> Iterator[dict[str, torch.Tensor]]:
        total = self.ptr * self.n_envs
        indices = generator.permutation(total)
        flattened: dict[str, np.ndarray] = {
            "global_obs": self.global_obs[: self.ptr].reshape(total, -1),
            "advantages": self.advantages[: self.ptr].reshape(total),
            "returns": self.returns[: self.ptr].reshape(total),
            "values": self.values[: self.ptr].reshape(total),
        }
        for name in self.agent_obs:
            flattened[f"obs_{name}"] = self.agent_obs[name][: self.ptr].reshape(total, -1)
            flattened[f"actions_{name}"] = self.agent_actions[name][: self.ptr].reshape(total, 1)
            flattened[f"log_probs_{name}"] = self.agent_log_probs[name][: self.ptr].reshape(total)
        for start in range(0, total, batch_size):
            selected = indices[start : start + batch_size]
            yield {
                key: torch.as_tensor(value[selected], dtype=torch.float32, device=self.device)
                for key, value in flattened.items()
            }


class AtomicTorchCheckpointManager:
    """Retrying, checksummed, same-directory atomic MAPPO checkpoint bundles."""

    def __init__(
        self,
        run_dir: Path,
        config_hash: str,
        *,
        save_retries: int = 5,
        replace_func: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None] = os.replace,
    ):
        self.config_hash = config_hash
        self.save_retries = int(save_retries)
        self.replace_func = replace_func
        self.directory = Path(run_dir) / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / "latest.json"
        self.best_path = self.directory / "best.json"

    def _replace(self, source: Path, destination: Path) -> None:
        self.replace_func(source, destination)

    def _write_pointer(self, destination: Path, payload: Mapping[str, Any]) -> None:
        temporary = self.directory / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(core.json_ready(payload), handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            self._replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _verify_payload(path: Path) -> Mapping[str, Any]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Empty MAPPO checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "actors" not in payload or "critic" not in payload:
            raise RuntimeError(f"Incomplete MAPPO checkpoint: {path}")
        return payload

    def save(
        self,
        payload: Mapping[str, Any],
        *,
        epoch: int,
        global_step: int,
    ) -> dict[str, Any]:
        prefix = f"epoch_{epoch:04d}_steps_{global_step:09d}"
        bundle_name = f"{prefix}.mappo.pt"
        manifest_name = f"{prefix}.manifest.json"
        final_bundle = self.directory / bundle_name
        final_manifest = self.directory / manifest_name
        last_error: Exception | None = None
        for attempt in range(self.save_retries):
            token = uuid.uuid4().hex
            temporary_bundle = self.directory / f".bundle.{token}.tmp.pt"
            temporary_manifest = self.directory / f".manifest.{token}.tmp.json"
            try:
                with temporary_bundle.open("wb") as handle:
                    torch.save(dict(payload), handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                loaded = self._verify_payload(temporary_bundle)
                if int(loaded.get("committed_epoch", -1)) != int(epoch):
                    raise RuntimeError("Serialized MAPPO epoch failed integrity check")
                manifest = {
                    "schema": "phase1.5f-checkpoint-v3-mappo",
                    "created_at": core.utc_now(),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "config_hash": self.config_hash,
                    "bundle": {
                        "name": bundle_name,
                        "sha256": core.sha256_file(temporary_bundle),
                    },
                }
                with temporary_manifest.open("w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, indent=2, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._replace(temporary_bundle, final_bundle)
                self._replace(temporary_manifest, final_manifest)
                self._verify_payload(final_bundle)
                if core.sha256_file(final_bundle) != manifest["bundle"]["sha256"]:
                    raise RuntimeError("MAPPO checkpoint checksum mismatch after commit")
                pointer = {
                    "schema": "phase1.5f-pointer-v3-mappo",
                    "manifest": manifest_name,
                    "manifest_sha256": core.sha256_file(final_manifest),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "config_hash": self.config_hash,
                    "committed_at": core.utc_now(),
                }
                self._write_pointer(self.latest_path, pointer)
                self._prune()
                return pointer
            except Exception as exc:
                last_error = exc
                temporary_bundle.unlink(missing_ok=True)
                temporary_manifest.unlink(missing_ok=True)
                if attempt + 1 < self.save_retries:
                    time.sleep(0.25 * (2**attempt))
        raise RuntimeError(
            f"Atomic MAPPO checkpoint failed after {self.save_retries} attempts; "
            "the prior latest pointer remains authoritative"
        ) from last_error

    def mark_best(self, pointer: Mapping[str, Any], score: float) -> None:
        manifest_path = self.directory / str(pointer["manifest"])
        best = {
            **dict(pointer),
            "schema": "phase1.5f-best-v3-mappo",
            "manifest_sha256": core.sha256_file(manifest_path),
            "score": float(score),
            "committed_at": core.utc_now(),
        }
        self._write_pointer(self.best_path, best)
        self._prune()

    def _read(self, pointer_path: Path) -> dict[str, Any]:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if pointer.get("config_hash") != self.config_hash:
            raise RuntimeError("MAPPO checkpoint config hash mismatch; refusing silent resume")
        manifest_path = self.directory / str(pointer["manifest"])
        if core.sha256_file(manifest_path) != pointer["manifest_sha256"]:
            raise RuntimeError("MAPPO manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != self.config_hash:
            raise RuntimeError("MAPPO checkpoint manifest config hash mismatch")
        bundle_path = self.directory / manifest["bundle"]["name"]
        if core.sha256_file(bundle_path) != manifest["bundle"]["sha256"]:
            raise RuntimeError("MAPPO bundle checksum mismatch")
        payload = dict(self._verify_payload(bundle_path))
        if int(payload.get("committed_epoch", -1)) != int(pointer["epoch"]):
            raise RuntimeError("MAPPO pointer and payload epoch mismatch")
        return {
            "pointer": pointer,
            "manifest": manifest,
            "payload": payload,
            "bundle_path": bundle_path,
        }

    def load_latest(self) -> dict[str, Any] | None:
        return self._read(self.latest_path) if self.latest_path.exists() else None

    def load_best(self) -> dict[str, Any]:
        if not self.best_path.exists():
            raise FileNotFoundError("No MAPPO train-week best checkpoint exists")
        return self._read(self.best_path)

    def _prune(self) -> None:
        manifests = sorted(self.directory.glob("epoch_*.manifest.json"))
        keep = {path.name for path in manifests[-2:]}
        for pointer_path in (self.latest_path, self.best_path):
            if pointer_path.exists():
                try:
                    keep.add(json.loads(pointer_path.read_text(encoding="utf-8"))["manifest"])
                except Exception:
                    pass
        for manifest_path in manifests:
            if manifest_path.name in keep:
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                (self.directory / manifest["bundle"]["name"]).unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
            except Exception:
                pass


class MAPPOPolicyAdapter:
    """Expose the two MAPPO actors through the deterministic SB3-style API."""

    def __init__(
        self,
        actors: Mapping[str, ActorNetwork],
        config: HydronicV3Config,
        device: str | torch.device = "cpu",
    ):
        self.actors = dict(actors)
        self.config = config
        self.device = torch.device(device)
        self.indices = _agent_indices(config)

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        global_observation = np.asarray(observation, dtype=np.float32)
        single = global_observation.ndim == 1
        if single:
            global_observation = global_observation[None, :]
        actions: list[np.ndarray] = []
        with torch.no_grad():
            for name in AGENT_NAMES:
                local = torch.as_tensor(
                    global_observation[:, self.indices[name]],
                    dtype=torch.float32,
                    device=self.device,
                )
                raw, _, _ = self.actors[name].get_action(
                    local, deterministic=deterministic
                )
                actions.append(raw.cpu().numpy().reshape(-1))
        joint = np.stack(actions, axis=1).clip(-1.0, 1.0).astype(np.float32)
        return (joint[0] if single else joint), None


def _linear_epoch_lambda(max_epochs: int) -> Callable[[int], float]:
    def schedule(completed_updates: int) -> float:
        return max(0.1, 1.0 - 0.9 * completed_updates / max(max_epochs, 1))

    return schedule


def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    variance = float(np.var(returns))
    if variance <= 1e-12:
        return float("nan")
    return float(1.0 - np.var(returns - values) / variance)


class MAPPOTrainer:
    """Four-environment MAPPO with the approved v3 science and GAE semantics."""

    def __init__(
        self,
        config: HydronicV3Config,
        environments: Sequence[gym.Env],
        forecast_data: Mapping[str, Sequence[float]],
        checkpoint: AtomicTorchCheckpointManager,
        wandb_logger: V3SafeWandb,
        *,
        environment_fingerprint: str,
        restored: Mapping[str, Any] | None = None,
        device: str | torch.device | None = None,
    ):
        self.config = config
        self.environments = list(environments)
        self.forecast_data = forecast_data
        self.checkpoint = checkpoint
        self.wandb = wandb_logger
        self.environment_fingerprint = environment_fingerprint
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.indices = _agent_indices(config)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        self.generator = np.random.default_rng(config.seed)
        self.actors = {
            name: ActorNetwork(
                len(LOCAL_OBSERVATION_COLUMNS[name]),
                hidden_sizes=config.policy_net,
            ).to(self.device)
            for name in AGENT_NAMES
        }
        self.critic = CentralizedCritic(
            len(config.observation_columns), hidden_sizes=config.policy_net
        ).to(self.device)
        self.actor_optimizers = {
            name: optim.Adam(actor.parameters(), lr=config.actor_learning_rate, eps=1e-5)
            for name, actor in self.actors.items()
        }
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate, eps=1e-5
        )
        schedule = _linear_epoch_lambda(config.max_epochs)
        self.actor_schedulers = {
            name: optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
            for name, optimizer in self.actor_optimizers.items()
        }
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(
            self.critic_optimizer, lr_lambda=schedule
        )
        self.buffer = MAPPORolloutBuffer(
            config.n_steps,
            config.num_envs,
            {name: len(columns) for name, columns in LOCAL_OBSERVATION_COLUMNS.items()},
            len(config.observation_columns),
            self.device,
        )
        restored_payload = dict(restored or {})
        self.committed_epoch = int(restored_payload.get("committed_epoch", 0))
        self.global_step = int(restored_payload.get("global_step", 0))
        self.training_rows: list[dict[str, Any]] = list(
            restored_payload.get("training_rows", [])
        )
        self.update_rows: list[dict[str, Any]] = list(
            restored_payload.get("update_rows", restored_payload.get("sb3_rows", []))
        )
        self.train_eval_rows: list[dict[str, Any]] = list(
            restored_payload.get("train_eval_rows", [])
        )
        self.best_score = core.safe_float(
            restored_payload.get("best_score"), -float("inf")
        )
        self.best_epoch = (
            int(restored_payload["best_epoch"])
            if restored_payload.get("best_epoch") is not None
            else None
        )
        self.convergence = dict(
            restored_payload.get("convergence", {"converged": False})
        )
        self.resume_count = int(restored_payload.get("resume_count", 0)) + int(
            bool(restored)
        )
        if restored:
            self._restore(restored_payload)
        self.observations: np.ndarray | None = None
        self.writer: Any | None = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(config.run_dir / "tensorboard"))
        except Exception:
            self.writer = None

    def _restore(self, payload: Mapping[str, Any]) -> None:
        if payload.get("config_hash") != self.config.config_hash:
            raise RuntimeError("MAPPO payload config hash mismatch")
        if payload.get("environment_fingerprint") != self.environment_fingerprint:
            raise RuntimeError("MAPPO environment fingerprint changed; refusing silent resume")
        for name in AGENT_NAMES:
            self.actors[name].load_state_dict(payload["actors"][name])
            self.actor_optimizers[name].load_state_dict(payload["actor_optimizers"][name])
            self.actor_schedulers[name].load_state_dict(payload["actor_schedulers"][name])
        self.critic.load_state_dict(payload["critic"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.critic_scheduler.load_state_dict(payload["critic_scheduler"])
        if "rng" in payload:
            core.restore_rng_state(payload["rng"])
        if "generator_state" in payload:
            self.generator.bit_generator.state = payload["generator_state"]

    def _reset_all(self) -> np.ndarray:
        observations: list[np.ndarray | None] = [None] * len(self.environments)
        with ThreadPoolExecutor(max_workers=len(self.environments)) as executor:
            futures = {
                executor.submit(environment.reset, seed=self.config.seed + index): index
                for index, environment in enumerate(self.environments)
            }
            for future in as_completed(futures):
                index = futures[future]
                observation, _ = future.result()
                observations[index] = np.asarray(observation, dtype=np.float32)
        return np.stack([value for value in observations if value is not None])

    def _parallel_step(
        self, actions: np.ndarray
    ) -> list[tuple[np.ndarray, float, bool, bool, dict[str, Any]]]:
        results: list[Any] = [None] * len(self.environments)
        with ThreadPoolExecutor(max_workers=len(self.environments)) as executor:
            futures = {
                executor.submit(environment.step, actions[index]): index
                for index, environment in enumerate(self.environments)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def _reset_done(
        self, next_observations: np.ndarray, done: np.ndarray
    ) -> np.ndarray:
        indices = np.where(done)[0].tolist()
        if not indices:
            return next_observations
        with ThreadPoolExecutor(max_workers=len(indices)) as executor:
            futures = {
                executor.submit(
                    self.environments[index].reset,
                    seed=self.config.seed + index + self.committed_epoch + 1,
                ): index
                for index in indices
            }
            for future in as_completed(futures):
                observation, _ = future.result()
                next_observations[futures[future]] = np.asarray(
                    observation, dtype=np.float32
                )
        return next_observations

    def _collect_rollout(self) -> dict[str, Any]:
        if self.observations is None:
            self.observations = self._reset_all()
        self.buffer.ptr = 0
        episode_returns = np.zeros(self.config.num_envs, dtype=float)
        completed_returns: list[float] = []
        components = {"energy": [], "comfort": [], "smooth": []}
        action_samples: list[tuple[int, float, float]] = []
        for _ in range(self.config.n_steps):
            global_observations = self.observations.copy()
            tensor_global = torch.as_tensor(
                global_observations, dtype=torch.float32, device=self.device
            )
            local_observations: dict[str, np.ndarray] = {}
            raw_actions: dict[str, np.ndarray] = {}
            old_log_probs: dict[str, np.ndarray] = {}
            with torch.no_grad():
                values = self.critic(tensor_global).cpu().numpy()
                for name in AGENT_NAMES:
                    local = global_observations[:, self.indices[name]]
                    local_observations[name] = local.copy()
                    action, log_probability, _ = self.actors[name].get_action(
                        torch.as_tensor(local, dtype=torch.float32, device=self.device)
                    )
                    raw_actions[name] = action.cpu().numpy()
                    old_log_probs[name] = log_probability.cpu().numpy()
            joint_actions = np.stack(
                [raw_actions[name].reshape(-1) for name in AGENT_NAMES], axis=1
            ).clip(-1.0, 1.0)
            results = self._parallel_step(joint_actions)
            terminal_observations = np.stack(
                [np.asarray(result[0], dtype=np.float32) for result in results]
            )
            rewards = np.asarray([result[1] for result in results], dtype=np.float32)
            terminated = np.asarray([result[2] for result in results], dtype=np.float32)
            truncated = np.asarray([result[3] for result in results], dtype=np.float32)
            infos = [result[4] for result in results]
            with torch.no_grad():
                next_values = self.critic(
                    torch.as_tensor(
                        terminal_observations,
                        dtype=torch.float32,
                        device=self.device,
                    )
                ).cpu().numpy()
            self.buffer.add(
                agent_obs=local_observations,
                agent_actions=raw_actions,
                agent_log_probs=old_log_probs,
                global_obs=global_observations,
                terminal_observations=terminal_observations,
                rewards=rewards,
                values=values,
                next_values=next_values,
                terminated=terminated,
                truncated=truncated,
            )
            episode_returns += rewards
            for environment_index, info in enumerate(infos):
                components["energy"].append(core.safe_float(info.get("rew_energy")))
                components["comfort"].append(core.safe_float(info.get("rew_comfort")))
                components["smooth"].append(core.safe_float(info.get("rew_smooth")))
                for zone_index, (action, occupancy) in enumerate(
                    zip(info.get("actions", []), info.get("occupancies", []))
                ):
                    action_samples.append(
                        (zone_index, core.safe_float(action), core.safe_float(occupancy))
                    )
                if terminated[environment_index] or truncated[environment_index]:
                    completed_returns.append(float(episode_returns[environment_index]))
                    episode_returns[environment_index] = 0.0
            done = (terminated + truncated) > 0
            self.observations = self._reset_done(terminal_observations.copy(), done)
            self.global_step += self.config.num_envs
        self.buffer.compute_gae(self.config.gamma, self.config.gae_lambda)
        returns = completed_returns or episode_returns.tolist()
        action_health = core._summarize_action_health(
            action_samples, self.config.n_zones
        )
        return {
            "reward_mean": float(np.mean(returns)),
            "reward_std": float(np.std(returns)),
            "reward_energy_mean": float(np.mean(components["energy"])),
            "reward_comfort_mean": float(np.mean(components["comfort"])),
            "reward_smooth_mean": float(np.mean(components["smooth"])),
            **action_health,
        }

    def _update(self, epoch: int) -> dict[str, Any]:
        advantages = self.buffer.advantages[: self.buffer.ptr]
        advantage_mean = float(np.mean(advantages))
        advantage_std = float(np.std(advantages)) + 1e-8
        self.buffer.advantages[: self.buffer.ptr] = (
            advantages - advantage_mean
        ) / advantage_std
        actor_losses = {name: [] for name in AGENT_NAMES}
        actor_gradients = {name: [] for name in AGENT_NAMES}
        approximate_kls = {name: [] for name in AGENT_NAMES}
        clip_fractions = {name: [] for name in AGENT_NAMES}
        entropies = {name: [] for name in AGENT_NAMES}
        critic_losses: list[float] = []
        critic_gradients: list[float] = []
        for _ in range(self.config.n_epochs):
            for batch in self.buffer.batches(
                self.config.batch_size, generator=self.generator
            ):
                predicted_values = self.critic(batch["global_obs"])
                critic_loss = torch.mean((predicted_values - batch["returns"]) ** 2)
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_loss_coef * critic_loss).backward()
                critic_gradient = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                self.critic_optimizer.step()
                critic_losses.append(float(critic_loss.detach().cpu()))
                critic_gradients.append(float(critic_gradient.detach().cpu()))
                for name in AGENT_NAMES:
                    log_probability, entropy = self.actors[name].evaluate_action(
                        batch[f"obs_{name}"], batch[f"actions_{name}"]
                    )
                    log_ratio = log_probability - batch[f"log_probs_{name}"]
                    ratio = torch.exp(log_ratio)
                    policy_loss_1 = -batch["advantages"] * ratio
                    policy_loss_2 = -batch["advantages"] * torch.clamp(
                        ratio,
                        1.0 - self.config.clip_range,
                        1.0 + self.config.clip_range,
                    )
                    actor_loss = (
                        torch.max(policy_loss_1, policy_loss_2).mean()
                        - self.config.ent_coef * entropy.mean()
                    )
                    self.actor_optimizers[name].zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_gradient = nn.utils.clip_grad_norm_(
                        self.actors[name].parameters(), self.config.max_grad_norm
                    )
                    self.actor_optimizers[name].step()
                    with torch.no_grad():
                        approximate_kl = torch.mean(
                            torch.exp(log_ratio) - 1.0 - log_ratio
                        )
                        clip_fraction = torch.mean(
                            (torch.abs(ratio - 1.0) > self.config.clip_range).float()
                        )
                    actor_losses[name].append(float(actor_loss.detach().cpu()))
                    actor_gradients[name].append(float(actor_gradient.detach().cpu()))
                    approximate_kls[name].append(float(approximate_kl.cpu()))
                    clip_fractions[name].append(float(clip_fraction.cpu()))
                    entropies[name].append(float(entropy.mean().detach().cpu()))
        flat_global = torch.as_tensor(
            self.buffer.global_obs[: self.buffer.ptr].reshape(
                -1, self.buffer.global_obs.shape[-1]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            post_values = self.critic(flat_global).cpu().numpy()
        flat_returns = self.buffer.returns[: self.buffer.ptr].reshape(-1)
        actor_lr = self.actor_optimizers[AGENT_NAMES[0]].param_groups[0]["lr"]
        critic_lr = self.critic_optimizer.param_groups[0]["lr"]
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "explained_variance": _explained_variance(post_values, flat_returns),
            "clip_fraction": float(
                np.mean([value for name in AGENT_NAMES for value in clip_fractions[name]])
            ),
            "approx_kl": float(
                np.mean([value for name in AGENT_NAMES for value in approximate_kls[name]])
            ),
            "entropy": float(
                np.mean([value for name in AGENT_NAMES for value in entropies[name]])
            ),
            "value_loss": float(np.mean(critic_losses)),
            "critic_gradient": float(np.mean(critic_gradients)),
            "learning_rate": float(actor_lr),
            "critic_learning_rate": float(critic_lr),
            "timestamp": core.utc_now(),
        }
        for name in AGENT_NAMES:
            row.update(
                {
                    f"actor_loss_{name}": float(np.mean(actor_losses[name])),
                    f"actor_gradient_{name}": float(np.mean(actor_gradients[name])),
                    f"approx_kl_{name}": float(np.mean(approximate_kls[name])),
                    f"clip_fraction_{name}": float(np.mean(clip_fractions[name])),
                    f"policy_std_{name}": float(
                        torch.clamp(
                            self.actors[name].log_std.exp(), min=0.01, max=1.0
                        ).mean().detach().cpu()
                    ),
                }
            )
        row["policy_std"] = float(
            np.mean([row[f"policy_std_{name}"] for name in AGENT_NAMES])
        )
        return row

    def _checkpoint_payload(
        self, epoch: int, convergence: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "schema": "phase1.5f-state-v3-mappo",
            "config_hash": self.config.config_hash,
            "environment_fingerprint": self.environment_fingerprint,
            "committed_epoch": int(epoch),
            "global_step": int(self.global_step),
            "actors": {name: actor.state_dict() for name, actor in self.actors.items()},
            "critic": self.critic.state_dict(),
            "actor_optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.actor_optimizers.items()
            },
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_schedulers": {
                name: scheduler.state_dict()
                for name, scheduler in self.actor_schedulers.items()
            },
            "critic_scheduler": self.critic_scheduler.state_dict(),
            "training_rows": self.training_rows,
            "update_rows": self.update_rows,
            "sb3_rows": self.update_rows,
            "train_eval_rows": self.train_eval_rows,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "convergence": dict(convergence),
            "resume_count": self.resume_count,
            "wandb_run_id": self.wandb.run_id,
            "wandb_url": self.wandb.url,
            "rng": core.capture_rng_state(),
            "generator_state": self.generator.bit_generator.state,
            "updated_at": core.utc_now(),
        }

    def _log_epoch(
        self,
        training_row: Mapping[str, Any],
        update_row: Mapping[str, Any],
        convergence: Mapping[str, Any],
    ) -> None:
        epoch = int(training_row["epoch"])
        metrics: dict[str, Any] = {
            "epoch": epoch,
            "train/reward_mean": training_row["reward_mean"],
            "train/rolling_30": training_row["rolling_30"],
            "reward/energy": training_row["reward_energy_mean"],
            "reward/comfort": training_row["reward_comfort_mean"],
            "reward/smooth": training_row["reward_smooth_mean"],
            "mappo/explained_variance": update_row["explained_variance"],
            "mappo/clip_fraction": update_row["clip_fraction"],
            "mappo/approx_kl": update_row["approx_kl"],
            "mappo/policy_std": update_row["policy_std"],
            "mappo/actor_learning_rate": update_row["learning_rate"],
            "mappo/critic_learning_rate": update_row["critic_learning_rate"],
            **{
                f"convergence/{name}": convergence.get(name, False)
                for name in ("C1", "C2", "C3", "C4", "converged")
            },
        }
        for name in AGENT_NAMES:
            metrics.update(
                {
                    f"mappo/policy_std_{name}": update_row[f"policy_std_{name}"],
                    f"mappo/approx_kl_{name}": update_row[f"approx_kl_{name}"],
                    f"mappo/clip_fraction_{name}": update_row[f"clip_fraction_{name}"],
                    f"mappo/actor_gradient_{name}": update_row[f"actor_gradient_{name}"],
                }
            )
        if self.train_eval_rows and int(self.train_eval_rows[-1]["epoch"]) == epoch:
            latest = self.train_eval_rows[-1]
            metrics.update(
                {
                    "train_week/return": latest["return"],
                    "train_week/cost": latest["cost"],
                    "train_week/pmv_hours": latest["pmv_hours"],
                    "train_week/occupied_saturation": latest["occupied_saturation"],
                    "train_week/occupancy_action_gap": latest["occupancy_action_gap"],
                }
            )
        self.wandb.log(metrics, self.global_step)
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                    self.writer.add_scalar(key, value, self.global_step)

    def train(self) -> dict[str, Any]:
        evaluator = V3TrainWeekEvaluator(self.config, self.forecast_data)
        if self.observations is None:
            self.observations = self._reset_all()
        for epoch in range(self.committed_epoch + 1, self.config.max_epochs + 1):
            started = time.perf_counter()
            rollout = self._collect_rollout()
            update_row = self._update(epoch)
            previous_rewards = [
                core.safe_float(row["reward_mean"]) for row in self.training_rows
            ]
            training_row = {
                "epoch": int(epoch),
                "global_step": int(self.global_step),
                **rollout,
                "rolling_30": float(
                    np.mean([*previous_rewards, rollout["reward_mean"]][-30:])
                ),
                "epoch_wall_seconds": time.perf_counter() - started,
                "timestamp": core.utc_now(),
            }
            self.training_rows.append(training_row)
            self.update_rows.append(update_row)
            new_best = False
            if epoch % self.config.eval_interval == 0:
                evaluation = evaluator(
                    MAPPOPolicyAdapter(self.actors, self.config, self.device),
                    epoch,
                    self.global_step,
                )
                self.train_eval_rows.append(evaluation)
                if core.safe_float(evaluation["return"]) > self.best_score:
                    self.best_score = core.safe_float(evaluation["return"])
                    self.best_epoch = epoch
                    new_best = True
            convergence = evaluate_convergence(
                self.training_rows,
                self.update_rows,
                self.train_eval_rows,
                best_epoch=self.best_epoch,
                min_epoch=self.config.min_early_stop_epoch,
                algorithm="mappo",
                eval_interval=self.config.eval_interval,
            )
            for scheduler in self.actor_schedulers.values():
                scheduler.step()
            self.critic_scheduler.step()
            payload = self._checkpoint_payload(epoch, convergence)
            pointer = self.checkpoint.save(
                payload, epoch=epoch, global_step=self.global_step
            )
            if new_best:
                self.checkpoint.mark_best(pointer, self.best_score)
            self.committed_epoch = epoch
            self.convergence = dict(convergence)
            core.write_csv(self.config.run_dir / "training_metrics.csv", self.training_rows)
            core.write_csv(self.config.run_dir / "sb3_updates.csv", self.update_rows)
            core.write_csv(self.config.run_dir / "train_week_eval.csv", self.train_eval_rows)
            self._log_epoch(training_row, update_row, convergence)
            print(
                f"[HYDRO MAPPO v3 Epoch {epoch:03d}] "
                f"reward={training_row['reward_mean']:.3f} "
                f"rolling30={training_row['rolling_30']:.3f} "
                f"actor_lr={update_row['learning_rate']:.3e} "
                f"critic_lr={update_row['critic_learning_rate']:.3e} "
                f"best_train_week={self.best_score:.3f}"
            )
            if convergence.get("converged"):
                break
        if self.writer is not None:
            self.writer.flush()
        latest = self.checkpoint.load_latest()
        if latest is None:
            raise RuntimeError("MAPPO stopped before the first checkpoint commit")
        return latest["payload"]

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass


def _ppo_checkpoint_state(config: HydronicV3Config) -> dict[str, Any] | None:
    metadata = core.AtomicCheckpointManager(
        config.run_dir, config.config_hash
    ).load_latest_metadata()
    return metadata["state"] if metadata else None


def _require_ppo_finished(suite: HydronicV3Suite) -> None:
    state = _ppo_checkpoint_state(suite.ppo)
    if state is None or not _checkpoint_complete_state(suite.ppo, state):
        raise RuntimeError(
            "PPO must reach its smoke/full cap or satisfy C1-C4 before MAPPO starts"
        )


class V3MAPPOExperiment:
    def __init__(self, suite: HydronicV3Suite):
        self.suite = suite
        self.config = suite.mappo

    def train(self) -> dict[str, Any]:
        run_preflight(self.suite, online=False)
        _require_ppo_finished(self.suite)
        config = self.config
        checkpoint = AtomicTorchCheckpointManager(config.run_dir, config.config_hash)
        latest = checkpoint.load_latest()
        if latest and not config.resume:
            raise RuntimeError("MAPPO checkpoint exists and RESUME=False; use a new directory")
        if latest and _checkpoint_complete_state(config, latest["payload"]):
            core.generate_training_diagnostics(config, latest["payload"])
            return latest["payload"]
        preferred = latest["payload"].get("wandb_run_id") if latest else None
        identity = core.get_or_create_run_identity(config, preferred)
        _write_run_manifest(config, identity, status="starting")
        wandb_logger = V3SafeWandb(config, identity["wandb_run_id"])
        wandb_logger.start()
        environments: list[gym.Env] = []
        trainer: MAPPOTrainer | None = None
        old_sigterm: Any = None
        start_wall = time.perf_counter()
        status = "complete"
        error: str | None = None
        online_started = False
        lock_path = config.run_dir / "runtime" / "training.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with core.RunLock(lock_path):
                online_started = True
                old_sigterm, _ = core._install_termination_handler()
                forecast = V3ForecastProvider(config, include_validation=False)
                forecast.prefetch_all()
                core.atomic_write_json(
                    config.run_dir / "forecast_composition.json",
                    audit_calendar_and_forecast(config, forecast.data),
                )
                boptest = core.query_boptest_version(config)
                fingerprint = core.environment_fingerprint(config, boptest, forecast.data)
                if latest and latest["payload"].get("environment_fingerprint") != fingerprint:
                    raise RuntimeError(
                        "MAPPO environment fingerprint changed; refusing silent resume"
                    )
                _write_run_manifest(
                    config,
                    identity,
                    status="training",
                    boptest=boptest,
                    environment_fingerprint=fingerprint,
                )
                environments = [
                    build_environment_v3(
                        config,
                        forecast.data,
                        rank=rank,
                        phase="training",
                    )
                    for rank in range(config.num_envs)
                ]
                trainer = MAPPOTrainer(
                    config,
                    environments,
                    forecast.data,
                    checkpoint,
                    wandb_logger,
                    environment_fingerprint=fingerprint,
                    restored=latest["payload"] if latest else None,
                )
                trainer.train()
        except KeyboardInterrupt:
            status = "interrupted"
            error = "KeyboardInterrupt"
        except Exception:
            status = "failed"
            error = traceback.format_exc()
            raise
        finally:
            if old_sigterm is not None:
                core._restore_termination_handler(old_sigterm)
            if trainer is not None:
                trainer.close()
            for environment in environments:
                try:
                    environment.close()
                except Exception:
                    pass
            core.cleanup_stale_testids(config, force=online_started)
            lifecycle_path = core.combine_lifecycle_logs(config.run_dir)
            committed = checkpoint.load_latest()
            state = committed["payload"] if committed else {}
            if state:
                core.generate_training_diagnostics(config, state)
            _finish_manifest(
                config,
                identity,
                state,
                status=status,
                wall_seconds=time.perf_counter() - start_wall,
                error=error,
            )
            wandb_logger.finish(
                lifecycle_path,
                {
                    "status": status,
                    "actual_epochs": int(state.get("committed_epoch", 0)),
                    "global_step": int(state.get("global_step", 0)),
                    "converged": bool(
                        state.get("convergence", {}).get("converged", False)
                    ),
                },
            )
        final = checkpoint.load_latest()
        if final is None:
            raise RuntimeError("MAPPO ended before the first checkpoint was committed")
        return final["payload"]


def train_mappo(suite: HydronicV3Suite) -> dict[str, Any]:
    return V3MAPPOExperiment(suite).train()


def _load_mappo_policy(
    config: HydronicV3Config,
    payload: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> MAPPOPolicyAdapter:
    target = torch.device(device)
    actors = {
        name: ActorNetwork(
            len(LOCAL_OBSERVATION_COLUMNS[name]), hidden_sizes=config.policy_net
        ).to(target)
        for name in AGENT_NAMES
    }
    states = payload.get("actors")
    if not isinstance(states, Mapping):
        states = {
            "nz": payload.get("actor_nz_state_dict"),
            "sz": payload.get("actor_sz_state_dict"),
        }
    for name in AGENT_NAMES:
        if not isinstance(states.get(name), Mapping):
            raise RuntimeError(f"MAPPO checkpoint is missing actor state: {name}")
        actors[name].load_state_dict(states[name])
        actors[name].eval()
    return MAPPOPolicyAdapter(actors, config, target)


def load_legacy_mappo_policy(config: HydronicV3Config) -> MAPPOPolicyAdapter:
    payload = torch.load(
        config.legacy_mappo_path, map_location="cpu", weights_only=False
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError("Archived v1 MAPPO checkpoint has an invalid payload")
    return _load_mappo_policy(config, payload, device="cpu")


def _final_policy_filename(label: str) -> str:
    names = {
        "rbc": "rbc_validation_hydronic_2zone.csv",
        "v1_ppo": "ppo_v1_validation_hydronic_2zone.csv",
        "v1_mappo": "mappo_v1_validation_hydronic_2zone.csv",
        "v3_ppo": "ppo_validation_hydronic_2zone.csv",
        "v3_mappo": "mappo_validation_hydronic_2zone.csv",
    }
    return names[label]


def _save_final_trajectory(
    config: HydronicV3Config,
    results_dir: Path,
    label: str,
    frame: pd.DataFrame,
) -> tuple[Path, Path]:
    trajectory_path = results_dir / _final_policy_filename(label)
    export = frame[validation_columns(config)]
    export.to_csv(trajectory_path, index=False)
    if len(export) != config.episode_steps:
        raise RuntimeError(
            f"{label} held-out row count {len(export)} != {config.episode_steps}"
        )
    if list(pd.read_csv(trajectory_path, nrows=1).columns) != validation_columns(config):
        raise RuntimeError(f"{label} validation schema is not the v3 schema")
    actions = np.asarray([json.loads(value) for value in frame["_actions"]], dtype=float)
    action_frame = pd.DataFrame(
        {
            "step": np.arange(len(actions), dtype=int),
            "time": frame["time"].to_numpy(),
            **{
                f"action_{zone}": actions[:, index]
                for index, zone in enumerate(config.zones)
            },
        }
    )
    actions_path = results_dir / f"{label}_actions.csv"
    action_frame.to_csv(actions_path, index=False)
    return trajectory_path, actions_path


def _augment_final_metrics(
    frame: pd.DataFrame, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **dict(metrics),
        "energy_kwh": float(frame["energy_step_kwh"].sum()),
    }


def _plot_v3_final_validation(
    config: HydronicV3Config,
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
        axes[1].plot(hours, frame["energy_step_kwh"].cumsum(), label=label)
    axes[0].set_ylabel("Cumulative cost")
    axes[1].set_ylabel("Cumulative energy [kWh]")
    reference = trajectories["v3_mappo"]
    hours = np.arange(len(reference)) * config.control_period / 3600.0
    for zone in config.zones:
        axes[2].plot(hours, reference[f"pmv_{zone}"], label=f"v3 MAPPO PMV {zone}")
    axes[2].axhspan(-0.5, 0.5, color="green", alpha=0.12)
    axes[2].set_ylabel("PMV")
    axes[2].set_xlabel("Held-out time [h]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _require_both_finished(
    suite: HydronicV3Suite,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ppo_state = _ppo_checkpoint_state(suite.ppo)
    mappo_latest = AtomicTorchCheckpointManager(
        suite.mappo.run_dir, suite.mappo.config_hash
    ).load_latest()
    mappo_state = mappo_latest["payload"] if mappo_latest else None
    if ppo_state is None or not _checkpoint_complete_state(suite.ppo, ppo_state):
        raise RuntimeError("PPO training has not completed; held-out validation is blocked")
    if mappo_state is None or not _checkpoint_complete_state(suite.mappo, mappo_state):
        raise RuntimeError("MAPPO training has not completed; held-out validation is blocked")
    return ppo_state, mappo_state


def evaluate_final(
    suite: HydronicV3Suite,
    *,
    force: bool | None = None,
) -> dict[str, Any]:
    """Run each held-out policy once after both training jobs are complete."""
    run_preflight(suite, online=False)
    ppo_state, mappo_state = _require_both_finished(suite)
    force = suite.force_final_eval if force is None else bool(force)
    config = suite.final
    results_dir = suite.run_dir / "final_validation"
    results_dir.mkdir(parents=True, exist_ok=True)
    ppo_checkpoint = core.AtomicCheckpointManager(
        suite.ppo.run_dir, suite.ppo.config_hash
    )
    mappo_checkpoint = AtomicTorchCheckpointManager(
        suite.mappo.run_dir, suite.mappo.config_hash
    )
    ppo_best = ppo_checkpoint.load_best_metadata()
    mappo_best = mappo_checkpoint.load_best()
    signature = {
        "schema": "phase1.5f-final-signature-v3",
        "ppo_config_hash": suite.ppo.config_hash,
        "mappo_config_hash": suite.mappo.config_hash,
        "legacy_v1_ppo_sha256": core.sha256_file(suite.ppo.legacy_model_path),
        "legacy_v1_mappo_sha256": core.sha256_file(suite.mappo.legacy_mappo_path),
        "ppo_best_manifest_sha256": core.sha256_file(
            ppo_checkpoint.directory / ppo_best["pointer"]["manifest"]
        ),
        "mappo_best_manifest_sha256": core.sha256_file(
            mappo_checkpoint.directory / mappo_best["pointer"]["manifest"]
        ),
    }
    signature_hash = core.sha256_payload(signature)
    report_path = results_dir / "final_validation_report.json"
    progress_path = results_dir / "evaluation_manifest.json"
    if report_path.exists() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("signature_hash") == signature_hash:
            return report
        raise RuntimeError(
            "A completed final validation references different best checkpoints; "
            "set FORCE_FINAL_EVAL=True only for an intentional rerun"
        )
    progress: dict[str, Any]
    if progress_path.exists() and not force:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("signature_hash") != signature_hash:
            raise RuntimeError(
                "Incomplete final evaluation has a different signature; explicit force is required"
            )
    else:
        progress = {
            "schema": "phase1.5f-final-progress-v3",
            "signature": signature,
            "signature_hash": signature_hash,
            "started_at": core.utc_now(),
            "methods": {},
            "status": "running",
        }
        core.atomic_write_json(progress_path, progress)

    cleanup: dict[str, Any] = {}
    for algorithm_config in (suite.ppo, suite.mappo, config):
        rows = core.cleanup_stale_testids(algorithm_config)
        cleanup[algorithm_config.algorithm] = rows
        if any(row.get("status") == "live_owner_skipped" for row in rows):
            raise RuntimeError("A training TestID still has a live owner; final evaluation is blocked")

    forecast = V3ForecastProvider(config, include_validation=True)
    forecast.prefetch_all()
    policy_factories: dict[str, Callable[[], tuple[Any | None, bool, bool]]] = {
        "rbc": lambda: (None, False, True),
        "v1_ppo": lambda: (
            PPO.load(suite.ppo.legacy_model_path, device="auto"),
            True,
            False,
        ),
        "v1_mappo": lambda: (
            load_legacy_mappo_policy(suite.mappo),
            True,
            False,
        ),
        "v3_ppo": lambda: (
            core.Phase15fPPO.load(ppo_best["model_path"], device="auto"),
            False,
            False,
        ),
        "v3_mappo": lambda: (
            _load_mappo_policy(suite.mappo, mappo_best["payload"]),
            False,
            False,
        ),
    }
    trajectories: dict[str, pd.DataFrame] = {}
    metrics: dict[str, Any] = {}
    for label in ("rbc", "v1_ppo", "v1_mappo", "v3_ppo", "v3_mappo"):
        existing = progress["methods"].get(label, {})
        if existing.get("status") == "complete" and not force:
            frame = pd.read_csv(existing["trajectory"])
            action_frame = pd.read_csv(existing["actions"])
            frame["_actions"] = [
                json.dumps(row)
                for row in action_frame[
                    [f"action_{zone}" for zone in config.zones]
                ].to_numpy(dtype=float).tolist()
            ]
            trajectories[label] = frame
            metrics[label] = existing["metrics"]
            continue
        progress["methods"][label] = {
            "status": "running",
            "started_at": core.utc_now(),
        }
        core.atomic_write_json(progress_path, progress)
        try:
            model, legacy_fixed_baseline, zero_policy = policy_factories[label]()
            frame, policy_metrics = rollout_policy(
                config,
                forecast.data,
                model=model,
                is_validation=True,
                phase=f"held_out_{label}",
                legacy_fixed_baseline=legacy_fixed_baseline,
                zero_policy=zero_policy,
            )
            policy_metrics = _augment_final_metrics(frame, policy_metrics)
            trajectory_path, actions_path = _save_final_trajectory(
                config, results_dir, label, frame
            )
            trajectories[label] = frame
            metrics[label] = policy_metrics
            progress["methods"][label] = {
                "status": "complete",
                "completed_at": core.utc_now(),
                "trajectory": str(trajectory_path),
                "actions": str(actions_path),
                "trajectory_sha256": core.sha256_file(trajectory_path),
                "actions_sha256": core.sha256_file(actions_path),
                "rows": len(frame),
                "metrics": policy_metrics,
            }
            core.atomic_write_json(progress_path, progress)
        except Exception:
            progress["methods"][label] = {
                **progress["methods"][label],
                "status": "failed",
                "failed_at": core.utc_now(),
                "error": traceback.format_exc(),
            }
            core.atomic_write_json(progress_path, progress)
            raise

    _plot_v3_final_validation(
        config, trajectories, results_dir / "final_validation.png"
    )
    comparison = [{"policy": label, **values} for label, values in metrics.items()]
    core.write_csv(results_dir / "kpi_comparison.csv", comparison)
    report = {
        "schema": "phase1.5f-final-validation-v3",
        "generated_at": core.utc_now(),
        "signature": signature,
        "signature_hash": signature_hash,
        "validation_start_day": config.start_day + 7,
        "validation_steps": config.episode_steps,
        "deterministic": True,
        "checkpoint_selection": "deterministic training-week return only",
        "held_out_used_for_selection": False,
        "legacy_ppo_provenance": audit_legacy_ppo_provenance(suite.ppo),
        "metrics": metrics,
        "convergence": {
            "ppo": ppo_state.get("convergence", {}),
            "mappo": mappo_state.get("convergence", {}),
        },
        "best": {
            "ppo": {
                "epoch": int(ppo_best["pointer"]["epoch"]),
                "global_step": int(ppo_best["pointer"]["num_timesteps"]),
            },
            "mappo": {
                "epoch": int(mappo_best["pointer"]["epoch"]),
                "global_step": int(mappo_best["pointer"]["global_step"]),
            },
        },
        "cost_ranking_is_diagnostic_only": True,
        "reward_comparison_warning": (
            "v1 and v3 rewards are not directly comparable because the PMV threshold, "
            "comfort scaler and action mapping changed; compare physical KPIs."
        ),
        "boptest": core.query_boptest_version(config),
        "stale_cleanup": cleanup,
        "wandb": {
            "ppo_run_id": ppo_state.get("wandb_run_id"),
            "mappo_run_id": mappo_state.get("wandb_run_id"),
            "ppo_url": ppo_state.get("wandb_url"),
            "mappo_url": mappo_state.get("wandb_url"),
        },
    }
    core.atomic_write_json(report_path, report)
    progress["status"] = "complete"
    progress["completed_at"] = core.utc_now()
    progress["report"] = str(report_path)
    core.atomic_write_json(progress_path, progress)
    lines = [
        "# Phase 1.5f v3 Hydronic unified final validation",
        "",
        "The held-out next week was never used for checkpoint selection.",
        "Cost ranking is recorded but is not a convergence criterion.",
        "",
        "| Policy | Cost | Energy (kWh) | Zone-h | PMV·h | Occ. saturation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, values in metrics.items():
        lines.append(
            f"| {label} | {values['cost']:.6f} | {values['energy_kwh']:.3f} | "
            f"{values['zone_hours']:.3f} | {values['pmv_hours']:.3f} | "
            f"{100*values['occupied_saturation']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"- PPO C1–C4 converged: {bool(ppo_state.get('convergence', {}).get('converged'))}",
            f"- MAPPO C1–C4 converged: {bool(mappo_state.get('convergence', {}).get('converged'))}",
            f"- PPO W&B run ID: {ppo_state.get('wandb_run_id')}",
            f"- MAPPO W&B run ID: {mappo_state.get('wandb_run_id')}",
            f"- PPO W&B URL: {ppo_state.get('wandb_url')}",
            f"- MAPPO W&B URL: {mappo_state.get('wandb_url')}",
        ]
    )
    (results_dir / "final_validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    core.combine_lifecycle_logs(config.run_dir)
    return report
