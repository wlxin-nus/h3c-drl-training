from __future__ import annotations

import json
import math
import os
import random
import time
import uuid
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from . import checkpointing as checkpoint_core
from .config import OBSERVATION_CONTRACT_ID, WANDB_PROJECT, TaskSpec, validate_task
from .continuation import (
    completion_status,
    linear_lr_multiplier,
    next_validation_block_epochs,
    post_cap_learning_rate,
    training_should_start,
    validate_continuation_mode,
)
from .early_stop import PlateauState
from .environment import cleanup_run_testids, make_env, prefetch_forecast
from .io import atomic_json, utc_now
from .tracking import SafeWandb


class FrozenLinearSchedule:
    """Absolute experiment schedule represented in SB3 progress coordinates."""

    def __init__(self, spec: TaskSpec):
        self.initial = float(spec.learning_rate)
        self.max_epochs = int(spec.max_epochs)
        self.decay_epochs = int(spec.lr_decay_epochs)

    def __call__(self, progress_remaining: float) -> float:
        elapsed = (1.0 - float(progress_remaining)) * self.max_epochs
        return self.initial * linear_lr_multiplier(elapsed, self.decay_epochs)


class ConstantFloorSchedule:
    """Keep post-cap updates at the preregistered 10% learning-rate floor."""

    def __init__(self, spec: TaskSpec):
        self.value = post_cap_learning_rate(spec.learning_rate)

    def __call__(self, _progress_remaining: float) -> float:
        return self.value


def _factory(
    spec: TaskSpec,
    seed: int,
    forecast: Mapping[str, Sequence[float]],
    endpoint: str,
    run_dir: Path,
    rank: int,
) -> gym.Env:
    return Monitor(make_env(spec, seed, forecast, endpoint, run_dir, rank, "training"))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    pd.DataFrame(list(rows)).to_csv(temporary, index=False)
    os.replace(temporary, path)


def evaluate_train_window(
    model: Any,
    spec: TaskSpec,
    seed: int,
    forecast: Mapping[str, Sequence[float]],
    endpoint: str,
    run_dir: Path,
    epoch: int,
) -> dict[str, Any]:
    env = make_env(
        spec,
        seed,
        forecast,
        endpoint,
        run_dir,
        "eval",
        f"train_week_eval_{epoch:04d}",
        evaluation=False,
    )
    rewards: list[float] = []
    action_rows: list[list[float]] = []
    costs: list[float] = []
    try:
        observation, _ = env.reset(seed=seed)
        for _step in range(spec.episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            action_rows.append(np.asarray(action, dtype=float).reshape(-1).tolist())
            costs.append(float(info["phys_cost"]))
            if terminated or truncated:
                break
    finally:
        env.close()
    if len(rewards) != spec.episode_steps:
        raise RuntimeError(f"Incomplete deterministic train-window rollout: {len(rewards)}")
    directory = run_dir / "diagnostics" / "train_window"
    directory.mkdir(parents=True, exist_ok=True)
    _write_rows(
        directory / f"epoch_{epoch:04d}.csv",
        [
            {
                "step": index,
                "reward": reward,
                "cost": costs[index],
                "actions": json.dumps(action_rows[index]),
            }
            for index, reward in enumerate(rewards)
        ],
    )
    actions = np.asarray(action_rows, dtype=float)
    return {
        "epoch": int(epoch),
        "return": float(np.sum(rewards)),
        "cost": float(np.sum(costs)),
        "action_mean": float(actions.mean()),
        "action_std": float(actions.std()),
        "action_saturation": float(np.mean(np.abs(actions) > 0.95)),
        "timestamp": utc_now(),
    }


class TransactionalPPOCallback(BaseCallback):
    def __init__(
        self,
        *,
        spec: TaskSpec,
        seed: int,
        run_dir: Path,
        mode: str,
        endpoint: str,
        forecast: Mapping[str, Sequence[float]],
        checkpoint: checkpoint_core.AtomicCheckpointManager,
        tracker: SafeWandb,
        restored: Mapping[str, Any] | None,
        environment_fingerprint: str,
        continue_until_converged: bool,
    ):
        super().__init__(verbose=0)
        self.spec = spec
        self.seed = seed
        self.run_dir = Path(run_dir)
        self.mode = mode
        self.endpoint = endpoint
        self.forecast = forecast
        self.checkpoint = checkpoint
        self.tracker = tracker
        state = dict(restored or {})
        self.committed_epoch = int(state.get("committed_epoch", 0))
        self.training_rows = list(state.get("training_rows", []))
        self.update_rows = list(state.get("update_rows", []))
        self.evaluation_rows = list(state.get("evaluation_rows", []))
        self.plateau = PlateauState.from_dict(state.get("early_stop"))
        self.plateau.min_epoch = spec.early_stop_min_epoch
        self.plateau.patience = spec.early_stop_patience
        self.plateau.min_delta_fraction = spec.early_stop_min_delta_fraction
        self.environment_fingerprint = environment_fingerprint
        self.continue_until_converged = bool(continue_until_converged)
        self.resume_count = int(state.get("resume_count", 0)) + int(restored is not None)
        self._started = time.perf_counter()
        self._clear()

    def _clear(self) -> None:
        self.running = np.zeros(self.spec.num_envs, dtype=float)
        self.episodes: list[float] = []
        self.energy: list[float] = []
        self.comfort: list[float] = []
        self.smooth: list[float] = []
        self.actions: list[float] = []
        self.occupied_actions: list[float] = []
        self.unoccupied_actions: list[float] = []
        self.occupied_pmv: list[float] = []

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        infos = self.locals.get("infos", [])
        if rewards.size == self.running.size:
            self.running += rewards
        for index, info in enumerate(infos):
            self.energy.append(_finite(info.get("rew_energy")))
            self.comfort.append(_finite(info.get("rew_comfort")))
            self.smooth.append(_finite(info.get("rew_smooth")))
            actions = info.get("actions", [])
            occupancies = info.get("occupancies", [])
            pmvs = info.get("pmvs", [])
            for action, occupancy, pmv in zip(actions, occupancies, pmvs, strict=False):
                action_value = _finite(action)
                self.actions.append(action_value)
                if _finite(occupancy) > 0:
                    self.occupied_actions.append(action_value)
                    self.occupied_pmv.append(_finite(pmv))
                else:
                    self.unoccupied_actions.append(action_value)
            if index < dones.size and dones[index]:
                self.episodes.append(_finite(info.get("episode", {}).get("r"), self.running[index]))
                self.running[index] = 0.0
        return True

    def _state(self, epoch: int) -> dict[str, Any]:
        return {
            "schema": "h3c-drl-ppo-state-v1",
            "config_hash": self.spec.scientific_hash(self.seed),
            "protocol_hash": self.spec.protocol_hash(),
            "observation_contract": OBSERVATION_CONTRACT_ID,
            "environment_fingerprint": self.environment_fingerprint,
            "committed_epoch": int(epoch),
            "global_step": int(self.model.num_timesteps),
            "training_rows": self.training_rows,
            "update_rows": self.update_rows,
            "evaluation_rows": self.evaluation_rows,
            "early_stop": self.plateau.to_dict(),
            "best_score": None
            if not math.isfinite(self.plateau.best_score)
            else self.plateau.best_score,
            "best_epoch": self.plateau.best_epoch,
            "resume_count": self.resume_count,
            "early_stop_protocol": self.spec.early_stop_protocol,
            "registered_max_epochs": self.spec.max_epochs,
            "continue_until_converged": self.continue_until_converged,
            "wandb_run_id": self.tracker.run_id,
            "updated_at": utc_now(),
            "code_commit": _run_identity(self.run_dir, self.spec, self.seed).get("code_commit"),
            "code_fingerprint": _run_identity(self.run_dir, self.spec, self.seed).get(
                "code_fingerprint"
            ),
        }

    def on_post_update(self, model: Any) -> bool:
        epoch = int(model.num_timesteps // self.spec.steps_per_epoch)
        if epoch <= self.committed_epoch:
            self._clear()
            return True
        if model.num_timesteps % self.spec.steps_per_epoch:
            raise RuntimeError("Checkpoint attempted away from an epoch boundary")
        episode_returns = self.episodes or self.running.tolist()
        reward_mean = float(np.mean(episode_returns))
        prior = [float(row["reward_mean"]) for row in self.training_rows]
        train_row = {
            "run_uuid": self.tracker.run_id,
            "epoch": epoch,
            "global_step": int(model.num_timesteps),
            "reward_mean": reward_mean,
            "reward_std": float(np.std(episode_returns)),
            "rolling_30": float(np.mean([*prior, reward_mean][-30:])),
            "reward_energy_mean": float(np.mean(self.energy)),
            "reward_comfort_mean": float(np.mean(self.comfort)),
            "reward_smooth_mean": float(np.mean(self.smooth)),
            "action_mean": float(np.mean(self.actions)) if self.actions else float("nan"),
            "action_std": float(np.std(self.actions)) if self.actions else float("nan"),
            "occupied_action_saturation": float(np.mean(np.abs(self.occupied_actions) > 0.95))
            if self.occupied_actions
            else 0.0,
            "action_occ_unocc_gap": float(
                abs(np.mean(self.occupied_actions) - np.mean(self.unoccupied_actions))
            )
            if self.occupied_actions and self.unoccupied_actions
            else 0.0,
            "occupied_pmv_violation_rate": float(
                np.mean(np.abs(self.occupied_pmv) > self.spec.comfort_threshold)
            )
            if self.occupied_pmv
            else 0.0,
            "epoch_wall_seconds": time.perf_counter() - self._started,
            "timestamp": utc_now(),
        }
        logger = getattr(model.logger, "name_to_value", {})
        gradient_norm = (
            float(
                torch.sqrt(
                    sum(
                        torch.sum(parameter.grad.detach() ** 2)
                        for parameter in model.policy.parameters()
                        if parameter.grad is not None
                    )
                ).cpu()
            )
            if any(parameter.grad is not None for parameter in model.policy.parameters())
            else float("nan")
        )
        update_row = {
            "run_uuid": self.tracker.run_id,
            "epoch": epoch,
            "global_step": int(model.num_timesteps),
            "learning_rate": float(model.policy.optimizer.param_groups[0]["lr"]),
            "policy_std": float(torch.exp(model.policy.log_std.detach()).mean().cpu()),
            "explained_variance": _finite(logger.get("train/explained_variance"), float("nan")),
            "clip_fraction": _finite(logger.get("train/clip_fraction"), float("nan")),
            "approx_kl": _finite(logger.get("train/approx_kl"), float("nan")),
            "value_loss": _finite(logger.get("train/value_loss"), float("nan")),
            "entropy_loss": _finite(logger.get("train/entropy_loss"), float("nan")),
            "policy_gradient_loss": _finite(logger.get("train/policy_gradient_loss"), float("nan")),
            "gradient_norm": gradient_norm,
            "timestamp": utc_now(),
        }
        self.training_rows.append(train_row)
        self.update_rows.append(update_row)
        actual_best = False
        current_evaluation = None
        interval = 1 if self.mode == "smoke" else self.spec.eval_interval
        if epoch % interval == 0:
            current_evaluation = evaluate_train_window(
                model, self.spec, self.seed, self.forecast, self.endpoint, self.run_dir, epoch
            )
            stop_update = self.plateau.update(epoch, float(current_evaluation["return"]))
            current_evaluation.update(stop_update)
            self.evaluation_rows.append(current_evaluation)
            actual_best = bool(stop_update["actual_best"])
        # The checkpoint transaction owns progress. CSV/W&B are projections of this state.
        pointer = self.checkpoint.save(model, self._state(epoch), epoch)
        if actual_best:
            self.checkpoint.mark_best(pointer, self.plateau.best_score)
        self.committed_epoch = epoch
        _write_rows(self.run_dir / "training_metrics.csv", self.training_rows)
        _write_rows(self.run_dir / "updates.csv", self.update_rows)
        _write_rows(self.run_dir / "train_window_eval.csv", self.evaluation_rows)
        self.tracker.log(
            {
                "epoch": epoch,
                "train/reward_mean": reward_mean,
                "train/rolling_30": train_row["rolling_30"],
                "ppo/learning_rate": update_row["learning_rate"],
                "ppo/policy_std": update_row["policy_std"],
                "ppo/approx_kl": update_row["approx_kl"],
                "ppo/clip_fraction": update_row["clip_fraction"],
                "ppo/explained_variance": update_row["explained_variance"],
                "ppo/value_loss": update_row["value_loss"],
                "ppo/gradient_norm": update_row["gradient_norm"],
                "early_stop/misses": self.plateau.misses,
                "early_stop/stopped": self.plateau.stopped,
                "train_window/return": self.evaluation_rows[-1]["return"]
                if self.evaluation_rows and self.evaluation_rows[-1]["epoch"] == epoch
                else float("nan"),
                "train_window/cost": float("nan")
                if current_evaluation is None
                else current_evaluation["cost"],
                "train_window/action_saturation": float("nan")
                if current_evaluation is None
                else current_evaluation["action_saturation"],
            },
            step=int(model.num_timesteps),
        )
        print(
            f"[{self.spec.key} seed={self.seed} epoch={epoch:04d}] "
            f"reward={reward_mean:.3f} lr={update_row['learning_rate']:.3e} "
            f"best_eval={self.plateau.best_score:.3f} misses={self.plateau.misses}"
        )
        self._started = time.perf_counter()
        self._clear()
        smoke_complete = self.mode == "smoke" and epoch >= 2
        return not self.plateau.stopped and not smoke_complete


def _run_identity(run_dir: Path, spec: TaskSpec, seed: int) -> dict[str, Any]:
    path = run_dir / "run_identity.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    identity = {
        "run_uuid": f"{spec.key}-seed{seed}-{uuid.uuid4().hex[:12]}",
        "created_at": utc_now(),
    }
    atomic_json(path, identity)
    return identity


def train_ppo(
    spec: TaskSpec,
    *,
    seed: int,
    mode: str,
    endpoint: str,
    run_dir: Path,
    resume: bool,
    wandb_mode: str,
    device: str,
    continue_until_converged: bool = False,
) -> dict[str, Any]:
    validate_task(spec, seed)
    validate_continuation_mode(mode, continue_until_converged)
    if spec.algorithm != "ppo":
        raise ValueError("PPO trainer received a non-PPO task")
    run_dir.mkdir(parents=True, exist_ok=True)
    cleanup_run_testids(run_dir, endpoint)
    identity = _run_identity(run_dir, spec, seed)
    config_hash = spec.scientific_hash(seed)
    checkpoint = checkpoint_core.AtomicCheckpointManager(run_dir, config_hash)
    latest = checkpoint.load_latest_metadata()
    if latest and not resume:
        raise RuntimeError("Checkpoint exists; use -Resume or a new output directory")
    forecast = prefetch_forecast(spec, endpoint=endpoint, run_dir=run_dir, seed=seed)
    fingerprint = checkpoint_core.sha256_payload({"spec": spec.payload(seed), "forecast": forecast})
    if latest and latest["state"].get("environment_fingerprint") != fingerprint:
        raise RuntimeError("Environment fingerprint changed; refusing silent resume")
    tracker = SafeWandb(
        run_dir,
        project=WANDB_PROJECT,
        group=f"seed{seed}",
        name=f"{spec.key}-seed{seed}-{mode}",
        run_id=identity["run_uuid"],
        mode=wandb_mode,
        config=spec.payload(seed),
    )
    tracker.start()
    vector: SubprocVecEnv | None = None
    try:
        vector = SubprocVecEnv(
            [partial(_factory, spec, seed, forecast, endpoint, run_dir, rank) for rank in range(4)],
            start_method="spawn",
        )
        schedule = FrozenLinearSchedule(spec)
        floor_schedule = ConstantFloorSchedule(spec)
        if latest:
            restored_steps = int(latest["state"].get("global_step", 0))
            load_schedule = (
                floor_schedule
                if continue_until_converged and restored_steps >= spec.max_steps
                else schedule
            )
            model = checkpoint_core.TransactionalPPO.load(
                latest["model_path"],
                env=vector,
                device=device,
                custom_objects={"learning_rate": load_schedule, "lr_schedule": load_schedule},
            )
            checkpoint_core.restore_rng_state(latest["rng"])
            restored = latest["state"]
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            policy_kwargs: dict[str, Any] = {
                "net_arch": {"pi": list(spec.policy_net), "vf": list(spec.policy_net)}
            }
            policy_kwargs["log_std_init"] = spec.log_std_init
            model = checkpoint_core.TransactionalPPO(
                "MlpPolicy",
                vector,
                learning_rate=schedule,
                n_steps=spec.n_steps,
                batch_size=spec.batch_size,
                n_epochs=spec.n_epochs,
                gamma=spec.gamma,
                gae_lambda=spec.gae_lambda,
                clip_range=spec.clip_range,
                ent_coef=spec.ent_coef,
                vf_coef=spec.vf_coef,
                max_grad_norm=spec.max_grad_norm,
                policy_kwargs=policy_kwargs,
                seed=seed,
                device=device,
                verbose=1,
                tensorboard_log=str(run_dir / "tensorboard"),
            )
            restored = None
        callback = TransactionalPPOCallback(
            spec=spec,
            seed=seed,
            run_dir=run_dir,
            mode=mode,
            endpoint=endpoint,
            forecast=forecast,
            checkpoint=checkpoint,
            tracker=tracker,
            restored=restored,
            environment_fingerprint=fingerprint,
            continue_until_converged=continue_until_converged,
        )
        model._transaction_hook = callback.on_post_update
        # Always expose the registered absolute experiment cap to SB3 so its
        # progress-dependent schedule is identical in smoke, full and resume.
        # The callback, not a shortened learn horizon, stops smoke at epoch 2.
        cap = spec.max_steps
        remaining = cap - int(model.num_timesteps)
        if remaining > 0 and training_should_start(
            mode=mode,
            committed_epoch=callback.committed_epoch,
            stopped=callback.plateau.stopped,
        ):
            model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=callback)
        if continue_until_converged and not callback.plateau.stopped:
            # Each extension call ends exactly at the next 25-epoch validation
            # boundary. The callback remains the sole convergence authority.
            model.learning_rate = floor_schedule
            model.lr_schedule = floor_schedule
            while not callback.plateau.stopped:
                current_epoch = int(model.num_timesteps // spec.steps_per_epoch)
                block_epochs = next_validation_block_epochs(current_epoch, spec.eval_interval)
                model.learn(
                    total_timesteps=block_epochs * spec.steps_per_epoch,
                    reset_num_timesteps=False,
                    callback=callback,
                )
        model._transaction_hook = None
    finally:
        if vector is not None:
            vector.close()
        cleanup_run_testids(run_dir, endpoint)
        final = checkpoint.load_latest_metadata()
        tracker.finish(
            {
                "completed_epoch": 0 if final is None else final["state"]["committed_epoch"],
                "early_stopped": False
                if final is None
                else final["state"]["early_stop"]["stopped"],
            }
        )
    final = checkpoint.load_latest_metadata()
    if final is None:
        raise RuntimeError("Training ended before the first committed epoch")
    completed_epoch = int(final["state"]["committed_epoch"])
    stopped = bool(final["state"]["early_stop"]["stopped"])
    atomic_json(
        run_dir / "run_manifest.json",
        {
            "schema": "h3c-drl-run-v1",
            "task": spec.key,
            "seed": seed,
            "mode": mode,
            "protocol_version": identity.get("protocol_version"),
            "protocol_hash": spec.protocol_hash(),
            "config_hash": config_hash,
            "boptest_version": identity.get("boptest_version"),
            "run_uuid": identity["run_uuid"],
            "status": completion_status(
                stopped=stopped, reached_cap=completed_epoch >= spec.max_epochs
            ),
            "completed_epoch": completed_epoch,
            "global_step": final["state"]["global_step"],
            "best_epoch": final["state"]["best_epoch"],
            "best_score": final["state"]["best_score"],
            "updated_at": utc_now(),
            "selection_basis": "highest deterministic training-window return at registered validation points",
            "held_out_used_for_selection": False,
            "training_environments": 4,
            "observation_dimension": spec.observation_dim,
            "action_dimension": spec.action_dim,
            "observation_contract": OBSERVATION_CONTRACT_ID,
            "wandb_project": WANDB_PROJECT,
            "registered_max_epochs": spec.max_epochs,
            "continue_until_converged": bool(continue_until_converged),
            "extended_past_registered_cap": completed_epoch > spec.max_epochs,
            "extension_epochs": max(0, completed_epoch - spec.max_epochs),
            "convergence_rule_met": stopped,
            "post_cap_learning_rate": post_cap_learning_rate(spec.learning_rate),
            "code_commit": identity.get("code_commit"),
            "code_fingerprint": identity.get("code_fingerprint"),
            "early_stop_risk": "Three-point plateau rule can miss slow late-stage improvement.",
            "early_stop_protocol": spec.early_stop_protocol,
        },
    )
    return final["state"]
