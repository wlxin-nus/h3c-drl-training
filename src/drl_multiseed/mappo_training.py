from __future__ import annotations

import math
import os
import random
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from .checkpointing import AtomicMAPPOCheckpointManager
from .config import OBSERVATION_CONTRACT_ID, WANDB_PROJECT, TaskSpec, validate_task
from .continuation import (
    completion_status,
    epoch_iterator,
    linear_lr_multiplier,
    post_cap_learning_rate,
    validate_continuation_mode,
)
from .early_stop import PlateauState
from .environment import cleanup_run_testids, make_env, prefetch_forecast
from .gae import compute_gae_arrays
from .io import atomic_json, utc_now
from .networks import ActorNetwork, CentralizedCritic
from .tracking import SafeWandb


class RolloutBuffer:
    def __init__(
        self,
        spec: TaskSpec,
        actor_names: Sequence[str],
        *,
        time_limit_safe_gae: bool | None = None,
        advantage_normalization: str | None = None,
    ):
        shape = (spec.n_steps, spec.num_envs)
        self.spec = spec
        self.actor_names = tuple(actor_names)
        self.global_obs = np.zeros((*shape, spec.observation_dim), np.float32)
        self.local_obs = {
            name: np.zeros((*shape, spec.local_observation_dim), np.float32) for name in actor_names
        }
        self.actions = {name: np.zeros((*shape, 1), np.float32) for name in actor_names}
        self.log_probs = {name: np.zeros(shape, np.float32) for name in actor_names}
        self.rewards = np.zeros(shape, np.float32)
        self.values = np.zeros(shape, np.float32)
        self.next_values = np.zeros(shape, np.float32)
        self.terminated = np.zeros(shape, np.float32)
        self.truncated = np.zeros(shape, np.float32)
        self.advantages = np.zeros(shape, np.float32)
        self.returns = np.zeros(shape, np.float32)
        self.ptr = 0
        self.time_limit_safe_gae = (
            spec.case_key == "mz_hydro"
            if time_limit_safe_gae is None
            else bool(time_limit_safe_gae)
        )
        self.advantage_normalization = advantage_normalization or (
            "rollout" if spec.case_key == "mz_hydro" else "minibatch"
        )
        if self.advantage_normalization not in {"rollout", "minibatch"}:
            raise ValueError("advantage_normalization must be rollout or minibatch")

    def add(self, **values: Any) -> None:
        index = self.ptr
        for name in self.actor_names:
            self.local_obs[name][index] = values["local_obs"][name]
            self.actions[name][index] = values["actions"][name]
            self.log_probs[name][index] = values["log_probs"][name]
        for key in ("global_obs", "rewards", "values", "next_values", "terminated", "truncated"):
            getattr(self, key)[index] = values[key]
        self.ptr += 1

    def compute_gae(self) -> None:
        advantages, returns = compute_gae_arrays(
            self.rewards[: self.ptr],
            self.values[: self.ptr],
            self.next_values[: self.ptr],
            self.terminated[: self.ptr],
            self.truncated[: self.ptr],
            gamma=self.spec.gamma,
            gae_lambda=self.spec.gae_lambda,
            time_limit_safe=self.time_limit_safe_gae,
        )
        self.advantages[: self.ptr] = advantages
        self.returns[: self.ptr] = returns

    def batches(self, batch_size: int, generator: np.random.Generator, device: torch.device):
        total = self.ptr * self.spec.num_envs
        indices = generator.permutation(total)
        global_obs = self.global_obs[: self.ptr].reshape(total, -1)
        advantages = self.advantages[: self.ptr].reshape(total)
        returns = self.returns[: self.ptr].reshape(total)
        for start in range(0, total, batch_size):
            selected = indices[start : start + batch_size]
            batch: dict[str, torch.Tensor] = {
                "global_obs": torch.as_tensor(global_obs[selected], device=device),
                "advantages": torch.as_tensor(advantages[selected], device=device),
                "returns": torch.as_tensor(returns[selected], device=device),
            }
            for name in self.actor_names:
                batch[f"obs_{name}"] = torch.as_tensor(
                    self.local_obs[name][: self.ptr].reshape(total, -1)[selected], device=device
                )
                batch[f"actions_{name}"] = torch.as_tensor(
                    self.actions[name][: self.ptr].reshape(total, 1)[selected], device=device
                )
                batch[f"log_probs_{name}"] = torch.as_tensor(
                    self.log_probs[name][: self.ptr].reshape(total)[selected], device=device
                )
            yield batch


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    pd.DataFrame(list(rows)).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _capture_rng() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and "cuda" in payload:
        torch.cuda.set_rng_state_all(payload["cuda"])


class MAPPOExperiment:
    def __init__(
        self,
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
    ):
        validate_task(spec, seed)
        validate_continuation_mode(mode, continue_until_converged)
        if spec.algorithm != "mappo" or spec.local_observation_dim is None:
            raise ValueError("MAPPO trainer received an incompatible task")
        self.spec, self.seed, self.mode, self.endpoint = spec, seed, mode, endpoint
        self.run_dir, self.resume = Path(run_dir), resume
        self.continue_until_converged = bool(continue_until_converged)
        self.wandb_project = WANDB_PROJECT
        self.wandb_group = f"seed{seed}"
        self.time_limit_safe_gae = bool(spec.time_limit_safe_gae)
        self.advantage_normalization = str(spec.advantage_normalization)
        if self.advantage_normalization not in {"rollout", "minibatch"}:
            raise ValueError("MAPPO advantage normalization must be rollout or minibatch")
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cleanup_run_testids(self.run_dir, endpoint)
        identity_path = self.run_dir / "run_identity.json"
        if identity_path.exists():
            import json

            self.identity = json.loads(identity_path.read_text(encoding="utf-8"))
        else:
            self.identity = {
                "run_uuid": f"{spec.key}-seed{seed}-{uuid.uuid4().hex[:12]}",
                "created_at": utc_now(),
            }
            atomic_json(identity_path, self.identity)
        self.tracker = SafeWandb(
            self.run_dir,
            project=self.wandb_project,
            group=self.wandb_group,
            name=f"{spec.key}-seed{seed}-{mode}",
            run_id=self.identity["run_uuid"],
            mode=wandb_mode,
            config=spec.payload(seed),
        )
        self.checkpoint = AtomicMAPPOCheckpointManager(self.run_dir, spec.scientific_hash(seed))

    def _parallel_reset(self, envs: Sequence[Any]) -> np.ndarray:
        rows: list[np.ndarray | None] = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=len(envs)) as pool:
            futures = {
                pool.submit(env.reset, seed=self.seed + index): index
                for index, env in enumerate(envs)
            }
            for future in as_completed(futures):
                rows[futures[future]] = np.asarray(future.result()[0], np.float32)
        return np.stack([row for row in rows if row is not None])

    def _parallel_step(self, envs: Sequence[Any], actions: np.ndarray):
        rows: list[Any] = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=len(envs)) as pool:
            futures = {
                pool.submit(env.step, actions[index]): index for index, env in enumerate(envs)
            }
            for future in as_completed(futures):
                rows[futures[future]] = future.result()
        return rows

    def _evaluate(
        self,
        actors: Mapping[str, ActorNetwork],
        forecast: Mapping[str, Sequence[float]],
        epoch: int,
    ) -> dict[str, Any]:
        env = make_env(
            self.spec,
            self.seed,
            forecast,
            self.endpoint,
            self.run_dir,
            "eval",
            f"train_week_eval_{epoch:04d}",
        )
        rewards: list[float] = []
        costs: list[float] = []
        actions_log: list[list[float]] = []
        try:
            observation, _ = env.reset(seed=self.seed)
            for _ in range(self.spec.episode_steps):
                action_values: list[float] = []
                with torch.no_grad():
                    for raw_zone, name in zip(env.zone_order, actors, strict=True):
                        local = torch.as_tensor(
                            env.local_observations[raw_zone],
                            dtype=torch.float32,
                            device=self.device,
                        ).unsqueeze(0)
                        action_values.append(
                            float(actors[name].sample(local, deterministic=True)[0].item())
                        )
                observation, reward, terminated, truncated, info = env.step(
                    np.asarray(action_values, np.float32)
                )
                rewards.append(float(reward))
                costs.append(float(info["phys_cost"]))
                actions_log.append(action_values)
                if terminated or truncated:
                    break
        finally:
            env.close()
        if len(rewards) != self.spec.episode_steps:
            raise RuntimeError("Incomplete MAPPO deterministic train-window rollout")
        actions = np.asarray(actions_log)
        row = {
            "epoch": epoch,
            "return": float(np.sum(rewards)),
            "cost": float(np.sum(costs)),
            "action_mean": float(actions.mean()),
            "action_std": float(actions.std()),
            "action_saturation": float(np.mean(np.abs(actions) > 0.95)),
            "timestamp": utc_now(),
        }
        directory = self.run_dir / "diagnostics" / "train_window"
        directory.mkdir(parents=True, exist_ok=True)
        _write_rows(
            directory / f"epoch_{epoch:04d}.csv",
            [
                {
                    "step": index,
                    "reward": rewards[index],
                    "cost": costs[index],
                    "actions": str(actions_log[index]),
                }
                for index in range(len(rewards))
            ],
        )
        return row

    def train(self) -> dict[str, Any]:
        latest = self.checkpoint.load_latest()
        if latest and not self.resume:
            raise RuntimeError("Checkpoint exists; use -Resume or a new output directory")
        self.tracker.start()
        forecast = prefetch_forecast(
            self.spec, endpoint=self.endpoint, run_dir=self.run_dir, seed=self.seed
        )
        from .checkpointing import sha256_payload

        fingerprint = sha256_payload({"spec": self.spec.payload(self.seed), "forecast": forecast})
        restored = latest["payload"] if latest else {}
        if restored and restored.get("environment_fingerprint") != fingerprint:
            raise RuntimeError("Environment fingerprint changed; refusing silent resume")
        probe = make_env(
            self.spec, self.seed, forecast, self.endpoint, self.run_dir, "probe", "contract_probe"
        )
        try:
            raw_zone_order = tuple(probe.zone_order)
        finally:
            probe.close()
        actor_names = tuple(zone.lower() for zone in raw_zone_order)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        generator = np.random.default_rng(self.seed)
        actors = {
            name: ActorNetwork(self.spec.local_observation_dim, self.spec.policy_net).to(
                self.device
            )
            for name in actor_names
        }
        critic = CentralizedCritic(self.spec.observation_dim, self.spec.policy_net).to(self.device)
        actor_optimizers = {
            name: optim.Adam(actor.parameters(), lr=self.spec.learning_rate, eps=1e-5)
            for name, actor in actors.items()
        }
        critic_optimizer = optim.Adam(
            critic.parameters(), lr=float(self.spec.critic_learning_rate), eps=1e-5
        )
        decay_epochs = self.spec.lr_decay_epochs

        def schedule(epoch: int) -> float:
            return linear_lr_multiplier(epoch, decay_epochs)

        actor_schedulers = {
            name: optim.lr_scheduler.LambdaLR(opt, lr_lambda=schedule)
            for name, opt in actor_optimizers.items()
        }
        critic_scheduler = optim.lr_scheduler.LambdaLR(critic_optimizer, lr_lambda=schedule)
        plateau = PlateauState.from_dict(restored.get("early_stop"))
        plateau.min_epoch = self.spec.early_stop_min_epoch
        plateau.patience = self.spec.early_stop_patience
        plateau.min_delta_fraction = self.spec.early_stop_min_delta_fraction
        training_rows = list(restored.get("training_rows", []))
        update_rows = list(restored.get("update_rows", []))
        evaluation_rows = list(restored.get("evaluation_rows", []))
        committed = int(restored.get("committed_epoch", 0))
        global_step = int(restored.get("global_step", 0))
        resume_count = int(restored.get("resume_count", 0)) + int(bool(restored))
        if restored:
            for name in actor_names:
                actors[name].load_state_dict(restored["actors"][name])
                actor_optimizers[name].load_state_dict(restored["actor_optimizers"][name])
                actor_schedulers[name].load_state_dict(restored["actor_schedulers"][name])
            critic.load_state_dict(restored["critic"])
            critic_optimizer.load_state_dict(restored["critic_optimizer"])
            critic_scheduler.load_state_dict(restored["critic_scheduler"])
            generator.bit_generator.state = restored["generator_state"]
            _restore_rng(restored["rng"])
        envs: list[Any] = []
        try:
            for rank in range(4):
                envs.append(
                    make_env(
                        self.spec,
                        self.seed,
                        forecast,
                        self.endpoint,
                        self.run_dir,
                        rank,
                        "training",
                    )
                )
            observations = self._parallel_reset(envs)
            buffer = RolloutBuffer(
                self.spec,
                actor_names,
                time_limit_safe_gae=self.time_limit_safe_gae,
                advantage_normalization=self.advantage_normalization,
            )
            epochs = (
                iter(())
                if plateau.stopped
                else epoch_iterator(
                    committed_epoch=committed,
                    registered_max_epochs=self.spec.max_epochs,
                    mode=self.mode,
                    continue_until_converged=self.continue_until_converged,
                )
            )
        except BaseException:
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass
            cleanup_run_testids(self.run_dir, self.endpoint)
            self.tracker.finish({"completed_epoch": committed, "early_stopped": plateau.stopped})
            raise
        try:
            for epoch in epochs:
                started = time.perf_counter()
                buffer.ptr = 0
                episode_returns = np.zeros(4)
                completed: list[float] = []
                component_values = {"energy": [], "comfort": [], "smooth": []}
                for _step in range(self.spec.n_steps):
                    global_obs = observations.copy()
                    tensor_global = torch.as_tensor(global_obs, device=self.device)
                    local: dict[str, np.ndarray] = {}
                    sampled: dict[str, np.ndarray] = {}
                    log_probs: dict[str, np.ndarray] = {}
                    with torch.no_grad():
                        values = critic(tensor_global).cpu().numpy()
                        for raw_zone, name in zip(raw_zone_order, actor_names, strict=True):
                            batch = np.stack(
                                [env.local_observations[raw_zone] for env in envs]
                            ).astype(np.float32)
                            action, log_prob, _ = actors[name].sample(
                                torch.as_tensor(batch, device=self.device)
                            )
                            local[name] = batch
                            sampled[name] = action.cpu().numpy()
                            log_probs[name] = log_prob.cpu().numpy()
                    joint = np.stack(
                        [sampled[name].reshape(-1) for name in actor_names], axis=1
                    ).clip(-1, 1)
                    results = self._parallel_step(envs, joint)
                    terminal_obs = np.stack(
                        [np.asarray(result[0], np.float32) for result in results]
                    )
                    rewards = np.asarray([result[1] for result in results], np.float32)
                    terminated = np.asarray([result[2] for result in results], np.float32)
                    truncated = np.asarray([result[3] for result in results], np.float32)
                    with torch.no_grad():
                        next_values = (
                            critic(torch.as_tensor(terminal_obs, device=self.device)).cpu().numpy()
                        )
                    buffer.add(
                        local_obs=local,
                        actions=sampled,
                        log_probs=log_probs,
                        global_obs=global_obs,
                        rewards=rewards,
                        values=values,
                        next_values=next_values,
                        terminated=terminated,
                        truncated=truncated,
                    )
                    global_step += 4
                    episode_returns += rewards
                    for index, result in enumerate(results):
                        info = result[4]
                        component_values["energy"].append(float(info["rew_energy"]))
                        component_values["comfort"].append(float(info["rew_comfort"]))
                        component_values["smooth"].append(float(info["rew_smooth"]))
                        if terminated[index] or truncated[index]:
                            completed.append(float(episode_returns[index]))
                            episode_returns[index] = 0.0
                    observations = terminal_obs
                    done_indices = [
                        index for index in range(4) if terminated[index] or truncated[index]
                    ]
                    if done_indices:
                        with ThreadPoolExecutor(max_workers=len(done_indices)) as pool:
                            futures = {
                                pool.submit(
                                    envs[index].reset, seed=self.seed + index + epoch
                                ): index
                                for index in done_indices
                            }
                            for future in as_completed(futures):
                                observations[futures[future]] = future.result()[0]
                buffer.compute_gae()
                advantages = buffer.advantages[: buffer.ptr]
                if self.advantage_normalization == "rollout":
                    buffer.advantages[: buffer.ptr] = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                actor_losses = {name: [] for name in actor_names}
                actor_entropies = {name: [] for name in actor_names}
                actor_kls = {name: [] for name in actor_names}
                actor_clips = {name: [] for name in actor_names}
                actor_gradients = {name: [] for name in actor_names}
                critic_losses = []
                critic_gradients = []
                for _ in range(self.spec.n_epochs):
                    for batch in buffer.batches(self.spec.batch_size, generator, self.device):
                        predicted = critic(batch["global_obs"])
                        critic_loss = ((predicted - batch["returns"]) ** 2).mean()
                        critic_optimizer.zero_grad(set_to_none=True)
                        (self.spec.vf_coef * critic_loss).backward()
                        critic_grad = nn.utils.clip_grad_norm_(
                            critic.parameters(), self.spec.max_grad_norm
                        )
                        critic_optimizer.step()
                        critic_losses.append(float(critic_loss.detach().cpu()))
                        critic_gradients.append(float(critic_grad.detach().cpu()))
                        advantage = batch["advantages"]
                        if self.advantage_normalization == "minibatch":
                            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                        for name in actor_names:
                            distribution = actors[name].distribution(batch[f"obs_{name}"])
                            new_log = distribution.log_prob(batch[f"actions_{name}"]).sum(-1)
                            entropy = distribution.entropy().sum(-1)
                            log_ratio = new_log - batch[f"log_probs_{name}"]
                            ratio = torch.exp(log_ratio)
                            loss = (
                                -torch.min(
                                    ratio * advantage,
                                    torch.clamp(
                                        ratio, 1 - self.spec.clip_range, 1 + self.spec.clip_range
                                    )
                                    * advantage,
                                ).mean()
                                - self.spec.ent_coef * entropy.mean()
                            )
                            actor_optimizers[name].zero_grad(set_to_none=True)
                            loss.backward()
                            actor_grad = nn.utils.clip_grad_norm_(
                                actors[name].parameters(), self.spec.max_grad_norm
                            )
                            actor_optimizers[name].step()
                            actor_losses[name].append(float(loss.detach().cpu()))
                            actor_gradients[name].append(float(actor_grad.detach().cpu()))
                            actor_entropies[name].append(float(entropy.mean().detach().cpu()))
                            with torch.no_grad():
                                actor_kls[name].append(
                                    float((torch.exp(log_ratio) - 1 - log_ratio).mean().cpu())
                                )
                                actor_clips[name].append(
                                    float(
                                        (torch.abs(ratio - 1) > self.spec.clip_range)
                                        .float()
                                        .mean()
                                        .cpu()
                                    )
                                )
                for scheduler in actor_schedulers.values():
                    scheduler.step()
                critic_scheduler.step()
                reward_mean = float(np.mean(completed or episode_returns.tolist()))
                prior = [float(row["reward_mean"]) for row in training_rows]
                training_row = {
                    "run_uuid": self.identity["run_uuid"],
                    "epoch": epoch,
                    "global_step": global_step,
                    "reward_mean": reward_mean,
                    "rolling_30": float(np.mean([*prior, reward_mean][-30:])),
                    "reward_energy_mean": float(np.mean(component_values["energy"])),
                    "reward_comfort_mean": float(np.mean(component_values["comfort"])),
                    "reward_smooth_mean": float(np.mean(component_values["smooth"])),
                    "epoch_wall_seconds": time.perf_counter() - started,
                    "timestamp": utc_now(),
                }
                with torch.no_grad():
                    flat_observation = torch.as_tensor(
                        buffer.global_obs[: buffer.ptr].reshape(-1, self.spec.observation_dim),
                        device=self.device,
                    )
                    predicted_values = critic(flat_observation).cpu().numpy()
                target_values = buffer.returns[: buffer.ptr].reshape(-1)
                target_variance = float(np.var(target_values))
                explained_variance = (
                    float(1.0 - np.var(target_values - predicted_values) / target_variance)
                    if target_variance > 1e-12
                    else float("nan")
                )
                update_row = {
                    "run_uuid": self.identity["run_uuid"],
                    "epoch": epoch,
                    "global_step": global_step,
                    "learning_rate": actor_optimizers[actor_names[0]].param_groups[0]["lr"],
                    "critic_learning_rate": critic_optimizer.param_groups[0]["lr"],
                    "policy_std": float(
                        np.mean(
                            [actor.log_std.exp().mean().detach().cpu() for actor in actors.values()]
                        )
                    ),
                    "clip_fraction": float(
                        np.mean([value for rows in actor_clips.values() for value in rows])
                    ),
                    "approx_kl": float(
                        np.mean([value for rows in actor_kls.values() for value in rows])
                    ),
                    "entropy": float(
                        np.mean([value for rows in actor_entropies.values() for value in rows])
                    ),
                    "actor_gradient_norm": float(
                        np.mean([value for rows in actor_gradients.values() for value in rows])
                    ),
                    "critic_gradient_norm": float(np.mean(critic_gradients)),
                    "value_loss": float(np.mean(critic_losses)),
                    "explained_variance": explained_variance,
                    "timestamp": utc_now(),
                }
                for name in actor_names:
                    update_row[f"policy_std_{name}"] = float(
                        actors[name].log_std.exp().mean().detach().cpu()
                    )
                    update_row[f"approx_kl_{name}"] = float(np.mean(actor_kls[name]))
                    update_row[f"clip_fraction_{name}"] = float(np.mean(actor_clips[name]))
                training_rows.append(training_row)
                update_rows.append(update_row)
                actual_best = False
                current_evaluation = None
                interval = 1 if self.mode == "smoke" else self.spec.eval_interval
                if epoch % interval == 0:
                    current_evaluation = self._evaluate(actors, forecast, epoch)
                    stop_update = plateau.update(epoch, current_evaluation["return"])
                    current_evaluation.update(stop_update)
                    evaluation_rows.append(current_evaluation)
                    actual_best = stop_update["actual_best"]
                payload = {
                    "schema": "h3c-drl-mappo-state-v1",
                    "config_hash": self.spec.scientific_hash(self.seed),
                    "protocol_hash": self.spec.protocol_hash(),
                    "observation_contract": OBSERVATION_CONTRACT_ID,
                    "environment_fingerprint": fingerprint,
                    "committed_epoch": epoch,
                    "global_step": global_step,
                    "actors": {name: actor.state_dict() for name, actor in actors.items()},
                    "critic": critic.state_dict(),
                    "actor_optimizers": {
                        name: value.state_dict() for name, value in actor_optimizers.items()
                    },
                    "critic_optimizer": critic_optimizer.state_dict(),
                    "actor_schedulers": {
                        name: value.state_dict() for name, value in actor_schedulers.items()
                    },
                    "critic_scheduler": critic_scheduler.state_dict(),
                    "training_rows": training_rows,
                    "update_rows": update_rows,
                    "evaluation_rows": evaluation_rows,
                    "early_stop": plateau.to_dict(),
                    "early_stop_protocol": self.spec.early_stop_protocol,
                    "best_score": None
                    if not math.isfinite(plateau.best_score)
                    else plateau.best_score,
                    "best_epoch": plateau.best_epoch,
                    "resume_count": resume_count,
                    "registered_max_epochs": self.spec.max_epochs,
                    "continue_until_converged": self.continue_until_converged,
                    "wandb_run_id": self.identity["run_uuid"],
                    "wandb_project": self.wandb_project,
                    "task": self.spec.key,
                    "time_limit_safe_gae": self.time_limit_safe_gae,
                    "advantage_normalization": self.advantage_normalization,
                    "code_commit": self.identity.get("code_commit"),
                    "code_fingerprint": self.identity.get("code_fingerprint"),
                    "rng": _capture_rng(),
                    "generator_state": generator.bit_generator.state,
                    "updated_at": utc_now(),
                }
                pointer = self.checkpoint.save(payload, epoch=epoch, global_step=global_step)
                if actual_best:
                    self.checkpoint.mark_best(pointer, plateau.best_score)
                committed = epoch
                _write_rows(self.run_dir / "training_metrics.csv", training_rows)
                _write_rows(self.run_dir / "updates.csv", update_rows)
                _write_rows(self.run_dir / "train_window_eval.csv", evaluation_rows)
                self.tracker.log(
                    {
                        "epoch": epoch,
                        "train/reward_mean": reward_mean,
                        "train/rolling_30": training_row["rolling_30"],
                        "mappo/policy_std": update_row["policy_std"],
                        "mappo/learning_rate": update_row["learning_rate"],
                        "mappo/critic_learning_rate": update_row["critic_learning_rate"],
                        "mappo/approx_kl": update_row["approx_kl"],
                        "mappo/clip_fraction": update_row["clip_fraction"],
                        "mappo/entropy": update_row["entropy"],
                        "mappo/value_loss": update_row["value_loss"],
                        "mappo/explained_variance": update_row["explained_variance"],
                        "mappo/actor_gradient_norm": update_row["actor_gradient_norm"],
                        "mappo/critic_gradient_norm": update_row["critic_gradient_norm"],
                        "early_stop/misses": plateau.misses,
                        "early_stop/stopped": plateau.stopped,
                        "train_window/return": float("nan")
                        if current_evaluation is None
                        else current_evaluation["return"],
                        "train_window/cost": float("nan")
                        if current_evaluation is None
                        else current_evaluation["cost"],
                        "train_window/action_saturation": float("nan")
                        if current_evaluation is None
                        else current_evaluation["action_saturation"],
                    },
                    step=global_step,
                )
                print(
                    f"[{self.spec.key} seed={self.seed} epoch={epoch:04d}] reward={reward_mean:.3f} best_eval={plateau.best_score:.3f} misses={plateau.misses}"
                )
                if plateau.stopped:
                    break
        finally:
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass
            cleanup_run_testids(self.run_dir, self.endpoint)
            self.tracker.finish({"completed_epoch": committed, "early_stopped": plateau.stopped})
        final = self.checkpoint.load_latest()
        if final is None:
            raise RuntimeError("Training ended before first committed epoch")
        payload = final["payload"]
        completed_epoch = int(payload["committed_epoch"])
        stopped = bool(payload["early_stop"]["stopped"])
        atomic_json(
            self.run_dir / "run_manifest.json",
            {
                "schema": "h3c-drl-run-v1",
                "task": self.spec.key,
                "seed": self.seed,
                "mode": self.mode,
                "protocol_version": self.identity.get("protocol_version"),
                "protocol_hash": self.spec.protocol_hash(),
                "config_hash": self.spec.scientific_hash(self.seed),
                "boptest_version": self.identity.get("boptest_version"),
                "observation_contract": OBSERVATION_CONTRACT_ID,
                "wandb_project": self.wandb_project,
                "run_uuid": self.identity["run_uuid"],
                "status": completion_status(
                    stopped=stopped, reached_cap=completed_epoch >= self.spec.max_epochs
                ),
                "completed_epoch": completed_epoch,
                "global_step": payload["global_step"],
                "best_epoch": payload["best_epoch"],
                "best_score": payload["best_score"],
                "updated_at": utc_now(),
                "selection_basis": "highest deterministic training-window return at registered validation points",
                "held_out_used_for_selection": False,
                "training_environments": 4,
                "observation_dimension": self.spec.observation_dim,
                "local_observation_dimension": self.spec.local_observation_dim,
                "action_dimension": self.spec.action_dim,
                "registered_max_epochs": self.spec.max_epochs,
                "continue_until_converged": self.continue_until_converged,
                "extended_past_registered_cap": completed_epoch > self.spec.max_epochs,
                "extension_epochs": max(0, completed_epoch - self.spec.max_epochs),
                "convergence_rule_met": stopped,
                "time_limit_safe_gae": self.time_limit_safe_gae,
                "advantage_normalization": self.advantage_normalization,
                "post_cap_actor_learning_rate": post_cap_learning_rate(self.spec.learning_rate),
                "post_cap_critic_learning_rate": post_cap_learning_rate(
                    float(self.spec.critic_learning_rate)
                ),
                "code_commit": self.identity.get("code_commit"),
                "code_fingerprint": self.identity.get("code_fingerprint"),
                "early_stop_risk": "Three-point plateau rule can miss slow late-stage improvement.",
                "early_stop_protocol": self.spec.early_stop_protocol,
            },
        )
        return payload


def train_mappo(spec: TaskSpec, **kwargs: Any) -> dict[str, Any]:
    return MAPPOExperiment(spec, **kwargs).train()
