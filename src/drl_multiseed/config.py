from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "1.0.0"
ALLOWED_SEEDS = (42, 1337, 2026)
DEFAULT_SEEDS = ALLOWED_SEEDS
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "h3c-drl-paper-reproduction")
OBSERVATION_CONTRACT_ID = "refined-temporal-v1"
EARLY_STOP_PROTOCOL_ID = "post-warmup-patience-v2"


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """Complete algorithm-level contract for one preregistered training task."""

    key: str
    case_key: str
    case_name: str
    algorithm: str
    start_day: int
    test_day: int
    episode_steps: int
    n_steps: int
    batch_size: int
    n_epochs: int
    max_epochs: int
    observation_dim: int
    action_dim: int
    local_observation_dim: int | None
    learning_rate: float
    lr_decay_epochs: int
    critic_learning_rate: float | None = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_envs: int = 4
    eval_interval: int = 25
    early_stop_min_epoch: int = 100
    early_stop_patience: int = 3
    early_stop_min_delta_fraction: float = 0.01
    early_stop_protocol: str = EARLY_STOP_PROTOCOL_ID
    forecast_samples: int = 5
    policy_net: tuple[int, int] = (256, 256)
    action_contract: str = "fixed_25_residual_5"
    comfort_threshold: float = 0.5
    log_std_init: float = 0.0
    time_limit_safe_gae: bool | None = None
    advantage_normalization: str | None = None

    @property
    def steps_per_epoch(self) -> int:
        return self.n_steps * self.num_envs

    @property
    def max_steps(self) -> int:
        return self.steps_per_epoch * self.max_epochs

    def protocol_payload(self) -> dict[str, Any]:
        """Return the seed-independent scientific protocol payload."""

        from .observation_contract import refined_contract_payload
        from .profiles import load_profile

        value = dataclasses.asdict(self)
        value.update(
            {
                "schema": "h3c-drl-training-task-v1",
                "protocol_version": PROTOCOL_VERSION,
                "case_profile": load_profile(self.case_key),
                "observation_contract": refined_contract_payload(self.key),
            }
        )
        return value

    def payload(self, seed: int) -> dict[str, Any]:
        """Return the complete seed-specific scientific payload."""

        value = self.protocol_payload()
        value["seed"] = int(seed)
        return value

    def protocol_hash(self) -> str:
        encoded = json.dumps(
            self.protocol_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def scientific_hash(self, seed: int) -> str:
        encoded = json.dumps(
            self.payload(seed), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


TASKS: dict[str, TaskSpec] = {
    "sz_air_ppo": TaskSpec(
        key="sz_air_ppo",
        case_key="sz_air",
        case_name="bestest_air",
        algorithm="ppo",
        start_day=196,
        test_day=203,
        episode_steps=672,
        n_steps=672,
        batch_size=168,
        n_epochs=10,
        max_epochs=700,
        observation_dim=36,
        action_dim=1,
        local_observation_dim=None,
        learning_rate=5e-4,
        lr_decay_epochs=300,
        log_std_init=0.0,
    ),
    "mz_air_ppo": TaskSpec(
        key="mz_air_ppo",
        case_key="mz_air",
        case_name="multizone_office_simple_air",
        algorithm="ppo",
        start_day=192,
        test_day=199,
        episode_steps=672,
        n_steps=672,
        batch_size=168,
        n_epochs=10,
        max_epochs=700,
        observation_dim=96,
        action_dim=5,
        local_observation_dim=None,
        learning_rate=3e-4,
        lr_decay_epochs=300,
        log_std_init=0.0,
    ),
    "mz_air_mappo": TaskSpec(
        key="mz_air_mappo",
        case_key="mz_air",
        case_name="multizone_office_simple_air",
        algorithm="mappo",
        start_day=192,
        test_day=199,
        episode_steps=672,
        n_steps=672,
        batch_size=168,
        n_epochs=10,
        max_epochs=700,
        observation_dim=96,
        action_dim=5,
        local_observation_dim=36,
        learning_rate=3e-4,
        critic_learning_rate=5e-4,
        lr_decay_epochs=300,
        log_std_init=-1.0,
        time_limit_safe_gae=False,
        advantage_normalization="minibatch",
    ),
    "mz_hydro_ppo": TaskSpec(
        key="mz_hydro_ppo",
        case_key="mz_hydro",
        case_name="multizone_office_simple_hydronic",
        algorithm="ppo",
        start_day=213,
        test_day=220,
        episode_steps=480,
        n_steps=480,
        batch_size=120,
        n_epochs=10,
        max_epochs=700,
        observation_dim=51,
        action_dim=2,
        local_observation_dim=None,
        learning_rate=3e-4,
        lr_decay_epochs=500,
        action_contract="occupied_25_unoccupied_30_residual_5",
        log_std_init=-1.0,
    ),
    "mz_hydro_mappo": TaskSpec(
        key="mz_hydro_mappo",
        case_key="mz_hydro",
        case_name="multizone_office_simple_hydronic",
        algorithm="mappo",
        start_day=213,
        test_day=220,
        episode_steps=480,
        n_steps=480,
        batch_size=120,
        n_epochs=10,
        max_epochs=700,
        observation_dim=51,
        action_dim=2,
        local_observation_dim=36,
        learning_rate=3e-4,
        critic_learning_rate=5e-4,
        lr_decay_epochs=500,
        action_contract="occupied_25_unoccupied_30_residual_5",
        log_std_init=-1.0,
        time_limit_safe_gae=True,
        advantage_normalization="rollout",
    ),
}


def get_task(key: str) -> TaskSpec:
    try:
        return TASKS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown task {key!r}; choose from {sorted(TASKS)}") from exc


def validate_task(spec: TaskSpec, seed: int) -> None:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Seed must be one of {ALLOWED_SEEDS}")
    if spec.algorithm not in {"ppo", "mappo"}:
        raise ValueError("Algorithm must be PPO or MAPPO")
    if spec.num_envs != 4:
        raise ValueError("The published protocol requires four training environments")
    if spec.steps_per_epoch % spec.batch_size:
        raise ValueError("Rollout size must be divisible by batch size")
    if spec.n_steps != spec.episode_steps:
        raise ValueError("One epoch must contain one full episode per environment")
    if spec.eval_interval != 25:
        raise ValueError("The deterministic training-window interval must be 25 epochs")
    if spec.max_epochs != 700:
        raise ValueError("All published tasks use a 700-epoch safety cap")
    if spec.comfort_threshold != 0.5:
        raise ValueError("All published tasks use an occupied PMV threshold of 0.5")
    if spec.ent_coef != 0.02:
        raise ValueError("All published tasks use an entropy coefficient of 0.02")
    if spec.early_stop_protocol != EARLY_STOP_PROTOCOL_ID:
        raise ValueError("Unexpected early-stop protocol")
    if not 0 < spec.lr_decay_epochs <= spec.max_epochs:
        raise ValueError("The learning-rate decay horizon is invalid")
    if spec.forecast_samples != 5:
        raise ValueError("The published forecast input uses t through t+4")
    if spec.algorithm == "mappo":
        if spec.local_observation_dim != 36 or spec.critic_learning_rate is None:
            raise ValueError("MAPPO requires 36-dimensional local observations and a critic LR")
        if spec.advantage_normalization not in {"rollout", "minibatch"}:
            raise ValueError("MAPPO advantage normalization is not explicit")


def source_checkout_root() -> Path | None:
    """Return this package's source checkout without inspecting an unrelated CWD."""

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "drl_multiseed"
        ).is_dir():
            return candidate
    return None


def repository_root() -> Path:
    """Return the source checkout, or the current directory for an installed wheel."""

    checkout = source_checkout_root()
    if checkout is not None:
        return checkout
    return Path.cwd().resolve()
