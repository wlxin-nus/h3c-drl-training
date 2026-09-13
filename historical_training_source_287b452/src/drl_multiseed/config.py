from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_SEEDS = (42, 1337, 2026)
DEFAULT_SEEDS = (1337, 2026)
WANDB_PROJECT = "h3c-drl-multiseed-refine-v2"
OBSERVATION_CONTRACT_ID = "refined-temporal-v1"
EARLY_STOP_PROTOCOL_ID = "post-warmup-patience-v2"


@dataclasses.dataclass(frozen=True)
class TaskSpec:
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
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_envs: int = 4
    eval_interval: int = 25
    early_stop_min_epoch: int = 100
    early_stop_patience: int = 3
    early_stop_min_delta_fraction: float = 0.01
    early_stop_protocol: str = EARLY_STOP_PROTOCOL_ID
    forecast_steps: int = 5
    policy_net: tuple[int, int] = (256, 256)
    action_contract: str = "fixed_25_residual_5"
    comfort_threshold: float = 0.5
    source_owner: str = ""
    notes: tuple[str, ...] = ()

    @property
    def steps_per_epoch(self) -> int:
        return self.n_steps * self.num_envs

    @property
    def max_steps(self) -> int:
        return self.steps_per_epoch * self.max_epochs

    def payload(self, seed: int) -> dict[str, Any]:
        from .observation_contract import refined_contract_payload

        value = dataclasses.asdict(self)
        value["seed"] = int(seed)
        value["schema"] = "h3c-drl-multiseed-refined-task-v1"
        value["observation_contract"] = refined_contract_payload(self.key)
        return value

    def scientific_hash(self, seed: int) -> str:
        encoded = json.dumps(
            self.payload(seed), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


TASKS: dict[str, TaskSpec] = {
    "sz_air_ppo": TaskSpec(
        key="sz_air_ppo", case_key="sz_air", case_name="bestest_air",
        algorithm="ppo", start_day=196, test_day=203, episode_steps=672,
        n_steps=672, batch_size=168, n_epochs=10, max_epochs=700,
        observation_dim=36, action_dim=1, local_observation_dim=None,
        learning_rate=5e-4, lr_decay_epochs=300,
        ent_coef=0.02, comfort_threshold=0.5,
        source_owner="FinalSZAIR.ipynb model card + unified refine acceptance settings",
        notes=("SB3 PPO", "fixed 25 C residual base", "refined temporal contract", "LR decay horizon preserved at 300 epochs"),
    ),
    "mz_air_ppo": TaskSpec(
        key="mz_air_ppo", case_key="mz_air", case_name="multizone_office_simple_air",
        algorithm="ppo", start_day=192, test_day=199, episode_steps=672,
        n_steps=672, batch_size=168, n_epochs=10, max_epochs=700,
        observation_dim=96, action_dim=5, local_observation_dim=None,
        learning_rate=3e-4, lr_decay_epochs=300,
        ent_coef=0.02, comfort_threshold=0.5,
        source_owner="FinalMZAIR.ipynb settings + unified refine acceptance settings",
        notes=("SB3 PPO", "fixed 25 C residual base", "effective-count occupancy", "LR decay horizon preserved at 300 epochs"),
    ),
    "mz_air_mappo": TaskSpec(
        key="mz_air_mappo", case_key="mz_air", case_name="multizone_office_simple_air",
        algorithm="mappo", start_day=192, test_day=199, episode_steps=672,
        n_steps=672, batch_size=168, n_epochs=10, max_epochs=700,
        observation_dim=96, action_dim=5, local_observation_dim=36,
        learning_rate=3e-4, lr_decay_epochs=300, critic_learning_rate=5e-4,
        ent_coef=0.02, comfort_threshold=0.5,
        source_owner="MAPPO.ipynb settings + unified refine acceptance settings",
        notes=("five actors", "central critic", "legacy done-based GAE", "zone-then-shared local layout", "LR decay horizon preserved at 300 epochs"),
    ),
    "mz_hydro_ppo": TaskSpec(
        key="mz_hydro_ppo", case_key="mz_hydro", case_name="multizone_office_simple_hydronic",
        algorithm="ppo", start_day=213, test_day=220, episode_steps=480,
        n_steps=480, batch_size=120, n_epochs=10, max_epochs=700,
        observation_dim=51, action_dim=2, local_observation_dim=None,
        learning_rate=3e-4, lr_decay_epochs=500,
        ent_coef=0.02, comfort_threshold=0.5,
        source_owner="Phase1.5f Wave-2 + 1 h continuation",
        action_contract="occupied_25_unoccupied_30_residual_5",
        notes=("log_std_init=-1", "LR reaches 3e-5 at epoch 500 and remains fixed"),
    ),
    "mz_hydro_mappo": TaskSpec(
        key="mz_hydro_mappo", case_key="mz_hydro", case_name="multizone_office_simple_hydronic",
        algorithm="mappo", start_day=213, test_day=220, episode_steps=480,
        n_steps=480, batch_size=120, n_epochs=10, max_epochs=700,
        observation_dim=51, action_dim=2, local_observation_dim=36,
        learning_rate=3e-4, lr_decay_epochs=500, critic_learning_rate=5e-4,
        ent_coef=0.02, comfort_threshold=0.5,
        source_owner="Phase1.5f v3 + 1 h continuation",
        action_contract="occupied_25_unoccupied_30_residual_5",
        notes=("two actors", "central critic", "time-limit-safe GAE", "LR floors after epoch 500"),
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
    if spec.num_envs != 4:
        raise ValueError("n_envs=4 is a frozen scientific setting")
    if spec.steps_per_epoch % spec.batch_size:
        raise ValueError("Rollout size must be divisible by batch_size")
    if spec.n_steps != spec.episode_steps:
        raise ValueError("One epoch must contain one full episode per environment")
    if spec.eval_interval != 25:
        raise ValueError("Deterministic train-window evaluation interval must be 25")
    if spec.max_epochs != 700:
        raise ValueError("All refined tasks use the same 700-epoch safety cap")
    if spec.comfort_threshold != 0.5:
        raise ValueError("All refined tasks use the same occupied PMV threshold of 0.5")
    if spec.ent_coef != 0.02:
        raise ValueError("All refined tasks use the same entropy coefficient of 0.02")
    if spec.early_stop_protocol != EARLY_STOP_PROTOCOL_ID:
        raise ValueError("All refined tasks require post-warm-up patience semantics")
    if not 0 < spec.lr_decay_epochs <= spec.max_epochs:
        raise ValueError("LR decay horizon must be positive and no greater than max_epochs")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]
