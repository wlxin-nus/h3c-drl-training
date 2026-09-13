"""Frozen model registry and fail-closed checkpoint loading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

from h3c.experiments.profiles import repository_root


def load_registry() -> dict[str, Any]:
    path = repository_root() / "models" / "registry.json"
    value = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if value.get("schema") != "h3c_frozen_policy_registry" or value.get("schema_version") != 3:
        raise ValueError("frozen policy registry schema is invalid")
    models = value.get("models")
    if not isinstance(models, dict) or len(models) != 5:
        raise ValueError("frozen policy registry must contain exactly five models")
    return value


def model_entry(case: str, controller: str) -> dict[str, Any]:
    matches = [
        dict(entry)
        for entry in load_registry()["models"].values()
        if entry.get("case") == case and entry.get("controller") == controller
    ]
    if len(matches) != 1:
        raise ValueError(f"no unique frozen model for {case}/{controller}")
    return matches[0]


def verify_checkpoint(entry: Mapping[str, Any]) -> dict[str, Any]:
    root = repository_root().resolve()
    path = (root / str(entry["path"])).resolve()
    if not path.is_relative_to(root / "models") or not path.is_file():
        raise ValueError("checkpoint path is missing or escaped models/")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if len(content) != int(entry["bytes"]) or digest != str(entry["sha256"]):
        raise ValueError(f"checkpoint identity mismatch: {path}")
    return {"path": str(path), "bytes": len(content), "sha256": digest}


def load_ppo_checkpoint(entry: Mapping[str, Any]) -> Any:
    """Load inference weights without deserializing legacy training-only schedules."""
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise RuntimeError("install H3C[baselines] to load PPO checkpoints") from error
    path = verify_checkpoint(entry)["path"]
    return PPO.load(
        path,
        device="cpu",
        custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
    )


def verify_all_checkpoints(*, load_cpu: bool = False) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, entry in load_registry()["models"].items():
        identity = verify_checkpoint(entry)
        if load_cpu:
            path = identity["path"]
            if entry["algorithm"] == "ppo":
                model = load_ppo_checkpoint(entry)
                if int(model.observation_space.shape[0]) != int(entry["observation_dimension"]):
                    raise ValueError("PPO observation dimension does not match the registry")
                if int(model.action_space.shape[0]) != int(entry["action_dimension"]):
                    raise ValueError("PPO action dimension does not match the registry")
            else:
                try:
                    import torch
                except ImportError as error:
                    raise RuntimeError(
                        "install H3C[baselines] to load MAPPO checkpoints"
                    ) from error
                candidate = torch.load(path, map_location="cpu", weights_only=False)
                if not isinstance(candidate, dict):
                    raise ValueError("MAPPO checkpoint is not a mapping")
            identity["cpu_load_verified"] = True
        results[name] = identity
    return {"models": results, "verified": True}
