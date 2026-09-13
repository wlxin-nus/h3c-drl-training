from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import checkpointing as checkpoint_core
from .checkpointing import AtomicMAPPOCheckpointManager
from .config import OBSERVATION_CONTRACT_ID, PROTOCOL_VERSION, TaskSpec
from .environment import cleanup_run_testids, make_env, prefetch_forecast
from .io import atomic_json, sha256_file, utc_now
from .networks import ActorNetwork
from .observation_contract import refined_model_entry
from .source_identity import code_fingerprint


def _run_metadata(spec: TaskSpec, seed: int, run_dir: Path) -> dict[str, Any]:
    """Load and cross-check the identities required for a formal evaluation."""

    paths = {
        "identity": run_dir / "run_identity.json",
        "manifest": run_dir / "run_manifest.json",
        "preflight": run_dir / "preflight.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Formal evaluation requires run metadata: {missing}")
    values = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    identity = values["identity"]
    manifest = values["manifest"]
    preflight = values["preflight"]
    expected = {
        "task": spec.key,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": spec.protocol_hash(),
        "config_hash": spec.scientific_hash(seed),
    }
    for source_name, source in (("identity", identity), ("manifest", manifest)):
        for field, expected_value in expected.items():
            if source.get(field) != expected_value:
                raise RuntimeError(
                    f"{source_name} {field} does not match the frozen evaluation task"
                )
    task_preflight = preflight.get("tasks", {}).get(spec.key, {})
    if task_preflight.get("scientific_hash") != expected["config_hash"]:
        raise RuntimeError("preflight scientific hash does not match the evaluated run")
    versions = {
        identity.get("boptest_version"),
        manifest.get("boptest_version"),
        preflight.get("boptest_version"),
    }
    if None in versions or len(versions) != 1:
        raise RuntimeError("BOPTEST version identity is missing or inconsistent")
    if identity.get("code_fingerprint") != manifest.get("code_fingerprint"):
        raise RuntimeError("source fingerprint differs between run identity and manifest")
    if identity.get("code_fingerprint") != code_fingerprint():
        raise RuntimeError(
            "current source fingerprint differs from the trained run; use the exact release"
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": expected["protocol_hash"],
        "config_hash": expected["config_hash"],
        "code_commit": identity.get("code_commit"),
        "code_fingerprint": identity.get("code_fingerprint"),
        "boptest_version": versions.pop(),
        "run_uuid": identity.get("run_uuid"),
    }


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    pd.DataFrame(list(rows)).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _best_ppo(spec: TaskSpec, seed: int, run_dir: Path, device: str):
    manager = checkpoint_core.AtomicCheckpointManager(run_dir, spec.scientific_hash(seed))
    best = manager.load_best_metadata()
    model = checkpoint_core.TransactionalPPO.load(
        best["model_path"],
        device=device,
        custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
    )
    best["state"] = json.loads(Path(best["state_path"]).read_text(encoding="utf-8"))
    return model, best


def _best_mappo(spec: TaskSpec, seed: int, run_dir: Path, device: str):
    manager = AtomicMAPPOCheckpointManager(run_dir, spec.scientific_hash(seed))
    best = manager.load_best()
    payload = best["payload"]
    actors: dict[str, ActorNetwork] = {}
    for name, state in payload["actors"].items():
        actor = ActorNetwork(int(spec.local_observation_dim), spec.policy_net).to(device)
        actor.load_state_dict(state)
        actor.eval()
        actors[name] = actor
    return actors, best


def _metrics(rows: Sequence[Mapping[str, Any]], zones: int) -> dict[str, float]:
    costs = np.asarray([row["cost"] for row in rows], dtype=float)
    energy = np.asarray([row["energy_step_kwh"] for row in rows], dtype=float)
    pmv = np.asarray([json.loads(str(row["pmv"])) for row in rows], dtype=float)
    occ = np.asarray([json.loads(str(row["occupancy"])) for row in rows], dtype=float)
    actions = np.asarray([json.loads(str(row["action"])) for row in rows], dtype=float)
    occupied = occ > 0
    violation = occupied & (np.abs(pmv) > 0.5)
    deep = occupied & (np.abs(pmv) > 0.6)
    occupied_count = max(1, int(occupied.sum()))
    return {
        "return": float(sum(float(row["reward"]) for row in rows)),
        "cost": float(costs.sum()),
        "energy_kwh": float(energy.sum()),
        "occupied_zone_hours": float(violation.sum() * 0.25),
        "pmv_hours": float((np.maximum(0.0, np.abs(pmv) - 0.5) * occupied).sum() * 0.25),
        "occupied_pmv_violation_rate": float(violation.sum() / occupied_count),
        "deep_pmv_violation_rate": float(deep.sum() / occupied_count),
        "occupied_action_saturation": float(
            ((np.abs(actions) > 0.95) & occupied).sum() / occupied_count
        ),
        "action_mean_occupied": float(actions[occupied].mean()) if occupied.any() else 0.0,
        "action_mean_unoccupied": float(actions[~occupied].mean()) if (~occupied).any() else 0.0,
        "action_occ_unocc_gap": float(
            abs(
                (actions[occupied].mean() if occupied.any() else 0.0)
                - (actions[~occupied].mean() if (~occupied).any() else 0.0)
            )
        ),
        "steps": float(len(rows)),
        "zones": float(zones),
    }


def export_best(spec: TaskSpec, *, seed: int, run_dir: Path, device: str = "cpu") -> dict[str, Any]:
    metadata = _run_metadata(spec, seed, run_dir)
    output = run_dir / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    if spec.algorithm == "ppo":
        _model, best = _best_ppo(spec, seed, run_dir, device)
        source = Path(best["model_path"])
        destination = output / "best_model_ppo.zip"
        shutil.copy2(source, destination)
        epoch = int(best["state"]["committed_epoch"])
        step = int(best["state"]["global_step"])
    else:
        _actors, best = _best_mappo(spec, seed, run_dir, device)
        source = Path(best["bundle_path"])
        destination = output / "mappo_best.pt"
        shutil.copy2(source, destination)
        epoch = int(best["payload"]["committed_epoch"])
        step = int(best["payload"]["global_step"])
    entry = refined_model_entry(spec.key)
    entry.update(
        {
            "path": destination.relative_to(run_dir).as_posix(),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
            "epoch": epoch,
            "training_steps": step,
            "seed": seed,
            "selection_basis": "maximum preregistered deterministic training-window return",
        }
    )
    # Verify the exported artifact independently of the in-memory trainer.
    if spec.algorithm == "ppo":
        checkpoint_core.TransactionalPPO.load(
            destination,
            device="cpu",
            custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
        )
    else:
        exported = torch.load(destination, map_location="cpu", weights_only=False)
        for _name, state in exported["actors"].items():
            actor = ActorNetwork(int(spec.local_observation_dim), spec.policy_net)
            actor.load_state_dict(state)
            actor.eval()
    manifest = {
        "schema": "h3c-drl-export-v1",
        "task": spec.key,
        "algorithm": spec.algorithm,
        "seed": seed,
        "epoch": epoch,
        "global_step": step,
        "observation_dimension": spec.observation_dim,
        "local_observation_dimension": spec.local_observation_dim,
        "action_dimension": spec.action_dim,
        "policy_net": list(spec.policy_net),
        "observation_contract": OBSERVATION_CONTRACT_ID,
        "checkpoint": destination.name,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "deployment_contract": entry,
        "export_load_verified": True,
        **metadata,
        "exported_at": utc_now(),
    }
    atomic_json(output / "model_manifest.json", manifest)
    return manifest


def evaluate_best(
    spec: TaskSpec,
    *,
    seed: int,
    endpoint: str,
    run_dir: Path,
    device: str = "cpu",
    force: bool = False,
) -> dict[str, Any]:
    metadata = _run_metadata(spec, seed, run_dir)
    result_dir = run_dir / "formal_evaluation"
    report_path = result_dir / "metrics.json"
    export = export_best(spec, seed=seed, run_dir=run_dir, device=device)
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        same_identity = all(existing.get(key) == value for key, value in metadata.items())
        if existing.get("model_sha256") == export["sha256"] and same_identity:
            return existing
        raise RuntimeError(
            "Existing formal evaluation belongs to a different model or run identity"
        )
    cleanup_run_testids(run_dir, endpoint)
    forecast = prefetch_forecast(
        spec, endpoint=endpoint, run_dir=run_dir, seed=seed, evaluation=True
    )
    env = make_env(
        spec,
        seed,
        forecast,
        endpoint,
        run_dir,
        "formal",
        "formal_evaluation",
        evaluation=True,
    )
    if spec.algorithm == "ppo":
        policy, _best = _best_ppo(spec, seed, run_dir, device)
    else:
        policy, _best = _best_mappo(spec, seed, run_dir, device)
    rows: list[dict[str, Any]] = []
    try:
        observation, _ = env.reset(seed=seed)
        for step in range(spec.episode_steps):
            if spec.algorithm == "ppo":
                action = np.asarray(policy.predict(observation, deterministic=True)[0], np.float32)
            else:
                values: list[float] = []
                with torch.no_grad():
                    for raw_zone in env.zone_order:
                        name = raw_zone.lower()
                        local = torch.as_tensor(
                            env.local_observations[raw_zone], dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        values.append(
                            float(policy[name].sample(local, deterministic=True)[0].item())
                        )
                action = np.asarray(values, np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            rows.append(
                {
                    "step": step,
                    "time": info["time"],
                    "reward": reward,
                    "cost": info["phys_cost"],
                    "power_total_W": info["power_total"],
                    "energy_step_kwh": float(info["power_total"]) * 0.25 / 1000.0,
                    "price": info["price"],
                    "action": json.dumps(info["actions"]),
                    "setpoint_c": json.dumps(info["setpoints"]),
                    "temperature_c": json.dumps(info["temperatures"]),
                    "pmv": json.dumps(info["pmvs"]),
                    "occupancy": json.dumps(info["occupancies"]),
                    "normalized_observation": json.dumps(info["normalized_observation"]),
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()
        cleanup_run_testids(run_dir, endpoint)
    if len(rows) != spec.episode_steps:
        raise RuntimeError(
            f"Formal evaluation incomplete ({len(rows)}/{spec.episode_steps}); no result committed"
        )
    expected = [spec.test_day * 86400 + index * 900 for index in range(spec.episode_steps)]
    if [int(row["time"]) for row in rows] != expected:
        raise RuntimeError("Formal evaluation timestamps are incomplete or discontinuous")
    result_dir.mkdir(parents=True, exist_ok=True)
    trajectory = result_dir / "trajectory.csv"
    _atomic_csv(trajectory, rows)
    report: dict[str, Any] = {
        "schema": "h3c-drl-formal-evaluation-v1",
        "task": spec.key,
        "seed": seed,
        "test_start_day": spec.test_day,
        "steps": spec.episode_steps,
        "model_sha256": export["sha256"],
        "trajectory_sha256": sha256_file(trajectory),
        "metrics": _metrics(rows, spec.action_dim),
        "completed_at": utc_now(),
        "selection_leakage": False,
        "short_window_limitation": "The 5/7-day window does not fully satisfy a longer-period evaluation request.",
        **metadata,
    }
    atomic_json(report_path, report)
    return report
