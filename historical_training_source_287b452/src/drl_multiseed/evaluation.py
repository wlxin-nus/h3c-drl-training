from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from CASE_TEST import rl_retraining_v2 as checkpoint_core
from CASE_TEST.rl_retraining_v3 import AtomicTorchCheckpointManager
from h3c_baselines.policies.mappo_adapter import HierarchicalMappoPolicy
from h3c_baselines.policies.ppo_adapter import CentralizedPpoPolicy

from .config import OBSERVATION_CONTRACT_ID, TaskSpec, repository_root
from .environment import cleanup_run_testids, make_env, prefetch_forecast
from .io import atomic_json, sha256_file, utc_now
from .networks import ActorNetwork
from .observation_contract import refined_model_entry


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    pd.DataFrame(list(rows)).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _best_ppo(spec: TaskSpec, seed: int, run_dir: Path, device: str):
    manager = checkpoint_core.AtomicCheckpointManager(run_dir, spec.scientific_hash(seed))
    best = manager.load_best_metadata()
    model = checkpoint_core.Phase15fPPO.load(
        best["model_path"], device=device,
        custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
    )
    best["state"] = json.loads(Path(best["state_path"]).read_text(encoding="utf-8"))
    return model, best


def _best_mappo(spec: TaskSpec, seed: int, run_dir: Path, device: str):
    manager = AtomicTorchCheckpointManager(run_dir, spec.scientific_hash(seed))
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
        "occupied_action_saturation": float(((np.abs(actions) > 0.95) & occupied).sum() / occupied_count),
        "action_mean_occupied": float(actions[occupied].mean()) if occupied.any() else 0.0,
        "action_mean_unoccupied": float(actions[~occupied].mean()) if (~occupied).any() else 0.0,
        "action_occ_unocc_gap": float(abs(
            (actions[occupied].mean() if occupied.any() else 0.0)
            - (actions[~occupied].mean() if (~occupied).any() else 0.0)
        )),
        "steps": float(len(rows)),
        "zones": float(zones),
    }


def export_best(spec: TaskSpec, *, seed: int, run_dir: Path, device: str = "cpu") -> dict[str, Any]:
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
    generated = (
        repository_root() / "models" / "generated" / "refine" / spec.key / f"seed{seed}"
    )
    generated.mkdir(parents=True, exist_ok=True)
    adapter_copy = generated / destination.name
    shutil.copy2(destination, adapter_copy)
    entry = refined_model_entry(spec.key)
    entry.update({
        "path": adapter_copy.relative_to(repository_root()).as_posix(),
        "sha256": sha256_file(adapter_copy), "bytes": adapter_copy.stat().st_size,
        "epoch": epoch, "training_steps": step, "seed": seed,
        "selection_basis": "maximum preregistered deterministic training-window return",
    })
    # This is the final compatibility gate: load the exported artifact through
    # the same inference class used by H3C, not merely through the trainer.
    if spec.algorithm == "ppo":
        CentralizedPpoPolicy(entry)
    else:
        HierarchicalMappoPolicy(entry)
    manifest = {
        "schema": "h3c-drl-refined-export-v1", "task": spec.key, "algorithm": spec.algorithm,
        "seed": seed, "epoch": epoch, "global_step": step,
        "observation_dimension": spec.observation_dim,
        "local_observation_dimension": spec.local_observation_dim,
        "action_dimension": spec.action_dim, "policy_net": list(spec.policy_net),
        "observation_contract": OBSERVATION_CONTRACT_ID,
        "checkpoint": destination.name, "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "h3c_adapter": "CentralizedPpoPolicy" if spec.algorithm == "ppo" else "HierarchicalMappoPolicy",
        "h3c_registry_entry": entry,
        "h3c_adapter_load_verified": True,
        "config_hash": spec.scientific_hash(seed), "exported_at": utc_now(),
    }
    atomic_json(output / "model_manifest.json", manifest)
    return manifest


def evaluate_best(
    spec: TaskSpec, *, seed: int, endpoint: str, run_dir: Path,
    device: str = "cpu", force: bool = False,
) -> dict[str, Any]:
    result_dir = run_dir / "formal_evaluation"
    report_path = result_dir / "metrics.json"
    export = export_best(spec, seed=seed, run_dir=run_dir, device=device)
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("model_sha256") == export["sha256"]:
            return existing
        raise RuntimeError("Existing formal evaluation belongs to a different best model")
    cleanup_run_testids(run_dir, endpoint)
    forecast = prefetch_forecast(spec, endpoint=endpoint, run_dir=run_dir, seed=seed, evaluation=True)
    env = make_env(
        spec, seed, forecast, endpoint, run_dir, "formal", "formal_evaluation",
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
                        values.append(float(policy[name].sample(local, deterministic=True)[0].item()))
                action = np.asarray(values, np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            rows.append({
                "step": step, "time": info["time"], "reward": reward,
                "cost": info["phys_cost"], "power_total_W": info["power_total"],
                "energy_step_kwh": float(info["power_total"]) * 0.25 / 1000.0,
                "price": info["price"], "action": json.dumps(info["actions"]),
                "setpoint_c": json.dumps(info["setpoints"]),
                "temperature_c": json.dumps(info["temperatures"]),
                "pmv": json.dumps(info["pmvs"]),
                "occupancy": json.dumps(info["occupancies"]),
                "normalized_observation": json.dumps(info["normalized_observation"]),
            })
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
        "schema": "h3c-drl-refined-formal-evaluation-v1", "task": spec.key,
        "seed": seed, "test_start_day": spec.test_day,
        "steps": spec.episode_steps, "model_sha256": export["sha256"],
        "trajectory_sha256": sha256_file(trajectory),
        "metrics": _metrics(rows, spec.action_dim), "completed_at": utc_now(),
        "selection_leakage": False,
        "short_window_limitation": "The 5/7-day window does not fully satisfy a longer-period evaluation request.",
    }
    atomic_json(report_path, report)
    return report
