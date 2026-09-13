"""Isolated continuation of the final 1 h Hydronic PPO/MAPPO runs.

The completed Wave-2 PPO and v3 MAPPO result trees are immutable parents.  A
continuation run imports their epoch-500 *latest* optimizer/RNG state into a
new checkpoint namespace, keeps the already reached learning-rate floor, and
trains epochs 501..700 (501..502 for smoke).  Historical deterministic
training-week best checkpoints remain eligible for final selection.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import time
import traceback
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from CASE_TEST import rl_retraining_v2 as core
from CASE_TEST import rl_retraining_v3 as v3
from CASE_TEST import rl_retraining_wave2 as wave2


SCHEMA = "hydronic-1h-continuation-700-v1"
PARENT_EPOCH = 500
PARENT_GLOBAL_STEP = 960_000
FULL_EXTENSION_EPOCHS = 200
SMOKE_EXTENSION_EPOCHS = 2
PPO_FLOOR_LR = 3e-5
MAPPO_ACTOR_FLOOR_LR = 3e-5
MAPPO_CRITIC_FLOOR_LR = 5e-5
EXPECTED_GLOBAL_OBSERVATION_DIM = 45
EXPECTED_LOCAL_OBSERVATION_DIM = 33
AUDIT_EPOCHS = (350, 499, 500)
AUDIT_RETURN_ABS_TOLERANCE = 0.1
AUDIT_KPI_RELATIVE_TOLERANCE = 0.001
DELIVERY_PREFIX = "MZ_Hydronic_Final_PPO_MAPPO_1h_Epoch700"


class ConstantSchedule:
    """Cloudpickle-safe constant SB3 learning-rate schedule."""

    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, _progress_remaining: float) -> float:
        return self.value


@dataclasses.dataclass
class HydronicContinuationConfig(wave2.HydronicWave2Config):
    parent_config_hash: str = ""
    parent_epoch: int = PARENT_EPOCH
    extension_epochs: int = FULL_EXTENSION_EPOCHS
    smoke_extension_epochs: int = SMOKE_EXTENSION_EPOCHS
    fixed_actor_learning_rate: float = PPO_FLOOR_LR
    fixed_critic_learning_rate: float = MAPPO_CRITIC_FLOOR_LR

    @property
    def max_epochs(self) -> int:
        additional = (
            self.smoke_extension_epochs
            if self.run_mode == "smoke"
            else self.extension_epochs
        )
        return int(self.parent_epoch + additional)

    @property
    def scientific_payload(self) -> dict[str, Any]:
        payload = dict(super().scientific_payload)
        payload.update(
            {
                "schema": SCHEMA,
                "implementation_sha256": core.sha256_file(Path(__file__).resolve()),
                "parent_config_hash": self.parent_config_hash,
                "continuation": {
                    "parent_epoch": self.parent_epoch,
                    "target_epoch": self.max_epochs,
                    "extension_epochs": self.max_epochs - self.parent_epoch,
                    "learning_rate_after_parent": "constant_at_original_floor",
                    "actor_floor": self.fixed_actor_learning_rate,
                    "critic_floor": self.fixed_critic_learning_rate,
                },
            }
        )
        return payload


@dataclasses.dataclass(frozen=True)
class HydronicContinuationSuite:
    ppo: HydronicContinuationConfig
    mappo: HydronicContinuationConfig
    force_final_eval: bool = False

    @property
    def output_root(self) -> Path:
        return self.ppo.output_root

    @property
    def run_dir(self) -> Path:
        return self.ppo.suite_dir

    @property
    def audit_dir(self) -> Path:
        return self.output_root / "pre_extension_audit"

    @property
    def source_wave2(self) -> wave2.HydronicWave2Suite:
        return wave2.build_hydronic_wave2_suite(
            self.ppo.project_root,
            run_mode="full",
            resume=True,
            wandb_mode="disabled",
            seed=42,
        )

    @property
    def source_v3(self) -> v3.HydronicV3Suite:
        return v3.build_hydronic_v3_suite(
            self.ppo.project_root,
            run_mode="full",
            resume=True,
            wandb_mode="disabled",
            seed=42,
        )

    @property
    def evaluation_config(self) -> HydronicContinuationConfig:
        return dataclasses.replace(
            self.ppo,
            algorithm="evaluation",
            run_tag=f"seed{self.ppo.seed}",
            wandb_mode="disabled",
        )


def _dataclass_values(instance: Any, target: type[Any]) -> dict[str, Any]:
    return {
        field.name: getattr(instance, field.name)
        for field in dataclasses.fields(target)
        if hasattr(instance, field.name)
    }


def build_hydronic_1h_continuation_suite(
    project_root: str | Path,
    *,
    run_mode: str = "smoke",
    resume: bool = True,
    wandb_mode: str = "online",
    seed: int = 42,
    extension_epochs: int = FULL_EXTENSION_EPOCHS,
    force_final_eval: bool = False,
) -> HydronicContinuationSuite:
    if run_mode not in {"smoke", "full"}:
        raise ValueError("RUN_MODE must be 'smoke' or 'full'")
    if seed != 42:
        raise ValueError("The registered continuation is seed-42 only")
    if int(extension_epochs) != FULL_EXTENSION_EPOCHS:
        raise ValueError("The full continuation is preregistered for exactly 200 epochs")
    root = Path(project_root).resolve()
    source_wave2 = wave2.build_hydronic_wave2_suite(
        root, run_mode="full", resume=True, wandb_mode="disabled", seed=42
    )
    source_v3 = v3.build_hydronic_v3_suite(
        root, run_mode="full", resume=True, wandb_mode="disabled", seed=42
    )
    output_root = (
        root / "CASE_TEST" / "MZ_OFFICE_HYDRONIC" / "phase1_5f_1h_continue_700"
    )
    common = _dataclass_values(source_wave2.ppo, wave2.HydronicWave2Config)
    common.update(
        {
            "output_root": output_root,
            "run_mode": run_mode,
            "resume": bool(resume),
            "wandb_mode": wandb_mode,
            "run_tag": f"seed{seed}",
            "seed": seed,
            "wandb_project": "drl-hydronic-1h-continuation",
            "wandb_group": f"mz-hydro-1h-continue700-seed{seed}",
        }
    )
    ppo_values = dict(common)
    ppo_values.update(
        {
            "algorithm": "ppo",
            "parent_config_hash": source_wave2.ppo.config_hash,
            "extension_epochs": int(extension_epochs),
            "fixed_actor_learning_rate": PPO_FLOOR_LR,
            "fixed_critic_learning_rate": MAPPO_CRITIC_FLOOR_LR,
        }
    )
    ppo = HydronicContinuationConfig(**ppo_values)
    mappo_values = dict(common)
    for name in (
        "actor_learning_rate",
        "critic_learning_rate",
        "value_loss_coef",
        "max_grad_norm",
    ):
        mappo_values[name] = getattr(source_v3.mappo, name)
    mappo_values.update(
        {
            "algorithm": "mappo",
            "parent_config_hash": source_v3.mappo.config_hash,
            "extension_epochs": int(extension_epochs),
            "fixed_actor_learning_rate": MAPPO_ACTOR_FLOOR_LR,
            "fixed_critic_learning_rate": MAPPO_CRITIC_FLOOR_LR,
        }
    )
    mappo = HydronicContinuationConfig(**mappo_values)
    return HydronicContinuationSuite(
        ppo=ppo,
        mappo=mappo,
        force_final_eval=force_final_eval,
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_path_isolation(suite: HydronicContinuationSuite) -> None:
    output = suite.output_root.resolve()
    forbidden = (
        suite.source_wave2.ppo.output_root.resolve(),
        suite.source_v3.ppo.output_root.resolve(),
        (suite.ppo.project_root / "DELIVERY").resolve(),
        (
            suite.ppo.project_root
            / "CASE_TEST"
            / "MZ_OFFICE_HYDRONIC"
            / "forecast_horizon_3h"
        ).resolve(),
    )
    if "phase1_5f_1h_continue_700" not in output.parts:
        raise RuntimeError(f"Unexpected continuation output root: {output}")
    for source in forbidden:
        if output == source or _is_relative_to(output, source):
            raise RuntimeError(f"Continuation output overlaps immutable source: {source}")


def _pointer_artifacts(directory: Path, pointer_name: str) -> set[Path]:
    pointer_path = directory / pointer_name
    if not pointer_path.exists():
        return set()
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = directory / str(pointer["manifest"])
    result = {pointer_path, manifest_path}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "files" in manifest:
        result.update(
            directory / str(record["name"])
            for record in manifest["files"].values()
        )
    if "bundle" in manifest:
        result.add(directory / str(manifest["bundle"]["name"]))
    return result


def _source_files(suite: HydronicContinuationSuite) -> list[Path]:
    root = suite.ppo.project_root
    paths: set[Path] = {
        Path(core.__file__).resolve(),
        Path(v3.__file__).resolve(),
        Path(wave2.__file__).resolve(),
        root / "CASE_TEST" / "MZ_OFFICE_HYDRONIC" / "PPO_Phase1_5f_Wave2.ipynb",
        root
        / "CASE_TEST"
        / "MZ_OFFICE_HYDRONIC"
        / "PPO_MAPPO_Phase1_5f_Retrain.ipynb",
    }
    for config in (suite.source_wave2.ppo, suite.source_v3.mappo):
        for name in (
            "run_manifest.json",
            "training_metrics.csv",
            "sb3_updates.csv",
            "train_week_eval.csv",
            "forecast_composition.json",
        ):
            paths.add(config.run_dir / name)
        paths.update(_pointer_artifacts(config.run_dir / "checkpoints", "best.json"))
        paths.update(_pointer_artifacts(config.run_dir / "checkpoints", "latest.json"))
    mappo_499 = (
        suite.source_v3.mappo.run_dir
        / "checkpoints"
        / "epoch_0499_steps_000958080.manifest.json"
    )
    paths.add(mappo_499)
    if mappo_499.exists():
        manifest = json.loads(mappo_499.read_text(encoding="utf-8"))
        paths.add(mappo_499.parent / str(manifest["bundle"]["name"]))
    h3_module = root / "CASE_TEST" / "rl_hydronic_forecast_3h.py"
    h3_notebook = (
        root / "CASE_TEST" / "MZ_OFFICE_HYDRONIC" / "PPO_MAPPO_3h_Forecast.ipynb"
    )
    paths.update((h3_module, h3_notebook))
    for report in (
        root
        / "CASE_TEST"
        / "MZ_OFFICE_HYDRONIC"
        / "forecast_horizon_3h"
        / "full"
        / "seed42"
        / "comparison"
        / "forecast_horizon_comparison_report.md",
        root
        / "CASE_TEST"
        / "MZ_OFFICE_HYDRONIC"
        / "forecast_horizon_audit"
        / "seed42"
        / "final_audit_report.md",
    ):
        paths.add(report)
    old_delivery = root / "DELIVERY" / "MZ_Hydronic_Final_PPO_MAPPO_20260811"
    if old_delivery.exists():
        paths.update(path for path in old_delivery.rglob("*") if path.is_file())
    missing = sorted(str(path) for path in paths if not path.exists())
    if missing:
        raise RuntimeError("Required immutable continuation input missing: " + "; ".join(missing))
    return sorted(path.resolve() for path in paths)


def build_source_inventory(suite: HydronicContinuationSuite) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in _source_files(suite):
        try:
            display = str(path.relative_to(suite.ppo.project_root)).replace("\\", "/")
        except ValueError:
            display = str(path)
        files.append(
            {
                "path": display,
                "size": int(path.stat().st_size),
                "sha256": core.sha256_file(path),
            }
        )
    return {
        "schema": f"{SCHEMA}-source-inventory",
        "files": files,
        "inventory_sha256": core.sha256_payload(files),
    }


def establish_source_inventory(suite: HydronicContinuationSuite) -> dict[str, Any]:
    assert_path_isolation(suite)
    current = build_source_inventory(suite)
    path = suite.output_root / "source_inventory.json"
    if path.exists():
        baseline = json.loads(path.read_text(encoding="utf-8"))
        if baseline.get("inventory_sha256") != current["inventory_sha256"]:
            raise RuntimeError("An immutable parent/delivery/3 h artifact changed")
        return baseline
    suite.output_root.mkdir(parents=True, exist_ok=True)
    baseline = {**current, "registered_at": core.utc_now()}
    core.atomic_write_json(path, baseline)
    return baseline


def verify_sources_unchanged(suite: HydronicContinuationSuite) -> dict[str, Any]:
    path = suite.output_root / "source_inventory.json"
    if not path.exists():
        raise RuntimeError("Continuation source inventory has not been registered")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    current = build_source_inventory(suite)
    passed = baseline.get("inventory_sha256") == current.get("inventory_sha256")
    result = {
        "schema": f"{SCHEMA}-source-isolation",
        "checked_at": core.utc_now(),
        "passed": passed,
        "registered_sha256": baseline.get("inventory_sha256"),
        "current_sha256": current.get("inventory_sha256"),
    }
    core.atomic_write_json(suite.output_root / "source_isolation_verification.json", result)
    if not passed:
        raise RuntimeError("Immutable source verification failed")
    return result


def _load_mappo_epoch(
    config: v3.HydronicV3Config, epoch: int
) -> dict[str, Any]:
    manager = v3.AtomicTorchCheckpointManager(config.run_dir, config.config_hash)
    if epoch == 350:
        loaded = manager.load_best()
    elif epoch == 500:
        loaded = manager.load_latest()
        if loaded is None:
            raise RuntimeError("MAPPO latest checkpoint is missing")
    else:
        manifest_path = (
            manager.directory / f"epoch_{epoch:04d}_steps_{epoch * config.steps_per_epoch:09d}.manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != config.config_hash:
            raise RuntimeError(f"MAPPO epoch {epoch} manifest config mismatch")
        bundle_path = manager.directory / str(manifest["bundle"]["name"])
        if core.sha256_file(bundle_path) != manifest["bundle"]["sha256"]:
            raise RuntimeError(f"MAPPO epoch {epoch} bundle checksum mismatch")
        payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
        loaded = {
            "pointer": {
                "epoch": epoch,
                "global_step": epoch * config.steps_per_epoch,
                "config_hash": config.config_hash,
                "manifest": manifest_path.name,
                "manifest_sha256": core.sha256_file(manifest_path),
            },
            "manifest": manifest,
            "payload": payload,
            "bundle_path": bundle_path,
        }
    if int(loaded["pointer"]["epoch"]) != epoch:
        raise RuntimeError(f"MAPPO epoch {epoch} pointer mismatch")
    return loaded


def _mappo_static_audit(suite: HydronicContinuationSuite) -> dict[str, Any]:
    config = suite.source_v3.mappo
    training = pd.read_csv(config.run_dir / "training_metrics.csv")
    evaluations = pd.read_csv(config.run_dir / "train_week_eval.csv")
    reward_row = training.loc[pd.to_numeric(training["reward_mean"]).idxmax()]
    rolling_row = training.loc[pd.to_numeric(training["rolling_30"]).idxmax()]
    eval_row = evaluations.loc[pd.to_numeric(evaluations["return"]).idxmax()]
    checkpoints: dict[str, Any] = {}
    payloads: dict[int, Mapping[str, Any]] = {}
    for epoch in AUDIT_EPOCHS:
        loaded = _load_mappo_epoch(config, epoch)
        payload = loaded["payload"]
        payloads[epoch] = payload
        policy = v3._load_mappo_policy(config, payload, device="cpu")
        local_dims = {name: int(len(index)) for name, index in policy.indices.items()}
        first_actor = next(iter(policy.actors.values()))
        first_linear = next(
            layer for layer in first_actor.modules() if isinstance(layer, torch.nn.Linear)
        )
        critic_linear = next(
            layer
            for layer in v3.CentralizedCritic(
                len(config.observation_columns), hidden_sizes=config.policy_net
            ).modules()
            if isinstance(layer, torch.nn.Linear)
        )
        checks = {
            "payload_epoch": int(payload.get("committed_epoch", -1)) == epoch,
            "payload_step": int(payload.get("global_step", -1))
            == epoch * config.steps_per_epoch,
            "config_hash": payload.get("config_hash") == config.config_hash,
            "manifest_hash": core.sha256_file(
                Path(loaded["bundle_path"]).parent / loaded["pointer"]["manifest"]
            )
            == loaded["pointer"]["manifest_sha256"],
            "bundle_hash": core.sha256_file(Path(loaded["bundle_path"]))
            == loaded["manifest"]["bundle"]["sha256"],
            "global_dim": int(critic_linear.in_features)
            == EXPECTED_GLOBAL_OBSERVATION_DIM,
            "local_dim": int(first_linear.in_features)
            == EXPECTED_LOCAL_OBSERVATION_DIM,
            "adapter_dims": all(
                value == EXPECTED_LOCAL_OBSERVATION_DIM
                for value in local_dims.values()
            ),
        }
        checkpoints[str(epoch)] = {
            "epoch": epoch,
            "global_step": int(payload["global_step"]),
            "manifest": loaded["pointer"]["manifest"],
            "manifest_sha256": loaded["pointer"]["manifest_sha256"],
            "bundle": str(loaded["bundle_path"]),
            "bundle_sha256": core.sha256_file(Path(loaded["bundle_path"])),
            "checks": checks,
            "passed": bool(all(checks.values())),
        }
        if not checkpoints[str(epoch)]["passed"]:
            failed = [name for name, ok in checks.items() if not ok]
            raise RuntimeError(f"MAPPO epoch {epoch} identity failed: {failed}")
    pointer = json.loads(
        (config.run_dir / "checkpoints" / "best.json").read_text(encoding="utf-8")
    )
    result = {
        "training_reward_argmax": {
            "epoch": int(reward_row["epoch"]),
            "return": float(reward_row["reward_mean"]),
        },
        "rolling_30_argmax": {
            "epoch": int(rolling_row["epoch"]),
            "return": float(rolling_row["rolling_30"]),
        },
        "deterministic_train_week_argmax": {
            "epoch": int(eval_row["epoch"]),
            "return": float(eval_row["return"]),
        },
        "best_pointer": pointer,
        "checkpoints": checkpoints,
        "selection_rule": "deterministic training-week return at 25-epoch boundaries",
        "epoch_499_is_diagnostic_only": True,
    }
    return {"report": result, "payloads": payloads}


def _plot_mappo_return_audit(
    training: pd.DataFrame,
    evaluations: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(training["epoch"], training["reward_mean"], alpha=0.3, label="epoch stochastic return")
    axes[0].plot(training["epoch"], training["rolling_30"], linewidth=2, label="rolling-30")
    axes[0].axvline(419, color="tab:gray", linestyle=":", label="epoch-return maximum: 419")
    axes[0].axvline(499, color="tab:purple", linestyle="--", label="rolling-30 maximum: 499")
    axes[0].set_ylabel("Training return")
    axes[0].legend()
    axes[1].plot(evaluations["epoch"], evaluations["return"], marker="o", label="deterministic train-week return")
    axes[1].axvline(350, color="tab:red", linestyle="--", label="registered best: 350")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Deterministic return")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("MAPPO model-selection audit: training and selection metrics are different")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _relative_error(actual: float, reference: float) -> float:
    return abs(actual - reference) / max(abs(reference), 1e-12)


def audit_mappo_best(
    suite: HydronicContinuationSuite,
    *,
    online: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Audit MAPPO identities and, online, replay epochs 350/499/500."""
    inventory = establish_source_inventory(suite)
    report_path = suite.audit_dir / "mappo_best_audit.json"
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("source_inventory_sha256") != inventory["inventory_sha256"]:
            raise RuntimeError("Existing MAPPO audit has a different source inventory")
        if not online or existing.get("online_checked"):
            if existing.get("passed"):
                return existing
            raise RuntimeError("Existing MAPPO audit did not pass")
    suite.audit_dir.mkdir(parents=True, exist_ok=True)
    static = _mappo_static_audit(suite)
    source_config = suite.source_v3.mappo
    training = pd.read_csv(source_config.run_dir / "training_metrics.csv")
    evaluations = pd.read_csv(source_config.run_dir / "train_week_eval.csv")
    _plot_mappo_return_audit(
        training,
        evaluations,
        suite.audit_dir / "mappo_training_vs_selection_return.png",
    )
    result: dict[str, Any] = {
        "schema": f"{SCHEMA}-mappo-best-audit",
        "generated_at": core.utc_now(),
        "source_inventory_sha256": inventory["inventory_sha256"],
        "online_checked": bool(online),
        **static["report"],
        "reruns": {},
        "passed": True,
    }
    if online:
        audit_config = dataclasses.replace(
            suite.mappo,
            output_root=suite.audit_dir / "runtime",
            run_mode="full",
            run_tag="seed42",
            algorithm="mappo_audit",
            wandb_mode="disabled",
        )
        core.cleanup_stale_testids(audit_config)
        forecast = v3.V3ForecastProvider(audit_config, include_validation=False)
        forecast.prefetch_all()
        boptest = core.query_boptest_version(audit_config)
        parent_fingerprints = {
            "ppo": core.environment_fingerprint(
                suite.source_wave2.ppo, boptest, forecast.data
            ),
            "mappo": core.environment_fingerprint(
                suite.source_v3.mappo, boptest, forecast.data
            ),
        }
        manifests = {
            "ppo": json.loads(
                (suite.source_wave2.ppo.run_dir / "run_manifest.json").read_text(encoding="utf-8")
            ),
            "mappo": json.loads(
                (suite.source_v3.mappo.run_dir / "run_manifest.json").read_text(encoding="utf-8")
            ),
        }
        fingerprint_checks = {
            name: parent_fingerprints[name] == manifests[name].get("environment_fingerprint")
            for name in parent_fingerprints
        }
        result["boptest"] = boptest
        result["parent_environment_fingerprints"] = parent_fingerprints
        result["environment_fingerprint_checks"] = fingerprint_checks
        if not all(fingerprint_checks.values()):
            raise RuntimeError("Current BOPTEST/forecast differs from a parent environment fingerprint")
        try:
            for epoch in AUDIT_EPOCHS:
                policy = v3._load_mappo_policy(
                    audit_config, static["payloads"][epoch], device="cpu"
                )
                frame, metrics = v3.rollout_policy(
                    audit_config,
                    forecast.data,
                    model=policy,
                    is_validation=False,
                    phase=f"mappo_best_audit_epoch_{epoch:04d}",
                )
                if len(frame) != audit_config.episode_steps:
                    raise RuntimeError(f"MAPPO audit epoch {epoch} produced {len(frame)} rows")
                trajectory = suite.audit_dir / f"mappo_epoch_{epoch:04d}_train_week.csv"
                frame[v3.validation_columns(audit_config)].to_csv(trajectory, index=False)
                c2 = wave2.c2_trajectory_metrics(frame, audit_config)
                result["reruns"][str(epoch)] = {
                    **metrics,
                    "c2": c2,
                    "rows": len(frame),
                    "trajectory": str(trajectory),
                    "trajectory_sha256": core.sha256_file(trajectory),
                }
        finally:
            core.cleanup_stale_testids(audit_config, force=True)
        recorded = {
            int(row["epoch"]): row
            for _, row in evaluations.iterrows()
            if int(row["epoch"]) in {350, 500}
        }
        comparisons: dict[str, Any] = {}
        for epoch in (350, 500):
            rerun = result["reruns"][str(epoch)]
            original = recorded[epoch]
            return_error = abs(float(rerun["return"]) - float(original["return"]))
            kpi_errors = {
                name: _relative_error(float(rerun[name]), float(original[name]))
                for name in ("cost", "zone_hours", "pmv_hours")
            }
            passed = bool(
                return_error <= AUDIT_RETURN_ABS_TOLERANCE
                and all(value <= AUDIT_KPI_RELATIVE_TOLERANCE for value in kpi_errors.values())
            )
            comparisons[str(epoch)] = {
                "return_absolute_error": return_error,
                "kpi_relative_errors": kpi_errors,
                "passed": passed,
            }
        ordering_passed = bool(
            float(result["reruns"]["350"]["return"])
            >= float(result["reruns"]["500"]["return"])
            - AUDIT_RETURN_ABS_TOLERANCE
        )
        result["recorded_comparisons"] = comparisons
        result["350_vs_500_ordering_passed"] = ordering_passed
        result["passed"] = bool(
            all(item["passed"] for item in comparisons.values()) and ordering_passed
        )
    lines = [
        "# MAPPO best checkpoint audit",
        "",
        f"- Stochastic epoch-return maximum: epoch {result['training_reward_argmax']['epoch']}",
        f"- Rolling-30 maximum: epoch {result['rolling_30_argmax']['epoch']}",
        f"- Registered deterministic train-week best: epoch {result['deterministic_train_week_argmax']['epoch']}",
        "- Epoch 499 is diagnostic only because it was not a registered 25-epoch selection boundary.",
        f"- Online replay completed: {result['online_checked']}",
        f"- Audit passed: {result['passed']}",
    ]
    core.atomic_write_json(report_path, result)
    (suite.audit_dir / "mappo_best_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if not result["passed"]:
        raise RuntimeError("MAPPO best audit failed; continuation is blocked")
    return result


_FROZEN_SCIENCE_FIELDS = (
    "case_key",
    "case_name",
    "start_day",
    "simulation_days",
    "control_period",
    "n_zones",
    "zones",
    "max_system_power",
    "observation_columns",
    "forecast_points",
    "observations_config",
    "reward_weights",
    "physical_config",
    "learning_rate",
    "final_lr_fraction",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "policy_net",
    "seed",
    "residual_scale",
    "occupied_base_k",
    "unoccupied_base_k",
    "action_min_k",
    "action_max_k",
    "comfort_threshold",
    "num_envs",
    "actor_learning_rate",
    "critic_learning_rate",
    "value_loss_coef",
    "max_grad_norm",
)


def _science_compatibility(
    child: HydronicContinuationConfig,
    parent: v3.HydronicV3Config,
    *,
    algorithm: str,
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    for field in _FROZEN_SCIENCE_FIELDS:
        child_value = getattr(child, field)
        parent_value = getattr(parent, field)
        if core.canonical_json(child_value) != core.canonical_json(parent_value):
            differences.append(
                {
                    "field": field,
                    "parent": core.json_ready(parent_value),
                    "continuation": core.json_ready(child_value),
                }
            )
    extra_checks = {
        "parent_config_hash": child.parent_config_hash == parent.config_hash,
        "parent_epoch": child.parent_epoch == PARENT_EPOCH,
        "target_epoch": child.max_epochs
        == PARENT_EPOCH
        + (
            SMOKE_EXTENSION_EPOCHS
            if child.run_mode == "smoke"
            else FULL_EXTENSION_EPOCHS
        ),
        "global_observation_dim": len(child.observation_columns)
        == EXPECTED_GLOBAL_OBSERVATION_DIM,
        "forecast_is_1h": all(
            not column.endswith(tuple(f"f{i}" for i in range(5, 13)))
            for column in child.observation_columns
        ),
        "aux_disabled": not any(
            key in core.hydronic_action_payload((298.15, 298.15))
            for key in core.AUXILIARY_HYDRONIC_KEYS
        ),
    }
    if algorithm == "ppo":
        extra_checks["log_std_init"] = math.isclose(child.log_std_init, -1.0)
        extra_checks["floor_lr"] = math.isclose(
            child.fixed_actor_learning_rate, PPO_FLOOR_LR
        )
    else:
        extra_checks.update(
            {
                "local_observation_dim": all(
                    len(columns) == EXPECTED_LOCAL_OBSERVATION_DIM
                    for columns in v3.LOCAL_OBSERVATION_COLUMNS.values()
                ),
                "actor_floor_lr": math.isclose(
                    child.fixed_actor_learning_rate, MAPPO_ACTOR_FLOOR_LR
                ),
                "critic_floor_lr": math.isclose(
                    child.fixed_critic_learning_rate, MAPPO_CRITIC_FLOOR_LR
                ),
            }
        )
    return {
        "algorithm": algorithm,
        "frozen_field_differences": differences,
        "allowed_changes": {
            "max_epochs": [PARENT_EPOCH, child.max_epochs],
            "learning_rate_after_epoch_500": "constant_at_original_floor",
            "output_and_wandb_metadata": True,
        },
        "checks": extra_checks,
        "passed": not differences and all(extra_checks.values()),
    }


def _source_checkpoint_summary(suite: HydronicContinuationSuite) -> dict[str, Any]:
    ppo_manager = core.AtomicCheckpointManager(
        suite.source_wave2.ppo.run_dir, suite.source_wave2.ppo.config_hash
    )
    ppo_latest = ppo_manager.load_latest_metadata()
    ppo_best = ppo_manager.load_best_metadata()
    if ppo_latest is None:
        raise RuntimeError("Wave-2 PPO latest checkpoint is missing")
    mappo_manager = v3.AtomicTorchCheckpointManager(
        suite.source_v3.mappo.run_dir, suite.source_v3.mappo.config_hash
    )
    mappo_latest = mappo_manager.load_latest()
    mappo_best = mappo_manager.load_best()
    if mappo_latest is None:
        raise RuntimeError("v3 MAPPO latest checkpoint is missing")
    checks = {
        "ppo_latest_epoch_500": int(ppo_latest["pointer"]["epoch"]) == 500,
        "ppo_latest_steps": int(ppo_latest["pointer"]["num_timesteps"])
        == PARENT_GLOBAL_STEP,
        "ppo_best_epoch_500": int(ppo_best["pointer"]["epoch"]) == 500,
        "mappo_latest_epoch_500": int(mappo_latest["pointer"]["epoch"]) == 500,
        "mappo_latest_steps": int(mappo_latest["pointer"]["global_step"])
        == PARENT_GLOBAL_STEP,
        "mappo_best_epoch_350": int(mappo_best["pointer"]["epoch"]) == 350,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Parent checkpoint preflight failed: {failed}")
    return {
        "checks": checks,
        "ppo": {
            "latest": ppo_latest["pointer"],
            "best": ppo_best["pointer"],
            "latest_model_sha256": core.sha256_file(Path(ppo_latest["model_path"])),
        },
        "mappo": {
            "latest": mappo_latest["pointer"],
            "best": mappo_best["pointer"],
            "latest_bundle_sha256": core.sha256_file(
                Path(mappo_latest["bundle_path"])
            ),
            "best_bundle_sha256": core.sha256_file(Path(mappo_best["bundle_path"])),
        },
    }


def run_preflight(
    suite: HydronicContinuationSuite,
    *,
    online: bool = True,
) -> dict[str, Any]:
    assert_path_isolation(suite)
    inventory = establish_source_inventory(suite)
    audit = audit_mappo_best(suite, online=online)
    if suite.ppo.run_mode == "full" and not audit.get("online_checked"):
        raise RuntimeError("Full continuation requires the online MAPPO best audit")
    science = {
        "ppo": _science_compatibility(
            suite.ppo, suite.source_wave2.ppo, algorithm="ppo"
        ),
        "mappo": _science_compatibility(
            suite.mappo, suite.source_v3.mappo, algorithm="mappo"
        ),
    }
    if not all(item["passed"] for item in science.values()):
        raise RuntimeError("Continuation scientific configuration differs from a parent")
    parent_checkpoints = _source_checkpoint_summary(suite)
    cleanup = {
        "ppo": core.cleanup_stale_testids(suite.ppo),
        "mappo": core.cleanup_stale_testids(suite.mappo),
        "evaluation": core.cleanup_stale_testids(suite.evaluation_config),
    }
    blocked = [
        row
        for rows in cleanup.values()
        for row in rows
        if row.get("status") in {"live_owner_skipped", "cleanup_failed"}
    ]
    if blocked:
        raise RuntimeError("A live or uncleanable TestID blocks continuation")
    report = {
        "schema": f"{SCHEMA}-preflight",
        "created_at": core.utc_now(),
        "passed": True,
        "run_mode": suite.ppo.run_mode,
        "seed": suite.ppo.seed,
        "parent_epoch": PARENT_EPOCH,
        "target_epoch": suite.ppo.max_epochs,
        "target_global_step": suite.ppo.total_cap_steps,
        "source_inventory_sha256": inventory["inventory_sha256"],
        "mappo_best_audit": {
            "path": str(suite.audit_dir / "mappo_best_audit.json"),
            "passed": audit["passed"],
            "online_checked": audit["online_checked"],
            "registered_best_epoch": audit["deterministic_train_week_argmax"]["epoch"],
        },
        "science_compatibility": science,
        "parent_checkpoints": parent_checkpoints,
        "cleanup": cleanup,
        "runtime_versions": core.runtime_versions(),
        "extension_config_hashes": {
            "ppo": suite.ppo.config_hash,
            "mappo": suite.mappo.config_hash,
        },
    }
    suite.run_dir.mkdir(parents=True, exist_ok=True)
    core.atomic_write_json(suite.run_dir / "preflight.json", report)
    return report


def _median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def evaluate_continuation_convergence(
    training_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    train_eval_rows: Sequence[Mapping[str, Any]],
    *,
    best_epoch: int | None,
    algorithm: str,
    min_epoch: int = 100,
    eval_interval: int = 25,
) -> dict[str, Any]:
    result = dict(
        v3.evaluate_convergence(
            training_rows,
            update_rows,
            train_eval_rows,
            best_epoch=best_epoch,
            min_epoch=min_epoch,
            algorithm=algorithm,
            eval_interval=eval_interval,
        )
    )
    result["schema"] = f"{SCHEMA}-c1-c4"
    result["C2_legacy"] = bool(result.get("C2", False))
    result["C2_legacy_details"] = result.get("C2_details", {})
    if not result.get("eligible"):
        result["C2_wave2"] = False
        result["C2"] = False
        result["converged"] = False
        result["reasons"] = ["not_an_eligible_validation_boundary"]
        return result
    recent = list(train_eval_rows[-5:])
    ratios = [
        core.safe_float(
            row.get("comfort_energy_ratio_violating_steps"), float("nan")
        )
        for row in recent
        if int(core.safe_float(row.get("violating_steps"), 0)) > 0
    ]
    ratios = [value for value in ratios if np.isfinite(value)]
    ratio_median = float(np.median(ratios)) if ratios else None
    latest = recent[-1]
    latest_violations = int(core.safe_float(latest.get("violating_steps"), 0))
    latest_ratio = core.safe_float(
        latest.get("comfort_energy_ratio_violating_steps"), float("nan")
    )
    latest_pass = bool(
        latest_violations == 0
        or (np.isfinite(latest_ratio) and latest_ratio >= wave2.C2_SIGNAL_THRESHOLD)
    )
    median_pass = bool(
        ratio_median is None or ratio_median >= wave2.C2_SIGNAL_THRESHOLD
    )
    formula_pass = bool(
        len(recent) == 5
        and all(
            core.safe_float(row.get("reward_formula_max_abs_error"), 1.0) <= 1e-8
            for row in recent
        )
    )
    deep_pass = bool(
        len(recent) == 5
        and all(
            core.safe_float(row.get("deep_violation_rate"), 1.0)
            < wave2.C2_DEEP_RATE_LIMIT
            for row in recent
        )
    )
    c2_wave2 = bool(
        len(recent) == 5 and latest_pass and median_pass and formula_pass and deep_pass
    )
    result["C2_wave2"] = c2_wave2
    result["C2"] = c2_wave2
    result["C2_details"] = {
        "latest_violating_steps": latest_violations,
        "latest_violating_step_ratio": (
            latest_ratio if np.isfinite(latest_ratio) else None
        ),
        "recent_five_violating_step_ratio_median": ratio_median,
        "recent_five_complete": len(recent) == 5,
        "latest_signal_pass": latest_pass,
        "median_signal_pass": median_pass,
        "reward_formula_pass": formula_pass,
        "deep_violation_rate_pass": deep_pass,
        "legacy_all_step_result": result["C2_legacy"],
    }
    result["reasons"] = [
        name for name in ("C1", "C2", "C3", "C4") if not result.get(name)
    ]
    result["converged"] = not result["reasons"]
    return result


def _set_optimizer_floor(optimizer: torch.optim.Optimizer, value: float) -> None:
    value = float(value)
    optimizer.defaults["lr"] = value
    for group in optimizer.param_groups:
        group["lr"] = value
        group["initial_lr"] = value


def _set_ppo_floor(model: core.Phase15fPPO, value: float = PPO_FLOOR_LR) -> None:
    model.learning_rate = float(value)
    model.lr_schedule = ConstantSchedule(value)
    _set_optimizer_floor(model.policy.optimizer, value)


def _source_inventory_hash(suite: HydronicContinuationSuite) -> str:
    return json.loads(
        (suite.output_root / "source_inventory.json").read_text(encoding="utf-8")
    )["inventory_sha256"]


def _write_run_manifest(
    config: HydronicContinuationConfig,
    identity: Mapping[str, Any],
    **updates: Any,
) -> dict[str, Any]:
    path = config.run_dir / "run_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != config.config_hash:
            raise RuntimeError("Continuation run manifest config hash mismatch")
    else:
        manifest = {
            "schema": f"{SCHEMA}-run",
            "case": "mz_hydro",
            "algorithm": config.algorithm,
            "started_at": core.utc_now(),
            "config_hash": config.config_hash,
            "parent_config_hash": config.parent_config_hash,
            "source_inventory_sha256": _source_inventory_hash_from_config(config),
            "scientific_config": config.scientific_payload,
            "wandb_run_id": identity["wandb_run_id"],
            "status": "starting",
        }
    manifest.update(core.json_ready(updates))
    core.atomic_write_json(path, manifest)
    return manifest


def _source_inventory_hash_from_config(config: HydronicContinuationConfig) -> str:
    path = config.output_root / "source_inventory.json"
    return json.loads(path.read_text(encoding="utf-8"))["inventory_sha256"]


def _finish_run_manifest(
    config: HydronicContinuationConfig,
    identity: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    status: str,
    wall_seconds: float,
    error: str | None,
) -> dict[str, Any]:
    epoch = int(state.get("committed_epoch", 0))
    return _write_run_manifest(
        config,
        identity,
        status=status,
        completed_at=core.utc_now(),
        actual_epochs=epoch,
        extension_epochs_completed=max(0, epoch - PARENT_EPOCH),
        global_step=int(state.get("global_step", epoch * config.steps_per_epoch)),
        converged=bool(state.get("convergence", {}).get("converged", False)),
        training_outcome=(
            "target_700_completed"
            if epoch >= config.max_epochs
            else status
        ),
        wall_seconds=wall_seconds,
        error=error,
        best_epoch=state.get("best_epoch"),
        best_score=state.get("best_score"),
    )


def _checkpoint_complete(
    config: HydronicContinuationConfig, state: Mapping[str, Any]
) -> bool:
    return int(state.get("committed_epoch", 0)) >= config.max_epochs


class ContinuationPPOCallback(wave2.Wave2PPOCallback):
    def __init__(self, *args: Any, **kwargs: Any):
        restored = dict(kwargs.get("restored_state") or {})
        super().__init__(*args, **kwargs)
        if restored.get("source_import") and int(restored.get("committed_epoch", 0)) == PARENT_EPOCH:
            self.resume_count = 0

    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA}-ppo-state",
            "config_hash": self.config.config_hash,
            "parent_config_hash": self.config.parent_config_hash,
            "parent_epoch": PARENT_EPOCH,
            "committed_epoch": self.committed_epoch,
            "global_step": self.committed_epoch * self.config.steps_per_epoch,
            "extension_epochs_completed": max(0, self.committed_epoch - PARENT_EPOCH),
            "training_rows": self.training_rows,
            "sb3_rows": self.sb3_rows,
            "train_eval_rows": self.train_eval_rows,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "resume_count": self.resume_count,
            "convergence": self.convergence,
            "failure_stop": self.failure_stop,
            "wandb_run_id": self.wandb.run_id,
            "wandb_url": self.wandb.url,
            "environment_fingerprint": self.environment_fingerprint,
            "source_inventory_sha256": _source_inventory_hash_from_config(self.config),
            "learning_rate_extension": {
                "mode": "constant_floor",
                "value": PPO_FLOOR_LR,
            },
            "source_import": False,
            "updated_at": core.utc_now(),
        }

    def on_post_update(self, model: Any) -> bool:
        super().on_post_update(model)
        return self.committed_epoch < self.config.max_epochs


def _prepare_ppo_import(
    suite: HydronicContinuationSuite,
    config: HydronicContinuationConfig,
    checkpoint: core.AtomicCheckpointManager,
    model: core.Phase15fPPO,
    parent: Mapping[str, Any],
    identity: Mapping[str, Any],
    environment_fingerprint: str,
) -> dict[str, Any]:
    original_lr = float(model.policy.optimizer.param_groups[0]["lr"])
    if not math.isclose(original_lr, PPO_FLOOR_LR, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(
            f"PPO parent optimizer LR {original_lr} is not the expected 3e-5 floor"
        )
    _set_ppo_floor(model, PPO_FLOOR_LR)
    state = dict(parent["state"])
    state.update(
        {
            "schema": f"{SCHEMA}-ppo-import",
            "config_hash": config.config_hash,
            "parent_config_hash": config.parent_config_hash,
            "parent_epoch": PARENT_EPOCH,
            "committed_epoch": PARENT_EPOCH,
            "global_step": PARENT_GLOBAL_STEP,
            "extension_epochs_completed": 0,
            "environment_fingerprint": environment_fingerprint,
            "source_inventory_sha256": _source_inventory_hash(suite),
            "wandb_run_id": identity["wandb_run_id"],
            "wandb_url": None,
            "resume_count": 0,
            "source_import": True,
            "learning_rate_extension": {
                "mode": "constant_floor",
                "value": PPO_FLOOR_LR,
            },
        }
    )
    core.restore_rng_state(parent["rng"])
    pointer = checkpoint.save(model, state, PARENT_EPOCH)
    checkpoint.mark_best(pointer, core.safe_float(state.get("best_score")))
    return state


class ContinuationPPOExperiment:
    def __init__(self, suite: HydronicContinuationSuite):
        self.suite = suite
        self.config = suite.ppo

    def train(self) -> dict[str, Any]:
        run_preflight(self.suite, online=False)
        verify_sources_unchanged(self.suite)
        config = self.config
        checkpoint = core.AtomicCheckpointManager(config.run_dir, config.config_hash)
        latest = checkpoint.load_latest_metadata()
        if latest and not config.resume:
            raise RuntimeError("Continuation PPO checkpoint exists and RESUME=False")
        if latest and _checkpoint_complete(config, latest["state"]):
            core.generate_training_diagnostics(config, latest["state"])
            return latest["state"]
        identity = core.get_or_create_run_identity(
            config, latest["state"].get("wandb_run_id") if latest else None
        )
        _write_run_manifest(config, identity, status="starting")
        wandb_logger = v3.V3SafeWandb(config, identity["wandb_run_id"])
        wandb_logger.start()
        vector_environment: VecEnv | None = None
        callback: ContinuationPPOCallback | None = None
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
                forecast = v3.V3ForecastProvider(config, include_validation=False)
                forecast.prefetch_all()
                core.atomic_write_json(
                    config.run_dir / "forecast_composition.json",
                    v3.audit_calendar_and_forecast(config, forecast.data),
                )
                boptest = core.query_boptest_version(config)
                fingerprint = core.environment_fingerprint(config, boptest, forecast.data)
                _write_run_manifest(
                    config,
                    identity,
                    status="training",
                    boptest=boptest,
                    environment_fingerprint=fingerprint,
                )
                factories = [
                    partial(v3._subprocess_environment_v3, config, forecast.data, rank)
                    for rank in range(config.num_envs)
                ]
                vector_environment = SubprocVecEnv(factories, start_method="spawn")
                if latest:
                    if latest["state"].get("environment_fingerprint") != fingerprint:
                        raise RuntimeError("PPO continuation environment fingerprint changed")
                    model = core.Phase15fPPO.load(
                        latest["model_path"], env=vector_environment, device="auto"
                    )
                    restored_state = latest["state"]
                    core.restore_rng_state(latest["rng"])
                    _set_ppo_floor(model, PPO_FLOOR_LR)
                else:
                    parent_manager = core.AtomicCheckpointManager(
                        self.suite.source_wave2.ppo.run_dir,
                        self.suite.source_wave2.ppo.config_hash,
                    )
                    parent = parent_manager.load_latest_metadata()
                    if parent is None:
                        raise RuntimeError("PPO parent latest checkpoint disappeared")
                    model = core.Phase15fPPO.load(
                        parent["model_path"], env=vector_environment, device="auto"
                    )
                    restored_state = _prepare_ppo_import(
                        self.suite,
                        config,
                        checkpoint,
                        model,
                        parent,
                        identity,
                        fingerprint,
                    )
                if int(model.num_timesteps) != int(restored_state["global_step"]):
                    raise RuntimeError("PPO imported timestep does not match checkpoint state")
                callback = ContinuationPPOCallback(
                    config,
                    checkpoint,
                    wave2.Wave2TrainWeekEvaluator(config, forecast.data),
                    wandb_logger,
                    environment_fingerprint=fingerprint,
                    restored_state=restored_state,
                )
                model._phase15f_hook = callback.on_post_update
                remaining = config.total_cap_steps - int(model.num_timesteps)
                if remaining > 0:
                    model.learn(
                        total_timesteps=remaining,
                        reset_num_timesteps=False,
                        callback=callback,
                        tb_log_name="mz_hydro_ppo_1h_continue700",
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
            lifecycle = core.combine_lifecycle_logs(config.run_dir)
            committed = checkpoint.load_latest_metadata()
            state = dict(committed["state"]) if committed else {}
            if state:
                core.generate_training_diagnostics(config, state)
            _finish_run_manifest(
                config,
                identity,
                state,
                status=status,
                wall_seconds=time.perf_counter() - start_wall,
                error=error,
            )
            verify_sources_unchanged(self.suite)
            wandb_logger.finish(
                lifecycle,
                {
                    "status": status,
                    "actual_epochs": int(state.get("committed_epoch", 0)),
                    "extension_epochs_completed": max(
                        0, int(state.get("committed_epoch", 0)) - PARENT_EPOCH
                    ),
                    "converged": bool(
                        state.get("convergence", {}).get("converged", False)
                    ),
                },
            )
        final = checkpoint.load_latest_metadata()
        if final is None:
            raise RuntimeError("PPO continuation ended before parent import")
        return final["state"]


def continue_ppo(suite: HydronicContinuationSuite) -> dict[str, Any]:
    return ContinuationPPOExperiment(suite).train()


def _rewrite_mappo_parent_payload(
    suite: HydronicContinuationSuite,
    config: HydronicContinuationConfig,
    payload: Mapping[str, Any],
    identity: Mapping[str, Any],
    environment_fingerprint: str,
) -> dict[str, Any]:
    epoch = int(payload["committed_epoch"])
    result = dict(payload)
    result.update(
        {
            "schema": f"{SCHEMA}-mappo-import",
            "config_hash": config.config_hash,
            "parent_config_hash": config.parent_config_hash,
            "parent_epoch": PARENT_EPOCH,
            "extension_epochs_completed": max(0, epoch - PARENT_EPOCH),
            "environment_fingerprint": environment_fingerprint,
            "source_inventory_sha256": _source_inventory_hash(suite),
            "wandb_run_id": identity["wandb_run_id"],
            "wandb_url": None,
            "resume_count": 0,
            "source_import": True,
            "learning_rate_extension": {
                "mode": "constant_floor",
                "actor": MAPPO_ACTOR_FLOOR_LR,
                "critic": MAPPO_CRITIC_FLOOR_LR,
            },
        }
    )
    return result


def _optimizer_lr_from_state(state: Mapping[str, Any]) -> float:
    groups = list(state.get("param_groups", []))
    if not groups:
        return float("nan")
    return float(groups[0].get("lr", float("nan")))


def _prepare_mappo_import(
    suite: HydronicContinuationSuite,
    config: HydronicContinuationConfig,
    checkpoint: v3.AtomicTorchCheckpointManager,
    identity: Mapping[str, Any],
    environment_fingerprint: str,
) -> dict[str, Any]:
    source_manager = v3.AtomicTorchCheckpointManager(
        suite.source_v3.mappo.run_dir, suite.source_v3.mappo.config_hash
    )
    source_best = source_manager.load_best()
    source_latest = source_manager.load_latest()
    if source_latest is None:
        raise RuntimeError("MAPPO parent latest checkpoint disappeared")
    latest_payload = source_latest["payload"]
    actor_lrs = [
        _optimizer_lr_from_state(state)
        for state in latest_payload["actor_optimizers"].values()
    ]
    critic_lr = _optimizer_lr_from_state(latest_payload["critic_optimizer"])
    if not all(
        math.isclose(value, MAPPO_ACTOR_FLOOR_LR, rel_tol=0.0, abs_tol=1e-10)
        for value in actor_lrs
    ):
        raise RuntimeError(f"MAPPO parent Actor LRs are not at 3e-5: {actor_lrs}")
    if not math.isclose(
        critic_lr, MAPPO_CRITIC_FLOOR_LR, rel_tol=0.0, abs_tol=1e-10
    ):
        raise RuntimeError(f"MAPPO parent Critic LR is not at 5e-5: {critic_lr}")
    best_payload = _rewrite_mappo_parent_payload(
        suite,
        config,
        source_best["payload"],
        identity,
        environment_fingerprint,
    )
    best_pointer = checkpoint.save(
        best_payload,
        epoch=int(best_payload["committed_epoch"]),
        global_step=int(best_payload["global_step"]),
    )
    checkpoint.mark_best(best_pointer, float(source_best["pointer"]["score"]))
    imported_latest = _rewrite_mappo_parent_payload(
        suite,
        config,
        latest_payload,
        identity,
        environment_fingerprint,
    )
    checkpoint.save(
        imported_latest,
        epoch=PARENT_EPOCH,
        global_step=PARENT_GLOBAL_STEP,
    )
    return imported_latest


class ContinuationMAPPOTrainer(v3.MAPPOTrainer):
    def __init__(self, *args: Any, **kwargs: Any):
        restored = dict(kwargs.get("restored") or {})
        super().__init__(*args, **kwargs)
        if restored.get("source_import") and self.committed_epoch == PARENT_EPOCH:
            self.resume_count = 0
        self._install_floor_schedulers(restored)

    def _install_floor_schedulers(self, restored: Mapping[str, Any]) -> None:
        restored_is_extension = bool(
            str(restored.get("schema", "")).startswith(SCHEMA)
            and not restored.get("source_import")
        )
        actor_states = restored.get("actor_schedulers", {})
        new_actor_schedulers: dict[str, Any] = {}
        for name, optimizer in self.actor_optimizers.items():
            _set_optimizer_floor(optimizer, MAPPO_ACTOR_FLOOR_LR)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=ConstantSchedule(1.0)
            )
            if restored_is_extension and name in actor_states:
                scheduler.load_state_dict(actor_states[name])
            _set_optimizer_floor(optimizer, MAPPO_ACTOR_FLOOR_LR)
            new_actor_schedulers[name] = scheduler
        self.actor_schedulers = new_actor_schedulers
        _set_optimizer_floor(self.critic_optimizer, MAPPO_CRITIC_FLOOR_LR)
        critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optimizer, lr_lambda=ConstantSchedule(1.0)
        )
        if restored_is_extension and restored.get("critic_scheduler"):
            critic_scheduler.load_state_dict(restored["critic_scheduler"])
        _set_optimizer_floor(self.critic_optimizer, MAPPO_CRITIC_FLOOR_LR)
        self.critic_scheduler = critic_scheduler

    def _checkpoint_payload(
        self, epoch: int, convergence: Mapping[str, Any]
    ) -> dict[str, Any]:
        payload = dict(super()._checkpoint_payload(epoch, convergence))
        payload.update(
            {
                "schema": f"{SCHEMA}-mappo-state",
                "parent_config_hash": self.config.parent_config_hash,
                "parent_epoch": PARENT_EPOCH,
                "extension_epochs_completed": max(0, epoch - PARENT_EPOCH),
                "source_inventory_sha256": _source_inventory_hash_from_config(
                    self.config
                ),
                "source_import": False,
                "learning_rate_extension": {
                    "mode": "constant_floor",
                    "actor": MAPPO_ACTOR_FLOOR_LR,
                    "critic": MAPPO_CRITIC_FLOOR_LR,
                },
            }
        )
        return payload

    def train(self) -> dict[str, Any]:
        evaluator = wave2.Wave2TrainWeekEvaluator(self.config, self.forecast_data)
        if self.observations is None:
            self.observations = self._reset_all()
        for epoch in range(self.committed_epoch + 1, self.config.max_epochs + 1):
            started = time.perf_counter()
            rollout = self._collect_rollout()
            update_row = self._update(epoch)
            for key in ("reward_mean",):
                if not np.isfinite(core.safe_float(rollout.get(key), float("nan"))):
                    raise RuntimeError(f"MAPPO produced non-finite {key} at epoch {epoch}")
            for key in ("policy_std", "learning_rate", "critic_learning_rate"):
                if not np.isfinite(core.safe_float(update_row.get(key), float("nan"))):
                    raise RuntimeError(f"MAPPO produced non-finite {key} at epoch {epoch}")
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
                    v3.MAPPOPolicyAdapter(self.actors, self.config, self.device),
                    epoch,
                    self.global_step,
                )
                self.train_eval_rows.append(evaluation)
                if core.safe_float(evaluation["return"]) > self.best_score:
                    self.best_score = core.safe_float(evaluation["return"])
                    self.best_epoch = epoch
                    new_best = True
            convergence = evaluate_continuation_convergence(
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
            core.write_csv(
                self.config.run_dir / "training_metrics.csv", self.training_rows
            )
            core.write_csv(
                self.config.run_dir / "sb3_updates.csv", self.update_rows
            )
            core.write_csv(
                self.config.run_dir / "train_week_eval.csv", self.train_eval_rows
            )
            self._log_epoch(training_row, update_row, convergence)
            print(
                f"[HYDRO MAPPO 1h Continue Epoch {epoch:03d}] "
                f"reward={training_row['reward_mean']:.3f} "
                f"rolling30={training_row['rolling_30']:.3f} "
                f"actor_lr={update_row['learning_rate']:.3e} "
                f"critic_lr={update_row['critic_learning_rate']:.3e} "
                f"best_train_week={self.best_score:.3f}"
            )
        if self.writer is not None:
            self.writer.flush()
        latest = self.checkpoint.load_latest()
        if latest is None:
            raise RuntimeError("MAPPO continuation stopped before parent import")
        return latest["payload"]


def _require_ppo_extension_finished(suite: HydronicContinuationSuite) -> None:
    metadata = core.AtomicCheckpointManager(
        suite.ppo.run_dir, suite.ppo.config_hash
    ).load_latest_metadata()
    if metadata is None or not _checkpoint_complete(suite.ppo, metadata["state"]):
        raise RuntimeError("PPO continuation must finish before MAPPO starts")


class ContinuationMAPPOExperiment:
    def __init__(self, suite: HydronicContinuationSuite):
        self.suite = suite
        self.config = suite.mappo

    def train(self) -> dict[str, Any]:
        run_preflight(self.suite, online=False)
        verify_sources_unchanged(self.suite)
        _require_ppo_extension_finished(self.suite)
        config = self.config
        checkpoint = v3.AtomicTorchCheckpointManager(
            config.run_dir, config.config_hash
        )
        latest = checkpoint.load_latest()
        if latest and not config.resume:
            raise RuntimeError("Continuation MAPPO checkpoint exists and RESUME=False")
        if latest and _checkpoint_complete(config, latest["payload"]):
            core.generate_training_diagnostics(config, latest["payload"])
            return latest["payload"]
        identity = core.get_or_create_run_identity(
            config, latest["payload"].get("wandb_run_id") if latest else None
        )
        _write_run_manifest(config, identity, status="starting")
        wandb_logger = v3.V3SafeWandb(config, identity["wandb_run_id"])
        wandb_logger.start()
        environments: list[gym.Env] = []
        trainer: ContinuationMAPPOTrainer | None = None
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
                forecast = v3.V3ForecastProvider(config, include_validation=False)
                forecast.prefetch_all()
                core.atomic_write_json(
                    config.run_dir / "forecast_composition.json",
                    v3.audit_calendar_and_forecast(config, forecast.data),
                )
                boptest = core.query_boptest_version(config)
                fingerprint = core.environment_fingerprint(config, boptest, forecast.data)
                _write_run_manifest(
                    config,
                    identity,
                    status="training",
                    boptest=boptest,
                    environment_fingerprint=fingerprint,
                )
                if latest and latest["payload"].get("environment_fingerprint") != fingerprint:
                    raise RuntimeError("MAPPO continuation environment fingerprint changed")
                if latest is None:
                    imported = _prepare_mappo_import(
                        self.suite,
                        config,
                        checkpoint,
                        identity,
                        fingerprint,
                    )
                    latest = checkpoint.load_latest()
                    if latest is None or int(imported["committed_epoch"]) != PARENT_EPOCH:
                        raise RuntimeError("MAPPO parent import did not commit epoch 500")
                environments = [
                    v3.build_environment_v3(
                        config,
                        forecast.data,
                        rank=rank,
                        phase="training",
                    )
                    for rank in range(config.num_envs)
                ]
                trainer = ContinuationMAPPOTrainer(
                    config,
                    environments,
                    forecast.data,
                    checkpoint,
                    wandb_logger,
                    environment_fingerprint=fingerprint,
                    restored=latest["payload"],
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
            lifecycle = core.combine_lifecycle_logs(config.run_dir)
            committed = checkpoint.load_latest()
            state = dict(committed["payload"]) if committed else {}
            if state:
                core.generate_training_diagnostics(config, state)
            _finish_run_manifest(
                config,
                identity,
                state,
                status=status,
                wall_seconds=time.perf_counter() - start_wall,
                error=error,
            )
            verify_sources_unchanged(self.suite)
            wandb_logger.finish(
                lifecycle,
                {
                    "status": status,
                    "actual_epochs": int(state.get("committed_epoch", 0)),
                    "extension_epochs_completed": max(
                        0, int(state.get("committed_epoch", 0)) - PARENT_EPOCH
                    ),
                    "converged": bool(
                        state.get("convergence", {}).get("converged", False)
                    ),
                },
            )
        final = checkpoint.load_latest()
        if final is None:
            raise RuntimeError("MAPPO continuation ended before parent import")
        return final["payload"]


def continue_mappo(suite: HydronicContinuationSuite) -> dict[str, Any]:
    return ContinuationMAPPOExperiment(suite).train()


def _require_both_finished(
    suite: HydronicContinuationSuite,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ppo = core.AtomicCheckpointManager(
        suite.ppo.run_dir, suite.ppo.config_hash
    ).load_latest_metadata()
    mappo = v3.AtomicTorchCheckpointManager(
        suite.mappo.run_dir, suite.mappo.config_hash
    ).load_latest()
    if ppo is None or not _checkpoint_complete(suite.ppo, ppo["state"]):
        raise RuntimeError("PPO continuation has not reached its target epoch")
    if mappo is None or not _checkpoint_complete(suite.mappo, mappo["payload"]):
        raise RuntimeError("MAPPO continuation has not reached its target epoch")
    return ppo["state"], mappo["payload"]


def _save_evaluation_trajectory(
    config: HydronicContinuationConfig,
    directory: Path,
    label: str,
    frame: pd.DataFrame,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    trajectory = directory / f"{label}_validation_hydronic_2zone.csv"
    actions_path = directory / f"{label}_actions.csv"
    export = frame[wave2.validation_columns(config)]
    export.to_csv(trajectory, index=False)
    actions = np.asarray(
        [json.loads(value) for value in frame["_actions"]], dtype=float
    )
    pd.DataFrame(
        {
            "step": np.arange(len(frame), dtype=int),
            "time": frame["time"].to_numpy(dtype=float),
            **{
                f"action_{zone}": actions[:, index]
                for index, zone in enumerate(config.zones)
            },
        }
    ).to_csv(actions_path, index=False)
    if len(export) != config.episode_steps:
        raise RuntimeError(f"{label} validation has {len(export)} rows, expected 480")
    times = export["time"].to_numpy(dtype=float)
    if len(times) != 480 or not np.allclose(np.diff(times), config.control_period):
        raise RuntimeError(f"{label} validation time grid is incomplete")
    return trajectory, actions_path


def _load_evaluation_trajectory(
    config: HydronicContinuationConfig,
    trajectory: Path,
    actions_path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(trajectory)
    actions = pd.read_csv(actions_path)
    frame["_actions"] = [
        json.dumps(row)
        for row in actions[
            [f"action_{zone}" for zone in config.zones]
        ].to_numpy(dtype=float).tolist()
    ]
    return frame


def _evaluation_metrics(
    frame: pd.DataFrame,
    metrics: Mapping[str, Any],
    config: HydronicContinuationConfig,
) -> dict[str, Any]:
    actions = [json.loads(value) for value in frame["_actions"]]
    c2 = wave2.c2_trajectory_metrics(frame, config)
    health = v3.action_health_by_zone(frame, actions, config)
    deep = 0
    occupied = 0
    for zone in config.zones:
        mask = frame[f"occ_{zone}"].to_numpy(dtype=float) > 0
        pmv = np.abs(frame[f"pmv_{zone}"].to_numpy(dtype=float))
        deep += int(np.sum(mask & (pmv > 0.6)))
        occupied += int(np.sum(mask))
    return {
        **dict(metrics),
        "energy_kwh": float(frame["energy_step_kwh"].sum()),
        "deep_pmv_violation_rate": float(deep / max(occupied, 1)),
        "action_health_by_zone": health,
        "c2_legacy": bool(
            core.safe_float(metrics.get("pmv_hours")) <= 0
            or c2["comfort_energy_ratio_all_steps"] >= 0.05
        ),
        "c2_wave2": bool(
            c2["signal_pass"] and c2["formula_pass"] and c2["deep_rate_pass"]
        ),
        "c2_details": c2,
    }


def _plot_continuation_curve(
    config: HydronicContinuationConfig,
    best_epoch: int,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    training = pd.read_csv(config.run_dir / "training_metrics.csv")
    evaluations = pd.read_csv(config.run_dir / "train_week_eval.csv")
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(training["epoch"], training["reward_mean"], alpha=0.25, label="epoch stochastic return")
    axes[0].plot(training["epoch"], training["rolling_30"], linewidth=2, label="rolling-30")
    axes[1].plot(evaluations["epoch"], evaluations["return"], marker="o", label="deterministic train-week return")
    for axis in axes:
        axis.axvline(PARENT_EPOCH, color="black", linestyle="--", label="continuation starts")
        axis.axvline(best_epoch, color="tab:red", linestyle=":", label=f"selected best: {best_epoch}")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Training return")
    axes[1].set_ylabel("Selection return")
    axes[1].set_xlabel("Epoch")
    figure.suptitle(f"Hydronic 1 h {config.algorithm.upper()} epochs 1-{config.max_epochs}")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_validation_overview(
    config: HydronicContinuationConfig,
    frames: Mapping[str, pd.DataFrame],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "rbc": "#6b7280",
        "ppo_parent": "#93c5fd",
        "ppo_final": "#2563eb",
        "mappo_parent": "#fdba74",
        "mappo_final": "#ea580c",
    }
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    for label, frame in frames.items():
        hours = np.arange(len(frame)) * config.control_period / 3600.0
        axes[0, 0].plot(hours, frame["cost"].cumsum(), label=label, color=colors[label])
        axes[0, 1].plot(hours, frame["energy_step_kwh"].cumsum(), label=label, color=colors[label])
    for label in ("ppo_final", "mappo_final"):
        frame = frames[label]
        hours = np.arange(len(frame)) * config.control_period / 3600.0
        for zone in config.zones:
            axes[1, 0].plot(hours, frame[f"pmv_{zone}"], label=f"{label} {zone}")
    for label in ("ppo_final", "mappo_final"):
        frame = frames[label]
        hours = np.arange(len(frame)) * config.control_period / 3600.0
        axes[1, 1].plot(
            hours,
            frame["power_total"].rolling(4, min_periods=1).mean() / 1000.0,
            label=label,
            color=colors[label],
        )
    axes[0, 0].set_ylabel("Cumulative cost")
    axes[0, 1].set_ylabel("Cumulative energy [kWh]")
    axes[1, 0].axhspan(-0.5, 0.5, color="green", alpha=0.1)
    axes[1, 0].set_ylabel("PMV")
    axes[1, 1].set_ylabel("Rolling 1 h power [kW]")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[1, 0].set_xlabel("Validation time [h]")
    axes[1, 1].set_xlabel("Validation time [h]")
    figure.suptitle("Hydronic 1 h parent vs continued best - day 220-224")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_final_actions(
    config: HydronicContinuationConfig,
    frames: Mapping[str, pd.DataFrame],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    for column, label in enumerate(("ppo_final", "mappo_final")):
        frame = frames[label]
        actions = np.asarray(
            [json.loads(value) for value in frame["_actions"]], dtype=float
        )
        hours = np.arange(len(frame)) * config.control_period / 3600.0
        for index, zone in enumerate(config.zones):
            axes[0, column].plot(hours, actions[:, index], label=f"action {zone}")
            axes[1, column].plot(hours, frame[f"setpoint_{zone}"], label=f"setpoint {zone}")
        occupied = np.maximum.reduce(
            [frame[f"occ_{zone}"].to_numpy(dtype=float) for zone in config.zones]
        ) > 0
        for axis in axes[:, column]:
            axis.fill_between(hours, axis.get_ylim()[0], axis.get_ylim()[1], where=occupied, color="gray", alpha=0.08)
            axis.grid(alpha=0.25)
            axis.legend()
        axes[0, column].set_title(label)
        axes[0, column].set_ylabel("Normalized action")
        axes[1, column].set_ylabel("Setpoint [K]")
        axes[1, column].set_xlabel("Validation time [h]")
    figure.suptitle("Hydronic 1 h final best actions and physical setpoints")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_kpi_summary(metrics: Mapping[str, Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(metrics)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    specifications = (
        ("cost", "Electricity cost"),
        ("energy_kwh", "Electricity use [kWh]"),
        ("zone_hours", "Occupied zone-h"),
        ("pmv_hours", "PMV*h"),
    )
    for axis, (key, title) in zip(axes.flat, specifications):
        values = [core.safe_float(metrics[label].get(key)) for label in labels]
        bars = axis.bar(labels, values)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("Hydronic 1 h whole-week KPI summary")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _copy_checkpoint_bundle(
    loaded: Mapping[str, Any], destination: Path, *, algorithm: str
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    if algorithm == "ppo":
        manifest = loaded["manifest"]
        pointer = loaded["pointer"]
        source_directory = Path(loaded["model_path"]).parent
        for record in manifest["files"].values():
            shutil.copy2(source_directory / record["name"], destination / record["name"])
    else:
        manifest = loaded["manifest"]
        pointer = loaded["pointer"]
        source_directory = Path(loaded["bundle_path"]).parent
        shutil.copy2(
            source_directory / manifest["bundle"]["name"],
            destination / manifest["bundle"]["name"],
        )
    shutil.copy2(
        source_directory / pointer["manifest"], destination / pointer["manifest"]
    )
    core.atomic_write_json(destination / "best.json", pointer)
    return {
        "epoch": int(pointer["epoch"]),
        "global_step": int(pointer.get("num_timesteps", pointer.get("global_step"))),
        "score": float(pointer["score"]),
        "config_hash": pointer["config_hash"],
        "manifest": pointer["manifest"],
        "origin": (
            "immutable_parent"
            if int(pointer["epoch"]) <= PARENT_EPOCH
            else "continuation"
        ),
    }


def _delivery_loader_source() -> str:
    return '''"""Offline integrity check for the delivered continuation models."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))
from CASE_TEST import rl_retraining_v2 as core
from CASE_TEST import rl_retraining_v3 as v3
from CASE_TEST import rl_hydronic_1h_continue_700 as ext
def read(path): return json.loads(path.read_text(encoding="utf-8"))
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()
def main():
    suite=ext.build_hydronic_1h_continuation_suite(ROOT / "project", run_mode="full", wandb_mode="disabled")
    pdir=ROOT / "models" / "ppo_1h"
    pp=read(pdir / "best.json"); pm=read(pdir / pp["manifest"])
    assert sha(pdir / pp["manifest"]) == pp["manifest_sha256"]
    for item in pm["files"].values(): assert sha(pdir/item["name"]) == item["sha256"]
    model=core.Phase15fPPO.load(pdir / pm["files"]["model"]["name"], device="cpu")
    assert model.num_timesteps == pp["num_timesteps"]
    mdir=ROOT / "models" / "mappo_1h"
    mp=read(mdir / "best.json"); mm=read(mdir / mp["manifest"])
    assert sha(mdir / mp["manifest"]) == mp["manifest_sha256"]
    bundle=mdir / mm["bundle"]["name"]; assert sha(bundle) == mm["bundle"]["sha256"]
    payload=torch.load(bundle, map_location="cpu", weights_only=False)
    v3._load_mappo_policy(suite.mappo, payload, device="cpu")
    print(f"PPO OK epoch={pp['epoch']}; MAPPO OK epoch={mp['epoch']}")
if __name__ == "__main__": main()
'''


def _delivery_verify_source() -> str:
    return '''$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
foreach ($Line in Get-Content -LiteralPath (Join-Path $Root "CHECKSUMS.sha256")) {
    if ([string]::IsNullOrWhiteSpace($Line)) { continue }
    $Parts = $Line -split "  ", 2
    $Actual = (Get-FileHash -LiteralPath (Join-Path $Root $Parts[1]) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Parts[0].Trim().ToLowerInvariant()) { throw "Hash mismatch: $($Parts[1])" }
}
Write-Host "Package checksum verification passed."
'''


def _write_delivery_checksums(root: Path) -> None:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name != "CHECKSUMS.sha256"),
        key=lambda path: str(path.relative_to(root)),
    )
    lines = [
        f"{core.sha256_file(path)}  {str(path.relative_to(root)).replace('/', os.sep)}"
        for path in files
    ]
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_delivery_package(
    suite: HydronicContinuationSuite,
    report: Mapping[str, Any],
) -> Path:
    if suite.ppo.run_mode != "full":
        raise RuntimeError("A delivery package is created only for the full continuation")
    verify_sources_unchanged(suite)
    date_stamp = datetime.now().strftime("%Y%m%d")
    destination = suite.ppo.project_root / "DELIVERY" / f"{DELIVERY_PREFIX}_{date_stamp}"
    signature = {
        "schema": f"{SCHEMA}-delivery-signature",
        "evaluation_signature_hash": report["signature_hash"],
        "ppo_best_manifest_sha256": report["signature"]["ppo_best_manifest_sha256"],
        "mappo_best_manifest_sha256": report["signature"]["mappo_best_manifest_sha256"],
    }
    signature_path = destination / "delivery_signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise RuntimeError(f"Delivery path already contains a different result: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    ppo_manager = core.AtomicCheckpointManager(suite.ppo.run_dir, suite.ppo.config_hash)
    mappo_manager = v3.AtomicTorchCheckpointManager(suite.mappo.run_dir, suite.mappo.config_hash)
    ppo_best = ppo_manager.load_best_metadata()
    mappo_best = mappo_manager.load_best()
    ppo_record = _copy_checkpoint_bundle(
        ppo_best, destination / "models" / "ppo_1h", algorithm="ppo"
    )
    mappo_record = _copy_checkpoint_bundle(
        mappo_best, destination / "models" / "mappo_1h", algorithm="mappo"
    )
    for config, target in (
        (suite.ppo, destination / "models" / "ppo_1h"),
        (suite.mappo, destination / "models" / "mappo_1h"),
    ):
        shutil.copy2(config.run_dir / "run_manifest.json", target / "run_manifest.json")
    project_case_test = destination / "project" / "CASE_TEST"
    project_hydro = project_case_test / "MZ_OFFICE_HYDRONIC"
    project_hydro.mkdir(parents=True, exist_ok=True)
    for module in (
        Path(core.__file__),
        Path(v3.__file__),
        Path(wave2.__file__),
        Path(__file__),
    ):
        shutil.copy2(module, project_case_test / module.name)
    (project_case_test / "__init__.py").write_text("", encoding="utf-8")
    for notebook in (
        suite.ppo.case_dir / "PPO_Phase1_5f_Wave2.ipynb",
        suite.ppo.case_dir / "PPO_MAPPO_Phase1_5f_Retrain.ipynb",
        suite.ppo.case_dir / "PPO_MAPPO_1h_Continue_700.ipynb",
    ):
        shutil.copy2(notebook, project_hydro / notebook.name)
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for source in (
        suite.ppo.run_dir / "diagnostics",
        suite.mappo.run_dir / "diagnostics",
        suite.run_dir / "comparison" / "figures",
        suite.audit_dir,
    ):
        if not source.exists():
            continue
        prefix = source.parent.name + "_" + source.name
        for image in source.rglob("*.png"):
            relative = image.relative_to(source)
            target = figures / prefix / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
    reports = destination / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for source in (
        suite.audit_dir / "mappo_best_audit.json",
        suite.audit_dir / "mappo_best_audit.md",
        suite.run_dir / "comparison" / "extension_evaluation_report.json",
        suite.run_dir / "comparison" / "extension_evaluation_report.md",
        suite.run_dir / "comparison" / "whole_week_kpis.csv",
    ):
        if source.exists():
            shutil.copy2(source, reports / source.name)
    requirements_source = (
        suite.ppo.project_root
        / "DELIVERY"
        / "MZ_Hydronic_Final_PPO_MAPPO_20260811"
        / "requirements.txt"
    )
    if requirements_source.exists():
        shutil.copy2(requirements_source, destination / "requirements.txt")
    configs = {
        "schema": f"{SCHEMA}-delivery",
        "created_on": date_stamp,
        "case": "MZ_OFFICE_HYDRONIC",
        "forecast_horizon": "1 h",
        "seed": 42,
        "selection": "deterministic training-week return at 25-epoch boundaries",
        "parent_epoch": PARENT_EPOCH,
        "target_epoch": 700,
        "models": {"ppo": ppo_record, "mappo": mappo_record},
        "parent_sources": _source_checkpoint_summary(suite),
        "evaluation_signature_hash": report["signature_hash"],
    }
    core.atomic_write_json(destination / "MODEL_CONFIGS.json", configs)
    core.atomic_write_json(signature_path, signature)
    tools = destination / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    (tools / "load_models.py").write_text(_delivery_loader_source(), encoding="utf-8")
    (tools / "verify_package.ps1").write_text(_delivery_verify_source(), encoding="utf-8")
    readme = f"""# MZ Hydronic final 1 h PPO/MAPPO delivery\n\nThis versioned package was generated after continuing both latest epoch-500 training states to epoch 700. The delivered checkpoint for each algorithm is still selected only by deterministic training-week return; it is not forced to be epoch 700.\n\n- PPO selected epoch: {ppo_record['epoch']} ({ppo_record['origin']})\n- MAPPO selected epoch: {mappo_record['epoch']} ({mappo_record['origin']})\n- Final training global step: 1,344,000 per algorithm\n- Learning rates for epochs 501-700: PPO/Actor 3e-5, Critic 5e-5\n- Forecast horizon: 1 h only; no 3 h result is included.\n\nRun `python tools/load_models.py` for an offline model load test and `powershell -ExecutionPolicy Bypass -File tools/verify_package.ps1` for checksum verification.\n"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    _write_delivery_checksums(destination)
    for line in (destination / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if core.sha256_file(destination / relative) != expected:
            raise RuntimeError(f"Delivery checksum verification failed: {relative}")
    verify_sources_unchanged(suite)
    return destination


def evaluate_extension(
    suite: HydronicContinuationSuite,
    *,
    force: bool | None = None,
) -> dict[str, Any]:
    run_preflight(suite, online=False)
    ppo_state, mappo_state = _require_both_finished(suite)
    force = suite.force_final_eval if force is None else bool(force)
    if suite.ppo.run_mode != "full":
        report = {
            "schema": f"{SCHEMA}-smoke-complete",
            "generated_at": core.utc_now(),
            "ppo_epoch": int(ppo_state["committed_epoch"]),
            "mappo_epoch": int(mappo_state["committed_epoch"]),
            "heldout_not_run": True,
            "delivery_not_created": True,
        }
        core.atomic_write_json(suite.run_dir / "smoke_completion.json", report)
        return report
    comparison_dir = suite.run_dir / "comparison"
    figures_dir = comparison_dir / "figures"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    ppo_manager = core.AtomicCheckpointManager(suite.ppo.run_dir, suite.ppo.config_hash)
    mappo_manager = v3.AtomicTorchCheckpointManager(suite.mappo.run_dir, suite.mappo.config_hash)
    ppo_best = ppo_manager.load_best_metadata()
    mappo_best = mappo_manager.load_best()
    signature = {
        "schema": f"{SCHEMA}-evaluation-signature",
        "source_inventory_sha256": _source_inventory_hash(suite),
        "ppo_best_manifest_sha256": core.sha256_file(
            ppo_manager.directory / ppo_best["pointer"]["manifest"]
        ),
        "mappo_best_manifest_sha256": core.sha256_file(
            mappo_manager.directory / mappo_best["pointer"]["manifest"]
        ),
        "validation_start_day": 220,
        "validation_steps": 480,
    }
    signature_hash = core.sha256_payload(signature)
    report_path = comparison_dir / "extension_evaluation_report.json"
    progress_path = comparison_dir / "evaluation_manifest.json"
    if report_path.exists() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("signature_hash") != signature_hash:
            raise RuntimeError("Existing extension evaluation uses different best checkpoints")
        build_delivery_package(suite, report)
        return report
    progress = {
        "schema": f"{SCHEMA}-evaluation-progress",
        "signature": signature,
        "signature_hash": signature_hash,
        "started_at": core.utc_now(),
        "status": "running",
        "methods": {},
    }
    if progress_path.exists() and not force:
        prior = json.loads(progress_path.read_text(encoding="utf-8"))
        if prior.get("signature_hash") != signature_hash:
            raise RuntimeError("Incomplete extension evaluation signature changed")
        progress = prior
    else:
        core.atomic_write_json(progress_path, progress)
    cleanup = {
        name: core.cleanup_stale_testids(config)
        for name, config in (
            ("ppo", suite.ppo),
            ("mappo", suite.mappo),
            ("evaluation", suite.evaluation_config),
        )
    }
    if any(
        row.get("status") in {"live_owner_skipped", "cleanup_failed"}
        for rows in cleanup.values()
        for row in rows
    ):
        raise RuntimeError("A live or uncleanable TestID blocks final evaluation")
    source_ppo_manager = core.AtomicCheckpointManager(
        suite.source_wave2.ppo.run_dir, suite.source_wave2.ppo.config_hash
    )
    source_mappo_manager = v3.AtomicTorchCheckpointManager(
        suite.source_v3.mappo.run_dir, suite.source_v3.mappo.config_hash
    )
    source_ppo_best = source_ppo_manager.load_best_metadata()
    source_mappo_best = source_mappo_manager.load_best()
    config = suite.evaluation_config
    forecast = v3.V3ForecastProvider(config, include_validation=True)
    forecast.prefetch_all()
    factories: dict[str, Callable[[], tuple[Any | None, bool]]] = {
        "rbc": lambda: (None, True),
        "ppo_parent": lambda: (
            core.Phase15fPPO.load(source_ppo_best["model_path"], device="auto"),
            False,
        ),
        "ppo_final": lambda: (
            core.Phase15fPPO.load(ppo_best["model_path"], device="auto"),
            False,
        ),
        "mappo_parent": lambda: (
            v3._load_mappo_policy(config, source_mappo_best["payload"]),
            False,
        ),
        "mappo_final": lambda: (
            v3._load_mappo_policy(config, mappo_best["payload"]),
            False,
        ),
    }
    frames: dict[str, pd.DataFrame] = {}
    metrics: dict[str, Any] = {}
    for label in ("rbc", "ppo_parent", "ppo_final", "mappo_parent", "mappo_final"):
        existing = progress["methods"].get(label, {})
        if existing.get("status") == "complete" and not force:
            frame = _load_evaluation_trajectory(
                config, Path(existing["trajectory"]), Path(existing["actions"])
            )
            frames[label] = frame
            metrics[label] = existing["metrics"]
            continue
        progress["methods"][label] = {"status": "running", "started_at": core.utc_now()}
        core.atomic_write_json(progress_path, progress)
        try:
            model, zero_policy = factories[label]()
            frame, base_metrics = wave2.rollout_policy(
                config,
                forecast.data,
                model=model,
                is_validation=True,
                phase=f"extension_heldout_{label}",
                zero_policy=zero_policy,
            )
            final_metrics = _evaluation_metrics(frame, base_metrics, config)
            trajectory, actions = _save_evaluation_trajectory(
                config, comparison_dir / "day220", label, frame
            )
            frames[label] = frame
            metrics[label] = final_metrics
            progress["methods"][label] = {
                "status": "complete",
                "completed_at": core.utc_now(),
                "trajectory": str(trajectory),
                "actions": str(actions),
                "trajectory_sha256": core.sha256_file(trajectory),
                "actions_sha256": core.sha256_file(actions),
                "rows": len(frame),
                "metrics": final_metrics,
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
    core.write_csv(
        comparison_dir / "whole_week_kpis.csv",
        [{"controller": label, **values} for label, values in metrics.items()],
    )
    _plot_continuation_curve(
        suite.ppo,
        int(ppo_best["pointer"]["epoch"]),
        figures_dir / "ppo_epochs_1_700.png",
    )
    _plot_continuation_curve(
        suite.mappo,
        int(mappo_best["pointer"]["epoch"]),
        figures_dir / "mappo_epochs_1_700.png",
    )
    _plot_validation_overview(
        config, frames, figures_dir / "validation_week_overview.png"
    )
    _plot_final_actions(config, frames, figures_dir / "final_actions_setpoints.png")
    _plot_kpi_summary(metrics, figures_dir / "whole_week_kpi_summary.png")
    isolation = verify_sources_unchanged(suite)
    report = {
        "schema": f"{SCHEMA}-evaluation",
        "generated_at": core.utc_now(),
        "signature": signature,
        "signature_hash": signature_hash,
        "checkpoint_selection": "deterministic training-week return at 25-epoch boundaries only",
        "heldout_used_for_selection": False,
        "training_target": {
            "parent_epoch": PARENT_EPOCH,
            "target_epoch": 700,
            "target_global_step": 1_344_000,
        },
        "best": {
            "ppo": {
                "epoch": int(ppo_best["pointer"]["epoch"]),
                "global_step": int(ppo_best["pointer"]["num_timesteps"]),
                "score": float(ppo_best["pointer"]["score"]),
                "origin": (
                    "immutable_parent"
                    if int(ppo_best["pointer"]["epoch"]) <= PARENT_EPOCH
                    else "continuation"
                ),
            },
            "mappo": {
                "epoch": int(mappo_best["pointer"]["epoch"]),
                "global_step": int(mappo_best["pointer"]["global_step"]),
                "score": float(mappo_best["pointer"]["score"]),
                "origin": (
                    "immutable_parent"
                    if int(mappo_best["pointer"]["epoch"]) <= PARENT_EPOCH
                    else "continuation"
                ),
            },
        },
        "convergence": {
            "ppo": ppo_state.get("convergence", {}),
            "mappo": mappo_state.get("convergence", {}),
        },
        "metrics": metrics,
        "mappo_best_audit": json.loads(
            (suite.audit_dir / "mappo_best_audit.json").read_text(encoding="utf-8")
        ),
        "source_isolation": isolation,
        "cleanup": cleanup,
        "three_hour_results_included": False,
    }
    core.atomic_write_json(report_path, report)
    lines = [
        "# Hydronic 1 h continuation evaluation",
        "",
        "Both latest epoch-500 optimizer states were continued through epoch 700.",
        "Best checkpoints were selected only by deterministic training-week return.",
        f"PPO final selected epoch: {report['best']['ppo']['epoch']}",
        f"MAPPO final selected epoch: {report['best']['mappo']['epoch']}",
        f"PPO C1/C2_wave2/C3/C4: {[report['convergence']['ppo'].get(k) for k in ('C1','C2_wave2','C3','C4')]}",
        f"MAPPO C1/C2_wave2/C3/C4: {[report['convergence']['mappo'].get(k) for k in ('C1','C2_wave2','C3','C4')]}",
        "The day-220 held-out week was not used for checkpoint selection.",
        "No 3 h model or result is included.",
    ]
    (comparison_dir / "extension_evaluation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    progress["status"] = "complete"
    progress["completed_at"] = core.utc_now()
    progress["report"] = str(report_path)
    core.atomic_write_json(progress_path, progress)
    delivery = build_delivery_package(suite, report)
    report["delivery_package"] = str(delivery)
    core.atomic_write_json(report_path, report)
    return report


__all__ = [
    "FULL_EXTENSION_EPOCHS",
    "HydronicContinuationConfig",
    "HydronicContinuationSuite",
    "audit_mappo_best",
    "build_delivery_package",
    "build_hydronic_1h_continuation_suite",
    "continue_mappo",
    "continue_ppo",
    "evaluate_continuation_convergence",
    "evaluate_extension",
    "run_preflight",
    "verify_sources_unchanged",
]
