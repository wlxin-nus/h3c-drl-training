"""Phase 1.5f Wave-2: isolated PPO retraining for MZ Hydronic.

This module deliberately does not mutate the Phase 1.5f v3 implementation or
its result tree.  It reuses the frozen v3 environment semantics, changes only
the PPO initial log standard deviation, and pre-registers a violation-
conditioned C2 diagnostic that is not diluted by unoccupied/deadband steps.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
import traceback
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


WAVE2_SCHEMA = "phase1.5f-wave2"
APPROVED_SEEDS = (42, 1337, 2026)
LOG_STD_INIT = -1.0
C2_SIGNAL_THRESHOLD = 0.05
C2_DEEP_PMV_LIMIT = 0.60
C2_DEEP_RATE_LIMIT = 0.01
FAIL_FAST_EPOCH = 150
FAIL_FAST_STD_LIMIT = 0.50
FAIL_FAST_SATURATION_LIMIT = 0.50
PRIMARY_VALIDATION_DAY = 220
CONFIRMATION_DAY_CANDIDATES = tuple(range(227, 360, 7))


@dataclasses.dataclass
class HydronicWave2Config(v3.HydronicV3Config):
    """Hydronic v3 scientific bundle plus the one approved PPO change."""

    log_std_init: float = LOG_STD_INIT
    fail_fast_epoch: int = FAIL_FAST_EPOCH
    c2_signal_threshold: float = C2_SIGNAL_THRESHOLD
    c2_deep_pmv_limit: float = C2_DEEP_PMV_LIMIT
    c2_deep_rate_limit: float = C2_DEEP_RATE_LIMIT

    @property
    def scientific_payload(self) -> dict[str, Any]:
        payload = dict(super().scientific_payload)
        payload.update(
            {
                "schema": WAVE2_SCHEMA,
                "implementation_sha256": core.sha256_file(Path(__file__).resolve()),
                "parent_v3_implementation_sha256": core.sha256_file(
                    Path(v3.__file__).resolve()
                ),
                "algorithm": self.algorithm,
                "log_std_init": self.log_std_init,
                "c2_acceptance": {
                    "legacy_reported": True,
                    "conditioning": "steps_with_any_occupied_abs_pmv_gt_0.5",
                    "signal_ratio_minimum": self.c2_signal_threshold,
                    "recent_evaluations": 5,
                    "deep_pmv_limit": self.c2_deep_pmv_limit,
                    "deep_violation_rate_limit": self.c2_deep_rate_limit,
                },
                "failure_stop": {
                    "epoch": self.fail_fast_epoch,
                    "policy_std_limit": FAIL_FAST_STD_LIMIT,
                    "two_consecutive_occupied_saturation_limit": (
                        FAIL_FAST_SATURATION_LIMIT
                    ),
                },
            }
        )
        return payload


@dataclasses.dataclass(frozen=True)
class HydronicWave2Suite:
    ppo: HydronicWave2Config
    force_final_eval: bool = False

    @property
    def run_dir(self) -> Path:
        return self.ppo.suite_dir

    @property
    def final(self) -> HydronicWave2Config:
        return dataclasses.replace(self.ppo, algorithm="final")

    @property
    def confirmation(self) -> HydronicWave2Config:
        return dataclasses.replace(self.ppo, algorithm="confirmation")

    @property
    def v3_source_suite(self) -> v3.HydronicV3Suite:
        return v3.build_hydronic_v3_suite(
            self.ppo.project_root,
            run_mode="full",
            resume=True,
            wandb_mode="disabled",
            seed=42,
        )


def _dataclass_values(instance: Any, target: type[Any]) -> dict[str, Any]:
    return {
        field.name: getattr(instance, field.name)
        for field in dataclasses.fields(target)
        if hasattr(instance, field.name)
    }


def build_hydronic_wave2_suite(
    project_root: str | Path,
    *,
    run_mode: str = "smoke",
    resume: bool = True,
    wandb_mode: str = "online",
    seed: int = 42,
    force_final_eval: bool = False,
) -> HydronicWave2Suite:
    if seed not in APPROVED_SEEDS:
        raise ValueError(f"Wave-2 seed must be one of {APPROVED_SEEDS}")
    if run_mode not in {"smoke", "full"}:
        raise ValueError("RUN_MODE must be 'smoke' or 'full'")
    root = Path(project_root).resolve()
    frozen = v3.build_hydronic_v3_suite(
        root,
        run_mode=run_mode,
        resume=resume,
        wandb_mode=wandb_mode,
        seed=42,
    ).ppo
    values = _dataclass_values(frozen, v3.HydronicV3Config)
    values.update(
        {
            "output_root": (
                root
                / "CASE_TEST"
                / "MZ_OFFICE_HYDRONIC"
                / "phase1_5f_wave2"
            ),
            "algorithm": "ppo",
            "seed": int(seed),
            "run_tag": f"seed{seed}",
            "wandb_project": "drl-phase1-5f-wave2",
            "wandb_group": f"mz-hydro-wave2-seed{seed}",
            "run_mode": run_mode,
            "resume": bool(resume),
            "wandb_mode": wandb_mode,
        }
    )
    config = HydronicWave2Config(**values)
    return HydronicWave2Suite(config, force_final_eval=force_final_eval)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_path_isolation(suite: HydronicWave2Suite) -> None:
    output = suite.ppo.output_root.resolve()
    v3_root = suite.v3_source_suite.ppo.output_root.resolve()
    if output == v3_root or _is_relative_to(output, v3_root) or _is_relative_to(v3_root, output):
        raise RuntimeError(
            f"Wave-2 output is not isolated from v3: output={output}, v3={v3_root}"
        )
    if "phase1_5f_wave2" not in output.parts:
        raise RuntimeError(f"Unexpected Wave-2 output root: {output}")
    if "phase1_5f_v3" in output.parts:
        raise RuntimeError(f"Wave-2 output resolves inside the v3 tree: {output}")


def _checkpoint_inventory_files(run_dir: Path) -> list[Path]:
    directory = run_dir / "checkpoints"
    if not directory.exists():
        return []
    files: set[Path] = set()
    for pointer_name in ("best.json", "latest.json"):
        pointer_path = directory / pointer_name
        if not pointer_path.exists():
            continue
        files.add(pointer_path)
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        manifest_name = pointer.get("manifest")
        if not manifest_name:
            continue
        manifest_path = directory / str(manifest_name)
        files.add(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest.get("files", {}).values():
            name = record.get("name") if isinstance(record, Mapping) else None
            if name:
                files.add(directory / str(name))
    return sorted(files)


def _v3_inventory_paths(suite: HydronicWave2Suite) -> list[Path]:
    source = suite.v3_source_suite
    paths: set[Path] = {
        Path(v3.__file__).resolve(),
        source.ppo.case_dir / "PPO_MAPPO_Phase1_5f_Retrain.ipynb",
    }
    for config in (source.ppo, source.mappo):
        for name in (
            "run_manifest.json",
            "training_metrics.csv",
            "sb3_updates.csv",
            "train_week_eval.csv",
        ):
            paths.add(config.run_dir / name)
        paths.update(_checkpoint_inventory_files(config.run_dir))
        train_week = config.run_dir / "diagnostics" / "train_week"
        if train_week.exists():
            paths.update(train_week.glob("train_week_epoch_*.csv"))
    final_dir = source.run_dir / "final_validation"
    if final_dir.exists():
        paths.update(path for path in final_dir.iterdir() if path.is_file())
    missing = sorted(str(path) for path in paths if not path.exists())
    if missing:
        raise RuntimeError("Required immutable v3 artifacts are missing: " + "; ".join(missing))
    return sorted(path.resolve() for path in paths)


def build_v3_inventory(suite: HydronicWave2Suite) -> dict[str, Any]:
    project = suite.ppo.project_root.resolve()
    files: dict[str, Any] = {}
    for path in _v3_inventory_paths(suite):
        try:
            name = str(path.relative_to(project)).replace("\\", "/")
        except ValueError:
            name = str(path)
        files[name] = {
            "size": int(path.stat().st_size),
            "sha256": core.sha256_file(path),
        }
    return {
        "schema": "phase1.5f-wave2-v3-inventory-v1",
        "source_root": str(suite.v3_source_suite.run_dir),
        "files": files,
        "inventory_sha256": core.sha256_payload(files),
    }


def establish_or_verify_v3_inventory(suite: HydronicWave2Suite) -> dict[str, Any]:
    assert_path_isolation(suite)
    current = build_v3_inventory(suite)
    path = suite.run_dir / "v3_source_inventory.json"
    if path.exists():
        baseline = json.loads(path.read_text(encoding="utf-8"))
        if baseline.get("inventory_sha256") != current["inventory_sha256"]:
            raise RuntimeError(
                "Immutable v3 source inventory changed after Wave-2 registration"
            )
        return baseline
    suite.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {**current, "registered_at": core.utc_now()}
    core.atomic_write_json(path, payload)
    return payload


def verify_v3_unchanged(suite: HydronicWave2Suite) -> dict[str, Any]:
    path = suite.run_dir / "v3_source_inventory.json"
    if not path.exists():
        raise RuntimeError("Wave-2 v3 source inventory has not been registered")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    current = build_v3_inventory(suite)
    passed = baseline.get("inventory_sha256") == current.get("inventory_sha256")
    report = {
        "schema": "phase1.5f-wave2-isolation-verification-v1",
        "checked_at": core.utc_now(),
        "passed": passed,
        "registered_sha256": baseline.get("inventory_sha256"),
        "current_sha256": current.get("inventory_sha256"),
    }
    core.atomic_write_json(suite.run_dir / "v3_isolation_verification.json", report)
    if not passed:
        raise RuntimeError("One or more immutable v3 artifacts changed")
    return report


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Required diagnostic CSV is missing: {path}")
    return pd.read_csv(path)


def reconstruct_reward_components(
    frame: pd.DataFrame,
    config: HydronicWave2Config,
) -> pd.DataFrame:
    required = {
        "cost",
        "reward",
        *(f"pmv_{zone}" for zone in config.zones),
        *(f"occ_{zone}" for zone in config.zones),
        *(f"setpoint_{zone}" for zone in config.zones),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Trajectory is missing reward reconstruction columns: {missing}")
    weights = config.reward_weights
    previous = np.full(config.n_zones, 298.15 - 273.15, dtype=float)
    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        setpoints = np.asarray(
            [core.safe_float(record[f"setpoint_{zone}"]) for zone in config.zones],
            dtype=float,
        )
        excesses = np.asarray(
            [
                max(
                    0.0,
                    abs(core.safe_float(record[f"pmv_{zone}"]))
                    - config.comfort_threshold,
                )
                if core.safe_float(record[f"occ_{zone}"]) > 0.0
                else 0.0
                for zone in config.zones
            ],
            dtype=float,
        )
        energy = (
            weights["w_energy"]
            * (core.safe_float(record["cost"]) / config.n_zones)
            * weights["scaler_energy"]
        )
        comfort = (
            weights["w_comfort"]
            * float(np.sum(excesses**2) / config.n_zones)
            * weights["scaler_comfort"]
        )
        smooth = (
            weights["w_smooth"]
            * float(np.sum(np.abs(setpoints - previous)) / config.n_zones)
            * weights["scaler_smooth"]
        )
        reconstructed = -(energy + comfort + smooth)
        rows.append(
            {
                "reward_energy_reconstructed": energy,
                "reward_comfort_reconstructed": comfort,
                "reward_smooth_reconstructed": smooth,
                "reward_reconstructed": reconstructed,
                "reward_abs_error": abs(
                    reconstructed - core.safe_float(record["reward"])
                ),
                "violating_step": bool(np.any(excesses > 0.0)),
                "violating_zone_count": int(np.sum(excesses > 0.0)),
                "deep_violating_zone_count": int(
                    np.sum(
                        [
                            core.safe_float(record[f"occ_{zone}"]) > 0.0
                            and abs(core.safe_float(record[f"pmv_{zone}"]))
                            > config.c2_deep_pmv_limit
                            for zone in config.zones
                        ]
                    )
                ),
                "occupied_zone_count": int(
                    np.sum(
                        [
                            core.safe_float(record[f"occ_{zone}"]) > 0.0
                            for zone in config.zones
                        ]
                    )
                ),
            }
        )
        previous = setpoints
    return pd.DataFrame(rows)


def c2_trajectory_metrics(
    frame: pd.DataFrame,
    config: HydronicWave2Config,
) -> dict[str, Any]:
    components = reconstruct_reward_components(frame, config)
    energy = components["reward_energy_reconstructed"].to_numpy(dtype=float)
    comfort = components["reward_comfort_reconstructed"].to_numpy(dtype=float)
    violating = components["violating_step"].to_numpy(dtype=bool)
    all_ratio = float(np.sum(comfort) / max(np.sum(energy), 1e-12))
    if np.any(violating):
        active_ratio: float | None = float(
            np.sum(comfort[violating]) / max(np.sum(energy[violating]), 1e-12)
        )
    else:
        active_ratio = None
    occupied_zone_steps = int(components["occupied_zone_count"].sum())
    deep_zone_steps = int(components["deep_violating_zone_count"].sum())
    deep_rate = deep_zone_steps / max(occupied_zone_steps, 1)
    return {
        "reward_formula_max_abs_error": float(components["reward_abs_error"].max()),
        "reward_formula_mean_abs_error": float(components["reward_abs_error"].mean()),
        "comfort_energy_ratio_all_steps": all_ratio,
        "comfort_energy_ratio_violating_steps": active_ratio,
        "violating_steps": int(np.sum(violating)),
        "violating_zone_steps": int(components["violating_zone_count"].sum()),
        "deep_violating_zone_steps": deep_zone_steps,
        "occupied_zone_steps": occupied_zone_steps,
        "deep_violation_rate": float(deep_rate),
        "signal_pass": bool(
            active_ratio is None or active_ratio >= config.c2_signal_threshold
        ),
        "formula_pass": bool(components["reward_abs_error"].max() <= 1e-8),
        "deep_rate_pass": bool(deep_rate < config.c2_deep_rate_limit),
    }


def _heldout_action_saturation(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    zones: Sequence[str],
) -> float:
    values: list[float] = []
    for zone in zones:
        occupied = frame[f"occ_{zone}"].to_numpy(dtype=float) > 0.0
        zone_actions = actions[f"action_{zone}"].to_numpy(dtype=float)
        values.extend(zone_actions[occupied].tolist())
    return float(np.mean(np.abs(values) > 0.95)) if values else float("nan")


def _observation_audit(frame: pd.DataFrame, zones: Sequence[str]) -> dict[str, Any]:
    mismatches = 0
    precursors = 0
    for zone in zones:
        actual = frame[f"occ_{zone}"].to_numpy(dtype=float) > 0.0
        observed = frame[f"obs_Occ_{zone}_0"].to_numpy(dtype=float) > 0.0
        mismatches += int(np.sum(actual != observed))
        future = np.zeros(len(frame), dtype=bool)
        for index in range(1, 5):
            future |= frame[f"obs_Occ_{zone}_f{index}"].to_numpy(dtype=float) > 0.0
        precursors += int(np.sum(~observed & future))
    return {
        "current_occupancy_mismatches": mismatches,
        "future_occupancy_precursor_rows": precursors,
        "price_min": float(frame["obs_Price_0"].min()),
        "price_max": float(frame["obs_Price_0"].max()),
        "passed": bool(mismatches == 0 and precursors > 0),
    }


def _diagnose_v3_algorithm(
    suite: HydronicWave2Suite,
    algorithm: str,
) -> dict[str, Any]:
    source = suite.v3_source_suite
    config = source.ppo if algorithm == "ppo" else source.mappo
    training = _read_csv(config.run_dir / "training_metrics.csv")
    updates = _read_csv(config.run_dir / "sb3_updates.csv")
    evaluations = _read_csv(config.run_dir / "train_week_eval.csv")
    latest_eval = evaluations.iloc[-1]
    best_row = evaluations.loc[evaluations["return"].idxmax()]
    recent = training.tail(60)
    legacy_ratios = (
        recent["reward_comfort_mean"].abs()
        / recent["reward_energy_mean"].abs().clip(lower=1e-12)
    )
    latest_trajectory = _read_csv(Path(str(latest_eval["trajectory"])))
    latest_c2 = c2_trajectory_metrics(latest_trajectory, suite.ppo)
    final_dir = source.run_dir / "final_validation"
    trajectory_name = (
        "ppo_validation_hydronic_2zone.csv"
        if algorithm == "ppo"
        else "mappo_validation_hydronic_2zone.csv"
    )
    action_name = "v3_ppo_actions.csv" if algorithm == "ppo" else "v3_mappo_actions.csv"
    heldout = _read_csv(final_dir / trajectory_name)
    actions = _read_csv(final_dir / action_name)
    heldout_c2 = c2_trajectory_metrics(heldout, suite.ppo)
    final_report = json.loads(
        (final_dir / "final_validation_report.json").read_text(encoding="utf-8")
    )
    convergence = final_report["convergence"][algorithm]
    policy_std_column = "policy_std"
    result = {
        "algorithm": algorithm,
        "epochs": int(training["epoch"].iloc[-1]),
        "global_step": int(training["global_step"].iloc[-1]),
        "original_convergence": convergence,
        "policy_std_first": float(updates[policy_std_column].iloc[0]),
        "policy_std_final": float(updates[policy_std_column].iloc[-1]),
        "train_week_occupied_saturation_first": float(
            evaluations["occupied_saturation"].iloc[0]
        ),
        "train_week_occupied_saturation_last": float(
            latest_eval["occupied_saturation"]
        ),
        "best_epoch": int(best_row["epoch"]),
        "best_train_week_occupied_saturation": float(best_row["occupied_saturation"]),
        "heldout_occupied_saturation": _heldout_action_saturation(
            heldout, actions, suite.ppo.zones
        ),
        "legacy_c2_minimum_recent_ratio": float(legacy_ratios.min()),
        "latest_train_week_c2_wave2": latest_c2,
        "heldout_c2_wave2": heldout_c2,
        "observation_audit": _observation_audit(latest_trajectory, suite.ppo.zones),
        "median_precooling_lead_hours_latest": float(
            latest_eval["median_precooling_lead_hours"]
        ),
    }
    if algorithm == "ppo":
        result["diagnostic_classification"] = (
            "retrain_required_policy_distribution_and_action_saturation"
        )
    else:
        result["diagnostic_classification"] = (
            "no_retrain_required_legacy_c2_dilution_only"
        )
    return result


def run_failure_diagnosis(suite: HydronicWave2Suite) -> dict[str, Any]:
    inventory = establish_or_verify_v3_inventory(suite)
    ppo = _diagnose_v3_algorithm(suite, "ppo")
    mappo = _diagnose_v3_algorithm(suite, "mappo")
    report = {
        "schema": "phase1.5f-wave2-failure-diagnosis-v1",
        "generated_at": core.utc_now(),
        "v3_inventory_sha256": inventory["inventory_sha256"],
        "v3_original_status_is_not_modified": True,
        "ppo": ppo,
        "mappo": mappo,
        "decision": {
            "retrain_ppo": True,
            "retrain_mappo": False,
            "continue_v3_training": False,
            "reason": (
                "PPO plateaued with high policy std and occupied action saturation; "
                "MAPPO is optimization/action healthy and only fails the diluted legacy C2."
            ),
        },
    }
    output = suite.run_dir / "failure_diagnosis.json"
    core.atomic_write_json(output, report)
    markdown = [
        "# Phase 1.5f Wave-2 failure diagnosis",
        "",
        "The Phase 1.5f v3 artifacts remain immutable and retain their original result.",
        "",
        "- PPO: retraining is required; C1 passed, but policy std and occupied saturation remained unhealthy.",
        "- MAPPO: retraining is not supported by the evidence; C1/C3/C4 passed and legacy C2 is diluted by zero-penalty steps.",
        f"- PPO std: {ppo['policy_std_first']:.3f} -> {ppo['policy_std_final']:.3f}.",
        f"- PPO occupied saturation: {ppo['train_week_occupied_saturation_first']:.3f} -> {ppo['train_week_occupied_saturation_last']:.3f}.",
        f"- MAPPO violating-step comfort/energy ratio: {mappo['latest_train_week_c2_wave2']['comfort_energy_ratio_violating_steps']:.3f}.",
        "",
        "The legacy C2 result is still reported; Wave-2 uses the pre-registered violation-conditioned C2.",
    ]
    (suite.run_dir / "failure_diagnosis.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    c2_rows: list[dict[str, Any]] = []
    for algorithm, item in (("ppo", ppo), ("mappo", mappo)):
        for scope, metrics in (
            ("latest_train_week", item["latest_train_week_c2_wave2"]),
            ("heldout_best", item["heldout_c2_wave2"]),
        ):
            c2_rows.append(
                {
                    "algorithm": algorithm,
                    "scope": scope,
                    "legacy_minimum_recent_training_ratio": item[
                        "legacy_c2_minimum_recent_ratio"
                    ],
                    **metrics,
                }
            )
    core.write_csv(suite.run_dir / "failure_diagnosis_c2.csv", c2_rows)
    _plot_failure_diagnosis(suite, report)
    return report


def _plot_failure_diagnosis(
    suite: HydronicWave2Suite,
    report: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = suite.v3_source_suite
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for algorithm, config, color in (
        ("ppo", source.ppo, "tab:blue"),
        ("mappo", source.mappo, "tab:orange"),
    ):
        training = _read_csv(config.run_dir / "training_metrics.csv")
        updates = _read_csv(config.run_dir / "sb3_updates.csv")
        evaluations = _read_csv(config.run_dir / "train_week_eval.csv")
        axes[0, 0].plot(
            training["epoch"], training["rolling_30"], label=algorithm.upper(), color=color
        )
        axes[0, 1].plot(
            updates["epoch"], updates["policy_std"], label=algorithm.upper(), color=color
        )
        axes[1, 0].plot(
            evaluations["epoch"],
            evaluations["occupied_saturation"],
            marker="o",
            markersize=3,
            label=algorithm.upper(),
            color=color,
        )
    axes[0, 0].set_title("v3 reward rolling mean")
    axes[0, 0].set_ylabel("30-epoch mean return")
    axes[0, 1].axhspan(0.1, 0.4, color="green", alpha=0.12)
    axes[0, 1].axhline(0.5, color="red", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Policy standard deviation")
    axes[1, 0].axhline(0.5, color="red", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Deterministic occupied action saturation")
    labels: list[str] = []
    all_ratios: list[float] = []
    active_ratios: list[float] = []
    for algorithm in ("ppo", "mappo"):
        item = report[algorithm]["latest_train_week_c2_wave2"]
        labels.append(algorithm.upper())
        all_ratios.append(item["comfort_energy_ratio_all_steps"])
        active_ratios.append(item["comfort_energy_ratio_violating_steps"] or 0.0)
    positions = np.arange(len(labels))
    axes[1, 1].bar(positions - 0.18, all_ratios, 0.36, label="all steps")
    axes[1, 1].bar(positions + 0.18, active_ratios, 0.36, label="violating steps")
    axes[1, 1].axhline(0.05, color="red", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].set_title("Comfort / energy signal ratio")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
        axis.set_xlabel("Epoch" if axis is not axes[1, 1] else "Algorithm")
    figure.tight_layout()
    figure.savefig(suite.run_dir / "failure_diagnosis.png", dpi=170)
    plt.close(figure)


def _scientific_config_errors(config: HydronicWave2Config) -> list[str]:
    errors: list[str] = []
    approved = v3.build_hydronic_v3_suite(
        config.project_root,
        run_mode=config.run_mode,
        resume=config.resume,
        wandb_mode=config.wandb_mode,
        seed=42,
    ).ppo
    if config.algorithm not in {"ppo", "final", "confirmation"}:
        errors.append(f"Unknown Wave-2 algorithm: {config.algorithm}")
    if config.seed not in APPROVED_SEEDS or config.num_envs != 4:
        errors.append("Wave-2 requires an approved seed and four environments")
    frozen_fields = (
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
        "residual_scale",
        "occupied_base_k",
        "unoccupied_base_k",
        "action_min_k",
        "action_max_k",
        "comfort_threshold",
        "max_full_epochs",
        "smoke_epochs",
        "min_early_stop_epoch",
        "train_eval_interval",
        "value_loss_coef",
        "max_grad_norm",
    )
    for name in frozen_fields:
        if getattr(config, name) != getattr(approved, name):
            errors.append(f"Frozen v3 field changed in Wave-2: {name}")
    if config.log_std_init != LOG_STD_INIT:
        errors.append("The sole PPO change must be log_std_init=-1.0")
    if core.AUXILIARY_HYDRONIC_KEYS.intersection(
        core.hydronic_action_payload((298.15, 303.15))
    ):
        errors.append("Forbidden aux-overwrite keys are active")
    if config.steps_per_epoch % config.batch_size:
        errors.append("Rollout buffer is not divisible by batch_size")
    return errors


def _require_seed42_pass_for_supplemental_seed(suite: HydronicWave2Suite) -> None:
    if suite.ppo.seed == 42 or suite.ppo.run_mode != "full":
        return
    manifest = (
        suite.ppo.output_root / "full" / "seed42" / "ppo" / "run_manifest.json"
    )
    if not manifest.exists():
        raise RuntimeError("Wave-2 seed42 must finish and pass before supplemental seeds")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not payload.get("converged"):
        raise RuntimeError("Wave-2 seed42 did not pass C1-C4; supplemental seeds are blocked")
    expected = payload.get("frozen_cross_seed_hash")
    if expected and expected != cross_seed_scientific_hash(suite.ppo):
        raise RuntimeError("Supplemental seed scientific configuration differs from seed42")


def cross_seed_scientific_hash(config: HydronicWave2Config) -> str:
    payload = dict(config.scientific_payload)
    payload.pop("seed", None)
    return core.sha256_payload(payload)


class Wave2ForecastProvider(core.ForecastProvider):
    """Forecast prefetch with Wave-2 lifecycle ownership and no v3 writes."""

    def __init__(self, config: HydronicWave2Config, *, extra_days: int = 0):
        super().__init__(config)
        self.total_horizon = (
            (config.simulation_days + int(extra_days)) * 24 * 3600
            + 5 * config.control_period
        )

    def prefetch_all(self) -> None:
        config = self.config
        lifecycle = core.TestIDLifecycle(
            config,
            owner=f"mz_hydro_{config.algorithm}_wave2_forecast",
            algorithm=f"mz_hydro_{config.algorithm}_wave2",
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
                key: np.asarray(
                    [0.0 if item is None else item for item in values], dtype=float
                )
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


class Wave2HydronicResidualEnv(core.HydronicResidualEnv):
    """The unchanged v3 Hydronic environment with Wave-2 ID labels."""

    def __init__(self, config: HydronicWave2Config, *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)
        self.lifecycle = core.TestIDLifecycle(
            config,
            owner=f"mz_hydro_{config.algorithm}_{self.phase}_{self.rank}",
            algorithm=f"mz_hydro_{config.algorithm}_wave2",
            phase=self.phase,
            rank=self.rank,
        )


def build_environment_wave2(
    config: HydronicWave2Config,
    forecast_data: Mapping[str, Sequence[float]],
    *,
    is_validation: bool = False,
    rank: int | str = 0,
    phase: str = "training",
    capture_series: bool = False,
    monitored: bool = False,
) -> gym.Env:
    environment: gym.Env = Wave2HydronicResidualEnv(
        config,
        forecast_data,
        is_validation=is_validation,
        legacy_fixed_baseline=False,
        rank=rank,
        phase=phase,
        capture_series=capture_series,
    )
    environment = core.NormalizedObservationWrapper(environment)
    return core.Monitor(environment) if monitored else environment


def _subprocess_environment_wave2(
    config: HydronicWave2Config,
    forecast_data: Mapping[str, Sequence[float]],
    rank: int,
) -> gym.Env:
    return build_environment_wave2(
        config,
        forecast_data,
        rank=rank,
        phase="training",
        monitored=True,
    )


def validation_columns(config: HydronicWave2Config) -> list[str]:
    return v3.validation_columns(config)


def _rollout_row(
    config: HydronicWave2Config,
    info: Mapping[str, Any],
    reward: float,
) -> dict[str, Any]:
    return v3._rollout_row(config, info, reward)


def rollout_policy(
    config: HydronicWave2Config,
    forecast_data: Mapping[str, Sequence[float]],
    *,
    model: Any | None,
    is_validation: bool,
    phase: str,
    zero_policy: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    environment = build_environment_wave2(
        config,
        forecast_data,
        is_validation=is_validation,
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


def _used_validation_start_days(case_dir: Path) -> dict[int, list[str]]:
    used: dict[int, list[str]] = {}
    for path in case_dir.rglob("*validation*.csv"):
        if "phase1_5f_wave2" in path.parts:
            continue
        try:
            first = pd.read_csv(path, usecols=["time"], nrows=1)
            if first.empty:
                continue
            day = int(round(float(first["time"].iloc[0]) / 86400.0))
            used.setdefault(day, []).append(str(path))
        except Exception:
            continue
    return used


def select_confirmation_day(suite: HydronicWave2Suite) -> dict[str, Any]:
    path = suite.run_dir / "confirmation_week.json"
    used = _used_validation_start_days(suite.ppo.case_dir)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        day = int(payload["start_day"])
        if day in used:
            raise RuntimeError(
                "The registered Wave-2 confirmation week now appears in historical data"
            )
        return payload
    day = next((candidate for candidate in CONFIRMATION_DAY_CANDIDATES if candidate not in used), None)
    if day is None:
        raise RuntimeError("No unused confirmation week was found")
    payload = {
        "schema": "phase1.5f-wave2-confirmation-week-v1",
        "selected_at": core.utc_now(),
        "start_day": int(day),
        "selection_rule": "first unused 7-day offset candidate beginning at day 227",
        "used_start_days_at_registration": sorted(used),
        "not_used_for_checkpoint_or_hyperparameter_selection": True,
    }
    suite.run_dir.mkdir(parents=True, exist_ok=True)
    core.atomic_write_json(path, payload)
    return payload


def _preflight_online_snapshot(
    suite: HydronicWave2Suite,
) -> tuple[dict[str, Any], dict[str, Any]]:
    forecast = Wave2ForecastProvider(suite.ppo, extra_days=7)
    forecast.prefetch_all()
    boptest = core.query_boptest_version(suite.ppo)
    calendar = v3.audit_calendar_and_forecast(suite.ppo, forecast.data)
    return boptest, calendar


def run_preflight(
    suite: HydronicWave2Suite,
    *,
    online: bool = False,
) -> dict[str, Any]:
    assert_path_isolation(suite)
    suite.run_dir.mkdir(parents=True, exist_ok=True)
    errors = _scientific_config_errors(suite.ppo)
    _require_seed42_pass_for_supplemental_seed(suite)
    inventory = establish_or_verify_v3_inventory(suite)
    diagnosis = run_failure_diagnosis(suite)
    confirmation = select_confirmation_day(suite)
    cleanup = {
        "ppo": core.cleanup_stale_testids(suite.ppo),
        "final": core.cleanup_stale_testids(suite.final),
        "confirmation": core.cleanup_stale_testids(suite.confirmation),
    }
    if any(
        row.get("status") == "cleanup_failed"
        for rows in cleanup.values()
        for row in rows
    ):
        errors.append("One or more stale Wave-2 TestIDs could not be stopped")
    aux = v3.audit_aux_provenance(suite.ppo)
    if not aux.get("passed"):
        errors.append("The frozen aux-overwrite-disabled provenance no longer passes")
    online_path = suite.run_dir / "preflight_online.json"
    boptest: dict[str, Any] = {"status": "not_queried"}
    calendar = v3.audit_calendar_and_forecast(suite.ppo)
    source = "offline"
    if online:
        boptest, calendar = _preflight_online_snapshot(suite)
        source = "online-query"
    elif online_path.exists():
        prior = json.loads(online_path.read_text(encoding="utf-8"))
        if prior.get("config_hash") == suite.ppo.config_hash:
            boptest = prior.get("boptest", boptest)
            calendar = prior.get("calendar_and_forecast", calendar)
            source = "offline-recheck-with-preserved-online-evidence"
    report = {
        "schema": "phase1.5f-wave2-preflight-v1",
        "timestamp": core.utc_now(),
        "passed": not errors,
        "preflight_source": source,
        "errors": errors,
        "config_hash": suite.ppo.config_hash,
        "cross_seed_scientific_hash": cross_seed_scientific_hash(suite.ppo),
        "scientific_config": suite.ppo.scientific_payload,
        "sole_training_change": {"log_std_init": suite.ppo.log_std_init},
        "v3_inventory_sha256": inventory["inventory_sha256"],
        "diagnosis_decision": diagnosis["decision"],
        "confirmation_week": confirmation,
        "stale_cleanup": cleanup,
        "aux_provenance": aux,
        "boptest": boptest,
        "calendar_and_forecast": calendar,
    }
    core.atomic_write_json(suite.run_dir / "preflight.json", report)
    if online:
        core.atomic_write_json(online_path, report)
    notes = [
        "P1: aux-overwrite-disabled remains frozen; v3 artifacts are read-only.",
        "Wave-2 changes only PPO log_std_init from the SB3 default 0.0 to -1.0.",
        "The legacy C2 result remains reported; C2_wave2 is pre-registered before training.",
        f"Confirmation week start day: {confirmation['start_day']} (not used for selection).",
    ]
    (suite.run_dir / "return_notes.md").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError("Phase 1.5f Wave-2 preflight failed: " + "; ".join(errors))
    return report


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
    eval_interval: int = 25,
) -> dict[str, Any]:
    legacy = v3.evaluate_convergence(
        training_rows,
        update_rows,
        train_eval_rows,
        best_epoch=best_epoch,
        min_epoch=min_epoch,
        algorithm="ppo",
        eval_interval=eval_interval,
    )
    result = dict(legacy)
    result["schema"] = "phase1.5f-wave2-c1-c4"
    result["C2_legacy"] = bool(legacy.get("C2", False))
    result["C2_legacy_details"] = legacy.get("C2_details", {})
    if not result.get("eligible"):
        result["C2_wave2"] = False
        result["C2"] = False
        result["reasons"] = ["not_an_eligible_validation_boundary"]
        result["converged"] = False
        return result
    recent = list(train_eval_rows[-5:])
    recent_five_complete = len(recent) == 5
    latest = recent[-1]
    numeric_ratios = [
        core.safe_float(row.get("comfort_energy_ratio_violating_steps"), float("nan"))
        for row in recent
        if int(core.safe_float(row.get("violating_steps"), 0)) > 0
    ]
    numeric_ratios = [value for value in numeric_ratios if np.isfinite(value)]
    median_ratio: float | None = (
        float(np.median(numeric_ratios)) if numeric_ratios else None
    )
    latest_violations = int(core.safe_float(latest.get("violating_steps"), 0))
    latest_ratio = core.safe_float(
        latest.get("comfort_energy_ratio_violating_steps"), float("nan")
    )
    latest_signal_pass = bool(
        latest_violations == 0
        or (np.isfinite(latest_ratio) and latest_ratio >= C2_SIGNAL_THRESHOLD)
    )
    median_signal_pass = bool(
        median_ratio is None or median_ratio >= C2_SIGNAL_THRESHOLD
    )
    formula_pass = bool(
        all(core.safe_float(row.get("reward_formula_max_abs_error"), 1.0) <= 1e-8 for row in recent)
    )
    deep_pass = bool(
        all(
            core.safe_float(row.get("deep_violation_rate"), 1.0)
            < C2_DEEP_RATE_LIMIT
            for row in recent
        )
    )
    c2_wave2 = bool(
        recent_five_complete
        and latest_signal_pass
        and median_signal_pass
        and formula_pass
        and deep_pass
    )
    result["C2_wave2"] = c2_wave2
    result["C2"] = c2_wave2
    result["C2_details"] = {
        "latest_violating_steps": latest_violations,
        "recent_five_complete": recent_five_complete,
        "latest_violating_step_ratio": (
            latest_ratio if np.isfinite(latest_ratio) else None
        ),
        "recent_five_violating_step_ratio_median": median_ratio,
        "latest_signal_pass": latest_signal_pass,
        "median_signal_pass": median_signal_pass,
        "reward_formula_pass": formula_pass,
        "deep_violation_rate_pass": deep_pass,
        "legacy_all_step_result": result["C2_legacy"],
    }
    result["reasons"] = [name for name in ("C1", "C2", "C3", "C4") if not result[name]]
    result["converged"] = not result["reasons"]
    return result


def evaluate_failure_stop(
    update_rows: Sequence[Mapping[str, Any]],
    train_eval_rows: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    eval_interval: int,
) -> dict[str, Any]:
    result = {
        "triggered": False,
        "eligible": False,
        "epoch": int(epoch),
        "reasons": [],
    }
    if epoch < FAIL_FAST_EPOCH or epoch % eval_interval != 0:
        return result
    result["eligible"] = True
    std_median = _median(
        core.safe_float(row.get("policy_std"), float("nan"))
        for row in list(update_rows[-20:])
    )
    recent_saturation = [
        core.safe_float(row.get("occupied_saturation"), float("nan"))
        for row in list(train_eval_rows[-2:])
    ]
    reasons: list[str] = []
    if np.isfinite(std_median) and std_median > FAIL_FAST_STD_LIMIT:
        reasons.append("policy_std_above_0.5")
    if len(recent_saturation) == 2 and all(
        np.isfinite(value) and value >= FAIL_FAST_SATURATION_LIMIT
        for value in recent_saturation
    ):
        reasons.append("two_consecutive_occupied_saturation_at_or_above_0.5")
    result.update(
        {
            "triggered": bool(reasons),
            "reasons": reasons,
            "recent_20_policy_std_median": std_median,
            "recent_two_occupied_saturation": recent_saturation,
        }
    )
    return result


class Wave2TrainWeekEvaluator:
    def __init__(
        self,
        config: HydronicWave2Config,
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
        c2 = c2_trajectory_metrics(frame, self.config)
        return {
            "epoch": int(epoch),
            "global_step": int(global_step),
            **metrics,
            **c2,
            "action_health_by_zone": v3.action_health_by_zone(
                frame, actions, self.config
            ),
            "trajectory": str(output),
            "timestamp": core.utc_now(),
        }


class Wave2SafeWandb(v3.V3SafeWandb):
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
                        f"mz-hydro-ppo-phase1-5f-wave2-"
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


class Wave2PPOCallback(core.Phase15fCallback):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        restored = dict(kwargs.get("restored_state") or {})
        self.failure_stop = dict(
            restored.get("failure_stop", {"triggered": False, "reasons": []})
        )

    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema": "phase1.5f-state-wave2-ppo",
            "config_hash": self.config.config_hash,
            "committed_epoch": self.committed_epoch,
            "global_step": self.committed_epoch * self.config.steps_per_epoch,
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
            "v3_inventory_sha256": json.loads(
                (self.config.suite_dir / "v3_source_inventory.json").read_text(
                    encoding="utf-8"
                )
            )["inventory_sha256"],
            "updated_at": core.utc_now(),
        }

    def on_post_update(self, model: Any) -> bool:
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
            eval_interval=self.config.eval_interval,
        )
        self.failure_stop = evaluate_failure_stop(
            self.sb3_rows,
            self.train_eval_rows,
            epoch=epoch,
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
            "convergence/C2_legacy": self.convergence.get("C2_legacy", False),
            "convergence/C2_wave2": self.convergence.get("C2_wave2", False),
            "convergence/C3": self.convergence.get("C3", False),
            "convergence/C4": self.convergence.get("C4", False),
            "convergence/converged": self.convergence.get("converged", False),
            "failure_stop/triggered": self.failure_stop.get("triggered", False),
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
                    "train_week/c2_all_step_ratio": latest[
                        "comfort_energy_ratio_all_steps"
                    ],
                    "train_week/c2_violating_step_ratio": latest[
                        "comfort_energy_ratio_violating_steps"
                    ],
                    "train_week/deep_violation_rate": latest[
                        "deep_violation_rate"
                    ],
                }
            )
        self.wandb.log(metrics, int(model.num_timesteps))
        print(
            f"[HYDRO PPO Wave-2 Epoch {epoch:03d}] "
            f"reward={training_row['reward_mean']:.3f} "
            f"rolling30={training_row['rolling_30']:.3f} "
            f"std={update_row['policy_std']:.3f} "
            f"lr={update_row['learning_rate']:.3e} "
            f"best_train_week={self.best_score:.3f}"
        )
        should_continue = not bool(
            self.convergence.get("converged") or self.failure_stop.get("triggered")
        )
        self.started_at = time.perf_counter()
        self._reset_epoch_accumulators()
        return should_continue


def _checkpoint_complete_state(
    config: HydronicWave2Config,
    state: Mapping[str, Any],
) -> bool:
    return bool(
        state.get("convergence", {}).get("converged")
        or state.get("failure_stop", {}).get("triggered")
        or int(state.get("committed_epoch", 0)) >= config.max_epochs
    )


def _write_run_manifest(
    config: HydronicWave2Config,
    identity: Mapping[str, Any],
    **updates: Any,
) -> dict[str, Any]:
    path = config.run_dir / "run_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != config.config_hash:
            raise RuntimeError("Existing Wave-2 manifest has a different config hash")
    else:
        manifest = {
            "schema": "phase1.5f-run-wave2",
            "case": "mz_hydro",
            "algorithm": "ppo",
            "started_at": core.utc_now(),
            "config_hash": config.config_hash,
            "cross_seed_scientific_hash": cross_seed_scientific_hash(config),
            "frozen_cross_seed_hash": cross_seed_scientific_hash(config),
            "scientific_config": config.scientific_payload,
            "sole_training_change": {"log_std_init": config.log_std_init},
            "wandb_run_id": identity["wandb_run_id"],
            "status": "starting",
        }
    manifest.update(core.json_ready(updates))
    core.atomic_write_json(path, manifest)
    return manifest


def _finish_manifest(
    config: HydronicWave2Config,
    identity: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    status: str,
    wall_seconds: float,
    error: str | None,
) -> dict[str, Any]:
    epoch = int(state.get("committed_epoch", 0))
    converged = bool(state.get("convergence", {}).get("converged", False))
    failure_stop = bool(state.get("failure_stop", {}).get("triggered", False))
    if converged:
        outcome = "converged"
    elif failure_stop:
        outcome = "failure_stop_policy_distribution"
    elif epoch >= config.max_epochs:
        outcome = "cap_reached_not_fully_converged"
    else:
        outcome = status
    return _write_run_manifest(
        config,
        identity,
        status=status,
        completed_at=core.utc_now(),
        actual_epochs=epoch,
        global_step=int(state.get("global_step", epoch * config.steps_per_epoch)),
        converged=converged,
        early_stopped=bool(converged and epoch < config.max_epochs),
        failure_stop=state.get("failure_stop", {}),
        resume_count=int(state.get("resume_count", 0)),
        training_outcome=outcome,
        wall_seconds=wall_seconds,
        error=error,
    )


def generate_training_report(
    config: HydronicWave2Config,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    report = {
        "schema": "phase1.5f-wave2-training-report-v1",
        "generated_at": core.utc_now(),
        "seed": config.seed,
        "actual_epochs": int(state.get("committed_epoch", 0)),
        "global_step": int(state.get("global_step", 0)),
        "best_epoch": state.get("best_epoch"),
        "best_train_week_return": state.get("best_score"),
        "convergence": state.get("convergence", {}),
        "failure_stop": state.get("failure_stop", {}),
        "wandb_run_id": state.get("wandb_run_id"),
        "wandb_url": state.get("wandb_url"),
        "v3_inventory_sha256": state.get("v3_inventory_sha256"),
    }
    directory = config.run_dir / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    core.atomic_write_json(directory / "training_report_wave2.json", report)
    convergence = report["convergence"]
    lines = [
        "# Phase 1.5f Wave-2 PPO training report",
        "",
        f"- Seed: {config.seed}",
        f"- Epochs: {report['actual_epochs']}",
        f"- Best epoch: {report['best_epoch']}",
        f"- C1: {convergence.get('C1', False)}",
        f"- C2 legacy: {convergence.get('C2_legacy', False)}",
        f"- C2 Wave-2: {convergence.get('C2_wave2', False)}",
        f"- C3: {convergence.get('C3', False)}",
        f"- C4: {convergence.get('C4', False)}",
        f"- Converged: {convergence.get('converged', False)}",
        f"- Failure stop: {report['failure_stop'].get('triggered', False)}",
    ]
    (directory / "training_report_wave2.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


class Wave2PPOExperiment:
    def __init__(self, suite: HydronicWave2Suite):
        self.suite = suite
        self.config = suite.ppo

    def train(self) -> dict[str, Any]:
        run_preflight(self.suite, online=False)
        config = self.config
        checkpoint = core.AtomicCheckpointManager(config.run_dir, config.config_hash)
        latest = checkpoint.load_latest_metadata()
        if latest and not config.resume:
            raise RuntimeError("Wave-2 checkpoint exists and RESUME=False")
        if latest and _checkpoint_complete_state(config, latest["state"]):
            core.generate_training_diagnostics(config, latest["state"])
            generate_training_report(config, latest["state"])
            verify_v3_unchanged(self.suite)
            return latest["state"]
        preferred = latest["state"].get("wandb_run_id") if latest else None
        identity = core.get_or_create_run_identity(config, preferred)
        _write_run_manifest(config, identity, status="starting")
        wandb_logger = Wave2SafeWandb(config, identity["wandb_run_id"])
        wandb_logger.start()
        vector_environment: VecEnv | None = None
        callback: Wave2PPOCallback | None = None
        old_sigterm: Any = None
        started = time.perf_counter()
        status = "complete"
        error: str | None = None
        online_started = False
        lock_path = config.run_dir / "runtime" / "training.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with core.RunLock(lock_path):
                online_started = True
                old_sigterm, _ = core._install_termination_handler()
                forecast = Wave2ForecastProvider(config)
                forecast.prefetch_all()
                core.atomic_write_json(
                    config.run_dir / "forecast_composition.json",
                    v3.audit_calendar_and_forecast(config, forecast.data),
                )
                boptest = core.query_boptest_version(config)
                fingerprint = core.environment_fingerprint(config, boptest, forecast.data)
                if latest and latest["state"].get("environment_fingerprint") not in {
                    None,
                    fingerprint,
                }:
                    raise RuntimeError(
                        "Wave-2 environment fingerprint changed; refusing silent resume"
                    )
                _write_run_manifest(
                    config,
                    identity,
                    status="training",
                    boptest=boptest,
                    environment_fingerprint=fingerprint,
                )
                factories = [
                    partial(_subprocess_environment_wave2, config, forecast.data, rank)
                    for rank in range(config.num_envs)
                ]
                vector_environment = SubprocVecEnv(factories, start_method="spawn")
                if latest:
                    model = core.Phase15fPPO.load(
                        latest["model_path"], env=vector_environment, device="auto"
                    )
                    if int(model.num_timesteps) != int(
                        latest["pointer"]["num_timesteps"]
                    ):
                        raise RuntimeError(
                            "Loaded Wave-2 timestep does not match checkpoint pointer"
                        )
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
                        vf_coef=config.value_loss_coef,
                        max_grad_norm=config.max_grad_norm,
                        policy_kwargs={
                            "net_arch": {
                                "pi": list(config.policy_net),
                                "vf": list(config.policy_net),
                            },
                            "log_std_init": config.log_std_init,
                        },
                        tensorboard_log=str(config.run_dir / "tensorboard"),
                        seed=config.seed,
                        verbose=1,
                        device="auto",
                    )
                    restored_state = None
                callback = Wave2PPOCallback(
                    config,
                    checkpoint,
                    Wave2TrainWeekEvaluator(config, forecast.data),
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
                        tb_log_name="mz_hydro_ppo_phase1_5f_wave2",
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
            state = dict(committed["state"]) if committed else {}
            if state:
                state["global_step"] = int(state.get("committed_epoch", 0)) * config.steps_per_epoch
                core.generate_training_diagnostics(config, state)
                generate_training_report(config, state)
            _finish_manifest(
                config,
                identity,
                state,
                status=status,
                wall_seconds=time.perf_counter() - started,
                error=error,
            )
            verify_v3_unchanged(self.suite)
            wandb_logger.finish(
                lifecycle_path,
                {
                    "status": status,
                    "actual_epochs": int(state.get("committed_epoch", 0)),
                    "converged": bool(
                        state.get("convergence", {}).get("converged", False)
                    ),
                    "failure_stop": bool(
                        state.get("failure_stop", {}).get("triggered", False)
                    ),
                },
            )
        final = checkpoint.load_latest_metadata()
        if not final:
            raise RuntimeError("Wave-2 PPO ended before the first checkpoint commit")
        return final["state"]


def train_ppo(suite: HydronicWave2Suite) -> dict[str, Any]:
    return Wave2PPOExperiment(suite).train()


def _ppo_checkpoint_state(config: HydronicWave2Config) -> dict[str, Any] | None:
    metadata = core.AtomicCheckpointManager(
        config.run_dir, config.config_hash
    ).load_latest_metadata()
    return metadata["state"] if metadata else None


def _require_ppo_finished(
    suite: HydronicWave2Suite,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _ppo_checkpoint_state(suite.ppo)
    if state is None or not _checkpoint_complete_state(suite.ppo, state):
        raise RuntimeError("Wave-2 PPO training must finish before final evaluation")
    checkpoint = core.AtomicCheckpointManager(
        suite.ppo.run_dir, suite.ppo.config_hash
    )
    best = checkpoint.load_best_metadata()
    if best is None:
        raise RuntimeError("Wave-2 PPO has no deterministic train-week best checkpoint")
    return state, best


def _trajectory_and_action_paths(
    results_dir: Path,
    label: str,
) -> tuple[Path, Path]:
    return (
        results_dir / f"{label}_validation_hydronic_2zone.csv",
        results_dir / f"{label}_actions.csv",
    )


def _save_trajectory(
    config: HydronicWave2Config,
    results_dir: Path,
    label: str,
    frame: pd.DataFrame,
) -> tuple[Path, Path]:
    trajectory_path, actions_path = _trajectory_and_action_paths(results_dir, label)
    export = frame[validation_columns(config)]
    export.to_csv(trajectory_path, index=False)
    if len(export) != config.episode_steps:
        raise RuntimeError(
            f"{label} row count {len(export)} != {config.episode_steps}"
        )
    if list(pd.read_csv(trajectory_path, nrows=1).columns) != validation_columns(config):
        raise RuntimeError(f"{label} validation schema differs from Wave-2")
    actions = np.asarray(
        [json.loads(value) for value in frame["_actions"]], dtype=float
    )
    action_frame = pd.DataFrame(
        {
            "step": np.arange(len(actions), dtype=int),
            "time": frame["time"].to_numpy(dtype=float),
            **{
                f"action_{zone}": actions[:, index]
                for index, zone in enumerate(config.zones)
            },
        }
    )
    action_frame.to_csv(actions_path, index=False)
    return trajectory_path, actions_path


def _load_saved_trajectory(
    config: HydronicWave2Config,
    trajectory_path: Path,
    actions_path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(trajectory_path)
    actions = pd.read_csv(actions_path)
    frame["_actions"] = [
        json.dumps(row)
        for row in actions[
            [f"action_{zone}" for zone in config.zones]
        ].to_numpy(dtype=float).tolist()
    ]
    return frame


def _evaluation_metrics(
    config: HydronicWave2Config,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    actions = [json.loads(value) for value in frame["_actions"]]
    base = core.calculate_rollout_metrics(frame, actions, config)
    return {
        **base,
        "energy_kwh": float(frame["energy_step_kwh"].sum()),
        "c2_wave2": c2_trajectory_metrics(frame, config),
        "action_health_by_zone": v3.action_health_by_zone(frame, actions, config),
    }


def _run_evaluation_method(
    *,
    config: HydronicWave2Config,
    forecast_data: Mapping[str, Sequence[float]],
    results_dir: Path,
    label: str,
    model: Any | None,
    zero_policy: bool,
    is_validation: bool,
    phase: str,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    frame, _ = rollout_policy(
        config,
        forecast_data,
        model=model,
        zero_policy=zero_policy,
        is_validation=is_validation,
        phase=phase,
    )
    metrics = _evaluation_metrics(config, frame)
    trajectory, actions = _save_trajectory(config, results_dir, label, frame)
    return frame, metrics, trajectory, actions


def _record_completed_method(
    progress: dict[str, Any],
    path: Path,
    label: str,
    metrics: Mapping[str, Any],
    trajectory: Path,
    actions: Path,
) -> None:
    progress["methods"][label] = {
        "status": "complete",
        "completed_at": core.utc_now(),
        "trajectory": str(trajectory),
        "actions": str(actions),
        "trajectory_sha256": core.sha256_file(trajectory),
        "actions_sha256": core.sha256_file(actions),
        "rows": int(len(pd.read_csv(trajectory, usecols=["time"]))),
        "metrics": dict(metrics),
    }
    core.atomic_write_json(path, progress)


def _plot_final_validation(
    config: HydronicWave2Config,
    primary: pd.DataFrame,
    confirmation: Mapping[str, pd.DataFrame],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    hours = np.arange(len(primary)) * config.control_period / 3600.0
    axes[0].plot(hours, primary["cost"].cumsum(), label="Wave-2 PPO")
    axes[0].set_ylabel("Primary cumulative cost")
    for zone in config.zones:
        axes[1].plot(hours, primary[f"pmv_{zone}"], label=f"Wave-2 PMV {zone}")
    axes[1].axhspan(-0.5, 0.5, color="green", alpha=0.12)
    axes[1].set_ylabel("Primary PMV")
    for label, frame in confirmation.items():
        confirm_hours = np.arange(len(frame)) * config.control_period / 3600.0
        axes[2].plot(confirm_hours, frame["cost"].cumsum(), label=label)
    axes[2].set_ylabel("Confirmation cumulative cost")
    axes[2].set_xlabel("Time [h]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _load_v3_policies(
    suite: HydronicWave2Suite,
) -> tuple[Any, Any]:
    source = suite.v3_source_suite
    ppo_checkpoint = core.AtomicCheckpointManager(
        source.ppo.run_dir, source.ppo.config_hash
    )
    ppo_best = ppo_checkpoint.load_best_metadata()
    if ppo_best is None:
        raise RuntimeError("Immutable v3 PPO best checkpoint is missing")
    mappo_checkpoint = v3.AtomicTorchCheckpointManager(
        source.mappo.run_dir, source.mappo.config_hash
    )
    mappo_best = mappo_checkpoint.load_best()
    if mappo_best is None:
        raise RuntimeError("Immutable v3 MAPPO best checkpoint is missing")
    return (
        core.Phase15fPPO.load(ppo_best["model_path"], device="auto"),
        v3._load_mappo_policy(source.mappo, mappo_best["payload"]),
    )


def _numeric_metric_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "return",
        "cost",
        "energy_kwh",
        "zone_hours",
        "pmv_hours",
        "occupied_saturation",
        "occupancy_action_gap",
        "median_precooling_lead_hours",
    )
    summary: dict[str, Any] = {}
    for key in keys:
        values = np.asarray(
            [core.safe_float(report.get(key), float("nan")) for report in reports],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "values": values.tolist(),
            }
    return summary


def generate_multiseed_summary(suite: HydronicWave2Suite) -> dict[str, Any] | None:
    if suite.ppo.run_mode != "full":
        return None
    reports: list[dict[str, Any]] = []
    for seed in APPROVED_SEEDS:
        path = (
            suite.ppo.output_root
            / "full"
            / f"seed{seed}"
            / "final_validation"
            / "final_validation_report.json"
        )
        if not path.exists():
            return None
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("training_convergence", {}).get("converged"):
            return None
        reports.append(report)
    payload = {
        "schema": "phase1.5f-wave2-multiseed-summary-v1",
        "generated_at": core.utc_now(),
        "seeds": list(APPROVED_SEEDS),
        "all_configs_frozen": len(
            {report["cross_seed_scientific_hash"] for report in reports}
        )
        == 1,
        "primary_wave2_ppo": _numeric_metric_summary(
            [report["primary_metrics"]["wave2_ppo"] for report in reports]
        ),
    }
    if not payload["all_configs_frozen"]:
        raise RuntimeError("Wave-2 supplemental seed configurations are not identical")
    output = suite.ppo.output_root / "full" / "multiseed_summary.json"
    core.atomic_write_json(output, payload)
    return payload


def evaluate_final(
    suite: HydronicWave2Suite,
    *,
    force: bool | None = None,
) -> dict[str, Any]:
    run_preflight(suite, online=False)
    verify_v3_unchanged(suite)
    state, best = _require_ppo_finished(suite)
    force = suite.force_final_eval if force is None else bool(force)
    config = suite.final
    results_dir = suite.run_dir / "final_validation"
    results_dir.mkdir(parents=True, exist_ok=True)
    source_inventory = json.loads(
        (suite.run_dir / "v3_source_inventory.json").read_text(encoding="utf-8")
    )
    signature = {
        "schema": "phase1.5f-wave2-final-signature-v1",
        "config_hash": suite.ppo.config_hash,
        "best_manifest_sha256": core.sha256_file(
            Path(best["manifest_path"])
            if best.get("manifest_path")
            else core.AtomicCheckpointManager(
                suite.ppo.run_dir, suite.ppo.config_hash
            ).directory
            / best["pointer"]["manifest"]
        ),
        "v3_inventory_sha256": source_inventory["inventory_sha256"],
    }
    signature_hash = core.sha256_payload(signature)
    report_path = results_dir / "final_validation_report.json"
    progress_path = results_dir / "evaluation_manifest.json"
    if report_path.exists() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("signature_hash") == signature_hash:
            verify_v3_unchanged(suite)
            generate_multiseed_summary(suite)
            return report
        raise RuntimeError(
            "Existing Wave-2 final validation references a different best checkpoint"
        )
    if progress_path.exists() and not force:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("signature_hash") != signature_hash:
            raise RuntimeError("Incomplete Wave-2 evaluation has a different signature")
    else:
        progress = {
            "schema": "phase1.5f-wave2-evaluation-progress-v1",
            "signature": signature,
            "signature_hash": signature_hash,
            "started_at": core.utc_now(),
            "status": "running",
            "methods": {},
        }
        core.atomic_write_json(progress_path, progress)
    cleanup = {
        "final": core.cleanup_stale_testids(config),
        "confirmation": core.cleanup_stale_testids(suite.confirmation),
    }
    if any(
        row.get("status") == "live_owner_skipped"
        for rows in cleanup.values()
        for row in rows
    ):
        raise RuntimeError("A Wave-2 TestID still has a live owner")

    wave2_model = core.Phase15fPPO.load(best["model_path"], device="auto")
    primary_label = "wave2_ppo"
    existing = progress["methods"].get(primary_label, {})
    if existing.get("status") == "complete" and not force:
        primary_frame = _load_saved_trajectory(
            config, Path(existing["trajectory"]), Path(existing["actions"])
        )
        primary_metrics = existing["metrics"]
    else:
        progress["methods"][primary_label] = {
            "status": "running",
            "started_at": core.utc_now(),
        }
        core.atomic_write_json(progress_path, progress)
        forecast = Wave2ForecastProvider(config, extra_days=7)
        forecast.prefetch_all()
        primary_frame, primary_metrics, trajectory, actions = _run_evaluation_method(
            config=config,
            forecast_data=forecast.data,
            results_dir=results_dir,
            label=primary_label,
            model=wave2_model,
            zero_policy=False,
            is_validation=True,
            phase="primary_held_out_wave2_ppo",
        )
        _record_completed_method(
            progress,
            progress_path,
            primary_label,
            primary_metrics,
            trajectory,
            actions,
        )

    old_report_path = (
        suite.v3_source_suite.run_dir
        / "final_validation"
        / "final_validation_report.json"
    )
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    old_metrics = dict(old_report["metrics"])
    primary_comparison = {
        **old_metrics,
        primary_label: primary_metrics,
    }
    core.write_csv(
        results_dir / "primary_kpi_comparison.csv",
        [{"policy": label, **values} for label, values in primary_comparison.items()],
    )

    confirmation_frames: dict[str, pd.DataFrame] = {}
    confirmation_metrics: dict[str, Any] = {}
    confirmation_info = select_confirmation_day(suite)
    if suite.ppo.run_mode == "full" and suite.ppo.seed == 42:
        confirmation_config = dataclasses.replace(
            suite.confirmation,
            start_day=int(confirmation_info["start_day"]),
        )
        confirmation_forecast = Wave2ForecastProvider(confirmation_config)
        confirmation_forecast.prefetch_all()
        v3_ppo, v3_mappo = _load_v3_policies(suite)
        factories: dict[str, tuple[Any | None, bool]] = {
            "confirmation_rbc": (None, True),
            "confirmation_v3_ppo": (v3_ppo, False),
            "confirmation_v3_mappo": (v3_mappo, False),
            "confirmation_wave2_ppo": (wave2_model, False),
        }
        for label, (model, zero_policy) in factories.items():
            existing = progress["methods"].get(label, {})
            if existing.get("status") == "complete" and not force:
                frame = _load_saved_trajectory(
                    confirmation_config,
                    Path(existing["trajectory"]),
                    Path(existing["actions"]),
                )
                metrics = existing["metrics"]
            else:
                progress["methods"][label] = {
                    "status": "running",
                    "started_at": core.utc_now(),
                }
                core.atomic_write_json(progress_path, progress)
                frame, metrics, trajectory, actions = _run_evaluation_method(
                    config=confirmation_config,
                    forecast_data=confirmation_forecast.data,
                    results_dir=results_dir,
                    label=label,
                    model=model,
                    zero_policy=zero_policy,
                    is_validation=False,
                    phase=label,
                )
                _record_completed_method(
                    progress,
                    progress_path,
                    label,
                    metrics,
                    trajectory,
                    actions,
                )
            confirmation_frames[label] = frame
            confirmation_metrics[label] = metrics
        core.write_csv(
            results_dir / "confirmation_kpi_comparison.csv",
            [
                {"policy": label, **values}
                for label, values in confirmation_metrics.items()
            ],
        )

    _plot_final_validation(
        config,
        primary_frame,
        confirmation_frames,
        results_dir / "final_validation.png",
    )
    isolation = verify_v3_unchanged(suite)
    report = {
        "schema": "phase1.5f-wave2-final-validation-v1",
        "generated_at": core.utc_now(),
        "signature": signature,
        "signature_hash": signature_hash,
        "seed": suite.ppo.seed,
        "cross_seed_scientific_hash": cross_seed_scientific_hash(suite.ppo),
        "primary_validation_start_day": PRIMARY_VALIDATION_DAY,
        "confirmation_week": confirmation_info,
        "checkpoint_selection": "deterministic training-week return only",
        "heldout_used_for_selection": False,
        "primary_metrics": primary_comparison,
        "confirmation_metrics": confirmation_metrics,
        "training_convergence": state.get("convergence", {}),
        "failure_stop": state.get("failure_stop", {}),
        "best": {
            "epoch": int(best["pointer"]["epoch"]),
            "global_step": int(best["pointer"]["num_timesteps"]),
        },
        "legacy_c2_is_preserved": True,
        "v3_original_status_is_not_modified": True,
        "v3_isolation": isolation,
        "stale_cleanup": cleanup,
        "wandb": {
            "run_id": state.get("wandb_run_id"),
            "url": state.get("wandb_url"),
        },
    }
    core.atomic_write_json(report_path, report)
    progress["status"] = "complete"
    progress["completed_at"] = core.utc_now()
    progress["report"] = str(report_path)
    core.atomic_write_json(progress_path, progress)
    lines = [
        "# Phase 1.5f Wave-2 PPO final validation",
        "",
        "The immutable v3 results were read as comparators and were not rewritten.",
        "The primary held-out week was not used for checkpoint selection.",
        f"C2 legacy: {state.get('convergence', {}).get('C2_legacy', False)}",
        f"C2 Wave-2: {state.get('convergence', {}).get('C2_wave2', False)}",
        f"Converged: {state.get('convergence', {}).get('converged', False)}",
    ]
    (results_dir / "final_validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    generate_multiseed_summary(suite)
    return report


__all__ = [
    "APPROVED_SEEDS",
    "HydronicWave2Config",
    "HydronicWave2Suite",
    "build_hydronic_wave2_suite",
    "c2_trajectory_metrics",
    "cross_seed_scientific_hash",
    "evaluate_convergence",
    "evaluate_failure_stop",
    "evaluate_final",
    "reconstruct_reward_components",
    "run_failure_diagnosis",
    "run_preflight",
    "train_ppo",
    "verify_v3_unchanged",
]
