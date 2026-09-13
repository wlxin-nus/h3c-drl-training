from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import TASKS
from .io import atomic_json, utc_now


def aggregate_results(output_root: Path, mode: str = "full") -> dict[str, Any]:
    output_root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted((output_root / mode).glob("seed*")):
        for task in TASKS:
            path = seed_dir / task / "formal_evaluation" / "metrics.json"
            manifest_path = seed_dir / task / "run_manifest.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {}
            )
            row = {
                "task": task, "seed": int(payload["seed"]),
                "best_epoch": manifest.get("best_epoch"),
                "stop_reason": manifest.get("status"),
                "actual_training_steps": manifest.get("global_step"),
                **payload["metrics"],
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    target = output_root / "aggregate"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "per_seed_results.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    metric_names = [
        "return", "cost", "energy_kwh", "occupied_zone_hours", "pmv_hours",
        "occupied_pmv_violation_rate", "deep_pmv_violation_rate",
        "occupied_action_saturation", "action_occ_unocc_gap",
    ]
    for task, group in frame.groupby("task") if not frame.empty else ():
        summary: dict[str, Any] = {"task": task, "n": len(group)}
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            summary[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else None
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(target / "mean_std_results.csv", index=False)
    fairness = [
        {
            "case": spec.case_key, "task": spec.key,
            "forecast_horizon": "t0..t+60 min at 15-min resolution",
            "forecast_variables": "outdoor temperature; solar; dynamic price; zone occupancy",
            "constraints": "same H3C profile static payload and 20-30 C cooling setpoint bounds",
            "observation_contract": f"H3C frozen adapter ({spec.observation_dim} global dims)",
            "formal_test_days": spec.episode_steps / 96,
        }
        for spec in TASKS.values()
    ]
    pd.DataFrame(fairness).to_csv(target / "information_fairness_matrix.csv", index=False)
    report = {
        "schema": "h3c-drl-multiseed-refined-aggregate-v1", "generated_at": utc_now(),
        "mode": mode, "rows": len(rows), "tasks": sorted(frame.task.unique().tolist()) if not frame.empty else [],
        "statistical_note": "Use clean seeds 42/1337/2026 for the primary mean+/-std once all are available.",
        "evaluation_limitation": "The registered 5/7-day windows do not fully answer the request for longer test periods.",
    }
    atomic_json(target / "aggregate_manifest.json", report)
    return report
