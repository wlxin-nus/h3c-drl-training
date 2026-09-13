from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PROTOCOL_VERSION, TASKS
from .io import atomic_json, utc_now


def aggregate_results(output_root: Path, mode: str = "full") -> dict[str, Any]:
    output_root = Path(output_root)
    rows: list[dict[str, Any]] = []
    identities: dict[str, set[tuple[str, str, str, str]]] = {task: set() for task in TASKS}
    for seed_dir in sorted((output_root / mode).glob("seed*")):
        for task in TASKS:
            path = seed_dir / task / "formal_evaluation" / "metrics.json"
            manifest_path = seed_dir / task / "run_manifest.json"
            if not path.exists():
                continue
            if not manifest_path.is_file():
                raise RuntimeError(f"Formal evaluation has no run manifest: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            seed = int(payload["seed"])
            spec = TASKS[task]
            expected = {
                "task": task,
                "seed": seed,
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": spec.protocol_hash(),
                "config_hash": spec.scientific_hash(seed),
            }
            for source_name, source in (("evaluation", payload), ("manifest", manifest)):
                for field, expected_value in expected.items():
                    if source.get(field) != expected_value:
                        raise RuntimeError(
                            f"{source_name} identity mismatch for {task} seed {seed}: {field}"
                        )
            identity_fields = (
                "protocol_version",
                "protocol_hash",
                "code_fingerprint",
                "boptest_version",
            )
            identity = tuple(str(payload.get(field) or "") for field in identity_fields)
            if any(not value for value in identity):
                raise RuntimeError(f"Incomplete aggregation identity for {task} seed {seed}")
            if any(
                str(manifest.get(field) or "") != value
                for field, value in zip(identity_fields, identity, strict=True)
            ):
                raise RuntimeError(
                    f"Evaluation and manifest identities differ for {task} seed {seed}"
                )
            identities[task].add(identity)
            row = {
                "task": task,
                "seed": seed,
                "best_epoch": manifest.get("best_epoch"),
                "stop_reason": manifest.get("status"),
                "actual_training_steps": manifest.get("global_step"),
                **dict(zip(identity_fields, identity, strict=True)),
                **payload["metrics"],
            }
            rows.append(row)
    incompatible = {task: sorted(values) for task, values in identities.items() if len(values) > 1}
    if incompatible:
        raise RuntimeError(
            f"Refusing to aggregate mixed protocol/source/BOPTEST identities: {incompatible}"
        )
    frame = pd.DataFrame(rows)
    target = output_root / "aggregate"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "per_seed_results.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    metric_names = [
        "return",
        "cost",
        "energy_kwh",
        "occupied_zone_hours",
        "pmv_hours",
        "occupied_pmv_violation_rate",
        "deep_pmv_violation_rate",
        "occupied_action_saturation",
        "action_occ_unocc_gap",
    ]
    for task, group in frame.groupby("task") if not frame.empty else ():
        summary: dict[str, Any] = {"task": task, "n": len(group)}
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            summary[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else None
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(target / "mean_std_results.csv", index=False)
    information_contract = [
        {
            "scope": "DRL baselines only; H3C is documented in the companion method archive",
            "case": spec.case_key,
            "task": spec.key,
            "forecast_horizon": "t0..t+60 min at 15-min resolution",
            "forecast_variables": "outdoor temperature; solar; dynamic price; zone occupancy",
            "constraints": (
                "frozen static payload; occupied 20-30 C and unoccupied 25-30 C setpoints"
                if spec.case_key == "mz_hydro"
                else "frozen static payload; 20-30 C cooling setpoints"
            ),
            "observation_contract": f"frozen DRL contract ({spec.observation_dim} global dims)",
            "formal_test_days": spec.episode_steps / 96,
        }
        for spec in TASKS.values()
    ]
    pd.DataFrame(information_contract).to_csv(target / "drl_information_contract.csv", index=False)
    report = {
        "schema": "h3c-drl-aggregate-v1",
        "generated_at": utc_now(),
        "mode": mode,
        "rows": len(rows),
        "tasks": sorted(frame.task.unique().tolist()) if not frame.empty else [],
        "identity_by_task": {
            task: dict(
                zip(
                    ("protocol_version", "protocol_hash", "code_fingerprint", "boptest_version"),
                    next(iter(values)),
                    strict=True,
                )
            )
            for task, values in identities.items()
            if values
        },
        "statistical_note": "Use clean seeds 42/1337/2026 for the primary mean+/-std once all are available.",
        "evaluation_limitation": "The registered 5/7-day windows do not fully answer the request for longer test periods.",
        "information_contract_scope": "DRL baselines only; cross-method fairness requires the companion H3C method archive.",
    }
    atomic_json(target / "aggregate_manifest.json", report)
    return report
