"""Baseline metrics projected onto the shared H3C physical owner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from h3c.outputs.physical_metrics import compute_physical_metrics
from h3c_baselines.outputs.artifacts import PERFORMANCE_COLUMNS


def _json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        candidate = json.loads(line)
        if not isinstance(candidate, dict):
            raise ValueError(f"{path.name} contains a non-object row")
        rows.append(candidate)
    return rows


def _performance(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != PERFORMANCE_COLUMNS:
            raise ValueError("baseline performance header is invalid")
        rows = [dict(row) for row in reader]
    if any(any(value is None for value in row.values()) for row in rows):
        raise ValueError("baseline performance row is incomplete")
    return rows


def compute_baseline_metrics(run_dir: Path) -> dict[str, Any]:
    performance = _performance(run_dir / "performance.csv")
    actions = _json_rows(run_dir / "actions.jsonl")
    physical = compute_physical_metrics(
        performance,
        [
            {
                "zone": row["zone"],
                "step": row["step"],
                "final_setpoint_c": row["final_setpoint_c"],
                "effective_occupancy": row["outcome"]["effective_occupancy"],
                "pmv": row["outcome"]["pmv"],
            }
            for row in actions
        ],
    )
    diagnostics = _json_rows(run_dir / "controller_diagnostics.jsonl")
    fallback_count = sum(row.get("method_degraded") is True for row in diagnostics)
    return {
        "metrics_schema": "h3c_baseline_metrics",
        "schema_version": 1,
        **physical,
        "controller": {
            "decision_count": len(diagnostics),
            "fallback_count": fallback_count,
            "method_degraded": fallback_count > 0,
        },
    }
