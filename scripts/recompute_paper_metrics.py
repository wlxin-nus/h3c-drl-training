"""Recompute paper-level comfort and setpoint metrics from released DRL time series."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "reference_results" / "paper_2026"
DEFAULT_INPUT = RESULT_ROOT / "drl_evaluation_timeseries.csv"
DEFAULT_EXPECTED = RESULT_ROOT / "drl_run_metrics.csv"
CASE_CONTRACT = {
    "SZ_Air": (672, 1),
    "MZ_Hydro": (480, 2),
    "MZ_Air": (672, 5),
}
RECOMPUTED_METRICS = (
    "discomfort_zone_hours",
    "discomfort_pmv_hours",
    "occupied_peak_absolute_pmv",
    "total_variation_c",
    "direction_reversals",
    "occupied_comfort_band_crossings",
)
REQUIRED_COLUMNS = {
    "model_id",
    "case",
    "algorithm",
    "seed",
    "zone",
    "step",
    "cooling_setpoint_c",
    "pmv",
    "effective_occupancy",
}


@dataclass(frozen=True)
class SeriesRow:
    zone: str
    step: int
    cooling_setpoint_c: float
    pmv: float
    effective_occupancy: float


@dataclass(frozen=True)
class ModelSeries:
    model_id: str
    case: str
    algorithm: str
    seed: int
    rows: tuple[SeriesRow, ...]


def _finite_float(value: str, field: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{field} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _integer(value: str, field: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(f"{field} is not an integer: {value!r}") from error
    return number


def load_timeseries(path: Path) -> list[ModelSeries]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - fields)
        if missing:
            raise ValueError(f"time-series columns are missing: {missing}")
        source_rows = list(reader)

    metadata: dict[str, tuple[str, str, int]] = {}
    grouped: dict[str, list[SeriesRow]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for source in source_rows:
        model_id = source["model_id"].strip()
        case = source["case"].strip()
        algorithm = source["algorithm"].strip()
        seed = _integer(source["seed"], "seed")
        zone = source["zone"].strip()
        step = _integer(source["step"], "step")
        if not model_id or not zone:
            raise ValueError("model_id and zone must be non-empty")
        if case not in CASE_CONTRACT:
            raise ValueError(f"unregistered case: {case}")
        candidate_metadata = (case, algorithm, seed)
        if model_id in metadata and metadata[model_id] != candidate_metadata:
            raise ValueError(f"metadata changes within model {model_id}")
        metadata[model_id] = candidate_metadata
        key = (model_id, zone, step)
        if key in seen:
            raise ValueError(f"duplicate model-zone-step row: {key}")
        seen.add(key)

        setpoint = _finite_float(source["cooling_setpoint_c"], "cooling_setpoint_c")
        if not 20.0 <= setpoint <= 30.0:
            raise ValueError(f"cooling_setpoint_c is outside [20, 30] for {key}")
        occupancy = _finite_float(source["effective_occupancy"], "effective_occupancy")
        if occupancy < 0:
            raise ValueError(f"effective_occupancy is negative for {key}")
        grouped[model_id].append(
            SeriesRow(
                zone=zone,
                step=step,
                cooling_setpoint_c=setpoint,
                pmv=_finite_float(source["pmv"], "pmv"),
                effective_occupancy=occupancy,
            )
        )

    models: list[ModelSeries] = []
    for model_id in sorted(grouped):
        case, algorithm, seed = metadata[model_id]
        expected_steps, expected_zones = CASE_CONTRACT[case]
        rows = grouped[model_id]
        zones = sorted({row.zone for row in rows})
        if len(zones) != expected_zones:
            raise ValueError(f"{model_id} has {len(zones)} zones; expected {expected_zones}")
        for zone in zones:
            steps = sorted(row.step for row in rows if row.zone == zone)
            if steps != list(range(expected_steps)):
                raise ValueError(
                    f"{model_id}/{zone} does not contain steps 0..{expected_steps - 1}"
                )
        models.append(
            ModelSeries(
                model_id=model_id,
                case=case,
                algorithm=algorithm,
                seed=seed,
                rows=tuple(sorted(rows, key=lambda row: (row.zone, row.step))),
            )
        )
    return models


def calculate_metrics(rows: Iterable[SeriesRow]) -> dict[str, float | int]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("at least one time-series row is required")

    occupied_rows = [row for row in materialized if row.effective_occupancy > 0]
    zone_hours = 0.25 * sum(abs(row.pmv) > 0.5 for row in occupied_rows)
    pmv_hours = 0.25 * sum(max(0.0, abs(row.pmv) - 0.5) for row in occupied_rows)
    peak = max((abs(row.pmv) for row in occupied_rows), default=0.0)

    by_zone: dict[str, list[SeriesRow]] = defaultdict(list)
    for row in materialized:
        by_zone[row.zone].append(row)
    total_variation = 0.0
    reversals = 0
    crossings = 0
    for zone_rows in by_zone.values():
        ordered = sorted(zone_rows, key=lambda row: row.step)
        changes = [
            right.cooling_setpoint_c - left.cooling_setpoint_c for left, right in pairwise(ordered)
        ]
        total_variation += sum(abs(change) for change in changes)
        signs = [change > 0 for change in changes if abs(change) > 1e-12]
        reversals += sum(left != right for left, right in pairwise(signs))
        comfort_states = [abs(row.pmv) <= 0.5 for row in ordered if row.effective_occupancy > 0]
        crossings += sum(left != right for left, right in pairwise(comfort_states))

    return {
        "discomfort_zone_hours": float(zone_hours),
        "discomfort_pmv_hours": float(pmv_hours),
        "occupied_peak_absolute_pmv": float(peak),
        "total_variation_c": float(total_variation),
        "direction_reversals": reversals,
        "occupied_comfort_band_crossings": crossings,
    }


def recompute(path: Path) -> list[dict[str, str | float | int]]:
    results: list[dict[str, str | float | int]] = []
    for model in load_timeseries(path):
        results.append(
            {
                "model_id": model.model_id,
                "case": model.case,
                "algorithm": model.algorithm,
                "seed": model.seed,
                **calculate_metrics(model.rows),
            }
        )
    return results


def compare_expected(actual_rows: list[dict[str, str | float | int]], expected_path: Path) -> float:
    with expected_path.open("r", encoding="utf-8-sig", newline="") as stream:
        expected_rows = list(csv.DictReader(stream))
    actual = {str(row["model_id"]): row for row in actual_rows}
    expected = {row["model_id"]: row for row in expected_rows}
    if set(actual) != set(expected):
        raise RuntimeError("model IDs differ between the time series and expected run metrics")

    maximum_error = 0.0
    failures: list[str] = []
    for model_id in sorted(actual):
        current = actual[model_id]
        reference = expected[model_id]
        for field in ("case", "algorithm"):
            if str(current[field]) != reference[field]:
                failures.append(f"{model_id}: {field} differs")
        if int(current["seed"]) != int(reference["seed"]):
            failures.append(f"{model_id}: seed differs")
        for metric in RECOMPUTED_METRICS:
            observed = float(current[metric])
            target = float(reference[metric])
            error = abs(observed - target)
            maximum_error = max(maximum_error, error)
            if not math.isclose(observed, target, rel_tol=1e-10, abs_tol=1e-8):
                failures.append(f"{model_id}: {metric} differs ({observed:.12g} != {target:.12g})")
    if failures:
        raise RuntimeError("paper-metric comparison failed:\n- " + "\n- ".join(failures))
    return maximum_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    args = parser.parse_args()
    calculated = recompute(args.input)
    maximum_error = compare_expected(calculated, args.expected)
    print(
        f"Verified {len(calculated)} DRL models across {len(RECOMPUTED_METRICS)} "
        f"recomputed metrics; maximum absolute difference={maximum_error:.3g}."
    )


if __name__ == "__main__":
    main()
