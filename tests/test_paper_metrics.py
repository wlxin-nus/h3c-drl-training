from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.recompute_paper_metrics import SeriesRow, calculate_metrics, load_timeseries
from scripts.verify_reference_results import verify


def test_hand_calculated_two_zone_metrics() -> None:
    rows = [
        SeriesRow("zone1", 0, 25.0, 0.4, 1.0),
        SeriesRow("zone1", 1, 24.0, 0.6, 1.0),
        SeriesRow("zone1", 2, 26.0, 0.7, 1.0),
        SeriesRow("zone2", 0, 25.0, -0.4, 1.0),
        SeriesRow("zone2", 1, 25.0, -0.4, 1.0),
        SeriesRow("zone2", 2, 24.0, -0.8, 1.0),
    ]
    metrics = calculate_metrics(rows)
    assert metrics == pytest.approx(
        {
            "discomfort_zone_hours": 0.75,
            "discomfort_pmv_hours": 0.15,
            "occupied_peak_absolute_pmv": 0.8,
            "total_variation_c": 4.0,
            "direction_reversals": 1,
            "occupied_comfort_band_crossings": 2,
        }
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pmv", "nan", "must be finite"),
        ("cooling_setpoint_c", "31", "outside"),
    ],
)
def test_invalid_released_rows_fail_closed(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    path = tmp_path / "invalid.csv"
    row = {
        "model_id": "bad",
        "case": "SZ_Air",
        "algorithm": "PPO",
        "seed": "42",
        "zone": "zone1",
        "step": "0",
        "cooling_setpoint_c": "25",
        "pmv": "0",
        "effective_occupancy": "0",
    }
    row[field] = value
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match=message):
        load_timeseries(path)


def test_released_reference_results_are_integrity_and_metric_closed() -> None:
    result = verify()
    assert result["models"] == 15
    assert result["historical_configurations"] == 15
    assert result["historical_preflight_models"] == 5
    assert result["recomputed_metrics"] == 6
