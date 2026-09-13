"""Pure physical metrics shared by H3C and independent baselines."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def compute_physical_metrics(
    performance_rows: Sequence[Mapping[str, Any]],
    zone_outcome_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute controller-independent metrics from canonical evaluation rows."""
    total_cost = sum(float(row["step_cost"]) for row in performance_rows)
    energy_kwh = sum(float(row["total_power_w"]) * 0.25 / 1000.0 for row in performance_rows)
    reward = sum(float(row["step_reward"]) for row in performance_rows)
    discomfort_zone_hours = 0.0
    discomfort_pmv_hours = 0.0
    occupied_peak_absolute_pmv = 0.0
    setpoints: dict[str, list[tuple[int, float]]] = defaultdict(list)
    comfort_state: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for row in zone_outcome_rows:
        occupancy = float(row["effective_occupancy"])
        pmv = float(row["pmv"])
        if occupancy > 0:
            absolute = abs(pmv)
            occupied_peak_absolute_pmv = max(occupied_peak_absolute_pmv, absolute)
            if absolute > 0.5:
                discomfort_zone_hours += 0.25
                discomfort_pmv_hours += (absolute - 0.5) * 0.25
            comfort_state[str(row["zone"])].append((int(row["step"]), absolute <= 0.5))
        setpoints[str(row["zone"])].append((int(row["step"]), float(row["final_setpoint_c"])))

    total_variation = 0.0
    reversals = 0
    comfort_crossings = 0
    for setpoint_values in setpoints.values():
        ordered = [value for _, value in sorted(setpoint_values)]
        deltas = [right - left for left, right in zip(ordered, ordered[1:], strict=False)]
        total_variation += sum(abs(delta) for delta in deltas)
        directions = [1 if delta > 0 else -1 for delta in deltas if abs(delta) > 1e-12]
        reversals += sum(
            left != right for left, right in zip(directions, directions[1:], strict=False)
        )
    for comfort_values in comfort_state.values():
        ordered = [value for _, value in sorted(comfort_values)]
        comfort_crossings += sum(
            left != right for left, right in zip(ordered, ordered[1:], strict=False)
        )
    return {
        "physical": {
            "total_cost": total_cost,
            "energy_kwh": energy_kwh,
            "reward": reward,
            "discomfort_zone_hours": discomfort_zone_hours,
            "discomfort_pmv_hours": discomfort_pmv_hours,
            "occupied_peak_absolute_pmv": occupied_peak_absolute_pmv,
            "evaluation_steps": len(performance_rows),
        },
        "setpoint_dynamics": {
            "total_variation_c": total_variation,
            "direction_reversals": reversals,
            "occupied_comfort_band_crossings": comfort_crossings,
        },
    }
