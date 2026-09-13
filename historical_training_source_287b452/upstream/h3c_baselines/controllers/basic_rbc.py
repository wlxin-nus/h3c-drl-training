"""Minimal occupancy-scheduled rule-based controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def basic_rbc_setpoints(zones: Sequence[str], occupancy: Mapping[str, float]) -> dict[str, float]:
    if set(zones) != set(occupancy):
        raise ValueError("basic RBC occupancy must cover exactly the configured zones")
    return {zone: 25.0 if float(occupancy[zone]) > 0 else 30.0 for zone in zones}
