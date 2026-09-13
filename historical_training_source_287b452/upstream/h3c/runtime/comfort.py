"""Frozen comfort, reward, and KPI calculations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pythermalcomfort.models import pmv_ppd_iso  # type: ignore[import-untyped]

COMFORT_BAND = 0.5
STEP_HOURS = 0.25
HEADROOM_SEARCH_SPAN_C = 20.0


class ComfortModel:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.clothing_insulation = float(config["summer_clothing_insulation"])
        self.previous_day = -1

    def update_clothing(self, time_seconds: float, outdoor_daily_mean_c: float) -> None:
        if not self.config["dynamic_clothing"]:
            return
        day = int(time_seconds // 86400)
        if day == self.previous_day:
            return
        low = float(self.config["clothing_transition_low_c"])
        high = float(self.config["clothing_transition_high_c"])
        winter = float(self.config["winter_clothing_insulation"])
        summer = float(self.config["summer_clothing_insulation"])
        if outdoor_daily_mean_c < low:
            self.clothing_insulation = winter
        elif outdoor_daily_mean_c > high:
            self.clothing_insulation = summer
        else:
            ratio = (outdoor_daily_mean_c - low) / (high - low)
            self.clothing_insulation = winter + ratio * (summer - winter)
        self.previous_day = day

    def pmv(self, air_temperature_c: float) -> float:
        result = pmv_ppd_iso(
            tdb=float(air_temperature_c),
            tr=float(air_temperature_c),
            vr=float(self.config["air_velocity_m_s"]),
            rh=float(self.config["relative_humidity_percent"]),
            met=float(self.config["metabolic_rate"]),
            clo=self.clothing_insulation,
            limit_inputs=False,
        )
        value = result["pmv"] if isinstance(result, dict) else result.pmv
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("comfort model returned a non-finite PMV")
        return max(-3.0, min(3.0, number))


def _headroom_root(
    pmv_of_temperature: Any, temperature_c: float, target: float, direction: int
) -> float | None:
    lower = temperature_c
    upper = temperature_c + direction * HEADROOM_SEARCH_SPAN_C
    lower_value = float(pmv_of_temperature(lower))
    upper_value = float(pmv_of_temperature(upper))
    if (lower_value - target) * (upper_value - target) > 0:
        return None
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        midpoint_value = float(pmv_of_temperature(midpoint))
        if (lower_value - target) * (midpoint_value - target) <= 0:
            upper = midpoint
        else:
            lower = midpoint
            lower_value = midpoint_value
        if abs(upper - lower) < 0.01:
            break
    return 0.5 * (lower + upper)


def comfort_headroom(pmv_of_temperature: Any, temperature_c: float) -> dict[str, float] | None:
    """Return measured zone-temperature distance to each score-band edge."""
    try:
        temperature = float(temperature_c)
        current = float(pmv_of_temperature(temperature))
    except (TypeError, ValueError):
        return None
    result: dict[str, float] = {}
    if current >= COMFORT_BAND:
        result["warmer_c"] = 0.0
    else:
        warmer = _headroom_root(pmv_of_temperature, temperature, COMFORT_BAND, 1)
        if warmer is not None:
            result["warmer_c"] = round(max(0.0, warmer - temperature), 2)
    if current <= -COMFORT_BAND:
        result["cooler_c"] = 0.0
    else:
        cooler = _headroom_root(pmv_of_temperature, temperature, -COMFORT_BAND, -1)
        if cooler is not None:
            result["cooler_c"] = round(max(0.0, temperature - cooler), 2)
    return result or None


def step_reward(
    *,
    cost: float,
    pmv: Sequence[float],
    occupancy: Sequence[float],
    setpoints_c: Sequence[float],
    previous_setpoints_c: Sequence[float],
    objective: Mapping[str, Any],
) -> float:
    """Return the frozen scalar reward without changing its historical arithmetic."""
    return float(
        step_reward_breakdown(
            cost=cost,
            pmv=pmv,
            occupancy=occupancy,
            setpoints_c=setpoints_c,
            previous_setpoints_c=previous_setpoints_c,
            objective=objective,
        )["reward"]
    )


def step_reward_breakdown(
    *,
    cost: float,
    pmv: Sequence[float],
    occupancy: Sequence[float],
    setpoints_c: Sequence[float],
    previous_setpoints_c: Sequence[float],
    objective: Mapping[str, Any],
    zone_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return the frozen reward and its exact, retrospective penalty decomposition."""
    if not (len(pmv) == len(occupancy) == len(setpoints_c) == len(previous_setpoints_c) > 0):
        raise ValueError("reward inputs must have one aligned value per zone")
    zone_count = len(pmv)
    names = (
        list(zone_names)
        if zone_names is not None
        else [f"zone_{index}" for index in range(zone_count)]
    )
    if len(names) != zone_count or len(set(names)) != zone_count or any(not name for name in names):
        raise ValueError("reward zone names must be unique and aligned with the reward inputs")

    comfort_excess_squared = [
        max(0.0, abs(float(value)) - COMFORT_BAND) ** 2
        for value, count in zip(pmv, occupancy, strict=True)
        if float(count) > 0
    ]
    # Keep one entry per configured zone, including an explicit zero for unoccupied zones.
    comfort_excess_by_zone = [
        max(0.0, abs(float(value)) - COMFORT_BAND) ** 2 if float(count) > 0 else 0.0
        for value, count in zip(pmv, occupancy, strict=True)
    ]
    smoothness_by_zone = [
        abs(float(current) - float(previous))
        for current, previous in zip(setpoints_c, previous_setpoints_c, strict=True)
    ]
    comfort_penalty = sum(comfort_excess_squared)
    smoothness = sum(smoothness_by_zone)
    site_energy_penalty = (
        float(objective["energy_weight"])
        * float(objective["energy_scale"])
        * float(cost)
        / zone_count
    )
    # Preserve the frozen expression's operation order before deriving explanatory shares.
    site_comfort_penalty = (
        float(objective["comfort_weight"])
        * float(objective["comfort_scale"])
        * comfort_penalty
        / zone_count
    )
    site_smoothness_penalty = (
        float(objective["smoothness_weight"])
        * float(objective["smoothness_scale"])
        * smoothness
        / zone_count
    )
    reward = -(site_energy_penalty + site_comfort_penalty + site_smoothness_penalty)
    comfort_factor = (
        float(objective["comfort_weight"]) * float(objective["comfort_scale"]) / zone_count
    )
    smoothness_factor = (
        float(objective["smoothness_weight"]) * float(objective["smoothness_scale"]) / zone_count
    )
    return {
        "reward": reward,
        "site_energy_penalty": site_energy_penalty,
        "site_comfort_penalty": site_comfort_penalty,
        "site_smoothness_penalty": site_smoothness_penalty,
        "zone_comfort_penalty_contributions": {
            name: comfort_factor * value
            for name, value in zip(names, comfort_excess_by_zone, strict=True)
        },
        "zone_smoothness_penalty_contributions": {
            name: smoothness_factor * value
            for name, value in zip(names, smoothness_by_zone, strict=True)
        },
    }


class MetricsAccumulator:
    def __init__(self) -> None:
        self.cost = 0.0
        self.energy_kwh = 0.0
        self.reward = 0.0
        self.discomfort_zone_hours = 0.0
        self.discomfort_pmv_hours = 0.0
        self.rows = 0

    def add(
        self,
        *,
        cost: float,
        power_w: float,
        reward: float,
        pmv: Sequence[float],
        occupancy: Sequence[float],
    ) -> None:
        self.cost += float(cost)
        self.energy_kwh += float(power_w) * STEP_HOURS / 1000.0
        self.reward += float(reward)
        for value, count in zip(pmv, occupancy, strict=True):
            excess = max(0.0, abs(float(value)) - COMFORT_BAND)
            if float(count) > 0 and excess > 0:
                self.discomfort_zone_hours += STEP_HOURS
                self.discomfort_pmv_hours += excess * STEP_HOURS
        self.rows += 1

    def resolved(self) -> dict[str, Any]:
        return {
            "total_cost": self.cost,
            "energy_kwh": self.energy_kwh,
            "reward": self.reward,
            "discomfort_zone_hours": self.discomfort_zone_hours,
            "discomfort_pmv_hours": self.discomfort_pmv_hours,
            "rows": self.rows,
        }
