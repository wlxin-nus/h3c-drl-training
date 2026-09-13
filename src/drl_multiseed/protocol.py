"""Physical BOPTEST payload and forecast contracts used during training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .occupancy import resolve_missing_occupancy_values


def setpoint_from_action(residual_base: str, occupancy: float, action: float) -> float:
    """Map one policy sample to the frozen physical cooling setpoint in degrees C."""

    if not math.isfinite(float(occupancy)) or not math.isfinite(float(action)):
        raise ValueError("occupancy and action must be finite")
    clipped_action = max(-1.0, min(1.0, float(action)))
    if residual_base == "fixed_25":
        base = 25.0
    elif residual_base == "occupancy_25_30":
        base = 25.0 if float(occupancy) > 0 else 30.0
    else:
        raise ValueError(f"unsupported residual base: {residual_base}")
    return max(20.0, min(30.0, base + 5.0 * clipped_action))


def forecast_points(profile: Mapping[str, Any]) -> list[str]:
    global_inputs = profile["global_inputs"]
    points = [
        global_inputs["outdoor_temperature"],
        global_inputs["solar_irradiance"],
        global_inputs["electricity_price"],
    ]
    for zone in profile["zones"].values():
        point = zone["occupancy_forecast"]
        if point not in points:
            points.append(point)
    return points


def validate_forecast(
    forecast: Mapping[str, Sequence[object]], points: Sequence[str], minimum_length: int
) -> None:
    missing = sorted(set(points) - set(forecast))
    if missing:
        raise ValueError(f"forecast bundle is missing configured points: {missing}")
    for point in points:
        values = forecast[point]
        if len(values) < minimum_length:
            raise ValueError(
                f"forecast point {point} has {len(values)} values; "
                f"at least {minimum_length} are required"
            )
        for index, value in enumerate(values):
            finite = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            )
            if not finite:
                raise ValueError(f"forecast point {point} at index {index} is not finite")


def resolve_forecast_missing_occupancy(
    profile: Mapping[str, Any],
    forecast: Mapping[str, Sequence[float | None]],
    points: Sequence[str],
    minimum_length: int,
    *,
    forecast_phase: str,
    start_time_seconds: int,
    step_seconds: int,
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    missing = sorted(set(points) - set(forecast))
    if missing:
        raise ValueError(f"forecast bundle is missing configured points: {missing}")
    candidate: dict[str, list[object]] = {point: list(forecast[point]) for point in points}
    events: list[dict[str, Any]] = []
    occupancy_points = dict.fromkeys(
        zone["occupancy_forecast"] for zone in profile["zones"].values()
    )
    for point in occupancy_points:
        values, point_events = resolve_missing_occupancy_values(
            profile["occupancy"],
            forecast[point],
            start_time_seconds=start_time_seconds,
            step_seconds=step_seconds,
        )
        candidate[point] = list(values)
        events.extend(
            {
                "phase": "occupancy_forecast_missing_value_resolution",
                "forecast_phase": forecast_phase,
                "point": point,
                **event,
            }
            for event in point_events
        )
    validate_forecast(candidate, points, minimum_length)
    return {point: [float(value) for value in candidate[point]] for point in points}, events


def control_input(profile: Mapping[str, Any], setpoints_c: Mapping[str, float]) -> dict[str, float]:
    zones = profile["zones"]
    if set(setpoints_c) != set(zones):
        raise ValueError("setpoints must cover exactly the configured zones")
    controls = {key: float(value) for key, value in profile["static_controls"].items()}
    for zone, value in setpoints_c.items():
        controls[zones[zone]["cooling_setpoint_actuator"]] = float(value) + 273.15
    return controls


def site_power(profile: Mapping[str, Any], state: Mapping[str, Any]) -> float:
    total = 0.0
    for point in profile["global_inputs"]["power_meters"]:
        if point not in state:
            raise ValueError(f"physical state is missing power meter: {point}")
        value = state[point]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"power meter {point} is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"power meter {point} is non-finite")
        total += number
    return total


def zone_temperature_c(profile: Mapping[str, Any], state: Mapping[str, Any], zone: str) -> float:
    point = profile["zones"][zone]["temperature_sensor"]
    if point not in state:
        raise ValueError(f"physical state is missing zone temperature: {point}")
    value = float(state[point]) - 273.15
    if not math.isfinite(value):
        raise ValueError("zone temperature is non-finite")
    return value
