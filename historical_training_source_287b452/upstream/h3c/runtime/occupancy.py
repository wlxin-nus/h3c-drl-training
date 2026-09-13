"""Configuration-owned effective occupancy and hourly request routing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

DAY_SECONDS = 24 * 60 * 60
STEP_SECONDS = 15 * 60


def documented_occupancy_active(policy: Mapping[str, Any], time_seconds: int) -> bool:
    resolution = policy.get("missing_value_resolution")
    if not isinstance(resolution, Mapping):
        raise ValueError("missing occupancy resolution is not configured")
    try:
        origin = datetime.fromisoformat(str(resolution["calendar_origin_utc"]))
        current = origin + timedelta(seconds=int(time_seconds))
        weekdays = resolution["occupied_weekdays"]
        holidays = resolution["holiday_month_days"]
        start = int(resolution["occupied_window_start_minute"])
        end = int(resolution["occupied_window_end_minute"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("missing occupancy resolution is invalid") from error
    if resolution.get("interval") != "half_open":
        raise ValueError("missing occupancy resolution is invalid")
    if current.strftime("%m-%d") in holidays or current.strftime("%A") not in weekdays:
        return False
    minute = current.hour * 60 + current.minute
    return start <= minute < end


def resolve_missing_occupancy_values(
    policy: Mapping[str, Any],
    values: Sequence[float | None],
    *,
    start_time_seconds: int,
    step_seconds: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    resolved: list[float] = []
    events: list[dict[str, Any]] = []
    resolution = policy.get("missing_value_resolution")
    for index, value in enumerate(values):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"occupancy forecast at index {index} is not numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"occupancy forecast at index {index} is not finite")
            resolved.append(number)
            continue
        if not isinstance(resolution, Mapping):
            raise ValueError(f"occupancy forecast at index {index} is missing without a policy")
        time_seconds = start_time_seconds + index * step_seconds
        occupied = documented_occupancy_active(policy, time_seconds)
        preceding_value: float | None = None
        if occupied:
            if not resolved or not math.isfinite(resolved[-1]):
                raise ValueError(
                    f"occupied occupancy forecast at index {index} has no finite previous step"
                )
            preceding_value = resolved[-1]
            replacement = preceding_value
            rule = "documented_occupancy_previous_step"
        else:
            replacement = 0.0
            rule = "documented_nonoccupancy_zero"
        resolved.append(replacement)
        events.append(
            {
                "forecast_index": index,
                "time_seconds": time_seconds,
                "source_value": None,
                "documented_occupied": occupied,
                "resolution_rule": rule,
                "preceding_value": preceding_value,
                "resolved_value": replacement,
                "documentation_source": resolution["source"],
            }
        )
    return resolved, events


def verify_missing_occupancy_resolution_evidence(
    profile: Mapping[str, Any],
    evaluation_hours: int,
    recorded_count: Any,
    events: Sequence[Mapping[str, Any]],
    *,
    evaluation_start_seconds: int | None = None,
) -> bool:
    """Recompute every permitted missing-occupancy resolution event."""

    if (
        isinstance(recorded_count, bool)
        or not isinstance(recorded_count, int)
        or recorded_count != len(events)
    ):
        return False
    zones = profile["zones"]
    policy = profile["occupancy"]
    resolution = policy.get("missing_value_resolution")
    if resolution is None and events:
        return False
    event_keys = {
        "phase",
        "forecast_phase",
        "point",
        "forecast_index",
        "time_seconds",
        "source_value",
        "documented_occupied",
        "resolution_rule",
        "preceding_value",
        "resolved_value",
        "documentation_source",
    }
    points = {mapping["occupancy_forecast"] for mapping in zones.values()}
    seen: set[tuple[str, str, int]] = set()
    maximum_index = int(evaluation_hours) * 4 + 96
    evaluation_start = (
        int(profile["evaluation_start_day"]) * DAY_SECONDS
        if evaluation_start_seconds is None
        else evaluation_start_seconds
    )
    for raw_event in events:
        event = dict(raw_event)
        if set(event) != event_keys or event["point"] not in points:
            return False
        phase = event["forecast_phase"]
        index = event["forecast_index"]
        if (
            phase != "evaluation"
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index > maximum_index
        ):
            return False
        identity = (phase, event["point"], index)
        if identity in seen:
            return False
        seen.add(identity)
        time_seconds = evaluation_start + index * STEP_SECONDS
        if (
            event["time_seconds"] != time_seconds
            or event["source_value"] is not None
            or not isinstance(event["documented_occupied"], bool)
            or not isinstance(resolution, Mapping)
            or event["documentation_source"] != resolution["source"]
        ):
            return False
        occupied = documented_occupancy_active(policy, time_seconds)
        if event["documented_occupied"] != occupied:
            return False
        if occupied:
            preceding = event["preceding_value"]
            if (
                event["resolution_rule"] != "documented_occupancy_previous_step"
                or isinstance(preceding, bool)
                or not isinstance(preceding, (int, float))
                or not math.isfinite(float(preceding))
                or event["resolved_value"] != preceding
            ):
                return False
        elif (
            event["resolution_rule"] != "documented_nonoccupancy_zero"
            or event["preceding_value"] is not None
            or event["resolved_value"] != 0.0
        ):
            return False
    return True


def effective_count(policy: Mapping[str, Any], time_seconds: float, raw_count: float) -> float:
    if isinstance(raw_count, bool) or not math.isfinite(float(raw_count)):
        raise ValueError("raw occupancy must be finite")
    if float(raw_count) <= 0:
        return 0.0
    if policy.get("mode") == "raw_count_positive":
        return float(raw_count)
    if (
        set(policy)
        != {
            "mode",
            "window_start_minute",
            "window_end_minute",
            "interval",
            "source",
        }
        or policy["mode"] != "official_hvac_window"
        or policy["interval"] != "half_open"
    ):
        raise ValueError("occupancy policy is invalid")
    minute = (float(time_seconds) % DAY_SECONDS) / 60.0
    start = int(policy["window_start_minute"])
    end = int(policy["window_end_minute"])
    active = start <= minute < end if start < end else minute >= start or minute < end
    return float(raw_count) if active else 0.0


def effective_forecast(
    policy: Mapping[str, Any], time_seconds: float, raw_counts: Sequence[float]
) -> list[float]:
    return [
        effective_count(policy, time_seconds + index * STEP_SECONDS, value)
        for index, value in enumerate(raw_counts)
    ]


def hourly_route(
    hour: int,
    current_occupancy: Mapping[str, float],
    next_hour_occupancy: Mapping[str, float],
) -> dict[str, Any]:
    if not current_occupancy or set(current_occupancy) != set(next_hour_occupancy):
        raise ValueError("hourly occupancy route requires the same non-empty zone set")
    current = any(float(value) > 0 for value in current_occupancy.values())
    forecast = any(float(value) > 0 for value in next_hour_occupancy.values())
    triggers = []
    if current:
        triggers.append("current_occupancy")
    if forecast:
        triggers.append("forecast_occupancy")
    return {
        "hour": int(hour),
        "thinking_mode": "low" if triggers else "disabled",
        "route_triggers": triggers,
        "current_occupied": current,
        "next_hour_occupied": forecast,
    }
