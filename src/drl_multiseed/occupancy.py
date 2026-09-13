"""Case-owned occupancy interpretation and missing-forecast resolution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

DAY_SECONDS = 86_400


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
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("missing occupancy resolution is invalid") from exc
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


def effective_count(policy: Mapping[str, Any], time_seconds: float, raw_count: float) -> float:
    """Return the occupancy count used by comfort and observation calculations."""

    if isinstance(raw_count, bool) or not math.isfinite(float(raw_count)):
        raise ValueError("raw occupancy must be finite")
    if float(raw_count) <= 0:
        return 0.0
    mode = policy.get("mode")
    if mode == "raw_count_positive":
        return float(raw_count)
    if mode != "official_hvac_window" or policy.get("interval") != "half_open":
        raise ValueError("occupancy policy is invalid")
    minute = (float(time_seconds) % DAY_SECONDS) / 60.0
    start = int(policy["window_start_minute"])
    end = int(policy["window_end_minute"])
    active = start <= minute < end if start < end else minute >= start or minute < end
    return float(raw_count) if active else 0.0
