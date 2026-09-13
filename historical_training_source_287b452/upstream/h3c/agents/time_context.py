"""Deterministic clock-only semantics for Agent-facing H3C contexts."""

from __future__ import annotations

from typing import Any

CONTROL_PERIOD_SECONDS = 15 * 60
COORDINATION_PERIOD_SECONDS = 60 * 60
SECONDS_PER_DAY = 24 * 60 * 60


def _require_aligned_time(time_seconds: int) -> None:
    if isinstance(time_seconds, bool) or not isinstance(time_seconds, int):
        raise ValueError("context time must be an integer number of seconds")
    if time_seconds < 0 or time_seconds % CONTROL_PERIOD_SECONDS:
        raise ValueError("context time must be a nonnegative 15-minute boundary")


def clock_time(time_seconds: int, *, reference_seconds: int) -> str:
    """Render one clock time, marking a day crossing without exposing a date."""
    _require_aligned_time(time_seconds)
    _require_aligned_time(reference_seconds)
    day_delta = time_seconds // SECONDS_PER_DAY - reference_seconds // SECONDS_PER_DAY
    minute_of_day = (time_seconds % SECONDS_PER_DAY) // 60
    clock = f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"
    if day_delta == 0:
        return clock
    if day_delta == 1:
        return f"next day {clock}"
    if day_delta == -1:
        return f"previous day {clock}"
    direction = "later" if day_delta > 0 else "earlier"
    return f"{abs(day_delta)} days {direction} {clock}"


def interval_label(start_seconds: int, end_seconds: int, *, reference_seconds: int) -> str:
    """Render a half-open interval using only clock semantics."""
    if end_seconds <= start_seconds:
        raise ValueError("clock interval must have positive duration")
    return (
        f"[{clock_time(start_seconds, reference_seconds=reference_seconds)}, "
        f"{clock_time(end_seconds, reference_seconds=reference_seconds)})"
    )


def action_and_outcome_times(start_seconds: int) -> tuple[list[str], list[str]]:
    """Return four action clocks and their exactly 15-minute-later outcome clocks."""
    _require_aligned_time(start_seconds)
    actions = [start_seconds + index * CONTROL_PERIOD_SECONDS for index in range(4)]
    outcomes = [value + CONTROL_PERIOD_SECONDS for value in actions]
    return (
        [clock_time(value, reference_seconds=start_seconds) for value in actions],
        [clock_time(value, reference_seconds=start_seconds) for value in outcomes],
    )


def decision_window(time_seconds: int) -> dict[str, Any]:
    """Build the sole Agent-facing owner for a one-hour decision window."""
    _require_aligned_time(time_seconds)
    action_times, outcome_times = action_and_outcome_times(time_seconds)
    return {
        "current_time": clock_time(time_seconds, reference_seconds=time_seconds),
        "control_interval": interval_label(
            time_seconds,
            time_seconds + COORDINATION_PERIOD_SECONDS,
            reference_seconds=time_seconds,
        ),
        "action_times": action_times,
        "forecast_outcome_times": outcome_times,
        "control_period": "15 min",
        "coordination_period": "60 min",
    }


def completed_interval(start_seconds: int) -> dict[str, Any]:
    """Build clock semantics for one completed four-action control interval."""
    _require_aligned_time(start_seconds)
    action_times, outcome_times = action_and_outcome_times(start_seconds)
    return {
        "interval": interval_label(
            start_seconds,
            start_seconds + COORDINATION_PERIOD_SECONDS,
            reference_seconds=start_seconds,
        ),
        "action_times": action_times,
        "outcome_times": outcome_times,
        "control_period": "15 min",
    }
