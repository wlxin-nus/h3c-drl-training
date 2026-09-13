"""Bounded completed-history working memory."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

WORKING_MEMORY_HOURS = (1, 2, 3)
FUTURE_WEATHER_FIELDS = {
    "weather_next_steps",
    "outdoor_temp_change_next_1h_c",
    "solar_irr_max_next_1h_w_m2",
    "solar_irr_mean_next_1h_w_m2",
}


def _rounded(value: Any) -> Any:
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else value
    if isinstance(value, Mapping):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rounded(item) for item in value)
    return copy.deepcopy(value)


def without_future_weather(value: Any) -> Any:
    """Recursively remove every future-weather channel from history-facing data."""
    if isinstance(value, Mapping):
        return {
            key: without_future_weather(item)
            for key, item in value.items()
            if key not in FUTURE_WEATHER_FIELDS
        }
    if isinstance(value, list):
        return [without_future_weather(item) for item in value]
    if isinstance(value, tuple):
        return tuple(without_future_weather(item) for item in value)
    return copy.deepcopy(value)


def completed_summary_frame(
    *,
    hour: int,
    zone: str,
    step_rows: Sequence[Mapping[str, Any]],
    program_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the final W deterministic completed-hour view for O/R memory."""
    rows = [dict(row) for row in step_rows]
    if len(rows) != 4 or any(row.get("zone") != zone for row in rows):
        raise ValueError("a completed memory frame requires four aligned zone results")
    if [row.get("step") for row in rows] != list(range(hour * 4, hour * 4 + 4)):
        raise ValueError("memory frame steps do not cover one complete hour")
    decision = without_future_weather(program_decision)
    residuals = [float(row["interpreter"]["residual"]) for row in rows]
    pmv_values = [float(row["outcome"]["pmv"]) for row in rows]
    occupancies = [float(row["outcome"]["effective_occupancy"]) for row in rows]
    zone_hours = [
        0.25 if occupancy > 0 and abs(pmv) > 0.5 else 0.0
        for pmv, occupancy in zip(pmv_values, occupancies, strict=True)
    ]
    pmv_hours = [
        max(0.0, abs(pmv) - 0.5) * 0.25 if occupancy > 0 else 0.0
        for pmv, occupancy in zip(pmv_values, occupancies, strict=True)
    ]
    patch = decision.get("patch", {})
    applied_patch_ids = (
        [int(decision["step"])]
        if decision.get("status") == "accepted" and patch.get("op") != "no_change"
        else []
    )
    return cast(
        dict[str, Any],
        _rounded(
            {
                "hour": hour,
                "zone": zone,
                "program_version": int(decision["current_program_version"]),
                "applied_patch_ids": applied_patch_ids,
                "mean_residual_c": sum(residuals) / len(residuals),
                "mean_setpoint_c": sum(float(row["final_setpoint_c"]) for row in rows) / len(rows),
                "mean_pmv": sum(pmv_values) / len(pmv_values),
                "cost": sum(float(row["outcome"]["cost"]) for row in rows),
                "zone_h": sum(zone_hours),
                "pmv_h": sum(pmv_hours),
                "complete": True,
            }
        ),
    )


def completed_executor_records(
    *,
    hour: int,
    zone: str,
    step_rows: Sequence[Mapping[str, Any]],
    program_decision: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build four completed, future-weather-free Executor records for one hour."""
    rows = [dict(row) for row in step_rows]
    if len(rows) != 4 or any(row.get("zone") != zone for row in rows):
        raise ValueError("Executor memory requires four aligned zone results")
    expected_steps = list(range(hour * 4, hour * 4 + 4))
    if [row.get("step") for row in rows] != expected_steps:
        raise ValueError("Executor memory steps do not cover one complete hour")
    decision = cast(dict[str, Any], without_future_weather(program_decision))
    patch = decision.get("patch")
    if not isinstance(patch, Mapping):
        raise ValueError("Executor memory requires the settled program proposal")
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        observation = row["observation"]
        assurance = row["action_assurance"]
        outcome = row["outcome"]
        called = index == 0
        proposal = (
            copy.deepcopy(dict(patch))
            if called
            else {"status": "not_called", "program_source_step": hour * 4}
        )
        rejection = decision.get("rejection")
        validation: dict[str, Any] = {
            "accepted": bool(
                called and decision.get("status") == "accepted" and patch.get("op") != "no_change"
            ),
            "rejected": bool(
                called and decision.get("status") in {"rejected", "model_output_rejected"}
            ),
            "matched_rule": row["interpreter"].get("matched_rule"),
            "shield": {
                "branch": (
                    "comfort_recovery"
                    if assurance["comfort_recovery_triggered"]
                    else (
                        "setpoint_rate_limit"
                        if assurance["setpoint_rate_limit_triggered"]
                        else (
                            "actuator_bounds"
                            if assurance["actuator_bounds_triggered"]
                            else "normal"
                        )
                    )
                ),
                "proposed": assurance["interpreter_setpoint"],
                "setpoint": assurance["final_setpoint"],
                "actuator_limit_applied": bool(assurance["actuator_bounds_triggered"]),
                "rate_limit_applied": bool(assurance["setpoint_rate_limit_triggered"]),
                "comfort_interlock_applied": bool(assurance["comfort_recovery_triggered"]),
            },
        }
        if called and isinstance(rejection, Mapping) and rejection.get("code") is not None:
            validation["rejection_code"] = rejection["code"]
        pmv = float(outcome["pmv"])
        occupancy = float(outcome["effective_occupancy"])
        measured = {
            "zone_temp": outcome["zone_temperature_c"],
            "pmv": pmv,
            "cost": outcome["cost"],
            "zone_h": 0.25 if occupancy > 0 and abs(pmv) > 0.5 else 0.0,
            "pmv_h": max(0.0, abs(pmv) - 0.5) * 0.25 if occupancy > 0 else 0.0,
            "occupancy": occupancy,
        }
        record = {
            "time": {"step": row["step"], "hour": hour},
            "scope": {"zone": zone},
            "observation": {
                "current_occupancy": observation["current_occupancy"],
                "last_occupancy": observation["last_occupancy"],
                "last_pmv": observation["last_pmv"],
                "last_setpoint": observation["last_setpoint"],
                "zone_temp_before_action_c": observation["zone_temperature_c"],
                "next_hour_occupancy": observation["next_hour_occupancy"],
            },
            "proposal": proposal,
            "validation": validation,
            "outcome": {
                "residual": row["interpreter"]["residual"],
                "final_setpoint": row["final_setpoint_c"],
                "measured": measured,
            },
            "program": {"version": int(decision["current_program_version"])},
            "executor_called": called,
        }
        cleaned = without_future_weather(record)
        records.append(cast(dict[str, Any], _rounded(cleaned)))
    return records


def reflector_results_view(
    current_results: Sequence[Mapping[str, Any]],
    program_decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project current results without creating a future-weather memory side channel."""
    frames = [
        {key: copy.deepcopy(value) for key, value in frame.items() if key != "complete"}
        for frame in current_results
    ]
    applied: list[dict[str, Any]] = []
    for decision in program_decisions:
        patch = decision.get("patch")
        if (
            decision.get("status") != "accepted"
            or not isinstance(patch, Mapping)
            or patch.get("op") == "no_change"
        ):
            continue
        row: dict[str, Any] = {
            "zone": decision["zone"],
            "patch_ref": decision["step"],
            "operation": patch["op"],
        }
        if patch.get("param") is not None:
            row["param"] = patch["param"]
        if patch.get("to") is not None:
            row["to"] = patch["to"]
        applied.append(row)
    return cast(
        dict[str, Any],
        without_future_weather({"frames": frames, "applied_patches": applied or None}),
    )


def select_executor_records(
    records: Sequence[Mapping[str, Any]],
    *,
    current_step: int,
    zone: str,
    working_memory_hours: int,
) -> list[dict[str, Any]]:
    """Select exactly 4*k prior completed records for one zone or omit the block."""
    if working_memory_hours not in WORKING_MEMORY_HOURS:
        raise ValueError("working_memory_hours must be one, two, or three")
    selected = [
        copy.deepcopy(dict(record))
        for record in records
        if record.get("scope", {}).get("zone") == zone
        and isinstance(record.get("time", {}).get("step"), int)
        and int(record["time"]["step"]) < current_step
    ]
    selected.sort(key=lambda record: int(record["time"]["step"]))
    selected = selected[-4 * working_memory_hours :]
    expected = list(range(current_step - 4 * working_memory_hours, current_step))
    if [int(record["time"]["step"]) for record in selected] != expected:
        return []
    if any(FUTURE_WEATHER_FIELDS & _all_keys(record) for record in selected):
        raise ValueError("future weather entered Executor working memory")
    exposed = [
        {key: value for key, value in frame.items() if key != "complete"} for frame in selected
    ]
    return cast(list[dict[str, Any]], _rounded(exposed))


def recent_outcome_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    current_step: int,
    zone: str,
    working_memory_hours: int,
) -> dict[str, Any] | None:
    """Summarize the same exact 4*k Executor window without adding a direction."""
    selected = select_executor_records(
        records,
        current_step=current_step,
        zone=zone,
        working_memory_hours=working_memory_hours,
    )
    if not selected:
        return None
    measured = [record["outcome"]["measured"] for record in selected]
    costs = [float(outcome["cost"]) for outcome in measured]
    pmv = [float(outcome["pmv"]) for outcome in measured]
    return {
        "covered_steps": len(selected),
        "mean_cost": round(sum(costs) / len(costs), 4),
        "first_last_cost_difference": round(costs[-1] - costs[0], 4),
        "mean_pmv": round(sum(pmv) / len(pmv), 4),
        "pmv_range": [round(min(pmv), 4), round(max(pmv), 4)],
        "accepted_edits": sum(record["validation"].get("accepted") is True for record in selected),
    }


def select_completed_frames(
    frames: Sequence[Mapping[str, Any]],
    *,
    current_hour: int,
    zone: str | None,
    working_memory_hours: int,
    zones: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Select complete prior hourly frames, ordered oldest to newest."""
    if working_memory_hours not in WORKING_MEMORY_HOURS:
        raise ValueError("working_memory_hours must be one, two, or three")
    first_hour = current_hour - working_memory_hours
    selected: list[dict[str, Any]] = []
    for frame in frames:
        hour = frame.get("hour")
        if not isinstance(hour, int) or hour >= current_hour:
            if isinstance(hour, int) and hour >= current_hour:
                continue
            raise ValueError("working memory frame has invalid hour")
        if first_hour <= hour < current_hour and (zone is None or frame.get("zone") == zone):
            if frame.get("complete") is not True:
                return []
            selected.append(copy.deepcopy(dict(frame)))
    selected.sort(key=lambda item: (item["hour"], str(item.get("zone", ""))))
    expected_hours = set(range(first_hour, current_hour))
    if {frame["hour"] for frame in selected} != expected_hours:
        return []
    if zones is not None:
        expected_zones = {zone} if zone is not None else set(zones)
        for hour in expected_hours:
            if {frame.get("zone") for frame in selected if frame["hour"] == hour} != expected_zones:
                return []
    if any(FUTURE_WEATHER_FIELDS & set(_all_keys(frame)) for frame in selected):
        raise ValueError("future weather entered working memory")
    return cast(list[dict[str, Any]], _rounded(selected))


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | set().union(
            *(_all_keys(item) for item in value.values()), set()
        )
    if isinstance(value, (list, tuple)):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()
