"""Strict, configuration-owned case profiles."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

PROFILE_FIELDS = {
    "profile_schema",
    "schema_version",
    "profile",
    "testcase",
    "evaluation_start_day",
    "control_step_seconds",
    "forecast_steps",
    "zones",
    "global_inputs",
    "static_controls",
    "occupancy",
    "protocol",
    "comfort",
    "objective",
    "performance",
    "graph",
    "program",
}
ZONE_FIELDS = {
    "description",
    "temperature_sensor",
    "cooling_setpoint_actuator",
    "occupancy_forecast",
}
GLOBAL_FIELDS = {
    "outdoor_temperature",
    "solar_irradiance",
    "electricity_price",
    "power_meters",
}
PROTOCOL_FIELDS = {
    "initialization_mode",
    "internal_warmup_days",
    "formal_evaluation_days",
    "initial_setpoint_c",
}
MISSING_OCCUPANCY_RESOLUTION_FIELDS = {
    "calendar_origin_utc",
    "occupied_weekdays",
    "occupied_window_start_minute",
    "occupied_window_end_minute",
    "interval",
    "holiday_month_days",
    "documented_nonoccupancy_rule",
    "documented_occupancy_rule",
    "source",
}
WEEKDAY_NAMES = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}


class ProfileError(ValueError):
    pass


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"profile cannot be read: {path}") from error
    if not isinstance(value, dict):
        raise ProfileError("profile root must be an object")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ProfileError(f"{name} must be finite")
    return number


def _validate_missing_occupancy_resolution(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != MISSING_OCCUPANCY_RESOLUTION_FIELDS:
        raise ProfileError("missing occupancy resolution fields are invalid")
    try:
        origin = datetime.fromisoformat(value["calendar_origin_utc"])
    except (TypeError, ValueError) as error:
        raise ProfileError("missing occupancy calendar origin is invalid") from error
    utc_offset = origin.utcoffset()
    if origin.tzinfo is None or utc_offset is None or utc_offset.total_seconds() != 0:
        raise ProfileError("missing occupancy calendar origin must be UTC")
    weekdays = value["occupied_weekdays"]
    if (
        not isinstance(weekdays, list)
        or not weekdays
        or any(not isinstance(day, str) for day in weekdays)
        or len(set(weekdays)) != len(weekdays)
        or any(day not in WEEKDAY_NAMES for day in weekdays)
    ):
        raise ProfileError("missing occupancy weekdays are invalid")
    start = value["occupied_window_start_minute"]
    end = value["occupied_window_end_minute"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not 0 <= start < end <= 1440
        or value["interval"] != "half_open"
    ):
        raise ProfileError("missing occupancy window is invalid")
    holidays = value["holiday_month_days"]
    if (
        not isinstance(holidays, list)
        or any(not isinstance(holiday, str) for holiday in holidays)
        or len(set(holidays)) != len(holidays)
    ):
        raise ProfileError("missing occupancy holiday dates are invalid")
    try:
        for holiday in holidays:
            if datetime.strptime(f"2000-{holiday}", "%Y-%m-%d").strftime("%m-%d") != holiday:
                raise ValueError
    except ValueError as error:
        raise ProfileError("missing occupancy holiday dates are invalid") from error
    if (
        value["documented_nonoccupancy_rule"] != "zero"
        or value["documented_occupancy_rule"] != "previous_step"
    ):
        raise ProfileError("missing occupancy rules are invalid")
    if not isinstance(value["source"], str) or not value["source"]:
        raise ProfileError("missing occupancy documentation source is required")


def validate_profile(
    raw: Mapping[str, Any],
    *,
    root: Path | None = None,
    allow_missing_graph: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != PROFILE_FIELDS:
        raise ProfileError("case profile fields do not match the schema")
    if raw["profile_schema"] != "h3c_case_profile" or raw["schema_version"] != 1:
        raise ProfileError("unsupported case profile schema")
    for field in ("profile", "testcase", "graph", "program"):
        if not isinstance(raw[field], str) or not raw[field]:
            raise ProfileError(f"{field} must be a non-empty string")
    if (
        isinstance(raw["evaluation_start_day"], bool)
        or not isinstance(raw["evaluation_start_day"], int)
        or raw["evaluation_start_day"] < 14
    ):
        raise ProfileError("evaluation start day must leave room for conditioning")
    if raw["control_step_seconds"] != 900 or raw["forecast_steps"] != 4:
        raise ProfileError(
            "control step and forecast horizon are frozen at 900 seconds and four steps"
        )

    zones = raw["zones"]
    if not isinstance(zones, Mapping) or not zones:
        raise ProfileError("profile must contain at least one zone")
    for zone, mapping in zones.items():
        if not isinstance(zone, str) or not zone or not isinstance(mapping, Mapping):
            raise ProfileError("zone identity is invalid")
        if set(mapping) != ZONE_FIELDS or any(
            not isinstance(mapping[field], str) or not mapping[field] for field in ZONE_FIELDS
        ):
            raise ProfileError("zone mapping fields are invalid")

    global_inputs = raw["global_inputs"]
    if not isinstance(global_inputs, Mapping) or set(global_inputs) != GLOBAL_FIELDS:
        raise ProfileError("global input mapping fields are invalid")
    if any(
        not isinstance(global_inputs[field], str) or not global_inputs[field]
        for field in GLOBAL_FIELDS - {"power_meters"}
    ):
        raise ProfileError("global forecast point names are invalid")
    meters = global_inputs["power_meters"]
    if (
        not isinstance(meters, list)
        or not meters
        or any(not isinstance(point, str) or not point for point in meters)
    ):
        raise ProfileError("power meter mapping is invalid")

    controls = raw["static_controls"]
    if not isinstance(controls, Mapping) or not controls:
        raise ProfileError("static controls must be a non-empty object")
    for key, value in controls.items():
        if not isinstance(key, str) or not key:
            raise ProfileError("static control name is invalid")
        _finite(value, key)

    occupancy = raw["occupancy"]
    if not isinstance(occupancy, Mapping):
        raise ProfileError("occupancy policy must be an object")
    mode = occupancy.get("mode")
    if mode == "raw_count_positive":
        if set(occupancy) not in (
            {"mode", "source"},
            {"mode", "source", "missing_value_resolution"},
        ):
            raise ProfileError("raw occupancy policy fields are invalid")
        if "missing_value_resolution" in occupancy:
            _validate_missing_occupancy_resolution(occupancy["missing_value_resolution"])
    elif mode == "official_hvac_window":
        if set(occupancy) != {
            "mode",
            "window_start_minute",
            "window_end_minute",
            "interval",
            "source",
        }:
            raise ProfileError("windowed occupancy policy fields are invalid")
        if occupancy["interval"] != "half_open" or not (
            0 <= occupancy["window_start_minute"] < 1440
            and 0 <= occupancy["window_end_minute"] < 1440
            and occupancy["window_start_minute"] != occupancy["window_end_minute"]
        ):
            raise ProfileError("occupancy window is invalid")
    else:
        raise ProfileError("occupancy mode is unsupported")
    if not isinstance(occupancy.get("source"), str) or not occupancy["source"]:
        raise ProfileError("occupancy source is required")

    protocol = raw["protocol"]
    if not isinstance(protocol, Mapping) or set(protocol) != PROTOCOL_FIELDS:
        raise ProfileError("physical protocol fields are invalid")
    if (
        protocol["initialization_mode"] != "evaluation_start_internal_warmup"
        or protocol["internal_warmup_days"] != 7
        or protocol["formal_evaluation_days"] not in (5, 7)
        or _finite(protocol["initial_setpoint_c"], "initial setpoint") != 25.0
    ):
        raise ProfileError(
            "physical protocol must initialize at the evaluation start with a seven-day "
            "internal warm-up, a supported five- or seven-day evaluation, and a 25 C "
            "initial controller state"
        )

    comfort = raw["comfort"]
    if not isinstance(comfort, Mapping) or set(comfort) != {
        "dynamic_clothing",
        "metabolic_rate",
        "relative_humidity_percent",
        "air_velocity_m_s",
        "winter_clothing_insulation",
        "summer_clothing_insulation",
        "clothing_transition_low_c",
        "clothing_transition_high_c",
    }:
        raise ProfileError("comfort configuration fields are invalid")
    for key, value in comfort.items():
        if key == "dynamic_clothing":
            if not isinstance(value, bool):
                raise ProfileError("dynamic clothing flag must be boolean")
        else:
            _finite(value, key)
    objective = raw["objective"]
    if not isinstance(objective, Mapping) or set(objective) != {
        "energy_weight",
        "comfort_weight",
        "smoothness_weight",
        "energy_scale",
        "comfort_scale",
        "smoothness_scale",
    }:
        raise ProfileError("objective configuration fields are invalid")
    for key, value in objective.items():
        if _finite(value, key) <= 0:
            raise ProfileError("objective values must be positive")
    performance = raw["performance"]
    if (
        not isinstance(performance, Mapping)
        or set(performance) != {"maximum_power_w"}
        or _finite(performance["maximum_power_w"], "maximum power") <= 0
    ):
        raise ProfileError("performance configuration is invalid")

    resolved_root = repository_root() if root is None else root
    for field in ("graph", "program"):
        path = (resolved_root / str(raw[field])).resolve()
        missing_allowed = field == "graph" and allow_missing_graph
        if not path.is_relative_to(resolved_root.resolve()) or (
            not missing_allowed and not path.is_file()
        ):
            raise ProfileError(f"{field} path is missing or outside the repository")
    return copy.deepcopy(dict(raw))


def profiles(config_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    root = repository_root()
    directory = root / "configs" / "cases" if config_dir is None else config_dir
    loaded: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        profile = validate_profile(_read_object(path), root=root)
        name = str(profile["profile"])
        if name in loaded:
            raise ProfileError(f"duplicate profile name: {name}")
        loaded[name] = profile
    if not loaded:
        raise ProfileError("no case profiles were found")
    return loaded


def load_profile(name: str) -> dict[str, Any]:
    loaded = profiles()
    try:
        return loaded[name]
    except KeyError as error:
        raise ProfileError(f"unknown case profile: {name}") from error
