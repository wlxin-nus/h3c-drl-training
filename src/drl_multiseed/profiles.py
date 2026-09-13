"""Load and validate the three BOPTEST case profiles used for DRL training."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

CASE_FILES = {
    "sz_air": "sz_air.json",
    "SZ_Air": "sz_air.json",
    "mz_air": "mz_air.json",
    "MZ_Air": "mz_air.json",
    "mz_hydro": "mz_hydro.json",
    "MZ_Hydro": "mz_hydro.json",
}


class ProfileError(ValueError):
    """Raised when a case profile is incomplete or internally inconsistent."""


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProfileError(f"{field} must be finite")
    return result


def validate_profile(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = {
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
    }
    if set(raw) != required:
        raise ProfileError(
            f"case profile fields differ from the DRL schema: {sorted(set(raw) ^ required)}"
        )
    if raw["profile_schema"] != "h3c_case_profile" or raw["schema_version"] != 1:
        raise ProfileError("unsupported case profile schema")
    if raw["control_step_seconds"] != 900 or raw["forecast_steps"] != 4:
        raise ProfileError("the protocol requires a 15-minute step and four future steps")
    if not isinstance(raw["testcase"], str) or not raw["testcase"]:
        raise ProfileError("testcase must be a non-empty string")
    if not isinstance(raw["evaluation_start_day"], int) or raw["evaluation_start_day"] < 14:
        raise ProfileError("evaluation_start_day must leave room for warm-up")

    zones = raw["zones"]
    if not isinstance(zones, Mapping) or not zones:
        raise ProfileError("at least one zone is required")
    zone_fields = {
        "description",
        "temperature_sensor",
        "cooling_setpoint_actuator",
        "occupancy_forecast",
    }
    for zone, values in zones.items():
        if not isinstance(zone, str) or not isinstance(values, Mapping):
            raise ProfileError("zone definitions are invalid")
        if set(values) != zone_fields or any(
            not isinstance(values[field], str) or not values[field] for field in zone_fields
        ):
            raise ProfileError(f"zone mapping is invalid: {zone}")

    global_inputs = raw["global_inputs"]
    global_fields = {
        "outdoor_temperature",
        "solar_irradiance",
        "electricity_price",
        "power_meters",
    }
    if not isinstance(global_inputs, Mapping) or set(global_inputs) != global_fields:
        raise ProfileError("global input mapping is invalid")
    if not isinstance(global_inputs["power_meters"], list) or not global_inputs["power_meters"]:
        raise ProfileError("at least one power meter is required")

    controls = raw["static_controls"]
    if not isinstance(controls, Mapping) or not controls:
        raise ProfileError("static controls are required")
    for name, value in controls.items():
        if not isinstance(name, str) or not name:
            raise ProfileError("static control names must be non-empty")
        _finite(value, name)

    occupancy = raw["occupancy"]
    if not isinstance(occupancy, Mapping) or occupancy.get("mode") not in {
        "raw_count_positive",
        "official_hvac_window",
    }:
        raise ProfileError("occupancy policy is invalid")
    if occupancy["mode"] == "official_hvac_window":
        start = occupancy.get("window_start_minute")
        end = occupancy.get("window_end_minute")
        if (
            occupancy.get("interval") != "half_open"
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= 1440
        ):
            raise ProfileError("official occupancy window is invalid")

    protocol = raw["protocol"]
    if not isinstance(protocol, Mapping) or (
        protocol.get("initialization_mode") != "evaluation_start_internal_warmup"
        or protocol.get("internal_warmup_days") != 7
        or protocol.get("formal_evaluation_days") not in {5, 7}
        or _finite(protocol.get("initial_setpoint_c"), "initial_setpoint_c") != 25.0
    ):
        raise ProfileError("physical initialization protocol is invalid")

    comfort = raw["comfort"]
    expected_comfort = {
        "dynamic_clothing",
        "metabolic_rate",
        "relative_humidity_percent",
        "air_velocity_m_s",
        "winter_clothing_insulation",
        "summer_clothing_insulation",
        "clothing_transition_low_c",
        "clothing_transition_high_c",
    }
    if not isinstance(comfort, Mapping) or set(comfort) != expected_comfort:
        raise ProfileError("comfort configuration is invalid")
    for field, value in comfort.items():
        if field == "dynamic_clothing":
            if not isinstance(value, bool):
                raise ProfileError("dynamic_clothing must be boolean")
        else:
            _finite(value, field)

    objective = raw["objective"]
    expected_objective = {
        "energy_weight",
        "comfort_weight",
        "smoothness_weight",
        "energy_scale",
        "comfort_scale",
        "smoothness_scale",
    }
    if not isinstance(objective, Mapping) or set(objective) != expected_objective:
        raise ProfileError("objective configuration is invalid")
    for field, value in objective.items():
        if _finite(value, field) <= 0:
            raise ProfileError(f"objective field must be positive: {field}")
    performance = raw["performance"]
    if (
        not isinstance(performance, Mapping)
        or set(performance) != {"maximum_power_w"}
        or _finite(performance["maximum_power_w"], "maximum_power_w") <= 0
    ):
        raise ProfileError("performance configuration is invalid")
    return copy.deepcopy(dict(raw))


def load_profile(name: str) -> dict[str, Any]:
    try:
        filename = CASE_FILES[name]
    except KeyError as exc:
        raise ProfileError(f"unknown case profile: {name}") from exc
    resource = files("drl_multiseed").joinpath("data", "cases", filename)
    try:
        raw = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read packaged case profile: {filename}") from exc
    if not isinstance(raw, Mapping):
        raise ProfileError("case profile root must be an object")
    return validate_profile(raw)


def load_all_profiles() -> dict[str, dict[str, Any]]:
    return {name: load_profile(name) for name in ("sz_air", "mz_air", "mz_hydro")}
