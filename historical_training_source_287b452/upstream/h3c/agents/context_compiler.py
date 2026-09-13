"""Lossless typed-context compiler for compact Agent and expanded audit views."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

from h3c.agents.time_context import COORDINATION_PERIOD_SECONDS, completed_interval

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
SectionKind = Literal["json", "common_rows", "working_memory", "control_specification"]

_REQUIRED_STEP_ARRAY_PATHS = {
    "/action/actual_setpoints_c": "actual_setpoint_c",
    "/action/matched_rules": "matched_rule",
    "/outcome/zone_temperatures_c": "zone_temperature_c",
    "/outcome/pmv": "pmv",
    "/outcome/effective_occupancy": "effective_occupancy",
}
_DERIVED_ACTION_STEP_ARRAY_PATHS = {
    "/action/regime_base_setpoints_c": "regime_base_setpoint_c",
    "/action/setpoint_offsets_from_regime_base_c": "setpoint_offset_from_regime_base_c",
    "/action/cooling_effects_relative_to_regime_base": ("cooling_effect_relative_to_regime_base"),
}
_DERIVED_OUTCOME_FIELDS = (
    "discomfort_zone_hours",
    "discomfort_pmv_hours",
    "occupied_peak_absolute_pmv",
    "setpoint_total_variation_c",
    "setpoint_direction_reversals",
)
_OBSERVED_CONTEXT_FIELDS = (
    "outdoor_temperature_c",
    "solar_irradiance_w_m2",
    "electricity_price",
    "temp_rise_to_warm_pmv_edge_c",
    "temp_drop_to_cool_pmv_edge_c",
)
_SITE_OBJECTIVE_HISTORY_FIELDS = (
    "site_step_reward",
    "site_energy_penalty",
    "site_comfort_penalty",
    "site_smoothness_penalty",
)
_ZONE_OBJECTIVE_HISTORY_FIELDS = (
    "zone_comfort_penalty_contribution",
    "zone_smoothness_penalty_contribution",
)


def _normalize(value: Any) -> JsonValue:
    """Normalize mappings while omitting unavailable values and preserving empty collections."""
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for raw_key, raw_item in value.items():
            if raw_item is None:
                continue
            normalized[str(raw_key)] = _normalize(raw_item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"context value is not JSON-compatible: {type(value).__name__}")


def normalized_context(value: Any) -> JsonValue:
    """Return the exact canonical representation used by the compiler."""
    return copy.deepcopy(_normalize(value))


def _pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _pointer_unescape(value: str) -> str:
    return value.replace("~1", "/").replace("~0", "~")


def _pointer(parts: Sequence[str]) -> str:
    return "/" + "/".join(_pointer_escape(part) for part in parts)


def _parts(pointer: str) -> tuple[str, ...]:
    if not pointer.startswith("/"):
        raise ValueError(f"invalid field pointer: {pointer}")
    return tuple(_pointer_unescape(part) for part in pointer[1:].split("/"))


def _flatten(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    leaves: dict[str, JsonValue] = {}

    def visit(item: JsonValue, path: tuple[str, ...]) -> None:
        if isinstance(item, dict) and item:
            for key, child in item.items():
                visit(child, (*path, key))
            return
        leaves[_pointer(path)] = copy.deepcopy(item)

    for key, child in value.items():
        visit(child, (key,))
    return leaves


def _put(root: dict[str, JsonValue], pointer: str, value: JsonValue) -> None:
    parts = _parts(pointer)
    if not parts:
        raise ValueError("root field cannot be assigned by pointer")
    cursor = root
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            child: dict[str, JsonValue] = {}
            cursor[part] = child
            cursor = child
        elif isinstance(current, dict):
            cursor = current
        else:
            raise ValueError(f"field owner conflict at {pointer}")
    leaf = parts[-1]
    if leaf in cursor and cursor[leaf] != value:
        raise ValueError(f"field owner conflict at {pointer}")
    cursor[leaf] = copy.deepcopy(value)


def _nested(leaves: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for pointer, value in leaves.items():
        _put(result, pointer, value)
    return result


def _is_scalar(value: JsonValue) -> bool:
    return not isinstance(value, (dict, list))


def _strict_integer(value: JsonValue, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def factor_common_rows(
    records: Sequence[Mapping[str, Any]], *, identity_fields: Sequence[str]
) -> dict[str, JsonValue]:
    """Factor structurally identical paths while keeping row cells scalar-only."""
    normalized = cast(list[dict[str, JsonValue]], _normalize(list(records)))
    if not normalized:
        return {
            "kind": "common_rows",
            "identity_fields": list(identity_fields),
            "common": {},
            "columns": [],
            "rows": [],
            "details": [],
        }
    identities = set(identity_fields)
    if not identities:
        raise ValueError("common+rows requires at least one identity field")
    flattened = [_flatten(record) for record in normalized]
    identity_pointers = {_pointer((field,)) for field in identity_fields}
    for index, leaves in enumerate(flattened):
        missing = identity_pointers - set(leaves)
        if missing:
            raise ValueError(f"row {index} lacks identity fields: {sorted(missing)}")
        if any(not _is_scalar(leaves[pointer]) for pointer in identity_pointers):
            raise ValueError("identity fields must be scalar")

    first_paths = list(flattened[0])
    common_paths = [
        path
        for path in first_paths
        if path not in identity_pointers
        and all(path in row and row[path] == flattened[0][path] for row in flattened[1:])
    ]
    common_set = set(common_paths)
    row_columns: list[str] = []
    for flattened_row in flattened:
        for path, value in flattened_row.items():
            if path in identity_pointers or path in common_set or not _is_scalar(value):
                continue
            if path not in row_columns:
                row_columns.append(path)

    compact_rows: list[JsonValue] = []
    details: list[JsonValue] = []
    for leaves in flattened:
        compact_row: dict[str, JsonValue] = {
            field: copy.deepcopy(leaves[_pointer((field,))]) for field in identity_fields
        }
        values: dict[str, JsonValue] = {}
        nested_values: dict[str, JsonValue] = {}
        for path, value in leaves.items():
            if path in identity_pointers or path in common_set:
                continue
            if _is_scalar(value):
                values[path] = copy.deepcopy(value)
            else:
                nested_values[path] = copy.deepcopy(value)
        compact_row["values"] = values
        compact_rows.append(compact_row)
        if nested_values:
            detail: dict[str, JsonValue] = {
                field: copy.deepcopy(leaves[_pointer((field,))]) for field in identity_fields
            }
            detail["values"] = _nested(nested_values)
            details.append(detail)

    view: dict[str, JsonValue] = {
        "kind": "common_rows",
        "identity_fields": cast(JsonValue, list(identity_fields)),
        "common": _nested({path: flattened[0][path] for path in common_paths}),
        "columns": cast(JsonValue, row_columns),
        "rows": compact_rows,
        "details": details,
    }
    if decode_common_rows(view) != normalized:
        raise ValueError("common+rows round-trip failed")
    return view


def decode_common_rows(view: Mapping[str, Any]) -> list[dict[str, JsonValue]]:
    """Recover normalized records from a common+rows compact view."""
    identity_fields = [str(field) for field in view["identity_fields"]]
    common = cast(dict[str, JsonValue], _normalize(view["common"]))
    details_index: dict[tuple[JsonScalar, ...], dict[str, JsonValue]] = {}
    for raw_detail in cast(Sequence[Mapping[str, Any]], view["details"]):
        key = tuple(cast(JsonScalar, raw_detail[field]) for field in identity_fields)
        if key in details_index:
            raise ValueError("duplicate details identity")
        details_index[key] = cast(dict[str, JsonValue], _normalize(raw_detail["values"]))

    decoded: list[dict[str, JsonValue]] = []
    for raw_row in cast(Sequence[Mapping[str, Any]], view["rows"]):
        record = copy.deepcopy(common)
        identity = tuple(cast(JsonScalar, raw_row[field]) for field in identity_fields)
        for field, value in zip(identity_fields, identity, strict=True):
            _put(record, _pointer((field,)), value)
        values = cast(Mapping[str, Any], raw_row["values"])
        for path, value in values.items():
            _put(record, str(path), _normalize(value))
        detail = details_index.pop(identity, None)
        if detail is not None:
            for path, detail_value in _flatten(detail).items():
                _put(record, path, detail_value)
        decoded.append(record)
    if details_index:
        raise ValueError("details row has no matching scalar row")
    return decoded


def _pop_path(record: dict[str, JsonValue], pointer: str) -> JsonValue | None:
    parts = _parts(pointer)
    cursor: dict[str, JsonValue] = record
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            return None
        cursor = child
    value = cursor.pop(parts[-1], None)
    for depth in range(len(parts) - 1, 0, -1):
        parent = record
        for part in parts[: depth - 1]:
            child = parent[part]
            if not isinstance(child, dict):
                raise ValueError("invalid nested context")
            parent = child
        child = parent.get(parts[depth - 1])
        if isinstance(child, dict) and not child:
            parent.pop(parts[depth - 1])
    return value


def compile_working_memory(
    records: Sequence[Mapping[str, Any]],
    *,
    reference_hour: int | None = None,
    reference_time_seconds: int | None = None,
    presentation: Literal["decision_history", "completed_interval"] = "decision_history",
    decision_rationale_visible: bool = True,
) -> dict[str, JsonValue]:
    """Compile completed records into time, state, action, and derived-feature layers."""
    canonical = cast(list[dict[str, JsonValue]], _normalize(list(records)))
    grouped: dict[int, list[dict[str, JsonValue]]] = {}
    for record in canonical:
        hour = record.get("hour")
        zone = record.get("zone")
        if not isinstance(hour, int) or isinstance(hour, bool) or not isinstance(zone, str):
            raise ValueError("working-memory records require integer hour and string zone")
        grouped.setdefault(hour, []).append(copy.deepcopy(record))

    if (reference_hour is None) != (reference_time_seconds is None):
        raise ValueError("working-memory clock requires both reference hour and reference time")
    if reference_hour is None:
        reference_hour = max(grouped, default=-1) + 1
        reference_time_seconds = reference_hour * COORDINATION_PERIOD_SECONDS
    if isinstance(reference_hour, bool) or reference_hour < 0:
        raise ValueError("working-memory reference hour must be nonnegative")
    if presentation not in {"decision_history", "completed_interval"}:
        raise ValueError("working-memory presentation is invalid")
    assert reference_time_seconds is not None

    hours: list[JsonValue] = []
    for hour, hour_records in grouped.items():
        interval_start_seconds = (
            reference_time_seconds + (hour - reference_hour) * COORDINATION_PERIOD_SECONDS
        )
        clock = completed_interval(interval_start_seconds)
        base_records: list[dict[str, JsonValue]] = []
        state_history_rows: list[dict[str, JsonValue]] = []
        action_history_rows: list[dict[str, JsonValue]] = []
        observed_context_rows: list[dict[str, JsonValue]] = []
        current_state_rows: list[dict[str, JsonValue]] = []
        derived_feature_rows: list[dict[str, JsonValue]] = []
        forecast_rows: list[JsonValue] = []
        site_result: dict[str, JsonValue] | None = None
        site_objective_feedback: dict[str, JsonValue] | None = None
        site_objective_rows: list[dict[str, JsonValue]] = []
        zone_objective_rows: list[dict[str, JsonValue]] = []
        objective_feedback_present: bool | None = None
        for record in hour_records:
            zone = cast(str, record["zone"])
            base = copy.deepcopy(record)
            arrays: dict[str, list[JsonValue]] = {}
            for path, column in _REQUIRED_STEP_ARRAY_PATHS.items():
                value = _pop_path(base, path)
                if not isinstance(value, list) or len(value) != 4:
                    raise ValueError(f"working-memory field {path} must contain four steps")
                arrays[column] = value
            derived_action_values = {
                path: _pop_path(base, path) for path in _DERIVED_ACTION_STEP_ARRAY_PATHS
            }
            derived_action_present = {
                path for path, value in derived_action_values.items() if value is not None
            }
            if derived_action_present and derived_action_present != set(
                _DERIVED_ACTION_STEP_ARRAY_PATHS
            ):
                raise ValueError("derived action facts must appear as one complete field group")
            for path, column in _DERIVED_ACTION_STEP_ARRAY_PATHS.items():
                value = derived_action_values[path]
                if value is None:
                    continue
                if not isinstance(value, list) or len(value) != 4:
                    raise ValueError(f"working-memory field {path} must contain four steps")
                arrays[column] = value
            shield_raw = _pop_path(base, "/action/shield")
            if not isinstance(shield_raw, list) or len(shield_raw) != 4:
                raise ValueError("working-memory shield must contain four steps")
            regime_raw = _pop_path(base, "/context/regime_step_coverage")
            if not isinstance(regime_raw, dict):
                raise ValueError("working-memory regime coverage is missing")
            regimes_by_step: dict[int, str] = {}
            for regime, covered_steps in regime_raw.items():
                if not isinstance(covered_steps, list):
                    raise ValueError("regime coverage must be a step list")
                for covered_step in covered_steps:
                    step_number = int(cast(int | float, covered_step))
                    if step_number in regimes_by_step:
                        raise ValueError("a physical step has more than one regime owner")
                    regimes_by_step[step_number] = regime
            observed_raw = _pop_path(base, "/context/observed_context_history")
            score_limit = _pop_path(base, "/context/abs_pmv_score_limit")
            if (observed_raw is None) != (score_limit is None):
                raise ValueError(
                    "observed context history and PMV score limit must appear together"
                )
            observed_arrays: dict[str, list[JsonValue]] = {}
            if observed_raw is not None:
                if not isinstance(observed_raw, dict) or not observed_raw:
                    raise ValueError("observed context history must be a nonempty object")
                unknown_observed = set(observed_raw) - set(_OBSERVED_CONTEXT_FIELDS)
                if unknown_observed:
                    raise ValueError(
                        f"observed context history has unknown fields: {sorted(unknown_observed)}"
                    )
                for field, values in observed_raw.items():
                    if not isinstance(values, list) or len(values) != 4:
                        raise ValueError(
                            f"observed context field {field} must contain four action-time values"
                        )
                    if any(not _is_scalar(value) for value in values):
                        raise ValueError("observed context history values must be scalar")
                    observed_arrays[str(field)] = copy.deepcopy(values)
            initial_raw = _pop_path(base, "/context/initial_observation")
            if not isinstance(initial_raw, dict):
                raise ValueError("working-memory initial observation is missing")
            initial = copy.deepcopy(initial_raw)
            forecast_raw = initial.pop("occupancy_next_steps", None)
            if not isinstance(forecast_raw, list) or len(forecast_raw) != 4:
                raise ValueError("working-memory occupancy forecast must contain four steps")
            for step_ahead, occupancy in enumerate(forecast_raw, start=1):
                forecast_rows.append(
                    {"zone": zone, "step_ahead": step_ahead, "occupancy": occupancy}
                )

            required_initial = {
                "zone_temperature_c",
                "current_occupancy",
                "last_occupancy",
                "last_pmv",
                "last_setpoint_c",
            }
            if not required_initial <= set(initial):
                raise ValueError("working-memory initial state is incomplete")
            initial_temperature = float(cast(int | float, initial.pop("zone_temperature_c")))
            initial_occupancy = initial.pop("current_occupancy")
            initial_pmv = float(cast(int | float, initial.pop("last_pmv")))
            initial_setpoint = float(cast(int | float, initial.pop("last_setpoint_c")))
            if initial:
                _put(base, "/context/initial_observation", initial)

            current_site = {
                "site_cost": _pop_path(base, "/outcome/site_cost"),
                "site_energy_kwh": _pop_path(base, "/outcome/site_energy_kwh"),
            }
            if any(value is None for value in current_site.values()):
                raise ValueError("working-memory site result is incomplete")
            if site_result is None:
                site_result = current_site
            elif site_result != current_site:
                raise ValueError("repeated site-result owners disagree across zones")

            objective_raw = _pop_path(base, "/outcome/objective_feedback")
            current_objective_present = objective_raw is not None
            if objective_feedback_present is None:
                objective_feedback_present = current_objective_present
            elif objective_feedback_present != current_objective_present:
                raise ValueError("objective feedback must cover every zone in a completed hour")
            if objective_raw is not None:
                if not isinstance(objective_raw, dict):
                    raise ValueError("objective feedback must be an object")
                expected_objective_fields = {
                    "interval_reward",
                    *_SITE_OBJECTIVE_HISTORY_FIELDS,
                    *_ZONE_OBJECTIVE_HISTORY_FIELDS,
                }
                if set(objective_raw) != expected_objective_fields:
                    raise ValueError("objective feedback does not match its exact field contract")
                interval_reward = objective_raw["interval_reward"]
                if not _is_scalar(interval_reward):
                    raise ValueError("interval reward must be scalar")
                site_arrays: dict[str, list[JsonValue]] = {}
                zone_arrays: dict[str, list[JsonValue]] = {}
                for field in _SITE_OBJECTIVE_HISTORY_FIELDS:
                    values = objective_raw[field]
                    if not isinstance(values, list) or len(values) != 4:
                        raise ValueError(
                            f"objective feedback field {field} must contain four steps"
                        )
                    if any(not _is_scalar(value) for value in values):
                        raise ValueError("objective feedback history values must be scalar")
                    site_arrays[field] = copy.deepcopy(values)
                for field in _ZONE_OBJECTIVE_HISTORY_FIELDS:
                    values = objective_raw[field]
                    if not isinstance(values, list) or len(values) != 4:
                        raise ValueError(
                            f"objective feedback field {field} must contain four steps"
                        )
                    if any(not _is_scalar(value) for value in values):
                        raise ValueError("zone objective contributions must be scalar")
                    zone_arrays[field] = copy.deepcopy(values)
                current_site_objective = {
                    "interval_reward": copy.deepcopy(interval_reward),
                    **copy.deepcopy(site_arrays),
                }
                if site_objective_feedback is None:
                    site_objective_feedback = current_site_objective
                elif site_objective_feedback != current_site_objective:
                    raise ValueError(
                        "repeated site objective-feedback owners disagree across zones"
                    )

            first_step = cast(dict[str, JsonValue], shield_raw[0]).get("step")
            if not isinstance(first_step, int) or isinstance(first_step, bool):
                raise ValueError("working-memory first physical step is invalid")
            state_history_rows.append(
                {
                    "zone": zone,
                    "sample_index": 0,
                    "physical_step": first_step - 1,
                    "state_time": "hour_start_before_first_action",
                    "zone_temperature_c": initial_temperature,
                    "pmv": initial_pmv,
                    "effective_occupancy": initial_occupancy,
                    "setpoint_c": initial_setpoint,
                }
            )
            for index, raw_shield in enumerate(shield_raw):
                if not isinstance(raw_shield, dict):
                    raise ValueError("working-memory shield row must be an object")
                step = raw_shield.get("step")
                if not isinstance(step, int) or isinstance(step, bool):
                    raise ValueError("working-memory shield step must be an integer")
                if step not in regimes_by_step:
                    raise ValueError("regime coverage does not own every physical step")
                state_row: dict[str, JsonValue] = {
                    "zone": zone,
                    "sample_index": index + 1,
                    "physical_step": step,
                    "state_time": "outcome_after_applied_step",
                    "zone_temperature_c": arrays["zone_temperature_c"][index],
                    "pmv": arrays["pmv"][index],
                    "effective_occupancy": arrays["effective_occupancy"][index],
                    "setpoint_c": arrays["actual_setpoint_c"][index],
                }
                action_row: dict[str, JsonValue] = {
                    "zone": zone,
                    "physical_step": step,
                    "regime": regimes_by_step[step],
                    "actual_setpoint_c": arrays["actual_setpoint_c"][index],
                    "matched_rule": arrays["matched_rule"][index],
                    "actuator_bounds": raw_shield.get("actuator_bounds"),
                    "setpoint_rate_limit": raw_shield.get("setpoint_rate_limit"),
                    "comfort_recovery": raw_shield.get("comfort_recovery"),
                }
                for column in _DERIVED_ACTION_STEP_ARRAY_PATHS.values():
                    if column in arrays:
                        action_row[column] = arrays[column][index]
                if observed_arrays:
                    observed_context_rows.append(
                        {
                            "zone": zone,
                            "physical_step": step,
                            "abs_pmv_score_limit": score_limit,
                            **{field: values[index] for field, values in observed_arrays.items()},
                        }
                    )
                if any(value is None for value in (*state_row.values(), *action_row.values())):
                    raise ValueError("working-memory history row is incomplete")
                state_history_rows.append(state_row)
                action_history_rows.append(action_row)
                if objective_raw is not None:
                    if not site_objective_rows or len(site_objective_rows) <= index:
                        site_objective_rows.append(
                            {
                                "physical_step": step,
                                **{
                                    field: site_arrays[field][index]
                                    for field in _SITE_OBJECTIVE_HISTORY_FIELDS
                                },
                            }
                        )
                    zone_objective_rows.append(
                        {
                            "zone": zone,
                            "physical_step": step,
                            **{
                                field: zone_arrays[field][index]
                                for field in _ZONE_OBJECTIVE_HISTORY_FIELDS
                            },
                        }
                    )

            final_state = state_history_rows[-1]
            current_state_rows.append(
                {
                    "zone": zone,
                    "last_completed_step": final_state["physical_step"],
                    "zone_temperature_c": final_state["zone_temperature_c"],
                    "pmv": final_state["pmv"],
                    "effective_occupancy": final_state["effective_occupancy"],
                    "setpoint_c": final_state["setpoint_c"],
                }
            )
            source_metrics: dict[str, JsonValue] = {}
            for name in _DERIVED_OUTCOME_FIELDS:
                value = _pop_path(base, f"/outcome/{name}")
                if value is None:
                    raise ValueError(f"working-memory derived metric {name} is missing")
                source_metrics[name] = value
            temperatures = [
                float(cast(int | float, row["zone_temperature_c"]))
                for row in state_history_rows[-5:]
            ]
            pmv_values = [float(cast(int | float, row["pmv"])) for row in state_history_rows[-5:]]
            setpoints = [
                float(cast(int | float, row["setpoint_c"])) for row in state_history_rows[-5:]
            ]
            derived_feature_rows.append(
                {
                    "zone": zone,
                    "zone_temperature_change_last_step_c": round(
                        temperatures[-1] - temperatures[-2], 6
                    ),
                    "zone_temperature_change_hour_c": round(temperatures[-1] - temperatures[0], 6),
                    "zone_temperature_slope_c_per_hour": round(
                        temperatures[-1] - temperatures[0], 6
                    ),
                    "pmv_change_last_step": round(pmv_values[-1] - pmv_values[-2], 6),
                    "pmv_change_hour": round(pmv_values[-1] - pmv_values[0], 6),
                    "setpoint_change_last_step_c": round(setpoints[-1] - setpoints[-2], 6),
                    "setpoint_change_hour_c": round(setpoints[-1] - setpoints[0], 6),
                    **source_metrics,
                }
            )
            base_records.append(base)
        hours.append(
            {
                "hour": hour,
                "clock": cast(JsonValue, clock),
                "time_semantics": {
                    "agent_decision_interval_minutes": 60,
                    "physical_step_interval_minutes": 15,
                    "current_state": "last completed outcome",
                    "state_history": (
                        "hour-start state followed by four post-action physical outcomes"
                    ),
                    "action_history": "setpoint and assurance applied before each outcome",
                },
                "site_result": site_result or {},
                **(
                    {
                        "objective_feedback": {
                            "interval_reward": site_objective_feedback["interval_reward"],
                            "site_history": factor_common_rows(
                                site_objective_rows, identity_fields=("physical_step",)
                            ),
                            "zone_contributions": factor_common_rows(
                                zone_objective_rows,
                                identity_fields=("zone", "physical_step"),
                            ),
                        }
                    }
                    if site_objective_feedback is not None
                    else {}
                ),
                "hourly_decision": factor_common_rows(base_records, identity_fields=("zone",)),
                "current_state": factor_common_rows(current_state_rows, identity_fields=("zone",)),
                "recent_state_history": factor_common_rows(
                    state_history_rows, identity_fields=("zone", "sample_index")
                ),
                "action_history": factor_common_rows(
                    action_history_rows, identity_fields=("zone", "physical_step")
                ),
                **(
                    {
                        "observed_context_history": factor_common_rows(
                            observed_context_rows,
                            identity_fields=("zone", "physical_step"),
                        )
                    }
                    if observed_context_rows
                    else {}
                ),
                "occupancy_forecast": factor_common_rows(
                    cast(Sequence[Mapping[str, Any]], forecast_rows),
                    identity_fields=("zone", "step_ahead"),
                ),
                "derived_features": factor_common_rows(
                    derived_feature_rows, identity_fields=("zone",)
                ),
            }
        )
    view: dict[str, JsonValue] = {
        "kind": "working_memory",
        "presentation": presentation,
        "decision_rationale_visible": decision_rationale_visible,
        "hours": hours,
    }
    if decode_working_memory(view) != canonical:
        raise ValueError("working-memory round-trip failed")
    return view


def decode_working_memory(view: Mapping[str, Any]) -> list[dict[str, JsonValue]]:
    """Recover normalized completed-hour records from a compact working-memory view."""
    decoded: list[dict[str, JsonValue]] = []
    for raw_hour in cast(Sequence[Mapping[str, Any]], view["hours"]):
        hour = int(raw_hour["hour"])
        records = decode_common_rows(cast(Mapping[str, Any], raw_hour["hourly_decision"]))
        forecast_index: dict[str, list[Mapping[str, Any]]] = {}
        forecast_rows = decode_common_rows(cast(Mapping[str, Any], raw_hour["occupancy_forecast"]))
        for row in forecast_rows:
            forecast_index.setdefault(str(row["zone"]), []).append(row)
        state_index: dict[str, list[Mapping[str, Any]]] = {}
        for row in decode_common_rows(cast(Mapping[str, Any], raw_hour["recent_state_history"])):
            state_index.setdefault(str(row["zone"]), []).append(row)
        action_index: dict[str, list[Mapping[str, Any]]] = {}
        for row in decode_common_rows(cast(Mapping[str, Any], raw_hour["action_history"])):
            action_index.setdefault(str(row["zone"]), []).append(row)
        observed_index: dict[str, list[Mapping[str, Any]]] = {}
        if "observed_context_history" in raw_hour:
            for row in decode_common_rows(
                cast(Mapping[str, Any], raw_hour["observed_context_history"])
            ):
                observed_index.setdefault(str(row["zone"]), []).append(row)
        derived_index = {
            str(row["zone"]): row
            for row in decode_common_rows(cast(Mapping[str, Any], raw_hour["derived_features"]))
        }
        site_result = cast(Mapping[str, JsonValue], raw_hour["site_result"])
        objective_block = raw_hour.get("objective_feedback")
        site_objective_rows: list[dict[str, JsonValue]] = []
        zone_objective_index: dict[str, list[dict[str, JsonValue]]] = {}
        interval_reward: JsonValue | None = None
        if objective_block is not None:
            if not isinstance(objective_block, Mapping):
                raise ValueError("objective-feedback compact view must be an object")
            interval_reward = objective_block["interval_reward"]
            site_objective_rows = sorted(
                decode_common_rows(cast(Mapping[str, Any], objective_block["site_history"])),
                key=lambda row: _strict_integer(
                    row["physical_step"], field="site objective physical_step"
                ),
            )
            if len(site_objective_rows) != 4:
                raise ValueError("site objective feedback lost a four-step sequence")
            for row in decode_common_rows(
                cast(Mapping[str, Any], objective_block["zone_contributions"])
            ):
                zone_objective_index.setdefault(str(row["zone"]), []).append(row)
        for record in records:
            zone = str(record["zone"])
            forecast = sorted(forecast_index.pop(zone), key=lambda row: int(row["step_ahead"]))
            states = sorted(state_index.pop(zone), key=lambda row: int(row["sample_index"]))
            actions = sorted(action_index.pop(zone), key=lambda row: int(row["physical_step"]))
            observed = sorted(
                observed_index.pop(zone, []), key=lambda row: int(row["physical_step"])
            )
            derived = derived_index.pop(zone)
            if len(forecast) != 4 or len(states) != 5 or len(actions) != 4:
                raise ValueError("working-memory compact view lost a four-step sequence")
            if observed:
                if len(observed) != 4:
                    raise ValueError("observed context history lost a four-step sequence")
                limits = {row.get("abs_pmv_score_limit") for row in observed}
                if len(limits) != 1 or None in limits:
                    raise ValueError("observed context history PMV score limit disagrees")
                _put(
                    record,
                    "/context/abs_pmv_score_limit",
                    cast(JsonValue, next(iter(limits))),
                )
                observed_fields = [
                    field
                    for field in _OBSERVED_CONTEXT_FIELDS
                    if any(field in row for row in observed)
                ]
                for field in observed_fields:
                    if any(field not in row for row in observed):
                        raise ValueError(f"observed context field {field} has a partial sequence")
                    _put(
                        record,
                        f"/context/observed_context_history/{field}",
                        [cast(JsonValue, row[field]) for row in observed],
                    )
            _put(
                record,
                "/context/initial_observation/occupancy_next_steps",
                [cast(JsonValue, row["occupancy"]) for row in forecast],
            )
            initial = states[0]
            _put(
                record,
                "/context/initial_observation/zone_temperature_c",
                cast(JsonValue, initial["zone_temperature_c"]),
            )
            _put(
                record,
                "/context/initial_observation/current_occupancy",
                cast(JsonValue, initial["effective_occupancy"]),
            )
            _put(
                record,
                "/context/initial_observation/last_pmv",
                cast(JsonValue, initial["pmv"]),
            )
            _put(
                record,
                "/context/initial_observation/last_setpoint_c",
                cast(JsonValue, initial["setpoint_c"]),
            )
            coverage: dict[str, JsonValue] = {}
            for decoded_action in actions:
                regime = str(decoded_action["regime"])
                physical_step = decoded_action["physical_step"]
                if not isinstance(physical_step, int) or isinstance(physical_step, bool):
                    raise ValueError("action-history physical step is invalid")
                cast(list[JsonValue], coverage.setdefault(regime, [])).append(physical_step)
            _put(record, "/context/regime_step_coverage", coverage)
            _put(
                record,
                "/action/actual_setpoints_c",
                [cast(JsonValue, row["actual_setpoint_c"]) for row in actions],
            )
            present_derived_columns = {
                column
                for column in _DERIVED_ACTION_STEP_ARRAY_PATHS.values()
                if all(column in row for row in actions)
            }
            partial_derived_columns = {
                column
                for column in _DERIVED_ACTION_STEP_ARRAY_PATHS.values()
                if any(column in row for row in actions)
            }
            if partial_derived_columns and present_derived_columns != set(
                _DERIVED_ACTION_STEP_ARRAY_PATHS.values()
            ):
                raise ValueError("derived action history lost a complete field group")
            if present_derived_columns:
                for path, column in _DERIVED_ACTION_STEP_ARRAY_PATHS.items():
                    _put(
                        record,
                        path,
                        [cast(JsonValue, row[column]) for row in actions],
                    )
            _put(
                record,
                "/action/matched_rules",
                [cast(JsonValue, row["matched_rule"]) for row in actions],
            )
            _put(
                record,
                "/outcome/zone_temperatures_c",
                [cast(JsonValue, row["zone_temperature_c"]) for row in states[1:]],
            )
            _put(record, "/outcome/pmv", [cast(JsonValue, row["pmv"]) for row in states[1:]])
            _put(
                record,
                "/outcome/effective_occupancy",
                [cast(JsonValue, row["effective_occupancy"]) for row in states[1:]],
            )
            _put(
                record,
                "/action/shield",
                [
                    {
                        "step": int(row["physical_step"]),
                        "actuator_bounds": row["actuator_bounds"],
                        "setpoint_rate_limit": row["setpoint_rate_limit"],
                        "comfort_recovery": row["comfort_recovery"],
                    }
                    for row in actions
                ],
            )
            for name in _DERIVED_OUTCOME_FIELDS:
                _put(record, f"/outcome/{name}", derived[name])
            _put(record, "/outcome/site_cost", site_result["site_cost"])
            _put(record, "/outcome/site_energy_kwh", site_result["site_energy_kwh"])
            if objective_block is not None:
                zone_objective_rows = sorted(
                    zone_objective_index.pop(zone, []),
                    key=lambda row: _strict_integer(
                        row["physical_step"], field="zone objective physical_step"
                    ),
                )
                if len(zone_objective_rows) != 4:
                    raise ValueError("zone objective feedback lost a four-step sequence")
                feedback: dict[str, JsonValue] = {"interval_reward": interval_reward}
                for field in _SITE_OBJECTIVE_HISTORY_FIELDS:
                    feedback[field] = [row[field] for row in site_objective_rows]
                for field in _ZONE_OBJECTIVE_HISTORY_FIELDS:
                    feedback[field] = [row[field] for row in zone_objective_rows]
                _put(record, "/outcome/objective_feedback", feedback)
            decoded.append(record)
        if (
            forecast_index
            or state_index
            or action_index
            or observed_index
            or derived_index
            or zone_objective_index
        ):
            raise ValueError("working-memory step rows have no matching zone record")
        record_hours = [record["hour"] for record in records]
        if any(
            not isinstance(record_hour, int) or isinstance(record_hour, bool) or record_hour != hour
            for record_hour in record_hours
        ):
            raise ValueError("working-memory hour owner conflict")
    return decoded


def _cell(value: JsonValue) -> str:
    if not _is_scalar(value):
        raise ValueError("table cells must be scalar")
    if isinstance(value, str) and value and not any(char in value for char in "|\r\n"):
        looks_numeric = re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value)
        if value not in {"true", "false", "null"} and looks_numeric is None:
            return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("|", "\\u007c")


def _table(rows: Sequence[Mapping[str, JsonValue]], columns: Sequence[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join((header, separator, *body))


def render_properties(value: JsonValue) -> str:
    """Render a typed object without JSON punctuation around its top-level field owners."""
    if not isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return "\n".join(
        f"{key}: "
        + (
            _cell(item)
            if _is_scalar(item)
            else json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        )
        for key, item in value.items()
    )


_DISPLAY_FIELD_ALIASES = {
    "effective_occupancy": "occupancy",
    "actual_setpoint_c": "setpoint_c",
    "zone_temperature_c": "temp_c",
    "zone_temperature_change_last_step_c": "temp_delta_15min_c",
    "zone_temperature_change_hour_c": "temp_delta_1h_c",
    "zone_temperature_slope_c_per_hour": "temp_slope_c_per_hour",
    "pmv_change_last_step": "pmv_delta_15min",
    "pmv_change_hour": "pmv_delta_1h",
    "setpoint_change_last_step_c": "setpoint_delta_15min_c",
    "setpoint_change_hour_c": "setpoint_delta_1h_c",
    "discomfort_zone_hours": "discomfort_zone_h",
    "discomfort_pmv_hours": "discomfort_pmv_h",
    "occupied_peak_absolute_pmv": "occupied_peak_abs_pmv",
    "setpoint_total_variation_c": "setpoint_tv_c",
    "setpoint_direction_reversals": "setpoint_reversals",
}


def _short_column_names(paths: Sequence[str], *, reserved: Sequence[str] = ()) -> list[str]:
    """Choose the shortest unambiguous, self-describing suffix for model-facing tables."""
    split = [_parts(path) for path in paths]
    chosen: list[str] = []
    reserved_set = set(reserved)
    for index, parts in enumerate(split):
        candidate = parts[-1]
        depth = 1
        while (
            candidate in reserved_set
            or candidate in chosen
            or any(
                other_index != index and ".".join(other[-depth:]) == candidate
                for other_index, other in enumerate(split)
            )
        ):
            depth += 1
            if depth > len(parts):
                candidate = "/".join(parts)
                break
            candidate = ".".join(parts[-depth:])
        chosen.append(_DISPLAY_FIELD_ALIASES.get(candidate, candidate))
    if len(set(chosen)) != len(chosen):
        raise ValueError("model-facing field aliases are ambiguous")
    return chosen


def render_common_rows(view: Mapping[str, Any]) -> str:
    """Render explicit common data, scalar rows, and separately typed nested details."""
    identities = [str(value) for value in view["identity_fields"]]
    paths = [str(value) for value in view["columns"]]
    common = cast(Mapping[str, JsonValue], view["common"])
    display_paths = _short_column_names(paths, reserved=identities)
    rows: list[dict[str, JsonValue]] = []
    row_by_identity: dict[tuple[JsonScalar, ...], dict[str, JsonValue]] = {}
    for raw_row in cast(Sequence[Mapping[str, Any]], view["rows"]):
        values = cast(Mapping[str, JsonValue], raw_row["values"])
        row = {field: cast(JsonValue, raw_row[field]) for field in identities}
        row.update(
            {
                display: values[path]
                for path, display in zip(paths, display_paths, strict=True)
                if path in values
            }
        )
        rows.append(row)
        row_by_identity[tuple(cast(JsonScalar, raw_row[field]) for field in identities)] = row
    remaining_details: list[JsonValue] = []
    for raw_detail in cast(Sequence[Mapping[str, Any]], view["details"]):
        detail_values = cast(Mapping[str, JsonValue], raw_detail["values"])
        expandable = bool(detail_values) and all(
            isinstance(value, list) and value and all(_is_scalar(item) for item in value)
            for value in detail_values.values()
        )
        identity = tuple(cast(JsonScalar, raw_detail[field]) for field in identities)
        target = row_by_identity.get(identity)
        if expandable and target is not None:
            for field, raw_values in detail_values.items():
                detail_items = cast(list[JsonValue], raw_values)
                for index, value in enumerate(detail_items, start=1):
                    target[f"{field}_{index}"] = value
        else:
            remaining_details.append(cast(JsonValue, copy.deepcopy(dict(raw_detail))))
    parts: list[str] = []
    if common:
        rendered_common: Mapping[str, JsonValue] = common
        if all(not isinstance(value, (dict, list)) for value in common.values()):
            rendered_common = {
                _DISPLAY_FIELD_ALIASES.get(field, field): value for field, value in common.items()
            }
        parts.append(
            "values_shared_by_all_rows:\n"
            + render_properties(cast(JsonValue, dict(rendered_common)))
        )
    grouped_rows: dict[tuple[str, ...], list[dict[str, JsonValue]]] = {}
    for row in rows:
        columns = tuple(row)
        grouped_rows.setdefault(columns, []).append(row)
    grouped_items = list(grouped_rows.items())
    for columns, group in grouped_items:
        label = "rows"
        if len(grouped_items) > 1:
            operations = {str(row["op"]) for row in group if "op" in row}
            if len(operations) == 1:
                label = f"{next(iter(operations))}_rows"
            else:
                safe_columns = [re.sub(r"[^a-zA-Z0-9_]+", "_", column) for column in columns]
                label = "rows_with_" + "_and_".join(safe_columns)
        parts.append(label + ":\n" + _table(group, columns))
    if remaining_details:
        parts.append(
            "details:\n```json\n"
            + json.dumps(remaining_details, ensure_ascii=False, separators=(",", ":"))
            + "\n```"
        )
    return "\n".join(parts)


def _assert_group_value(rows: Sequence[Mapping[str, JsonValue]], field: str) -> JsonValue:
    values = [row[field] for row in rows]
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"working-memory shared axis disagrees for {field}")
    return values[0]


def _direct_common(view: Mapping[str, Any]) -> dict[str, JsonValue]:
    common = cast(Mapping[str, JsonValue], view["common"])
    if any(isinstance(value, (dict, list)) for value in common.values()):
        raise ValueError("working-memory layer common fields must be direct scalars")
    return {
        _DISPLAY_FIELD_ALIASES.get(field, field): copy.deepcopy(value)
        for field, value in common.items()
    }


def _zone_constant_partition(
    records: Sequence[Mapping[str, JsonValue]],
    *,
    zones: Sequence[str],
    fields: Sequence[str],
    axis_field: str,
) -> tuple[list[dict[str, JsonValue]], list[str], list[dict[str, JsonValue]]]:
    """Extract per-zone constants and leave only time-varying values in axis rows."""
    constant_rows: list[dict[str, JsonValue]] = []
    variable_pairs: list[tuple[str, str]] = []
    for zone in zones:
        zone_records = [record for record in records if str(record["zone"]) == zone]
        constant_row: dict[str, JsonValue] = {"zone": zone}
        for field in fields:
            values = [record[field] for record in zone_records]
            if values and all(value == values[0] for value in values[1:]):
                constant_row[_DISPLAY_FIELD_ALIASES.get(field, field)] = values[0]
            else:
                variable_pairs.append((zone, field))
        if len(constant_row) > 1:
            constant_rows.append(constant_row)

    grouped: dict[JsonScalar, list[Mapping[str, JsonValue]]] = {}
    for record in records:
        axis_value = record[axis_field]
        if not _is_scalar(axis_value):
            raise ValueError("working-memory time axis must be scalar")
        grouped.setdefault(cast(JsonScalar, axis_value), []).append(record)
    variable_rows: list[dict[str, JsonValue]] = []
    for axis_value, axis_records in grouped.items():
        row: dict[str, JsonValue] = {_DISPLAY_FIELD_ALIASES.get(axis_field, axis_field): axis_value}
        by_zone = {str(record["zone"]): record for record in axis_records}
        if set(by_zone) != set(zones):
            raise ValueError("working-memory time axis has an incomplete zone axis")
        for zone, field in variable_pairs:
            display_field = _DISPLAY_FIELD_ALIASES.get(field, field)
            row[f"{zone}.{display_field}"] = by_zone[zone][field]
        variable_rows.append(row)
    variable_columns = [
        _DISPLAY_FIELD_ALIASES.get(axis_field, axis_field),
        *(f"{zone}.{_DISPLAY_FIELD_ALIASES.get(field, field)}" for zone, field in variable_pairs),
    ]
    return constant_rows, variable_columns, variable_rows


def _render_history(
    view: Mapping[str, Any], *, state_times: Sequence[str], include_terminal: bool
) -> str:
    records = decode_common_rows(view)
    if len(state_times) != 5:
        raise ValueError("working-memory state clock requires five boundaries")
    projected: list[dict[str, JsonValue]] = []
    for record in records:
        sample = record["sample_index"]
        if not isinstance(sample, int) or isinstance(sample, bool) or not 0 <= sample <= 4:
            raise ValueError("working-memory state sample is invalid")
        if not include_terminal and sample == 4:
            continue
        item = copy.deepcopy(record)
        item["state_time"] = state_times[sample]
        item.pop("sample_index")
        item.pop("physical_step")
        projected.append(item)
    zones = list(dict.fromkeys(str(record["zone"]) for record in projected))
    common = _direct_common(view)
    common.pop("state_time", None)
    source_fields = (
        "zone_temperature_c",
        "pmv",
        "effective_occupancy",
        "setpoint_c",
    )
    variable_fields = tuple(
        field for field in source_fields if _DISPLAY_FIELD_ALIASES.get(field, field) not in common
    )
    constants, variable_columns, variable_rows = _zone_constant_partition(
        projected,
        zones=zones,
        fields=variable_fields,
        axis_field="state_time",
    )
    parts = []
    if common:
        parts.append("values_shared_by_all_rows:\n" + render_properties(cast(JsonValue, common)))
    if constants:
        parts.append(
            "values_constant_within_each_zone:\n"
            + "\n".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in constants
            )
        )
    parts.append(
        "state_times: "
        + json.dumps(
            list(state_times if include_terminal else state_times[:-1]),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    if len(variable_columns) > 1:
        parts.append("time_varying_measurements:\n" + _table(variable_rows, variable_columns))
    return "\n".join(parts)


def _render_action_history(
    view: Mapping[str, Any], *, action_times: Sequence[str], outcome_times: Sequence[str]
) -> str:
    records = decode_common_rows(view)
    if len(action_times) != 4 or len(outcome_times) != 4:
        raise ValueError("working-memory action clock requires four action/outcome pairs")
    ordered_steps = sorted(
        {
            record["physical_step"]
            for record in records
            if isinstance(record["physical_step"], int)
            and not isinstance(record["physical_step"], bool)
        }
    )
    if len(ordered_steps) != 4:
        raise ValueError("working-memory action history requires four internal steps")
    clock_by_step = {
        step: (action_times[index], outcome_times[index])
        for index, step in enumerate(ordered_steps)
    }
    projected: list[dict[str, JsonValue]] = []
    for record in records:
        step = cast(int, record["physical_step"])
        action_time, outcome_time = clock_by_step[step]
        item = copy.deepcopy(record)
        item.pop("physical_step")
        item["action_time"] = action_time
        item["outcome_time"] = outcome_time
        projected.append(item)
    zones = list(dict.fromkeys(str(record["zone"]) for record in projected))
    common = _direct_common(view)
    source_fields = (
        "regime",
        "actual_setpoint_c",
        "regime_base_setpoint_c",
        "setpoint_offset_from_regime_base_c",
        "cooling_effect_relative_to_regime_base",
        "matched_rule",
        "actuator_bounds",
        "setpoint_rate_limit",
        "comfort_recovery",
    )
    available_fields = tuple(
        field for field in source_fields if all(field in record for record in projected)
    )
    variable_fields = tuple(
        field
        for field in available_fields
        if _DISPLAY_FIELD_ALIASES.get(field, field) not in common
    )
    constants, variable_columns, variable_rows = _zone_constant_partition(
        projected,
        zones=zones,
        fields=variable_fields,
        axis_field="action_time",
    )
    outcome_by_action = {str(record["action_time"]): record["outcome_time"] for record in projected}
    for row in variable_rows:
        row["outcome_time"] = outcome_by_action[str(row["action_time"])]
    if variable_rows:
        variable_columns.insert(1, "outcome_time")
    parts = []
    if common:
        parts.append("values_shared_by_all_rows:\n" + render_properties(cast(JsonValue, common)))
    if constants:
        parts.append(
            "values_constant_within_each_zone:\n"
            + "\n".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in constants
            )
        )
    if len(variable_columns) > 1:
        parts.append("time_varying_actions:\n" + _table(variable_rows, variable_columns))
    else:
        pairs = [
            {"action_time": action, "outcome_time": outcome}
            for action, outcome in zip(action_times, outcome_times, strict=True)
        ]
        parts.append("action_outcome_times:\n" + _table(pairs, ("action_time", "outcome_time")))
    return "\n".join(parts)


def _render_objective_feedback(
    view: Mapping[str, Any],
    *,
    action_times: Sequence[str],
    outcome_times: Sequence[str],
    show_zone_contributions: bool,
) -> str:
    """Render retrospective reward facts with site quantities owned once per step."""
    if len(action_times) != 4 or len(outcome_times) != 4:
        raise ValueError("objective feedback requires four action/outcome clock pairs")
    site_rows = sorted(
        decode_common_rows(cast(Mapping[str, Any], view["site_history"])),
        key=lambda row: _strict_integer(row["physical_step"], field="site objective physical_step"),
    )
    if len(site_rows) != 4:
        raise ValueError("objective feedback requires four site rows")
    rendered_site: list[dict[str, JsonValue]] = []
    step_clock: dict[int, tuple[str, str]] = {}
    for index, source in enumerate(site_rows):
        step = source["physical_step"]
        if not isinstance(step, int) or isinstance(step, bool):
            raise ValueError("objective-feedback physical step must be an integer")
        step_clock[step] = (action_times[index], outcome_times[index])
        rendered_site.append(
            {
                "action_time": action_times[index],
                "outcome_time": outcome_times[index],
                **{field: source[field] for field in _SITE_OBJECTIVE_HISTORY_FIELDS},
            }
        )
    parts = [f"interval_reward: {_cell(cast(JsonValue, view['interval_reward']))}"]
    parts.append(
        "site_reward_history:\n"
        + _table(
            rendered_site,
            ("action_time", "outcome_time", *_SITE_OBJECTIVE_HISTORY_FIELDS),
        )
    )
    if show_zone_contributions:
        contribution_rows = decode_common_rows(cast(Mapping[str, Any], view["zone_contributions"]))
        rendered_zone: list[dict[str, JsonValue]] = []
        for source in contribution_rows:
            step = source["physical_step"]
            if not isinstance(step, int) or isinstance(step, bool) or step not in step_clock:
                raise ValueError("zone objective contribution has no site-step owner")
            action_time, outcome_time = step_clock[step]
            rendered_zone.append(
                {
                    "zone": source["zone"],
                    "action_time": action_time,
                    "outcome_time": outcome_time,
                    **{field: source[field] for field in _ZONE_OBJECTIVE_HISTORY_FIELDS},
                }
            )
        parts.append(
            "zone_penalty_contributions:\n"
            + _table(
                rendered_zone,
                (
                    "zone",
                    "action_time",
                    "outcome_time",
                    *_ZONE_OBJECTIVE_HISTORY_FIELDS,
                ),
            )
        )
    return "\n".join(parts)


def _render_occupancy_forecast(view: Mapping[str, Any], *, outcome_times: Sequence[str]) -> str:
    records = decode_common_rows(view)
    grouped: dict[int, list[dict[str, JsonValue]]] = {}
    for record in records:
        grouped.setdefault(cast(int, record["step_ahead"]), []).append(record)
    zones = list(dict.fromkeys(str(record["zone"]) for record in records))
    if len(outcome_times) != len(grouped):
        raise ValueError("working-memory forecast clock does not match forecast rows")
    time_by_index = {index: outcome_times[index - 1] for index in sorted(grouped)}
    common = _direct_common(view)
    if "occupancy" in common:
        return (
            "values_shared_by_all_rows:\n"
            + render_properties(cast(JsonValue, common))
            + "\noutcome_times: "
            + json.dumps(list(outcome_times), ensure_ascii=False, separators=(",", ":"))
        )
    rows: list[dict[str, JsonValue]] = []
    for step_ahead, step_records in grouped.items():
        by_zone = {str(record["zone"]): record for record in step_records}
        if set(by_zone) != set(zones):
            raise ValueError("working-memory occupancy forecast has an incomplete zone axis")
        row: dict[str, JsonValue] = {"outcome_time": time_by_index[step_ahead]}
        row.update({zone: by_zone[zone]["occupancy"] for zone in zones})
        rows.append(row)
    return _table(rows, ["outcome_time", *zones])


def _render_scoped_common_rows(view: Mapping[str, Any], *, omit_common: Sequence[str] = ()) -> str:
    display_view = copy.deepcopy(dict(view))
    common = cast(dict[str, JsonValue], display_view["common"])
    for field in omit_common:
        common.pop(field, None)
    rows = cast(Sequence[Mapping[str, Any]], display_view["rows"])
    details = cast(Sequence[JsonValue], display_view["details"])
    if rows and not details and all(not cast(Mapping[str, Any], row["values"]) for row in rows):
        if common:
            rendered_common: Mapping[str, JsonValue] = common
            if all(not isinstance(value, (dict, list)) for value in common.values()):
                rendered_common = {
                    _DISPLAY_FIELD_ALIASES.get(field, field): value
                    for field, value in common.items()
                }
            return "values_shared_by_all_rows:\n" + render_properties(
                cast(JsonValue, dict(rendered_common))
            )
        return ""
    return render_common_rows(display_view)


def _render_observed_context_history(
    view: Mapping[str, Any], *, action_times: Sequence[str]
) -> str:
    """Render completed action-time conditions with site fields owned once per time."""
    records = decode_common_rows(view)
    ordered_steps = sorted(
        {
            cast(int, record["physical_step"])
            for record in records
            if isinstance(record.get("physical_step"), int)
            and not isinstance(record.get("physical_step"), bool)
        }
    )
    if len(ordered_steps) != 4 or len(action_times) != 4:
        raise ValueError("observed context history requires four action-time rows")
    time_by_step = {step: action_times[index] for index, step in enumerate(ordered_steps)}
    zones = list(dict.fromkeys(str(record["zone"]) for record in records))
    limits = {record.get("abs_pmv_score_limit") for record in records}
    if len(limits) != 1 or None in limits:
        raise ValueError("observed context history has no single PMV score limit owner")
    site_fields = (
        "outdoor_temperature_c",
        "solar_irradiance_w_m2",
        "electricity_price",
    )
    zone_fields = (
        "temp_rise_to_warm_pmv_edge_c",
        "temp_drop_to_cool_pmv_edge_c",
    )
    rows: list[dict[str, JsonValue]] = []
    for step in ordered_steps:
        step_records = [record for record in records if record.get("physical_step") == step]
        by_zone = {str(record["zone"]): record for record in step_records}
        if set(by_zone) != set(zones):
            raise ValueError("observed context time row has an incomplete zone axis")
        row: dict[str, JsonValue] = {"action_time": time_by_step[step]}
        for field in site_fields:
            present = [record[field] for record in step_records if field in record]
            if present:
                if len(present) != len(zones) or any(value != present[0] for value in present[1:]):
                    raise ValueError(f"site observed context field {field} disagrees across zones")
                row[field] = present[0]
        for zone in zones:
            for field in zone_fields:
                if field in by_zone[zone]:
                    row[f"{zone}.{field}"] = by_zone[zone][field]
        rows.append(row)
    columns = list(dict.fromkeys(field for row in rows for field in row))
    return f"abs_pmv_score_limit: {_cell(next(iter(limits)))}\n" + _table(rows, columns)


def _working_decision_agent_view(
    records: Sequence[Mapping[str, JsonValue]],
    *,
    rationale_visible: bool,
    initial_context_visible: bool,
) -> dict[str, JsonValue]:
    """Project audit-only proof detail out of the next decision's model-facing history."""
    projected: list[dict[str, JsonValue]] = []
    audit_only_paths = (
        "/action/admission/completed_validation_stages",
        "/action/proposal/causal_edge_ids",
        "/action/proposal/expected_effects",
        "/action/proposal/consistent_program_direction_proof",
    )
    for source in records:
        record = copy.deepcopy(dict(source))
        for path in audit_only_paths:
            _pop_path(record, path)
        if not rationale_visible:
            _pop_path(record, "/action/proposal/rationale")
        if not initial_context_visible:
            _pop_path(record, "/context/initial_observation")
        projected.append(record)
    return factor_common_rows(projected, identity_fields=("zone",))


def render_working_memory(view: Mapping[str, Any]) -> str:
    """Render explicit completed intervals without exposing internal indices."""
    presentation = str(view.get("presentation", "decision_history"))
    parts: list[str] = []
    for raw_hour in cast(Sequence[Mapping[str, Any]], view["hours"]):
        decision_records = decode_common_rows(cast(Mapping[str, Any], raw_hour["hourly_decision"]))
        zones = list(dict.fromkeys(str(record["zone"]) for record in decision_records))
        clock = cast(Mapping[str, JsonValue], raw_hour["clock"])
        interval = cast(str, clock["interval"])
        action_times = cast(Sequence[str], clock["action_times"])
        outcome_times = cast(Sequence[str], clock["outcome_times"])
        state_times = [action_times[0], *outcome_times]
        if presentation == "completed_interval":
            parts.append(f"interval: {interval}")
            parts.append(
                "action_times: "
                + json.dumps(action_times, ensure_ascii=False, separators=(",", ":"))
            )
            parts.append(
                "outcome_times: "
                + json.dumps(outcome_times, ensure_ascii=False, separators=(",", ":"))
            )
        else:
            parts.append(f"completed_interval: {interval}")
        parts.append("zones: " + json.dumps(zones, ensure_ascii=False, separators=(",", ":")))
        if presentation == "completed_interval":
            parts.append("control_period: 15 min")
        site = cast(Mapping[str, JsonValue], raw_hour["site_result"])
        parts.append("site_result: " + json.dumps(site, ensure_ascii=False, separators=(",", ":")))
        if "objective_feedback" in raw_hour:
            parts.append(
                "OBJECTIVE FEEDBACK:\n"
                + _render_objective_feedback(
                    cast(Mapping[str, Any], raw_hour["objective_feedback"]),
                    action_times=action_times,
                    outcome_times=outcome_times,
                    show_zone_contributions=presentation == "completed_interval" or len(zones) == 1,
                )
            )
        parts.append(
            "completed_decision:\n"
            + _render_scoped_common_rows(
                _working_decision_agent_view(
                    decision_records,
                    rationale_visible=bool(view.get("decision_rationale_visible", True)),
                    initial_context_visible=presentation == "completed_interval",
                ),
                omit_common=("hour",),
            )
        )
        parts.append(
            "recent_state_history:\n"
            + _render_history(
                cast(Mapping[str, Any], raw_hour["recent_state_history"]),
                state_times=state_times,
                include_terminal=presentation == "completed_interval",
            )
        )
        if "observed_context_history" in raw_hour:
            parts.append(
                "OBSERVED CONTEXT HISTORY:\n"
                + _render_observed_context_history(
                    cast(Mapping[str, Any], raw_hour["observed_context_history"]),
                    action_times=action_times,
                )
            )
        parts.append(
            "control_action_history:\n"
            + _render_action_history(
                cast(Mapping[str, Any], raw_hour["action_history"]),
                action_times=action_times,
                outcome_times=outcome_times,
            )
        )
        parts.append(
            "previous_decision_forecast:\n"
            + _render_occupancy_forecast(
                cast(Mapping[str, Any], raw_hour["occupancy_forecast"]),
                outcome_times=outcome_times,
            )
        )
        parts.append(f"derived_feature_interval: {interval}")
        parts.append(
            "derived_features:\n"
            + _render_scoped_common_rows(cast(Mapping[str, Any], raw_hour["derived_features"]))
        )
    return "\n".join(part for part in parts if part and not part.endswith(":\n"))


def render_control_specification(view: Mapping[str, Any]) -> str:
    """Render the executable program and its edit limits without repeated record keys."""
    specification = copy.deepcopy(dict(view))
    version = specification.pop("program_version")
    parameters = cast(Sequence[Mapping[str, JsonValue]], specification.pop("parameters"))
    rules = cast(Sequence[Mapping[str, JsonValue]], specification.pop("rules"))
    parts = [f"program_version: {_cell(cast(JsonValue, version))}"]
    parameter_rows: list[dict[str, JsonValue]] = []
    for parameter in parameters:
        row = {
            "param": copy.deepcopy(parameter["param"]),
            "current": copy.deepcopy(parameter["current"]),
            "min": copy.deepcopy(parameter["min"]),
            "max": copy.deepcopy(parameter["max"]),
            "source": copy.deepcopy(parameter["bounds_source"]),
        }
        parameter_rows.append(row)
    parts.append(
        "parameters:\n" + _table(parameter_rows, ("param", "current", "min", "max", "source"))
    )

    parts.append(
        "rules:\n"
        + "\n".join(json.dumps(rule, ensure_ascii=False, separators=(",", ":")) for rule in rules)
    )
    display_text = {
        "use the named parameter value": "named parameter value",
        "use the opposite sign of the named parameter value": "negative named parameter value",
        "a number, or the name of a parameter above": "number or parameter name",
        "finite numeric literals": "finite numbers",
        "not allowed": "forbidden",
    }
    display_fields = {
        "a_rule_you_add": "new_rule",
        "conditions_may_test": "condition_fields",
        "compared_against": "comparison_value",
        "actions": "action_types",
        "most_rules_at_once": "max_rules",
        "weather_condition_values": "weather_literals",
        "parameter_references": "parameter_refs",
    }

    def concise(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return {display_fields.get(key, key): concise(child) for key, child in value.items()}
        if isinstance(value, list):
            return [concise(child) for child in value]
        if isinstance(value, str):
            return display_text.get(value, value)
        return value

    for field, value in specification.items():
        parts.append(
            f"{field}: "
            + json.dumps(concise(cast(JsonValue, value)), ensure_ascii=False, separators=(",", ":"))
        )
    return "\n".join(parts)


def _field_manifest(value: JsonValue, prefix: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, dict):
        if not value:
            return [_pointer(prefix)]
        result: list[str] = []
        for key, item in value.items():
            result.extend(_field_manifest(item, (*prefix, key)))
        return result
    if isinstance(value, list):
        if not value:
            return [_pointer((*prefix, "[]"))]
        result = []
        for item in value:
            result.extend(_field_manifest(item, (*prefix, "[]")))
        return list(dict.fromkeys(result))
    return [_pointer(prefix)]


def _hash(value: JsonValue) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_INTERNAL_COORDINATE_FIELDS = {
    "hour",
    "step",
    "sample_index",
    "physical_step",
    "step_ahead",
    "last_completed_step",
    "time_seconds",
    "action_time_seconds",
}


def _internal_coordinates(value: JsonValue) -> list[dict[str, JsonValue]]:
    """Collect internal temporal coordinates for audit without rendering them to Agents."""
    coordinates: list[dict[str, JsonValue]] = []

    def visit(item: JsonValue, path: tuple[str, ...]) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = (*path, key)
                if key in _INTERNAL_COORDINATE_FIELDS or key.endswith("_time_seconds"):
                    coordinates.append(
                        {"path": _pointer(child_path), "value": copy.deepcopy(child)}
                    )
                visit(child, child_path)
        elif isinstance(item, list):
            for child in item:
                visit(child, (*path, "[]"))

    visit(value, ())
    return coordinates


@dataclass(frozen=True)
class CompiledContext:
    agent_view: str
    human_view: str
    canonical_ir: dict[str, JsonValue]
    compact_view: list[dict[str, JsonValue]]
    audit_view: dict[str, JsonValue]


class ContextBuilder:
    """Build three deterministic views from one canonical typed context."""

    def __init__(self) -> None:
        self._sections: list[dict[str, JsonValue]] = []
        self._canonical: dict[str, JsonValue] = {}
        self._audit_metadata: dict[str, JsonValue] = {}

    def add_audit_metadata(self, **values: Any) -> None:
        """Register internal coordinates that must never enter an Agent-facing section."""
        normalized = _normalize(values)
        if not isinstance(normalized, dict):
            raise ValueError("context audit metadata must be an object")
        overlap = set(self._audit_metadata) & set(normalized)
        if overlap:
            raise ValueError(f"duplicate audit metadata owner: {sorted(overlap)}")
        self._audit_metadata.update(normalized)

    def add_json(self, title: str, value: Any) -> None:
        if value is None:
            return
        canonical = _normalize(value)
        self._add(title, "json", canonical, canonical)

    def add_common_rows(
        self, title: str, records: Sequence[Mapping[str, Any]], *, identity_fields: Sequence[str]
    ) -> None:
        canonical = _normalize(list(records))
        compact = factor_common_rows(records, identity_fields=identity_fields)
        self._add(title, "common_rows", canonical, cast(JsonValue, compact))

    def add_working_memory(
        self,
        title: str,
        records: Sequence[Mapping[str, Any]],
        *,
        reference_hour: int | None = None,
        reference_time_seconds: int | None = None,
        presentation: Literal["decision_history", "completed_interval"] = "decision_history",
        decision_rationale_visible: bool = True,
    ) -> None:
        canonical = _normalize(list(records))
        compact = compile_working_memory(
            records,
            reference_hour=reference_hour,
            reference_time_seconds=reference_time_seconds,
            presentation=presentation,
            decision_rationale_visible=decision_rationale_visible,
        )
        self._add(title, "working_memory", canonical, cast(JsonValue, compact))

    def add_control_specification(self, title: str, specification: Mapping[str, Any]) -> None:
        canonical = _normalize(specification)
        if not isinstance(canonical, dict):
            raise ValueError("control specification must be an object")
        self._add(title, "control_specification", canonical, canonical)

    def _add(self, title: str, kind: SectionKind, canonical: JsonValue, compact: JsonValue) -> None:
        if title in self._canonical:
            raise ValueError(f"duplicate context-section owner: {title}")
        self._canonical[title] = canonical
        self._sections.append({"title": title, "kind": kind, "view": compact})

    def build(self) -> CompiledContext:
        decoded = decode_compact_context(self._sections)
        if decoded != self._canonical:
            raise ValueError("compiled context is not lossless")
        agent_parts: list[str] = []
        human_parts: list[str] = []
        manifest: dict[str, JsonValue] = {}
        for section in self._sections:
            title = cast(str, section["title"])
            kind = cast(SectionKind, section["kind"])
            view = section["view"]
            if kind == "json":
                rendered = render_properties(view)
            elif kind == "common_rows":
                rendered = render_common_rows(cast(Mapping[str, Any], view))
            elif kind == "working_memory":
                rendered = render_working_memory(cast(Mapping[str, Any], view))
            else:
                rendered = render_control_specification(cast(Mapping[str, Any], view))
            agent_parts.append(f"### {title}\n{rendered}")
            human_parts.append(f"### {title}\n{rendered}")
            manifest[title] = cast(list[JsonValue], _field_manifest(self._canonical[title]))
        audit: dict[str, JsonValue] = {
            "schema": "h3c_context_audit_v1",
            "canonical_sha256": _hash(self._canonical),
            "compact_sha256": _hash(cast(JsonValue, self._sections)),
            "field_manifest": manifest,
            "round_trip_equal": True,
            "internal_context": copy.deepcopy(self._audit_metadata),
            "internal_coordinates": cast(JsonValue, _internal_coordinates(self._canonical)),
        }
        return CompiledContext(
            agent_view="\n\n".join(agent_parts) + ("\n" if agent_parts else ""),
            human_view="\n\n".join(human_parts) + ("\n" if human_parts else ""),
            canonical_ir=copy.deepcopy(self._canonical),
            compact_view=copy.deepcopy(self._sections),
            audit_view=audit,
        )


def decode_compact_context(sections: Sequence[Mapping[str, Any]]) -> dict[str, JsonValue]:
    """Decode all compact sections to their canonical typed values."""
    decoded: dict[str, JsonValue] = {}
    for section in sections:
        title = str(section["title"])
        if title in decoded:
            raise ValueError(f"duplicate context-section owner: {title}")
        kind = str(section["kind"])
        view = section["view"]
        if kind == "json":
            decoded[title] = _normalize(view)
        elif kind == "common_rows":
            decoded[title] = cast(JsonValue, decode_common_rows(cast(Mapping[str, Any], view)))
        elif kind == "working_memory":
            decoded[title] = cast(JsonValue, decode_working_memory(cast(Mapping[str, Any], view)))
        elif kind == "control_specification":
            decoded[title] = _normalize(view)
        else:
            raise ValueError(f"unknown compact section kind: {kind}")
    return decoded
