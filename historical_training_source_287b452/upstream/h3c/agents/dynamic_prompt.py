"""Cooling-only dynamic prompt projections and historical compact rendering."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

from h3c.control.program import (
    ACTUATOR_BOUNDS_C,
    MAX_PROGRAM_RULES,
    OCCUPIED_BASE_SETPOINT_C,
    PARAMETER_BOUNDS,
    RESIDUAL_BOUNDS_C,
    RULE_ACTIONS,
    RULE_OPERATORS,
    UNOCCUPIED_BASE_SETPOINT_C,
    condition_fields,
)
from h3c.runtime.comfort import COMFORT_BAND

DISPLAY_NAMES = {
    "site_residual_c": "shared_pool_available_before_settlement_c",
    "remaining_c": "zone_reserved_allowance_c",
    "priority_rank": "shared_settlement_priority_rank",
    "granted_c": "total_reserved_allowance_c",
    "used_c": "total_reserved_consumption_c",
    "residual_initial_c": "initial_shared_unreserved_pool_c",
    "residual_left_c": "remaining_shared_unreserved_pool_c",
    "residual_used_by": "shared_pool_consumption_by_zone_c",
    "per_zone_max_c": "most_any_one_zone_may_be_given_c",
    "site_max_c": "most_the_whole_group_may_be_given_c",
    "orch_fallback_reason": "why_this_allowance_was_reused",
}

GLOSSARY = {
    "comfort_headroom_c": (
        "how far zone temperature would have to rise, or fall, to reach the band edge. "
        "Room in zone temperature, not in the setpoint; the edge of the score, not of "
        "what people feel"
    ),
    "occupied_share_with_no_room_to_ask_for_more": (
        "share of occupied steps with the actuator already at the end that asks most of "
        "the shared equipment. Its own knob cannot ask for more"
    ),
    "occupied_share_with_no_room_to_ask_for_less": (
        "the same at the other end. A zone can sit here and still be served more than it "
        "asked for, because the shared side is set by the others"
    ),
    "mean_pmv": (
        "the mean PMV recorded for the represented observations; it does not establish "
        "a comfort judgment or a target"
    ),
    "mean_residual_c": (
        "the mean residual value recorded for the represented observations; it does not "
        "establish a control direction or target"
    ),
    "neg_param": (
        "the name of a numeric parameter whose value is used with the opposite sign; "
        "it does not establish which action to take"
    ),
    "zone_h": "how long comfort spent outside the band. Counts time, not depth",
    "pmv_h": (
        "how far comfort went past the band, summed over the time it was past. "
        "Sizes excursions; not when they happened"
    ),
    "code": "a short label for why a patch was not accepted",
    "program_version": (
        "the program-version label recorded with a result; it does not establish result "
        "quality or control suitability"
    ),
    "outdoor_temp_change_next_1h_c": (
        "outdoor dry-bulb temperature at the fourth future 15-minute step minus the "
        "current value, in degrees Celsius; it describes the forecast change and does not "
        "prescribe an action"
    ),
    "solar_irr_max_next_1h_w_m2": (
        "the maximum forecast solar irradiance over the next four 15-minute steps, in "
        "watts per square metre; it describes the forecast magnitude and does not "
        "prescribe an action"
    ),
    "solar_irr_mean_next_1h_w_m2": (
        "the mean forecast solar irradiance over the next four 15-minute steps, in watts "
        "per square metre; it describes the forecast magnitude and does not prescribe an "
        "action"
    ),
}

EXECUTOR_MEMORY_COLUMNS = (
    "time",
    "observation",
    "proposal",
    "validation",
    "outcome",
    "program version",
)

COOLING_CONTROL_DOMAIN = {
    "domain_id": "cooling",
    "controlled_setpoint": "zone cooling setpoint",
    "unit": "degC",
    "hard_bounds_c": list(ACTUATOR_BOUNDS_C),
    "residual_bounds_c": list(RESIDUAL_BOUNDS_C),
    "occupied_base_c": OCCUPIED_BASE_SETPOINT_C,
    "unoccupied_base_c": UNOCCUPIED_BASE_SETPOINT_C,
    "abs_pmv_score_limit": COMFORT_BAND,
    "preconditioning": {"label": "precool", "lead_steps": 4, "target_c": 25.0},
    "energy_intensive_setpoint_direction": "decrease",
}


def omit_missing(value: Any) -> Any:
    """Recursively omit unavailable values while preserving measured zero and false."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            projected = omit_missing(item)
            if projected is None:
                continue
            if isinstance(projected, (dict, list, str)) and not projected:
                continue
            result[str(key)] = projected
        return result
    if isinstance(value, (list, tuple)):
        sequence_result: list[Any] = []
        for item in value:
            projected = omit_missing(item)
            if projected is not None:
                sequence_result.append(projected)
        return sequence_result
    return value


def strip_audit_fields(value: Any) -> Any:
    """Remove internal program hashes before any value reaches a model."""
    if isinstance(value, Mapping):
        return {
            key: strip_audit_fields(item) for key, item in value.items() if key != "program_hash"
        }
    if isinstance(value, list):
        return [strip_audit_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(strip_audit_fields(item) for item in value)
    return value


def display(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {DISPLAY_NAMES.get(str(key), str(key)): display(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [display(item) for item in value]
    return value


def _leaf_count(value: Mapping[str, Any]) -> int:
    return sum(_leaf_count(item) if isinstance(item, Mapping) else 1 for item in value.values())


def _flatten(
    value: Mapping[str, Any], result: dict[str, Any] | None = None, prefix: str = ""
) -> dict[str, Any]:
    flattened = {} if result is None else result
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, Mapping):
            _flatten(item, flattened, name + ".")
        elif isinstance(item, list):
            flattened[name] = json.dumps(
                item, ensure_ascii=False, separators=(",", ":"), default=float
            )
        else:
            flattened[name] = item
    return flattened


def table(rows: Sequence[Any]) -> str | None:
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        return None
    flattened: list[dict[str, Any]] = []
    for row in rows:
        current = _flatten(row)
        if len(current) != _leaf_count(row):
            return None
        flattened.append(current)
    columns: list[str] = []
    for row in flattened:
        for key in row:
            if key not in columns:
                columns.append(key)

    def cell(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=float).replace(
            "|", "\\u007c"
        )

    def header(value: str) -> str:
        return value.replace("|", "\\u007c").replace("\n", "\\n").replace("\r", "\\r")

    cells = [
        [cell(row[column]) if column in row else "" for column in columns] for row in flattened
    ]
    if any("\t" in item or "\n" in item or "\r" in item for row in cells for item in row):
        return None
    return "\n".join(
        [
            "| " + " | ".join(header(column) for column in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *("| " + " | ".join(row) + " |" for row in cells),
        ]
    )


def render_executor_working_memory(records: Sequence[Mapping[str, Any]] | None) -> str:
    """Render the final W six-group completed-step memory without audit-only fields."""
    if not records:
        return ""
    projections: list[dict[str, Any]] = []
    for record in records:
        required = {"time", "observation", "proposal", "validation", "outcome", "program"}
        if not required <= set(record):
            raise ValueError("Executor working-memory record is incomplete")
        projections.append(
            {
                "time": copy.deepcopy(record["time"]),
                "observation": copy.deepcopy(record["observation"]),
                "proposal": copy.deepcopy(record["proposal"]),
                "validation": copy.deepcopy(record["validation"]),
                "outcome": copy.deepcopy(record["outcome"]),
                "program version": copy.deepcopy(record["program"]),
            }
        )

    def cell(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=float).replace(
            "|", "\\u007c"
        )

    header = "| " + " | ".join(EXECUTOR_MEMORY_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in EXECUTOR_MEMORY_COLUMNS) + " |"
    rows = [
        "| " + " | ".join(cell(projection[column]) for column in EXECUTOR_MEMORY_COLUMNS) + " |"
        for projection in projections
    ]
    return "\n".join((header, separator, *rows))


def block(title: str, body: Any, *, tabular: bool = False) -> str:
    if body is None:
        return ""
    if tabular and isinstance(body, (list, tuple)):
        rendered = table(display(omit_missing(body)))
        if rendered:
            return f"\n### {title}\n{rendered}\n"
    if isinstance(body, str):
        text = body
    else:
        cleaned = display(omit_missing(body))
        if isinstance(cleaned, (dict, list)) and not cleaned:
            return ""
        text = "```json\n{}\n```".format(
            json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"), default=float)
        )
    text = text.strip()
    if not text or text in ("{}", "[]"):
        return ""
    return f"\n### {title}\n{text}\n"


def merged_block(title: str, parts: Sequence[tuple[str, Any, bool]]) -> str:
    chunks: list[str] = []
    for label, body, tabular in parts:
        rendered = block("_", body, tabular=tabular)
        if rendered:
            chunks.append(f"{label}:\n{rendered.split(chr(10), 2)[2].rstrip(chr(10))}")
    if not chunks:
        return ""
    return f"\n### {title}\n" + "\n".join(chunks) + "\n"


def visible_glossary_lines(*objects: Any) -> str | None:
    names: set[str] = set()

    def collect(value: Any) -> None:
        projected = display(omit_missing(value))
        if isinstance(projected, Mapping):
            for key, item in projected.items():
                names.add(str(key))
                collect(item)
        elif isinstance(projected, list):
            for item in projected:
                collect(item)

    for value in objects:
        collect(value)
    visible = sorted(names & set(GLOSSARY))
    return "\n".join(f"{name} -- {GLOSSARY[name]}" for name in visible) or None


def executor_observation_view(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Project runtime names to the final cooling production observation surface."""
    keys = (
        ("current_occupancy", "current_occupancy"),
        ("last_occupancy", "last_occupancy"),
        ("occ_ahead", "occupancy_next_steps"),
        ("zone_temperature_c", "zone_temp_c"),
        ("last_pmv", "last_pmv"),
        ("last_setpoint", "last_setpoint_c"),
        ("outdoor_temp_c", "outdoor_temp_c"),
        ("solar_irr", "solar_irr"),
        ("electricity_price", "price_now"),
        ("comfort_headroom_c", "comfort_headroom_c"),
        ("outdoor_temp_change_next_1h_c", "outdoor_temp_change_next_1h_c"),
        ("solar_irr_max_next_1h_w_m2", "solar_irr_max_next_1h_w_m2"),
        ("solar_irr_mean_next_1h_w_m2", "solar_irr_mean_next_1h_w_m2"),
        ("weather_next_steps", "weather_next_steps"),
    )
    return {
        exposed: copy.deepcopy(observation[internal])
        for internal, exposed in keys
        if internal in observation
    }


def parameter_rule_limits() -> dict[str, Any]:
    sources = {
        "precool_lead_steps": "visible forecast horizon",
        "precool_residual_c": "residual safety bound",
        "pmv_band_lo": "PMV evaluation range",
        "pmv_band_hi": "PMV evaluation range",
        "pmv_step_c": "residual safety bound",
    }
    limits: dict[str, Any] = {
        name: {"bounds": [bounds[0], bounds[1]], "bounds_come_from": sources[name]}
        for name, bounds in PARAMETER_BOUNDS.items()
    }
    limits["a_rule_you_add"] = {
        "conditions_may_test": sorted(condition_fields()),
        "comparisons": list(RULE_OPERATORS),
        "compared_against": "a number, or the name of a parameter above",
        "actions": list(RULE_ACTIONS),
        "most_rules_at_once": MAX_PROGRAM_RULES,
    }
    limits["weather_condition_values"] = {
        "must_be": "finite numeric literals",
        "parameter_references": "not allowed",
    }
    return limits
