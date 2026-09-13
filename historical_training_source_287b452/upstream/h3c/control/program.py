"""Canonical W-only executable cooling program and deterministic interpreter."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from h3c.agents.contracts import PATCH_OPERATIONS, patch_contract

SPEC_VERSION = 2
OCCUPIED_BASE_SETPOINT_C = 25.0
UNOCCUPIED_BASE_SETPOINT_C = 30.0
RESIDUAL_BOUNDS_C = (-5.0, 5.0)
ACTUATOR_BOUNDS_C = (20.0, 30.0)
MAX_PROGRAM_RULES = 8
PARAMETER_BOUNDS: dict[str, tuple[float, float, type[int] | type[float]]] = {
    "precool_lead_steps": (0, 4, int),
    "precool_residual_c": (-5.0, 5.0, float),
    "pmv_band_lo": (-0.5, 0.5, float),
    "pmv_band_hi": (-0.5, 0.5, float),
    "pmv_step_c": (0.0, 5.0, float),
}
BASE_CONDITION_FIELDS = ("occupied_now", "occupied_last", "precool_due", "last_pmv")
WEATHER_CONDITION_FIELDS = (
    "outdoor_temp_change_next_1h_c",
    "solar_irr_max_next_1h_w_m2",
    "solar_irr_mean_next_1h_w_m2",
)
WEATHER_DRIVER_NODES = {
    "outdoor_temp_change_next_1h_c": "outdoor_temp",
    "solar_irr_max_next_1h_w_m2": "solar_irr",
    "solar_irr_mean_next_1h_w_m2": "solar_irr",
}
RULE_PATCH_OPERATIONS = ("add_rule", "replace_rule", "remove_rule", "move_rule")
RULE_OPERATORS = ("<", "<=", ">", ">=", "==", "!=")
RULE_ACTIONS = ("set_residual", "step_setpoint", "hold_setpoint")
_RULE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,23}$")
MAX_DIRECTION_PROOF_WITNESSES = 250_000


class ProgramError(ValueError):
    """Raised when a program or patch violates its deterministic contract."""

    def __init__(self, message: str, *, code: str = "program_invalid") -> None:
        super().__init__(message)
        self.code = code


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgramError(f"{name} must be numeric", code="invalid_number")
    number = float(value)
    if not math.isfinite(number):
        raise ProgramError(f"{name} must be finite", code="invalid_number")
    return number


def _resolved_value(program: Mapping[str, Any], value: Any) -> float:
    if isinstance(value, Mapping):
        if set(value) == {"neg_param"} and value["neg_param"] in PARAMETER_BOUNDS:
            return -float(program["params"][value["neg_param"]])
        if set(value) == {"param"} and value["param"] in PARAMETER_BOUNDS:
            return float(program["params"][value["param"]])
        raise ProgramError("value parameter reference is invalid", code="invalid_parameter")
    return _finite_number(value, "rule value")


def load_program(path: str | Path, zone: str) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    program = {
        key: copy.deepcopy(raw[key])
        for key in ("spec_version", "precool_trigger", "params", "rules")
    }
    program["zone"] = zone
    program["provenance"] = {}
    validate_program(program)
    return program


def program_hash(program: Mapping[str, Any]) -> str:
    payload = {key: program[key] for key in ("spec_version", "precool_trigger", "params", "rules")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def condition_fields(*, weather_enabled: bool = True) -> tuple[str, ...]:
    return BASE_CONDITION_FIELDS + (WEATHER_CONDITION_FIELDS if weather_enabled else ())


def regime_base_setpoint(current_occupancy: Any) -> float:
    """Return the executable program base for one currently observed regime."""
    return (
        OCCUPIED_BASE_SETPOINT_C
        if _finite_number(current_occupancy, "current occupancy") > 0
        else UNOCCUPIED_BASE_SETPOINT_C
    )


def setpoint_effect_facts(
    *, current_occupancy: Any, applied_setpoint_c: Any
) -> dict[str, float | str]:
    """Derive model-facing setpoint facts without influencing program execution."""
    base = regime_base_setpoint(current_occupancy)
    offset = round(_finite_number(applied_setpoint_c, "applied setpoint") - base, 10)
    if abs(offset) < 1e-9:
        effect = "at_regime_base"
        offset = 0.0
    elif offset < 0:
        effect = "more_cooling_than_regime_base"
    else:
        effect = "less_cooling_than_regime_base"
    return {
        "regime_base_setpoint_c": base,
        "setpoint_offset_from_regime_base_c": offset,
        "cooling_effect_relative_to_regime_base": effect,
    }


def interpreter_semantics() -> dict[str, Any]:
    """Return the concise Agent view of the same semantics used by ``run_program``."""
    return {
        "rule_evaluation": "top_to_bottom_first_matching_rule_only",
        "action_formulas": {
            "set_residual": "candidate_setpoint_c = regime_base_setpoint_c + value",
            "step_setpoint": (
                "candidate_setpoint_c = last_physical_setpoint_c + value when "
                "previous_occupancy > 0; otherwise regime_base_setpoint_c + value"
            ),
            "hold_setpoint": "candidate_setpoint_c = last_physical_setpoint_c",
        },
        "postprocessing": (
            "clip candidate offset from regime_base_setpoint_c to residual_bounds_c, "
            "then clip candidate_setpoint_c to hard_bounds_c"
        ),
    }


def _execute_rule_action(
    program: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    current_occupancy: float,
    previous_occupancy: float,
    last_physical_setpoint_c: float,
) -> dict[str, Any]:
    """Apply one rule action using the same arithmetic as the program interpreter."""
    base = regime_base_setpoint(current_occupancy)
    operation = str(action["op"])
    resolved_value: float | None = None
    if operation == "set_residual":
        resolved_value = _resolved_value(program, action["value"])
        candidate = base + resolved_value
        residual = resolved_value
        formula = "regime_base_setpoint_c + resolved_value_c"
        depends_on_last = False
    elif operation == "step_setpoint":
        resolved_value = _resolved_value(program, action["value"])
        step_base = last_physical_setpoint_c if previous_occupancy > 0 else base
        candidate = step_base + resolved_value
        residual = candidate - base
        formula = (
            "last_physical_setpoint_c + resolved_value_c"
            if previous_occupancy > 0
            else "regime_base_setpoint_c + resolved_value_c"
        )
        depends_on_last = previous_occupancy > 0
    elif operation == "hold_setpoint":
        candidate = last_physical_setpoint_c
        residual = candidate - base
        formula = "last_physical_setpoint_c"
        depends_on_last = True
    else:
        raise ProgramError("rule action is unsupported", code="rule_invalid")

    residual = max(RESIDUAL_BOUNDS_C[0], min(RESIDUAL_BOUNDS_C[1], residual))
    setpoint = max(ACTUATOR_BOUNDS_C[0], min(ACTUATOR_BOUNDS_C[1], base + residual))
    return {
        "operation": operation,
        **({"resolved_value_c": resolved_value} if resolved_value is not None else {}),
        "formula": formula,
        "depends_on_last_physical_setpoint": depends_on_last,
        "regime_base_setpoint_c": base,
        "candidate_setpoint_c": candidate,
        "residual_after_clamp_c": residual,
        "final_setpoint_after_existing_clips_c": setpoint,
    }


def _binary_condition_matches(
    program: Mapping[str, Any], condition: Mapping[str, Any], value: float
) -> bool:
    right = _resolved_value(program, condition["value"])
    return {
        "<": value < right,
        "<=": value <= right,
        ">": value > right,
        ">=": value >= right,
        "==": abs(value - right) < 1e-9,
        "!=": abs(value - right) >= 1e-9,
    }[str(condition["op"])]


def rule_effects_if_matched(
    program: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Project each rule's effect in the occupancy states where that rule can match.

    These are explanatory facts only. They do not participate in matching, admission,
    settlement, or physical execution.
    """
    last_setpoint = _finite_number(observation["last_setpoint"], "last setpoint")

    def visible_number(value: Any) -> float:
        number = round(_finite_number(value, "projected interpreter value"), 10)
        return 0.0 if abs(number) < 1e-9 else number

    rows_by_action: dict[str, list[dict[str, Any]]] = {action: [] for action in RULE_ACTIONS}
    for rule_order_index, rule in enumerate(program["rules"]):
        occupancy_conditions = [
            condition
            for condition in rule["when"]
            if condition["field"] in {"occupied_now", "occupied_last"}
        ]
        allowed_current = [
            value
            for value in (0.0, 1.0)
            if all(
                _binary_condition_matches(program, condition, value)
                for condition in occupancy_conditions
                if condition["field"] == "occupied_now"
            )
        ]
        allowed_previous = [
            value
            for value in (0.0, 1.0)
            if all(
                _binary_condition_matches(program, condition, value)
                for condition in occupancy_conditions
                if condition["field"] == "occupied_last"
            )
        ]
        for current, previous in itertools.product(allowed_current, allowed_previous):
            executed = _execute_rule_action(
                program,
                rule["then"],
                current_occupancy=current,
                previous_occupancy=previous,
                last_physical_setpoint_c=last_setpoint,
            )
            effect = setpoint_effect_facts(
                current_occupancy=current,
                applied_setpoint_c=executed["final_setpoint_after_existing_clips_c"],
            )
            rows_by_action[str(rule["then"]["op"])].append(
                {
                    "rule_id": rule["id"],
                    "rule_order_index": rule_order_index,
                    "matching_current_occupancy": int(current),
                    "matching_previous_occupancy": int(previous),
                    **(
                        {"resolved_value_c": visible_number(executed["resolved_value_c"])}
                        if "resolved_value_c" in executed
                        else {}
                    ),
                    "regime_base_setpoint_c": visible_number(executed["regime_base_setpoint_c"]),
                    "formula": executed["formula"],
                    "last_setpoint_basis": (
                        "uses_visible_current_last_physical_setpoint"
                        if executed["depends_on_last_physical_setpoint"]
                        else "independent_of_visible_current_last_physical_setpoint"
                    ),
                    "visible_current_last_physical_setpoint_c": visible_number(last_setpoint),
                    "candidate_setpoint_c": visible_number(executed["candidate_setpoint_c"]),
                    "interpreter_setpoint_c_after_residual_and_hard_clips_before_assurance": visible_number(
                        executed["final_setpoint_after_existing_clips_c"]
                    ),
                    "offset_from_regime_base_c": visible_number(
                        effect["setpoint_offset_from_regime_base_c"]
                    ),
                    "cooling_effect_relative_to_regime_base": effect[
                        "cooling_effect_relative_to_regime_base"
                    ],
                }
            )
    return rows_by_action


def current_interpreter_derivation(
    program: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Explain the current program result using the same executable interpreter."""
    result = run_program(program, observation)
    matched_rule = result["matched_rule"]
    matched_action: dict[str, Any] | None = None
    if matched_rule is not None:
        rule = next(item for item in program["rules"] if item["id"] == matched_rule)
        action = rule["then"]
        matched_action = {"op": action["op"]}
        if action["op"] != "hold_setpoint":
            matched_action["resolved_value_c"] = _resolved_value(program, action["value"])
    last_setpoint = _finite_number(observation["last_setpoint"], "last setpoint")
    setpoint_delta = round(float(result["setpoint"]) - last_setpoint, 10)
    if abs(setpoint_delta) < 1e-9:
        direction = "unchanged_from_last_physical_setpoint"
        setpoint_delta = 0.0
    elif setpoint_delta < 0:
        direction = "more_cooling_than_last_physical_setpoint"
    else:
        direction = "less_cooling_than_last_physical_setpoint"
    return {
        "rule_match_status": "matched" if matched_rule is not None else "no_match_residual_zero",
        **({"first_matching_rule_id": matched_rule} if matched_rule is not None else {}),
        **({"first_matching_action": matched_action} if matched_action is not None else {}),
        "regime_base_setpoint_c": result["base_setpoint"],
        "last_physical_setpoint_c": last_setpoint,
        "residual_after_clamp_c": result["residual"],
        "interpreter_setpoint_before_assurance_c": result["setpoint"],
        "setpoint_change_from_last_physical_c": setpoint_delta,
        "cooling_effect_relative_to_last_physical_setpoint": direction,
        "interpreter_branch": result["branch"],
    }


def validate_program(
    program: Mapping[str, Any], *, weather_enabled: bool = True
) -> Mapping[str, Any]:
    if program.get("spec_version") != SPEC_VERSION:
        raise ProgramError("unsupported program schema", code="schema_invalid")
    if program.get("precool_trigger") != "rolling":
        raise ProgramError("preconditioning trigger must be rolling", code="trigger_invalid")
    params = program.get("params")
    if not isinstance(params, Mapping) or set(params) != set(PARAMETER_BOUNDS):
        raise ProgramError(
            "program parameters do not match the canonical set", code="params_invalid"
        )
    for name, (lower, upper, kind) in PARAMETER_BOUNDS.items():
        value = _finite_number(params[name], name)
        if kind is int and value != int(value):
            raise ProgramError(f"{name} must be an integer", code="out_of_box")
        if not lower <= value <= upper:
            raise ProgramError(f"{name} is outside its bound", code="out_of_box")
    if not float(params["pmv_band_lo"]) < float(params["pmv_band_hi"]):
        raise ProgramError("PMV lower band must be below upper band", code="out_of_box")

    allowed_fields = set(condition_fields(weather_enabled=weather_enabled))
    rules = program.get("rules")
    if not isinstance(rules, list) or len(rules) > MAX_PROGRAM_RULES:
        raise ProgramError("rules must be a list with at most eight entries", code="rules_invalid")
    seen_ids: set[str] = set()
    seen_shapes: set[tuple[tuple[str, str, str], ...]] = set()
    for rule in rules:
        if not isinstance(rule, Mapping) or set(rule) != {"id", "when", "then"}:
            raise ProgramError("each rule must contain id, when, and then", code="rule_invalid")
        rule_id = rule["id"]
        if not isinstance(rule_id, str) or not _RULE_ID.fullmatch(rule_id) or rule_id in seen_ids:
            raise ProgramError("rule identifiers must be unique and short", code="rule_invalid")
        seen_ids.add(rule_id)
        conditions = rule["when"]
        if not isinstance(conditions, list) or not conditions:
            raise ProgramError("a rule needs at least one condition", code="rule_invalid")
        shape: list[tuple[str, str, str]] = []
        for condition in conditions:
            if not isinstance(condition, Mapping) or set(condition) != {"field", "op", "value"}:
                raise ProgramError(
                    "conditions must contain field, op, and value", code="rule_invalid"
                )
            if condition["field"] not in allowed_fields or condition["op"] not in RULE_OPERATORS:
                raise ProgramError(
                    "condition field or operator is unsupported", code="rule_invalid"
                )
            if condition["field"] in WEATHER_CONDITION_FIELDS and isinstance(
                condition["value"], Mapping
            ):
                raise ProgramError(
                    "weather conditions require numeric literals", code="rule_invalid"
                )
            _resolved_value(program, condition["value"])
            shape.append(
                (
                    str(condition["field"]),
                    str(condition["op"]),
                    json.dumps(condition["value"], sort_keys=True),
                )
            )
        shape_key = tuple(sorted(shape))
        if shape_key in seen_shapes:
            raise ProgramError("duplicate rule conditions", code="rule_invalid")
        seen_shapes.add(shape_key)
        action = rule["then"]
        if not isinstance(action, Mapping) or action.get("op") not in RULE_ACTIONS:
            raise ProgramError("rule action is unsupported", code="rule_invalid")
        if action["op"] == "hold_setpoint":
            if set(action) != {"op"}:
                raise ProgramError("hold_setpoint carries no value", code="rule_invalid")
        elif set(action) != {"op", "value"}:
            raise ProgramError("rule action requires a value", code="rule_invalid")
        else:
            value = _resolved_value(program, action["value"])
            if not RESIDUAL_BOUNDS_C[0] <= value <= RESIDUAL_BOUNDS_C[1]:
                raise ProgramError("rule action is outside residual bounds", code="out_of_box")
    return program


def validate_patch_shape(
    patch: Any, *, causal_enabled: bool = True, allow_internal: bool = False
) -> None:
    if not isinstance(patch, Mapping):
        raise ProgramError("patch must be an object", code="patch_not_an_object")
    operation = patch.get("op")
    if operation not in PATCH_OPERATIONS:
        raise ProgramError("operation is not supported", code="unknown_operation")
    contract = patch_contract(causal_enabled=causal_enabled)["operations"][operation]
    allowed = set(contract["required"]) | set(contract["optional"])
    if allow_internal:
        allowed |= {"expected_effects", "consistent_program_direction_proof"}
    unknown = sorted(set(patch) - allowed)
    if unknown:
        raise ProgramError(f"unknown patch fields: {unknown}", code="unknown_patch_fields")
    missing = [field for field in contract["required"] if field not in patch]
    if missing:
        raise ProgramError(f"patch is missing required fields: {missing}", code="missing_field")
    rationale = patch.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ProgramError("rationale must be a nonempty string", code="invalid_rationale")
    if causal_enabled and operation != "no_change":
        identifiers = patch.get("causal_edge_ids")
        if (
            not isinstance(identifiers, list)
            or not identifiers
            or len(identifiers) != len(set(identifiers))
            or any(not isinstance(item, str) for item in identifiers)
        ):
            raise ProgramError("causal edge identifiers are malformed", code="missing_field")


def apply_patch(
    program: Mapping[str, Any],
    patch: Mapping[str, Any],
    *,
    causal_enabled: bool = True,
    weather_enabled: bool = True,
) -> dict[str, Any]:
    validate_patch_shape(patch, causal_enabled=causal_enabled, allow_internal=True)
    operation = str(patch["op"])
    if operation == "no_change":
        return copy.deepcopy(dict(program))
    candidate = copy.deepcopy(dict(program))
    rules = candidate["rules"]
    if operation == "set_param":
        parameter = patch.get("param")
        if parameter not in PARAMETER_BOUNDS:
            raise ProgramError("unknown program parameter", code="unknown_param")
        if candidate["params"][parameter] == patch.get("to"):
            raise ProgramError("patch has no executable effect", code="semantic_noop")
        candidate["params"][parameter] = patch.get("to")
    elif operation in ("add_rule", "replace_rule"):
        rule = patch.get("rule")
        if not isinstance(rule, Mapping):
            raise ProgramError("rule patch is invalid", code="rule_invalid")
        identifiers = [item["id"] for item in rules]
        rule_id = rule.get("id")
        if operation == "add_rule":
            if rule_id in identifiers:
                raise ProgramError("rule already exists", code="rule_invalid")
            index = patch.get("index", len(rules))
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index <= len(rules)
            ):
                raise ProgramError("rule insertion index is invalid", code="rule_invalid")
            rules.insert(index, copy.deepcopy(dict(rule)))
        else:
            if rule_id not in identifiers:
                raise ProgramError("rule does not exist", code="rule_invalid")
            position = identifiers.index(rule_id)
            if rules[position] == rule:
                raise ProgramError("patch has no executable effect", code="semantic_noop")
            rules[position] = copy.deepcopy(dict(rule))
    elif operation == "remove_rule":
        identifiers = [item["id"] for item in rules]
        if patch.get("id") not in identifiers:
            raise ProgramError("rule does not exist", code="rule_invalid")
        rules.pop(identifiers.index(patch["id"]))
    else:
        identifiers = [item["id"] for item in rules]
        destination = patch.get("to_index")
        if (
            patch.get("id") not in identifiers
            or isinstance(destination, bool)
            or not isinstance(destination, int)
            or not 0 <= destination < len(rules)
        ):
            raise ProgramError("rule movement is invalid", code="rule_invalid")
        source = identifiers.index(patch["id"])
        if source == destination:
            raise ProgramError("patch has no executable effect", code="semantic_noop")
        rules.insert(destination, rules.pop(source))
    validate_program(candidate, weather_enabled=weather_enabled)
    if not program_delta(program, candidate, weather_enabled=weather_enabled)["changed"]:
        raise ProgramError("patch has no executable effect", code="semantic_noop")
    return candidate


def _state(program: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, float]:
    occupied = float(observation["current_occupancy"]) > 0
    lead = int(program["params"]["precool_lead_steps"])
    due = any(float(value) > 0 for value in (observation.get("occ_ahead") or [])[:lead])
    state = {
        "occupied_now": 1.0 if occupied else 0.0,
        "occupied_last": 1.0 if float(observation["last_occupancy"]) > 0 else 0.0,
        "precool_due": 1.0 if not occupied and due else 0.0,
        "last_pmv": float(observation["last_pmv"]),
    }
    used_weather = {
        condition["field"]
        for rule in program["rules"]
        for condition in rule["when"]
        if condition["field"] in WEATHER_CONDITION_FIELDS
    }
    for field in used_weather:
        if field not in observation:
            raise ProgramError(f"weather condition input is missing: {field}", code="missing_field")
        state[field] = _finite_number(observation[field], field)
    return state


def _matches(
    program: Mapping[str, Any], rule: Mapping[str, Any], state: Mapping[str, float]
) -> bool:
    for condition in rule["when"]:
        left = state[condition["field"]]
        right = _resolved_value(program, condition["value"])
        operation = condition["op"]
        matched = {
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
            "==": abs(left - right) < 1e-9,
            "!=": abs(left - right) >= 1e-9,
        }[operation]
        if not matched:
            return False
    return True


def run_program(program: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    state = _state(program, observation)
    base = regime_base_setpoint(state["occupied_now"])
    residual = 0.0
    matched_rule: str | None = None
    for rule in program["rules"]:
        if not _matches(program, rule, state):
            continue
        executed = _execute_rule_action(
            program,
            rule["then"],
            current_occupancy=state["occupied_now"],
            previous_occupancy=state["occupied_last"],
            last_physical_setpoint_c=float(observation["last_setpoint"]),
        )
        residual = float(executed["residual_after_clamp_c"])
        setpoint = float(executed["final_setpoint_after_existing_clips_c"])
        matched_rule = str(rule["id"])
        break
    else:
        residual = max(RESIDUAL_BOUNDS_C[0], min(RESIDUAL_BOUNDS_C[1], residual))
        setpoint = max(ACTUATOR_BOUNDS_C[0], min(ACTUATOR_BOUNDS_C[1], base + residual))
    rate_base = (
        float(observation["last_setpoint"])
        if state["occupied_now"] and state["occupied_last"]
        else base
    )
    if matched_rule == "precool":
        branch = "preconditioning"
    elif not state["occupied_now"]:
        branch = "unoccupied"
    else:
        branch_by_rule = {
            "pmv_raise": "occupied_pmv_raise",
            "pmv_lower": "occupied_pmv_lower",
        }
        branch = branch_by_rule.get(
            matched_rule if matched_rule is not None else "", "occupied_pmv_in_band"
        )
        if not state["occupied_last"]:
            branch = "occupancy_onset_" + branch
    return {
        "setpoint": setpoint,
        "base_setpoint": base,
        "base": rate_base,
        "residual": residual,
        "matched_rule": matched_rule,
        "branch": branch,
        "rules_fired": [matched_rule] if matched_rule is not None else [],
        "exempt_rate": not bool(state["occupied_now"]),
    }


def _legacy_witnesses(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    pmv_values: set[float] = {0.0}
    for program in (left, right):
        for rule in program["rules"]:
            for condition in rule["when"]:
                if condition["field"] == "last_pmv":
                    value = _resolved_value(program, condition["value"])
                    pmv_values.update((value - 1e-6, value, value + 1e-6))
    cuts = sorted(pmv_values)
    pmv_values.update((cuts[index] + cuts[index + 1]) / 2 for index in range(len(cuts) - 1))
    max_lead = max(
        int(left["params"]["precool_lead_steps"]),
        int(right["params"]["precool_lead_steps"]),
    )
    ahead_patterns = [tuple(0.0 for _ in range(max_lead))]
    ahead_patterns.extend(
        tuple(1.0 if index == active else 0.0 for index in range(max_lead))
        for active in range(max_lead)
    )
    ahead_patterns.append(tuple(1.0 for _ in range(max_lead)))
    weather_values: dict[str, set[float]] = {field: {0.0} for field in WEATHER_CONDITION_FIELDS}
    for program in (left, right):
        for rule in program["rules"]:
            for condition in rule["when"]:
                field = condition["field"]
                if field in weather_values:
                    value = _resolved_value(program, condition["value"])
                    weather_values[field].update((value - 1.0, value, value + 1.0))
    weather_axes = [sorted(weather_values[field]) for field in WEATHER_CONDITION_FIELDS]
    witness_upper_bound = (
        4
        * len(pmv_values)
        * 3
        * len(ahead_patterns)
        * math.prod(len(axis) for axis in weather_axes)
    )
    if witness_upper_bound > MAX_DIRECTION_PROOF_WITNESSES:
        raise ProgramError(
            "exact direction proof exceeds the registered witness safety bound",
            code="direction_proof_complexity_limit",
        )
    for occupied_now, occupied_last, pmv in itertools.product(
        (0.0, 1.0), (0.0, 1.0), sorted(pmv_values)
    ):
        for previous in (20.0, 25.0, 30.0):
            for ahead in ahead_patterns:
                for weather_row in itertools.product(*weather_axes):
                    yield {
                        "current_occupancy": occupied_now,
                        "last_occupancy": occupied_last,
                        "last_pmv": pmv,
                        "last_setpoint": previous,
                        "next_hour_occupancy": 1.0 if any(ahead) else 0.0,
                        "occ_ahead": list(ahead),
                        **dict(zip(WEATHER_CONDITION_FIELDS, weather_row, strict=True)),
                    }


def program_delta(
    left: Mapping[str, Any], right: Mapping[str, Any], *, weather_enabled: bool = True
) -> dict[str, Any]:
    del weather_enabled
    witnesses: list[dict[str, Any]] = []
    directions: set[int] = set()
    maximum_energy_intensive = 0.0
    for observation in _legacy_witnesses(left, right):
        before_result = run_program(left, observation)
        after_result = run_program(right, observation)
        before = float(before_result["setpoint"])
        after = float(after_result["setpoint"])
        if abs(after - before) <= 1e-9:
            continue
        directions.add(1 if after > before else -1)
        maximum_energy_intensive = max(maximum_energy_intensive, before - after)
        witnesses.append(
            {
                "observation": observation,
                "before": before,
                "after": after,
                "before_rule": before_result["matched_rule"],
                "after_rule": after_result["matched_rule"],
            }
        )
    return {
        "changed": bool(witnesses),
        "max_extra_energy_actuation_c": max(0.0, maximum_energy_intensive),
        "directions": tuple(sorted(directions)),
        "witnesses": witnesses,
    }


def _references_parameter(value: Any, parameter: str) -> bool:
    if isinstance(value, Mapping):
        if value.get("param") == parameter or value.get("neg_param") == parameter:
            return True
        return any(_references_parameter(item, parameter) for item in value.values())
    if isinstance(value, list):
        return any(_references_parameter(item, parameter) for item in value)
    return False


def _rules_affected_by_patch(
    patch: Mapping[str, Any], program: Mapping[str, Any]
) -> Iterable[Mapping[str, Any]]:
    """Return only rules whose behaviour or priority can change under one patch."""
    rules = list(program["rules"])
    operation = patch.get("op")
    if operation == "add_rule":
        rule = patch.get("rule")
        return [rule] if isinstance(rule, Mapping) else []
    if operation == "replace_rule":
        rule = patch.get("rule")
        if not isinstance(rule, Mapping):
            return []
        original = next((item for item in rules if item.get("id") == rule.get("id")), None)
        return [item for item in (original, rule) if item is not None]
    if operation == "remove_rule":
        return [item for item in rules if item.get("id") == patch.get("id")]
    if operation == "move_rule":
        identifiers = [item.get("id") for item in rules]
        rule_id = patch.get("id")
        destination = patch.get("to_index")
        if (
            rule_id not in identifiers
            or not isinstance(destination, int)
            or isinstance(destination, bool)
        ):
            return []
        source = identifiers.index(rule_id)
        lower, upper = sorted((source, destination))
        return rules[lower : upper + 1]
    if operation == "set_param":
        parameter = patch.get("param")
        if not isinstance(parameter, str):
            return []
        return [rule for rule in rules if _references_parameter(rule, parameter)]
    return []


def cited_weather_drivers(patch: Mapping[str, Any], program: Mapping[str, Any]) -> set[str]:
    rules = _rules_affected_by_patch(patch, program)
    return {
        WEATHER_DRIVER_NODES[condition["field"]]
        for rule in rules
        for condition in rule.get("when", [])
        if condition.get("field") in WEATHER_DRIVER_NODES
    }
