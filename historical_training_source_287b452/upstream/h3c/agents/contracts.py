"""Single owner for Agent-facing JSON contracts."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Literal

PATCH_OPERATIONS = (
    "set_param",
    "add_rule",
    "replace_rule",
    "remove_rule",
    "move_rule",
    "no_change",
)

DEFAULT_PER_ZONE_RESERVED_CAP_C = 5.0

ALLOCATION_CONTRACT_SPEC: dict[str, Any] = {
    "fields": ("site_cap_c", "zone_budgets_c", "priority", "rationale_per_zone"),
    "causal_field": "causal_edge_ids",
    "per_zone_reserved_cap_field": "per_zone_reserved_cap_c",
    "site_cap_rule": "must_equal_supplied_site_cap",
    "zone_set_rule": "all_and_only_configured_zones",
    "budget_rule": "finite_nonnegative_each_and_sum_not_above_site_cap",
    "site_cap_need_not_be_fully_allocated": True,
    "priority_rule": "each_configured_zone_exactly_once_in_descending_priority",
    "rationale_rule": "all_and_only_configured_zones_with_nonempty_strings",
    "causal_rule": (
        "unique_nonempty_subset_of_visible_ids_including_an_edge_targeting_power_meters"
    ),
}

_PATCH_CONTRACT: dict[str, Any] = {
    "root": {"required": ["patch"], "patch": "list with exactly one operation"},
    "operations": {
        "no_change": {"required": ["op", "rationale"], "optional": []},
        "set_param": {
            "required": ["op", "param", "to", "causal_edge_ids", "rationale"],
            "optional": [],
        },
        "add_rule": {
            "required": ["op", "rule", "causal_edge_ids", "rationale"],
            "optional": ["index"],
        },
        "replace_rule": {
            "required": ["op", "rule", "causal_edge_ids", "rationale"],
            "optional": [],
        },
        "remove_rule": {
            "required": ["op", "id", "causal_edge_ids", "rationale"],
            "optional": [],
        },
        "move_rule": {
            "required": ["op", "id", "to_index", "causal_edge_ids", "rationale"],
            "optional": [],
        },
    },
    "rule": {
        "required": ["id", "when", "then"],
        "when_item": {"required": ["field", "op", "value"]},
        "then": {"required": ["op"], "value_for": ["set_residual", "step_setpoint"]},
    },
    "causal_edge_ids": "stable IDs from the supplied structured edge objects",
    "rationale": "nonempty string",
    "identifier_semantics": {
        "add_rule": "rule.id must be new",
        "replace_rule": "rule.id must already exist",
        "remove_rule": "id must already exist",
        "move_rule": "id must already exist",
    },
    "index_semantics": {
        "add_rule.index": "zero-based integer in [0, number of current rules]; omitted means append",
        "move_rule.to_index": "zero-based integer in [0, number of current rules - 1]",
    },
    "then_value_semantics": (
        'finite number, {"param":name}, or {"neg_param":name}; hold_setpoint has no value'
    ),
}


def rationale_length_telemetry(role: str, rationales: Mapping[str, str]) -> dict[str, Any]:
    """Report rationale character lengths without influencing any decision."""
    if role not in {"orchestrator", "executor"}:
        raise ValueError("rationale telemetry role is invalid")
    if (
        not isinstance(rationales, Mapping)
        or not rationales
        or any(
            not isinstance(scope, str)
            or not scope
            or not isinstance(value, str)
            or not value.strip()
            for scope, value in rationales.items()
        )
    ):
        raise ValueError("rationale telemetry requires nonempty string values")
    lengths = {scope: len(value) for scope, value in rationales.items()}
    return {
        "telemetry_schema": "h3c_rationale_length_telemetry",
        "schema_version": 1,
        "role": role,
        "character_lengths": lengths,
        "maximum_character_length": max(lengths.values()),
        "decision_use": "none",
    }


def patch_contract(*, causal_enabled: bool = True) -> dict[str, Any]:
    """Return the exact public patch contract for one causal mode."""
    view = copy.deepcopy(_PATCH_CONTRACT)
    if not causal_enabled:
        for operation in view["operations"].values():
            operation["required"] = [
                field for field in operation["required"] if field != "causal_edge_ids"
            ]
        view.pop("causal_edge_ids")
    return view


def compact_patch_contract(*, causal_enabled: bool = True, language: str = "en") -> str:
    """Render the unchanged patch wire contract without repeating common fields per operation."""
    causal_line = (
        "Edits also require causal_edge_ids: unique, nonempty visible IDs.\n"
        if causal_enabled and language == "en"
        else "修改类操作还需 causal_edge_ids：当前可见 ID 的无重复非空列表。\n"
        if causal_enabled
        else ""
    )
    intro = (
        'Root object exactly: {"patch":[{...}]}. patch is a list containing exactly one '
        "operation. Operation common fields: op, rationale.\n"
        if language == "en"
        else '根对象必须精确为 {"patch":[{...}]}；patch 是仅含一个 operation 的列表；'
        "operation 公共字段：op、rationale。\n"
    )
    operation_rows = []
    for operation in PATCH_OPERATIONS:
        contract = _PATCH_CONTRACT["operations"][operation]
        additional = [
            field
            for field in contract["required"]
            if field not in {"op", "rationale", "causal_edge_ids"}
        ]
        required_text = ",".join(additional) or "none"
        optional_text = ",".join(contract["optional"])
        suffix = f"; optional={optional_text}" if optional_text else ""
        operation_rows.append(f"{operation}: required={required_text}{suffix}")
    operation_table = "Operations:\n" + "\n".join(operation_rows) + "\n"
    rule = (
        "rule={id,when:[{field,op,value}],then:{op,value?}}."
        if language == "en"
        else "rule={id,when:[{field,op,value}],then:{op,value?}}。"
    )
    semantics = _PATCH_CONTRACT
    if language == "en":
        detail = (
            " IDs: add=new rule.id; replace=existing rule.id; remove/move=existing id. "
            "Zero-based indices: add index 0..N (default N); move to_index 0..N-1. "
            'then.value: finite number | {"param":name} | {"neg_param":name}; omit for '
            "hold_setpoint."
        )
    else:
        detail = (
            " ID：add 用新 rule.id；replace 用已有 rule.id；remove/move 用已有 id。"
            "索引从 0 开始：add index 为 0..N（默认 N）；move to_index 为 0..N-1。"
            'then.value：有限数值 | {"param":name} | {"neg_param":name}；hold_setpoint 省略 value。'
        )
    if not semantics["identifier_semantics"] or not semantics["index_semantics"]:
        raise ValueError("patch contract semantics are incomplete")
    return intro + causal_line + operation_table + rule + detail


def allocation_contract(*, causal_enabled: bool = True) -> tuple[str, ...]:
    fields = list(ALLOCATION_CONTRACT_SPEC["fields"])
    if causal_enabled:
        fields.append(str(ALLOCATION_CONTRACT_SPEC["causal_field"]))
    return tuple(fields)


def allocation_constraint_text(*, causal_enabled: bool, language: str) -> str:
    """Render the exact allocation rules consumed by the deterministic validator."""
    causal = (
        " causal_edge_ids: unique, nonempty, visible-ID subset including at least one visible "
        "edge with target=power_meters."
        if causal_enabled and language == "en"
        else " causal_edge_ids：可见 ID 的无重复非空子集，且至少包含一条 "
        "target=power_meters 的可见边。"
        if causal_enabled
        else ""
    )
    if language == "en":
        return (
            "site_cap_c=input. zone_budgets_c: keys=zones; each finite in "
            "[0,per_zone_reserved_cap_c]; sum≤cap; unused allowed. "
            "priority=permutation(zones), highest first. rationale_per_zone: keys=zones; values "
            "nonempty." + causal
        )
    return (
        "site_cap_c = 输入值。budgets/rationales 的区域键 = 配置区域。"
        "每区额度：有限且在 [0,per_zone_reserved_cap_c]；总和 ≤ cap，允许未用满。"
        "priority：区域全排列，优先级从高到低。理由不能为空。" + causal
    )


def _nonempty_string() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _causal_identifiers() -> dict[str, Any]:
    return {
        "type": "array",
        "items": _nonempty_string(),
        "minItems": 1,
        "uniqueItems": True,
    }


def _resolved_value_schema() -> dict[str, Any]:
    def reference(key: str) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {key: _nonempty_string()},
            "required": [key],
            "additionalProperties": False,
        }

    return {
        "oneOf": [
            {"type": "number"},
            reference("param"),
            reference("neg_param"),
        ]
    }


def _rule_schema() -> dict[str, Any]:
    condition = {
        "type": "object",
        "properties": {
            "field": _nonempty_string(),
            "op": _nonempty_string(),
            "value": _resolved_value_schema(),
        },
        "required": ["field", "op", "value"],
        "additionalProperties": False,
    }
    then = {
        "oneOf": [
            {
                "type": "object",
                "properties": {"op": {"const": "hold_setpoint"}},
                "required": ["op"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "op": {"enum": ["set_residual", "step_setpoint"]},
                    "value": _resolved_value_schema(),
                },
                "required": ["op", "value"],
                "additionalProperties": False,
            },
        ]
    }
    return {
        "type": "object",
        "properties": {
            "id": _nonempty_string(),
            "when": {"type": "array", "items": condition, "minItems": 1},
            "then": then,
        },
        "required": ["id", "when", "then"],
        "additionalProperties": False,
    }


def _operation_schema(operation: str, *, causal_enabled: bool) -> dict[str, Any]:
    contract = patch_contract(causal_enabled=causal_enabled)["operations"][operation]
    properties: dict[str, Any] = {
        "op": {"const": operation},
        "rationale": _nonempty_string(),
    }
    if operation == "set_param":
        properties.update({"param": _nonempty_string(), "to": {"type": "number"}})
    elif operation in {"add_rule", "replace_rule"}:
        properties["rule"] = _rule_schema()
        if operation == "add_rule":
            properties["index"] = {"type": "integer", "minimum": 0}
    elif operation == "remove_rule":
        properties["id"] = _nonempty_string()
    elif operation == "move_rule":
        properties.update(
            {
                "id": _nonempty_string(),
                "to_index": {"type": "integer", "minimum": 0},
            }
        )
    if causal_enabled and operation != "no_change":
        properties["causal_edge_ids"] = _causal_identifiers()
    return {
        "type": "object",
        "properties": properties,
        "required": list(contract["required"]),
        "additionalProperties": False,
    }


def executor_response_schema(*, causal_enabled: bool, long_term_memory: bool) -> dict[str, Any]:
    """Return the exact provider-native Executor envelope schema."""
    properties: dict[str, Any] = {
        "patch": {
            "type": "array",
            "items": {
                "oneOf": [
                    _operation_schema(operation, causal_enabled=causal_enabled)
                    for operation in PATCH_OPERATIONS
                ]
            },
            "minItems": 1,
            "maxItems": 1,
        }
    }
    required = ["patch"]
    if long_term_memory:
        properties["memory_refs"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "regime": {
                        "enum": [
                            "unoccupied",
                            "occupancy_transition",
                            "steady_state_occupancy",
                        ]
                    },
                    "revision": {"type": "integer", "minimum": 1},
                },
                "required": ["regime", "revision"],
                "additionalProperties": False,
            },
            "uniqueItems": True,
        }
        required.append("memory_refs")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def orchestrator_response_schema(*, zones: tuple[str, ...], causal_enabled: bool) -> dict[str, Any]:
    """Return the exact provider-native Orchestrator envelope schema."""
    if not zones or len(zones) != len(set(zones)):
        raise ValueError("orchestrator response schema requires unique configured zones")
    zone_values = {zone: {"type": "number"} for zone in zones}
    zone_rationales = {zone: _nonempty_string() for zone in zones}
    properties: dict[str, Any] = {
        "site_cap_c": {"type": "number"},
        "zone_budgets_c": {
            "type": "object",
            "properties": zone_values,
            "required": list(zones),
            "additionalProperties": False,
        },
        "priority": {
            "type": "array",
            "items": {"enum": list(zones)},
            "minItems": len(zones),
            "maxItems": len(zones),
            "uniqueItems": True,
        },
        "rationale_per_zone": {
            "type": "object",
            "properties": zone_rationales,
            "required": list(zones),
            "additionalProperties": False,
        },
    }
    required = list(ALLOCATION_CONTRACT_SPEC["fields"])
    if causal_enabled:
        properties["causal_edge_ids"] = _causal_identifiers()
        required.append(str(ALLOCATION_CONTRACT_SPEC["causal_field"]))
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def reflector_response_schema(*, zones: tuple[str, ...], long_term_memory: bool) -> dict[str, Any]:
    """Return the exact provider-native Reflector envelope schema."""
    if not zones or len(zones) != len(set(zones)):
        raise ValueError("reflector response schema requires unique configured zones")
    lesson = {
        "type": "object",
        "properties": {
            "zone": {"enum": list(zones)},
            "lesson": _nonempty_string(),
        },
        "required": ["zone", "lesson"],
        "additionalProperties": False,
    }
    properties: dict[str, Any] = {
        "hourly_lessons": {
            "type": "array",
            "items": lesson,
            "minItems": len(zones),
            "maxItems": len(zones),
        }
    }
    required = ["hourly_lessons"]
    if long_term_memory:
        regimes = ["unoccupied", "occupancy_transition", "steady_state_occupancy"]
        operation_common = {
            "zone": {"enum": list(zones)},
            "regime": {"enum": regimes},
            "expected_revision": {"type": "integer", "minimum": 1},
            "experience": _nonempty_string(),
        }

        def operation(
            op: Literal["no_change", "add", "replace", "delete"],
        ) -> dict[str, Any]:
            fields = {"zone": operation_common["zone"], "op": {"const": op}}
            required_fields = ["zone", "op"]
            if op != "no_change":
                fields["regime"] = operation_common["regime"]
                required_fields.append("regime")
            if op in {"replace", "delete"}:
                fields["expected_revision"] = operation_common["expected_revision"]
                required_fields.append("expected_revision")
            if op in {"add", "replace"}:
                fields["experience"] = operation_common["experience"]
                required_fields.append("experience")
            return {
                "type": "object",
                "properties": fields,
                "required": required_fields,
                "additionalProperties": False,
            }

        properties["memory_operations"] = {
            "type": "array",
            "items": {"oneOf": [operation(op) for op in ("no_change", "add", "replace", "delete")]},
            "minItems": len(zones),
            "maxItems": len(zones),
        }
        required.append("memory_operations")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
