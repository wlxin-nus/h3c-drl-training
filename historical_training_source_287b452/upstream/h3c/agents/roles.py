"""Three bounded Agent roles with deterministic prompt and output contracts."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from h3c.agents.context_compiler import CompiledContext, ContextBuilder
from h3c.agents.contracts import (
    DEFAULT_PER_ZONE_RESERVED_CAP_C,
    allocation_contract,
    executor_response_schema,
    orchestrator_response_schema,
    rationale_length_telemetry,
    reflector_response_schema,
)
from h3c.agents.dynamic_prompt import (
    COOLING_CONTROL_DOMAIN,
    display,
    executor_observation_view,
    parameter_rule_limits,
    strip_audit_fields,
)
from h3c.agents.prompts import Role, system_prompt
from h3c.agents.time_context import decision_window
from h3c.control.budget import validate_allocation
from h3c.control.program import (
    PARAMETER_BOUNDS,
    RULE_ACTIONS,
    current_interpreter_derivation,
    interpreter_semantics,
    rule_effects_if_matched,
    setpoint_effect_facts,
    validate_patch_shape,
)
from h3c.memory.caol import (
    ReflectorResolution,
    agent_visible_number,
    resolve_reflector_payload,
)


@dataclass(frozen=True)
class ModelCallContext:
    hour: int
    step: int
    call_ordinal: int
    zone: str | None = None

    def __post_init__(self) -> None:
        if min(self.hour, self.step, self.call_ordinal) < 0:
            raise ValueError("model call context indices must be nonnegative")
        if self.zone is not None and not self.zone:
            raise ValueError("model call context zone must be nonempty when present")

    def as_mapping(self) -> dict[str, Any]:
        context: dict[str, Any] = {
            "hour": self.hour,
            "step": self.step,
            "call_ordinal": self.call_ordinal,
        }
        if self.zone is not None:
            context["zone"] = self.zone
        return context


class ModelClient(Protocol):
    async def complete(
        self,
        *,
        context: ModelCallContext,
        role: Role,
        system: str,
        user: str,
        thinking_mode: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> str: ...


def parse_bare_json(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or raw.lstrip().startswith("```"):
        raise ValueError("model output must be one bare JSON object")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("model output is not valid bare JSON") from error
    if not isinstance(value, dict):
        raise ValueError("model output root must be an object")
    return value


class ModelContractError(ValueError):
    """A model response failed the exact role contract without a transport failure."""

    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def _mapping_rows(value: Mapping[str, Any], *, identity: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, raw_row in value.items():
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{identity} mapping values must be objects")
        if identity in raw_row:
            raise ValueError(f"{identity} has more than one field owner")
        rows.append({identity: str(name), **copy.deepcopy(dict(raw_row))})
    return rows


def _site_state_views(
    site_state: Mapping[str, Any], *, outcome_times: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    current = copy.deepcopy(dict(site_state))
    raw_forecast = current.pop("weather_next_steps", [])
    if not isinstance(raw_forecast, Sequence) or isinstance(raw_forecast, (str, bytes)):
        raise ValueError("site weather forecast must be a sequence")
    forecast: list[dict[str, Any]] = []
    if raw_forecast and len(raw_forecast) != len(outcome_times):
        raise ValueError("site weather forecast and decision window disagree")
    for index, row in enumerate(raw_forecast):
        if not isinstance(row, Mapping) or "step_ahead" not in row:
            raise ValueError("site weather forecast rows require step_ahead")
        item = copy.deepcopy(dict(row))
        if int(item.pop("step_ahead")) != index + 1:
            raise ValueError("site weather forecast order is invalid")
        forecast.append({"forecast_outcome_time": outcome_times[index], **item})
    current_state: dict[str, Any] = {
        name: current.pop(name)
        for name in ("outdoor_temp_c", "solar_irr", "price_now")
        if name in current
    }
    if "price_next_hour" in current:
        current["price_at_interval_end"] = current.pop("price_next_hour")
    return current_state, current, forecast


def _executor_observation_views(
    observation: Mapping[str, Any],
    *,
    outcome_times: Sequence[str],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    current = copy.deepcopy(dict(observation))
    occupancy = current.pop("occupancy_next_steps", [])
    weather = current.pop("weather_next_steps", [])
    if not isinstance(occupancy, Sequence) or isinstance(occupancy, (str, bytes)):
        raise ValueError("occupancy forecast must be a sequence")
    if not isinstance(weather, Sequence) or isinstance(weather, (str, bytes)):
        raise ValueError("weather forecast must be a sequence")
    if occupancy and len(occupancy) != 4:
        raise ValueError("occupancy forecast must contain four steps")
    if weather and len(weather) != 4:
        raise ValueError("weather forecast must contain four steps")
    forecast_length = max(len(occupancy), len(weather))
    if forecast_length and forecast_length != len(outcome_times):
        raise ValueError("zone forecast and decision window disagree")
    rows: list[dict[str, Any]] = []
    for index in range(forecast_length):
        row: dict[str, Any] = {"forecast_outcome_time": outcome_times[index]}
        if occupancy:
            row["occupancy"] = occupancy[index]
        if weather:
            raw_weather = weather[index]
            if not isinstance(raw_weather, Mapping):
                raise ValueError("weather forecast row must be an object")
            weather_row = dict(raw_weather)
            weather_step = weather_row.pop("step_ahead", index + 1)
            if int(weather_step) != index + 1:
                raise ValueError("weather and occupancy forecast steps disagree")
            row.update(copy.deepcopy(weather_row))
        rows.append(row)
    headroom = current.pop("comfort_headroom_c", None)
    if headroom is not None and not isinstance(headroom, Mapping):
        raise ValueError("comfort headroom must be an object when available")
    zone_state = {}
    for source, target in (
        ("zone_temp_c", "zone_temperature_c"),
        ("last_pmv", "pmv"),
        ("current_occupancy", "occupancy"),
        ("last_setpoint_c", "setpoint_c"),
    ):
        if source in current:
            zone_state[target] = current.pop(source)
    if isinstance(headroom, Mapping):
        zone_state["temp_rise_to_warm_pmv_edge_c"] = headroom.get("warmer_c")
        zone_state["temp_drop_to_cool_pmv_edge_c"] = headroom.get("cooler_c")
    if "occupancy" in zone_state and "setpoint_c" in zone_state:
        zone_state.update(
            setpoint_effect_facts(
                current_occupancy=zone_state["occupancy"],
                applied_setpoint_c=zone_state["setpoint_c"],
            )
        )
    transition_state = (
        {"previous_occupancy": current.pop("last_occupancy")} if "last_occupancy" in current else {}
    )
    external_state: dict[str, Any] = {
        name: current.pop(name)
        for name in ("outdoor_temp_c", "solar_irr", "price_now")
        if name in current
    }
    return zone_state, transition_state, external_state, current, rows


def _orchestrator_zone_views(
    zone_coupling: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    current_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    current_fields = (
        "zone_temperature_c",
        "pmv",
        "occupancy",
        "setpoint_c",
        "temp_rise_to_warm_pmv_edge_c",
        "temp_drop_to_cool_pmv_edge_c",
    )
    control_fields = (
        "occupancy_at_interval_end",
        "precool_offset_from_unoccupied_base_c",
        "resulting_precool_setpoint_c",
    )
    for zone, raw in zone_coupling.items():
        if not isinstance(raw, Mapping):
            raise ValueError("zone coupling values must be objects")
        unknown = (
            set(raw)
            - set(current_fields)
            - set(control_fields)
            - {
                "occupied_share_with_no_room_to_ask_for_more",
                "occupied_share_with_no_room_to_ask_for_less",
            }
        )
        if unknown:
            raise ValueError(
                f"zone coupling contains fields without a semantic owner: {sorted(unknown)}"
            )
        current_rows.append(
            {"zone": zone, **{field: raw[field] for field in current_fields if field in raw}}
        )
        control_rows.append(
            {"zone": zone, **{field: raw[field] for field in control_fields if field in raw}}
        )
        signals = {
            field: raw[field]
            for field in (
                "occupied_share_with_no_room_to_ask_for_more",
                "occupied_share_with_no_room_to_ask_for_less",
            )
            if field in raw
        }
        if signals:
            signal_rows.append({"zone": zone, **signals})
    return current_rows, control_rows, signal_rows


def _working_memory_terminal_by_zone(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    if not records:
        return {}
    latest = max(int(record["hour"]) for record in records)
    result: dict[str, dict[str, float]] = {}
    for record in records:
        if int(record["hour"]) != latest:
            continue
        zone = str(record["zone"])
        outcome = record["outcome"]
        action = record["action"]
        if not isinstance(outcome, Mapping) or not isinstance(action, Mapping):
            raise ValueError("working-memory terminal record is malformed")
        result[zone] = {
            "zone_temperature_c": float(outcome["zone_temperatures_c"][-1]),
            "pmv": float(outcome["pmv"][-1]),
            "setpoint_c": float(action["actual_setpoints_c"][-1]),
        }
    return result


def _assert_current_state_matches_memory(
    current_rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> None:
    terminal = _working_memory_terminal_by_zone(records)
    for row in current_rows:
        zone = str(row["zone"])
        if zone not in terminal:
            continue
        for state_field in ("zone_temperature_c", "pmv", "setpoint_c"):
            if state_field not in row:
                raise ValueError(f"current decision state is missing {zone}.{state_field}")
            if agent_visible_number(float(row[state_field])) != terminal[zone][state_field]:
                raise ValueError(
                    "current decision state disagrees with working-memory endpoint for "
                    f"{zone}.{state_field}"
                )


def _previous_budget_view(
    previous_allocation: Mapping[str, Any] | None,
    previous_utilisation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if previous_allocation is None and previous_utilisation is None:
        return None
    if previous_allocation is None or previous_utilisation is None:
        raise ValueError("previous allocation and utilisation must be supplied together")
    reserved = previous_utilisation.get("reserved_allowance_by_zone_c")
    consumed = previous_utilisation.get("reserved_consumption_by_zone_c")
    shared_used = previous_utilisation.get("residual_used_by")
    if (
        not isinstance(reserved, Mapping)
        or not isinstance(consumed, Mapping)
        or not isinstance(shared_used, Mapping)
    ):
        raise ValueError("previous Budget view requires production ledger zone owners")
    if dict(reserved) != dict(previous_allocation["zone_budgets_c"]):
        raise ValueError("previous allocation and Budget ledger reserved allowances disagree")
    return {
        "site_cap_c": previous_allocation["site_cap_c"],
        "reserved_allowance_by_zone_c": copy.deepcopy(dict(reserved)),
        "reserved_consumption_by_zone_c": copy.deepcopy(dict(consumed)),
        "total_reserved_allowance_c": previous_utilisation["granted_c"],
        "total_reserved_consumption_c": previous_utilisation["used_c"],
        "reserved_utilisation": previous_utilisation["utilisation"],
        "initial_shared_unreserved_pool_c": previous_utilisation["residual_initial_c"],
        "shared_pool_consumption_by_zone_c": copy.deepcopy(dict(shared_used)),
        "remaining_shared_unreserved_pool_c": previous_utilisation["residual_left_c"],
        "previous_priority": copy.deepcopy(previous_allocation["priority"]),
        **(
            {"previous_causal_edge_ids": copy.deepcopy(previous_allocation["causal_edge_ids"])}
            if "causal_edge_ids" in previous_allocation
            else {}
        ),
    }


def _control_specification(
    program: Mapping[str, Any],
    limits: Mapping[str, Any],
    observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    specification = copy.deepcopy(dict(program))
    params = specification.pop("params", None)
    if not isinstance(params, Mapping):
        raise ValueError("current executable program lacks parameters")
    remaining_limits = copy.deepcopy(dict(limits))
    parameter_rows: list[dict[str, Any]] = []
    for name, current in params.items():
        raw_limit = remaining_limits.pop(str(name), None)
        if not isinstance(raw_limit, Mapping) or set(raw_limit) != {
            "bounds",
            "bounds_come_from",
        }:
            raise ValueError(f"parameter {name} lacks one exact limit owner")
        bounds = raw_limit["bounds"]
        if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 2:
            raise ValueError(f"parameter {name} limit is malformed")
        parameter_rows.append(
            {
                "param": str(name),
                "current": copy.deepcopy(current),
                "min": copy.deepcopy(bounds[0]),
                "max": copy.deepcopy(bounds[1]),
                "bounds_source": copy.deepcopy(raw_limit["bounds_come_from"]),
            }
        )
    rules = specification.pop("rules")
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise ValueError("current executable program lacks an ordered rule sequence")
    new_rule_limits = remaining_limits.get("a_rule_you_add")
    if not isinstance(new_rule_limits, dict):
        raise ValueError("current executable program lacks one rule-limit owner")
    maximum_rules = new_rule_limits.pop("most_rules_at_once", None)
    if not isinstance(maximum_rules, int) or isinstance(maximum_rules, bool):
        raise ValueError("current executable program rule maximum is malformed")
    if len(rules) > maximum_rules:
        raise ValueError("current executable program exceeds its rule maximum")
    required_derivation_fields = {
        "current_occupancy",
        "last_occupancy",
        "last_pmv",
        "last_setpoint",
        "occ_ahead",
    }
    derivation = (
        current_interpreter_derivation(program, observation)
        if observation is not None
        and required_derivation_fields <= set(observation)
        and set(PARAMETER_BOUNDS) <= set(params)
        else None
    )
    return {
        "program_version": specification.pop("program_version"),
        "parameters": parameter_rows,
        "rules": rules,
        "rule_capacity": {
            "used": len(rules),
            "maximum": maximum_rules,
            "remaining_add_slots": maximum_rules - len(rules),
        },
        "interpreter_semantics": interpreter_semantics(),
        **({"current_interpreter_derivation": derivation} if derivation is not None else {}),
        **(
            {
                "rule_effects_if_matched": rule_effects_if_matched(
                    program, cast(Mapping[str, Any], observation)
                )
            }
            if derivation is not None
            else {}
        ),
        "rule_and_weather_limits": remaining_limits,
        "control_domain": copy.deepcopy(COOLING_CONTROL_DOMAIN),
        **specification,
    }


def resolve_orchestrator_model_output(
    raw: str,
    zones: Sequence[str],
    *,
    causal_enabled: bool,
    allowed_causal_edge_ids: set[str] | None = None,
    site_causal_edge_ids: set[str] | None = None,
    expected_site_cap_c: float | None = None,
    expected_per_zone_reserved_cap_c: float = DEFAULT_PER_ZONE_RESERVED_CAP_C,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one raw Orchestrator response through the production contract."""
    try:
        allocation = parse_bare_json(raw)
        if set(allocation) == {"allocation_contract"}:
            wrapped = allocation["allocation_contract"]
            if not isinstance(wrapped, dict):
                raise ValueError("Orchestrator allocation_contract wrapper must contain an object")
            allocation = dict(wrapped)
        if set(allocation) != set(allocation_contract(causal_enabled=causal_enabled)):
            raise ValueError("Orchestrator output does not match the exact allocation contract")
        validate_allocation(
            allocation,
            zones,
            causal_enabled=causal_enabled,
            allowed_causal_edge_ids=allowed_causal_edge_ids,
            site_causal_edge_ids=site_causal_edge_ids,
            expected_site_cap_c=expected_site_cap_c,
            expected_per_zone_reserved_cap_c=expected_per_zone_reserved_cap_c,
        )
        telemetry = rationale_length_telemetry("orchestrator", allocation["rationale_per_zone"])
    except ValueError as error:
        raise ModelContractError(str(error), raw) from error
    return allocation, telemetry


def _resolve_executor_model_output(
    raw: str, *, causal_enabled: bool, long_term_memory: bool = False
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Resolve one raw Executor response without applying a control decision."""
    try:
        root = parse_bare_json(raw)
        root_fields = set(root)
        memory_refs: list[dict[str, Any]] = []
        if long_term_memory:
            if root_fields != {"patch", "memory_refs"} or not isinstance(root["memory_refs"], list):
                raise ValueError("Executor output does not match the memory-enabled root contract")
            for reference in root["memory_refs"]:
                if (
                    not isinstance(reference, Mapping)
                    or set(reference) != {"regime", "revision"}
                    or reference["regime"]
                    not in {
                        "unoccupied",
                        "occupancy_transition",
                        "steady_state_occupancy",
                    }
                    or not isinstance(reference["revision"], int)
                    or isinstance(reference["revision"], bool)
                    or int(reference["revision"]) < 1
                ):
                    raise ValueError("Executor memory_refs entry is structurally invalid")
                memory_refs.append(copy.deepcopy(dict(reference)))
        elif root_fields == {"patch", "rationale"}:
            audit_rationale = root["rationale"]
            if not isinstance(audit_rationale, str) or not audit_rationale.strip():
                raise ValueError("Executor root rationale must be a nonempty audit string")
        elif root_fields == {"patch", "root"}:
            duplicate = root["root"]
            if not isinstance(duplicate, dict) or set(duplicate) != {"patch"}:
                raise ValueError("Executor root wrapper must contain only the duplicate patch")
            if duplicate["patch"] != root["patch"]:
                raise ValueError("Executor root wrapper conflicts with the canonical patch")
        elif root_fields != {"patch"}:
            raise ValueError("Executor output contains unsupported root fields")
        if not isinstance(root["patch"], list) or len(root["patch"]) != 1:
            raise ValueError("Executor output must contain exactly one patch operation")
        raw_patch = root["patch"][0]
        if not isinstance(raw_patch, dict):
            raise ValueError("Executor patch operation must be an object")
        patch = dict(raw_patch)
        if patch.get("op") == "replace_rule" and "id" in patch:
            rule = patch.get("rule")
            if not isinstance(rule, dict) or patch["id"] != rule.get("id"):
                raise ValueError("Executor replace_rule id conflicts with rule.id")
            patch.pop("id")
        validate_patch_shape(patch, causal_enabled=causal_enabled)
        telemetry = rationale_length_telemetry("executor", {"operation": patch["rationale"]})
    except (KeyError, TypeError, ValueError) as error:
        raise ModelContractError(str(error), raw) from error
    return dict(patch), telemetry, memory_refs


def resolve_executor_model_output(
    raw: str, *, causal_enabled: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the memory-off Executor contract used by the default architecture."""
    patch, telemetry, _ = _resolve_executor_model_output(
        raw,
        causal_enabled=causal_enabled,
        long_term_memory=False,
    )
    return patch, telemetry


def resolve_executor_memory_model_output(
    raw: str, *, causal_enabled: bool
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Resolve the optional memory-enabled root without validating reference existence."""
    return _resolve_executor_model_output(
        raw,
        causal_enabled=causal_enabled,
        long_term_memory=True,
    )


@dataclass
class Orchestrator:
    client: ModelClient
    last_rationale_telemetry: dict[str, Any] | None = field(init=False, default=None)

    @staticmethod
    def build_context(
        *,
        hour: int,
        decision_time_seconds: int | None = None,
        zones: Sequence[str],
        site_state: Mapping[str, Any],
        zone_coupling: Mapping[str, Any],
        causal_edges: Sequence[Mapping[str, Any]] | None,
        previous_allocation: Mapping[str, Any] | None,
        previous_utilisation: Mapping[str, Any] | None,
        working_memory: Sequence[Mapping[str, Any]] | None,
        allocation_limits: Mapping[str, Any],
    ) -> CompiledContext:
        expected_limits = {"zones", "site_cap_c", "per_zone_reserved_cap_c"}
        if set(allocation_limits) != expected_limits:
            raise ValueError("allocation limits must use the exact resolved cooling fields")
        if list(allocation_limits["zones"]) != list(zones):
            raise ValueError("allocation-limit zones must preserve configured zone order")
        site = strip_audit_fields(site_state)
        coupling = strip_audit_fields(zone_coupling)
        edges = strip_audit_fields(causal_edges)
        previous = strip_audit_fields(previous_allocation)
        utilisation = strip_audit_fields(previous_utilisation)
        memory = strip_audit_fields(working_memory)
        builder = ContextBuilder()
        resolved_time = hour * 3600 if decision_time_seconds is None else decision_time_seconds
        builder.add_audit_metadata(
            decision_hour=hour,
            decision_time_seconds=resolved_time,
        )
        window = decision_window(resolved_time)
        outcome_times = cast(Sequence[str], window["forecast_outcome_times"])
        builder.add_json("DECISION WINDOW", window)
        current_site, forecast_summary, forecast_rows = _site_state_views(
            cast(Mapping[str, Any], site), outcome_times=outcome_times
        )
        builder.add_json("CURRENT SITE STATE", display(current_site))
        builder.add_json("FORECAST SUMMARY", display(forecast_summary))
        if forecast_rows:
            builder.add_common_rows(
                "SITE FORECAST",
                cast(Sequence[Mapping[str, Any]], display(forecast_rows)),
                identity_fields=("forecast_outcome_time",),
            )
        coupling_rows, control_rows, signal_rows = _orchestrator_zone_views(
            cast(Mapping[str, Any], coupling)
        )
        if memory:
            _assert_current_state_matches_memory(
                coupling_rows, cast(Sequence[Mapping[str, Any]], memory)
            )
        builder.add_common_rows(
            "CURRENT ZONE STATE",
            cast(Sequence[Mapping[str, Any]], display(coupling_rows)),
            identity_fields=("zone",),
        )
        builder.add_common_rows(
            "ZONE CONTROL CONTEXT",
            cast(Sequence[Mapping[str, Any]], display(control_rows)),
            identity_fields=("zone",),
        )
        if signal_rows:
            builder.add_common_rows(
                "RECENT ACTUATOR LIMIT SIGNALS",
                cast(Sequence[Mapping[str, Any]], display(signal_rows)),
                identity_fields=("zone",),
            )
        if edges is not None:
            builder.add_common_rows(
                "CAUSAL EVIDENCE",
                cast(Sequence[Mapping[str, Any]], display(edges)),
                identity_fields=("id",),
            )
        builder.add_json("ALLOCATION LIMITS", display(allocation_limits))
        builder.add_json("CONTROL DOMAIN", display(COOLING_CONTROL_DOMAIN))
        if memory:
            builder.add_working_memory(
                "WORKING MEMORY",
                cast(Sequence[Mapping[str, Any]], memory),
                reference_hour=hour,
                reference_time_seconds=resolved_time,
                decision_rationale_visible=False,
            )
        builder.add_json(
            "PREVIOUS BUDGET USE",
            display(
                _previous_budget_view(
                    cast(Mapping[str, Any] | None, previous),
                    cast(Mapping[str, Any] | None, utilisation),
                )
            ),
        )
        return builder.build()

    @staticmethod
    def build_user(
        *,
        hour: int,
        decision_time_seconds: int | None = None,
        zones: Sequence[str],
        site_state: Mapping[str, Any],
        zone_coupling: Mapping[str, Any],
        causal_edges: Sequence[Mapping[str, Any]] | None,
        previous_allocation: Mapping[str, Any] | None,
        previous_utilisation: Mapping[str, Any] | None,
        working_memory: Sequence[Mapping[str, Any]] | None,
        allocation_limits: Mapping[str, Any],
    ) -> str:
        return Orchestrator.build_context(
            hour=hour,
            decision_time_seconds=decision_time_seconds,
            zones=zones,
            site_state=site_state,
            zone_coupling=zone_coupling,
            causal_edges=causal_edges,
            previous_allocation=previous_allocation,
            previous_utilisation=previous_utilisation,
            working_memory=working_memory,
            allocation_limits=allocation_limits,
        ).agent_view

    async def allocate(
        self,
        *,
        context: ModelCallContext,
        zones: Sequence[str],
        user: str,
        causal_enabled: bool,
        thinking_mode: str,
        allowed_causal_edge_ids: set[str] | None = None,
        site_causal_edge_ids: set[str] | None = None,
        expected_site_cap_c: float | None = None,
        expected_per_zone_reserved_cap_c: float = DEFAULT_PER_ZONE_RESERVED_CAP_C,
    ) -> dict[str, Any]:
        self.last_rationale_telemetry = None
        raw = await self.client.complete(
            context=context,
            role="orchestrator",
            system=system_prompt("orchestrator", causal_enabled=causal_enabled),
            user=user,
            thinking_mode=thinking_mode,
            response_schema=orchestrator_response_schema(
                zones=tuple(zones), causal_enabled=causal_enabled
            ),
        )
        allocation, self.last_rationale_telemetry = resolve_orchestrator_model_output(
            raw,
            zones,
            causal_enabled=causal_enabled,
            allowed_causal_edge_ids=allowed_causal_edge_ids,
            site_causal_edge_ids=site_causal_edge_ids,
            expected_site_cap_c=expected_site_cap_c,
            expected_per_zone_reserved_cap_c=expected_per_zone_reserved_cap_c,
        )
        return allocation


@dataclass
class Executor:
    client: ModelClient
    last_rationale_telemetry: dict[str, Any] | None = field(init=False, default=None)
    last_memory_refs: list[dict[str, Any]] = field(init=False, default_factory=list)

    @staticmethod
    def build_context(
        *,
        hour: int,
        decision_time_seconds: int | None = None,
        zone: str,
        observation: Mapping[str, Any],
        current_executable_program: Mapping[str, Any],
        causal_edges: Sequence[Mapping[str, Any]] | None,
        allowance: Mapping[str, Any] | None,
        working_memory: Sequence[Mapping[str, Any]] | None,
        long_term_experiences: Sequence[Mapping[str, Any]] | None = None,
        rejection_feedback: Mapping[str, Any] | None = None,
    ) -> CompiledContext:
        observation_view = strip_audit_fields(executor_observation_view(observation))
        program_view = strip_audit_fields(current_executable_program)
        edges = strip_audit_fields(causal_edges)
        budget = strip_audit_fields(allowance)
        memory = strip_audit_fields(working_memory)
        experiences = strip_audit_fields(long_term_experiences)
        rejection = strip_audit_fields(rejection_feedback)
        limits = parameter_rule_limits()
        builder = ContextBuilder()
        resolved_time = hour * 3600 if decision_time_seconds is None else decision_time_seconds
        builder.add_audit_metadata(
            decision_hour=hour,
            decision_step=hour * 4,
            decision_time_seconds=resolved_time,
            zone=zone,
        )
        window = decision_window(resolved_time)
        outcome_times = cast(Sequence[str], window["forecast_outcome_times"])
        builder.add_json("DECISION WINDOW", window)
        (
            current_zone_state,
            transition_state,
            current_external_state,
            forecast_summary,
            forecast_rows,
        ) = _executor_observation_views(
            cast(Mapping[str, Any], observation_view), outcome_times=outcome_times
        )
        current_zone_row = {"zone": zone, **current_zone_state}
        if memory:
            _assert_current_state_matches_memory(
                [current_zone_row], cast(Sequence[Mapping[str, Any]], memory)
            )
        builder.add_json("CURRENT DECISION STATE", display(current_zone_row))
        builder.add_json("OCCUPANCY TRANSITION CONTEXT", display(transition_state))
        builder.add_json("CURRENT EXTERNAL STATE", display(current_external_state))
        builder.add_json("FORECAST SUMMARY", display(forecast_summary))
        if forecast_rows:
            builder.add_common_rows(
                "ZONE FORECAST",
                cast(Sequence[Mapping[str, Any]], display(forecast_rows)),
                identity_fields=("forecast_outcome_time",),
            )
        control_specification = _control_specification(
            cast(Mapping[str, Any], program_view),
            limits,
            observation=observation,
        )
        effect_rows = control_specification.pop("rule_effects_if_matched", None)
        builder.add_control_specification("CONTROL SPECIFICATION", display(control_specification))
        if isinstance(effect_rows, Mapping):
            for action in RULE_ACTIONS:
                rows = effect_rows.get(action)
                if isinstance(rows, list) and rows:
                    builder.add_common_rows(
                        f"RULE EFFECTS IF MATCHED — {action}",
                        cast(Sequence[Mapping[str, Any]], display(rows)),
                        identity_fields=(
                            "rule_id",
                            "matching_current_occupancy",
                            "matching_previous_occupancy",
                        ),
                    )
        if edges is not None:
            builder.add_common_rows(
                "CAUSAL EVIDENCE",
                cast(Sequence[Mapping[str, Any]], display(edges)),
                identity_fields=("id",),
            )
        builder.add_json(
            "ALLOCATION", display({"allocation": budget}) if budget is not None else None
        )
        if memory:
            builder.add_working_memory(
                "WORKING MEMORY",
                cast(Sequence[Mapping[str, Any]], memory),
                reference_hour=hour,
                reference_time_seconds=resolved_time,
            )
        if experiences:
            builder.add_common_rows(
                "ACTIVE LONG-TERM EXPERIENCE SLOTS",
                cast(Sequence[Mapping[str, Any]], display(experiences)),
                identity_fields=("regime",),
            )
        builder.add_json("LAST REJECTION", display(rejection))
        return builder.build()

    @staticmethod
    def build_user(
        *,
        hour: int,
        decision_time_seconds: int | None = None,
        zone: str,
        observation: Mapping[str, Any],
        current_executable_program: Mapping[str, Any],
        causal_edges: Sequence[Mapping[str, Any]] | None,
        allowance: Mapping[str, Any] | None,
        working_memory: Sequence[Mapping[str, Any]] | None,
        long_term_experiences: Sequence[Mapping[str, Any]] | None = None,
        rejection_feedback: Mapping[str, Any] | None = None,
    ) -> str:
        return Executor.build_context(
            hour=hour,
            decision_time_seconds=decision_time_seconds,
            zone=zone,
            observation=observation,
            current_executable_program=current_executable_program,
            causal_edges=causal_edges,
            allowance=allowance,
            working_memory=working_memory,
            long_term_experiences=long_term_experiences,
            rejection_feedback=rejection_feedback,
        ).agent_view

    async def propose(
        self,
        *,
        context: ModelCallContext,
        user: str,
        causal_enabled: bool,
        coordination_enabled: bool,
        thinking_mode: str,
        long_term_memory: bool = False,
    ) -> dict[str, Any]:
        raw = await self.client.complete(
            context=context,
            role="executor",
            system=system_prompt(
                "executor",
                causal_enabled=causal_enabled,
                coordination_enabled=coordination_enabled,
                long_term_memory=long_term_memory,
            ),
            user=user,
            thinking_mode=thinking_mode,
            response_schema=executor_response_schema(
                causal_enabled=causal_enabled,
                long_term_memory=long_term_memory,
            ),
        )
        self.last_rationale_telemetry = None
        self.last_memory_refs = []
        if long_term_memory:
            patch, self.last_rationale_telemetry, self.last_memory_refs = (
                resolve_executor_memory_model_output(raw, causal_enabled=causal_enabled)
            )
        else:
            patch, self.last_rationale_telemetry = resolve_executor_model_output(
                raw,
                causal_enabled=causal_enabled,
            )
        return patch


def clean_insight(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or len(text) > 480:
        return None
    return text


@dataclass
class Reflector:
    client: ModelClient

    @staticmethod
    def build_context(
        *,
        current_hour_cao: Sequence[Mapping[str, Any]],
        interval_start_time_seconds: int | None = None,
        long_term_slots: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> CompiledContext:
        cao = strip_audit_fields(current_hour_cao)
        slots = strip_audit_fields(long_term_slots)
        builder = ContextBuilder()
        completed_hours = {int(record["hour"]) for record in cast(Sequence[Mapping[str, Any]], cao)}
        if len(completed_hours) != 1:
            raise ValueError("Reflector evidence must belong to one completed interval")
        completed_hour = next(iter(completed_hours))
        resolved_start = (
            completed_hour * 3600
            if interval_start_time_seconds is None
            else interval_start_time_seconds
        )
        builder.add_audit_metadata(
            completed_hour=completed_hour,
            interval_start_time_seconds=resolved_start,
        )
        builder.add_working_memory(
            "COMPLETED CONTROL INTERVAL",
            cast(Sequence[Mapping[str, Any]], cao),
            reference_hour=completed_hour,
            reference_time_seconds=resolved_start,
            presentation="completed_interval",
        )
        if slots is not None:
            active_slot_rows: list[dict[str, Any]] = []
            empty_slot_rows: list[dict[str, Any]] = []
            for zone, zone_slots in cast(Mapping[str, Any], slots).items():
                if not isinstance(zone_slots, Sequence) or isinstance(zone_slots, (str, bytes)):
                    raise ValueError("eligible experience slots must be a sequence")
                for raw_slot in zone_slots:
                    if not isinstance(raw_slot, Mapping):
                        raise ValueError("eligible experience slot must be an object")
                    if "zone" in raw_slot:
                        raise ValueError("eligible experience slot has duplicate zone owner")
                    slot = {"zone": zone, **copy.deepcopy(dict(raw_slot))}
                    if slot.get("state") == "empty":
                        slot.pop("state")
                        empty_slot_rows.append(slot)
                    else:
                        active_slot_rows.append(slot)
            if active_slot_rows:
                builder.add_common_rows(
                    "ACTIVE LONG-TERM EXPERIENCE SLOTS",
                    cast(Sequence[Mapping[str, Any]], display(active_slot_rows)),
                    identity_fields=("zone", "regime"),
                )
            if empty_slot_rows:
                builder.add_common_rows(
                    "EMPTY LONG-TERM EXPERIENCE SLOTS",
                    cast(Sequence[Mapping[str, Any]], display(empty_slot_rows)),
                    identity_fields=("zone", "regime"),
                )
        return builder.build()

    @staticmethod
    def build_user(
        *,
        current_hour_cao: Sequence[Mapping[str, Any]],
        interval_start_time_seconds: int | None = None,
        long_term_slots: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> str:
        return Reflector.build_context(
            current_hour_cao=current_hour_cao,
            interval_start_time_seconds=interval_start_time_seconds,
            long_term_slots=long_term_slots,
        ).agent_view

    async def summarize(
        self,
        *,
        context: ModelCallContext,
        user: str,
        causal_enabled: bool,
        thinking_mode: str,
        zones: Sequence[str],
        long_term_memory: bool = False,
    ) -> ReflectorResolution:
        raw = await self.client.complete(
            context=context,
            role="reflector",
            system=system_prompt(
                "reflector",
                causal_enabled=causal_enabled,
                long_term_memory=long_term_memory,
            ),
            user=user,
            thinking_mode=thinking_mode,
            response_schema=reflector_response_schema(
                zones=tuple(zones), long_term_memory=long_term_memory
            ),
        )
        try:
            payload = parse_bare_json(raw)
            resolution = resolve_reflector_payload(
                payload,
                zones=zones,
                long_term_memory=long_term_memory,
            )
        except ValueError as error:
            raise ModelContractError(str(error), raw) from error
        return resolution
