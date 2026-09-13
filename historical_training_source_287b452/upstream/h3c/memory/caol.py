"""Deterministic CAOL working memory and optional regime experience storage."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from h3c.control.program import setpoint_effect_facts
from h3c.runtime.comfort import COMFORT_BAND

Regime = Literal[
    "unoccupied",
    "occupancy_transition",
    "steady_state_occupancy",
]

REGIMES: tuple[Regime, ...] = (
    "unoccupied",
    "occupancy_transition",
    "steady_state_occupancy",
)
_REGIME_SET = set(REGIMES)
_PROVENANCE_PATTERN = re.compile(
    r"(?:\b(?:hour|step)\s*#?\d+\b|\bcaol[_-]|小时\s*\d+|步骤\s*\d+)",
    re.IGNORECASE,
)
AGENT_VISIBLE_NUMERIC_DECIMALS = 4


def agent_visible_number(value: int | float) -> float:
    """Normalize one finite scalar to the precision used in Agent-visible memory."""
    number = float(value)
    return round(number, AGENT_VISIBLE_NUMERIC_DECIMALS) if math.isfinite(number) else number


def _rounded(value: Any) -> Any:
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return agent_visible_number(value)
    if isinstance(value, Mapping):
        return {str(key): _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    return copy.deepcopy(value)


def _normalized_text(value: Any, *, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or len(text) > maximum:
        return None
    return text


def classify_regime(observation: Mapping[str, Any]) -> Regime:
    """Classify one control step using only the visible occupancy channels."""
    current = float(observation["current_occupancy"]) > 0
    previous = float(observation["last_occupancy"]) > 0
    ahead_raw = observation.get("occupancy_next_steps", observation.get("occ_ahead"))
    if not isinstance(ahead_raw, Sequence) or isinstance(ahead_raw, (str, bytes)):
        raise ValueError("occupancy_next_steps must be a sequence")
    if len(ahead_raw) != 4:
        raise ValueError("regime classification requires exactly four visible future steps")
    enters_within_visible_window = any(float(value) > 0 for value in ahead_raw)
    if not current and not enters_within_visible_window:
        return "unoccupied"
    if (not current and enters_within_visible_window) or (current and not previous):
        return "occupancy_transition"
    if current and previous:
        return "steady_state_occupancy"
    raise ValueError("occupancy channels do not map to a registered regime")


def _direction_reversals(values: Sequence[float]) -> int:
    deltas = [right - left for left, right in zip(values, values[1:], strict=False)]
    directions = [1 if delta > 0 else -1 for delta in deltas if abs(delta) > 1e-12]
    return sum(left != right for left, right in zip(directions, directions[1:], strict=False))


def build_hourly_cao(
    *,
    hour: int,
    zone: str,
    step_rows: Sequence[Mapping[str, Any]],
    program_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Build deterministic Context/Action/Outcome evidence for one completed zone-hour."""
    rows = [dict(row) for row in step_rows]
    expected_steps = list(range(hour * 4, hour * 4 + 4))
    if len(rows) != 4 or [row.get("step") for row in rows] != expected_steps:
        raise ValueError("CAOL requires exactly one aligned four-step hour")
    if any(row.get("zone") != zone for row in rows):
        raise ValueError("CAOL rows must belong to one configured zone")

    regime_steps: dict[str, list[int]] = {regime: [] for regime in REGIMES}
    for row in rows:
        regime_steps[classify_regime(row["observation"])].append(int(row["step"]))
    covered = {regime: steps for regime, steps in regime_steps.items() if steps}

    setpoints = [float(row["final_setpoint_c"]) for row in rows]
    setpoint_facts = [
        setpoint_effect_facts(
            current_occupancy=row["observation"]["current_occupancy"],
            applied_setpoint_c=row["final_setpoint_c"],
        )
        for row in rows
    ]
    pmv_values = [float(row["outcome"]["pmv"]) for row in rows]
    occupancies = [float(row["outcome"]["effective_occupancy"]) for row in rows]
    patch = program_decision.get("patch")
    proposal = copy.deepcopy(dict(patch)) if isinstance(patch, Mapping) else None
    rejection = program_decision.get("rejection")
    rejection_code = rejection.get("code") if isinstance(rejection, Mapping) else None

    shield = []
    for row in rows:
        assurance = row["action_assurance"]
        shield.append(
            {
                "step": int(row["step"]),
                "actuator_bounds": bool(assurance["actuator_bounds_triggered"]),
                "setpoint_rate_limit": bool(assurance["setpoint_rate_limit_triggered"]),
                "comfort_recovery": bool(assurance["comfort_recovery_triggered"]),
            }
        )

    occupied_absolute_pmv = [
        abs(pmv) for pmv, occupancy in zip(pmv_values, occupancies, strict=True) if occupancy > 0
    ]
    zone_h = sum(
        0.25 if occupancy > 0 and abs(pmv) > 0.5 else 0.0
        for pmv, occupancy in zip(pmv_values, occupancies, strict=True)
    )
    pmv_h = sum(
        max(0.0, abs(pmv) - 0.5) * 0.25 if occupancy > 0 else 0.0
        for pmv, occupancy in zip(pmv_values, occupancies, strict=True)
    )
    setpoint_path = [float(rows[0]["observation"]["last_setpoint"]), *setpoints]
    total_variation = sum(
        abs(right - left) for left, right in zip(setpoint_path, setpoint_path[1:], strict=False)
    )

    objective_fields = (
        "site_step_reward",
        "site_energy_penalty",
        "site_comfort_penalty",
        "site_smoothness_penalty",
        "zone_comfort_penalty_contribution",
        "zone_smoothness_penalty_contribution",
    )
    raw_objective_feedback = [row["outcome"].get("objective_feedback") for row in rows]
    objective_feedback: dict[str, Any] | None = None
    if any(feedback is not None for feedback in raw_objective_feedback):
        if any(feedback is None for feedback in raw_objective_feedback):
            raise ValueError("objective feedback cannot cover only part of a completed hour")
        objective_history: dict[str, list[float]] = {field: [] for field in objective_fields}
        for feedback in raw_objective_feedback:
            if not isinstance(feedback, Mapping) or set(feedback) != set(objective_fields):
                raise ValueError("completed step lacks the exact objective-feedback contract")
            for field in objective_fields:
                value = feedback[field]
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError("objective-feedback values must be finite numbers")
                objective_history[field].append(agent_visible_number(float(value)))
        objective_feedback = {
            "interval_reward": agent_visible_number(sum(objective_history["site_step_reward"])),
            **objective_history,
        }

    observed_context_history: dict[str, list[float]] = {}
    scalar_observation_fields = {
        "outdoor_temperature_c": "outdoor_temp_c",
        "solar_irradiance_w_m2": "solar_irr",
        "electricity_price": "electricity_price",
    }
    for output_name, observation_name in scalar_observation_fields.items():
        values = [row["observation"].get(observation_name) for row in rows]
        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            observed_context_history[output_name] = [float(value) for value in values]
    headroom_fields = {
        "temp_rise_to_warm_pmv_edge_c": "warmer_c",
        "temp_drop_to_cool_pmv_edge_c": "cooler_c",
    }
    for output_name, headroom_name in headroom_fields.items():
        values = []
        for row in rows:
            headroom = row["observation"].get("comfort_headroom_c")
            values.append(headroom.get(headroom_name) if isinstance(headroom, Mapping) else None)
        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            observed_context_history[output_name] = [float(value) for value in values]

    deterministic_program_effect: dict[str, Any] | None = None
    if proposal is not None:
        expected_effects = proposal.get("expected_effects")
        proof = proposal.get("consistent_program_direction_proof")
        if isinstance(expected_effects, list) and isinstance(proof, Mapping):
            program_direction = proof.get("program_direction")
            if isinstance(program_direction, str):
                deterministic_program_effect = {
                    "program_direction": program_direction,
                    "expected_effects": copy.deepcopy(expected_effects),
                }

    cao = {
        "hour": hour,
        "zone": zone,
        "context": {
            "regime_step_coverage": covered,
            **(
                {
                    "abs_pmv_score_limit": COMFORT_BAND,
                    "observed_context_history": observed_context_history,
                }
                if observed_context_history
                else {}
            ),
            "initial_observation": {
                "zone_temperature_c": rows[0]["observation"]["zone_temperature_c"],
                "current_occupancy": rows[0]["observation"]["current_occupancy"],
                "last_occupancy": rows[0]["observation"]["last_occupancy"],
                "occupancy_next_steps": copy.deepcopy(
                    rows[0]["observation"].get(
                        "occupancy_next_steps", rows[0]["observation"].get("occ_ahead")
                    )
                ),
                "last_pmv": rows[0]["observation"]["last_pmv"],
                "last_setpoint_c": rows[0]["observation"]["last_setpoint"],
            },
        },
        "action": {
            "proposal": proposal,
            "admission": {
                "status": program_decision.get("status"),
                "completed_validation_stages": copy.deepcopy(
                    program_decision.get("completed_validation_stages", [])
                ),
                **({"rejection_code": rejection_code} if rejection_code is not None else {}),
            },
            "program_version_before": int(
                program_decision.get(
                    "program_version_before", program_decision["current_program_version"]
                )
            ),
            "program_version_after": int(program_decision["current_program_version"]),
            "actual_setpoints_c": setpoints,
            "regime_base_setpoints_c": [fact["regime_base_setpoint_c"] for fact in setpoint_facts],
            "setpoint_offsets_from_regime_base_c": [
                fact["setpoint_offset_from_regime_base_c"] for fact in setpoint_facts
            ],
            "cooling_effects_relative_to_regime_base": [
                fact["cooling_effect_relative_to_regime_base"] for fact in setpoint_facts
            ],
            "matched_rules": [row["interpreter"].get("matched_rule") for row in rows],
            "shield": shield,
            **(
                {"deterministic_program_effect": deterministic_program_effect}
                if deterministic_program_effect is not None
                else {}
            ),
        },
        "outcome": {
            "zone_temperatures_c": [float(row["outcome"]["zone_temperature_c"]) for row in rows],
            "pmv": pmv_values,
            "effective_occupancy": occupancies,
            "site_cost": sum(float(row["outcome"]["cost"]) for row in rows),
            "site_energy_kwh": sum(
                float(row["outcome"]["power_w"]) * 0.25 / 1000.0 for row in rows
            ),
            "discomfort_zone_hours": zone_h,
            "discomfort_pmv_hours": pmv_h,
            "occupied_peak_absolute_pmv": max(occupied_absolute_pmv, default=0.0),
            "setpoint_total_variation_c": total_variation,
            "setpoint_direction_reversals": _direction_reversals(setpoint_path),
            **(
                {"objective_feedback": objective_feedback} if objective_feedback is not None else {}
            ),
        },
    }
    return cast(dict[str, Any], _rounded(cao))


def select_caol_working_memory(
    records: Sequence[Mapping[str, Any]],
    *,
    current_hour: int,
    zone: str | None,
    working_memory_hours: int,
    zones: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Select the exact completed CAOL window for one zone or the full building."""
    if working_memory_hours not in {1, 2, 3}:
        raise ValueError("working memory must be one, two, or three hours")
    first_hour = current_hour - working_memory_hours
    selected = [
        copy.deepcopy(dict(record))
        for record in records
        if isinstance(record.get("hour"), int)
        and first_hour <= int(record["hour"]) < current_hour
        and (zone is None or record.get("zone") == zone)
    ]
    selected.sort(key=lambda row: (int(row["hour"]), str(row["zone"])))
    if current_hour < working_memory_hours:
        return []
    expected_zones = {zone} if zone is not None else set(zones or ())
    if not expected_zones:
        raise ValueError("configured zones are required for building CAOL selection")
    for expected_hour in range(first_hour, current_hour):
        observed = {str(row["zone"]) for row in selected if int(row["hour"]) == expected_hour}
        if observed != expected_zones:
            return []
    return cast(list[dict[str, Any]], _rounded(selected))


def attach_hourly_lessons(
    cao_records: Sequence[Mapping[str, Any]],
    lessons: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Attach only validated lessons; unavailable lessons are omitted entirely."""
    completed: list[dict[str, Any]] = []
    for record in cao_records:
        row = copy.deepcopy(dict(record))
        zone = str(row["zone"])
        if zone in lessons:
            row["lesson"] = lessons[zone]
        completed.append(row)
    return completed


def empty_regime_store(zones: Sequence[str]) -> dict[str, dict[str, dict[str, Any] | None]]:
    """Create three empty deterministic slots for every configured zone."""
    return {zone: {regime: None for regime in REGIMES} for zone in zones}


def active_experiences(
    store: Mapping[str, Mapping[str, Mapping[str, Any] | None]], zone: str
) -> list[dict[str, Any]]:
    """Return only active experiences in the fixed regime order for Executor exposure."""
    slots = store[zone]
    active: list[dict[str, Any]] = []
    for regime in REGIMES:
        entry = slots[regime]
        if entry is not None:
            active.append(copy.deepcopy(dict(entry)))
    return active


def reflector_slot_view(
    store: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
    zone: str,
    observed_regimes: Sequence[str],
) -> list[dict[str, Any]]:
    """Expose only this hour's eligible slots, in first-observed-step order."""
    view: list[dict[str, Any]] = []
    seen: set[str] = set()
    for regime in observed_regimes:
        if regime not in _REGIME_SET or regime in seen:
            raise ValueError("observed regimes must be unique registered values")
        seen.add(regime)
        entry = store[zone][regime]
        if entry is None:
            view.append({"regime": regime, "state": "empty"})
        else:
            view.append(copy.deepcopy(dict(entry)))
    return view


@dataclass(frozen=True)
class ReflectorResolution:
    lessons: dict[str, str]
    operations: dict[str, dict[str, Any]]
    issues: tuple[dict[str, str], ...]

    @property
    def clean(self) -> bool:
        return not self.issues


def _operation_shape(operation: Mapping[str, Any]) -> bool:
    op = operation.get("op")
    fields = set(operation)
    if op == "no_change":
        return fields == {"zone", "op"}
    if op == "add":
        return fields == {"zone", "op", "regime", "experience"}
    if op == "replace":
        return fields == {
            "zone",
            "op",
            "regime",
            "expected_revision",
            "experience",
        }
    if op == "delete":
        return fields == {"zone", "op", "regime", "expected_revision"}
    return False


def resolve_reflector_payload(
    payload: Mapping[str, Any],
    *,
    zones: Sequence[str],
    long_term_memory: bool,
) -> ReflectorResolution:
    """Resolve a structurally valid root while isolating per-zone model mistakes."""
    expected_root = (
        {"hourly_lessons", "memory_operations"} if long_term_memory else {"hourly_lessons"}
    )
    if set(payload) != expected_root or not isinstance(payload.get("hourly_lessons"), list):
        raise ValueError("Reflector output does not match the active root contract")
    if long_term_memory and not isinstance(payload.get("memory_operations"), list):
        raise ValueError("Reflector memory_operations must be a list")

    lessons: dict[str, str] = {}
    operations: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, str]] = []
    configured = set(zones)
    for row in payload["hourly_lessons"]:
        if not isinstance(row, Mapping) or set(row) != {"zone", "lesson"}:
            issues.append({"zone": "unknown", "code": "invalid_lesson_shape"})
            continue
        zone = str(row.get("zone"))
        lesson = _normalized_text(row.get("lesson"), maximum=480)
        if zone not in configured or zone in lessons or lesson is None:
            issues.append({"zone": zone, "code": "invalid_lesson"})
            continue
        lessons[zone] = lesson
    for zone in zones:
        if zone not in lessons:
            issues.append({"zone": zone, "code": "missing_lesson"})

    if long_term_memory:
        for row in payload["memory_operations"]:
            if not isinstance(row, Mapping) or not _operation_shape(row):
                issues.append({"zone": "unknown", "code": "invalid_operation_shape"})
                continue
            operation = copy.deepcopy(dict(row))
            zone = str(operation.get("zone"))
            regime = operation.get("regime")
            experience = operation.get("experience")
            revision = operation.get("expected_revision")
            valid = zone in configured and zone not in operations
            valid = valid and (regime in _REGIME_SET if operation["op"] != "no_change" else True)
            if operation["op"] in {"replace", "delete"}:
                revision_valid = isinstance(revision, int) and not isinstance(revision, bool)
                valid = valid and revision_valid
                if revision_valid:
                    revision_number = cast(int, revision)
                    valid = valid and revision_number >= 1
            if operation["op"] in {"add", "replace"}:
                cleaned = _normalized_text(experience, maximum=480)
                valid = valid and cleaned is not None
                if cleaned is not None:
                    valid = valid and _PROVENANCE_PATTERN.search(cleaned) is None
                    operation["experience"] = cleaned
            if not valid:
                issues.append({"zone": zone, "code": "invalid_memory_operation"})
                continue
            operations[zone] = operation
        for zone in zones:
            if zone not in operations:
                issues.append({"zone": zone, "code": "missing_memory_operation"})
    return ReflectorResolution(lessons, operations, tuple(issues))


def apply_memory_operations(
    store: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
    operations: Mapping[str, Mapping[str, Any]],
    *,
    zones: Sequence[str],
    hour: int,
    observed_regimes: Mapping[str, Sequence[str]],
) -> tuple[dict[str, dict[str, dict[str, Any] | None]], list[dict[str, Any]]]:
    """Apply at most one CAS-protected operation per zone and emit append-only audit rows."""
    updated: dict[str, dict[str, dict[str, Any] | None]] = {}
    for zone, slots in store.items():
        updated[zone] = {}
        for slot_regime in REGIMES:
            entry = slots[slot_regime]
            updated[zone][slot_regime] = copy.deepcopy(dict(entry)) if entry is not None else None
    audits: list[dict[str, Any]] = []
    for zone in zones:
        operation = copy.deepcopy(dict(operations[zone])) if zone in operations else None
        before_slots = copy.deepcopy(updated[zone])
        status = "accepted"
        rejection_code: str | None = None
        if operation is None:
            status = "rejected"
            rejection_code = "missing_or_invalid_operation"
        elif operation["op"] != "no_change":
            regime = str(operation["regime"])
            if regime not in set(observed_regimes.get(zone, ())):
                status = "rejected"
                rejection_code = "regime_not_observed_this_hour"
            else:
                current = updated[zone][regime]
                if operation["op"] == "add":
                    if current is not None:
                        status = "rejected"
                        rejection_code = "slot_not_empty"
                    else:
                        updated[zone][regime] = {
                            "regime": regime,
                            "revision": 1,
                            "experience": operation["experience"],
                        }
                elif operation["op"] in {"replace", "delete"}:
                    if current is None:
                        status = "rejected"
                        rejection_code = "slot_empty"
                    elif int(operation["expected_revision"]) != int(current["revision"]):
                        status = "rejected"
                        rejection_code = "revision_conflict"
                    elif operation["op"] == "delete":
                        updated[zone][regime] = None
                    else:
                        updated[zone][regime] = {
                            "regime": regime,
                            "revision": int(current["revision"]) + 1,
                            "experience": operation["experience"],
                        }
        audit: dict[str, Any] = {
            "hour": hour,
            "zone": zone,
            "requested_operation": operation,
            "status": status,
            "before": before_slots,
            "after": copy.deepcopy(updated[zone]),
        }
        if rejection_code is not None:
            audit["rejection_code"] = rejection_code
        audits.append(audit)
    return updated, audits


def validate_memory_refs(
    refs: Sequence[Mapping[str, Any]],
    exposed: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate valid and invalid audit-only references without changing a patch."""
    allowed = {(str(row["regime"]), int(row["revision"])) for row in exposed}
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in refs:
        row = copy.deepcopy(dict(raw))
        revision = row.get("revision")
        structural = (
            set(row) == {"regime", "revision"}
            and isinstance(revision, int)
            and not isinstance(revision, bool)
        )
        key = (str(row.get("regime")), revision if structural else -1)
        if structural and key in allowed and key not in seen:
            valid.append(row)
            seen.add(key)
        else:
            invalid.append(row)
    return valid, invalid
