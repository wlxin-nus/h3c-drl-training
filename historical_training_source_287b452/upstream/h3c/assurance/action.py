"""Mandatory ordered action assurance for every deterministic control action."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

COMFORT_INTERLOCK_PMV = 0.7
COMFORT_ANCHOR_C = 25.0
SETPOINT_RATE_LIMIT_C = 1.0
ACTUATOR_BOUNDS_C = (20.0, 30.0)
ACTION_ASSURANCE_ORDER = (
    "comfort_recovery",
    "setpoint_rate_limit",
    "actuator_bounds",
)


def action_assurance(
    proposal: Mapping[str, Any], observation: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    """Apply comfort recovery, rate limiting, then actuator bounds exactly once."""
    previous = float(observation["last_setpoint"])
    pmv = float(observation["last_pmv"])
    occupied_now = float(observation["current_occupancy"]) > 0
    occupied_last = float(observation["last_occupancy"]) > 0
    interpreter_setpoint = float(proposal["setpoint"])
    setpoint = interpreter_setpoint
    audit: dict[str, Any] = {
        "order": list(ACTION_ASSURANCE_ORDER),
        "interpreter_setpoint": interpreter_setpoint,
        "comfort_recovery_triggered": False,
        "comfort_recovery_reason": None,
        "comfort_recovery_input_setpoint": interpreter_setpoint,
        "comfort_recovery_output_setpoint": interpreter_setpoint,
        "setpoint_rate_limit_triggered": False,
        "setpoint_rate_limit_delta_before_c": 0.0,
        "setpoint_rate_limit_output_setpoint": interpreter_setpoint,
        "actuator_bounds_triggered": False,
        "actuator_bound": None,
        "final_setpoint": None,
    }
    if occupied_now and occupied_last and abs(pmv) > COMFORT_INTERLOCK_PMV:
        if pmv > COMFORT_INTERLOCK_PMV:
            setpoint = min(previous, COMFORT_ANCHOR_C)
            reason = "occupied_hot_pmv_reset"
        else:
            setpoint = max(previous, COMFORT_ANCHOR_C)
            reason = "occupied_cold_pmv_reset"
        audit["comfort_recovery_triggered"] = True
        audit["comfort_recovery_reason"] = reason
        audit["comfort_recovery_output_setpoint"] = setpoint
    elif not bool(proposal["exempt_rate"]):
        base = float(proposal["base"])
        lower, upper = base - SETPOINT_RATE_LIMIT_C, base + SETPOINT_RATE_LIMIT_C
        audit["setpoint_rate_limit_delta_before_c"] = setpoint - base
        if setpoint < lower or setpoint > upper:
            audit["setpoint_rate_limit_triggered"] = True
        setpoint = max(lower, min(upper, setpoint))
        audit["setpoint_rate_limit_output_setpoint"] = setpoint
    else:
        audit["setpoint_rate_limit_delta_before_c"] = setpoint - float(proposal["base"])
        audit["setpoint_rate_limit_output_setpoint"] = setpoint

    lower_bound, upper_bound = ACTUATOR_BOUNDS_C
    if setpoint < lower_bound or setpoint > upper_bound:
        audit["actuator_bounds_triggered"] = True
        audit["actuator_bound"] = "lower" if setpoint < lower_bound else "upper"
    setpoint = max(lower_bound, min(upper_bound, setpoint))
    audit["final_setpoint"] = setpoint
    return setpoint, audit
