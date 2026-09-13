"""Deterministic execution of one current control program per zone."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from h3c.assurance.action import action_assurance
from h3c.control.program import run_program


def build_program_observations(
    zones: Sequence[str],
    *,
    current_occupancy: Mapping[str, float],
    future_occupancy: Mapping[str, Sequence[float]],
    last_setpoints_c: Mapping[str, float],
    last_pmv: Mapping[str, float],
    last_occupancy: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """Build the exact observation surface consumed by the canonical program."""
    zone_set = set(zones)
    if any(
        set(values) != zone_set
        for values in (
            current_occupancy,
            future_occupancy,
            last_setpoints_c,
            last_pmv,
            last_occupancy,
        )
    ):
        raise ValueError("program observation inputs must cover exactly the configured zones")
    result: dict[str, dict[str, Any]] = {}
    for zone in zones:
        ahead = [float(value) for value in future_occupancy[zone]]
        if len(ahead) != 4:
            raise ValueError("program occupancy lookahead must contain exactly four steps")
        result[zone] = {
            "current_occupancy": float(current_occupancy[zone]),
            "occ_ahead": ahead,
            "last_setpoint": float(last_setpoints_c[zone]),
            "last_pmv": float(last_pmv[zone]),
            "last_occupancy": float(last_occupancy[zone]),
        }
    return result


def execute_zone_programs(
    programs: Mapping[str, Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, dict[str, Any]]]:
    """Interpret and assure each zone program exactly once."""
    if set(programs) != set(observations):
        raise ValueError("programs and observations must cover the same zones")
    proposals: dict[str, dict[str, Any]] = {}
    setpoints: dict[str, float] = {}
    audits: dict[str, dict[str, Any]] = {}
    for zone, program in programs.items():
        observation = observations[zone]
        proposal = run_program(program, observation)
        setpoint, audit = action_assurance(proposal, observation)
        proposals[zone] = proposal
        setpoints[zone] = setpoint
        audits[zone] = audit
    return proposals, setpoints, audits
