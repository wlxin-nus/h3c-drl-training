"""Graph-grounded admission for executable program patches."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from h3c.causal.direction import consistent_program_direction_proof
from h3c.causal.graph import ConfirmedGraph, Edge
from h3c.control.program import (
    apply_patch,
    cited_weather_drivers,
    program_delta,
)


@dataclass(frozen=True)
class CausalAdmission:
    patch: Mapping[str, Any]
    candidate_program: Mapping[str, Any]
    effect: Mapping[str, Any]
    cited_edges: tuple[Edge, ...]
    direction_proof: Mapping[str, Any]


class CausalAdmissionError(ValueError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def causal_admissibility(
    patch: Mapping[str, Any],
    program: Mapping[str, Any],
    graph: ConfirmedGraph,
    *,
    weather_enabled: bool = True,
) -> CausalAdmission:
    """Resolve every citation, enforce relevance, and prove one program direction."""
    canonical_patch = copy.deepcopy(dict(patch))
    candidate = apply_patch(
        program,
        canonical_patch,
        causal_enabled=True,
        weather_enabled=weather_enabled,
    )
    if canonical_patch["op"] == "no_change":
        return CausalAdmission(canonical_patch, candidate, {}, (), {})
    identifiers = list(canonical_patch["causal_edge_ids"])
    missing = [identifier for identifier in identifiers if identifier not in graph.by_id]
    if missing:
        raise CausalAdmissionError(
            "a cited edge identifier is not present in the resolved graph",
            code="unknown_edge_id",
        )
    edges = [graph.by_id[identifier] for identifier in identifiers]
    if not any(
        edge.source == "cooling_setpoint" or edge.target == "cooling_setpoint" for edge in edges
    ):
        raise CausalAdmissionError(
            "no cited edge touches the controlled actuator",
            code="irrelevant_edge",
        )
    drivers = cited_weather_drivers(canonical_patch, program)
    absent_drivers = sorted(
        driver
        for driver in drivers
        if not any(edge.source == driver or edge.target == driver for edge in edges)
    )
    if absent_drivers:
        raise CausalAdmissionError(
            f"weather-conditioned rule is missing confirmed edges for {absent_drivers}",
            code="missing_weather_driver_edge",
        )
    effect = program_delta(program, candidate, weather_enabled=weather_enabled)
    if effect["max_extra_energy_actuation_c"] > 0:
        aggregate = [
            edge
            for edge in graph.edges
            if edge.source == "cooling_setpoint"
            and edge.relation == "Negative Corr"
            and edge.target == "power_meters"
        ]
        if len(aggregate) != 1:
            raise CausalAdmissionError(
                "resolved graph lacks one mandatory actuator-to-site-power edge",
                code="mandatory_site_power_edge_missing",
            )
        if aggregate[0].identifier not in identifiers:
            identifiers.append(aggregate[0].identifier)
            canonical_patch["causal_edge_ids"] = identifiers
            edges.append(aggregate[0])
    try:
        direction_proof = consistent_program_direction_proof(effect, edges)
    except ValueError as error:
        raise CausalAdmissionError(str(error), code="program_direction_undetermined") from error
    canonical_patch["expected_effects"] = direction_proof["expected_effects"]
    canonical_patch["consistent_program_direction_proof"] = direction_proof
    return CausalAdmission(
        patch=canonical_patch,
        candidate_program=candidate,
        effect=effect,
        cited_edges=tuple(edges),
        direction_proof=direction_proof,
    )
