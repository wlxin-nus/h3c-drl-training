"""One ordered owner for patch admission and budget settlement."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from h3c.causal.admissibility import CausalAdmissionError, causal_admissibility
from h3c.causal.graph import ConfirmedGraph
from h3c.control.budget import BudgetLedger
from h3c.control.program import ProgramError, apply_patch, program_delta

VALIDATION_STAGES = (
    "program_validation",
    "causal_admissibility",
    "consistent_program_direction_proof",
    "energy_budget_validation",
)


@dataclass(frozen=True)
class Rejection:
    stage: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"stage": self.stage, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    patch: Mapping[str, Any]
    candidate_program: Mapping[str, Any] | None
    effect: Mapping[str, Any] | None
    completed_stages: tuple[str, ...]
    rejection: Rejection | None


def validate_candidate(
    patch: Mapping[str, Any],
    program: Mapping[str, Any],
    *,
    graph: ConfirmedGraph | None,
    ledger: BudgetLedger | None,
    zone: str,
    step: int,
    causal_enabled: bool = True,
    coordination_enabled: bool = True,
    weather_enabled: bool = True,
) -> ValidationResult:
    """Run the registered chain in order and never commit a partial result."""
    completed: list[str] = []
    canonical_patch = copy.deepcopy(dict(patch))
    try:
        candidate = apply_patch(
            program,
            canonical_patch,
            causal_enabled=causal_enabled,
            weather_enabled=weather_enabled,
        )
        effect = (
            {}
            if canonical_patch.get("op") == "no_change"
            else program_delta(program, candidate, weather_enabled=weather_enabled)
        )
        completed.append("program_validation")
    except ProgramError as error:
        return ValidationResult(
            False,
            canonical_patch,
            None,
            None,
            tuple(completed),
            Rejection("program_validation", error.code, str(error)),
        )

    if causal_enabled:
        if graph is None:
            return ValidationResult(
                False,
                canonical_patch,
                None,
                None,
                tuple(completed),
                Rejection(
                    "causal_admissibility",
                    "confirmed_graph_missing",
                    "causal validation requires a resolved confirmed graph",
                ),
            )
        try:
            admission = causal_admissibility(
                canonical_patch,
                program,
                graph,
                weather_enabled=weather_enabled,
            )
            canonical_patch = dict(admission.patch)
            candidate = dict(admission.candidate_program)
            effect = dict(admission.effect)
            completed.extend(("causal_admissibility", "consistent_program_direction_proof"))
        except CausalAdmissionError as error:
            stage = (
                "consistent_program_direction_proof"
                if error.code == "program_direction_undetermined"
                else "causal_admissibility"
            )
            return ValidationResult(
                False,
                canonical_patch,
                None,
                None,
                tuple(completed),
                Rejection(stage, error.code, str(error)),
            )

    if coordination_enabled:
        if ledger is None:
            return ValidationResult(
                False,
                canonical_patch,
                None,
                None,
                tuple(completed),
                Rejection(
                    "energy_budget_validation",
                    "budget_ledger_missing",
                    "coordinated validation requires the current hourly ledger",
                ),
            )
        amount = float((effect or {}).get("max_extra_energy_actuation_c", 0.0))
        rejection = ledger.energy_budget_validation(
            zone,
            amount,
            parameter=str(canonical_patch.get("param", canonical_patch.get("op"))),
            step=step,
        )
        if rejection is not None:
            return ValidationResult(
                False,
                canonical_patch,
                None,
                effect,
                tuple(completed),
                Rejection("energy_budget_validation", rejection.code, rejection.message),
            )
        completed.append("energy_budget_validation")
    return ValidationResult(
        True,
        canonical_patch,
        candidate,
        effect,
        tuple(completed),
        None,
    )
