"""Internal full accepted-patch ledger and deterministic program replay."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from h3c.control.program import apply_patch, program_hash


@dataclass(frozen=True)
class AcceptedProgramUpdate:
    step: int
    hour: int
    zone: str
    patch: Mapping[str, Any]
    version_before: int
    version_after: int
    program_hash_before: str
    program_hash_after: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "hour": self.hour,
            "zone": self.zone,
            "patch": copy.deepcopy(dict(self.patch)),
            "version_before": self.version_before,
            "version_after": self.version_after,
            "program_hash_before": self.program_hash_before,
            "program_hash_after": self.program_hash_after,
        }


class ProgramLedger:
    """Own current program state while retaining every accepted update internally."""

    def __init__(self, initial_program: Mapping[str, Any], *, causal_enabled: bool = True) -> None:
        self.initial_program = copy.deepcopy(dict(initial_program))
        self.current_program = copy.deepcopy(dict(initial_program))
        self.causal_enabled = causal_enabled
        self.version = 0
        self.accepted_updates: list[AcceptedProgramUpdate] = []

    def commit(
        self,
        patch: Mapping[str, Any],
        *,
        step: int,
        hour: int,
        weather_enabled: bool = True,
    ) -> AcceptedProgramUpdate:
        before_hash = program_hash(self.current_program)
        candidate = apply_patch(
            self.current_program,
            patch,
            causal_enabled=self.causal_enabled,
            weather_enabled=weather_enabled,
        )
        update = AcceptedProgramUpdate(
            step=step,
            hour=hour,
            zone=str(self.current_program["zone"]),
            patch=copy.deepcopy(dict(patch)),
            version_before=self.version,
            version_after=self.version + 1,
            program_hash_before=before_hash,
            program_hash_after=program_hash(candidate),
        )
        self.current_program = candidate
        self.version += 1
        self.accepted_updates.append(update)
        return update

    def prompt_view(self) -> dict[str, Any]:
        """Return the current complete program without the accepted ledger."""
        return {
            "program_version": self.version,
            "params": copy.deepcopy(self.current_program["params"]),
            "rules": copy.deepcopy(self.current_program["rules"]),
        }

    def replay(self, *, weather_enabled: bool = True) -> dict[str, Any]:
        program = copy.deepcopy(self.initial_program)
        version = 0
        for update in self.accepted_updates:
            if (
                program_hash(program) != update.program_hash_before
                or version != update.version_before
            ):
                raise ValueError("accepted ledger cannot be replayed from the initial program")
            program = apply_patch(
                program,
                update.patch,
                causal_enabled=self.causal_enabled,
                weather_enabled=weather_enabled,
            )
            version += 1
            if (
                program_hash(program) != update.program_hash_after
                or version != update.version_after
            ):
                raise ValueError("accepted ledger replay identity diverged")
        if program_hash(program) != program_hash(self.current_program) or version != self.version:
            raise ValueError("accepted ledger replay does not reproduce current state")
        return program
