"""Canonical H3C cooling program used without Agent program updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from h3c.control.program import load_program
from h3c.control.program_execution import build_program_observations, execute_zone_programs


class EnhancedRbcController:
    def __init__(self, zones: Sequence[str], program_path: Path) -> None:
        self.zones = tuple(zones)
        self.programs = {zone: load_program(program_path, zone) for zone in self.zones}

    def decide(
        self,
        *,
        occupancy: Mapping[str, float],
        future_occupancy: Mapping[str, Sequence[float]],
        last_setpoints_c: Mapping[str, float],
        last_pmv: Mapping[str, float],
        last_occupancy: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        observations = build_program_observations(
            self.zones,
            current_occupancy=occupancy,
            future_occupancy=future_occupancy,
            last_setpoints_c=last_setpoints_c,
            last_pmv=last_pmv,
            last_occupancy=last_occupancy,
        )
        proposals, setpoints, audits = execute_zone_programs(self.programs, observations)
        diagnostics = {
            zone: {"interpreter": proposals[zone], "action_assurance": audits[zone]}
            for zone in self.zones
        }
        return setpoints, diagnostics
