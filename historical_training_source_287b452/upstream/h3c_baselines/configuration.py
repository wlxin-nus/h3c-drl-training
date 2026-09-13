"""Typed baseline plans and the frozen formal matrix."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from h3c.experiments.profiles import load_profile, repository_root, validate_profile

_SUPPORTED_CONTROLLERS = (
    "basic-rbc",
    "enhanced-rbc",
    "c-drl",
    "h-drl",
    "hierarchical-mpc",
)


def _merge_profile(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get("__replace__") is True:
            result[key] = deepcopy(
                {name: item for name, item in value.items() if name != "__replace__"}
            )
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_profile(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass(frozen=True)
class BaselineRunPlan:
    case: str
    controller: str
    evaluation_hours: int = 168
    profile_overrides: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.controller not in _SUPPORTED_CONTROLLERS:
            raise ValueError(f"unknown baseline controller: {self.controller}")
        if self.evaluation_hours <= 0 or self.evaluation_hours > 168:
            raise ValueError("evaluation hours must be in [1, 168]")
        profile = load_profile(self.case)
        if self.controller == "h-drl" and len(profile["zones"]) == 1:
            raise ValueError("H-DRL is not defined for the single-zone case")

    def resolved(self) -> dict[str, Any]:
        overrides = deepcopy(self.profile_overrides or {})
        profile = validate_profile(_merge_profile(load_profile(self.case), overrides))
        value = {
            "schema": "h3c_baseline_run_plan",
            "schema_version": 1,
            "case": self.case,
            "controller": self.controller,
            "evaluation_hours": self.evaluation_hours,
            "profile_overrides": overrides,
            "case_profile": profile,
        }
        value["plan_identity"] = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return value


def load_formal_suite(path: Path | None = None) -> dict[str, Any]:
    source = path or repository_root() / "configs" / "baselines" / "formal.json"
    value = cast(dict[str, Any], json.loads(source.read_text(encoding="utf-8")))
    if value.get("schema") != "h3c_baseline_suite" or value.get("name") != "formal":
        raise ValueError("baseline suite schema is invalid")
    return value


def formal_evaluation_plans() -> list[BaselineRunPlan]:
    suite = load_formal_suite()
    return [
        BaselineRunPlan(
            case=case,
            controller=controller,
            evaluation_hours=int(load_profile(case)["protocol"]["formal_evaluation_days"]) * 24,
        )
        for case in suite["case_order"]
        for controller in suite["controller_order"][case]
    ]


def load_hierarchical_mpc_config(path: Path | None = None) -> dict[str, Any]:
    source = path or repository_root() / "configs" / "baselines" / "hierarchical_mpc.json"
    value = cast(dict[str, Any], json.loads(source.read_text(encoding="utf-8")))
    if value.get("schema") != "h3c_hierarchical_mpc" or value.get("schema_version") != 1:
        raise ValueError("hierarchical MPC configuration schema is invalid")
    if value.get("case_order") != ["SZ_Air", "MZ_Hydro", "MZ_Air"]:
        raise ValueError("hierarchical MPC case order is invalid")
    if value.get("fit_checkpoints") != [8, 16, 32, 64]:
        raise ValueError("hierarchical MPC fit checkpoints are invalid")
    if value.get("holdout_episodes") != 4 or value.get("warmup_days") != 7:
        raise ValueError("hierarchical MPC episode budget is invalid")
    excitation = value.get("excitation", {})
    if excitation.get("occupied_bounds_c") != [23.5, 26.5] or excitation.get(
        "unoccupied_bounds_c"
    ) != [20.0, 30.0]:
        raise ValueError("hierarchical MPC identification-support bounds are invalid")
    return value


def mpc_formal_evaluation_plans() -> list[BaselineRunPlan]:
    config = load_hierarchical_mpc_config()
    return [
        BaselineRunPlan(
            case=case,
            controller="hierarchical-mpc",
            evaluation_hours=int(load_profile(case)["protocol"]["formal_evaluation_days"]) * 24,
        )
        for case in config["case_order"]
    ]
