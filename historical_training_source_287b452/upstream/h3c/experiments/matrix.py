"""Registered experiment matrices and execution identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from h3c.causal.graph import derive_variant, load_graph
from h3c.experiments.profiles import profiles, repository_root
from h3c.experiments.settings import (
    evaluation_start_seconds,
    load_diagnostic_window_catalog,
    load_graph_mutation_catalog,
    load_runtime_contract,
    load_suite_contract,
)

SUITES = (
    "main",
    "memory",
    "causal-ablation",
    "graph-sensitivity",
    "coordination-ablation",
    "thinking-ablation",
)


@dataclass(frozen=True)
class RunPlan:
    profile: str
    controller: str
    working_memory_hours: int
    causal_enabled: bool
    coordination_enabled: bool
    thinking_policy: str
    graph_mutation: dict[str, Any] | None
    evaluation_hours: int = 168
    long_term_memory: bool = False
    model_provider: str | None = None
    diagnostic_window: str | None = None

    def __post_init__(self) -> None:
        loaded = profiles()
        if not isinstance(self.profile, str) or self.profile not in loaded:
            raise ValueError("run profile is not registered")
        if self.controller not in {"deterministic_baseline", "h3c_agent"}:
            raise ValueError("run controller is not registered")
        if (
            isinstance(self.working_memory_hours, bool)
            or not isinstance(self.working_memory_hours, int)
            or self.working_memory_hours not in {1, 2, 3}
        ):
            raise ValueError("working memory must be one, two, or three hours")
        if not isinstance(self.causal_enabled, bool) or not isinstance(
            self.coordination_enabled, bool
        ):
            raise ValueError("causal and coordination flags must be boolean")
        if not isinstance(self.long_term_memory, bool):
            raise ValueError("long-term memory flag must be boolean")
        if self.controller == "deterministic_baseline":
            if self.model_provider is not None:
                raise ValueError("baseline plans cannot select a model provider")
        elif self.effective_model_provider() not in load_runtime_contract()["model"]["providers"]:
            raise ValueError("model provider is not registered")
        if self.thinking_policy not in {"occupancy_routed", "all_roles_disabled"}:
            raise ValueError("thinking policy is not registered")
        formal_hours = int(loaded[self.profile]["protocol"]["formal_evaluation_days"]) * 24
        if (
            isinstance(self.evaluation_hours, bool)
            or not isinstance(self.evaluation_hours, int)
            or self.evaluation_hours not in {6, formal_hours}
        ):
            raise ValueError("evaluation hours must be a registered validation or formal duration")
        if not self.causal_enabled and self.graph_mutation is not None:
            raise ValueError("causal-off runs cannot carry graph mutations")
        if self.controller == "deterministic_baseline" and (
            self.causal_enabled
            or self.coordination_enabled
            or self.thinking_policy != "all_roles_disabled"
            or self.graph_mutation is not None
            or self.working_memory_hours != 1
            or self.long_term_memory
        ):
            raise ValueError("baseline plans cannot carry Agent-only factors")
        if self.graph_mutation is not None:
            registered = load_graph_mutation_catalog().values()
            if not isinstance(self.graph_mutation, dict) or self.graph_mutation not in registered:
                raise ValueError("graph mutation is not a registered declarative factor")
            profile = loaded[self.profile]
            graph = load_graph(repository_root() / profile["graph"])
            if graph.profile != self.profile or graph.zones != tuple(profile["zones"]):
                raise ValueError("graph identity does not match the run profile")
            derive_variant(graph, self.graph_mutation)
        if self.diagnostic_window is not None:
            windows = load_diagnostic_window_catalog()
            if self.diagnostic_window not in windows:
                raise ValueError("diagnostic window is not registered")
            window = windows[self.diagnostic_window]
            if (
                window["profile"] != self.profile
                or window["evaluation_hours"] != self.evaluation_hours
            ):
                raise ValueError("diagnostic window does not match the run profile and duration")
            if self.controller == "h3c_agent" and (
                self.working_memory_hours != 1
                or not self.causal_enabled
                or not self.coordination_enabled
                or self.thinking_policy != "occupancy_routed"
                or self.graph_mutation is not None
                or self.long_term_memory
            ):
                raise ValueError("diagnostic Agent protocol factors are frozen")

    def effective_model_provider(self) -> str | None:
        if self.controller == "deterministic_baseline":
            return None
        return self.model_provider or str(load_runtime_contract()["model"]["default_provider"])

    def evaluation_start_seconds(self, case_profile: dict[str, Any]) -> int:
        return evaluation_start_seconds(case_profile, self.diagnostic_window)

    def method_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "controller": self.controller,
            "working_memory_hours": self.working_memory_hours,
            "causal_enabled": self.causal_enabled,
            "coordination_enabled": self.coordination_enabled,
            "weather_enabled": True,
            "thinking_policy": self.thinking_policy,
            "action_assurance_required": True,
            "evaluation_hours": self.evaluation_hours,
        }
        if self.graph_mutation is not None:
            config["graph_mutation"] = self.graph_mutation
        if self.controller == "h3c_agent":
            config["working_memory_format"] = "caol"
            config["long_term_memory"] = self.long_term_memory
        if self.diagnostic_window is not None:
            config["diagnostic_window"] = self.diagnostic_window
        return config

    def identity_payload(self, case_profile: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_profile": case_profile,
            "method": self.method_config(),
            "evaluation_start_seconds": self.evaluation_start_seconds(case_profile),
        }
        provider = self.effective_model_provider()
        if provider is not None:
            payload["model_provider"] = provider
        return payload

    def identity(self, case_profile: dict[str, Any]) -> str:
        payload = json.dumps(
            self.identity_payload(case_profile),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def expected_agent_calls(self, zone_count: int) -> int:
        if self.controller == "deterministic_baseline":
            return 0
        roles_per_hour = zone_count + 1 + int(self.coordination_enabled)
        return self.evaluation_hours * roles_per_hour


def _agent(
    profile: str,
    *,
    memory: int = 1,
    causal: bool = True,
    coordination: bool = True,
    thinking: str = "occupancy_routed",
    mutation: dict[str, Any] | None = None,
    evaluation_hours: int = 168,
    long_term_memory: bool = False,
) -> RunPlan:
    return RunPlan(
        profile=profile,
        controller="h3c_agent",
        working_memory_hours=memory,
        causal_enabled=causal,
        coordination_enabled=coordination,
        thinking_policy=thinking,
        graph_mutation=mutation,
        evaluation_hours=evaluation_hours,
        long_term_memory=long_term_memory,
    )


def _baseline(profile: str, evaluation_hours: int = 168) -> RunPlan:
    return RunPlan(
        profile=profile,
        controller="deterministic_baseline",
        working_memory_hours=1,
        causal_enabled=False,
        coordination_enabled=False,
        thinking_policy="all_roles_disabled",
        graph_mutation=None,
        evaluation_hours=evaluation_hours,
    )


def graph_mutation(name: str) -> dict[str, Any]:
    try:
        return load_graph_mutation_catalog()[name]
    except KeyError as error:
        raise ValueError(f"unknown graph mutation: {name}") from error


def plan_suite(name: str) -> list[RunPlan]:
    suite_contract = load_suite_contract()
    case_names = tuple(suite_contract["profile_order"])
    resolved_profiles = profiles()
    if set(case_names) != set(resolved_profiles):
        raise ValueError("suite profile order does not cover exactly the configured profiles")
    evaluation_hours = {
        case: int(resolved_profiles[case]["protocol"]["formal_evaluation_days"]) * 24
        for case in case_names
    }
    if name == "main":
        return [
            *(_baseline(case, evaluation_hours[case]) for case in case_names),
            *(_agent(case, evaluation_hours=evaluation_hours[case]) for case in case_names),
        ]
    if name == "memory":
        return [
            _agent(case, memory=hours, evaluation_hours=evaluation_hours[case])
            for case in case_names
            for hours in (1, 2, 3)
        ]
    if name == "causal-ablation":
        return [
            _agent(case, causal=False, evaluation_hours=evaluation_hours[case])
            for case in case_names
        ]
    if name == "graph-sensitivity":
        timing_mutations = suite_contract["graph_timing_mutation_by_profile"]
        return [
            _agent(
                case,
                mutation=graph_mutation(kind),
                evaluation_hours=evaluation_hours[case],
            )
            for case in case_names
            for kind in ("missing_solar_zone_edge", str(timing_mutations[case]))
        ]
    if name == "coordination-ablation":
        return [
            _agent(case, coordination=False, evaluation_hours=evaluation_hours[case])
            for case in case_names
        ]
    if name == "thinking-ablation":
        return [
            _agent(
                case,
                thinking="all_roles_disabled",
                evaluation_hours=evaluation_hours[case],
            )
            for case in case_names
        ]
    if name == "all":
        resolved_profiles = profiles()
        unique: dict[str, RunPlan] = {}
        for suite in SUITES:
            for plan in plan_suite(suite):
                unique.setdefault(plan.identity(resolved_profiles[plan.profile]), plan)
        return list(unique.values())
    raise ValueError(f"unknown suite: {name}")
