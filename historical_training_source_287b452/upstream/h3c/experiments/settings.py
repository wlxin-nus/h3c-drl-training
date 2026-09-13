"""Frozen runtime request contract."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from h3c.experiments.profiles import repository_root

ENVIRONMENT_VARIABLE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")


def load_runtime_contract(path: Path | None = None) -> dict[str, Any]:
    source = path or repository_root() / "configs" / "experiments" / "runtime.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "runtime_schema",
        "schema_version",
        "model",
        "physical_service",
    }:
        raise ValueError("runtime contract fields are invalid")
    if value["runtime_schema"] != "h3c_runtime_contract" or value["schema_version"] != 4:
        raise ValueError("unsupported runtime contract schema")
    model = value["model"]
    if not isinstance(model, dict) or set(model) != {
        "default_provider",
        "providers",
        "thinking_reasoning_effort",
        "no_thinking_temperature",
        "no_thinking_top_p",
        "retry_count",
        "retry_backoff_seconds",
    }:
        raise ValueError("model runtime contract fields are invalid")
    if (
        model["default_provider"] != "baseten-deepseek"
        or model["thinking_reasoning_effort"] != "low"
        or model["no_thinking_temperature"] != 0.0
        or model["no_thinking_top_p"] != 1.0
        or model["retry_count"] != 2
        or model["retry_backoff_seconds"] != [1.0, 2.0]
    ):
        raise ValueError("model request contract is not frozen")
    providers = model["providers"]
    if not isinstance(providers, dict) or set(providers) != {
        "deepseek-official",
        "baseten-deepseek",
    }:
        raise ValueError("model provider catalog is invalid")
    expected_providers = {
        "deepseek-official": {
            "model": "deepseek-v4-flash",
            "endpoint_environment_variable": "H3C_MODEL_ENDPOINT",
            "fixed_endpoint": None,
            "api_key_environment_variable": "H3C_MODEL_API_KEY",
            "session_affinity_header": None,
            "retryable_status_codes": [429, 503],
            "response_format": "json_object",
        },
        "baseten-deepseek": {
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "endpoint_environment_variable": None,
            "fixed_endpoint": "https://inference.baseten.co/v1",
            "api_key_environment_variable": "BASETEN_API_KEY",
            "session_affinity_header": "x-session-affinity",
            "retryable_status_codes": [429, 500, 502, 503, 504, 529],
            "response_format": "json_schema",
        },
    }
    if providers != expected_providers:
        raise ValueError("model provider contracts are not frozen")
    physical = value["physical_service"]
    if (
        not isinstance(physical, dict)
        or set(physical) != {"endpoint_environment_variable", "retry_count"}
        or not isinstance(physical["endpoint_environment_variable"], str)
        or ENVIRONMENT_VARIABLE.fullmatch(physical["endpoint_environment_variable"]) is None
        or physical["retry_count"] != 0
    ):
        raise ValueError("physical service contract is invalid")
    environment_names = {
        "H3C_MODEL_ENDPOINT",
        "H3C_MODEL_API_KEY",
        "BASETEN_API_KEY",
        physical["endpoint_environment_variable"],
    }
    if len(environment_names) != 4:
        raise ValueError("runtime credential/environment owners must be distinct")
    return value


def load_model_provider_contract(provider: str) -> dict[str, Any]:
    runtime = load_runtime_contract()
    providers = runtime["model"]["providers"]
    if not isinstance(provider, str) or provider not in providers:
        raise ValueError("model provider is not registered")
    return copy.deepcopy(providers[provider])


def load_diagnostic_window_catalog(path: Path | None = None) -> dict[str, dict[str, Any]]:
    source = path or repository_root() / "configs" / "experiments" / "diagnostic_windows.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "diagnostic_window_schema",
        "schema_version",
        "windows",
    }:
        raise ValueError("diagnostic window catalog fields are invalid")
    if (
        value["diagnostic_window_schema"] != "h3c_diagnostic_window_catalog"
        or value["schema_version"] != 1
    ):
        raise ValueError("unsupported diagnostic window catalog schema")
    windows = value["windows"]
    if not isinstance(windows, dict) or not windows:
        raise ValueError("registered diagnostic windows are invalid")
    for name, window in windows.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(window, dict)
            or set(window)
            != {
                "profile",
                "evaluation_start_offset_seconds",
                "evaluation_hours",
                "description",
            }
            or not isinstance(window["profile"], str)
            or not window["profile"]
            or isinstance(window["evaluation_start_offset_seconds"], bool)
            or not isinstance(window["evaluation_start_offset_seconds"], int)
            or not 0 <= window["evaluation_start_offset_seconds"] < 86400
            or window["evaluation_start_offset_seconds"] % 900 != 0
            or isinstance(window["evaluation_hours"], bool)
            or not isinstance(window["evaluation_hours"], int)
            or window["evaluation_hours"] <= 0
            or not isinstance(window["description"], str)
            or not window["description"]
        ):
            raise ValueError("diagnostic window declaration is invalid")
    return copy.deepcopy(windows)


def evaluation_start_seconds(profile: dict[str, Any], diagnostic_window: str | None = None) -> int:
    offset = 0
    if diagnostic_window is not None:
        windows = load_diagnostic_window_catalog()
        if diagnostic_window not in windows:
            raise ValueError("diagnostic window is not registered")
        window = windows[diagnostic_window]
        if window["profile"] != profile.get("profile"):
            raise ValueError("diagnostic window does not match the case profile")
        offset = int(window["evaluation_start_offset_seconds"])
    return int(profile["evaluation_start_day"]) * 86400 + offset


def load_suite_contract(path: Path | None = None) -> dict[str, Any]:
    source = path or repository_root() / "configs" / "experiments" / "suites.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "suite_schema",
        "schema_version",
        "profile_order",
        "graph_timing_mutation_by_profile",
    }:
        raise ValueError("suite contract fields are invalid")
    if value["suite_schema"] != "h3c_suite_contract" or value["schema_version"] != 1:
        raise ValueError("unsupported suite contract schema")
    order = value["profile_order"]
    if (
        not isinstance(order, list)
        or not order
        or len(order) != len(set(order))
        or any(not isinstance(name, str) or not name for name in order)
    ):
        raise ValueError("suite profile declarations are invalid")
    timing_mutations = value["graph_timing_mutation_by_profile"]
    registered_mutations = load_graph_mutation_catalog()
    if (
        not isinstance(timing_mutations, dict)
        or set(timing_mutations) != set(order)
        or any(
            not isinstance(name, str) or name not in registered_mutations
            for name in timing_mutations.values()
        )
    ):
        raise ValueError("suite graph timing mutations are invalid")
    return value


def load_graph_mutation_catalog(path: Path | None = None) -> dict[str, dict[str, Any]]:
    source = path or repository_root() / "configs" / "graphs" / "mutations.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "graph_mutation_schema",
        "schema_version",
        "mutations",
    }:
        raise ValueError("graph mutation catalog fields are invalid")
    if (
        value["graph_mutation_schema"] != "h3c_graph_mutation_catalog"
        or value["schema_version"] != 1
    ):
        raise ValueError("unsupported graph mutation catalog schema")
    mutations = value["mutations"]
    if not isinstance(mutations, dict) or set(mutations) != {
        "missing_solar_zone_edge",
        "delayed_solar_zone_edge",
        "immediate_solar_zone_edge",
    }:
        raise ValueError("registered graph mutations are invalid")
    for mutation in mutations.values():
        if (
            not isinstance(mutation, dict)
            or set(mutation) != {"kind", "edge"}
            or mutation["kind"]
            not in {
                "remove_edge",
                "change_immediate_to_delayed",
                "change_delayed_to_immediate",
            }
            or not isinstance(mutation["edge"], dict)
            or set(mutation["edge"]) != {"source", "relation", "target"}
            or any(not isinstance(field, str) or not field for field in mutation["edge"].values())
        ):
            raise ValueError("graph mutation declaration is invalid")
    return copy.deepcopy(mutations)
