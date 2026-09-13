from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, cast


CONTRACT_SCHEMA = "h3c-drl-refined-observation-contract"
CONTRACT_VERSION = 1


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def contract_path() -> Path:
    return _repository_root() / "configs" / "refined_observation_contracts.json"


@lru_cache(maxsize=1)
def load_refined_contracts() -> dict[str, Any]:
    value = cast(
        dict[str, Any],
        json.loads(contract_path().read_text(encoding="utf-8")),
    )
    if value.get("schema") != CONTRACT_SCHEMA or value.get("schema_version") != CONTRACT_VERSION:
        raise ValueError("refined observation contract schema is invalid")
    temporal = value.get("temporal_contract")
    if not isinstance(temporal, Mapping):
        raise ValueError("refined temporal contract is missing")
    expected = {
        "control_step_minutes": 15,
        "temperature_offsets": [0, -1, -2, -3, -4],
        "temperature_missing": "repeat_earliest",
        "action_offsets": [-1, -2, -3, -4],
        "action_missing": "fixed_initial_setpoint",
        "power_offsets": [-1, -2, -3, -4],
        "power_missing": "zero",
        "forecast_offsets": [0, 1, 2, 3, 4],
        "forecast_offsets_minutes": [0, 15, 30, 45, 60],
        "mappo_local_layout": "zone_then_shared",
    }
    for key, required in expected.items():
        if temporal.get(key) != required:
            raise ValueError(f"refined temporal contract has invalid {key}")
    tasks = value.get("tasks")
    if not isinstance(tasks, Mapping) or len(tasks) != 5:
        raise ValueError("refined observation contract must contain exactly five tasks")
    return value


def refined_model_entry(task: str) -> dict[str, Any]:
    contracts = load_refined_contracts()
    tasks = cast(Mapping[str, Any], contracts["tasks"])
    try:
        task_entry = copy.deepcopy(dict(cast(Mapping[str, Any], tasks[task])))
    except KeyError as error:
        raise ValueError(f"no refined observation contract for {task!r}") from error
    temporal = cast(Mapping[str, Any], contracts["temporal_contract"])
    normalization = cast(Mapping[str, Any], contracts["common_normalization"])
    task_entry.update(
        {
            "contract_id": contracts["contract_id"],
            "action_history_steps": len(cast(list[int], temporal["action_offsets"])),
            "temperature_past_offset": 1,
            "action_past_offset": 0,
            "power_past_offset": 0,
            "temperature_missing": temporal["temperature_missing"],
            "local_observation_layout": temporal["mappo_local_layout"],
            "time_observation_bounds": list(normalization["time_sin_cos"]),
            "action_observation_bounds_k": list(
                normalization["physical_setpoint_history_k"]
            ),
        }
    )
    return task_entry


def refined_contract_payload(task: str) -> dict[str, Any]:
    contracts = load_refined_contracts()
    return {
        "schema": contracts["schema"],
        "schema_version": contracts["schema_version"],
        "contract_id": contracts["contract_id"],
        "temporal_contract": copy.deepcopy(contracts["temporal_contract"]),
        "common_normalization": copy.deepcopy(contracts["common_normalization"]),
        "task": refined_model_entry(task),
    }
