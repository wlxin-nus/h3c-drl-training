"""Offline and live checks for the published DRL training protocol."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROTOCOL_VERSION, TASKS, get_task, validate_task
from .io import atomic_json, utc_now
from .observation_contract import refined_model_entry
from .observations import PolicyObservationBuilder
from .occupancy import effective_count
from .profiles import load_profile
from .protocol import control_input

REFERENCE_BOPTEST_VERSION = "0.8.0-dev"
SUPPORTED_BOPTEST_VERSIONS = (REFERENCE_BOPTEST_VERSION, "1.0.0-dev")


def validate_boptest_version(version: str) -> str:
    if version not in SUPPORTED_BOPTEST_VERSIONS:
        raise RuntimeError(
            f"unsupported BOPTEST version {version!r}; expected one of {SUPPORTED_BOPTEST_VERSIONS}"
        )
    if version != REFERENCE_BOPTEST_VERSION:
        warnings.warn(
            f"BOPTEST {version} is supported, but the reference experiments used "
            f"{REFERENCE_BOPTEST_VERSION}. Do not aggregate mixed-version runs without "
            "an explicit compatibility analysis.",
            RuntimeWarning,
            stacklevel=2,
        )
    return version


def boptest_version(endpoint: str) -> str:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/version", timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    version = str(payload.get("payload", {}).get("version", ""))
    return validate_boptest_version(version)


def _state(profile: dict[str, Any]) -> dict[str, float]:
    result = {"time": float(profile["evaluation_start_day"] * 86_400)}
    for index, zone in enumerate(profile["zones"].values()):
        result[str(zone["temperature_sensor"])] = 297.15 + index * 0.1
    for meter in profile["global_inputs"]["power_meters"]:
        result[str(meter)] = 100.0
    return result


def _forecast(profile: dict[str, Any], length: int = 101) -> dict[str, list[float]]:
    global_inputs = profile["global_inputs"]
    result = {
        str(global_inputs["outdoor_temperature"]): [293.15 + i * 0.05 for i in range(length)],
        str(global_inputs["solar_irradiance"]): [100.0 + i for i in range(length)],
        str(global_inputs["electricity_price"]): [0.1 + i * 0.0001 for i in range(length)],
    }
    for index, zone in enumerate(profile["zones"].values()):
        result[str(zone["occupancy_forecast"])] = [float(index + 1)] * length
    return result


def _task_contract(task: str, seed: int) -> dict[str, Any]:
    spec = get_task(task)
    validate_task(spec, seed)
    profile = load_profile(spec.case_key)
    entry = refined_model_entry(task)
    if profile["testcase"] != spec.case_name:
        raise RuntimeError(f"{task}: testcase differs between task and case profile")
    if profile["evaluation_start_day"] != spec.test_day:
        raise RuntimeError(f"{task}: held-out evaluation day differs from the case profile")
    if profile["protocol"]["formal_evaluation_days"] * 96 != spec.episode_steps:
        raise RuntimeError(f"{task}: episode length differs from the case profile")
    if int(entry["observation_dimension"]) != spec.observation_dim:
        raise RuntimeError(f"{task}: global observation dimension is inconsistent")
    if int(entry["action_dimension"]) != spec.action_dim:
        raise RuntimeError(f"{task}: action dimension is inconsistent")
    if spec.algorithm == "mappo" and int(entry["local_observation_dimension"]) != 36:
        raise RuntimeError(f"{task}: local MAPPO observation dimension is inconsistent")

    builder = PolicyObservationBuilder(profile, entry)
    builder.reset(_state(profile))
    packet = builder.build(
        _forecast(profile),
        step=0,
        action_time_seconds=spec.start_day * 86_400,
        step_seconds=900,
    )
    if packet.normalized.shape != (spec.observation_dim,):
        raise RuntimeError(f"{task}: observation builder returned the wrong shape")
    if np.any(~np.isfinite(packet.normalized)):
        raise RuntimeError(f"{task}: observation builder returned non-finite values")
    if spec.local_observation_dim is not None and any(
        vector.shape != (spec.local_observation_dim,) for vector in packet.local_normalized.values()
    ):
        raise RuntimeError(f"{task}: local observation builder returned the wrong shape")

    setpoints = dict.fromkeys(entry["policy_zone_order"], 25.0)
    payload = control_input(profile, setpoints)
    expected = set(profile["static_controls"]) | {
        profile["zones"][zone]["cooling_setpoint_actuator"] for zone in profile["zones"]
    }
    if set(payload) != expected:
        raise RuntimeError(f"{task}: physical payload contains unexpected control points")
    return {
        "task": task,
        "scientific_hash": spec.scientific_hash(seed),
        "case": spec.case_key,
        "testcase": spec.case_name,
        "algorithm": spec.algorithm,
        "global_observation_dimension": spec.observation_dim,
        "local_observation_dimension": spec.local_observation_dim,
        "action_dimension": spec.action_dim,
        "observation_columns": list(packet.columns),
        "control_points": sorted(payload),
        "passed": True,
    }


def _package_versions() -> dict[str, str]:
    names = (
        "gymnasium",
        "numpy",
        "pandas",
        "pythermalcomfort",
        "requests",
        "stable-baselines3",
        "torch",
        "tensorboard",
        "wandb",
        "portalocker",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def run_preflight(
    *,
    task: str | None = None,
    seed: int = 1337,
    endpoint: str = "http://127.0.0.1:8000",
    online: bool = False,
    output: Path | None = None,
) -> dict[str, Any]:
    selected = [task] if task else list(TASKS)
    contracts = {name: _task_contract(name, seed) for name in selected}

    # MZ-Air uses the official HVAC window in addition to the forecast count.
    mz_air = load_profile("mz_air")
    day = int(mz_air["evaluation_start_day"]) * 86_400
    if effective_count(mz_air["occupancy"], day + 5 * 3600, 1.0) != 0.0:
        raise RuntimeError("MZ-Air occupancy must be zero outside the official HVAC window")
    if effective_count(mz_air["occupancy"], day + 7 * 3600, 1.0) != 1.0:
        raise RuntimeError("MZ-Air occupancy must retain counts inside the official HVAC window")

    version = boptest_version(endpoint) if online else "not-checked"
    report = {
        "schema": "h3c-drl-preflight-v1",
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "seed": seed,
        "online": online,
        "endpoint": endpoint,
        "boptest_version": version,
        "reference_boptest_version": REFERENCE_BOPTEST_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": _package_versions(),
        "tasks": contracts,
        "passed": True,
    }
    if output is not None:
        atomic_json(Path(output), report)
    return report
