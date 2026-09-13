from __future__ import annotations

import hashlib
import json
import platform
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from h3c.experiments.profiles import load_profile
from h3c.runtime.protocol import control_input
from h3c_baselines.controllers.drl import FrozenDrlController
from h3c_baselines.models import load_registry, model_entry, verify_all_checkpoints
from h3c_baselines.policies.observation_contracts import PolicyObservationBuilder

from .config import (
    EARLY_STOP_PROTOCOL_ID,
    OBSERVATION_CONTRACT_ID,
    TASKS,
    WANDB_PROJECT,
    TaskSpec,
    get_task,
    repository_root,
    validate_task,
)
from .io import atomic_json, sha256_file, utc_now
from .observation_contract import contract_path, refined_model_entry


H3C_SOURCE_COMMIT = "23186b2c499e02c042018f71f22ee61b5510b910"
H3C_REGISTRY_CANONICAL_SHA256 = "70bffd441fa71e25f818dcda14263bc5826e44d7e3aef7ee2f194c1f75ba3bca"
REFINED_CONTRACT_CANONICAL_SHA256 = "dd6008927c5b2e497e05cca1196b5f2730c093f7aa4a20d48d61fe2a4a6ce876"
REFERENCE_BOPTEST_VERSION = "0.8.0-dev"
SUPPORTED_BOPTEST_VERSIONS = (REFERENCE_BOPTEST_VERSION, "1.0.0-dev")
CASE_DISPLAY = {"sz_air": "SZ_Air", "mz_air": "MZ_Air", "mz_hydro": "MZ_Hydro"}
FROZEN_TRAINING_PROTOCOLS: dict[str, dict[str, Any]] = {
    "sz_air_ppo": dict(start_day=196, test_day=203, episode_steps=672, n_steps=672,
        batch_size=168, n_epochs=10, max_epochs=700, learning_rate=5e-4,
        lr_decay_epochs=300,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
        vf_coef=0.5, max_grad_norm=0.5, comfort_threshold=0.5,
        action_contract="fixed_25_residual_5"),
    "mz_air_ppo": dict(start_day=192, test_day=199, episode_steps=672, n_steps=672,
        batch_size=168, n_epochs=10, max_epochs=700, learning_rate=3e-4,
        lr_decay_epochs=300,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
        vf_coef=0.5, max_grad_norm=0.5, comfort_threshold=0.5,
        action_contract="fixed_25_residual_5"),
    "mz_air_mappo": dict(start_day=192, test_day=199, episode_steps=672, n_steps=672,
        batch_size=168, n_epochs=10, max_epochs=700, learning_rate=3e-4,
        lr_decay_epochs=300,
        critic_learning_rate=5e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.02, vf_coef=0.5, max_grad_norm=0.5, comfort_threshold=0.5,
        action_contract="fixed_25_residual_5"),
    "mz_hydro_ppo": dict(start_day=213, test_day=220, episode_steps=480, n_steps=480,
        batch_size=120, n_epochs=10, max_epochs=700, learning_rate=3e-4,
        lr_decay_epochs=500,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
        vf_coef=0.5, max_grad_norm=0.5, comfort_threshold=0.5,
        action_contract="occupied_25_unoccupied_30_residual_5"),
    "mz_hydro_mappo": dict(start_day=213, test_day=220, episode_steps=480, n_steps=480,
        batch_size=120, n_epochs=10, max_epochs=700, learning_rate=3e-4,
        lr_decay_epochs=500,
        critic_learning_rate=5e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.02, vf_coef=0.5, max_grad_norm=0.5, comfort_threshold=0.5,
        action_contract="occupied_25_unoccupied_30_residual_5"),
}


def _identity(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json_sha256(path: Path) -> str:
    """Hash JSON content independently of checkout line endings and formatting."""
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state(profile: Mapping[str, Any]) -> dict[str, float]:
    state = {"time": float(int(profile["evaluation_start_day"]) * 86400)}
    for index, zone in enumerate(profile["zones"].values()):
        state[str(zone["temperature_sensor"])] = 297.15 + index * 0.1
    for point in profile["global_inputs"]["power_meters"]:
        state[str(point)] = 100.0
    return state


def _forecast(profile: Mapping[str, Any], length: int = 101) -> dict[str, list[float]]:
    global_inputs = profile["global_inputs"]
    values = {
        str(global_inputs["outdoor_temperature"]): [293.15 + i * 0.05 for i in range(length)],
        str(global_inputs["solar_irradiance"]): [100.0 + i for i in range(length)],
        str(global_inputs["electricity_price"]): [0.1 + i * 0.0001 for i in range(length)],
    }
    for index, zone in enumerate(profile["zones"].values()):
        values[str(zone["occupancy_forecast"])] = [float(index + 1)] * length
    return values


def _golden_inference() -> dict[str, Any]:
    source = repository_root() / "configs" / "baseline_policy_inference_golden.json"
    golden = json.loads(source.read_text(encoding="utf-8"))["fixtures"]
    checks: dict[str, Any] = {}
    for key, fixture in golden.items():
        profile = load_profile(fixture["case"])
        entry = model_entry(fixture["case"], fixture["controller"])
        builder = PolicyObservationBuilder(profile, entry)
        builder.reset(_state(profile))
        packet = builder.build(
            _forecast(profile), step=0,
            action_time_seconds=int(profile["evaluation_start_day"]) * 86400,
            step_seconds=900,
        )
        occupancy = dict.fromkeys(profile["zones"], 1.0)
        setpoints, diagnostics = FrozenDrlController(
            fixture["case"], fixture["controller"]
        ).decide(packet, occupancy)
        actual = {
            "model_sha256": entry["sha256"],
            "columns_sha256": _identity(list(packet.columns)),
            "raw_sha256": _identity(packet.raw.tolist()),
            "normalized_sha256": _identity(packet.normalized.tolist()),
            "local_normalized_sha256": {
                zone: _identity(vector.tolist())
                for zone, vector in packet.local_normalized.items()
            },
            "raw_action": diagnostics["raw_action"],
            "setpoints_c": setpoints,
            "boptest_payload": control_input(profile, setpoints),
        }
        exact_fields = ("model_sha256", "columns_sha256", "raw_sha256", "normalized_sha256", "local_normalized_sha256")
        exact_passed = all(actual[name] == fixture[name] for name in exact_fields)
        action_error = float(np.max(np.abs(np.asarray(actual["raw_action"]) - np.asarray(fixture["raw_action"]))))
        setpoint_error = max(abs(float(actual["setpoints_c"][name]) - float(fixture["setpoints_c"][name])) for name in fixture["setpoints_c"])
        payload_error = max(abs(float(actual["boptest_payload"][name]) - float(fixture["boptest_payload"][name])) for name in fixture["boptest_payload"])
        numeric_keys_passed = (
            set(actual["setpoints_c"]) == set(fixture["setpoints_c"])
            and set(actual["boptest_payload"]) == set(fixture["boptest_payload"])
        )
        passed = exact_passed and numeric_keys_passed and max(action_error, setpoint_error, payload_error) <= 2e-6
        checks[key] = {"passed": passed, "actual": actual, "maximum_numeric_error": max(action_error, setpoint_error, payload_error), "numeric_tolerance": 2e-6}
        if not passed:
            mismatches = [name for name in exact_fields if actual[name] != fixture[name]]
            if not numeric_keys_passed or max(action_error, setpoint_error, payload_error) > 2e-6:
                mismatches.append("action/setpoint/payload")
            raise RuntimeError(f"H3C golden policy contract changed: {key}: {mismatches}")
    return checks


def _task_contract(spec: TaskSpec) -> dict[str, Any]:
    case = CASE_DISPLAY[spec.case_key]
    controller = "c-drl" if spec.algorithm == "ppo" else "h-drl"
    entry = refined_model_entry(spec.key)
    legacy_entry = model_entry(case, controller)
    expected_residual = (
        "occupancy_25_30"
        if spec.action_contract == "occupied_25_unoccupied_30_residual_5"
        else "fixed_25"
    )
    assertions = {
        "case": entry["case"] == case,
        "algorithm": entry["algorithm"] == spec.algorithm,
        "observation_dimension": int(entry["observation_dimension"]) == spec.observation_dim,
        "action_dimension": int(entry["action_dimension"]) == spec.action_dim,
        "forecast_points": spec.forecast_steps == 5,
        "residual_base": entry["residual_base"] == expected_residual,
        "temperature_offsets": entry["temperature_past_offset"] == 1,
        "action_offsets": entry["action_past_offset"] == 0,
        "power_offsets": entry["power_past_offset"] == 0,
        "temperature_missing": entry["temperature_missing"] == "repeat_earliest",
        "four_actions_per_zone": entry["action_history_steps"] == 4,
        "time_bounds": entry["time_observation_bounds"] == [-1.0, 1.0],
        "action_history_bounds": entry["action_observation_bounds_k"] == [293.15, 303.15],
        "effective_occupancy_count": entry["occupancy_encoding"] == "effective_count",
        "n_envs": spec.num_envs == 4,
        "full_episode_per_env": spec.n_steps == spec.episode_steps,
        "batch_divisibility": spec.steps_per_epoch % spec.batch_size == 0,
        "network": tuple(spec.policy_net) == (256, 256),
        "frozen_training_protocol": all(
            getattr(spec, name) == expected
            for name, expected in FROZEN_TRAINING_PROTOCOLS[spec.key].items()
        ),
    }
    if spec.algorithm == "mappo":
        assertions["local_observation_dimension"] = (
            int(entry["local_observation_dimension"]) == spec.local_observation_dim
        )
        assertions["canonical_local_layout"] = (
            entry["local_observation_layout"] == "zone_then_shared"
        )
        assertions["actor_input_dimension"] = spec.local_observation_dim == 36
        assertions["critic_input_dimension"] = (
            spec.observation_dim == int(entry["observation_dimension"])
        )
    if not all(assertions.values()):
        failed = [name for name, passed in assertions.items() if not passed]
        raise RuntimeError(f"{spec.key} violates refined H3C contract: {failed}")
    builder = PolicyObservationBuilder(load_profile(case), entry)
    profile = load_profile(case)
    builder.reset(_state(profile))
    packet = builder.build(
        _forecast(profile), step=0,
        action_time_seconds=int(profile["evaluation_start_day"]) * 86400,
        step_seconds=900,
    )
    if packet.raw.shape != (spec.observation_dim,):
        raise RuntimeError(f"{spec.key} refined global observation shape is invalid")
    if spec.local_observation_dim is not None and any(
        value.shape != (spec.local_observation_dim,)
        for value in packet.local_normalized.values()
    ):
        raise RuntimeError(f"{spec.key} refined local observation shape is invalid")
    return {
        "case": case,
        "controller": controller,
        "contract_id": entry["contract_id"],
        "legacy_registry_model_sha256": legacy_entry["sha256"],
        "legacy_observation_dimension": legacy_entry["observation_dimension"],
        "refined_observation_dimension": entry["observation_dimension"],
        "policy_zone_order": entry["policy_zone_order"],
        "occupancy_encoding": entry["occupancy_encoding"],
        "local_observation_layout": entry.get("local_observation_layout"),
        "assertions": assertions,
    }


def _validate_boptest_version(version: str) -> str:
    if not version:
        raise RuntimeError("BOPTEST /version returned an empty version")
    if version not in SUPPORTED_BOPTEST_VERSIONS:
        raise RuntimeError(
            f"Unsupported BOPTEST version {version!r}; supported versions are "
            f"{SUPPORTED_BOPTEST_VERSIONS}"
        )
    if version != REFERENCE_BOPTEST_VERSION:
        warnings.warn(
            f"BOPTEST {version} is accepted, but the reference experiments used "
            f"{REFERENCE_BOPTEST_VERSION}. The actual version will be recorded in "
            "the run metadata.",
            RuntimeWarning,
            stacklevel=2,
        )
    return version


def _boptest_version(endpoint: str) -> str:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/version", timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    version = str(payload.get("payload", {}).get("version", ""))
    return _validate_boptest_version(version)


def run_preflight(
    *, task: str | None = None, seed: int = 1337, endpoint: str = "http://127.0.0.1:8000",
    online: bool = False, output: Path | None = None,
) -> dict[str, Any]:
    selected = [get_task(task)] if task else list(TASKS.values())
    for spec in selected:
        validate_task(spec, seed)
    registry_path = repository_root() / "models" / "registry.json"
    registry_sha = sha256_file(registry_path)
    registry_canonical_sha = _canonical_json_sha256(registry_path)
    if registry_canonical_sha != H3C_REGISTRY_CANONICAL_SHA256:
        raise RuntimeError(
            "Frozen H3C registry content changed: "
            f"expected canonical SHA256 {H3C_REGISTRY_CANONICAL_SHA256}, "
            f"got {registry_canonical_sha}"
        )
    refined_path = contract_path()
    refined_canonical_sha = _canonical_json_sha256(refined_path)
    if refined_canonical_sha != REFINED_CONTRACT_CANONICAL_SHA256:
        raise RuntimeError(
            "Refined observation contract content changed: "
            f"expected canonical SHA256 {REFINED_CONTRACT_CANONICAL_SHA256}, "
            f"got {refined_canonical_sha}"
        )
    if json.loads(refined_path.read_text(encoding="utf-8"))["contract_id"] != OBSERVATION_CONTRACT_ID:
        raise RuntimeError("Refined observation contract ID does not match runtime configuration")
    boptest_version = _boptest_version(endpoint) if online else "not_checked"
    report = {
        "schema": "h3c-drl-multiseed-refined-preflight-v1",
        "generated_at": utc_now(),
        "h3c_source_commit": H3C_SOURCE_COMMIT,
        "h3c_registry_sha256": registry_sha,
        "h3c_registry_canonical_sha256": registry_canonical_sha,
        "refined_contract_canonical_sha256": refined_canonical_sha,
        "refined_contract_id": OBSERVATION_CONTRACT_ID,
        "wandb_project": WANDB_PROJECT,
        "uniform_training_settings": {
            "comfort_threshold": 0.5,
            "ent_coef": 0.02,
            "max_epochs": 700,
            "early_stop_min_epoch": 100,
            "early_stop_eval_interval": 25,
            "early_stop_min_delta_fraction": 0.01,
            "early_stop_patience": 3,
            "early_stop_protocol": EARLY_STOP_PROTOCOL_ID,
        },
        "legacy_models": verify_all_checkpoints(load_cpu=True),
        "golden_inference": _golden_inference(),
        "tasks": {spec.key: _task_contract(spec) for spec in selected},
        "worker_contract": {
            "capacity": 12, "train_ids_per_task": 4, "validation_ids_per_task": 1,
            "max_parallel": 2, "peak_ids": 10, "spare_workers": 2,
        },
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "online": bool(online),
        "boptest_version": boptest_version,
        "boptest_reference_version": REFERENCE_BOPTEST_VERSION,
        "boptest_version_matches_reference": (
            boptest_version == REFERENCE_BOPTEST_VERSION if online else None
        ),
        "passed": True,
    }
    if output is not None:
        atomic_json(Path(output), report)
    return report
