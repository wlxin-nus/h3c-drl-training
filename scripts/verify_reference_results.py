"""Verify integrity and statistical closure of the processed paper-result package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from scripts.recompute_paper_metrics import RECOMPUTED_METRICS, compare_expected, recompute

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "reference_results" / "paper_2026"
MANIFEST = RESULT_ROOT / "manifest.json"
HISTORICAL_SOURCE_ROOT = ROOT / "historical_training_source_287b452"
HISTORICAL_CONTRACT = HISTORICAL_SOURCE_ROOT / "contract.json"
EXPECTED_GROUPS = {
    ("SZ_Air", "PPO"),
    ("MZ_Hydro", "PPO"),
    ("MZ_Hydro", "MAPPO"),
    ("MZ_Air", "PPO"),
    ("MZ_Air", "MAPPO"),
}
PHYSICAL_METRICS = (
    "reward",
    "total_cost",
    "energy_kwh",
    "discomfort_zone_hours",
    "discomfort_pmv_hours",
    "occupied_peak_absolute_pmv",
    "total_variation_c",
    "direction_reversals",
    "occupied_comfort_band_crossings",
)
PRIVATE_COLUMNS = {
    "test_id",
    "run_identity",
    "audit_status",
    "method_classification",
    "model_sha256",
    "run_dir",
}
HISTORICAL_TRAINING_FINGERPRINT = "aca13c42236511fc7e428ce0c9f2969c39d53779cc2d5c9880dcaf8e2cf46272"
RELEASE_SOURCE_FINGERPRINT = "e6cde1aeffba4ac6dbc2dfb2c371ef61b556dc31ace513ce17b9870868d68efe"
PYTHERMALCOMFORT_VERSIONS = {
    "historical_training": "3.8.0",
    "canonical_held_out_evaluation": "3.9.8",
    "verified_equivalence_scope": (
        "pmv_ppd_iso with ISO/SI units, limit_inputs=False, and default rounding"
    ),
}
CANONICAL_EVALUATION_RUNTIME_VERSIONS = {
    "gymnasium": "1.2.1",
    "numpy": "2.2.6",
    "stable-baselines3": "2.7.0",
    "torch": "2.9.0",
}
HISTORICAL_SOURCE_COMMIT = "287b452c2874a36cf2777bd31696192b3622d706"
HISTORICAL_FINGERPRINT_FILES = (
    "src/drl_multiseed/config.py",
    "src/drl_multiseed/environment.py",
    "src/drl_multiseed/observation_contract.py",
    "src/drl_multiseed/networks.py",
    "src/drl_multiseed/gae.py",
    "src/drl_multiseed/early_stop.py",
    "src/drl_multiseed/continuation.py",
    "src/drl_multiseed/evaluation.py",
    "src/drl_multiseed/ppo_training.py",
    "src/drl_multiseed/mappo_training.py",
    "upstream/h3c/experiments/profiles.py",
    "upstream/h3c_baselines/policies/observation_contracts.py",
    "upstream/h3c_baselines/policies/ppo_adapter.py",
    "upstream/h3c_baselines/policies/mappo_adapter.py",
    "models/registry.json",
    "configs/refined_observation_contracts.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(name: str) -> tuple[list[str], list[dict[str, str]]]:
    path = RESULT_ROOT / name
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8)


def _historical_source_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in HISTORICAL_FINGERPRINT_FILES:
        path = HISTORICAL_SOURCE_ROOT / name
        digest.update(name.encode("utf-8"))
        content = path.read_text(encoding="utf-8")
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        digest.update(normalized.encode("utf-8"))
    return digest.hexdigest()


def _historical_configuration_hashes() -> dict[str, dict[str, str]]:
    command = (
        "from drl_multiseed.config import TASKS; import json; "
        "print(json.dumps({k:{str(s):TASKS[k].scientific_hash(s) "
        "for s in (42,1337,2026)} for k in sorted(TASKS)},sort_keys=True))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(HISTORICAL_SOURCE_ROOT / "src"), str(HISTORICAL_SOURCE_ROOT / "upstream")]
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=HISTORICAL_SOURCE_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("historical configuration hashes could not be recomputed")
    return cast(dict[str, dict[str, str]], json.loads(result.stdout))


def _verify_historical_preflight_models(historical: dict[str, Any]) -> int:
    registry_path = HISTORICAL_SOURCE_ROOT / "models" / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))["models"]
    expected = historical.get("legacy_preflight_models_included")
    if not isinstance(expected, list) or set(expected) != set(registry):
        raise RuntimeError("the historical preflight-model inventory differs from the registry")
    for model_id in expected:
        entry = registry[model_id]
        path = HISTORICAL_SOURCE_ROOT / entry["path"]
        if not path.is_file():
            raise RuntimeError(f"missing historical preflight model: {model_id}")
        if path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"historical preflight-model byte count differs: {model_id}")
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"historical preflight-model SHA-256 differs: {model_id}")
    return len(expected)


def verify() -> dict[str, int | float]:
    payload: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    historical: dict[str, Any] = json.loads(HISTORICAL_CONTRACT.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("historical_training_source_fingerprint") != HISTORICAL_TRAINING_FINGERPRINT:
        failures.append("the historical training source fingerprint differs from the manifest")
    if payload.get("release_source_fingerprint") != RELEASE_SOURCE_FINGERPRINT:
        failures.append("the publication source fingerprint differs from the manifest")
    if payload.get("historical_training_configurations_match_release") is not False:
        failures.append("the historical/release configuration boundary is not recorded")
    if payload.get("historical_training_source_archive") != HISTORICAL_SOURCE_ROOT.name:
        failures.append("the historical source archive is not recorded")
    if payload.get("historical_training_configurations_match_archive") is not True:
        failures.append("the historical/archive configuration match is not recorded")
    if payload.get("legacy_preflight_models_included") is not True:
        failures.append("the historical preflight-model inclusion is not recorded")
    if payload.get("evaluated_checkpoint_replay_supported") is not False:
        failures.append("the evaluated-checkpoint replay boundary is not recorded")
    if "zone air temperature" not in str(payload.get("pmv_temperature_contract", "")).lower():
        failures.append("the PMV temperature contract is not recorded")
    if payload.get("pythermalcomfort_versions") != PYTHERMALCOMFORT_VERSIONS:
        failures.append("the training/evaluation PMV versions differ from the manifest")
    if (
        payload.get("canonical_evaluation_runtime_versions")
        != CANONICAL_EVALUATION_RUNTIME_VERSIONS
    ):
        failures.append("the formal-evaluation runtime versions differ from the manifest")
    if historical.get("source_commit") != HISTORICAL_SOURCE_COMMIT:
        failures.append("the historical source commit differs from the contract")
    if historical.get("source_fingerprint") != HISTORICAL_TRAINING_FINGERPRINT:
        failures.append("the historical source fingerprint differs from the contract")
    if historical.get("checkpoint_suite_included") is not False:
        failures.append("the evaluated-checkpoint boundary differs from the historical contract")
    if _historical_source_fingerprint() != HISTORICAL_TRAINING_FINGERPRINT:
        failures.append("the exported historical source fingerprint differs")
    try:
        historical_hashes = _historical_configuration_hashes()
    except RuntimeError as error:
        failures.append(str(error))
    else:
        if historical_hashes != historical.get("configuration_hashes"):
            failures.append("the 15 historical task/seed configuration hashes differ")
    historical_preflight_models = 0
    try:
        historical_preflight_models = _verify_historical_preflight_models(historical)
    except RuntimeError as error:
        failures.append(str(error))
    for item in payload["files"]:
        path = RESULT_ROOT / item["path"]
        if not path.is_file():
            failures.append(f"missing file: {item['path']}")
            continue
        if path.stat().st_size != int(item["bytes"]):
            failures.append(f"byte count differs: {item['path']}")
        if _sha256(path) != item["sha256"]:
            failures.append(f"SHA-256 differs: {item['path']}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = sum(1 for _ in csv.reader(stream)) - 1
        if rows != int(item["data_rows"]):
            failures.append(f"row count differs: {item['path']}")

    run_fields, runs = _read_csv("drl_run_metrics.csv")
    leaked = PRIVATE_COLUMNS.intersection(run_fields)
    if leaked:
        failures.append("private run fields are present: " + ", ".join(sorted(leaked)))
    seeds_by_group: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in runs:
        seeds_by_group[(row["case"], row["algorithm"])].add(int(row["seed"]))
    if set(seeds_by_group) != EXPECTED_GROUPS or any(
        seeds != {42, 1337, 2026} for seeds in seeds_by_group.values()
    ):
        failures.append("the five task groups do not each contain seeds 42, 1337, and 2026")
    if len({row["model_id"] for row in runs}) != 15:
        failures.append("the run table does not contain 15 unique model IDs")
    source_counts = Counter(row["source_commit"] for row in runs)
    expected_sources = Counter(
        {item["commit"]: int(item["models"]) for item in payload["evaluation_sources"]}
    )
    if source_counts != expected_sources:
        failures.append("the evaluation source-commit counts differ from the manifest")

    _, summaries = _read_csv("drl_metric_summary.csv")
    summary_index = {(row["case"], row["algorithm"], row["metric"]): row for row in summaries}
    expected_summary_keys = {
        (case, algorithm, metric)
        for case, algorithm in EXPECTED_GROUPS
        for metric in PHYSICAL_METRICS
    }
    if set(summary_index) != expected_summary_keys:
        failures.append("the metric summary does not contain five groups by nine metrics")
    runs_by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        runs_by_group[(row["case"], row["algorithm"])].append(row)
    for (case, algorithm), group_rows in runs_by_group.items():
        for metric in PHYSICAL_METRICS:
            summary = summary_index.get((case, algorithm, metric))
            if summary is None:
                continue
            values = [float(row[metric]) for row in group_rows]
            if (
                int(summary["n"]) != 3
                or not _close(float(summary["mean"]), statistics.mean(values))
                or not _close(float(summary["sample_sd"]), statistics.stdev(values))
            ):
                failures.append(f"summary statistics differ: {case}/{algorithm}/{metric}")
            for seed in (42, 1337, 2026):
                target = next(float(row[metric]) for row in group_rows if int(row["seed"]) == seed)
                if not _close(float(summary[f"seed{seed}"]), target):
                    failures.append(f"seed value differs: {case}/{algorithm}/{metric}/seed{seed}")

    maximum_error = 0.0
    try:
        calculated = recompute(RESULT_ROOT / "drl_evaluation_timeseries.csv")
        maximum_error = compare_expected(calculated, RESULT_ROOT / "drl_run_metrics.csv")
    except (RuntimeError, ValueError) as error:
        failures.append(str(error))
    if failures:
        raise RuntimeError("reference-result verification failed:\n- " + "\n- ".join(failures))
    return {
        "files": len(payload["files"]),
        "models": len(runs),
        "historical_configurations": sum(
            len(seeds) for seeds in historical["configuration_hashes"].values()
        ),
        "historical_preflight_models": historical_preflight_models,
        "recomputed_metrics": len(RECOMPUTED_METRICS),
        "maximum_absolute_difference": maximum_error,
    }


def main() -> None:
    result = verify()
    print(
        f"Verified {result['files']} reference files and {result['models']} DRL models; "
        f"{result['historical_configurations']} historical configurations and "
        f"{result['historical_preflight_models']} historical preflight models; "
        f"{result['recomputed_metrics']} metrics close with maximum absolute difference "
        f"{result['maximum_absolute_difference']:.3g}."
    )


if __name__ == "__main__":
    main()
