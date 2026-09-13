from __future__ import annotations

import json

import pytest

from drl_multiseed.aggregate import aggregate_results
from drl_multiseed.config import PROTOCOL_VERSION, get_task


def _write_result(root, seed: int, boptest_version: str) -> None:
    task = "mz_hydro_ppo"
    spec = get_task(task)
    run_dir = root / "full" / f"seed{seed}" / task
    evaluation = run_dir / "formal_evaluation"
    evaluation.mkdir(parents=True)
    identity = {
        "task": task,
        "seed": seed,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": spec.protocol_hash(),
        "config_hash": spec.scientific_hash(seed),
        "code_fingerprint": "same-source",
        "boptest_version": boptest_version,
    }
    manifest = {
        **identity,
        "best_epoch": 200,
        "status": "early_stopped",
        "global_step": 384_000,
    }
    metrics = {
        "return": -500.0,
        "cost": 100.0,
        "energy_kwh": 200.0,
        "occupied_zone_hours": 1.0,
        "pmv_hours": 0.1,
        "occupied_pmv_violation_rate": 0.01,
        "deep_pmv_violation_rate": 0.0,
        "occupied_action_saturation": 0.1,
        "action_occ_unocc_gap": 0.5,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (evaluation / "metrics.json").write_text(
        json.dumps({**identity, "metrics": metrics}), encoding="utf-8"
    )


def test_aggregate_accepts_matching_cross_seed_identities(tmp_path) -> None:
    _write_result(tmp_path, 42, "0.8.0-dev")
    _write_result(tmp_path, 1337, "0.8.0-dev")
    report = aggregate_results(tmp_path)
    assert report["rows"] == 2
    assert report["identity_by_task"]["mz_hydro_ppo"]["boptest_version"] == "0.8.0-dev"
    contract = (tmp_path / "aggregate" / "drl_information_contract.csv").read_text(encoding="utf-8")
    assert "DRL baselines only" in contract
    assert "unoccupied 25-30 C" in contract


def test_aggregate_rejects_mixed_boptest_versions(tmp_path) -> None:
    _write_result(tmp_path, 42, "0.8.0-dev")
    _write_result(tmp_path, 1337, "1.0.0-dev")
    with pytest.raises(RuntimeError, match="Refusing to aggregate mixed"):
        aggregate_results(tmp_path)
