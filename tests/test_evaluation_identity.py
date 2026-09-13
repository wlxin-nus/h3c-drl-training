from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from drl_multiseed import cli, evaluation
from drl_multiseed.config import PROTOCOL_VERSION, get_task


def _write_identity_files(run_dir, *, status: str = "cap_reached") -> None:
    spec = get_task("mz_hydro_ppo")
    seed = 1337
    fingerprint = "trained-source"
    identity = {
        "task": spec.key,
        "seed": seed,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": spec.protocol_hash(),
        "config_hash": spec.scientific_hash(seed),
        "code_fingerprint": fingerprint,
        "boptest_version": "0.8.0-dev",
        "run_uuid": "identity-test",
    }
    run_dir.mkdir(parents=True)
    (run_dir / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({**identity, "status": status}), encoding="utf-8"
    )
    (run_dir / "preflight.json").write_text(
        json.dumps(
            {
                "boptest_version": "0.8.0-dev",
                "tasks": {spec.key: {"scientific_hash": spec.scientific_hash(seed)}},
            }
        ),
        encoding="utf-8",
    )


def test_formal_evaluation_rejects_current_source_drift(tmp_path, monkeypatch) -> None:
    spec = get_task("mz_hydro_ppo")
    run_dir = tmp_path / "run"
    _write_identity_files(run_dir)
    monkeypatch.setattr(evaluation, "code_fingerprint", lambda: "changed-source")
    with pytest.raises(RuntimeError, match="current source fingerprint"):
        evaluation._run_metadata(spec, 1337, run_dir)


def test_cli_blocks_formal_evaluation_before_training_is_terminal(tmp_path) -> None:
    run_dir = tmp_path / "full" / "seed1337" / "mz_hydro_ppo"
    _write_identity_files(run_dir, status="incomplete")
    args = SimpleNamespace(
        task="mz_hydro_ppo",
        seed=1337,
        mode="full",
        output_root=str(tmp_path),
        worker_capacity=12,
        lease_wait_seconds=0.0,
        endpoint="http://127.0.0.1:8000",
        device="cpu",
        force=False,
    )
    with pytest.raises(RuntimeError, match="terminal state"):
        cli._evaluate(args)
