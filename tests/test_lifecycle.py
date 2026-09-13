from __future__ import annotations

import json

import pytest

from drl_multiseed import cli
from drl_multiseed.environment import LifecycleRecorder, cleanup_run_testids
from drl_multiseed.leases import exclusive_run_lock


def test_lifecycle_persists_selected_before_initialize_and_clears_on_stop(tmp_path) -> None:
    recorder = LifecycleRecorder(tmp_path, "training_0", "mz_hydro_ppo", 1337)
    recorder({"event": "selected", "test_id": "owned-id"})
    assert recorder.active.exists()
    active = json.loads(recorder.active.read_text(encoding="utf-8"))
    assert active["test_id"] == "owned-id"
    recorder({"event": "initialized", "test_id": "owned-id"})
    assert recorder.active.exists()
    recorder({"event": "stopped", "test_id": "owned-id"})
    assert not recorder.active.exists()


def test_cleanup_only_touches_ids_inside_requested_run(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    LifecycleRecorder(first, "a", "task-a", 42)({"event": "selected", "test_id": "first-id"})
    LifecycleRecorder(second, "b", "task-b", 42)({"event": "selected", "test_id": "second-id"})
    called = []

    class Response:
        status_code = 200

    monkeypatch.setattr("requests.put", lambda url, timeout: (called.append(url), Response())[1])
    cleanup_run_testids(first, "http://example")
    assert called == ["http://example/stop/first-id"]
    assert (second / "lifecycle" / "b.active.json").exists()


def test_run_lock_prevents_duplicate_resume(tmp_path) -> None:
    with exclusive_run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with exclusive_run_lock(tmp_path):
                pass


def test_preflight_only_identity_can_follow_a_code_update(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(cli, "_code_commit", lambda: "old")
    monkeypatch.setattr(cli, "_code_fingerprint", lambda: "old-fingerprint")
    original = cli._identity(run_dir, "mz_air_ppo", 1337, False)

    monkeypatch.setattr(cli, "_code_commit", lambda: "new")
    monkeypatch.setattr(cli, "_code_fingerprint", lambda: "new-fingerprint")
    refreshed = cli._identity(run_dir, "mz_air_ppo", 1337, False)

    assert refreshed["run_uuid"] == original["run_uuid"]
    assert refreshed["code_commit"] == "new"
    assert refreshed["code_fingerprint"] == "new-fingerprint"


def test_identity_refuses_code_update_after_checkpoint_commit(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(cli, "_code_commit", lambda: "old")
    monkeypatch.setattr(cli, "_code_fingerprint", lambda: "old-fingerprint")
    cli._identity(run_dir, "mz_air_ppo", 1337, False)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "latest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli, "_code_commit", lambda: "new")
    monkeypatch.setattr(cli, "_code_fingerprint", lambda: "new-fingerprint")
    with pytest.raises(RuntimeError, match="after training progress was committed"):
        cli._identity(run_dir, "mz_air_ppo", 1337, False)


def test_identity_refuses_boptest_version_change(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(cli, "_code_commit", lambda: "same")
    monkeypatch.setattr(cli, "_code_fingerprint", lambda: "same-fingerprint")
    preflight = run_dir / "preflight.json"
    preflight.write_text(json.dumps({"boptest_version": "0.8.0-dev"}), encoding="utf-8")
    cli._identity(run_dir, "mz_air_ppo", 1337, False)
    preflight.write_text(json.dumps({"boptest_version": "1.0.0-dev"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="BOPTEST version changed"):
        cli._identity(run_dir, "mz_air_ppo", 1337, False)
