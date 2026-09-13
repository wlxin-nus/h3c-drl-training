from __future__ import annotations

import argparse
import contextlib
import json
import socket
import sys
import types
import urllib.error

import pytest

from drl_multiseed import cli
from drl_multiseed.http_recovery import (
    is_retryable_boptest_error,
    recovery_delay,
    wait_until_healthy,
)


@pytest.mark.parametrize("winerror", [109, 10048, 10055])
def test_windows_socket_pressure_is_retryable_through_url_chain(winerror: int) -> None:
    cause = OSError(winerror, "socket resources exhausted")
    cause.winerror = winerror
    wrapped = urllib.error.URLError(cause)
    outer = RuntimeError("BOPTEST worker failed")
    outer.__cause__ = wrapped
    assert is_retryable_boptest_error(outer)


def test_windows_109_plain_oserror_and_wrapped_text_are_retryable() -> None:
    # Depending on where multiprocessing notices the closed worker pipe,
    # Windows/Python may expose 109 as errno, winerror, or only message text.
    assert is_retryable_boptest_error(OSError(109, "The pipe has been ended"))
    assert is_retryable_boptest_error(
        RuntimeError("SubprocVecEnv recv failed: [WinError 109] The pipe has been ended")
    )


def test_worker_eof_and_transient_http_are_retryable() -> None:
    class TransportError(RuntimeError):
        def __init__(self):
            super().__init__("HTTP 503")
            self.retryable = False
            self.error_type = "http_503"

    assert is_retryable_boptest_error(EOFError("worker pipe closed"))
    assert is_retryable_boptest_error(TransportError())


def test_bare_checkpoint_eof_and_http_404_are_not_retried() -> None:
    assert not is_retryable_boptest_error(EOFError())
    assert not is_retryable_boptest_error(
        urllib.error.HTTPError(
            "http://example",
            404,
            "not found",
            {},
            None,
        )
    )


@pytest.mark.parametrize(
    "error", [ValueError("shape"), RuntimeError("checkpoint checksum mismatch")]
)
def test_non_transport_failures_are_not_retried(error: BaseException) -> None:
    assert not is_retryable_boptest_error(error)


def test_recovery_backoff_is_bounded() -> None:
    assert [recovery_delay(index, 15) for index in range(1, 6)] == [15, 30, 60, 60, 60]


def test_health_wait_retries_without_mutating_testcases() -> None:
    attempts = iter([False, False, True])
    clock = iter([0.0, 0.0, 1.0, 2.0])
    assert wait_until_healthy(
        "http://example",
        timeout_seconds=5,
        poll_seconds=1,
        health_check=lambda endpoint, timeout_seconds: next(attempts),
        sleep=lambda seconds: None,
        monotonic=lambda: next(clock),
    )


def test_train_auto_resumes_same_run_after_worker_eof(tmp_path, monkeypatch) -> None:
    calls: list[bool] = []

    class Manager:
        def __init__(self, *args, **kwargs):
            pass

        def cleanup_stale(self):
            return []

        @contextlib.contextmanager
        def hold(self, **kwargs):
            yield

    @contextlib.contextmanager
    def unlocked(*args, **kwargs):
        yield

    def fake_train(spec, **kwargs):
        calls.append(bool(kwargs["resume"]))
        if len(calls) == 1:
            raise EOFError("worker pipe closed after HTTP failure")

    monkeypatch.setattr(cli, "_output_root", lambda value: tmp_path)
    monkeypatch.setattr(cli, "_cleanup_stale_runs", lambda *args: [])
    monkeypatch.setattr(cli, "_preflight_with_http_recovery", lambda *args: None)
    monkeypatch.setattr(cli, "_identity", lambda *args, **kwargs: {"run_uuid": "same-run"})
    monkeypatch.setattr(cli, "CapacityManager", Manager)
    monkeypatch.setattr(cli, "exclusive_run_lock", unlocked)
    monkeypatch.setattr(cli, "_wait_for_clean_recovery", lambda **kwargs: True)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    fake_module = types.ModuleType("drl_multiseed.ppo_training")
    fake_module.train_ppo = fake_train
    monkeypatch.setitem(sys.modules, "drl_multiseed.ppo_training", fake_module)

    args = argparse.Namespace(
        task="sz_air_ppo",
        seed=1337,
        mode="full",
        endpoint="http://example",
        output_root=None,
        worker_capacity=12,
        allow_host_migration=False,
        continue_until_converged=False,
        lease_wait_seconds=0,
        resume=False,
        wandb_mode="disabled",
        device="cpu",
        http_auto_resume_attempts=2,
        http_resume_backoff_seconds=1.0,
        http_health_timeout_seconds=30.0,
    )
    assert cli._train(args) == 0
    assert calls == [False, True]
    failure = json.loads(
        (tmp_path / "full" / "seed1337" / "sz_air_ppo" / "last_failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["auto_resume_scheduled"] is True


def test_identity_allows_audited_runtime_only_update_after_checkpoint(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(cli, "_code_commit", lambda: "old")
    monkeypatch.setattr(cli, "_code_fingerprint", lambda: "same-science")
    original = cli._identity(run_dir, "mz_air_ppo", 1337, False)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "latest.json").write_text('{"epoch": 12}', encoding="utf-8")

    monkeypatch.setattr(cli, "_code_commit", lambda: "new")
    updated = cli._identity(run_dir, "mz_air_ppo", 1337, False)

    assert updated["run_uuid"] == original["run_uuid"]
    assert updated["hostname"] == socket.gethostname()
    assert updated["code_commit"] == "new"
    audit = (run_dir / "code_upgrades.jsonl").read_text(encoding="utf-8")
    assert "runtime_only_code_update" in audit
    assert '"scientific_configuration_changed": false' in audit
