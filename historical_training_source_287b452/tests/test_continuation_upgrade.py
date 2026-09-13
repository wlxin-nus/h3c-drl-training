from __future__ import annotations

import json
import socket

import pytest

from drl_multiseed import cli


def _legacy_identity(run_dir) -> None:
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "latest.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_identity.json").write_text(json.dumps({
        "run_uuid": "test-run",
        "task": "sz_air_ppo",
        "seed": 2026,
        "hostname": socket.gethostname(),
        "code_commit": "old-commit",
        "code_fingerprint": next(iter(cli._CONTINUATION_UPGRADE_SOURCE_FINGERPRINTS)),
    }), encoding="utf-8")


def test_committed_legacy_run_requires_explicit_continuation_upgrade(tmp_path, monkeypatch) -> None:
    _legacy_identity(tmp_path)
    monkeypatch.setattr(cli, "_code_commit", lambda: "new-commit")
    monkeypatch.setattr(cli, "_code_fingerprint", lambda: cli._CONTINUATION_UPGRADE_TARGET_FINGERPRINT)
    with pytest.raises(RuntimeError, match="refusing silent resume"):
        cli._identity(tmp_path, "sz_air_ppo", 2026, False)


def test_explicit_continuation_upgrade_is_audited(tmp_path, monkeypatch) -> None:
    _legacy_identity(tmp_path)
    monkeypatch.setattr(cli, "_code_commit", lambda: "new-commit")
    monkeypatch.setattr(cli, "_code_fingerprint", lambda: cli._CONTINUATION_UPGRADE_TARGET_FINGERPRINT)
    identity = cli._identity(
        tmp_path, "sz_air_ppo", 2026, False, allow_continuation_upgrade=True,
    )
    assert identity["code_commit"] == "new-commit"
    assert identity["code_fingerprint"] == cli._CONTINUATION_UPGRADE_TARGET_FINGERPRINT
    audit = (tmp_path / "code_upgrades.jsonl").read_text(encoding="utf-8")
    assert "audited_continuation_upgrade" in audit
    assert '"scientific_configuration_changed": false' in audit


def test_unknown_code_drift_is_never_accepted(tmp_path, monkeypatch) -> None:
    _legacy_identity(tmp_path)
    monkeypatch.setattr(cli, "_code_commit", lambda: "unknown-commit")
    monkeypatch.setattr(cli, "_code_fingerprint", lambda: "0" * 64)
    with pytest.raises(RuntimeError, match="only permits the audited"):
        cli._identity(
            tmp_path, "sz_air_ppo", 2026, False, allow_continuation_upgrade=True,
        )
