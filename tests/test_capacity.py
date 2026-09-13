from __future__ import annotations

import json

import pytest

from drl_multiseed.leases import CapacityManager


def test_two_jobs_fit_but_third_is_rejected(tmp_path) -> None:
    manager = CapacityManager(tmp_path, capacity=12)
    first = manager.acquire(task="a", seed=42, run_uuid="a", slots=5)
    second = manager.acquire(task="b", seed=42, run_uuid="b", slots=5)
    with pytest.raises(RuntimeError, match="Insufficient BOPTEST capacity"):
        manager.acquire(task="c", seed=42, run_uuid="c", slots=5)
    state = json.loads((tmp_path / "worker_leases.json").read_text(encoding="utf-8"))
    assert sum(item["slots"] for item in state["leases"].values()) == 10
    manager.release(first)
    manager.release(second)
