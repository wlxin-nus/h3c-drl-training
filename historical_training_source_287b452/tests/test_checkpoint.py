from __future__ import annotations

import os

import torch

from CASE_TEST.rl_retraining_v3 import AtomicTorchCheckpointManager


def _payload(epoch: int) -> dict:
    return {
        "committed_epoch": epoch,
        "actors": {"zone": {"weight": torch.tensor([float(epoch)])}},
        "critic": {"weight": torch.tensor([2.0])},
        "global_step": epoch * 4,
    }


def test_atomic_checkpoint_round_trip_and_retry(tmp_path) -> None:
    attempts = {"count": 0}

    def flaky_replace(source, destination):
        if str(destination).endswith(".mappo.pt") and attempts["count"] == 0:
            attempts["count"] += 1
            raise PermissionError("injected transient rename failure")
        os.replace(source, destination)

    manager = AtomicTorchCheckpointManager(
        tmp_path, "config", save_retries=2, replace_func=flaky_replace
    )
    pointer = manager.save(_payload(1), epoch=1, global_step=4)
    manager.mark_best(pointer, -10.0)
    assert manager.load_latest()["payload"]["committed_epoch"] == 1
    assert manager.load_best()["pointer"]["score"] == -10.0
    assert attempts["count"] == 1


def test_failed_commit_keeps_previous_latest(tmp_path) -> None:
    manager = AtomicTorchCheckpointManager(tmp_path, "config")
    manager.save(_payload(1), epoch=1, global_step=4)

    def always_fail(source, destination):
        raise PermissionError("injected permanent failure")

    failing = AtomicTorchCheckpointManager(
        tmp_path, "config", save_retries=1, replace_func=always_fail
    )
    try:
        failing.save(_payload(2), epoch=2, global_step=8)
    except RuntimeError:
        pass
    assert manager.load_latest()["payload"]["committed_epoch"] == 1

