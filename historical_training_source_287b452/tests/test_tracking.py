from __future__ import annotations

import sys

from drl_multiseed.tracking import SafeWandb


def test_missing_wandb_never_interrupts_training(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    tracker = SafeWandb(
        tmp_path, project="p", group="g", name="n", run_id="id",
        mode="online", config={"seed": 42},
    )
    tracker.start()
    tracker.log({"reward": -1.0}, step=1)
    tracker.finish({"completed_epoch": 1})
    assert (tmp_path / "wandb" / "wandb_disabled.json").exists()
    assert (tmp_path / "wandb" / "pending_history.jsonl").exists()
    assert (tmp_path / "wandb" / "pending_summary.json").exists()
