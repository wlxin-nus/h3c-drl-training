"""Fast offline validation for a source checkout."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import nbformat
import numpy as np

from drl_multiseed.config import EARLY_STOP_PROTOCOL_ID, TASKS, repository_root, validate_task
from drl_multiseed.continuation import epoch_iterator, linear_lr_multiplier
from drl_multiseed.early_stop import PlateauState
from drl_multiseed.gae import compute_gae_arrays
from drl_multiseed.leases import CapacityManager
from drl_multiseed.observation_contract import refined_model_entry
from drl_multiseed.preflight import run_preflight


def main() -> None:
    for spec in TASKS.values():
        validate_task(spec, 1337)
        entry = refined_model_entry(spec.key)
        assert entry["action_history_steps"] == 4
        assert entry["temperature_missing"] == "repeat_earliest"
        assert spec.early_stop_protocol == EARLY_STOP_PROTOCOL_ID

    plateau = PlateauState(min_epoch=100, patience=3, min_delta_fraction=0.01)
    for epoch, score in (
        (25, -600.0),
        (50, -599.0),
        (75, -598.0),
        (100, -599.5),
        (125, -599.0),
        (150, -600.0),
        (175, -601.0),
    ):
        result = plateau.update(epoch, score)
    assert result["should_stop"] and plateau.stop_epoch == 175

    extension = epoch_iterator(
        committed_epoch=700,
        registered_max_epochs=700,
        mode="full",
        continue_until_converged=True,
    )
    assert [next(extension) for _ in range(3)] == [701, 702, 703]
    assert np.isclose(linear_lr_multiplier(700, 300), 0.1)

    advantages, _ = compute_gae_arrays(
        np.ones((1, 4), np.float32),
        np.full((1, 4), 2.0, np.float32),
        np.full((1, 4), 3.0, np.float32),
        np.zeros((1, 4), np.float32),
        np.ones((1, 4), np.float32),
        gamma=0.99,
        gae_lambda=0.95,
        time_limit_safe=True,
    )
    assert np.isclose(advantages[0, 0], 1.0 + 0.99 * 3.0 - 2.0)

    with tempfile.TemporaryDirectory() as directory:
        manager = CapacityManager(Path(directory), capacity=12)
        first = manager.acquire(task="a", seed=42, run_uuid="a", slots=5)
        second = manager.acquire(task="b", seed=42, run_uuid="b", slots=5)
        try:
            try:
                manager.acquire(task="c", seed=42, run_uuid="c", slots=5)
            except RuntimeError:
                pass
            else:
                raise AssertionError("capacity manager admitted a third five-worker job")
        finally:
            manager.release(first)
            manager.release(second)

    notebook_path = repository_root() / "notebooks" / "Monitor_Training.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
    assert len(code) == 3
    for cell in code:
        assert cell.execution_count is None and not cell.outputs
        compile(cell.source, str(notebook_path), "exec")

    report = run_preflight(seed=1337, online=False)
    assert report["passed"] and len(report["tasks"]) == 5
    print(
        json.dumps(
            {
                "passed": True,
                "tasks": list(report["tasks"]),
                "protocol_version": report["protocol_version"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
