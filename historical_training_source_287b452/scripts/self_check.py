from __future__ import annotations

import json
import importlib.util
import tempfile
from pathlib import Path

import nbformat
import numpy as np

from drl_multiseed.config import (
    EARLY_STOP_PROTOCOL_ID,
    OBSERVATION_CONTRACT_ID,
    TASKS,
    WANDB_PROJECT,
    get_task,
    repository_root,
    validate_task,
)
from drl_multiseed.continuation import (
    epoch_iterator,
    linear_lr_multiplier,
    next_validation_block_epochs,
)
from drl_multiseed.early_stop import PlateauState
from drl_multiseed.leases import CapacityManager
from drl_multiseed.gae import compute_gae_arrays
from drl_multiseed.preflight import run_preflight
from drl_multiseed.observation_contract import refined_model_entry


def main() -> None:
    for module in ("h3c.runtime.clients", "h3c.runtime.protocol"):
        assert importlib.util.find_spec(module) is not None, f"missing required package: {module}"
    for spec in TASKS.values():
        validate_task(spec, 1337)
        assert spec.num_envs == 4 and spec.n_steps == spec.episode_steps
        assert spec.max_epochs == 700
        assert spec.comfort_threshold == 0.5
        assert spec.ent_coef == 0.02
        assert spec.early_stop_protocol == EARLY_STOP_PROTOCOL_ID
        entry = refined_model_entry(spec.key)
        assert entry["temperature_past_offset"] == 1
        assert entry["action_past_offset"] == 0
        assert entry["power_past_offset"] == 0
        assert entry["action_history_steps"] == 4
        assert entry["temperature_missing"] == "repeat_earliest"
    assert {key: spec.lr_decay_epochs for key, spec in TASKS.items()} == {
        "sz_air_ppo": 300,
        "mz_air_ppo": 300,
        "mz_air_mappo": 300,
        "mz_hydro_ppo": 500,
        "mz_hydro_mappo": 500,
    }

    plateau = PlateauState(min_epoch=100, patience=3, min_delta_fraction=0.01)
    for epoch, score in (
        (25, -600.0), (50, -599.0), (75, -598.0), (100, -599.5),
        (125, -599.0), (150, -600.0), (175, -601.0),
    ):
        result = plateau.update(epoch, score)
    assert result["should_stop"] and plateau.stop_epoch == 175
    assert plateau.best_score == -598.0

    extension = epoch_iterator(
        committed_epoch=300, registered_max_epochs=300, mode="full",
        continue_until_converged=True,
    )
    assert [next(extension) for _ in range(3)] == [301, 302, 303]
    assert next_validation_block_epochs(311, 25) == 14
    assert np.isclose(linear_lr_multiplier(375, 300), 0.1)

    advantages, _returns = compute_gae_arrays(
        np.ones((1, 4), np.float32), np.full((1, 4), 2.0, np.float32),
        np.full((1, 4), 3.0, np.float32), np.zeros((1, 4), np.float32),
        np.ones((1, 4), np.float32), gamma=0.99, gae_lambda=0.95,
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
                raise AssertionError("third five-slot job was incorrectly admitted")
        finally:
            manager.release(first); manager.release(second)

    notebook_path = repository_root() / "notebooks" / "Monitor_Training.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
    assert len(code) == 3
    for cell in code:
        assert cell.execution_count is None and not cell.outputs
        compile(cell.source, str(notebook_path), "exec")

    report = run_preflight(seed=1337, online=False)
    assert report["passed"] and len(report["tasks"]) == 5
    assert report["refined_contract_id"] == OBSERVATION_CONTRACT_ID
    assert report["wandb_project"] == WANDB_PROJECT == "h3c-drl-multiseed-refine-v2"
    assert report["uniform_training_settings"]["early_stop_protocol"] == EARLY_STOP_PROTOCOL_ID
    print(json.dumps({
        "passed": True, "tasks": list(report["tasks"]),
        "golden_contracts": list(report["golden_inference"]),
        "refined_contract": report["refined_contract_canonical_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
