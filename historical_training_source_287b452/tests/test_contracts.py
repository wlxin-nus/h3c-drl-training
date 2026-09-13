from __future__ import annotations

import json
import warnings
from pathlib import Path

import nbformat

from drl_multiseed.config import EARLY_STOP_PROTOCOL_ID, TASKS, validate_task
from drl_multiseed.preflight import (
    H3C_REGISTRY_CANONICAL_SHA256,
    REFINED_CONTRACT_CANONICAL_SHA256,
    _validate_boptest_version,
    _canonical_json_sha256,
    _golden_inference,
    _task_contract,
)


def test_all_tasks_freeze_four_full_episode_environments() -> None:
    for spec in TASKS.values():
        validate_task(spec, 1337)
        assert spec.num_envs == 4
        assert spec.n_steps == spec.episode_steps
        assert spec.steps_per_epoch == spec.episode_steps * 4
        assert spec.forecast_steps == 5


def test_all_tasks_share_acceptance_threshold_exploration_and_safety_cap() -> None:
    for spec in TASKS.values():
        assert spec.comfort_threshold == 0.5
        assert spec.ent_coef == 0.02
        assert spec.max_epochs == 700
        assert spec.early_stop_protocol == EARLY_STOP_PROTOCOL_ID
    assert {key: spec.lr_decay_epochs for key, spec in TASKS.items()} == {
        "sz_air_ppo": 300,
        "mz_air_ppo": 300,
        "mz_air_mappo": 300,
        "mz_hydro_ppo": 500,
        "mz_hydro_mappo": 500,
    }


def test_new_training_structures_match_refined_contract() -> None:
    for spec in TASKS.values():
        result = _task_contract(spec)
        assert all(result["assertions"].values())


def test_frozen_h3c_golden_inputs_outputs_and_payloads() -> None:
    assert all(item["passed"] for item in _golden_inference().values())


def test_registry_identity_is_portable_across_line_endings(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "models" / "registry.json"
    content = source.read_text(encoding="utf-8")
    lf_path = tmp_path / "registry-lf.json"
    crlf_path = tmp_path / "registry-crlf.json"
    lf_path.write_bytes(content.replace("\r\n", "\n").encode("utf-8"))
    crlf_path.write_bytes(content.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))

    assert _canonical_json_sha256(lf_path) == H3C_REGISTRY_CANONICAL_SHA256
    assert _canonical_json_sha256(crlf_path) == H3C_REGISTRY_CANONICAL_SHA256

    changed = json.loads(content)
    changed["schema_version"] = int(changed["schema_version"]) + 1
    changed_path = tmp_path / "registry-changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    assert _canonical_json_sha256(changed_path) != H3C_REGISTRY_CANONICAL_SHA256


def test_refined_contract_identity_is_portable_across_line_endings(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "refined_observation_contracts.json"
    content = source.read_text(encoding="utf-8")
    lf_path = tmp_path / "refined-lf.json"
    crlf_path = tmp_path / "refined-crlf.json"
    lf_path.write_bytes(content.replace("\r\n", "\n").encode("utf-8"))
    crlf_path.write_bytes(content.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
    assert _canonical_json_sha256(lf_path) == REFINED_CONTRACT_CANONICAL_SHA256
    assert _canonical_json_sha256(crlf_path) == REFINED_CONTRACT_CANONICAL_SHA256


def test_supported_boptest_versions_are_accepted_and_disclosed() -> None:
    assert _validate_boptest_version("0.8.0-dev") == "0.8.0-dev"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert _validate_boptest_version("1.0.0-dev") == "1.0.0-dev"
    assert any("reference experiments" in str(item.message) for item in captured)


def test_monitor_notebook_is_clean_and_has_exactly_three_code_cells() -> None:
    path = Path(__file__).parents[1] / "notebooks" / "Monitor_Training.ipynb"
    notebook = nbformat.read(path, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
    assert len(code) == 3
    for cell in code:
        assert cell.execution_count is None
        assert not cell.outputs
        compile(cell.source, str(path), "exec")
