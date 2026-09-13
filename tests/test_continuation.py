from __future__ import annotations

import itertools

import pytest

from drl_multiseed.config import TASKS
from drl_multiseed.continuation import (
    completion_status,
    epoch_iterator,
    linear_lr_multiplier,
    next_validation_block_epochs,
    post_cap_learning_rate,
    training_should_start,
    validate_continuation_mode,
)


def test_default_full_run_stops_at_registered_cap() -> None:
    assert list(
        epoch_iterator(
            committed_epoch=298,
            registered_max_epochs=300,
            mode="full",
            continue_until_converged=False,
        )
    ) == [299, 300]


def test_continuation_is_unbounded_and_starts_after_committed_epoch() -> None:
    epochs = epoch_iterator(
        committed_epoch=300,
        registered_max_epochs=300,
        mode="full",
        continue_until_converged=True,
    )
    assert list(itertools.islice(epochs, 4)) == [301, 302, 303, 304]


def test_continuation_is_rejected_for_smoke() -> None:
    with pytest.raises(ValueError, match="mode full"):
        validate_continuation_mode("smoke", True)


def test_completed_ppo_smoke_resume_is_idempotent() -> None:
    assert training_should_start(mode="smoke", committed_epoch=1, stopped=False)
    assert not training_should_start(mode="smoke", committed_epoch=2, stopped=False)
    assert training_should_start(mode="full", committed_epoch=2, stopped=False)
    assert not training_should_start(mode="full", committed_epoch=2, stopped=True)


def test_next_validation_block_handles_boundary_and_partial_block() -> None:
    assert next_validation_block_epochs(300, 25) == 25
    assert next_validation_block_epochs(311, 25) == 14


def test_learning_rate_reaches_floor_and_never_restarts_after_cap() -> None:
    assert linear_lr_multiplier(0, 300) == 1.0
    assert linear_lr_multiplier(300, 300) == pytest.approx(0.1)
    assert linear_lr_multiplier(375, 300) == pytest.approx(0.1)
    assert post_cap_learning_rate(5e-4) == pytest.approx(5e-5)


def test_uniform_cap_does_not_stretch_the_frozen_lr_decay_horizons() -> None:
    for spec in TASKS.values():
        assert spec.max_epochs == 700
        assert linear_lr_multiplier(spec.lr_decay_epochs, spec.lr_decay_epochs) == pytest.approx(
            0.1
        )
        assert linear_lr_multiplier(spec.max_epochs, spec.lr_decay_epochs) == pytest.approx(0.1)


def test_completion_status_preserves_existing_manifest_vocabulary() -> None:
    assert completion_status(stopped=True, reached_cap=True) == "early_stopped"
    assert completion_status(stopped=False, reached_cap=True) == "cap_reached"
    assert completion_status(stopped=False, reached_cap=False) == "incomplete"
