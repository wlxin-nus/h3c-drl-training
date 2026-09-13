from __future__ import annotations

from drl_multiseed.early_stop import PlateauState


def test_warmup_does_not_consume_patience_and_earliest_stop_is_epoch_175() -> None:
    state = PlateauState(min_epoch=100, patience=3, min_delta_fraction=0.01)
    first = state.update(25, -600.0)
    assert first["meaningful_improvement"]
    assert state.best_score == -600.0
    for epoch, score in ((50, -599.0), (75, -598.0), (100, -599.5)):
        warmup = state.update(epoch, score)
        assert not warmup["patience_active"]
        assert warmup["misses"] == 0
        assert not warmup["should_stop"]
    assert state.best_score == -598.0
    first_miss = state.update(125, -599.0)
    second_miss = state.update(150, -600.0)
    final = state.update(175, -601.0)
    assert first_miss["misses"] == 1
    assert second_miss["misses"] == 2
    assert final["should_stop"]
    assert state.stop_epoch == 175


def test_significant_improvement_resets_patience() -> None:
    state = PlateauState(min_epoch=100, patience=3, min_delta_fraction=0.01)
    state.update(25, -600.0)
    boundary = state.update(50, -594.0)
    assert not boundary["meaningful_improvement"]
    assert boundary["threshold"] == 6.0
    update = state.update(75, -593.9)
    assert update["meaningful_improvement"]
    assert state.misses == 0
    assert state.anchor == -593.9


def test_post_warmup_significant_improvement_resets_patience() -> None:
    state = PlateauState(min_epoch=100, patience=3, min_delta_fraction=0.01)
    state.update(25, -600.0)
    state.update(100, -599.0)
    state.update(125, -598.0)
    assert state.misses == 1
    improvement = state.update(150, -593.9)
    assert improvement["meaningful_improvement"]
    assert improvement["patience_active"]
    assert state.misses == 0


def test_post_warmup_patience_survives_checkpoint_round_trip() -> None:
    state = PlateauState(min_epoch=100, patience=3, min_delta_fraction=0.01)
    for epoch, score in ((25, -600.0), (100, -599.0), (125, -598.0), (150, -599.0)):
        state.update(epoch, score)
    assert state.misses == 2
    restored = PlateauState.from_dict(state.to_dict())
    result = restored.update(175, -600.0)
    assert result["should_stop"]
    assert restored.stop_epoch == 175
