from __future__ import annotations

import itertools
from collections.abc import Iterator


def validate_continuation_mode(mode: str, continue_until_converged: bool) -> None:
    """Reject an unbounded continuation request outside a formal full run."""
    if continue_until_converged and mode != "full":
        raise ValueError("--continue-until-converged is only valid with --mode full")


def epoch_iterator(
    *, committed_epoch: int, registered_max_epochs: int, mode: str,
    continue_until_converged: bool,
) -> Iterator[int]:
    """Yield transaction epochs, optionally continuing beyond the registered cap."""
    validate_continuation_mode(mode, continue_until_converged)
    first = int(committed_epoch) + 1
    if mode == "smoke":
        return iter(range(first, 3))
    if continue_until_converged:
        return itertools.count(first)
    return iter(range(first, int(registered_max_epochs) + 1))


def next_validation_block_epochs(current_epoch: int, eval_interval: int) -> int:
    """Return epochs through the next registered deterministic evaluation point."""
    if current_epoch < 0:
        raise ValueError("current_epoch cannot be negative")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    next_boundary = (int(current_epoch) // int(eval_interval) + 1) * int(eval_interval)
    return next_boundary - int(current_epoch)


def linear_lr_multiplier(completed_epoch: float, decay_epochs: int) -> float:
    """Linear decay to 10%, then a fixed floor for every post-cap epoch."""
    if completed_epoch < 0:
        raise ValueError("completed_epoch cannot be negative")
    if decay_epochs <= 0:
        raise ValueError("decay_epochs must be positive")
    return max(0.1, 1.0 - 0.9 * float(completed_epoch) / int(decay_epochs))


def post_cap_learning_rate(initial_learning_rate: float) -> float:
    if initial_learning_rate <= 0:
        raise ValueError("initial_learning_rate must be positive")
    return float(initial_learning_rate) * 0.1


def completion_status(*, stopped: bool, reached_cap: bool) -> str:
    if stopped:
        return "early_stopped"
    return "cap_reached" if reached_cap else "incomplete"
