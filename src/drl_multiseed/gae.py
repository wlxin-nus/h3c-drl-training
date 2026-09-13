from __future__ import annotations

import numpy as np


def compute_gae_arrays(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    time_limit_safe: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE without crossing reset boundaries.

    In the time-limit-safe contract a truncated terminal state is bootstrapped,
    while the recursive advantage is still stopped at the reset boundary.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_advantage = np.zeros(rewards.shape[1], np.float32)
    for index in reversed(range(rewards.shape[0])):
        if time_limit_safe:
            bootstrap = 1.0 - terminated[index]
            continuation = bootstrap * (1.0 - truncated[index])
        else:
            done = np.maximum(terminated[index], truncated[index])
            bootstrap = 1.0 - done
            continuation = bootstrap
        delta = rewards[index] + gamma * bootstrap * next_values[index] - values[index]
        next_advantage = delta + gamma * gae_lambda * continuation * next_advantage
        advantages[index] = next_advantage
    return advantages, advantages + values
