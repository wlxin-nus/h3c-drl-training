from __future__ import annotations

import numpy as np

from drl_multiseed.gae import compute_gae_arrays


def _one_step(terminated: float, truncated: float) -> float:
    advantages, _returns = compute_gae_arrays(
        np.ones((1, 4), np.float32),
        np.full((1, 4), 2.0, np.float32),
        np.full((1, 4), 3.0, np.float32),
        np.full((1, 4), terminated, np.float32),
        np.full((1, 4), truncated, np.float32),
        gamma=0.99,
        gae_lambda=0.95,
        time_limit_safe=True,
    )
    return float(advantages[0, 0])


def test_genuine_termination_does_not_bootstrap() -> None:
    assert np.isclose(_one_step(1.0, 0.0), 1.0 - 2.0)


def test_time_limit_truncation_bootstraps_terminal_value() -> None:
    assert np.isclose(_one_step(0.0, 1.0), 1.0 + 0.99 * 3.0 - 2.0)


def test_nonterminal_rollout_boundary_bootstraps() -> None:
    assert np.isclose(_one_step(0.0, 0.0), 1.0 + 0.99 * 3.0 - 2.0)
