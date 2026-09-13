"""Deterministic inference adapter for frozen centralized PPO policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from h3c_baselines.models import load_ppo_checkpoint


class CentralizedPpoPolicy:
    def __init__(self, entry: Mapping[str, Any]) -> None:
        self.entry = dict(entry)
        self.model = load_ppo_checkpoint(entry)

    def predict(self, observation: NDArray[np.float32]) -> NDArray[np.float64]:
        action, _ = self.model.predict(observation, deterministic=True)
        result = np.asarray(action, dtype=np.float64).reshape(-1)
        expected = int(self.entry["action_dimension"])
        if result.shape != (expected,) or np.any(~np.isfinite(result)):
            raise ValueError("PPO produced an invalid deterministic action")
        return np.clip(result, -1.0, 1.0)
