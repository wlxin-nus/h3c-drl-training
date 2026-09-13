"""Inference-only loader for the archived two-layer MAPPO actors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from h3c_baselines.models import verify_checkpoint


class HierarchicalMappoPolicy:
    def __init__(self, entry: Mapping[str, Any]) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as error:
            raise RuntimeError("install H3C[baselines] to evaluate MAPPO policies") from error

        class ActorNetwork(nn.Module):
            def __init__(self, observation_dimension: int) -> None:
                super().__init__()
                layers: list[nn.Module] = []
                previous = observation_dimension
                for hidden in (256, 256):
                    layers.extend((nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.ReLU()))
                    previous = hidden
                self.backbone = nn.Sequential(*layers)
                self.mean_linear = nn.Linear(previous, 1)
                self.log_std = nn.Parameter(torch.full((1,), -1.0))

            def forward(self, observation: Any) -> Any:
                return torch.tanh(self.mean_linear(self.backbone(observation)))

        identity = verify_checkpoint(entry)
        payload = torch.load(identity["path"], map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("MAPPO checkpoint payload is invalid")
        self.torch = torch
        self.entry = dict(entry)
        self.actors: dict[str, Any] = {}
        states = payload.get("actors")
        for zone in entry["policy_zone_order"]:
            key = str(zone).lower()
            state = states.get(key) if isinstance(states, Mapping) else None
            if state is None:
                state = payload.get(f"actor_{key}_state_dict")
            if not isinstance(state, Mapping):
                raise ValueError(f"MAPPO checkpoint is missing actor state: {zone}")
            actor = ActorNetwork(int(entry["local_observation_dimension"]))
            actor.load_state_dict(state)
            actor.eval()
            self.actors[str(zone)] = actor

    def predict(self, local_observations: Mapping[str, NDArray[np.float32]]) -> NDArray[np.float64]:
        actions: list[float] = []
        with self.torch.no_grad():
            for zone in self.entry["policy_zone_order"]:
                vector = local_observations.get(str(zone))
                if vector is None:
                    raise ValueError(f"local MAPPO observation is missing zone: {zone}")
                tensor = self.torch.as_tensor(vector, dtype=self.torch.float32).unsqueeze(0)
                actions.append(float(self.actors[str(zone)](tensor).item()))
        result = np.asarray(actions, dtype=np.float64)
        if np.any(~np.isfinite(result)):
            raise ValueError("MAPPO produced a non-finite deterministic action")
        return np.clip(result, -1.0, 1.0)
