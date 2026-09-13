from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """Frozen two-layer MAPPO actor used by the registered experiments."""

    def __init__(self, observation_dimension: int, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        previous = observation_dimension
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.ReLU()))
            previous = hidden
        self.backbone = nn.Sequential(*layers)
        self.mean_linear = nn.Linear(previous, 1)
        self.log_std = nn.Parameter(torch.full((1,), -1.0))
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_linear.weight, gain=0.01)
        nn.init.zeros_(self.mean_linear.bias)

    def distribution(self, observation: torch.Tensor) -> Normal:
        mean = torch.tanh(self.mean_linear(self.backbone(observation)))
        return Normal(mean, torch.clamp(self.log_std.exp(), min=0.01, max=1.0))

    def sample(self, observation: torch.Tensor, deterministic: bool = False):
        distribution = self.distribution(observation)
        action = distribution.mean if deterministic else distribution.rsample()
        return action, distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1)


class CentralizedCritic(nn.Module):
    def __init__(self, observation_dimension: int, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        previous = observation_dimension
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.ReLU()))
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        final = [layer for layer in self.network if isinstance(layer, nn.Linear)][-1]
        nn.init.orthogonal_(final.weight, gain=1.0)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)
