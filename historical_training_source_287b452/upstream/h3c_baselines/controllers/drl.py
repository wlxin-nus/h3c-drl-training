"""Frozen centralized and hierarchical DRL controller surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from h3c_baselines.models import model_entry
from h3c_baselines.policies.mappo_adapter import HierarchicalMappoPolicy
from h3c_baselines.policies.observation_contracts import ObservationPacket
from h3c_baselines.policies.ppo_adapter import CentralizedPpoPolicy


class FrozenDrlController:
    def __init__(self, case: str, controller: str) -> None:
        self.entry = model_entry(case, controller)
        self.zone_order = tuple(str(zone) for zone in self.entry["policy_zone_order"])
        self.policy: Any = (
            CentralizedPpoPolicy(self.entry)
            if controller == "c-drl"
            else HierarchicalMappoPolicy(self.entry)
        )

    def decide(
        self,
        packet: ObservationPacket,
        occupancy: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        raw_action = (
            self.policy.predict(packet.normalized)
            if self.entry["controller"] == "c-drl"
            else self.policy.predict(packet.local_normalized)
        )
        if raw_action.shape != (len(self.zone_order),):
            raise ValueError("frozen policy action dimension is inconsistent with zone order")
        base_mode = str(self.entry["residual_base"])
        if base_mode not in {"fixed_25", "occupancy_25_30"}:
            raise ValueError("frozen policy residual base is invalid")
        setpoints: dict[str, float] = {}
        for index, zone in enumerate(self.zone_order):
            base = 25.0 if base_mode == "fixed_25" or float(occupancy[zone]) > 0 else 30.0
            setpoints[zone] = max(20.0, min(30.0, base + 5.0 * float(raw_action[index])))
        return setpoints, {
            "policy_zone_order": list(self.zone_order),
            "raw_action": raw_action.tolist(),
            "setpoints_c": dict(setpoints),
            "observation_dimension": int(packet.normalized.size),
        }
