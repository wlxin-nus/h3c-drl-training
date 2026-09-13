"""Shared observation builder for legacy and refined H3C DRL policies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from h3c.runtime.occupancy import effective_count
from h3c.runtime.protocol import zone_temperature_c
from h3c_baselines.policies.normalization import symmetric_minmax


@dataclass(frozen=True)
class ObservationPacket:
    raw: NDArray[np.float64]
    normalized: NDArray[np.float32]
    local_normalized: dict[str, NDArray[np.float32]]
    columns: tuple[str, ...]


def _axis(values: Sequence[float], bound: tuple[float, float]) -> tuple[list[float], list[float]]:
    return [bound[0]] * len(values), [bound[1]] * len(values)


class PolicyObservationBuilder:
    """Stateful observation builder; state is reset at the formal evaluation boundary."""

    def __init__(self, profile: Mapping[str, Any], model: Mapping[str, Any]) -> None:
        self.profile = profile
        self.zone_order = tuple(str(zone) for zone in model["policy_zone_order"])
        if set(self.zone_order) != set(profile["zones"]):
            raise ValueError("policy zone order does not cover the profile zones")
        self.expected_dimension = int(model["observation_dimension"])
        self.local_dimension = int(model.get("local_observation_dimension", 0))
        self.action_history_steps = int(model["action_history_steps"])
        self.cold_start_action_c = float(model["cold_start_action_c"])
        time_bounds = model["time_observation_bounds"]
        if (
            not isinstance(time_bounds, list)
            or len(time_bounds) != 2
            or not all(isinstance(value, (int, float)) for value in time_bounds)
        ):
            raise ValueError("policy time-observation bounds are invalid")
        self.time_observation_bounds = (float(time_bounds[0]), float(time_bounds[1]))
        if (
            not all(math.isfinite(value) for value in self.time_observation_bounds)
            or self.time_observation_bounds[0] >= self.time_observation_bounds[1]
        ):
            raise ValueError("policy time-observation bounds are invalid")
        action_bounds = model["action_observation_bounds_k"]
        if (
            not isinstance(action_bounds, list)
            or len(action_bounds) != 2
            or not all(isinstance(value, (int, float)) for value in action_bounds)
        ):
            raise ValueError("policy action-observation bounds are invalid")
        self.action_observation_bounds_k = (float(action_bounds[0]), float(action_bounds[1]))
        if (
            not all(math.isfinite(value) for value in self.action_observation_bounds_k)
            or self.action_observation_bounds_k[0] >= self.action_observation_bounds_k[1]
        ):
            raise ValueError("policy action-observation bounds are invalid")
        self.temperature_past_offset = int(model["temperature_past_offset"])
        self.action_past_offset = int(model["action_past_offset"])
        self.power_past_offset = int(model["power_past_offset"])
        self.temperature_missing = str(model["temperature_missing"])
        self.local_observation_layout = str(model.get("local_observation_layout", ""))
        if any(
            offset not in {0, 1}
            for offset in (
                self.temperature_past_offset,
                self.action_past_offset,
                self.power_past_offset,
            )
        ):
            raise ValueError("policy history offset is invalid")
        if self.temperature_missing not in {"repeat_earliest", "fixed_25"}:
            raise ValueError("policy temperature missing-value rule is invalid")
        self.occupancy_encoding = str(model["occupancy_encoding"])
        if self.occupancy_encoding not in {"effective_count", "binary_raw"}:
            raise ValueError("policy occupancy encoding is invalid")
        if self.local_dimension and self.local_observation_layout not in {
            "zone_then_shared",
            "shared_then_zone",
        }:
            raise ValueError("local MAPPO observation layout is invalid")
        self.occupancy_upper = float(model["occupancy_upper"])
        self.temperature_history_k: dict[str, list[float]] = {}
        self.action_history_k: dict[str, list[float]] = {}
        self.power_history: list[float] = []
        self.current_pmv: dict[str, float] = {}
        self._initialized = False

    def reset(self, state: Mapping[str, Any]) -> None:
        self.temperature_history_k = {}
        for zone in self.zone_order:
            current = zone_temperature_c(self.profile, state, zone) + 273.15
            self.temperature_history_k[zone] = [current]
        self.action_history_k = {
            zone: [self.cold_start_action_c + 273.15] for zone in self.zone_order
        }
        self.power_history = [0.0]
        self.current_pmv = dict.fromkeys(self.zone_order, 0.0)
        self._initialized = True

    def update(
        self,
        state: Mapping[str, Any],
        setpoints_c: Mapping[str, float],
        pmv: Mapping[str, float],
        power_w: float,
    ) -> None:
        if not self._initialized:
            raise ValueError("policy observation history has not been reset")
        maximum_power = float(self.profile["performance"]["maximum_power_w"])
        for zone in self.zone_order:
            temperature_k = zone_temperature_c(self.profile, state, zone) + 273.15
            self.temperature_history_k[zone] = [
                *self.temperature_history_k[zone],
                temperature_k,
            ][-5:]
            self.action_history_k[zone] = [
                *self.action_history_k[zone],
                float(setpoints_c[zone]) + 273.15,
            ][-5:]
            self.current_pmv[zone] = float(pmv[zone])
        self.power_history = [*self.power_history, float(power_w) / maximum_power][-5:]

    @staticmethod
    def _past_value(
        history: Sequence[float],
        *,
        slot: int,
        offset: int,
        missing: str,
        fixed: float,
    ) -> float:
        index = len(history) - slot - offset
        if index >= 0:
            return float(history[index])
        if missing == "repeat_earliest":
            return float(history[0])
        return fixed

    def _temperature_values(self, zone: str) -> list[float]:
        history = self.temperature_history_k[zone]
        missing = self.temperature_missing
        return [
            float(history[-1]),
            *[
                self._past_value(
                    history,
                    slot=slot,
                    offset=self.temperature_past_offset,
                    missing=missing,
                    fixed=298.15,
                )
                for slot in range(1, 5)
            ],
        ]

    def _action_values(self, zone: str) -> list[float]:
        return [
            self._past_value(
                self.action_history_k[zone],
                slot=slot,
                offset=self.action_past_offset,
                missing="fixed",
                fixed=self.cold_start_action_c + 273.15,
            )
            for slot in range(1, self.action_history_steps + 1)
        ]

    def _power_values(self) -> list[float]:
        return [
            self._past_value(
                self.power_history,
                slot=slot,
                offset=self.power_past_offset,
                missing="fixed",
                fixed=0.0,
            )
            for slot in range(1, 5)
        ]

    def _occupancy_forecasts(
        self,
        forecast: Mapping[str, Sequence[float]],
        *,
        step: int,
        action_time_seconds: int,
        step_seconds: int,
    ) -> dict[str, list[float]]:
        values: dict[str, list[float]] = {}
        for zone in self.zone_order:
            point = self.profile["zones"][zone]["occupancy_forecast"]
            resolved = []
            for horizon in range(5):
                raw_count = float(forecast[point][step + horizon])
                if self.occupancy_encoding == "binary_raw":
                    resolved.append(1.0 if raw_count > 0 else 0.0)
                else:
                    resolved.append(
                        float(
                            effective_count(
                                self.profile["occupancy"],
                                action_time_seconds + horizon * step_seconds,
                                raw_count,
                            )
                        )
                    )
            values[zone] = resolved
        return values

    def build(
        self,
        forecast: Mapping[str, Sequence[float]],
        *,
        step: int,
        action_time_seconds: int,
        step_seconds: int,
    ) -> ObservationPacket:
        if not self._initialized:
            raise ValueError("policy observation history has not been reset")
        day_fraction = (action_time_seconds % 86400) / 86400.0
        time_values = [
            math.sin(2.0 * math.pi * day_fraction),
            math.cos(2.0 * math.pi * day_fraction),
        ]
        outdoor = [
            float(value)
            for value in forecast[self.profile["global_inputs"]["outdoor_temperature"]][
                step : step + 5
            ]
        ]
        solar = [
            float(value)
            for value in forecast[self.profile["global_inputs"]["solar_irradiance"]][
                step : step + 5
            ]
        ]
        price = [
            float(value)
            for value in forecast[self.profile["global_inputs"]["electricity_price"]][
                step : step + 5
            ]
        ]
        occupancy = self._occupancy_forecasts(
            forecast,
            step=step,
            action_time_seconds=action_time_seconds,
            step_seconds=step_seconds,
        )
        raw: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        columns: list[str] = []

        def add(name: str, values: Sequence[float], bound: tuple[float, float]) -> None:
            raw.extend(values)
            lo, hi = _axis(values, bound)
            lower.extend(lo)
            upper.extend(hi)
            columns.extend(
                name if len(values) == 1 else f"{name}_{index}" for index in range(len(values))
            )

        add("time", time_values, self.time_observation_bounds)
        for zone in self.zone_order:
            add(f"temperature_{zone}", self._temperature_values(zone), (288.15, 308.15))
        add("pmv", [self.current_pmv[zone] for zone in self.zone_order], (-3.0, 3.0))
        if self.action_history_steps == 4 and len(self.zone_order) == 1:
            add(
                "action_zone1",
                self._action_values(self.zone_order[0]),
                self.action_observation_bounds_k,
            )
        elif self.action_history_steps == 1:
            add(
                "last_action",
                [self._action_values(zone)[0] for zone in self.zone_order],
                self.action_observation_bounds_k,
            )
        elif self.action_history_steps == 4:
            for zone in self.zone_order:
                add(
                    f"action_{zone}",
                    self._action_values(zone),
                    self.action_observation_bounds_k,
                )
        else:
            raise ValueError("unsupported policy action-history contract")
        add("power_norm", self._power_values(), (0.0, 1.0))
        add("outdoor_temperature", outdoor, (263.15, 313.15))
        add("solar_irradiance", solar, (0.0, 1200.0))
        add("electricity_price", price, (0.0, 0.2))
        for zone in self.zone_order:
            add(f"occupancy_{zone}", occupancy[zone], (0.0, self.occupancy_upper))
        raw_array = np.asarray(raw, dtype=np.float64)
        if raw_array.shape != (self.expected_dimension,):
            raise ValueError(
                f"policy observation dimension is {raw_array.size}, expected {self.expected_dimension}"
            )
        normalized = symmetric_minmax(
            raw_array,
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )
        local: dict[str, NDArray[np.float32]] = {}
        if self.local_dimension:
            column_index = {name: index for index, name in enumerate(columns)}
            for zone_index, zone in enumerate(self.zone_order):
                action_columns = (
                    [f"last_action_{zone_index}"]
                    if self.action_history_steps == 1
                    else [
                        f"action_{zone}_{index}"
                        for index in range(self.action_history_steps)
                    ]
                )
                zone_columns = [
                    *[f"temperature_{zone}_{index}" for index in range(5)],
                    f"pmv_{zone_index}",
                    *action_columns,
                ]
                shared_columns = [
                    *[f"power_norm_{index}" for index in range(4)],
                    *[f"outdoor_temperature_{index}" for index in range(5)],
                    *[f"solar_irradiance_{index}" for index in range(5)],
                    *[f"electricity_price_{index}" for index in range(5)],
                ]
                local_columns = [
                    "time_0",
                    "time_1",
                    *(
                        [*zone_columns, *shared_columns]
                        if self.local_observation_layout == "zone_then_shared"
                        else [*shared_columns, *zone_columns]
                    ),
                    *[f"occupancy_{zone}_{index}" for index in range(5)],
                ]
                if len(local_columns) != self.local_dimension:
                    raise ValueError("local MAPPO observation dimension is invalid")
                try:
                    local[zone] = normalized[
                        np.asarray([column_index[name] for name in local_columns], dtype=np.int64)
                    ].copy()
                except KeyError as error:
                    raise ValueError("local MAPPO observation column is missing") from error
        return ObservationPacket(raw_array, normalized, local, tuple(columns))
