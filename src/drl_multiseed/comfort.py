"""Fanger PMV calculation shared by every training environment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pythermalcomfort.models import pmv_ppd_iso  # type: ignore[import-untyped]


class ComfortModel:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.clothing_insulation = float(config["summer_clothing_insulation"])
        self.previous_day = -1

    def update_clothing(self, time_seconds: float, outdoor_daily_mean_c: float) -> None:
        if not self.config["dynamic_clothing"]:
            return
        day = int(time_seconds // 86_400)
        if day == self.previous_day:
            return
        low = float(self.config["clothing_transition_low_c"])
        high = float(self.config["clothing_transition_high_c"])
        winter = float(self.config["winter_clothing_insulation"])
        summer = float(self.config["summer_clothing_insulation"])
        if outdoor_daily_mean_c < low:
            self.clothing_insulation = winter
        elif outdoor_daily_mean_c > high:
            self.clothing_insulation = summer
        else:
            ratio = (outdoor_daily_mean_c - low) / (high - low)
            self.clothing_insulation = winter + ratio * (summer - winter)
        self.previous_day = day

    def pmv(self, air_temperature_c: float) -> float:
        result = pmv_ppd_iso(
            tdb=float(air_temperature_c),
            tr=float(air_temperature_c),
            vr=float(self.config["air_velocity_m_s"]),
            rh=float(self.config["relative_humidity_percent"]),
            met=float(self.config["metabolic_rate"]),
            clo=self.clothing_insulation,
            limit_inputs=False,
        )
        value = result["pmv"] if isinstance(result, dict) else result.pmv
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("comfort model returned a non-finite PMV")
        return max(-3.0, min(3.0, number))
