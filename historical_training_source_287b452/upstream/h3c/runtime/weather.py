"""Four-step raw weather input, summaries, and weather-condition values."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _finite(values: Sequence[Any], name: str) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} forecast must contain finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} forecast must contain finite numbers")
        result.append(number)
    return result


def weather_view(
    outdoor_temperature_kelvin: Sequence[Any], solar_irradiance: Sequence[Any]
) -> dict[str, Any]:
    """Return raw next four steps and the canonical three one-hour summaries."""
    temperatures = _finite(outdoor_temperature_kelvin, "outdoor temperature")
    solar = _finite(solar_irradiance, "solar irradiance")
    if len(temperatures) < 5 or len(solar) < 5:
        raise ValueError("weather forecast must contain current plus four future steps")
    temperatures_c = [round(value - 273.15, 2) for value in temperatures[:5]]
    solar_values = [round(value, 1) for value in solar[:5]]
    future_solar = solar_values[1:5]
    return {
        "outdoor_temp_change_next_1h_c": round(temperatures_c[4] - temperatures_c[0], 2),
        "solar_irr_max_next_1h_w_m2": round(max(future_solar), 1),
        "solar_irr_mean_next_1h_w_m2": round(sum(future_solar) / 4.0, 1),
        "weather_next_steps": [
            {
                "step_ahead": offset,
                "outdoor_temp_c": temperatures_c[offset],
                "solar_irr": solar_values[offset],
            }
            for offset in range(1, 5)
        ],
    }


def weather_condition_inputs(view: Mapping[str, Any]) -> dict[str, float]:
    fields = (
        "outdoor_temp_change_next_1h_c",
        "solar_irr_max_next_1h_w_m2",
        "solar_irr_mean_next_1h_w_m2",
    )
    if any(field not in view for field in fields):
        raise ValueError("weather summaries are incomplete")
    return {field: float(view[field]) for field in fields}
