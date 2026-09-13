from __future__ import annotations

import numpy as np

from drl_multiseed.config import OBSERVATION_CONTRACT_ID, TASKS, WANDB_PROJECT
from drl_multiseed.observation_contract import (
    load_refined_contracts,
    refined_model_entry,
)
from h3c.experiments.profiles import load_profile
from h3c_baselines.policies.observation_contracts import PolicyObservationBuilder


CASE_NAMES = {"sz_air": "SZ_Air", "mz_air": "MZ_Air", "mz_hydro": "MZ_Hydro"}


def _state(profile: dict, increment: float = 0.0) -> dict[str, float]:
    state: dict[str, float] = {"time": float(profile["evaluation_start_day"] * 86400)}
    for index, zone in enumerate(profile["zones"].values()):
        state[str(zone["temperature_sensor"])] = 294.15 + index + increment
    for meter in profile["global_inputs"]["power_meters"]:
        state[str(meter)] = 0.0
    return state


def _forecast(profile: dict, length: int = 16) -> dict[str, list[float]]:
    inputs = profile["global_inputs"]
    result = {
        str(inputs["outdoor_temperature"]): [280.0 + index for index in range(length)],
        str(inputs["solar_irradiance"]): [100.0 + index for index in range(length)],
        str(inputs["electricity_price"]): [0.01 * index for index in range(length)],
    }
    for zone in profile["zones"].values():
        result[str(zone["occupancy_forecast"])] = [1.0] * length
    return result


def test_all_tasks_share_the_refined_temporal_contract() -> None:
    contracts = load_refined_contracts()
    temporal = contracts["temporal_contract"]
    assert contracts["contract_id"] == OBSERVATION_CONTRACT_ID == "refined-temporal-v1"
    assert WANDB_PROJECT == "h3c-drl-multiseed-refine-v2"
    assert temporal["temperature_offsets"] == [0, -1, -2, -3, -4]
    assert temporal["action_offsets"] == [-1, -2, -3, -4]
    assert temporal["power_offsets"] == [-1, -2, -3, -4]
    assert temporal["forecast_offsets"] == [0, 1, 2, 3, 4]
    assert temporal["temperature_missing"] == "repeat_earliest"
    assert temporal["action_missing"] == "fixed_initial_setpoint"
    assert temporal["power_missing"] == "zero"
    for task in TASKS:
        entry = refined_model_entry(task)
        assert entry["temperature_past_offset"] == 1
        assert entry["action_past_offset"] == 0
        assert entry["power_past_offset"] == 0
        assert entry["temperature_missing"] == "repeat_earliest"
        assert entry["action_history_steps"] == 4
        assert entry["time_observation_bounds"] == [-1.0, 1.0]
        assert entry["action_observation_bounds_k"] == [293.15, 303.15]
        assert entry["occupancy_encoding"] == "effective_count"
        if entry["algorithm"] == "mappo":
            assert entry["local_observation_layout"] == "zone_then_shared"


def test_dimensions_follow_one_canonical_formula() -> None:
    # time(2) + per-zone[temp(5)+pmv(1)+actions(4)+occupancy forecast(5)]
    # + shared[power(4)+weather/solar/price forecasts(15)] = 21 + 15*zones.
    for key, spec in TASKS.items():
        zones = len(refined_model_entry(key)["policy_zone_order"])
        assert spec.observation_dim == 21 + 15 * zones
        if spec.algorithm == "mappo":
            assert spec.local_observation_dim == 36


def test_temperature_padding_and_all_temporal_indices_are_exact() -> None:
    for key, spec in TASKS.items():
        profile = load_profile(CASE_NAMES[spec.case_key])
        entry = refined_model_entry(key)
        builder = PolicyObservationBuilder(profile, entry)
        initial = _state(profile)
        builder.reset(initial)

        for zone in entry["policy_zone_order"]:
            sensor = profile["zones"][zone]["temperature_sensor"]
            initial_k = float(initial[sensor])
            assert builder._temperature_values(zone) == [initial_k] * 5

        maximum_power = float(profile["performance"]["maximum_power_w"])
        for step in range(1, 5):
            setpoints = {
                zone: 20.0 + step + zone_index * 0.1
                for zone_index, zone in enumerate(entry["policy_zone_order"])
            }
            builder.update(
                _state(profile, float(step)),
                setpoints,
                dict.fromkeys(entry["policy_zone_order"], 0.0),
                float(step * 100),
            )

        for zone_index, zone in enumerate(entry["policy_zone_order"]):
            sensor = profile["zones"][zone]["temperature_sensor"]
            base = float(initial[sensor])
            assert np.allclose(
                builder._temperature_values(zone),
                [base + 4, base + 3, base + 2, base + 1, base],
            )
            expected_actions = [
                297.15 + zone_index * 0.1,
                296.15 + zone_index * 0.1,
                295.15 + zone_index * 0.1,
                294.15 + zone_index * 0.1,
            ]
            assert np.allclose(builder._action_values(zone), expected_actions)
        assert np.allclose(
            builder._power_values(),
            [400.0 / maximum_power, 300.0 / maximum_power,
             200.0 / maximum_power, 100.0 / maximum_power],
        )

        packet = builder.build(
            _forecast(profile), step=2,
            action_time_seconds=int(profile["evaluation_start_day"]) * 86400 + 2 * 900,
            step_seconds=900,
        )
        assert packet.raw.shape == (spec.observation_dim,)
        for name, expected in (
            ("outdoor_temperature", [282.0, 283.0, 284.0, 285.0, 286.0]),
            ("solar_irradiance", [102.0, 103.0, 104.0, 105.0, 106.0]),
            ("electricity_price", [0.02, 0.03, 0.04, 0.05, 0.06]),
        ):
            indices = [packet.columns.index(f"{name}_{index}") for index in range(5)]
            assert np.allclose(packet.raw[indices], expected)
        if spec.algorithm == "mappo":
            assert set(packet.local_normalized) == set(entry["policy_zone_order"])
            assert all(value.shape == (36,) for value in packet.local_normalized.values())


def test_ppo_and_mappo_share_the_same_case_observation_semantics() -> None:
    for ppo_key, mappo_key in (
        ("mz_air_ppo", "mz_air_mappo"),
        ("mz_hydro_ppo", "mz_hydro_mappo"),
    ):
        ppo = refined_model_entry(ppo_key)
        mappo = refined_model_entry(mappo_key)
        ignored = {"algorithm", "controller", "local_observation_dimension"}
        assert {
            key: value for key, value in ppo.items() if key not in ignored
        } == {
            key: value for key, value in mappo.items() if key not in ignored
        }
