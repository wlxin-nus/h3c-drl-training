from __future__ import annotations

import pytest

from drl_multiseed.environment import training_reward
from drl_multiseed.occupancy import effective_count
from drl_multiseed.profiles import load_profile
from drl_multiseed.protocol import control_input, setpoint_from_action


def test_air_and_hydronic_action_mappings_are_exact() -> None:
    assert [setpoint_from_action("fixed_25", 0.0, action) for action in (-1, 0, 1)] == [
        20.0,
        25.0,
        30.0,
    ]
    assert [setpoint_from_action("occupancy_25_30", 1.0, action) for action in (-1, 0, 1)] == [
        20.0,
        25.0,
        30.0,
    ]
    assert [setpoint_from_action("occupancy_25_30", 0.0, action) for action in (-1, 0, 1)] == [
        25.0,
        30.0,
        30.0,
    ]
    assert setpoint_from_action("fixed_25", 1.0, 4.0) == 30.0


def test_reward_decomposition_uses_registered_deadband_and_zone_average() -> None:
    objective = load_profile("mz_hydro")["objective"]
    reward, components = training_reward(
        cost=2.0,
        pmv=[0.4, -0.7],
        occupancy=[1.0, 1.0],
        setpoints=[24.0, 27.0],
        previous=[25.0, 25.0],
        objective=objective,
        comfort_threshold=0.5,
    )
    assert components["rew_energy"] == pytest.approx(3.0)
    assert components["rew_comfort"] == pytest.approx(4.0)
    assert components["rew_smooth"] == pytest.approx(0.026339925)
    assert reward == pytest.approx(-sum(components.values()))


def test_mz_air_occupancy_requires_count_and_official_window() -> None:
    policy = load_profile("mz_air")["occupancy"]
    day = 199 * 86_400
    assert effective_count(policy, day + 5 * 3600, 3.0) == 0.0
    assert effective_count(policy, day + 7 * 3600, 3.0) == 3.0
    assert effective_count(policy, day + 7 * 3600, 0.0) == 0.0
    assert effective_count(policy, day + 19 * 3600, 3.0) == 0.0


def test_control_payload_contains_only_registered_points() -> None:
    profile = load_profile("mz_hydro")
    payload = control_input(profile, {"NZ": 25.0, "SZ": 30.0})
    expected = set(profile["static_controls"]) | {
        zone["cooling_setpoint_actuator"] for zone in profile["zones"].values()
    }
    assert set(payload) == expected
    assert payload["bms_oveTZonSetMaxNz_u"] == pytest.approx(298.15)
    assert payload["bms_oveTZonSetMaxSz_u"] == pytest.approx(303.15)
