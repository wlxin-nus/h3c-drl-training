"""Shared physical initialization and evaluation-boundary protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from h3c.runtime.occupancy import resolve_missing_occupancy_values

if TYPE_CHECKING:
    from h3c.runtime.comfort import ComfortModel


class PhysicalClient(Protocol):
    test_id: str | None

    def initialize(
        self, testcase: str, start_time_seconds: int, warmup_period_seconds: int
    ) -> dict[str, Any]: ...

    def forecast(
        self, points: Sequence[str], horizon_seconds: int, interval_seconds: int
    ) -> dict[str, list[float | None]]: ...

    def advance(self, controls: Mapping[str, float]) -> dict[str, Any]: ...

    def stop(self) -> None: ...


class InitializationArtifactSink(Protocol):
    """Minimal artifact surface required by physical initialization."""

    def append_jsonl(self, name: str, value: Mapping[str, Any]) -> None: ...


@dataclass
class EvaluationBoundaryState:
    state: dict[str, Any]
    test_id: str
    last_setpoint_c: dict[str, float]
    last_pmv: dict[str, float]
    last_occupancy: dict[str, float]
    comfort: ComfortModel
    occupancy_missing_value_resolution_count: int
    conditioning_prefix_identity: str
    evaluation_boundary_identity: str


def forecast_points(profile: Mapping[str, Any]) -> list[str]:
    global_inputs = profile["global_inputs"]
    points = [
        global_inputs["outdoor_temperature"],
        global_inputs["solar_irradiance"],
        global_inputs["electricity_price"],
    ]
    for zone in profile["zones"].values():
        point = zone["occupancy_forecast"]
        if point not in points:
            points.append(point)
    return points


def validate_forecast(
    forecast: Mapping[str, Sequence[object]], points: Sequence[str], minimum_length: int
) -> None:
    missing = sorted(set(points) - set(forecast))
    if missing:
        raise ValueError(f"forecast bundle is missing configured points: {missing}")
    for point in points:
        values = forecast[point]
        if len(values) < minimum_length:
            raise ValueError(
                f"forecast point {point} has {len(values)} values; "
                f"at least {minimum_length} are required"
            )
        for index, value in enumerate(values):
            finite = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            )
            if not finite:
                raise ValueError(f"forecast point {point} at index {index} is not a finite number")


def resolve_forecast_missing_occupancy(
    profile: Mapping[str, Any],
    forecast: Mapping[str, Sequence[float | None]],
    points: Sequence[str],
    minimum_length: int,
    *,
    forecast_phase: str,
    start_time_seconds: int,
    step_seconds: int,
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    missing = sorted(set(points) - set(forecast))
    if missing:
        raise ValueError(f"forecast bundle is missing configured points: {missing}")
    candidate: dict[str, list[object]] = {point: list(forecast[point]) for point in points}
    events: list[dict[str, Any]] = []
    occupancy_points = dict.fromkeys(
        zone["occupancy_forecast"] for zone in profile["zones"].values()
    )
    for point in occupancy_points:
        values, point_events = resolve_missing_occupancy_values(
            profile["occupancy"],
            forecast[point],
            start_time_seconds=start_time_seconds,
            step_seconds=step_seconds,
        )
        resolved_values: list[object] = list(values)
        candidate[point] = resolved_values
        events.extend(
            {
                "phase": "occupancy_forecast_missing_value_resolution",
                "forecast_phase": forecast_phase,
                "point": point,
                **event,
            }
            for event in point_events
        )
    validate_forecast(candidate, points, minimum_length)
    normalized: dict[str, list[float]] = {}
    for point in points:
        normalized[point] = []
        for value in candidate[point]:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"forecast point {point} is not numeric after validation")
            normalized[point].append(float(value))
    return normalized, events


def build_forecast_evidence(
    points: Sequence[str],
    source: Mapping[str, Sequence[float | None]],
    resolved: Mapping[str, Sequence[float]],
    *,
    start_time_seconds: int,
    step_seconds: int,
) -> dict[str, Any]:
    """Serialize the exact raw and resolved forecast used by a run."""
    return {
        "artifact_schema": "h3c_forecast_inputs",
        "schema_version": 1,
        "start_time_seconds": int(start_time_seconds),
        "step_seconds": int(step_seconds),
        "points": list(points),
        "source": {point: list(source[point]) for point in points},
        "resolved": {point: [float(value) for value in resolved[point]] for point in points},
    }


def reconstruct_forecast_evidence(
    profile: Mapping[str, Any],
    evidence: Mapping[str, Any],
    minimum_length: int,
    *,
    forecast_phase: str,
    start_time_seconds: int,
    step_seconds: int,
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    """Validate stored forecast inputs and reconstruct their resolution events."""
    expected_fields = {
        "artifact_schema",
        "schema_version",
        "start_time_seconds",
        "step_seconds",
        "points",
        "source",
        "resolved",
    }
    points = forecast_points(profile)
    if (
        set(evidence) != expected_fields
        or evidence.get("artifact_schema") != "h3c_forecast_inputs"
        or evidence.get("schema_version") != 1
        or evidence.get("start_time_seconds") != start_time_seconds
        or evidence.get("step_seconds") != step_seconds
        or evidence.get("points") != points
        or not isinstance(evidence.get("source"), Mapping)
        or set(evidence["source"]) != set(points)
        or not isinstance(evidence.get("resolved"), Mapping)
        or set(evidence["resolved"]) != set(points)
    ):
        raise ValueError("forecast evidence contract is invalid")
    resolved, events = resolve_forecast_missing_occupancy(
        profile,
        evidence["source"],
        points,
        minimum_length,
        forecast_phase=forecast_phase,
        start_time_seconds=start_time_seconds,
        step_seconds=step_seconds,
    )
    if evidence["resolved"] != resolved:
        raise ValueError("resolved forecast evidence does not match the source forecast")
    return resolved, events


def control_input(profile: Mapping[str, Any], setpoints_c: Mapping[str, float]) -> dict[str, float]:
    zones = profile["zones"]
    if set(setpoints_c) != set(zones):
        raise ValueError("setpoints must cover exactly the configured zones")
    controls = {key: float(value) for key, value in profile["static_controls"].items()}
    for zone, value in setpoints_c.items():
        controls[zones[zone]["cooling_setpoint_actuator"]] = float(value) + 273.15
    return controls


def _time(state: Mapping[str, Any], expected: int) -> int:
    value = state.get("time")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("physical state time is missing")
    if not math.isfinite(float(value)) or int(value) != expected:
        raise ValueError("physical state time diverged from the registered timeline")
    return int(value)


def _power(profile: Mapping[str, Any], state: Mapping[str, Any]) -> float:
    total = 0.0
    for point in profile["global_inputs"]["power_meters"]:
        if point not in state:
            raise ValueError(f"physical state is missing power meter: {point}")
        raw = state[point]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"power meter {point} is not numeric")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"power meter {point} is non-finite")
        total += value
    return total


def canonical_evidence_bytes(value: Any) -> bytes:
    """Serialize physical evidence with the repository's exact identity contract."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def physical_evidence_identity(value: Any) -> str:
    """Return the SHA-256 identity of canonical physical evidence."""

    return hashlib.sha256(canonical_evidence_bytes(value)).hexdigest()


def _boundary_evidence(
    state: Mapping[str, Any],
    last_setpoint: Mapping[str, float],
    last_pmv: Mapping[str, float],
    last_occupancy: Mapping[str, float],
    comfort: ComfortModel,
) -> dict[str, Any]:
    return {
        "physical_state": dict(state),
        "last_setpoint_c": dict(last_setpoint),
        "last_pmv": dict(last_pmv),
        "last_occupancy": dict(last_occupancy),
        "clothing_insulation": comfort.clothing_insulation,
    }


def _temperature_c(profile: Mapping[str, Any], state: Mapping[str, Any], zone: str) -> float:
    point = profile["zones"][zone]["temperature_sensor"]
    value = float(state[point]) - 273.15
    if not math.isfinite(value):
        raise ValueError("zone temperature is non-finite")
    return value


def initialize_evaluation_boundary(
    client: PhysicalClient,
    profile: Mapping[str, Any],
    artifacts: InitializationArtifactSink,
    *,
    evaluation_start_seconds: int | None = None,
    on_initialized: Callable[[str], None] | None = None,
) -> EvaluationBoundaryState:
    from h3c.runtime.comfort import ComfortModel
    """Initialize directly at the registered evaluation boundary.

    BOPTEST owns the internal warm-up. No controlled prefix is executed, and the
    application-side controller state starts from the configured deterministic seed.
    """
    protocol = profile["protocol"]
    if protocol["initialization_mode"] != "evaluation_start_internal_warmup":
        raise ValueError("unsupported physical initialization mode")
    evaluation_start = (
        int(profile["evaluation_start_day"]) * 86400
        if evaluation_start_seconds is None
        else evaluation_start_seconds
    )
    if isinstance(evaluation_start, bool) or not isinstance(evaluation_start, int):
        raise ValueError("evaluation start must be an integer number of seconds")
    warmup_seconds = int(protocol["internal_warmup_days"]) * 86400
    state = client.initialize(profile["testcase"], evaluation_start, warmup_seconds)
    _time(state, evaluation_start)
    test_id = client.test_id
    if not isinstance(test_id, str) or not test_id:
        raise ValueError("physical initialize did not produce a test id")
    artifacts.append_jsonl(
        "timing.jsonl",
        {
            "phase": "physical_lifecycle",
            "event": "initialized",
            "time_seconds": evaluation_start,
            "warmup_period_seconds": warmup_seconds,
            "test_id": test_id,
        },
    )
    if on_initialized is not None:
        on_initialized(test_id)
    zones = tuple(profile["zones"])
    last_setpoint = {zone: float(protocol["initial_setpoint_c"]) for zone in zones}
    last_pmv = {zone: 0.0 for zone in zones}
    last_occupancy = {zone: 0.0 for zone in zones}
    comfort = ComfortModel(profile["comfort"])
    _time(state, evaluation_start)
    boundary = _boundary_evidence(state, last_setpoint, last_pmv, last_occupancy, comfort)
    boundary_identity = physical_evidence_identity(boundary)
    artifacts.append_jsonl(
        "timing.jsonl",
        {
            "phase": "evaluation_boundary",
            "boundary": boundary,
            "evaluation_boundary_identity": boundary_identity,
            "test_id": test_id,
        },
    )
    return EvaluationBoundaryState(
        state=state,
        test_id=test_id,
        last_setpoint_c=last_setpoint,
        last_pmv=last_pmv,
        last_occupancy=last_occupancy,
        comfort=comfort,
        occupancy_missing_value_resolution_count=0,
        conditioning_prefix_identity=hashlib.sha256(b"").hexdigest(),
        evaluation_boundary_identity=boundary_identity,
    )


def site_power(profile: Mapping[str, Any], state: Mapping[str, Any]) -> float:
    return _power(profile, state)


def zone_temperature_c(profile: Mapping[str, Any], state: Mapping[str, Any], zone: str) -> float:
    return _temperature_c(profile, state, zone)


def require_time(state: Mapping[str, Any], expected: int) -> int:
    return _time(state, expected)
