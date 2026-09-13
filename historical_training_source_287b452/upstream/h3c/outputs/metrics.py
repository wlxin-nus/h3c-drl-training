"""Deterministic run metrics recomputed only from immutable raw artifact streams."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from h3c.outputs.artifacts import PERFORMANCE_COLUMNS
from h3c.outputs.physical_metrics import compute_physical_metrics

PRICE_BOOK_ID = "project-fixed-v1"
USD_PER_MILLION = {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28}
CNY_PER_MILLION = {"cache_hit": 0.02, "cache_miss": 1.0, "output": 2.0}


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} contains a non-object row")
        rows.append(value)
    return rows


def _performance(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != PERFORMANCE_COLUMNS:
            raise ValueError("performance.csv header does not match the registered schema")
        rows = [dict(row) for row in reader]
    if any(
        set(row) != set(PERFORMANCE_COLUMNS) or any(value is None for value in row.values())
        for row in rows
    ):
        raise ValueError("performance.csv row does not match the registered schema")
    return rows


def _usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    if not calls:
        return {
            "available": False,
            "price_book": PRICE_BOOK_ID,
            "reason": "no_usage_rows",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "estimated_cost_usd": 0.0,
            "estimated_cost_cny": 0.0,
        }
    candidate_usage = [row.get("usage") for row in calls]
    if any(
        not isinstance(row, dict) or row.get("available") is not True for row in candidate_usage
    ):
        return {
            "available": False,
            "price_book": PRICE_BOOK_ID,
            "reason": "one_or_more_usage_rows_unavailable",
        }
    usage_rows = cast(list[dict[str, Any]], candidate_usage)
    names = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
    )
    totals = {name: sum(int(row[name]) for row in usage_rows) for name in names}
    usd = (
        totals["cache_hit_tokens"] * USD_PER_MILLION["cache_hit"]
        + totals["cache_miss_tokens"] * USD_PER_MILLION["cache_miss"]
        + totals["completion_tokens"] * USD_PER_MILLION["output"]
    ) / 1_000_000
    cny = (
        totals["cache_hit_tokens"] * CNY_PER_MILLION["cache_hit"]
        + totals["cache_miss_tokens"] * CNY_PER_MILLION["cache_miss"]
        + totals["completion_tokens"] * CNY_PER_MILLION["output"]
    ) / 1_000_000
    return {
        "available": True,
        "price_book": PRICE_BOOK_ID,
        **totals,
        "estimated_cost_usd": usd,
        "estimated_cost_cny": cny,
    }


def compute_run_metrics(run_dir: Path) -> dict[str, Any]:
    """Recompute every reported metric from raw evaluation and model streams."""
    performance = _performance(run_dir / "performance.csv")
    zone_steps = _rows(run_dir / "zone_steps.jsonl")
    updates = _rows(run_dir / "program_updates.jsonl")
    decisions = _rows(run_dir / "hourly_decisions.jsonl")
    calls = _rows(run_dir / "agent_calls.jsonl")
    attempts = _rows(run_dir / "model_request_attempts.jsonl")

    physical_metrics = compute_physical_metrics(
        performance,
        [
            {
                "zone": row["zone"],
                "step": row["step"],
                "final_setpoint_c": row["final_setpoint_c"],
                "effective_occupancy": row["outcome"]["effective_occupancy"],
                "pmv": row["outcome"]["pmv"],
            }
            for row in zone_steps
        ],
    )
    assurance_count = Counter[str]()
    assurance_magnitude: defaultdict[str, float] = defaultdict(float)
    for row in zone_steps:
        audit = row["action_assurance"]
        stages = (
            (
                "comfort_recovery",
                "comfort_recovery_triggered",
                "comfort_recovery_input_setpoint",
                "comfort_recovery_output_setpoint",
            ),
            (
                "setpoint_rate_limit",
                "setpoint_rate_limit_triggered",
                "comfort_recovery_output_setpoint",
                "setpoint_rate_limit_output_setpoint",
            ),
            (
                "actuator_bounds",
                "actuator_bounds_triggered",
                "setpoint_rate_limit_output_setpoint",
                "final_setpoint",
            ),
        )
        for stage, trigger, before, after in stages:
            assurance_count[stage] += int(bool(audit[trigger]))
            assurance_magnitude[stage] += abs(float(audit[after]) - float(audit[before]))

    status_counts = Counter(str(row.get("status")) for row in updates)
    accepted = sum(
        row.get("status") == "accepted" and row.get("patch", {}).get("op") != "no_change"
        for row in updates
    )
    no_change = sum(
        row.get("status") == "accepted" and row.get("patch", {}).get("op") == "no_change"
        for row in updates
    )
    stage_counts = Counter(
        str(stage) for row in updates for stage in row.get("completed_validation_stages", [])
    )
    rejection_codes = Counter(
        str(row["rejection"]["code"])
        for row in updates
        if isinstance(row.get("rejection"), dict) and row["rejection"].get("code")
    )
    budgets = [row["energy_budget"] for row in decisions if "energy_budget" in row]
    route_by_hour = {
        int(row["hour"]): str(row.get("route", {}).get("thinking_mode", "unavailable"))
        for row in decisions
    }
    utilisation_values = [
        float(row["utilisation"]) for row in budgets if row.get("utilisation") is not None
    ]
    orchestration = {
        "hours": len(budgets),
        "fallback_hours": sum(
            row.get("orchestration", {}).get("fallback", {}).get("used") is True
            for row in decisions
        ),
        "site_cap_c_total": sum(float(row.get("site_cap_c", 0.0)) for row in budgets),
        "granted_c_total": sum(float(row.get("granted_c", 0.0)) for row in budgets),
        "used_c_total": sum(float(row.get("used_c", 0.0)) for row in budgets),
        "residual_initial_c_total": sum(
            float(row.get("residual_initial_c", 0.0)) for row in budgets
        ),
        "residual_left_c_total": sum(float(row.get("residual_left_c", 0.0)) for row in budgets),
        "mean_utilisation": (
            sum(utilisation_values) / len(utilisation_values) if utilisation_values else None
        ),
    }
    latencies = [float(row["elapsed_seconds"]) for row in calls]
    if any(not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError("agent latency is invalid")
    attempt_latencies = [float(row["elapsed_seconds"]) for row in attempts]
    if any(not math.isfinite(value) or value < 0 for value in attempt_latencies):
        raise ValueError("model request attempt latency is invalid")

    rationale_by_role: dict[str, list[dict[str, Any]]] = {
        "orchestrator": [
            row["orchestration"]["rationale_telemetry"]
            for row in decisions
            if isinstance(row.get("orchestration"), dict)
            and isinstance(row["orchestration"].get("rationale_telemetry"), dict)
        ],
        "executor": [
            row["rationale_telemetry"]
            for row in updates
            if isinstance(row.get("rationale_telemetry"), dict)
        ],
    }
    rationale_metrics: dict[str, Any] = {}
    for role, telemetry_rows in rationale_by_role.items():
        lengths = [
            int(length)
            for telemetry in telemetry_rows
            for length in telemetry["character_lengths"].values()
        ]
        rationale_metrics[role] = {
            "call_count": len(telemetry_rows),
            "rationale_count": len(lengths),
            "total_characters": sum(lengths),
            "maximum_character_length": max(lengths, default=0),
        }
    return {
        "metrics_schema": "h3c_run_metrics",
        "schema_version": 3,
        **physical_metrics,
        "program_decisions": {
            "accepted": accepted,
            "rejected": status_counts["rejected"],
            "no_change": no_change,
            "model_output_rejected": status_counts["model_output_rejected"],
            "validation_stage_counts": dict(sorted(stage_counts.items())),
            "rejection_code_counts": dict(sorted(rejection_codes.items())),
        },
        "action_assurance": {
            "trigger_counts": dict(sorted(assurance_count.items())),
            "adjustment_magnitude_c": dict(sorted(assurance_magnitude.items())),
        },
        "orchestration": orchestration,
        "rationale_telemetry": {
            "decision_use": "none",
            "by_role": rationale_metrics,
        },
        "model_calls": {
            "count": len(calls),
            "by_role": dict(sorted(Counter(str(row.get("role")) for row in calls).items())),
            "by_thinking_mode": dict(
                sorted(Counter(str(row.get("thinking_mode")) for row in calls).items())
            ),
            "by_route": dict(
                sorted(
                    Counter(
                        route_by_hour.get(int(row["hour"]), "unavailable") for row in calls
                    ).items()
                )
            ),
            "latency_seconds": {
                "total": sum(latencies),
                "mean": sum(latencies) / len(latencies) if latencies else 0.0,
                "maximum": max(latencies, default=0.0),
            },
            "transport": {
                "attempt_count": len(attempts),
                "retry_count": sum(int(row["attempt_number"]) > 1 for row in attempts),
                "recovered_logical_call_count": sum(
                    row.get("outcome") == "response_received" and int(row["attempt_number"]) > 1
                    for row in attempts
                ),
                "failed_attempt_count": sum(
                    row.get("outcome") == "request_failed" for row in attempts
                ),
                "terminal_failed_logical_call_count": sum(
                    row.get("outcome") == "request_failed" and row.get("will_retry") is False
                    for row in attempts
                ),
                "confirmed_response_count": sum(
                    row.get("provider_charge_status") == "confirmed_response_usage_recorded"
                    for row in attempts
                ),
                "unknown_provider_charge_attempt_count": sum(
                    row.get("provider_charge_status") == "unknown_after_request_failure"
                    for row in attempts
                ),
                "attempt_latency_seconds": {
                    "total": sum(attempt_latencies),
                    "mean": (
                        sum(attempt_latencies) / len(attempt_latencies)
                        if attempt_latencies
                        else 0.0
                    ),
                    "maximum": max(attempt_latencies, default=0.0),
                },
            },
            "usage": _usage(calls),
        },
    }
