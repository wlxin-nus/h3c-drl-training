"""Shared energy-intensive setpoint-movement allocation and settlement."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from h3c.agents.contracts import DEFAULT_PER_ZONE_RESERVED_CAP_C

EDGE_IDENTIFIER = re.compile(r"^ce_[0-9a-f]{8}$")
ALLOCATION_RATIONALE_ERROR = "allocation rationale must cover every zone with a nonempty string"
BUDGET_ABS_TOLERANCE = 1e-9


@dataclass(frozen=True)
class BudgetRejection:
    code: str
    message: str


def site_cap_max(zones: Sequence[str], site_budget_fraction: float = 0.5) -> float:
    if (
        isinstance(site_budget_fraction, bool)
        or not isinstance(site_budget_fraction, (int, float))
        or not math.isfinite(float(site_budget_fraction))
        or not 0.0 < float(site_budget_fraction) <= 1.0
    ):
        raise ValueError("site budget fraction must satisfy zero < value <= one")
    return DEFAULT_PER_ZONE_RESERVED_CAP_C * len(zones) * float(site_budget_fraction)


def validate_allocation(
    allocation: Mapping[str, Any],
    zones: Sequence[str],
    *,
    causal_enabled: bool = True,
    allowed_causal_edge_ids: set[str] | None = None,
    site_causal_edge_ids: set[str] | None = None,
    expected_site_cap_c: float | None = None,
    expected_per_zone_reserved_cap_c: float = DEFAULT_PER_ZONE_RESERVED_CAP_C,
) -> None:
    required = {"site_cap_c", "zone_budgets_c", "priority", "rationale_per_zone"}
    if causal_enabled:
        required.add("causal_edge_ids")
    if not isinstance(allocation, Mapping) or set(allocation) != required:
        raise ValueError("allocation fields do not match the exact contract")
    cap = allocation["site_cap_c"]
    budgets = allocation["zone_budgets_c"]
    priority = allocation["priority"]
    if (
        isinstance(cap, bool)
        or not isinstance(cap, (int, float))
        or not 0 <= cap <= site_cap_max(zones)
    ):
        raise ValueError("site allocation is outside its bound")
    if expected_site_cap_c is not None and not math.isclose(
        float(cap),
        float(expected_site_cap_c),
        rel_tol=0.0,
        abs_tol=BUDGET_ABS_TOLERANCE,
    ):
        raise ValueError("site allocation must equal the supplied site cap")
    if (
        isinstance(expected_per_zone_reserved_cap_c, bool)
        or not isinstance(expected_per_zone_reserved_cap_c, (int, float))
        or not math.isfinite(float(expected_per_zone_reserved_cap_c))
        or float(expected_per_zone_reserved_cap_c) < 0.0
    ):
        raise ValueError("per-zone reserved cap must be finite and nonnegative")
    if not isinstance(budgets, Mapping) or set(budgets) != set(zones):
        raise ValueError("zone allocations must cover exactly the configured zones")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= float(expected_per_zone_reserved_cap_c)
        for value in budgets.values()
    ):
        raise ValueError("zone allocation is outside its bound")
    if sum(float(value) for value in budgets.values()) > float(cap) + BUDGET_ABS_TOLERANCE:
        raise ValueError("zone allocations exceed the site cap")
    if (
        not isinstance(priority, list)
        or len(priority) != len(zones)
        or any(not isinstance(zone, str) for zone in priority)
        or len(set(priority)) != len(priority)
        or set(priority) != set(zones)
    ):
        raise ValueError("priority must be a permutation of configured zones")
    rationale = allocation["rationale_per_zone"]
    if (
        not isinstance(rationale, Mapping)
        or set(rationale) != set(zones)
        or any(not isinstance(value, str) or not value.strip() for value in rationale.values())
    ):
        raise ValueError(ALLOCATION_RATIONALE_ERROR)
    if causal_enabled:
        identifiers = allocation["causal_edge_ids"]
        if (
            not isinstance(identifiers, list)
            or not identifiers
            or len(identifiers) != len(set(identifiers))
            or any(
                not isinstance(identifier, str) or EDGE_IDENTIFIER.fullmatch(identifier) is None
                for identifier in identifiers
            )
        ):
            raise ValueError("allocation must cite confirmed causal edge identifiers")
        cited = set(identifiers)
        if allowed_causal_edge_ids is not None and not cited <= allowed_causal_edge_ids:
            raise ValueError("allocation cites an edge outside the resolved graph")
        if site_causal_edge_ids is not None and not cited & site_causal_edge_ids:
            raise ValueError("allocation must cite a visible edge with target=power_meters")


def allocation_fallback_audit(
    *, used: bool, reason: str | None = None, source: str | None = None
) -> dict[str, Any]:
    """Build the exact fallback telemetry surface from its execution status."""
    if not isinstance(used, bool):
        raise ValueError("fallback used must be boolean")
    if used:
        if not isinstance(reason, str) or not reason:
            raise ValueError("used fallback requires a nonempty rejection reason")
        if source not in {"previous_valid_allocation", "equal_split_current_zones"}:
            raise ValueError("used fallback requires a registered allocation source")
    elif reason is not None or source is not None:
        raise ValueError("unused fallback cannot carry a reason or source")
    return {
        "used": used,
        "reason": reason,
        "source": source,
        "validated": used,
    }


def validated_fallback_allocation(
    zones: Sequence[str],
    previous_allocation: Mapping[str, Any] | None,
    *,
    site_cap_c: float,
    causal_enabled: bool,
    causal_edge_ids: Sequence[str] | None,
    allowed_causal_edge_ids: set[str] | None,
    site_causal_edge_ids: set[str] | None,
    per_zone_reserved_cap_c: float = DEFAULT_PER_ZONE_RESERVED_CAP_C,
) -> tuple[dict[str, Any], str]:
    """Return a previous valid allocation or a validated deterministic equal split."""
    if previous_allocation is not None:
        previous = copy.deepcopy(dict(previous_allocation))
        try:
            validate_allocation(
                previous,
                zones,
                causal_enabled=causal_enabled,
                allowed_causal_edge_ids=allowed_causal_edge_ids,
                site_causal_edge_ids=site_causal_edge_ids,
                expected_per_zone_reserved_cap_c=per_zone_reserved_cap_c,
            )
        except ValueError:
            pass
        else:
            return previous, "previous_valid_allocation"

    ordered_zones = list(zones)
    if not ordered_zones:
        raise ValueError("fallback allocation requires at least one zone")
    share = float(site_cap_c) / len(ordered_zones)
    allocation: dict[str, Any] = {
        "site_cap_c": float(site_cap_c),
        "zone_budgets_c": {zone: share for zone in ordered_zones},
        "priority": ordered_zones,
        "rationale_per_zone": {
            zone: "validated deterministic equal-share fallback" for zone in ordered_zones
        },
    }
    if causal_enabled:
        allocation["causal_edge_ids"] = list(causal_edge_ids or ())
    validate_allocation(
        allocation,
        ordered_zones,
        causal_enabled=causal_enabled,
        allowed_causal_edge_ids=allowed_causal_edge_ids,
        site_causal_edge_ids=site_causal_edge_ids,
        expected_site_cap_c=site_cap_c,
        expected_per_zone_reserved_cap_c=per_zone_reserved_cap_c,
    )
    return allocation, "equal_split_current_zones"


class BudgetLedger:
    """One-hour ledger; unused allowance never carries into another hour."""

    def __init__(
        self,
        allocation: Mapping[str, Any],
        zones: Sequence[str],
        *,
        per_zone_reserved_cap_c: float = DEFAULT_PER_ZONE_RESERVED_CAP_C,
    ) -> None:
        validate_allocation(
            allocation,
            zones,
            causal_enabled="causal_edge_ids" in allocation,
            expected_per_zone_reserved_cap_c=per_zone_reserved_cap_c,
        )
        self.zones = list(zones)
        self.cap = float(allocation["site_cap_c"])
        self.granted = {zone: float(allocation["zone_budgets_c"][zone]) for zone in zones}
        self.used = {zone: 0.0 for zone in zones}
        self.residual = max(0.0, self.cap - sum(self.granted.values()))
        self.residual_initial = self.residual
        self.residual_used = {zone: 0.0 for zone in zones}
        self.priority = list(allocation["priority"])
        self.events: list[dict[str, Any]] = []

    def remaining(self, zone: str) -> float:
        return max(0.0, self.granted[zone] - self.used[zone])

    def snapshot(self, zone: str) -> dict[str, Any]:
        return {
            "remaining_c": round(self.remaining(zone), 4),
            "site_residual_c": round(self.residual, 4),
            "priority_rank": self.priority.index(zone) + 1,
        }

    def energy_budget_validation(
        self, zone: str, amount_c: float, *, parameter: str, step: int
    ) -> BudgetRejection | None:
        amount = max(0.0, float(amount_c))
        if amount == 0:
            self.events.append(
                {
                    "step": step,
                    "zone": zone,
                    "parameter": parameter,
                    "amount_c": 0.0,
                    "source": "free_non_intensive",
                }
            )
            return None
        own = self.remaining(zone)
        if amount <= own + 1e-9:
            self.used[zone] += amount
            source = "zone_budget"
        else:
            needed = amount - own
            if needed > self.residual + 1e-9:
                self.events.append(
                    {
                        "step": step,
                        "zone": zone,
                        "parameter": parameter,
                        "amount_c": round(amount, 4),
                        "source": "rejected",
                    }
                )
                return BudgetRejection(
                    "energy_budget_exhausted",
                    f"patch needs {amount:.3f} C but zone and shared allowance are insufficient",
                )
            self.used[zone] += own
            self.residual -= needed
            self.residual_used[zone] += needed
            source = "zone_budget_and_shared_residual"
        self.events.append(
            {
                "step": step,
                "zone": zone,
                "parameter": parameter,
                "amount_c": round(amount, 4),
                "source": source,
            }
        )
        return None

    def utilisation(self) -> dict[str, Any]:
        granted = sum(self.granted.values())
        used = sum(self.used.values())
        return {
            "granted_c": round(granted, 4),
            "used_c": round(used, 4),
            "reserved_allowance_by_zone_c": {
                zone: round(value, 4) for zone, value in self.granted.items()
            },
            "reserved_consumption_by_zone_c": {
                zone: round(value, 4) for zone, value in self.used.items()
            },
            "utilisation": round(used / granted, 4) if granted else None,
            "site_cap_c": round(self.cap, 4),
            "residual_initial_c": round(self.residual_initial, 4),
            "residual_left_c": round(self.residual, 4),
            "residual_used_by": {
                zone: round(value, 4) for zone, value in self.residual_used.items() if value > 0
            },
        }
