"""Human-in-the-loop graph preparation, confirmation, and derivation."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from h3c.causal.graph import ConfirmedGraph, derive_variant, validate_graph

PROPOSAL_FIELDS = {
    "graph_schema",
    "schema_version",
    "profile",
    "workflow_status",
    "sources",
    "nodes",
    "edges",
    "zones",
    "adjacency",
}


class GraphWorkflowError(ValueError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GraphWorkflowError(f"graph document cannot be read: {path}") from error
    if not isinstance(value, dict):
        raise GraphWorkflowError("graph document root must be an object")
    return value


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def prepare_proposal(profile: str, zones: Sequence[str], sources: Sequence[str]) -> dict[str, Any]:
    if not profile or not zones or not sources or any(not item for item in (*zones, *sources)):
        raise GraphWorkflowError("profile, zones, and evidence sources are required")
    return {
        "graph_schema": "causal_graph_proposal",
        "schema_version": 1,
        "profile": profile,
        "workflow_status": "prepared",
        "sources": list(sources),
        "nodes": [
            {"id": "zone_temp", "scope": "zone"},
            {"id": "cooling_setpoint", "scope": "zone"},
            {"id": "occupancy", "scope": "zone"},
            {"id": "power_meters", "scope": "site"},
            {"id": "outdoor_temp", "scope": "site"},
            {"id": "solar_irr", "scope": "site"},
            {"id": "electricity_price", "scope": "site"},
        ],
        "edges": [],
        "zones": list(zones),
        "adjacency": [],
    }


def validate_proposal(raw: Mapping[str, Any], *, required_status: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != PROPOSAL_FIELDS:
        raise GraphWorkflowError("graph proposal fields do not match the schema")
    if (
        raw["graph_schema"] != "causal_graph_proposal"
        or raw["schema_version"] != 1
        or raw["workflow_status"] != required_status
    ):
        raise GraphWorkflowError("graph proposal schema or workflow status is invalid")
    candidate = {
        key: copy.deepcopy(raw[key])
        for key in ("profile", "sources", "nodes", "edges", "zones", "adjacency")
    }
    candidate.update(
        {
            "graph_schema": "confirmed_causal_graph",
            "schema_version": 1,
            "confirmation": {
                "status": "confirmed",
                "reviewer": "workflow validation placeholder",
                "date": "2000-01-01",
            },
        }
    )
    try:
        validate_graph(candidate)
    except ValueError as error:
        raise GraphWorkflowError(str(error)) from error
    return copy.deepcopy(dict(raw))


def propose_graph(raw: Mapping[str, Any]) -> dict[str, Any]:
    prepared = validate_proposal(raw, required_status="prepared")
    if not prepared["edges"]:
        raise GraphWorkflowError("a proposal requires at least one declared edge")
    prepared["workflow_status"] = "proposed"
    return prepared


def confirm_graph(raw: Mapping[str, Any], *, reviewer: str, date: str) -> ConfirmedGraph:
    proposed = validate_proposal(raw, required_status="proposed")
    if not reviewer.strip() or not date.strip():
        raise GraphWorkflowError("confirmation requires reviewer and date")
    confirmed = {
        key: copy.deepcopy(proposed[key])
        for key in ("profile", "sources", "nodes", "edges", "zones", "adjacency")
    }
    confirmed.update(
        {
            "graph_schema": "confirmed_causal_graph",
            "schema_version": 1,
            "confirmation": {
                "status": "confirmed",
                "reviewer": reviewer.strip(),
                "date": date.strip(),
            },
        }
    )
    return validate_graph(confirmed)


def derive_graph(graph: ConfirmedGraph, mutation: Mapping[str, Any]) -> ConfirmedGraph:
    return derive_variant(graph, mutation)


def document_identity(raw: Mapping[str, Any]) -> str:
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_document(path: Path) -> dict[str, Any]:
    return _read(path)
