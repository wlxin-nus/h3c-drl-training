"""Human-confirmed causal graph validation and deterministic edge identities."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

EDGE_IDENTIFIER = re.compile(r"^ce_[0-9a-f]{8}$")
EDGE_FIELDS = {"source", "relation", "target", "tags", "undirected"}
ALLOWED_RELATIONS = {"Positive Corr", "Negative Corr"}
ALLOWED_EDGE_TAGS = {
    "Immediate",
    "Delayed",
    "Strong Impact",
    "Weak Impact",
    "High Inertia",
}
TIMING_TAGS = {"Immediate", "Delayed"}


class GraphError(ValueError):
    pass


def edge_identity(
    source: str,
    relation: str,
    target: str,
    tags: Iterable[str] = (),
    undirected: bool = False,
) -> str:
    payload = {
        "source": str(source).strip(),
        "relation": str(relation).strip(),
        "target": str(target).strip(),
        "tags": sorted({str(tag).strip() for tag in tags}),
        "undirected": bool(undirected),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_edge_id(
    source: str,
    relation: str,
    target: str,
    tags: Iterable[str] = (),
    undirected: bool = False,
) -> str:
    identity = edge_identity(source, relation, target, tags, undirected).encode("utf-8")
    return "ce_" + hashlib.sha256(identity).hexdigest()[:8]


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str
    tags: tuple[str, ...]
    undirected: bool = False

    @property
    def identifier(self) -> str:
        return stable_edge_id(self.source, self.relation, self.target, self.tags, self.undirected)

    @property
    def sign(self) -> int | None:
        return {"Positive Corr": 1, "Negative Corr": -1}.get(self.relation)

    def as_object(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "tags": list(self.tags),
            "undirected": self.undirected,
        }


@dataclass(frozen=True)
class ConfirmedGraph:
    profile: str
    confirmation: Mapping[str, Any]
    sources: tuple[str, ...]
    nodes: tuple[Mapping[str, str], ...]
    edges: tuple[Edge, ...]
    zones: tuple[str, ...]
    adjacency: tuple[tuple[str, str], ...]
    mutation: Mapping[str, Any] | None = None

    @property
    def by_id(self) -> dict[str, Edge]:
        return {edge.identifier: edge for edge in self.edges}

    def resolved(self) -> dict[str, Any]:
        return {
            "graph_schema": "confirmed_causal_graph",
            "schema_version": 1,
            "profile": self.profile,
            "confirmation": copy.deepcopy(dict(self.confirmation)),
            "sources": list(self.sources),
            "nodes": [copy.deepcopy(dict(node)) for node in self.nodes],
            "edges": [edge.as_object() for edge in self.edges],
            "zones": list(self.zones),
            "adjacency": [list(pair) for pair in self.adjacency],
            **({"mutation": copy.deepcopy(dict(self.mutation))} if self.mutation else {}),
        }


def _edge_from_object(raw: Mapping[str, Any]) -> Edge:
    allowed = EDGE_FIELDS | {"timing", "id"}
    if (
        not isinstance(raw, Mapping)
        or set(raw) - allowed
        or not EDGE_FIELDS - {"undirected"} <= set(raw)
    ):
        raise GraphError("edge fields are invalid")
    for field in ("source", "relation", "target"):
        if (
            not isinstance(raw[field], str)
            or not raw[field].strip()
            or raw[field] != raw[field].strip()
        ):
            raise GraphError("edge endpoints and relation must be non-empty strings")
    if raw["relation"] not in ALLOWED_RELATIONS:
        raise GraphError("edge relation is not in the confirmed signed enumeration")
    if "undirected" in raw and not isinstance(raw["undirected"], bool):
        raise GraphError("edge undirected flag must be boolean")
    tags = raw["tags"]
    if (
        not isinstance(tags, list)
        or not tags
        or any(
            not isinstance(tag, str)
            or not tag.strip()
            or tag != tag.strip()
            or tag not in ALLOWED_EDGE_TAGS
            for tag in tags
        )
        or len(tags) != len(set(tags))
        or len(set(tags) & TIMING_TAGS) != 1
    ):
        raise GraphError("edge tags must use the confirmed enumeration with one timing tag")
    timing = raw.get("timing")
    if timing not in (None, "immediate", "delayed"):
        raise GraphError("edge timing must be immediate or delayed")
    if timing is not None and timing.title() not in tags:
        raise GraphError("edge timing and qualitative tag disagree")
    edge = Edge(
        source=str(raw["source"]),
        relation=str(raw["relation"]),
        target=str(raw["target"]),
        tags=tuple(sorted(set(tags))),
        undirected=bool(raw.get("undirected", False)),
    )
    supplied_identifier = raw.get("id")
    if supplied_identifier is not None and (
        not isinstance(supplied_identifier, str)
        or EDGE_IDENTIFIER.fullmatch(supplied_identifier) is None
        or supplied_identifier != edge.identifier
    ):
        raise GraphError("edge stable identifier does not match its complete content")
    return edge


def validate_graph(raw: Mapping[str, Any]) -> ConfirmedGraph:
    required = {
        "graph_schema",
        "schema_version",
        "profile",
        "confirmation",
        "sources",
        "nodes",
        "edges",
        "zones",
        "adjacency",
    }
    optional = {"mutation"}
    if not isinstance(raw, Mapping) or set(raw) - (required | optional) or not required <= set(raw):
        raise GraphError("confirmed graph fields are invalid")
    if raw["graph_schema"] != "confirmed_causal_graph" or raw["schema_version"] != 1:
        raise GraphError("unsupported confirmed graph schema")
    if (
        not isinstance(raw["profile"], str)
        or not raw["profile"].strip()
        or raw["profile"] != raw["profile"].strip()
    ):
        raise GraphError("confirmed graph profile must be a non-empty string")
    confirmation = raw["confirmation"]
    if not isinstance(confirmation, Mapping) or set(confirmation) != {
        "status",
        "reviewer",
        "date",
    }:
        raise GraphError("confirmation must record status, reviewer, and date")
    if confirmation["status"] != "confirmed":
        raise GraphError("only a fully confirmed graph may run")
    if any(
        not isinstance(confirmation[key], str)
        or not confirmation[key].strip()
        or confirmation[key] != confirmation[key].strip()
        for key in confirmation
    ):
        raise GraphError("confirmation fields must be non-empty strings")
    try:
        parsed_date = date.fromisoformat(str(confirmation["date"]))
    except ValueError as error:
        raise GraphError("confirmation date must be an ISO calendar date") from error
    if parsed_date.isoformat() != confirmation["date"]:
        raise GraphError("confirmation date must be a canonical ISO calendar date")
    sources = raw["sources"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(item, str) or not item.strip() or item != item.strip()
            for item in sources
        )
        or len(sources) != len(set(sources))
    ):
        raise GraphError("confirmed graph requires unique non-empty evidence sources")
    if not isinstance(raw["edges"], list) or not raw["edges"]:
        raise GraphError("confirmed graph requires at least one edge")
    edges = tuple(_edge_from_object(edge) for edge in raw["edges"])
    identities = [
        edge_identity(edge.source, edge.relation, edge.target, edge.tags, edge.undirected)
        for edge in edges
    ]
    identifiers = [edge.identifier for edge in edges]
    if len(identities) != len(set(identities)) or len(identifiers) != len(set(identifiers)):
        raise GraphError("confirmed graph contains duplicate edges or identifier collisions")
    raw_nodes = raw["nodes"]
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise GraphError("confirmed graph requires nodes")
    nodes = tuple(copy.deepcopy(raw_nodes))
    if any(
        not isinstance(node, Mapping)
        or set(node) != {"id", "scope"}
        or not isinstance(node["id"], str)
        or not node["id"].strip()
        or node["id"] != node["id"].strip()
        or node["scope"] not in {"zone", "site"}
        for node in nodes
    ):
        raise GraphError("node fields must be exact non-empty id and allowed scope")
    node_ids = {node["id"] for node in nodes}
    if len(node_ids) != len(nodes):
        raise GraphError("node identifiers must be unique")
    if any(edge.source not in node_ids or edge.target not in node_ids for edge in edges):
        raise GraphError("edge endpoint is not a declared node")
    if (
        not isinstance(raw["zones"], list)
        or not raw["zones"]
        or any(
            not isinstance(zone, str) or not zone.strip() or zone != zone.strip()
            for zone in raw["zones"]
        )
        or len(raw["zones"]) != len(set(raw["zones"]))
    ):
        raise GraphError("graph zones must be non-empty and unique")
    zones = tuple(raw["zones"])
    raw_adjacency = raw["adjacency"]
    if not isinstance(raw_adjacency, list):
        raise GraphError("graph adjacency must be a list")
    adjacency = tuple(tuple(pair) for pair in raw_adjacency if isinstance(pair, list))
    if len(adjacency) != len(raw_adjacency) or any(
        len(pair) != 2 or pair[0] not in zones or pair[1] not in zones or pair[0] == pair[1]
        for pair in adjacency
    ):
        raise GraphError("graph adjacency is invalid")
    normalized_adjacency = {tuple(sorted(pair)) for pair in adjacency}
    if len(normalized_adjacency) != len(adjacency):
        raise GraphError("graph adjacency contains duplicate undirected pairs")
    if len(zones) > 1 and {zone for pair in adjacency for zone in pair} != set(zones):
        raise GraphError("graph adjacency must cover every multizone endpoint")
    mutation = raw.get("mutation")
    if mutation is not None and not isinstance(mutation, Mapping):
        raise GraphError("graph mutation must be an object")
    return ConfirmedGraph(
        profile=str(raw["profile"]),
        confirmation=copy.deepcopy(confirmation),
        sources=tuple(sources),
        nodes=nodes,
        edges=edges,
        zones=zones,
        adjacency=adjacency,
        mutation=copy.deepcopy(mutation),
    )


def load_graph(path: str | Path) -> ConfirmedGraph:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_graph(raw)


def derive_variant(graph: ConfirmedGraph, mutation: Mapping[str, Any]) -> ConfirmedGraph:
    """Derive one declared single-difference graph without changing the canonical graph."""
    if not isinstance(mutation, Mapping) or set(mutation) != {"kind", "edge"}:
        raise GraphError("graph mutation must contain exactly kind and edge")
    selector = mutation["edge"]
    if not isinstance(selector, Mapping) or set(selector) != {"source", "relation", "target"}:
        raise GraphError("graph mutation edge selector is invalid")
    matching = [
        edge
        for edge in graph.edges
        if edge.source == selector["source"]
        and edge.relation == selector["relation"]
        and edge.target == selector["target"]
    ]
    if len(matching) != 1:
        raise GraphError("graph mutation must select exactly one canonical edge")
    selected = matching[0]
    if mutation["kind"] == "remove_edge":
        edges = tuple(edge for edge in graph.edges if edge != selected)
    elif mutation["kind"] == "change_immediate_to_delayed":
        if "Immediate" not in selected.tags or "Delayed" in selected.tags:
            raise GraphError("timing mutation requires one immediate edge")
        replacement = Edge(
            selected.source,
            selected.relation,
            selected.target,
            tuple(sorted({tag for tag in selected.tags if tag != "Immediate"} | {"Delayed"})),
            selected.undirected,
        )
        edges = tuple(replacement if edge == selected else edge for edge in graph.edges)
        if replacement.identifier == selected.identifier:
            raise GraphError("timing mutation must produce a new stable identifier")
    elif mutation["kind"] == "change_delayed_to_immediate":
        if "Delayed" not in selected.tags or "Immediate" in selected.tags:
            raise GraphError("timing mutation requires one delayed edge")
        replacement = Edge(
            selected.source,
            selected.relation,
            selected.target,
            tuple(sorted({tag for tag in selected.tags if tag != "Delayed"} | {"Immediate"})),
            selected.undirected,
        )
        edges = tuple(replacement if edge == selected else edge for edge in graph.edges)
        if replacement.identifier == selected.identifier:
            raise GraphError("timing mutation must produce a new stable identifier")
    else:
        raise GraphError("unsupported graph mutation")
    return ConfirmedGraph(
        profile=graph.profile,
        confirmation=graph.confirmation,
        sources=graph.sources,
        nodes=graph.nodes,
        edges=edges,
        zones=graph.zones,
        adjacency=graph.adjacency,
        mutation=copy.deepcopy(mutation),
    )
