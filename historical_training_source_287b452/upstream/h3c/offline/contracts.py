"""Strict input and proposal contracts for offline onboarding."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from h3c.causal.graph import stable_edge_id
from h3c.causal.workflow import prepare_proposal, propose_graph
from h3c.experiments.profiles import PROFILE_FIELDS, validate_profile

SPEC_FIELDS = {
    "onboarding_schema",
    "schema_version",
    "case_id",
    "building_document",
    "point_inventory",
    "case_profile_template",
    "requirements",
    "evidence_sources",
    "provider",
    "limits",
}
REQUIREMENT_FIELDS = {"observations", "actuators"}
EXPECTED_OBSERVATIONS = (
    "zone_temperature",
    "occupancy_forecast",
    "outdoor_temperature_forecast",
    "solar_irradiance_forecast",
    "electricity_price_forecast",
    "power_meters",
)
EXPECTED_ACTUATORS = ("cooling_setpoint",)
PROVIDER_FIELDS = {
    "kind",
    "endpoint_env",
    "api_key_env",
    "model_env",
    "supports_reasoning_effort_low",
}
LIMIT_FIELDS = {"mapping_model_calls", "causal_model_calls"}
EVIDENCE_SOURCE_FIELDS = {"id", "path"}
MAPPING_FIELDS = {"mapping_schema", "schema_version", "case_id", "zones", "global_inputs"}
MAPPING_ZONE_FIELDS = {
    "description",
    "temperature_sensor",
    "cooling_setpoint_actuator",
    "occupancy_forecast",
}
MAPPING_GLOBAL_FIELDS = {
    "outdoor_temperature",
    "solar_irradiance",
    "electricity_price",
    "power_meters",
}
CAUSAL_FIELDS = {
    "proposal_schema",
    "schema_version",
    "case_id",
    "edges",
    "adjacency",
}
CAUSAL_EDGE_FIELDS = {
    "source",
    "relation",
    "target",
    "tags",
    "timing",
    "undirected",
    "evidence_source_ids",
    "rationale",
}
ALLOWED_NODES = {
    "zone_temp",
    "cooling_setpoint",
    "occupancy",
    "power_meters",
    "outdoor_temp",
    "solar_irr",
    "electricity_price",
}
ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
CASE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
SOURCE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class OfflineContractError(ValueError):
    """Raised when an offline input or proposal fails closed."""


@dataclass(frozen=True)
class EvidenceSource:
    identifier: str
    path: Path


@dataclass(frozen=True)
class ProviderContract:
    endpoint_env: str
    api_key_env: str
    model_env: str


@dataclass(frozen=True)
class OnboardingSpec:
    case_id: str
    building_document: Path
    point_inventory: Path | None
    case_profile_template: Path
    evidence_sources: tuple[EvidenceSource, ...]
    provider: ProviderContract
    mapping_model_calls: int
    causal_model_calls: int
    source_path: Path
    repository_root: Path

    def public(self) -> dict[str, Any]:
        """Return the secret-free, repository-relative resolved specification."""

        root = self.repository_root.resolve()

        def relative(path: Path) -> str:
            return path.resolve().relative_to(root).as_posix()

        return {
            "onboarding_schema": "h3c_offline_onboarding",
            "schema_version": 1,
            "case_id": self.case_id,
            "building_document": relative(self.building_document),
            "point_inventory": (
                relative(self.point_inventory) if self.point_inventory is not None else None
            ),
            "case_profile_template": relative(self.case_profile_template),
            "requirements": {
                "observations": list(EXPECTED_OBSERVATIONS),
                "actuators": list(EXPECTED_ACTUATORS),
            },
            "evidence_sources": [
                {"id": source.identifier, "path": relative(source.path)}
                for source in self.evidence_sources
            ],
            "provider": {
                "kind": "openai_compatible",
                "endpoint_env": self.provider.endpoint_env,
                "api_key_env": self.provider.api_key_env,
                "model_env": self.provider.model_env,
                "supports_reasoning_effort_low": True,
            },
            "limits": {
                "mapping_model_calls": self.mapping_model_calls,
                "causal_model_calls": self.causal_model_calls,
            },
        }

    @property
    def identity(self) -> str:
        return object_identity(self.public())


def object_identity(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OfflineContractError(f"JSON object cannot be read: {path}") from error
    if not isinstance(value, dict):
        raise OfflineContractError(f"JSON root must be an object: {path}")
    return value


def read_text_document(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise OfflineContractError(f"document must be readable UTF-8 text: {path}") from error
    if not text.strip():
        raise OfflineContractError(f"document must not be empty: {path}")
    return text


def _strict_path(root: Path, raw: Any, name: str) -> Path:
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
        raise OfflineContractError(f"{name} must be a non-empty repository-relative path")
    root_resolved = root.resolve()
    path = (root_resolved / raw).resolve()
    if not path.is_relative_to(root_resolved) or not path.is_file():
        raise OfflineContractError(f"{name} is missing or outside the repository")
    return path


def load_onboarding_spec(path: Path, repository_root: Path) -> OnboardingSpec:
    raw = read_json_object(path)
    if set(raw) != SPEC_FIELDS:
        raise OfflineContractError("onboarding spec fields do not match the schema")
    if raw["onboarding_schema"] != "h3c_offline_onboarding" or raw["schema_version"] != 1:
        raise OfflineContractError("unsupported onboarding spec schema")
    case_id = raw["case_id"]
    if not isinstance(case_id, str) or CASE_ID.fullmatch(case_id) is None:
        raise OfflineContractError("case_id must use a portable semantic identifier")
    requirements = raw["requirements"]
    if not isinstance(requirements, dict) or set(requirements) != REQUIREMENT_FIELDS:
        raise OfflineContractError("requirements fields are invalid")
    if requirements["observations"] != list(EXPECTED_OBSERVATIONS) or requirements[
        "actuators"
    ] != list(EXPECTED_ACTUATORS):
        raise OfflineContractError("requirements must match the final cooling-only H3C interface")

    provider = raw["provider"]
    if not isinstance(provider, dict) or set(provider) != PROVIDER_FIELDS:
        raise OfflineContractError("provider fields are invalid")
    if (
        provider["kind"] != "openai_compatible"
        or provider["supports_reasoning_effort_low"] is not True
    ):
        raise OfflineContractError(
            "provider must explicitly support OpenAI-compatible thinking low"
        )
    environment_names = tuple(
        provider[field] for field in ("endpoint_env", "api_key_env", "model_env")
    )
    if any(
        not isinstance(name, str) or ENVIRONMENT_NAME.fullmatch(name) is None
        for name in environment_names
    ):
        raise OfflineContractError("provider credentials and identity must use environment names")
    if len(set(environment_names)) != len(environment_names):
        raise OfflineContractError("provider environment names must be distinct")

    limits = raw["limits"]
    if not isinstance(limits, dict) or set(limits) != LIMIT_FIELDS:
        raise OfflineContractError("model call limit fields are invalid")
    limit_values = tuple(limits[field] for field in ("mapping_model_calls", "causal_model_calls"))
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10
        for value in limit_values
    ):
        raise OfflineContractError("each model stage limit must be an integer from one through ten")

    sources = raw["evidence_sources"]
    if not isinstance(sources, list) or not sources:
        raise OfflineContractError("at least one evidence source is required")
    resolved_sources: list[EvidenceSource] = []
    for source in sources:
        if not isinstance(source, dict) or set(source) != EVIDENCE_SOURCE_FIELDS:
            raise OfflineContractError("evidence source fields are invalid")
        identifier = source["id"]
        if not isinstance(identifier, str) or SOURCE_ID.fullmatch(identifier) is None:
            raise OfflineContractError("evidence source ID is invalid")
        resolved_sources.append(
            EvidenceSource(
                identifier, _strict_path(repository_root, source["path"], "evidence source")
            )
        )
    if len({source.identifier for source in resolved_sources}) != len(resolved_sources):
        raise OfflineContractError("evidence source IDs must be unique")

    inventory_raw = raw["point_inventory"]
    inventory = (
        None
        if inventory_raw is None
        else _strict_path(repository_root, inventory_raw, "point inventory")
    )
    source_path = path.resolve()
    if not source_path.is_file():
        raise OfflineContractError("onboarding spec path is missing")
    return OnboardingSpec(
        case_id=case_id,
        building_document=_strict_path(
            repository_root, raw["building_document"], "building document"
        ),
        point_inventory=inventory,
        case_profile_template=_strict_path(
            repository_root, raw["case_profile_template"], "case profile template"
        ),
        evidence_sources=tuple(resolved_sources),
        provider=ProviderContract(*environment_names),
        mapping_model_calls=limit_values[0],
        causal_model_calls=limit_values[1],
        source_path=source_path,
        repository_root=repository_root.resolve(),
    )


def _point_inventory(spec: OnboardingSpec) -> set[str]:
    if spec.point_inventory is None:
        return set()
    raw = json.loads(spec.point_inventory.read_text(encoding="utf-8"))
    points: Any
    if isinstance(raw, list):
        points = raw
    elif isinstance(raw, dict) and set(raw) == {"points"}:
        points = raw["points"]
    else:
        raise OfflineContractError("point inventory must be a list or an object containing points")
    if (
        not isinstance(points, list)
        or not points
        or any(
            not isinstance(point, str) or not point.strip() or point != point.strip()
            for point in points
        )
        or len(points) != len(set(points))
    ):
        raise OfflineContractError("point inventory values must be unique non-empty strings")
    return set(points)


def validate_mapping(raw: Any, spec: OnboardingSpec) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != MAPPING_FIELDS:
        raise OfflineContractError("mapping proposal fields do not match the schema")
    if (
        raw["mapping_schema"] != "h3c_semantic_mapping"
        or raw["schema_version"] != 1
        or raw["case_id"] != spec.case_id
    ):
        raise OfflineContractError("mapping proposal identity is invalid")
    zones = raw["zones"]
    if not isinstance(zones, dict) or not zones:
        raise OfflineContractError("mapping requires at least one zone")
    point_roles: list[tuple[str, str]] = []
    for zone, mapping in zones.items():
        if (
            not isinstance(zone, str)
            or not zone.strip()
            or zone != zone.strip()
            or not isinstance(mapping, dict)
            or set(mapping) != MAPPING_ZONE_FIELDS
        ):
            raise OfflineContractError("zone mapping identity or fields are invalid")
        for field, value in mapping.items():
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise OfflineContractError("zone mapping values must be non-empty strings")
            if field != "description":
                point_roles.append((f"{zone}.{field}", value))
    global_inputs = raw["global_inputs"]
    if not isinstance(global_inputs, dict) or set(global_inputs) != MAPPING_GLOBAL_FIELDS:
        raise OfflineContractError("global mapping fields are invalid")
    for field in MAPPING_GLOBAL_FIELDS - {"power_meters"}:
        value = global_inputs[field]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise OfflineContractError("global mapping values must be non-empty strings")
        point_roles.append((field, value))
    meters = global_inputs["power_meters"]
    if (
        not isinstance(meters, list)
        or not meters
        or any(
            not isinstance(point, str) or not point.strip() or point != point.strip()
            for point in meters
        )
        or len(meters) != len(set(meters))
    ):
        raise OfflineContractError("power meters must be unique non-empty point identifiers")
    point_roles.extend(("power_meters", point) for point in meters)
    points = [point for _, point in point_roles]
    if len(points) != len(set(points)):
        raise OfflineContractError("one BMS point cannot serve conflicting mapping roles")

    inventory = _point_inventory(spec)
    document = read_text_document(spec.building_document)
    unknown = sorted(point for point in points if point not in inventory and point not in document)
    if unknown:
        raise OfflineContractError(f"mapping contains unknown source points: {unknown}")
    return copy.deepcopy(raw)


def mapping_warnings(mapping: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for zone, values in mapping["zones"].items():
        actuator = values["cooling_setpoint_actuator"]
        if not actuator.endswith("_u"):
            warnings.append(f"{zone} cooling actuator does not end in _u: {actuator}")
        for field in ("temperature_sensor", "occupancy_forecast"):
            point = values[field]
            if not point.endswith("_y"):
                warnings.append(f"{zone} {field} does not end in _y: {point}")
    for field in ("outdoor_temperature", "solar_irradiance", "electricity_price"):
        point = mapping["global_inputs"][field]
        if not point.endswith("_y"):
            warnings.append(f"{field} does not end in _y: {point}")
    for point in mapping["global_inputs"]["power_meters"]:
        if not point.endswith("_y"):
            warnings.append(f"power meter does not end in _y: {point}")
    return warnings


def merge_profile_template(spec: OnboardingSpec, mapping: dict[str, Any]) -> dict[str, Any]:
    template = read_json_object(spec.case_profile_template)
    if set(template) != PROFILE_FIELDS:
        raise OfflineContractError("case profile template fields do not match the profile schema")
    if template.get("profile") != spec.case_id:
        raise OfflineContractError(
            "case profile template identity differs from the onboarding case"
        )
    if template.get("zones") != {} or template.get("global_inputs") != {}:
        raise OfflineContractError("profile template mapping fields must be empty placeholders")
    candidate = copy.deepcopy(template)
    candidate["zones"] = copy.deepcopy(mapping["zones"])
    candidate["global_inputs"] = copy.deepcopy(mapping["global_inputs"])
    try:
        return validate_profile(candidate, root=spec.repository_root, allow_missing_graph=True)
    except ValueError as error:
        raise OfflineContractError(str(error)) from error


def standard_variables(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "zones": list(mapping["zones"]),
        "nodes": [
            {"id": "zone_temp", "scope": "zone"},
            {"id": "cooling_setpoint", "scope": "zone"},
            {"id": "occupancy", "scope": "zone"},
            {"id": "power_meters", "scope": "site"},
            {"id": "outdoor_temp", "scope": "site"},
            {"id": "solar_irr", "scope": "site"},
            {"id": "electricity_price", "scope": "site"},
        ],
    }


def validate_causal_proposal(
    raw: Any, spec: OnboardingSpec, mapping: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != CAUSAL_FIELDS:
        raise OfflineContractError("causal proposal fields do not match the schema")
    if (
        raw["proposal_schema"] != "h3c_causal_discovery_proposal"
        or raw["schema_version"] != 1
        or raw["case_id"] != spec.case_id
    ):
        raise OfflineContractError("causal proposal identity is invalid")
    declared_sources = {source.identifier for source in spec.evidence_sources}
    edges = raw["edges"]
    if not isinstance(edges, list) or not edges:
        raise OfflineContractError("causal proposal requires at least one edge")
    graph_edges: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict) or set(edge) != CAUSAL_EDGE_FIELDS:
            raise OfflineContractError("causal edge fields are invalid")
        if edge["source"] not in ALLOWED_NODES or edge["target"] not in ALLOWED_NODES:
            raise OfflineContractError("causal edge endpoint is not a standardized variable")
        source_ids = edge["evidence_source_ids"]
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or any(not isinstance(item, str) for item in source_ids)
            or len(source_ids) != len(set(source_ids))
            or not set(source_ids) <= declared_sources
        ):
            raise OfflineContractError("causal edge cites an unknown or duplicate evidence source")
        rationale = edge["rationale"]
        if (
            not isinstance(rationale, str)
            or not rationale.strip()
            or rationale != rationale.strip()
        ):
            raise OfflineContractError("causal edge rationale must be a non-empty string")
        graph_edge = {
            key: copy.deepcopy(edge[key])
            for key in ("source", "relation", "target", "tags", "timing", "undirected")
        }
        graph_edges.append(graph_edge)
        provenance.append(
            {
                "stable_edge_id": stable_edge_id(
                    graph_edge["source"],
                    graph_edge["relation"],
                    graph_edge["target"],
                    graph_edge["tags"],
                    graph_edge["undirected"],
                ),
                "evidence_source_ids": list(source_ids),
                "agent_rationale": rationale,
            }
        )
    prepared = prepare_proposal(
        spec.case_id,
        tuple(mapping["zones"]),
        tuple(source.identifier for source in spec.evidence_sources),
    )
    prepared["edges"] = graph_edges
    prepared["adjacency"] = copy.deepcopy(raw["adjacency"])
    try:
        graph_proposal = propose_graph(prepared)
    except ValueError as error:
        raise OfflineContractError(str(error)) from error
    return {
        "raw": copy.deepcopy(raw),
        "graph_proposal": graph_proposal,
        "edge_provenance": provenance,
    }


def bind_edge_provenance(
    graph_edges: list[dict[str, Any]],
    edge_provenance: list[dict[str, Any]],
    *,
    confirmed_round: int,
    human_feedback: list[str],
) -> list[dict[str, Any]]:
    """Bind explanatory evidence to graph edges by stable identity, never list position."""
    if not isinstance(confirmed_round, int) or confirmed_round < 1:
        raise OfflineContractError("confirmed causal round must be a positive integer")
    if any(not isinstance(item, str) or not item.strip() for item in human_feedback):
        raise OfflineContractError("human feedback entries must be non-empty strings")
    by_identifier: dict[str, dict[str, Any]] = {}
    for item in edge_provenance:
        if not isinstance(item, dict) or set(item) != {
            "stable_edge_id",
            "evidence_source_ids",
            "agent_rationale",
        }:
            raise OfflineContractError("edge provenance fields are invalid")
        identifier = item["stable_edge_id"]
        if not isinstance(identifier, str) or identifier in by_identifier:
            raise OfflineContractError("edge provenance identifiers must be unique strings")
        by_identifier[identifier] = item
    graph_identifiers: list[str] = []
    for edge in graph_edges:
        identifier = edge.get("id")
        if not isinstance(identifier, str):
            raise OfflineContractError("edge provenance does not exactly cover the confirmed graph")
        graph_identifiers.append(identifier)
    if len(graph_identifiers) != len(set(graph_identifiers)) or set(graph_identifiers) != set(
        by_identifier
    ):
        raise OfflineContractError("edge provenance does not exactly cover the confirmed graph")
    return [
        {
            "stable_edge_id": identifier,
            "evidence_source_ids": list(by_identifier[identifier]["evidence_source_ids"]),
            "agent_rationale": by_identifier[identifier]["agent_rationale"],
            "confirmed_round": confirmed_round,
            "human_feedback": list(human_feedback),
        }
        for identifier in graph_identifiers
    ]
