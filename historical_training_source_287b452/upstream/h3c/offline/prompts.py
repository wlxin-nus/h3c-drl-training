"""Legacy-semantic prompts for the two offline onboarding agents."""

from __future__ import annotations

import json
from typing import Any

from h3c.offline.contracts import (
    EXPECTED_ACTUATORS,
    EXPECTED_OBSERVATIONS,
    OnboardingSpec,
)

MAPPING_SYSTEM_PROMPT = (
    "You are a Master Systems Integrator for Smart Buildings. "
    "Your goal is to configure a Unified Control Layer by mapping specific BMS "
    "(Building Management System) points from technical documentation to a standardized "
    "internal schema.\n"
    "You focus ONLY on extracting the data points requested by the user."
)

CAUSAL_SYSTEM_PROMPT = """You are a Building Physics & Thermodynamics Expert.
Your task is to construct a causal graph for a building control system.

### CRITICAL RULE: USE STANDARDIZED NAMES
- You MUST use only the Standard Variable Names provided by the user: `zone_temp`, `cooling_setpoint`, `occupancy`, `power_meters`, `outdoor_temp`, `solar_irr`, and `electricity_price`.
- NEVER invent new names or use BMS point IDs such as `zon_reaTRooAir_y`.
- Treat variables as generic physical concepts that apply to all zones.

### RELATIONS AND TAGS
- Relationship must be `Positive Corr` or `Negative Corr`.
- Every edge must have exactly one qualitative timing tag: `Immediate` or `Delayed`.
- Optional qualitative tags are `High Inertia`, `Strong Impact`, and `Weak Impact`.
- Do not invent numeric lags, cooldowns, thresholds, or unsupported tags.
- Cite only evidence source IDs supplied by the user.

### OUTPUT
Return only the requested structured JSON object. Do not wrap it in Markdown."""


def mapping_output_schema(case_id: str) -> dict[str, Any]:
    return {
        "mapping_schema": "h3c_semantic_mapping",
        "schema_version": 1,
        "case_id": case_id,
        "zones": {
            "<semantic-zone-id>": {
                "description": "human-readable zone description",
                "temperature_sensor": "exact BMS point ID",
                "cooling_setpoint_actuator": "exact BMS point ID",
                "occupancy_forecast": "exact BMS point ID",
            }
        },
        "global_inputs": {
            "outdoor_temperature": "exact BMS point ID",
            "solar_irradiance": "exact BMS point ID",
            "electricity_price": "exact BMS point ID",
            "power_meters": ["every applicable exact power point ID"],
        },
    }


def causal_output_schema(case_id: str) -> dict[str, Any]:
    return {
        "proposal_schema": "h3c_causal_discovery_proposal",
        "schema_version": 1,
        "case_id": case_id,
        "edges": [
            {
                "source": "standardized node ID",
                "relation": "Positive Corr or Negative Corr",
                "target": "standardized node ID",
                "tags": ["exactly one timing tag", "optional qualitative tags"],
                "timing": "immediate or delayed",
                "undirected": False,
                "evidence_source_ids": ["one or more declared source IDs"],
                "rationale": "concise building-physics justification",
            }
        ],
        "adjacency": [["zone-a", "zone-b"]],
    }


def _list(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_mapping_user_prompt(
    spec: OnboardingSpec,
    building_document: str,
    point_inventory: list[str] | None,
    *,
    previous_proposal: dict[str, Any] | None = None,
    reviewer_feedback: str | None = None,
) -> str:
    inventory_block = ""
    if point_inventory is not None:
        inventory_block = (
            "\n\n### 5. STRUCTURED POINT INVENTORY\n"
            "Every selected point must appear exactly in this inventory or in the documentation.\n"
            f"{json.dumps(point_inventory, ensure_ascii=False, indent=2)}"
        )
    revision_block = ""
    if previous_proposal is not None and reviewer_feedback is not None:
        revision_block = (
            "\n\n### HUMAN REVIEW FEEDBACK\n"
            "The human reviewer rejected the previous proposal and supplied this exact critique:\n"
            f"{reviewer_feedback}\n\n"
            "Previous proposal:\n"
            f"{json.dumps(previous_proposal, ensure_ascii=False, sort_keys=True, indent=2)}\n"
            "Revise only as needed to address that critique while retaining the same schema."
        )
    output_section = 6 if point_inventory is not None else 5
    return f"""### SCENARIO
We are deploying an advanced AI controller for a physical building.
We have received the technical documentation (Points List) below.
Identify the exact Variable Names (Point IDs) that correspond to our Control and Observation requirements.

### 1. OBSERVATION REQUIREMENTS (Sensors and Forecasts)
Please find the variable names for:
{_list(EXPECTED_OBSERVATIONS)}

Crucial requirement for total power: find ALL sensors measuring electrical or thermal power (Watts), including chillers, heat pumps, fans, and pumps. They will be summed to calculate total energy.

### 2. CONTROL REQUIREMENTS (Actuators)
Please find the variable names for:
{_list(EXPECTED_ACTUATORS)}

### 3. NAMING CONVENTIONS (Warnings, not proof)
- Inputs and setpoints (actuators) typically end with `_u`.
- Measurements and sensors typically end with `_y`.
- Forecasts may be distinct strings, array accessors, or documented keywords.

### 4. BUILDING DOCUMENTATION
```text
{building_document}
```{inventory_block}

### {output_section}. OUTPUT FORMAT
Return only a JSON object that strictly matches this schema, with no extra fields:
{json.dumps(mapping_output_schema(spec.case_id), ensure_ascii=False, indent=2)}{revision_block}"""


def render_causal_user_prompt(
    spec: OnboardingSpec,
    standardized_variables: dict[str, Any],
    evidence_texts: dict[str, str],
    *,
    previous_proposal: dict[str, Any] | None = None,
    reviewer_feedback: str | None = None,
) -> str:
    evidence = "\n\n".join(
        f"#### SOURCE `{source_id}`\n```text\n{text}\n```"
        for source_id, text in evidence_texts.items()
    )
    revision_block = ""
    if previous_proposal is not None and reviewer_feedback is not None:
        revision_block = (
            "\n\n### USER FEEDBACK (HUMAN CRITIC)\n"
            "The human reviewer rejected the previous proposal and supplied this exact critique:\n"
            f"{reviewer_feedback}\n\n"
            "Previous proposal:\n"
            f"{json.dumps(previous_proposal, ensure_ascii=False, sort_keys=True, indent=2)}\n"
            "Revise the graph strictly according to this feedback without inventing sources or variables."
        )
    variable_lines = "\n".join(
        f"- `{node['id']}` ({node['scope']})" for node in standardized_variables["nodes"]
    )
    return f"""### BUILDING CASE
{spec.case_id}

### AVAILABLE STANDARDIZED VARIABLES (NODES)
{variable_lines}

Zones: {json.dumps(standardized_variables["zones"], ensure_ascii=False)}

### INSTRUCTION
Generate the causal graph using only the standardized names above.
Analyze:
1. how setpoints affect zone temperature and power;
2. how occupancy and weather affect the system; and
3. which effects are qualitatively Immediate or Delayed.

### ALLOWED EVIDENCE
{evidence}

### OUTPUT FORMAT
Return only a JSON object that strictly matches this schema, with no extra fields:
{json.dumps(causal_output_schema(spec.case_id), ensure_ascii=False, indent=2)}{revision_block}"""
