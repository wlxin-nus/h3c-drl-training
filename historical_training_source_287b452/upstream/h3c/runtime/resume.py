"""Read-only validation and lineage for atomic completed-hour recovery."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from h3c.control.program import load_program, program_hash
from h3c.experiments.matrix import RunPlan
from h3c.experiments.profiles import repository_root
from h3c.memory.ledger import ProgramLedger
from h3c.outputs.artifacts import PERFORMANCE_COLUMNS

RESUME_STREAMS = (
    "agent_calls.jsonl",
    "raw_model_io.jsonl",
    "model_request_attempts.jsonl",
    "program_updates.jsonl",
    "hourly_decisions.jsonl",
    "caol_records.jsonl",
    "zone_steps.jsonl",
)
ALLOWED_FAILURE_TYPES = {"terminal_transport_error"}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"resume artifact {path.name} must contain one object")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"resume stream {path.name} must contain objects")
        values.append(value)
    return values


def _performance(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != PERFORMANCE_COLUMNS:
            raise ValueError("resume performance header does not match the runtime owner")
        return [dict(row) for row in reader]


def plan_from_resolved(resolved: Mapping[str, Any]) -> RunPlan:
    profile = resolved.get("case_profile")
    method = resolved.get("method")
    if not isinstance(profile, Mapping) or not isinstance(method, Mapping):
        raise ValueError("resume source does not contain a resolved plan")
    provider = resolved.get("model_provider")
    return RunPlan(
        profile=str(profile["profile"]),
        controller=str(method["controller"]),
        working_memory_hours=int(method["working_memory_hours"]),
        causal_enabled=bool(method["causal_enabled"]),
        coordination_enabled=bool(method["coordination_enabled"]),
        thinking_policy=str(method["thinking_policy"]),
        graph_mutation=method.get("graph_mutation"),
        evaluation_hours=int(method["evaluation_hours"]),
        long_term_memory=bool(method.get("long_term_memory", False)),
        model_provider=str(provider) if provider is not None else None,
        diagnostic_window=(
            str(method["diagnostic_window"])
            if method.get("diagnostic_window") is not None
            else None
        ),
    )


def _prefix_identity(
    *,
    next_step: int,
    program_versions: Mapping[str, Any],
    forecast_inputs: Mapping[str, Any],
    performance: Sequence[Mapping[str, Any]],
    zone_steps: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    caol: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    raw_calls: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> str:
    normalized_zone_steps = [
        {key: value for key, value in row.items() if key != "test_id"} for row in zone_steps
    ]
    return _identity(
        {
            "next_step": next_step,
            "program_versions": dict(program_versions),
            "forecast_inputs": dict(forecast_inputs),
            "performance": list(performance),
            "zone_steps": normalized_zone_steps,
            "program_updates": list(updates),
            "hourly_decisions": list(decisions),
            "caol_records": list(caol),
            "agent_calls": list(calls),
            "raw_model_io": list(raw_calls),
            "model_request_attempts": list(attempts),
        }
    )


@dataclass(frozen=True)
class ResumePrefix:
    source_run: Path
    plan: RunPlan
    source_commit: str
    source_plan_identity: str
    source_run_identity: str
    source_test_id: str
    completed_hour: int
    completed_step: int
    next_step: int
    program_versions: Mapping[str, int]
    forecast_inputs: Mapping[str, Any]
    performance: tuple[Mapping[str, str], ...]
    zone_steps: tuple[Mapping[str, Any], ...]
    program_updates: tuple[Mapping[str, Any], ...]
    hourly_decisions: tuple[Mapping[str, Any], ...]
    caol_records: tuple[Mapping[str, Any], ...]
    agent_calls: tuple[Mapping[str, Any], ...]
    raw_model_io: tuple[Mapping[str, Any], ...]
    model_request_attempts: tuple[Mapping[str, Any], ...]
    prefix_identity: str

    @property
    def completed_hours(self) -> int:
        return self.completed_hour + 1

    def lineage(self) -> dict[str, Any]:
        return {
            "source_commit": self.source_commit,
            "source_run_identity": self.source_run_identity,
            "source_plan_identity": self.source_plan_identity,
            "source_test_id": self.source_test_id,
            "source_completed_hour": self.completed_hour,
            "source_completed_step": self.completed_step,
            "source_next_step": self.next_step,
            "source_program_versions": dict(self.program_versions),
            "source_prefix_identity": self.prefix_identity,
        }

    def calls_for_import(self, stream: str) -> tuple[Mapping[str, Any], ...]:
        if stream == "agent_calls.jsonl":
            return self.agent_calls
        if stream == "raw_model_io.jsonl":
            return self.raw_model_io
        if stream == "model_request_attempts.jsonl":
            return self.model_request_attempts
        raise ValueError("stream is not an importable model-evidence stream")

    def updates_for_hour(self, hour: int) -> list[Mapping[str, Any]]:
        return [row for row in self.program_updates if int(row["hour"]) == hour]

    def zone_rows_for_step(self, step: int) -> dict[str, Mapping[str, Any]]:
        return {str(row["zone"]): row for row in self.zone_steps if int(row["step"]) == step}

    def caol_for_hour(self, hour: int) -> list[Mapping[str, Any]]:
        return [row for row in self.caol_records if int(row["hour"]) == hour]

    def decision_for_hour(self, hour: int) -> Mapping[str, Any]:
        return self.hourly_decisions[hour]


def _validate_program_prefix(
    *,
    plan: RunPlan,
    profile: Mapping[str, Any],
    updates: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    program_versions: Mapping[str, Any],
) -> None:
    zones = tuple(profile["zones"])
    ledgers = {
        zone: ProgramLedger(
            load_program(repository_root() / str(profile["program"]), zone),
            causal_enabled=plan.causal_enabled,
        )
        for zone in zones
    }
    for row in updates:
        zone = str(row["zone"])
        if zone not in ledgers:
            raise ValueError("resume program update has an unknown zone")
        ledger = ledgers[zone]
        before_version = int(row["program_version_before"])
        before_hash = str(row["program_hash_before"])
        if ledger.version != before_version or program_hash(ledger.current_program) != before_hash:
            raise ValueError("resume program prefix does not start from the recorded version")
        patch = row["patch"]
        if (
            row.get("status") == "accepted"
            and isinstance(patch, Mapping)
            and patch.get("op") != "no_change"
        ):
            accepted = ledger.commit(
                patch,
                step=int(row["step"]),
                hour=int(row["hour"]),
            ).as_dict()
            if row.get("accepted_update") != accepted:
                raise ValueError("resume accepted program update cannot be replayed")
        if (
            ledger.version != int(row["program_version_after"])
            or program_hash(ledger.current_program) != row["program_hash_after"]
            or row.get("current_program_version") != ledger.version
            or row.get("current_program_hash") != program_hash(ledger.current_program)
        ):
            raise ValueError("resume program update after-state diverges")
    if {zone: ledgers[zone].version for zone in zones} != {
        str(zone): int(value) for zone, value in program_versions.items()
    }:
        raise ValueError("resume checkpoint program versions do not match replay")
    if not decisions:
        raise ValueError("resume requires at least one completed hourly decision")
    final_replay = decisions[-1].get("program_replay")
    if not isinstance(final_replay, Mapping):
        raise ValueError("resume final hourly decision lacks program replay evidence")
    for zone in zones:
        expected = final_replay.get(zone)
        if not isinstance(expected, Mapping) or (
            expected.get("verified") is not True
            or expected.get("version") != ledgers[zone].version
            or expected.get("hash") != program_hash(ledgers[zone].current_program)
        ):
            raise ValueError("resume final program replay identity diverges")


def load_resume_prefix(source_run: Path) -> ResumePrefix:
    directory = source_run.resolve()
    if not directory.is_dir():
        raise ValueError("resume source run does not exist")
    required = {
        "resolved_config.yaml",
        "manifest.json",
        "failure.json",
        "completed_hour_checkpoint.json",
        "performance.csv",
        "forecast_inputs.json",
        *RESUME_STREAMS,
    }
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        raise ValueError(f"resume source is missing artifacts: {missing}")
    if (directory / "completion.json").exists():
        raise ValueError("a completed run is not resume-eligible")

    resolved = _object(directory / "resolved_config.yaml")
    manifest = _object(directory / "manifest.json")
    failure = _object(directory / "failure.json")
    checkpoint = _object(directory / "completed_hour_checkpoint.json")
    plan = plan_from_resolved(resolved)
    if plan.controller != "h3c_agent":
        raise ValueError("only Agent runs are resume-eligible")
    if plan.long_term_memory:
        raise ValueError("long-term-memory CRUD resume is not implemented")
    if failure.get("failure_type") not in ALLOWED_FAILURE_TYPES:
        raise ValueError("resume source failure is not a registered transport interruption")
    if (
        manifest.get("secret_scan_status") != "completed"
        or manifest.get("secret_exposure_count") != 0
    ):
        raise ValueError("resume source secret evidence is not clean")
    source_run_identity = str(manifest.get("run_identity", ""))
    if re.fullmatch(r"[0-9a-f]{64}", source_run_identity) is None:
        raise ValueError("resume source run identity is invalid")
    if failure.get("run_identity") != source_run_identity:
        raise ValueError("resume failure identity does not match its manifest")

    completed_hour = checkpoint.get("completed_hour")
    completed_step = checkpoint.get("completed_step")
    next_step = checkpoint.get("next_step")
    if (
        isinstance(completed_hour, bool)
        or not isinstance(completed_hour, int)
        or isinstance(completed_step, bool)
        or not isinstance(completed_step, int)
        or isinstance(next_step, bool)
        or not isinstance(next_step, int)
        or completed_hour < 0
        or completed_step != completed_hour * 4 + 3
        or next_step != completed_step + 1
        or next_step >= plan.evaluation_hours * 4
    ):
        raise ValueError("resume checkpoint is not one atomic incomplete-run hour boundary")
    if checkpoint.get("run_identity") != source_run_identity:
        raise ValueError("resume checkpoint run identity is invalid")
    source_test_id = str(checkpoint.get("test_id", ""))
    if not source_test_id:
        raise ValueError("resume checkpoint test identity is missing")
    program_versions = checkpoint.get("program_versions")
    if not isinstance(program_versions, Mapping):
        raise ValueError("resume checkpoint program versions are missing")

    profile = resolved["case_profile"]
    execution_identity = resolved.get("execution_identity")
    if not isinstance(execution_identity, Mapping):
        raise ValueError("resume source execution identity is missing")
    source_plan_identity = str(execution_identity.get("plan_identity", ""))
    if source_plan_identity != plan.identity(dict(profile)):
        raise ValueError("resume source plan identity cannot be reconstructed")
    source_commit = str(execution_identity.get("source_commit", ""))
    if (
        re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or manifest.get("source_commit") != source_commit
    ):
        raise ValueError("resume source commit identity is invalid")
    zones = tuple(profile["zones"])
    completed_hours = completed_hour + 1
    expected_calls = completed_hours * (len(zones) + 1 + int(plan.coordination_enabled))
    all_rows = {name: _rows(directory / name) for name in RESUME_STREAMS}
    all_performance = _performance(directory / "performance.csv")
    forecast_inputs = _object(directory / "forecast_inputs.json")
    all_zone_steps = all_rows["zone_steps.jsonl"]
    all_updates = all_rows["program_updates.jsonl"]
    all_decisions = all_rows["hourly_decisions.jsonl"]
    all_caol = all_rows["caol_records.jsonl"]
    if [int(row["step"]) for row in all_performance] != list(range(len(all_performance))):
        raise ValueError("resume source performance timeline is not contiguous")
    if not next_step <= len(all_performance) <= next_step + 4:
        raise ValueError("resume source extends beyond one incomplete hour after its checkpoint")
    all_step_zone_pairs = [(int(row["step"]), str(row["zone"])) for row in all_zone_steps]
    if all_step_zone_pairs != [
        (step, zone) for step in range(len(all_performance)) for zone in zones
    ]:
        raise ValueError("resume source zone-step timeline is not canonical")
    performance = all_performance[:next_step]
    zone_steps = [row for row in all_zone_steps if int(row["step"]) < next_step]
    updates = [row for row in all_updates if int(row["hour"]) < completed_hours]
    decisions = [row for row in all_decisions if int(row["hour"]) < completed_hours]
    caol = [row for row in all_caol if int(row["hour"]) < completed_hours]
    calls = [row for row in all_rows["agent_calls.jsonl"] if int(row["hour"]) < completed_hours]
    raw_calls = [
        row for row in all_rows["raw_model_io.jsonl"] if int(row["hour"]) < completed_hours
    ]
    attempts = [
        row
        for row in all_rows["model_request_attempts.jsonl"]
        if int(row["hour"]) < completed_hours
    ]
    if (
        len(performance) != next_step
        or len(zone_steps) != next_step * len(zones)
        or len(updates) != completed_hours * len(zones)
        or len(decisions) != completed_hours
        or len(caol) != completed_hours * len(zones)
        or len(calls) != expected_calls
        or len(raw_calls) != expected_calls
    ):
        raise ValueError("resume source complete-prefix stream counts are inconsistent")
    step_zone_pairs = [(int(row["step"]), str(row["zone"])) for row in zone_steps]
    if step_zone_pairs != [(step, zone) for step in range(next_step) for zone in zones]:
        raise ValueError("resume zone-step timeline is not canonical")
    if {str(row.get("test_id")) for row in zone_steps} != {source_test_id}:
        raise ValueError("resume physical prefix changed test identity")
    if [int(row["hour"]) for row in decisions] != list(range(completed_hours)):
        raise ValueError("resume hourly decisions are not contiguous")
    if [
        int(row["call_ordinal"]) for row in sorted(calls, key=lambda row: row["call_ordinal"])
    ] != list(range(expected_calls)):
        raise ValueError("resume completed logical calls are not contiguous")
    call_ids = {str(row["logical_call_identity"]) for row in calls}
    if (
        len(call_ids) != expected_calls
        or {str(row["logical_call_identity"]) for row in raw_calls} != call_ids
        or {str(row["logical_call_identity"]) for row in attempts} != call_ids
    ):
        raise ValueError("resume completed model evidence is not identity-complete")
    if any(int(row["hour"]) > completed_hours for row in (*all_updates, *all_decisions, *all_caol)):
        raise ValueError("resume source has evidence beyond its one incomplete hour")

    _validate_program_prefix(
        plan=plan,
        profile=profile,
        updates=updates,
        decisions=decisions,
        program_versions=program_versions,
    )
    prefix_identity = _prefix_identity(
        next_step=next_step,
        program_versions=program_versions,
        forecast_inputs=forecast_inputs,
        performance=performance,
        zone_steps=zone_steps,
        updates=updates,
        decisions=decisions,
        caol=caol,
        calls=calls,
        raw_calls=raw_calls,
        attempts=attempts,
    )
    return ResumePrefix(
        source_run=directory,
        plan=plan,
        source_commit=source_commit,
        source_plan_identity=source_plan_identity,
        source_run_identity=source_run_identity,
        source_test_id=source_test_id,
        completed_hour=completed_hour,
        completed_step=completed_step,
        next_step=next_step,
        program_versions={str(zone): int(value) for zone, value in program_versions.items()},
        forecast_inputs=forecast_inputs,
        performance=tuple(performance),
        zone_steps=tuple(zone_steps),
        program_updates=tuple(updates),
        hourly_decisions=tuple(decisions),
        caol_records=tuple(caol),
        agent_calls=tuple(calls),
        raw_model_io=tuple(raw_calls),
        model_request_attempts=tuple(attempts),
        prefix_identity=prefix_identity,
    )


def recompute_resume_prefix_identity(run_dir: Path, lineage: Mapping[str, Any]) -> str:
    directory = run_dir.resolve()
    next_step = int(lineage["source_next_step"])
    completed_hours = int(lineage["source_completed_hour"]) + 1
    zone_steps = [
        row for row in _rows(directory / "zone_steps.jsonl") if int(row["step"]) < next_step
    ]
    updates = [
        row
        for row in _rows(directory / "program_updates.jsonl")
        if int(row["hour"]) < completed_hours
    ]
    decisions = _rows(directory / "hourly_decisions.jsonl")[:completed_hours]
    caol = [
        row for row in _rows(directory / "caol_records.jsonl") if int(row["hour"]) < completed_hours
    ]
    calls = [
        row for row in _rows(directory / "agent_calls.jsonl") if int(row["hour"]) < completed_hours
    ]
    raw_calls = [
        row for row in _rows(directory / "raw_model_io.jsonl") if int(row["hour"]) < completed_hours
    ]
    attempts = [
        row
        for row in _rows(directory / "model_request_attempts.jsonl")
        if int(row["hour"]) < completed_hours
    ]
    return _prefix_identity(
        next_step=next_step,
        program_versions=lineage["source_program_versions"],
        forecast_inputs=_object(directory / "forecast_inputs.json"),
        performance=_performance(directory / "performance.csv")[:next_step],
        zone_steps=zone_steps,
        updates=updates,
        decisions=decisions,
        caol=caol,
        calls=calls,
        raw_calls=raw_calls,
        attempts=attempts,
    )
