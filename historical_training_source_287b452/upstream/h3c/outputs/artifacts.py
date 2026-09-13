"""One owner for immutable run paths and atomic completion publication."""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

STREAM_FILES = (
    "physical_conditioning.jsonl",
    "zone_steps.jsonl",
    "hourly_decisions.jsonl",
    "program_updates.jsonl",
    "agent_calls.jsonl",
    "raw_model_io.jsonl",
    "model_request_attempts.jsonl",
    "timing.jsonl",
    "caol_records.jsonl",
    "long_term_memory_crud.jsonl",
)
JSON_FILES = (
    "resolved_config.yaml",
    "manifest.json",
    "metrics.json",
    "verification.json",
)
PERFORMANCE_COLUMNS = (
    "time_seconds",
    "hour",
    "step",
    "total_power_w",
    "step_cost",
    "step_reward",
    "zone_temperatures_c",
    "zone_setpoints_c",
    "zone_pmv",
    "zone_occupancy",
)
COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class ArtifactError(RuntimeError):
    pass


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _component(value: str, name: str) -> str:
    if not isinstance(value, str) or not COMPONENT.fullmatch(value):
        raise ArtifactError(f"{name} is not a safe path component")
    return value


class RunArtifacts:
    def __init__(self, root: Path, suite: str, case: str, run_id: str) -> None:
        base = root.resolve()
        self.run_dir = (
            base
            / _component(suite, "suite")
            / _component(case, "case")
            / _component(run_id, "run id")
        ).resolve()
        if not self.run_dir.is_relative_to(base):
            raise ArtifactError("run path escaped the configured output root")

    def create(self, resolved_config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
        if self.run_dir.exists():
            raise ArtifactError("fresh run directory already exists")
        self.run_dir.mkdir(parents=True)
        self._write_new_json("resolved_config.yaml", resolved_config)
        self._write_new_json("manifest.json", manifest)
        for name in STREAM_FILES:
            (self.run_dir / name).touch(exist_ok=False)
        with (self.run_dir / "performance.csv").open("x", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(PERFORMANCE_COLUMNS)

    def _write_new_json(self, name: str, value: Any) -> None:
        path = self.run_dir / name
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(_json_text(value) + "\n")

    def append_jsonl(self, name: str, value: Mapping[str, Any]) -> None:
        if name not in STREAM_FILES:
            raise ArtifactError("unknown JSON-lines artifact")
        with (self.run_dir / name).open("a", encoding="utf-8", newline="\n") as file:
            file.write(_json_text(value) + "\n")

    def append_performance(self, row: Sequence[Any]) -> None:
        if len(row) != len(PERFORMANCE_COLUMNS):
            raise ArtifactError("performance row does not match the registered columns")
        with (self.run_dir / "performance.csv").open("a", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(row)

    def replace_manifest(self, manifest: Mapping[str, Any]) -> None:
        path = self.run_dir / "manifest.json"
        pending = self.run_dir / ".manifest.pending"
        if (
            not path.is_file()
            or pending.exists()
            or (self.run_dir / "completion.json").exists()
            or (self.run_dir / "failure.json").exists()
        ):
            raise ArtifactError("manifest cannot be replaced in the current run state")
        with pending.open("x", encoding="utf-8", newline="\n") as file:
            file.write(_json_text(manifest) + "\n")
            file.flush()
            os.fsync(file.fileno())
        pending.replace(path)

    def _replace_runtime_state(self, name: str, value: Mapping[str, Any]) -> None:
        if name not in {"dispatch_state.json", "completed_hour_checkpoint.json"}:
            raise ArtifactError("unknown runtime-state artifact")
        if (self.run_dir / "completion.json").exists() or (self.run_dir / "failure.json").exists():
            raise ArtifactError("runtime state cannot change after completion")
        path = self.run_dir / name
        pending = self.run_dir / f".{name}.pending"
        if pending.exists():
            raise ArtifactError("runtime-state publication is already pending")
        with pending.open("x", encoding="utf-8", newline="\n") as file:
            file.write(_json_text(value) + "\n")
            file.flush()
            os.fsync(file.fileno())
        pending.replace(path)

    def replace_dispatch_state(self, value: Mapping[str, Any]) -> None:
        self._replace_runtime_state("dispatch_state.json", value)

    def replace_completed_hour_checkpoint(self, value: Mapping[str, Any]) -> None:
        self._replace_runtime_state("completed_hour_checkpoint.json", value)

    def write_metrics(self, metrics: Mapping[str, Any]) -> None:
        self._write_new_json("metrics.json", metrics)

    def write_forecast_inputs(self, value: Mapping[str, Any]) -> None:
        self._write_new_json("forecast_inputs.json", value)

    def write_verification(self, verification: Mapping[str, Any]) -> None:
        self._write_new_json("verification.json", verification)

    def publish_completion(self, completion: Mapping[str, Any]) -> Path:
        if (self.run_dir / "failure.json").exists():
            raise ArtifactError("completion has already been attempted")
        verification = json.loads((self.run_dir / "verification.json").read_text(encoding="utf-8"))
        if verification.get("completion_eligible") is not True:
            raise ArtifactError("completion is forbidden for an execution-invalid run")
        if completion.get("classification") != verification.get("classification"):
            raise ArtifactError("completion classification does not match verification")
        if not (self.run_dir / "metrics.json").is_file():
            raise ArtifactError("metrics and verification must exist before completion")
        completion_path = self.run_dir / "completion.json"
        pending = self.run_dir / ".completion.pending"
        if completion_path.exists() or pending.exists():
            raise ArtifactError("completion has already been attempted")
        with pending.open("x", encoding="utf-8", newline="\n") as file:
            file.write(_json_text(completion) + "\n")
            file.flush()
            os.fsync(file.fileno())
        pending.replace(completion_path)
        return completion_path

    def publish_failure(self, failure: Mapping[str, Any]) -> Path:
        failure_path = self.run_dir / "failure.json"
        pending = self.run_dir / ".failure.pending"
        if failure_path.exists() or pending.exists() or (self.run_dir / "completion.json").exists():
            raise ArtifactError("failure has already been attempted")
        with pending.open("x", encoding="utf-8", newline="\n") as file:
            file.write(_json_text(failure) + "\n")
            file.flush()
            os.fsync(file.fileno())
        pending.replace(failure_path)
        return failure_path

    def record_incomplete(
        self, *, metrics: Mapping[str, Any], verification: Mapping[str, Any]
    ) -> None:
        if verification.get("completion_eligible"):
            raise ArtifactError("a completion-eligible run must publish a completion marker")
        if not (self.run_dir / "metrics.json").exists():
            self.write_metrics(metrics)
        if not (self.run_dir / "verification.json").exists():
            self.write_verification(verification)
