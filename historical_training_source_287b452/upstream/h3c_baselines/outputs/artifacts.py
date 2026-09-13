"""Fresh baseline run artifacts without synthetic Agent evidence."""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
BASE_STREAMS = (
    "physical_conditioning.jsonl",
    "actions.jsonl",
    "controller_diagnostics.jsonl",
    "timing.jsonl",
)
DRL_STREAMS = ("observations.jsonl", "policy_inference.jsonl")
MPC_STREAMS = ("predictions.jsonl", "solver_trace.jsonl")
PERFORMANCE_COLUMNS = (
    "time_seconds",
    "step",
    "total_power_w",
    "step_cost",
    "step_reward",
    "zone_temperatures_c",
    "zone_setpoints_c",
    "zone_pmv",
    "zone_occupancy",
)


def _text(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


class BaselineArtifacts:
    def __init__(self, root: Path, suite: str, case: str, run_id: str, controller: str) -> None:
        for value, name in ((suite, "suite"), (case, "case"), (run_id, "run id")):
            if not COMPONENT.fullmatch(value):
                raise ValueError(f"{name} is not a safe path component")
        base = root.resolve()
        self.run_dir = (base / suite / case / run_id).resolve()
        if not self.run_dir.is_relative_to(base):
            raise ValueError("baseline output escaped its configured root")
        self.allowed_streams = BASE_STREAMS + (
            DRL_STREAMS
            if controller in {"c-drl", "h-drl"}
            else MPC_STREAMS
            if controller == "hierarchical-mpc"
            else ()
        )

    def create(
        self,
        resolved: Mapping[str, Any],
        manifest: Mapping[str, Any],
        *,
        reserved_by_execution_lock: bool = False,
    ) -> None:
        if reserved_by_execution_lock:
            if not self.run_dir.is_dir() or {path.name for path in self.run_dir.iterdir()} != {
                ".execution.lock"
            }:
                raise ValueError("reserved baseline run directory has unexpected contents")
        else:
            if self.run_dir.exists():
                raise ValueError("fresh baseline run directory already exists")
            self.run_dir.mkdir(parents=True)
        self.write_new_json("resolved_config.json", resolved)
        self.write_new_json("manifest.json", manifest)
        for name in self.allowed_streams:
            (self.run_dir / name).touch(exist_ok=False)
        with (self.run_dir / "performance.csv").open("x", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(PERFORMANCE_COLUMNS)

    def write_new_json(self, name: str, value: Any) -> None:
        with (self.run_dir / name).open("x", encoding="utf-8", newline="\n") as file:
            file.write(_text(value) + "\n")

    def replace_json(self, name: str, value: Any) -> None:
        path = self.run_dir / name
        pending = self.run_dir / f".{name}.pending"
        if not path.is_file() or pending.exists() or (self.run_dir / "completion.json").exists():
            raise ValueError(f"{name} cannot be replaced")
        with pending.open("x", encoding="utf-8", newline="\n") as file:
            file.write(_text(value) + "\n")
            file.flush()
            os.fsync(file.fileno())
        pending.replace(path)

    def append_jsonl(self, name: str, value: Mapping[str, Any]) -> None:
        if name not in self.allowed_streams:
            raise ValueError("unknown baseline JSON-lines artifact")
        with (self.run_dir / name).open("a", encoding="utf-8", newline="\n") as file:
            file.write(_text(value) + "\n")

    def append_performance(self, row: Sequence[Any]) -> None:
        if len(row) != len(PERFORMANCE_COLUMNS):
            raise ValueError("baseline performance row has the wrong length")
        with (self.run_dir / "performance.csv").open("a", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(row)

    def publish_completion(self, completion: Mapping[str, Any]) -> None:
        verification = json.loads((self.run_dir / "verification.json").read_text(encoding="utf-8"))
        if verification.get("completion_eligible") is not True:
            raise ValueError("baseline completion is forbidden by verification")
        pending = self.run_dir / ".completion.pending"
        target = self.run_dir / "completion.json"
        if pending.exists() or target.exists():
            raise ValueError("baseline completion already exists")
        with pending.open("x", encoding="utf-8", newline="\n") as file:
            file.write(_text(completion) + "\n")
            file.flush()
            os.fsync(file.fileno())
        pending.replace(target)

    def publish_failure(self, failure: Mapping[str, Any]) -> None:
        pending = self.run_dir / ".failure.pending"
        target = self.run_dir / "failure.json"
        if pending.exists() or target.exists() or (self.run_dir / "completion.json").exists():
            raise ValueError("baseline failure already exists or the run is complete")
        with pending.open("x", encoding="utf-8", newline="\n") as file:
            file.write(_text(failure) + "\n")
            file.flush()
            os.fsync(file.fileno())
        pending.replace(target)
