"""Build identity-safe run or suite comparisons from completed run artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

from h3c.outputs.verification import verify_run


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def _identity(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _completion_paths(target: Path) -> tuple[str, list[Path]]:
    resolved = target.resolve()
    if (resolved / "completion.json").is_file():
        return "run", [resolved / "completion.json"]
    paths = sorted(resolved.glob("*/*/completion.json"))
    if paths:
        return "suite", paths
    raise ValueError("report target is neither one completed run nor one suite directory")


def generate_report(target: Path, reports_root: Path) -> tuple[Path, Path]:
    target_kind, completions = _completion_paths(target)
    rows: list[dict[str, Any]] = []
    source_commits: set[str] = set()
    compatibility: dict[str, set[tuple[str, str, str, int]]] = {}
    for completion in completions:
        run_dir = completion.parent
        verification = verify_run(run_dir)
        if not verification["execution_integrity"]:
            raise ValueError(f"completed run failed execution-integrity verification: {run_dir}")
        manifest = _object(run_dir / "manifest.json")
        metrics = _object(run_dir / "metrics.json")
        resolved = _object(run_dir / "resolved_config.yaml")
        profile = resolved["case_profile"]
        method = resolved["method"]
        profile_name = str(profile["profile"])
        source_commits.add(str(manifest["source_commit"]))
        compatibility.setdefault(profile_name, set()).add(
            (
                _identity(profile["protocol"]),
                str(manifest["conditioning_prefix_identity"]),
                str(manifest["evaluation_boundary_identity"]),
                int(method["evaluation_hours"]),
            )
        )
        rows.append(
            {
                "run": run_dir.as_posix(),
                "profile": profile_name,
                "controller": manifest["controller"],
                "run_identity": manifest["run_identity"],
                "source_commit": manifest["source_commit"],
                "classification": verification["classification"],
                "execution_integrity": verification["execution_integrity"],
                "model_contract_clean": verification["model_contract_clean"],
                "conditioning_prefix_identity": manifest["conditioning_prefix_identity"],
                "evaluation_boundary_identity": manifest["evaluation_boundary_identity"],
                "protocol_identity": _identity(profile["protocol"]),
                "evaluation_hours": method["evaluation_hours"],
                "method": method,
                "metrics": metrics,
                "core_timeseries": {
                    "artifact": (run_dir / "performance.csv").as_posix(),
                    "fields": [
                        "time_seconds",
                        "total_power_w",
                        "step_cost",
                        "step_reward",
                        "zone_setpoints_c",
                        "zone_pmv",
                        "zone_occupancy",
                    ],
                },
            }
        )
    if not rows:
        raise ValueError("report target contains no eligible completed runs")
    if len(source_commits) != 1:
        raise ValueError("report target mixes completion artifacts from different source commits")
    incompatible = sorted(profile for profile, values in compatibility.items() if len(values) != 1)
    if incompatible:
        raise ValueError(
            "report comparison has incompatible protocol/prefix/boundary identities for: "
            + ", ".join(incompatible)
        )

    comparisons: list[dict[str, Any]] = []
    for profile in sorted({row["profile"] for row in rows}):
        profile_rows = [row for row in rows if row["profile"] == profile]
        baselines = [row for row in profile_rows if row["controller"] == "deterministic_baseline"]
        agents = [row for row in profile_rows if row["controller"] == "h3c_agent"]
        for baseline in baselines:
            for agent in agents:
                baseline_physical = baseline["metrics"]["physical"]
                agent_physical = agent["metrics"]["physical"]
                comparisons.append(
                    {
                        "profile": profile,
                        "baseline_run_identity": baseline["run_identity"],
                        "agent_run_identity": agent["run_identity"],
                        "agent_classification": agent["classification"],
                        "metric_delta_agent_minus_baseline": {
                            name: float(agent_physical[name]) - float(baseline_physical[name])
                            for name in (
                                "total_cost",
                                "energy_kwh",
                                "reward",
                                "discomfort_zone_hours",
                                "discomfort_pmv_hours",
                                "occupied_peak_absolute_pmv",
                            )
                        },
                        "mechanism": {
                            "program_decisions": agent["metrics"]["program_decisions"],
                            "action_assurance": agent["metrics"]["action_assurance"],
                            "orchestration": agent["metrics"]["orchestration"],
                            "rationale_telemetry": agent["metrics"]["rationale_telemetry"],
                        },
                        "api_efficiency": agent["metrics"]["model_calls"],
                    }
                )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    reports_root.mkdir(parents=True, exist_ok=True)
    json_path = reports_root / f"release-report-{timestamp}.json"
    markdown_path = reports_root / f"release-report-{timestamp}.md"
    payload = {
        "report_schema": "h3c_identity_safe_comparison",
        "schema_version": 2,
        "generated_at": timestamp,
        "target_kind": target_kind,
        "target": target.resolve().as_posix(),
        "source_commit": next(iter(source_commits)),
        "runs": rows,
        "comparisons": comparisons,
    }
    with json_path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")
    lines = [
        "# H3C identity-safe run report",
        "",
        f"Runs: {len(rows)}; comparisons: {len(comparisons)}",
        f"Source commit: `{next(iter(source_commits))}`",
        "",
    ]
    for row in rows:
        lines.extend(
            (
                f"## {row['profile']} — {row['run_identity']}",
                "",
                f"- Controller: `{row['controller']}`",
                f"- Classification: `{row['classification']}`",
                f"- Physical metrics: `{json.dumps(row['metrics']['physical'], sort_keys=True)}`",
                f"- API efficiency: `{json.dumps(row['metrics']['model_calls'], sort_keys=True)}`",
                f"- Rationale length telemetry: `{json.dumps(row['metrics']['rationale_telemetry'], sort_keys=True)}`",
                f"- Core time-series source: `{row['core_timeseries']['artifact']}`",
                "",
            )
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return json_path, markdown_path
