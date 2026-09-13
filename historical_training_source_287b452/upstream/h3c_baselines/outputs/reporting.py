"""Human-readable baseline tables and controller time-series figures."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

from h3c.experiments.profiles import repository_root
from h3c_baselines.configuration import formal_evaluation_plans, mpc_formal_evaluation_plans
from h3c_baselines.outputs.verification import (
    verify_baseline_run,
    verify_concurrent_suite_evidence,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not an object")
    return value


def _cumulative(values: list[float]) -> list[float]:
    total = 0.0
    result: list[float] = []
    for value in values:
        total += value
        result.append(total)
    return result


def _format_optional_metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.6f}"


def _run_summary(run_dir: Path) -> dict[str, Any]:
    manifest = _load(run_dir / "manifest.json")
    metrics = _load(run_dir / "metrics.json")
    completion = _load(run_dir / "completion.json")
    native_kpis = _load(run_dir / "native_boptest_kpis.json")
    resolved = _load(run_dir / "resolved_config.json")
    model_path = run_dir / "model_identity.json"
    model_identity = _load(model_path) if model_path.is_file() else None
    mpc_model_path = run_dir / "mpc_model_identity.json"
    mpc_model_identity = _load(mpc_model_path) if mpc_model_path.is_file() else None
    return {
        "case": manifest["case"],
        "controller": manifest["controller"],
        "classification": completion["classification"],
        "formal_execution_classification": completion["classification"],
        "source_commit": manifest["source_commit"],
        "run_identity": manifest["run_identity"],
        "plan_identity": resolved["plan_identity"],
        "model_sha256": None if model_identity is None else model_identity["sha256"],
        "mpc_model_identity": (
            None if mpc_model_identity is None else mpc_model_identity["model_identity"]
        ),
        "mpc_admission_mode": (
            None if mpc_model_identity is None else mpc_model_identity["admission_mode"]
        ),
        "mpc_validation_classification": (
            None if mpc_model_identity is None else mpc_model_identity["validation_classification"]
        ),
        "mpc_freeze_identity": (
            None if mpc_model_identity is None else mpc_model_identity["freeze_identity"]
        ),
        **metrics["physical"],
        **metrics["setpoint_dynamics"],
        "fallback_count": metrics["controller"]["fallback_count"],
        **{
            f"native_{key}": native_kpis.get(key)
            for key in (
                "cost_tot",
                "ener_tot",
                "emis_tot",
                "idis_tot",
                "tdis_tot",
                "pdih_tot",
                "pele_tot",
                "pgas_tot",
                "time_rat",
            )
        },
        "run_dir": str(run_dir),
    }


def _discover_runs(source: Path | Sequence[Path]) -> list[Path]:
    sources = [source] if isinstance(source, Path) else list(source)
    if not sources:
        raise ValueError("report requires at least one source")
    runs: list[Path] = []
    for item in sources:
        target = item.resolve()
        if (target / "completion.json").is_file():
            runs.append(target)
        else:
            runs.extend(
                path.parent.resolve()
                for path in target.rglob("completion.json")
                if (path.parent / "metrics.json").is_file()
            )
    if not runs:
        raise ValueError("report source contains no completed baseline runs")
    unique = sorted(set(runs), key=str)
    if len(unique) != len(runs):
        raise ValueError("report sources contain duplicate completed runs")
    return unique


def _registered_benchmark_keys() -> set[tuple[str, str, str]]:
    return {
        (plan.case, plan.controller, str(plan.resolved()["plan_identity"]))
        for plan in [*formal_evaluation_plans(), *mpc_formal_evaluation_plans()]
    }


_HISTORICAL_IDENTITY_COMPATIBILITY_ERRORS = frozenset(
    {"completion_identity", "execution_identity", "execution_identity_schema"}
)


def _historical_verification_is_compatible(
    current: dict[str, Any], persisted: dict[str, Any], completion: dict[str, Any]
) -> bool:
    errors = set(current.get("errors", []))
    checks = current.get("checks")
    return (
        bool(errors)
        and errors <= _HISTORICAL_IDENTITY_COMPATIBILITY_ERRORS
        and isinstance(checks, dict)
        and all(bool(value) for name, value in checks.items() if name not in errors)
        and persisted.get("execution_integrity") is True
        and persisted.get("completion_eligible") is True
        and persisted.get("errors") == []
        and persisted.get("classification") == completion.get("classification")
        and persisted.get("metrics_identity") == current.get("metrics_identity")
    )


def _verify_report_runs(runs: Sequence[Path]) -> dict[Path, str]:
    modes: dict[Path, str] = {}
    for run in runs:
        verification = verify_baseline_run(run)
        if (
            verification.get("execution_integrity") is not True
            or verification.get("completion_eligible") is not True
        ):
            persisted_path = run / "verification.json"
            if not persisted_path.is_file() or not _historical_verification_is_compatible(
                verification,
                _load(persisted_path),
                _load(run / "completion.json"),
            ):
                raise ValueError(f"baseline run is not report-eligible: {run}")
            modes[run] = "source-native-persisted-plus-current-metrics-replay"
            continue
        completion = _load(run / "completion.json")
        if verification.get("classification") != completion.get("classification"):
            raise ValueError(f"baseline classification is inconsistent: {run}")
        modes[run] = "current-replay"
    return modes


def _plot_run(run_dir: Path, destination: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install H3C[baselines] to generate baseline figures") from error
    with (run_dir / "performance.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("cannot plot an empty baseline run")
    hours = [index * 0.25 for index in range(len(rows))]
    temperatures = [json.loads(row["zone_temperatures_c"]) for row in rows]
    setpoints = [json.loads(row["zone_setpoints_c"]) for row in rows]
    pmv = [json.loads(row["zone_pmv"]) for row in rows]
    occupancy = [json.loads(row["zone_occupancy"]) for row in rows]
    power = [float(row["total_power_w"]) for row in rows]
    cumulative_cost = _cumulative([float(row["step_cost"]) for row in rows])
    resolved = _load(run_dir / "resolved_config.json")
    zones = list(resolved["case_profile"]["zones"])
    figure, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    for zone in range(len(temperatures[0])):
        label = zones[zone]
        axes[0].plot(hours, [row[zone] for row in temperatures], label=label)
        axes[1].plot(hours, [row[zone] for row in setpoints])
        axes[2].plot(hours, [row[zone] for row in pmv])
        axes[3].step(hours, [row[zone] for row in occupancy], where="post")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(ncol=min(5, len(temperatures[0])))
    axes[1].set_ylabel("Setpoint (°C)")
    axes[2].axhspan(-0.5, 0.5, color="#2ca02c", alpha=0.1)
    axes[2].set_ylabel("PMV")
    axes[3].set_ylabel("Occupancy")
    axes[4].plot(hours, cumulative_cost, color="#9467bd")
    axes[4].set_ylabel("Cumulative cost")
    axes[5].plot(hours, power, color="#d62728")
    axes[5].set_ylabel("Power (W)")
    axes[5].set_xlabel("Evaluation hour")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_case_comparison(
    case: str, runs: Sequence[tuple[Path, dict[str, Any]]], destination: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install H3C[baselines] to generate baseline figures") from error
    figure, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    for run_dir, summary in runs:
        with (run_dir / "performance.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"cannot plot an empty baseline run: {run_dir}")
        hours = [index * 0.25 for index in range(len(rows))]
        temperatures = [json.loads(row["zone_temperatures_c"]) for row in rows]
        setpoints = [json.loads(row["zone_setpoints_c"]) for row in rows]
        pmv = [json.loads(row["zone_pmv"]) for row in rows]
        occupancy = [json.loads(row["zone_occupancy"]) for row in rows]
        label = str(summary["controller"])
        axes[0].plot(
            hours,
            _cumulative([float(row["step_cost"]) for row in rows]),
            label=label,
        )
        axes[1].plot(hours, [float(row["total_power_w"]) for row in rows], label=label)
        axes[2].plot(
            hours,
            [sum(float(value) for value in values) / len(values) for values in temperatures],
            label=label,
        )
        axes[3].plot(
            hours,
            [
                max(
                    (
                        abs(float(value))
                        for value, occupied in zip(values, occupied_values, strict=True)
                        if float(occupied) > 0
                    ),
                    default=float("nan"),
                )
                for values, occupied_values in zip(pmv, occupancy, strict=True)
            ],
            label=label,
        )
        axes[4].plot(
            hours,
            [sum(float(value) for value in values) / len(values) for values in setpoints],
            label=label,
        )
        axes[5].step(
            hours,
            [sum(float(value) for value in values) for values in occupancy],
            where="post",
            label=label,
        )
    axes[0].set_ylabel("Cumulative cost")
    axes[0].legend(ncol=min(5, len(runs)))
    axes[1].set_ylabel("Power (W)")
    axes[2].set_ylabel("Mean temperature (°C)")
    axes[3].axhline(0.5, color="#2ca02c", alpha=0.5, linestyle="--")
    axes[3].set_ylabel("Occupied max |PMV|")
    axes[4].set_ylabel("Mean setpoint (°C)")
    axes[5].set_ylabel("Total occupancy")
    axes[5].set_xlabel("Evaluation hour")
    figure.suptitle(f"{case}: formal controller trajectories")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def generate_report(
    source: Path | Sequence[Path],
    output_root: Path | None = None,
    *,
    require_complete_benchmark: bool = False,
    mpc_suite_evidence: Path | None = None,
) -> dict[str, Any]:
    runs = _discover_runs(source)
    verification_modes = _verify_report_runs(runs)
    suite_identity: str | None = None
    if mpc_suite_evidence is not None:
        suite_path = mpc_suite_evidence.resolve()
        suite_verification = verify_concurrent_suite_evidence(suite_path)
        if suite_verification.get("valid") is not True:
            raise ValueError("MPC suite evidence is invalid")
        suite = _load(suite_path)
        suite_runs = {
            Path(str(arm["run_dir"])).resolve() for arm in suite["arms"] if isinstance(arm, dict)
        }
        report_mpc_runs = {
            run
            for run in runs
            if _load(run / "manifest.json").get("controller") == "hierarchical-mpc"
        }
        if suite_runs != report_mpc_runs:
            raise ValueError("MPC report runs do not exactly match suite evidence")
        suite_identity = str(suite["suite_identity"])
    destination = (
        output_root
        or repository_root()
        / "outputs"
        / "baselines"
        / "reports"
        / datetime.now(UTC).strftime("baseline-report-%Y%m%dT%H%M%SZ")
    ).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    summaries = [_run_summary(run) for run in runs]
    for index, run in enumerate(runs):
        summaries[index]["verification_mode"] = verification_modes[run]
    actual_keys = {
        (str(row["case"]), str(row["controller"]), str(row["plan_identity"])) for row in summaries
    }
    expected_keys = _registered_benchmark_keys()
    benchmark_complete = actual_keys == expected_keys and len(summaries) == len(expected_keys)
    if require_complete_benchmark and not benchmark_complete:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"registered benchmark is incomplete: missing={missing}, extra={extra}")
    report = {
        "schema": "h3c_baseline_report",
        "schema_version": 1,
        "run_count": len(summaries),
        "registered_benchmark_complete": benchmark_complete,
        "mpc_suite_identity": suite_identity,
        "runs": summaries,
    }
    (destination / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    columns = tuple(summaries[0])
    with (destination / "results.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# H3C baseline report",
        "",
        f"Registered 14-arm benchmark complete: **{str(benchmark_complete).lower()}**",
        "",
        "| Case | Controller | Execution | Model admission | Fallbacks | Cost | Energy (kWh) | Reward | Zone-h | PMV·h | Peak | TV (°C) | Reversals | Crossings |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        admission = row["mpc_validation_classification"] or "—"
        lines.append(
            f"| {row['case']} | {row['controller']} | {row['classification']} | "
            f"{admission} | {row['fallback_count']} | "
            f"{row['total_cost']:.6f} | {row['energy_kwh']:.6f} | {row['reward']:.6f} | "
            f"{row['discomfort_zone_hours']:.3f} | {row['discomfort_pmv_hours']:.6f} | "
            f"{row['occupied_peak_absolute_pmv']:.3f} | {row['total_variation_c']:.3f} | "
            f"{row['direction_reversals']} | {row['occupied_comfort_band_crossings']} |"
        )
    lines.extend(
        [
            "",
            "## Native BOPTEST KPIs",
            "",
            "| Case | Controller | cost_tot | ener_tot | emis_tot | idis_tot | tdis_tot |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['case']} | {row['controller']} | "
            f"{_format_optional_metric(row['native_cost_tot'])} | "
            f"{_format_optional_metric(row['native_ener_tot'])} | "
            f"{_format_optional_metric(row['native_emis_tot'])} | "
            f"{_format_optional_metric(row['native_idis_tot'])} | "
            f"{_format_optional_metric(row['native_tdis_tot'])} |"
        )
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for index, run in enumerate(runs):
        summary = summaries[index]
        _plot_run(
            run,
            destination / f"{summary['case']}_{summary['controller']}_timeseries.png",
        )
    for case in sorted({str(summary["case"]) for summary in summaries}):
        case_runs = [
            (run, summaries[index])
            for index, run in enumerate(runs)
            if summaries[index]["case"] == case
        ]
        _plot_case_comparison(
            case,
            case_runs,
            destination / f"{case}_controller_comparison.png",
        )
    return {
        "report_dir": str(destination),
        "run_count": len(runs),
        "registered_benchmark_complete": benchmark_complete,
    }
