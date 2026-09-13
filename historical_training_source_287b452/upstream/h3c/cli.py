"""Command-line interface for planning, executing, verifying, and reporting H3C runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from h3c.causal.graph import load_graph
from h3c.causal.workflow import (
    confirm_graph,
    derive_graph,
    document_identity,
    prepare_proposal,
    propose_graph,
    read_document,
    validate_proposal,
    write_new,
)
from h3c.experiments.matrix import RunPlan, graph_mutation, plan_suite
from h3c.experiments.profiles import load_profile, profiles, repository_root
from h3c.experiments.settings import (
    load_diagnostic_window_catalog,
    load_runtime_contract,
)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _plan_view(plans: list[RunPlan]) -> dict[str, Any]:
    loaded = profiles()
    rows = []
    for plan in plans:
        profile = loaded[plan.profile]
        rows.append(
            {
                "profile": plan.profile,
                "method": plan.method_config(),
                "model_provider": plan.effective_model_provider(),
                "evaluation_start_seconds": plan.evaluation_start_seconds(profile),
                "run_identity": plan.identity(profile),
                "expected_agent_calls": plan.expected_agent_calls(len(profile["zones"])),
            }
        )
    return {
        "mode": "dry_plan",
        "run_count": len(rows),
        "expected_agent_calls": sum(row["expected_agent_calls"] for row in rows),
        "runs": rows,
    }


def _execute(plans: list[RunPlan], suite: str) -> None:
    from h3c.runtime.engine import execute_serial

    result = execute_serial(plans, suite=suite)
    _print(result)


def _run_plan(args: argparse.Namespace) -> RunPlan:
    mutation = None
    if args.graph_mutation == "missing-solar-zone-edge":
        mutation = graph_mutation("missing_solar_zone_edge")
    elif args.graph_mutation == "delayed-solar-zone-edge":
        mutation = graph_mutation("delayed_solar_zone_edge")
    elif args.graph_mutation == "immediate-solar-zone-edge":
        mutation = graph_mutation("immediate_solar_zone_edge")
    baseline = bool(args.baseline)
    if baseline and (
        args.working_memory_hours != 1
        or args.causal_off
        or args.independent_coordination
        or args.no_thinking
        or args.long_term_memory
        or mutation is not None
        or args.model_provider is not None
    ):
        raise SystemExit("--baseline cannot be combined with Agent-only flags")
    profile = load_profile(args.profile)
    diagnostic_window = args.diagnostic_window
    evaluation_hours = (
        int(load_diagnostic_window_catalog()[diagnostic_window]["evaluation_hours"])
        if diagnostic_window is not None
        else int(profile["protocol"]["formal_evaluation_days"]) * 24
    )
    return RunPlan(
        profile=args.profile,
        controller="deterministic_baseline" if baseline else "h3c_agent",
        working_memory_hours=args.working_memory_hours,
        causal_enabled=False if baseline else not args.causal_off,
        coordination_enabled=False if baseline else not args.independent_coordination,
        thinking_policy=(
            "all_roles_disabled" if baseline or args.no_thinking else "occupancy_routed"
        ),
        graph_mutation=None if baseline else mutation,
        evaluation_hours=evaluation_hours,
        long_term_memory=False if baseline else bool(args.long_term_memory),
        model_provider=(
            None
            if baseline
            else args.model_provider or load_runtime_contract()["model"]["default_provider"]
        ),
        diagnostic_window=diagnostic_window,
    )


def _graph_command(args: argparse.Namespace) -> None:
    if args.graph_action == "prepare":
        profile = load_profile(args.profile)
        proposal = prepare_proposal(args.profile, tuple(profile["zones"]), args.source)
        write_new(args.output, proposal)
        _print({"output": str(args.output), "identity": document_identity(proposal)})
    elif args.graph_action == "propose":
        proposal = propose_graph(read_document(args.input))
        write_new(args.output, proposal)
        _print({"output": str(args.output), "identity": document_identity(proposal)})
    elif args.graph_action == "validate":
        proposal = validate_proposal(read_document(args.input), required_status="proposed")
        _print({"valid": True, "identity": document_identity(proposal)})
    elif args.graph_action == "confirm":
        graph = confirm_graph(read_document(args.input), reviewer=args.reviewer, date=args.date)
        resolved = graph.resolved()
        write_new(args.output, resolved)
        _print({"output": str(args.output), "identity": document_identity(resolved)})
    else:
        graph = load_graph(args.input)
        mutation = read_document(args.mutation)
        derived = derive_graph(graph, mutation).resolved()
        write_new(args.output, derived)
        _print({"output": str(args.output), "identity": document_identity(derived)})


def _offline_command(args: argparse.Namespace, root: Path) -> None:
    from h3c.offline.artifacts import OfflineWorkspace
    from h3c.offline.service import (
        discover,
        export_workspace,
        load_onboarding_spec,
        resolved_plan,
        resume,
        verify_workspace,
    )

    if args.offline_action == "discover":
        spec = load_onboarding_spec(args.spec, root)
        if not args.execute:
            _print(resolved_plan(spec))
            return
        if args.reviewer is None:
            raise SystemExit("--reviewer is required with --execute")
        _print(discover(args.spec, root, reviewer=args.reviewer))
    elif args.offline_action == "resume":
        workspace = OfflineWorkspace.open(args.workspace)
        spec = load_onboarding_spec(workspace.path / "resolved_spec.json", root)
        if not args.execute:
            plan = resolved_plan(spec)
            plan.update(
                {
                    "mode": "resume_dry_plan",
                    "workspace": str(workspace.path),
                    "current_state": workspace.state(),
                }
            )
            _print(plan)
            return
        if args.reviewer is None:
            raise SystemExit("--reviewer is required with --execute")
        _print(resume(args.workspace, reviewer=args.reviewer))
    elif args.offline_action == "verify":
        result = verify_workspace(args.workspace, require_completion=True)
        _print(result)
        if not result["passed"]:
            raise SystemExit(1)
    else:
        _print(
            export_workspace(
                args.workspace,
                case_profile=args.case_profile,
                graph=args.graph,
                provenance=args.provenance,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="h3c")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="plan or execute one physical run")
    run.add_argument("--profile", choices=tuple(profiles()), required=True)
    run.add_argument("--baseline", action="store_true")
    run.add_argument(
        "--model-provider",
        choices=tuple(load_runtime_contract()["model"]["providers"]),
        help="select the registered OpenAI-compatible provider for Agent calls",
    )
    run.add_argument(
        "--diagnostic-window",
        choices=tuple(load_diagnostic_window_catalog()),
        help="use one registered non-formal evaluation window",
    )
    run.add_argument("--working-memory-hours", type=int, choices=(1, 2, 3), default=1)
    run.add_argument("--causal-off", action="store_true")
    run.add_argument("--independent-coordination", action="store_true")
    run.add_argument("--no-thinking", action="store_true")
    run.add_argument(
        "--long-term-memory",
        action="store_true",
        help="enable the optional three-regime per-zone experience store",
    )
    run.add_argument(
        "--graph-mutation",
        choices=(
            "none",
            "missing-solar-zone-edge",
            "delayed-solar-zone-edge",
            "immediate-solar-zone-edge",
        ),
        default="none",
    )
    run.add_argument("--execute", action="store_true")

    resume_run = commands.add_parser(
        "resume",
        help="audit or resume one eligible failed run by replaying its atomic physical prefix",
    )
    resume_run.add_argument("source_run", type=Path)
    resume_run.add_argument("--execute", action="store_true")

    suite = commands.add_parser("suite", help="plan or execute a registered suite serially")
    suite.add_argument(
        "name",
        choices=(
            "main",
            "memory",
            "causal-ablation",
            "graph-sensitivity",
            "coordination-ablation",
            "thinking-ablation",
            "all",
        ),
    )
    suite.add_argument("--execute", action="store_true")
    suite.add_argument(
        "--arm-index",
        type=int,
        help="select one registered suite arm for independently supervised execution",
    )

    verify = commands.add_parser("verify", help="verify one completed run")
    verify.add_argument("run_dir", type=Path)

    recertify = commands.add_parser(
        "recertify",
        help="append a zero-call re-verification to one immutable historical run",
    )
    recertify.add_argument("run_dir", type=Path)
    recertify.add_argument("--output", type=Path, required=True)
    recertify.add_argument("--recertifier-source-commit", required=True)

    report = commands.add_parser("report", help="build a report for one run or suite directory")
    report.add_argument("target", type=Path)
    report.add_argument("--reports-root", type=Path)

    graph = commands.add_parser("graph", help="run the human-in-the-loop graph workflow")
    graph_commands = graph.add_subparsers(dest="graph_action", required=True)
    prepare = graph_commands.add_parser("prepare")
    prepare.add_argument("--profile", choices=tuple(profiles()), required=True)
    prepare.add_argument("--source", action="append", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    propose = graph_commands.add_parser("propose")
    propose.add_argument("--input", type=Path, required=True)
    propose.add_argument("--output", type=Path, required=True)
    validate = graph_commands.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)
    confirm = graph_commands.add_parser("confirm")
    confirm.add_argument("--input", type=Path, required=True)
    confirm.add_argument("--reviewer", required=True)
    confirm.add_argument("--date", required=True)
    confirm.add_argument("--output", type=Path, required=True)
    derive = graph_commands.add_parser("derive")
    derive.add_argument("--input", type=Path, required=True)
    derive.add_argument("--mutation", type=Path, required=True)
    derive.add_argument("--output", type=Path, required=True)

    offline = commands.add_parser(
        "offline", help="onboard a case with Mapping and HITL causal discovery"
    )
    offline_commands = offline.add_subparsers(dest="offline_action", required=True)
    discover = offline_commands.add_parser(
        "discover", help="plan or start a fresh offline workflow"
    )
    discover.add_argument("--spec", type=Path, required=True)
    discover.add_argument("--reviewer")
    discover.add_argument("--execute", action="store_true")
    resume = offline_commands.add_parser("resume", help="plan or resume one checkpointed workflow")
    resume.add_argument("workspace", type=Path)
    resume.add_argument("--reviewer")
    resume.add_argument("--execute", action="store_true")
    offline_verify = offline_commands.add_parser("verify", help="verify a completed workspace")
    offline_verify.add_argument("workspace", type=Path)
    export = offline_commands.add_parser(
        "export", help="export one approved workspace without overwrite"
    )
    export.add_argument("workspace", type=Path)
    export.add_argument("--case-profile", type=Path, required=True)
    export.add_argument("--graph", type=Path, required=True)
    export.add_argument("--provenance", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = repository_root()
    if args.command == "run":
        plans = [_run_plan(args)]
        _execute(plans, "single-run") if args.execute else _print(_plan_view(plans))
    elif args.command == "resume":
        from h3c.runtime.engine import execute_resume, resume_plan

        _print(execute_resume(args.source_run) if args.execute else resume_plan(args.source_run))
    elif args.command == "suite":
        plans = plan_suite(args.name)
        if args.arm_index is not None:
            if not 0 <= args.arm_index < len(plans):
                raise SystemExit(f"suite arm index must be within 0..{len(plans) - 1}")
            plans = [plans[args.arm_index]]
        _execute(plans, args.name) if args.execute else _print(_plan_view(plans))
    elif args.command == "verify":
        from h3c.outputs.verification import verify_run

        result = verify_run(args.run_dir)
        _print(result)
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "recertify":
        from h3c.outputs.verification import recertify_run

        _print(
            recertify_run(
                args.run_dir,
                output_path=args.output,
                recertifier_source_commit=args.recertifier_source_commit,
            )
        )
    elif args.command == "report":
        from h3c.outputs.reporting import generate_report

        reports_root = args.reports_root or root / "outputs" / "reports"
        json_path, markdown_path = generate_report(args.target, reports_root)
        _print({"json": str(json_path), "markdown": str(markdown_path)})
    elif args.command == "graph":
        _graph_command(args)
    else:
        _offline_command(args, root)


if __name__ == "__main__":
    main()
