"""Command-line interface for independent H3C baselines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from h3c.experiments.profiles import load_profile
from h3c.experiments.settings import load_runtime_contract
from h3c_baselines.configuration import (
    BaselineRunPlan,
    formal_evaluation_plans,
    mpc_formal_evaluation_plans,
)
from h3c_baselines.models import verify_all_checkpoints
from h3c_baselines.mpc.refit import (
    refit_hierarchical_mpc,
    resolved_refit_plan,
    verify_refit_workspace,
)
from h3c_baselines.mpc.registry import (
    freeze_method_degraded_mpc_suite,
    freeze_validated_mpc_suite,
    resolved_method_degraded_freeze_plan,
    verify_frozen_mpc_suite,
)
from h3c_baselines.mpc.training import (
    resolved_training_plan,
    train_hierarchical_mpc,
    verify_frozen_mpc_model,
)
from h3c_baselines.mpc.validation import (
    execute_fresh_validation,
    resolved_validation_plan,
    verify_validation_workspace,
)
from h3c_baselines.outputs.reporting import generate_report
from h3c_baselines.outputs.verification import verify_baseline_run
from h3c_baselines.runtime.runner import (
    execute_baseline_plans,
    execute_formal_suite,
    execute_mpc_formal_suite,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="h3c-baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="inspect frozen DRL checkpoints")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    verify_models = model_commands.add_parser("verify")
    verify_models.add_argument("--identity-only", action="store_true")
    run = commands.add_parser("run", help="resolve or execute one formal baseline")
    run.add_argument("--case", required=True, choices=("SZ_Air", "MZ_Hydro", "MZ_Air"))
    run.add_argument(
        "--controller",
        required=True,
        choices=("basic-rbc", "enhanced-rbc", "c-drl", "h-drl", "hierarchical-mpc"),
    )
    run.add_argument("--execute", action="store_true")
    suite = commands.add_parser("suite", help="resolve or execute a registered suite")
    suite.add_argument("name", choices=("formal", "mpc-formal"))
    suite.add_argument("--execute", action="store_true")
    mpc = commands.add_parser("mpc", help="train or inspect hierarchical MPC models")
    mpc_commands = mpc.add_subparsers(dest="mpc_command", required=True)
    train = mpc_commands.add_parser("train")
    train.add_argument("--case", choices=("all",), default="all")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--max-fit-episodes", type=int, default=64)
    train.add_argument("--execute", action="store_true")
    refit = mpc_commands.add_parser("refit", help="refit from one preserved failed data bank")
    refit.add_argument("--source-run", type=Path, required=True)
    refit.add_argument("--execute", action="store_true")
    verify_mpc = mpc_commands.add_parser("verify-model")
    verify_mpc.add_argument("--case", required=True, choices=("SZ_Air", "MZ_Hydro", "MZ_Air"))
    verify_refit = mpc_commands.add_parser("verify-refit")
    verify_refit.add_argument("workspace", type=Path)
    validate = mpc_commands.add_parser("validate", help="fresh-validate and freeze refit models")
    validate.add_argument("--refit-workspace", type=Path, required=True)
    validate.add_argument("--execute", action="store_true")
    verify_validation = mpc_commands.add_parser("verify-validation")
    verify_validation.add_argument("workspace", type=Path)
    freeze_adverse = mpc_commands.add_parser(
        "freeze-adverse", help="explicitly publish a fully evidenced method-degraded suite"
    )
    freeze_adverse.add_argument("--validation-workspace", type=Path, required=True)
    freeze_adverse.add_argument("--execute", action="store_true")
    mpc_commands.add_parser("verify-frozen-suite")
    verify = commands.add_parser("verify")
    verify.add_argument("run_directory", type=Path)
    report = commands.add_parser("report")
    report.add_argument("source", type=Path, nargs="+")
    report.add_argument("--require-complete-benchmark", action="store_true")
    report.add_argument("--mpc-suite-evidence", type=Path)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _verify_output(path: Path) -> dict[str, Any]:
    return verify_baseline_run(path)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "models":
        _print(verify_all_checkpoints(load_cpu=not arguments.identity_only))
        return 0
    if arguments.command == "mpc":
        if arguments.mpc_command == "verify-model":
            result = verify_frozen_mpc_model(arguments.case)
            _print(result)
            return 0 if result["valid"] else 1
        if arguments.mpc_command == "verify-refit":
            result = verify_refit_workspace(arguments.workspace)
            _print(result)
            return 0 if result["valid"] else 1
        if arguments.mpc_command == "verify-validation":
            result = verify_validation_workspace(arguments.workspace)
            _print(result)
            return 0 if result["valid"] else 1
        if arguments.mpc_command == "freeze-adverse":
            plan = resolved_method_degraded_freeze_plan(arguments.validation_workspace)
            if not arguments.execute:
                _print(plan)
                return 0 if plan["validation_valid_for_admission"] else 1
            _print(freeze_method_degraded_mpc_suite(arguments.validation_workspace))
            return 0
        if arguments.mpc_command == "verify-frozen-suite":
            result = verify_frozen_mpc_suite()
            _print(result)
            return 0 if result["valid"] else 1
        if arguments.mpc_command == "validate":
            validation_plan = resolved_validation_plan(arguments.refit_workspace)
            if not arguments.execute:
                _print(validation_plan)
                return 0
            runtime = load_runtime_contract()
            endpoint_name = runtime["physical_service"]["endpoint_environment_variable"]
            endpoint = os.environ.get(endpoint_name, "").rstrip("/")
            if not endpoint:
                raise ValueError(f"{endpoint_name} is required for fresh MPC validation")
            validation = execute_fresh_validation(
                arguments.refit_workspace,
                endpoint=endpoint,
            )
            frozen = freeze_validated_mpc_suite(Path(validation["run_dir"]))
            _print({"validation": validation, "freeze": frozen})
            return 0
        if arguments.mpc_command == "refit":
            refit_plan = resolved_refit_plan(arguments.source_run)
            if not arguments.execute:
                _print(refit_plan)
                return 0
            _print(refit_hierarchical_mpc(arguments.source_run))
            return 0
        training_plan = resolved_training_plan(
            workers=arguments.workers,
            max_fit_episodes=arguments.max_fit_episodes,
        )
        if not arguments.execute:
            _print(training_plan)
            return 0
        runtime = load_runtime_contract()
        endpoint_name = runtime["physical_service"]["endpoint_environment_variable"]
        endpoint = os.environ.get(endpoint_name, "").rstrip("/")
        if not endpoint:
            raise ValueError(f"{endpoint_name} is required for MPC training")
        _print(
            train_hierarchical_mpc(
                endpoint=endpoint,
                workers=arguments.workers,
                maximum_fit_episodes=arguments.max_fit_episodes,
            )
        )
        return 0
    if arguments.command == "run":
        evaluation_hours = (
            int(load_profile(arguments.case)["protocol"]["formal_evaluation_days"]) * 24
        )
        run_plan = BaselineRunPlan(arguments.case, arguments.controller, evaluation_hours)
        if not arguments.execute:
            _print({"execution": False, "plan": run_plan.resolved()})
            return 0
        _print(execute_baseline_plans([run_plan], suite="manual"))
        return 0
    if arguments.command == "suite":
        plans = (
            formal_evaluation_plans()
            if arguments.name == "formal"
            else mpc_formal_evaluation_plans()
        )
        dry = {
            "execution": False,
            "suite": arguments.name,
            "evaluation_runs": [plan.resolved() for plan in plans],
        }
        if not arguments.execute:
            _print(dry)
            return 0
        _print(execute_formal_suite() if arguments.name == "formal" else execute_mpc_formal_suite())
        return 0
    if arguments.command == "verify":
        result = _verify_output(arguments.run_directory.resolve())
        _print(result)
        return 0 if result["execution_integrity"] else 1
    if arguments.command == "report":
        _print(
            generate_report(
                arguments.source,
                require_complete_benchmark=arguments.require_complete_benchmark,
                mpc_suite_evidence=arguments.mpc_suite_evidence,
            )
        )
        return 0
    raise AssertionError("unreachable baseline command")


if __name__ == "__main__":
    raise SystemExit(main())
