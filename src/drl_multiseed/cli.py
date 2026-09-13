from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .aggregate import aggregate_results
from .config import PROTOCOL_VERSION, TASKS, get_task, repository_root
from .http_recovery import (
    HTTP_RECOVERY_PROTOCOL_ID,
    is_retryable_boptest_error,
    recovery_delay,
    wait_until_healthy,
)
from .io import append_jsonl, atomic_json, utc_now
from .leases import CapacityManager, exclusive_run_lock
from .preflight import run_preflight
from .source_identity import code_commit, code_fingerprint


def _output_root(value: str | None) -> Path:
    return Path(value).resolve() if value else repository_root() / "runs"


def _run_dir(root: Path, mode: str, seed: int, task: str) -> Path:
    return root / mode / f"seed{seed}" / task


def _code_commit() -> str:
    return code_commit()


def _code_fingerprint() -> str:
    return code_fingerprint()


def _identity(
    run_dir: Path,
    task: str,
    seed: int,
    allow_host_migration: bool,
) -> dict[str, Any]:
    path = run_dir / "run_identity.json"
    hostname = socket.gethostname()
    spec = get_task(task)
    expected = {
        "task": task,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": spec.protocol_hash(),
        "config_hash": spec.scientific_hash(seed),
    }
    preflight_path = run_dir / "preflight.json"
    preflight_version = None
    if preflight_path.exists():
        preflight_version = json.loads(preflight_path.read_text(encoding="utf-8")).get(
            "boptest_version"
        )
    if path.exists():
        identity = json.loads(path.read_text(encoding="utf-8"))
        for field, value in expected.items():
            if identity.get(field, value) != value:
                raise RuntimeError(f"Run identity mismatch for {field}; refusing resume")
            identity.setdefault(field, value)
        recorded_boptest = identity.get("boptest_version")
        if recorded_boptest and preflight_version and recorded_boptest != preflight_version:
            raise RuntimeError(
                "BOPTEST version changed within this run; use a new output directory"
            )
        if preflight_version and not recorded_boptest:
            identity["boptest_version"] = preflight_version
            atomic_json(path, identity)
        recorded_commit = identity.get("code_commit")
        current_commit = _code_commit()
        current_fingerprint = _code_fingerprint()
        recorded_fingerprint = identity.get("code_fingerprint")
        commit_changed = bool(recorded_commit and recorded_commit != current_commit)
        fingerprint_changed = recorded_fingerprint not in {None, current_fingerprint}
        latest_checkpoint = run_dir / "checkpoints" / "latest.json"
        if fingerprint_changed and latest_checkpoint.exists():
            raise RuntimeError(
                "Scientific source identity changed after training progress was committed; "
                "refusing silent resume"
            )
        elif commit_changed and latest_checkpoint.exists():
            # CLI/recovery/documentation updates do not enter the scientific
            # fingerprint.  Permit them without pretending the training
            # configuration changed, and leave a permanent audit record.
            append_jsonl(
                run_dir / "code_upgrades.jsonl",
                {
                    "timestamp": utc_now(),
                    "event": "runtime_only_code_update",
                    "from_commit": recorded_commit,
                    "to_commit": current_commit,
                    "code_fingerprint": current_fingerprint,
                    "scientific_configuration_changed": False,
                },
            )
            identity["code_commit"] = current_commit
            identity["runtime_update_applied_at"] = utc_now()
            atomic_json(path, identity)
        elif commit_changed or fingerprint_changed:
            append_jsonl(
                run_dir / "identity_refresh.jsonl",
                {
                    "timestamp": utc_now(),
                    "event": "preflight_only_identity_refreshed",
                    "from_commit": recorded_commit,
                    "to_commit": current_commit,
                    "reason": "no_committed_checkpoint",
                },
            )
            identity["code_commit"] = current_commit
            identity["code_fingerprint"] = current_fingerprint
            atomic_json(path, identity)
        if identity.get("code_fingerprint") not in {None, current_fingerprint}:
            raise RuntimeError("Training code fingerprint changed; refusing silent resume")
        original = identity.get("hostname")
        if original and original != hostname:
            if not allow_host_migration:
                raise RuntimeError(
                    f"Run belongs to host {original}; pass --allow-host-migration after copying the complete run directory"
                )
            append_jsonl(
                run_dir / "host_migrations.jsonl",
                {
                    "timestamp": utc_now(),
                    "from": original,
                    "to": hostname,
                    "run_uuid": identity["run_uuid"],
                },
            )
            identity["hostname"] = hostname
            atomic_json(path, identity)
        return identity
    identity = {
        "run_uuid": f"{task}-seed{seed}-{uuid.uuid4().hex[:12]}",
        **expected,
        "hostname": hostname,
        "boptest_version": preflight_version,
        "created_at": utc_now(),
        "code_commit": _code_commit(),
        "code_fingerprint": _code_fingerprint(),
    }
    atomic_json(path, identity)
    return identity


def _cleanup_stale_runs(root: Path, endpoint: str) -> list[dict[str, Any]]:
    """Clean TestIDs only when the owning run lock is demonstrably free."""
    from .environment import cleanup_run_testids

    cleaned: list[dict[str, Any]] = []
    if not root.exists():
        return cleaned
    run_dirs = {path.parent.parent for path in root.glob("**/lifecycle/*.active.json")}
    for run_dir in sorted(run_dirs):
        try:
            with exclusive_run_lock(run_dir):
                for row in cleanup_run_testids(run_dir, endpoint):
                    cleaned.append({"run_dir": str(run_dir), **row})
        except RuntimeError:
            # A live process owns the run. Never stop its IDs.
            continue
    return cleaned


def _latest_committed_epoch(run_dir: Path) -> int:
    path = run_dir / "checkpoints" / "latest.json"
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("epoch", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # Do not classify checkpoint corruption as a transport issue.  The
        # trainer will load it and raise the authoritative integrity error.
        return 0


def _active_testid_count(run_dir: Path) -> int:
    lifecycle = run_dir / "lifecycle"
    return len(list(lifecycle.glob("*.active.json"))) if lifecycle.exists() else 0


def _record_http_recovery(run_dir: Path, **values: Any) -> None:
    append_jsonl(
        run_dir / "http_auto_resume.jsonl",
        {
            "timestamp": utc_now(),
            "protocol": HTTP_RECOVERY_PROTOCOL_ID,
            **values,
        },
    )


def _wait_for_clean_recovery(
    *,
    run_dir: Path,
    endpoint: str,
    timeout_seconds: float,
) -> bool:
    """Wait for BOPTEST and remove only this run's persisted TestIDs."""

    from .environment import cleanup_run_testids

    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if not wait_until_healthy(
            endpoint,
            timeout_seconds=min(remaining, 30.0),
            poll_seconds=5.0,
        ):
            continue
        cleanup_results = cleanup_run_testids(run_dir, endpoint)
        active = _active_testid_count(run_dir)
        _record_http_recovery(
            run_dir,
            event="cleanup_probe",
            active_testids=active,
            cleanup_results=cleanup_results,
        )
        if active == 0:
            return True
        time.sleep(min(10.0, max(0.0, deadline - time.monotonic())))


def _preflight_with_http_recovery(
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    attempt = 0
    while True:
        try:
            run_preflight(
                task=args.task,
                seed=args.seed,
                endpoint=args.endpoint,
                online=True,
                output=run_dir / "preflight.json",
            )
            return
        except BaseException as exc:
            if not is_retryable_boptest_error(exc) or attempt >= args.http_auto_resume_attempts:
                raise
            attempt += 1
            delay = recovery_delay(attempt, args.http_resume_backoff_seconds)
            print(
                f"[http-recovery] preflight transport failure: {type(exc).__name__}: {exc}; "
                f"retry {attempt}/{args.http_auto_resume_attempts} after {delay:.0f}s",
                flush=True,
            )
            _record_http_recovery(
                run_dir,
                event="preflight_retry_scheduled",
                attempt=attempt,
                error_type=type(exc).__name__,
                error=str(exc),
                delay_seconds=delay,
            )
            time.sleep(delay)
            if not wait_until_healthy(
                args.endpoint,
                timeout_seconds=args.http_health_timeout_seconds,
            ):
                _record_http_recovery(
                    run_dir,
                    event="preflight_health_wait_exhausted",
                    attempt=attempt,
                    timeout_seconds=args.http_health_timeout_seconds,
                )
                continue


def _annotate_runtime_recovery(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    restarts: int,
) -> None:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "http_recovery_protocol": HTTP_RECOVERY_PROTOCOL_ID,
            "http_auto_resume_attempts": int(args.http_auto_resume_attempts),
            "http_restarts_used": int(restarts),
        }
    )
    atomic_json(path, manifest)
    failure_path = run_dir / "last_failure.json"
    if restarts and failure_path.exists():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        failure.update({"resolved": True, "resolved_at": utc_now()})
        atomic_json(failure_path, failure)


def _train(args: argparse.Namespace) -> int:
    if args.continue_until_converged and args.mode != "full":
        raise ValueError("--continue-until-converged is only valid with --mode full")
    if args.http_auto_resume_attempts < 0:
        raise ValueError("--http-auto-resume-attempts must be non-negative")
    if args.http_resume_backoff_seconds <= 0 or args.http_health_timeout_seconds <= 0:
        raise ValueError("HTTP recovery delays must be positive")
    run_key = args.task
    spec = get_task(args.task)
    root = _output_root(args.output_root)
    run_dir = _run_dir(root, args.mode, args.seed, run_key)
    run_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_runs(root, args.endpoint)
    CapacityManager(root / "_runtime", capacity=args.worker_capacity).cleanup_stale()
    _preflight_with_http_recovery(args, run_dir)
    identity = _identity(run_dir, run_key, args.seed, args.allow_host_migration)
    manager = CapacityManager(root / "_runtime", capacity=args.worker_capacity)
    with exclusive_run_lock(run_dir):
        with manager.hold(
            task=run_key,
            seed=args.seed,
            run_uuid=identity["run_uuid"],
            slots=5,
            wait_seconds=args.lease_wait_seconds,
        ):
            resume = bool(args.resume)
            consecutive_failures = 0
            failed_epoch = _latest_committed_epoch(run_dir)
            total_restarts = 0
            while True:
                kwargs = dict(
                    seed=args.seed,
                    mode=args.mode,
                    endpoint=args.endpoint,
                    run_dir=run_dir,
                    resume=resume,
                    wandb_mode=args.wandb_mode,
                    device=args.device,
                    continue_until_converged=args.continue_until_converged,
                )
                try:
                    if spec.algorithm == "ppo":
                        from .ppo_training import train_ppo

                        train_ppo(spec, **kwargs)
                    else:
                        from .mappo_training import train_mappo

                        train_mappo(spec, **kwargs)
                    _annotate_runtime_recovery(
                        run_dir,
                        args=args,
                        restarts=total_restarts,
                    )
                    if total_restarts:
                        _record_http_recovery(
                            run_dir,
                            event="training_recovered",
                            committed_epoch=_latest_committed_epoch(run_dir),
                            total_restarts=total_restarts,
                        )
                    break
                except BaseException as exc:
                    committed_epoch = _latest_committed_epoch(run_dir)
                    retryable = is_retryable_boptest_error(exc)
                    if committed_epoch > failed_epoch:
                        consecutive_failures = 1
                    else:
                        consecutive_failures += 1
                    failed_epoch = committed_epoch
                    can_retry = retryable and consecutive_failures <= args.http_auto_resume_attempts
                    failure = {
                        "task": run_key,
                        "seed": args.seed,
                        "timestamp": utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "committed_epoch": committed_epoch,
                        "retryable_boptest_transport": retryable,
                        "auto_resume_scheduled": can_retry,
                        "consecutive_failures": consecutive_failures,
                        "max_consecutive_retries": args.http_auto_resume_attempts,
                        "resume_instruction": (
                            "Automatic epoch-boundary resume is active. If retries are exhausted, "
                            "rerun the identical command with --resume."
                        ),
                    }
                    atomic_json(run_dir / "last_failure.json", failure)
                    if not can_retry:
                        raise
                    total_restarts += 1
                    delay = recovery_delay(
                        consecutive_failures,
                        args.http_resume_backoff_seconds,
                    )
                    _record_http_recovery(
                        run_dir,
                        event="training_retry_scheduled",
                        attempt=consecutive_failures,
                        total_restarts=total_restarts,
                        committed_epoch=committed_epoch,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        delay_seconds=delay,
                    )
                    print(
                        f"[http-recovery] BOPTEST transport failure at committed epoch "
                        f"{committed_epoch}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    print(
                        f"[http-recovery] retry {consecutive_failures}/"
                        f"{args.http_auto_resume_attempts} in {delay:.0f}s; "
                        "the incomplete epoch will be replayed from latest checkpoint",
                        flush=True,
                    )
                    while True:
                        time.sleep(delay)
                        if _wait_for_clean_recovery(
                            run_dir=run_dir,
                            endpoint=args.endpoint,
                            timeout_seconds=args.http_health_timeout_seconds,
                        ):
                            break
                        consecutive_failures += 1
                        if consecutive_failures > args.http_auto_resume_attempts:
                            atomic_json(
                                run_dir / "last_failure.json",
                                {
                                    **failure,
                                    "timestamp": utc_now(),
                                    "auto_resume_scheduled": False,
                                    "consecutive_failures": consecutive_failures,
                                    "recovery_error": (
                                        "BOPTEST health/owned-TestID cleanup wait exhausted"
                                    ),
                                },
                            )
                            raise RuntimeError(
                                "BOPTEST automatic recovery attempts were exhausted "
                                "while waiting for service health and TestID cleanup"
                            ) from exc
                        delay = recovery_delay(
                            consecutive_failures,
                            args.http_resume_backoff_seconds,
                        )
                        _record_http_recovery(
                            run_dir,
                            event="health_wait_retry_scheduled",
                            attempt=consecutive_failures,
                            total_restarts=total_restarts,
                            committed_epoch=committed_epoch,
                            delay_seconds=delay,
                            health_timeout_seconds=args.http_health_timeout_seconds,
                        )
                        print(
                            f"[http-recovery] BOPTEST is still unavailable; health wait "
                            f"{consecutive_failures}/{args.http_auto_resume_attempts} "
                            f"will retry after {delay:.0f}s",
                            flush=True,
                        )
                    resume = True
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_best

    run_key = args.task
    spec = get_task(args.task)
    root = _output_root(args.output_root)
    run_dir = _run_dir(root, args.mode, args.seed, run_key)
    manager = CapacityManager(root / "_runtime", capacity=args.worker_capacity)
    manager.cleanup_stale()
    with exclusive_run_lock(run_dir):
        manifest_path = run_dir / "run_manifest.json"
        identity_path = run_dir / "run_identity.json"
        if not manifest_path.is_file() or not identity_path.is_file():
            raise RuntimeError("Formal evaluation requires a completed training run")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"early_stopped", "cap_reached"}:
            raise RuntimeError(
                "Formal evaluation is blocked until training reaches a terminal state"
            )
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        with manager.hold(
            task=f"{run_key}:formal_evaluation",
            seed=args.seed,
            run_uuid=str(identity["run_uuid"]),
            slots=1,
            wait_seconds=args.lease_wait_seconds,
        ):
            report = evaluate_best(
                spec,
                seed=args.seed,
                endpoint=args.endpoint,
                run_dir=run_dir,
                device=args.device,
                force=args.force,
            )
    print(json.dumps(report, indent=2))
    return 0


def _suite(args: argparse.Namespace) -> int:
    if args.continue_until_converged and args.mode != "full":
        raise ValueError("--continue-until-converged is only valid with --mode full")
    if args.max_parallel not in {1, 2}:
        raise ValueError("--max-parallel must be 1 or 2")
    if args.gpu_slots not in {0, 1, 2}:
        raise ValueError("--gpu-slots must be 0, 1 or 2")
    if args.http_auto_resume_attempts < 0:
        raise ValueError("--http-auto-resume-attempts must be non-negative")
    if args.http_resume_backoff_seconds <= 0 or args.http_health_timeout_seconds <= 0:
        raise ValueError("HTTP recovery delays must be positive")
    queue = list(args.tasks or TASKS)
    active: list[tuple[subprocess.Popen[str], str, bool]] = []
    failures: list[str] = []
    root = _output_root(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_runs(root, args.endpoint)
    CapacityManager(root / "_runtime", capacity=args.worker_capacity).cleanup_stale()
    interrupted = False
    try:
        while queue or active:
            while queue and len(active) < args.max_parallel:
                running_gpu = sum(int(is_gpu) for _, _, is_gpu in active)
                selected_index = None
                preferred_algorithm = None
                running_cases = {TASKS[task].case_key for _, task, _ in active}
                if active:
                    running_algorithms = {TASKS[task].algorithm for _, task, _ in active}
                    if len(running_algorithms) == 1:
                        preferred_algorithm = "mappo" if "ppo" in running_algorithms else "ppo"
                candidate_indices = list(range(len(queue)))
                candidate_indices.sort(
                    key=lambda index: (
                        bool(preferred_algorithm)
                        and TASKS[queue[index]].algorithm != preferred_algorithm,
                    )
                )
                for index in candidate_indices:
                    task = queue[index]
                    # Two agents for the same FMU testcase contend heavily in
                    # BOPTEST and can reduce throughput by an order of magnitude.
                    # Parallelism therefore means distinct building cases.
                    if TASKS[task].case_key in running_cases:
                        continue
                    needs_gpu = TASKS[task].algorithm == "mappo" and args.gpu_slots > 0
                    if not needs_gpu or running_gpu < args.gpu_slots:
                        selected_index = index
                        break
                if selected_index is None:
                    break
                task = queue.pop(selected_index)
                is_gpu = TASKS[task].algorithm == "mappo" and args.gpu_slots > 0
                device = "cuda" if is_gpu else "cpu"
                command = [
                    sys.executable,
                    "-m",
                    "drl_multiseed.cli",
                    "train",
                    "--task",
                    task,
                    "--seed",
                    str(args.seed),
                    "--mode",
                    args.mode,
                    "--endpoint",
                    args.endpoint,
                    "--output-root",
                    str(root),
                    "--wandb-mode",
                    args.wandb_mode,
                    "--device",
                    device,
                    "--worker-capacity",
                    str(args.worker_capacity),
                    "--lease-wait-seconds",
                    "86400",
                    "--http-auto-resume-attempts",
                    str(args.http_auto_resume_attempts),
                    "--http-resume-backoff-seconds",
                    str(args.http_resume_backoff_seconds),
                    "--http-health-timeout-seconds",
                    str(args.http_health_timeout_seconds),
                ]
                if args.resume:
                    command.append("--resume")
                if args.allow_host_migration:
                    command.append("--allow-host-migration")
                if args.continue_until_converged:
                    command.append("--continue-until-converged")
                environment = os.environ.copy()
                environment.update(
                    {
                        "OMP_NUM_THREADS": str(args.threads_per_task),
                        "MKL_NUM_THREADS": str(args.threads_per_task),
                        "OPENBLAS_NUM_THREADS": str(args.threads_per_task),
                        "NUMEXPR_NUM_THREADS": str(args.threads_per_task),
                    }
                )
                process = subprocess.Popen(
                    command, cwd=repository_root(), env=environment, text=True
                )
                active.append((process, task, is_gpu))
                print(f"[suite] started {task} pid={process.pid} device={device}")
            time.sleep(2.0)
            survivors: list[tuple[subprocess.Popen[str], str, bool]] = []
            for process, task, is_gpu in active:
                code = process.poll()
                if code is None:
                    survivors.append((process, task, is_gpu))
                elif code != 0:
                    failures.append(task)
                    print(f"[suite] FAILED {task} exit={code}")
                else:
                    print(f"[suite] completed {task}")
            active = survivors
    except KeyboardInterrupt:
        interrupted = True
        print("[suite] interrupted; terminating child trainers and cleaning their resources")
        raise
    finally:
        if active:
            for process, task, _ in active:
                if process.poll() is None:
                    print(f"[suite] terminating {task} pid={process.pid}")
                    process.terminate()
            deadline = time.monotonic() + 10.0
            for process, task, _ in active:
                if process.poll() is None:
                    try:
                        process.wait(timeout=max(0.1, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        print(f"[suite] killing unresponsive {task} pid={process.pid}")
                        process.kill()
                        process.wait(timeout=5.0)
        if interrupted or active:
            # Child termination can bypass its Python finally block. Run-lock
            # checks ensure this cannot stop an unrelated live trainer.
            for attempt in range(3):
                _cleanup_stale_runs(root, args.endpoint)
                CapacityManager(root / "_runtime", capacity=args.worker_capacity).cleanup_stale()
                if not list(root.glob("**/lifecycle/*.active.json")):
                    break
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
    if failures:
        raise RuntimeError(f"Suite failures: {failures}; rerun with --resume")
    return 0


def _status(args: argparse.Namespace) -> int:
    root = _output_root(args.output_root) / args.mode
    rows = []
    for path in root.glob("seed*/*/run_manifest.json") if root.exists() else ():
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    print(json.dumps(rows, indent=2))
    return 0


def _cleanup(args: argparse.Namespace) -> int:
    root = _output_root(args.output_root)
    cleaned = _cleanup_stale_runs(root, args.endpoint)
    released = CapacityManager(root / "_runtime", capacity=args.worker_capacity).cleanup_stale()
    print(json.dumps({"testids": cleaned, "leases": released}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible PPO/MAPPO training against an external BOPTEST service"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--endpoint", default=os.getenv("BOPTEST_URL", "http://127.0.0.1:8000"))
    common.add_argument("--output-root")
    common.add_argument("--mode", choices=("smoke", "full"), default="full")
    common.add_argument("--seed", type=int, default=1337)
    common.add_argument("--device", default="auto")

    train = sub.add_parser("train", parents=[common])
    train.add_argument("--task", choices=tuple(TASKS), required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--allow-host-migration", action="store_true")
    train.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    train.add_argument("--worker-capacity", type=int, default=12)
    train.add_argument("--lease-wait-seconds", type=float, default=0)
    train.add_argument("--continue-until-converged", action="store_true")
    train.add_argument("--http-auto-resume-attempts", type=int, default=12)
    train.add_argument("--http-resume-backoff-seconds", type=float, default=15.0)
    train.add_argument("--http-health-timeout-seconds", type=float, default=600.0)
    train.set_defaults(func=_train)

    suite = sub.add_parser("suite", parents=[common])
    suite.add_argument("--tasks", nargs="*", choices=tuple(TASKS))
    suite.add_argument("--max-parallel", type=int, default=1)
    suite.add_argument("--gpu-slots", type=int, default=1)
    suite.add_argument("--threads-per-task", type=int, default=4)
    suite.add_argument("--resume", action="store_true")
    suite.add_argument("--allow-host-migration", action="store_true")
    suite.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    suite.add_argument("--worker-capacity", type=int, default=12)
    suite.add_argument("--continue-until-converged", action="store_true")
    suite.add_argument("--http-auto-resume-attempts", type=int, default=12)
    suite.add_argument("--http-resume-backoff-seconds", type=float, default=15.0)
    suite.add_argument("--http-health-timeout-seconds", type=float, default=600.0)
    suite.set_defaults(func=_suite)

    evaluate = sub.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--task", choices=tuple(TASKS), required=True)
    evaluate.add_argument("--force", action="store_true")
    evaluate.add_argument("--worker-capacity", type=int, default=12)
    evaluate.add_argument("--lease-wait-seconds", type=float, default=0)
    evaluate.set_defaults(func=_evaluate)

    preflight = sub.add_parser("preflight", parents=[common])
    preflight.add_argument("--task", choices=tuple(TASKS))
    preflight.add_argument("--online", action="store_true")
    preflight.set_defaults(
        func=lambda a: (
            print(
                json.dumps(
                    run_preflight(
                        task=a.task,
                        seed=a.seed,
                        endpoint=a.endpoint,
                        online=a.online,
                        output=_output_root(a.output_root) / "preflight.json",
                    ),
                    indent=2,
                )
            ),
            0,
        )[1]
    )

    aggregate = sub.add_parser("aggregate", parents=[common])
    aggregate.set_defaults(
        func=lambda a: (
            print(json.dumps(aggregate_results(_output_root(a.output_root), a.mode), indent=2)),
            0,
        )[1]
    )
    status = sub.add_parser("status", parents=[common])
    status.set_defaults(func=_status)
    cleanup = sub.add_parser("cleanup", parents=[common])
    cleanup.add_argument("--worker-capacity", type=int, default=12)
    cleanup.set_defaults(func=_cleanup)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
