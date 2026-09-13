"""Committed repository identity used by all executable workflows."""

from __future__ import annotations

import subprocess
from pathlib import Path

from h3c.experiments.profiles import repository_root


def committed_source_identity(root: Path | None = None) -> str:
    """Return the exact committed source identity or fail closed."""

    repository = (root or repository_root()).resolve()
    common = ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository)]
    top_level = subprocess.run(
        [*common, "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != repository:
        raise RuntimeError("execution requires the repository root as its Git worktree")
    status = subprocess.run(
        [*common, "status", "--porcelain=v1", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("execution requires a clean committed Git worktree")
    completed = subprocess.run(
        [
            *common,
            "rev-parse",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40:
        raise RuntimeError("execution requires a committed Git source identity")
    return commit
