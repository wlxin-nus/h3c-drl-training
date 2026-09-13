"""Single-owner physical execution lock shared by every H3C runner."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def physical_execution_lock(lock_root: Path) -> Iterator[Path]:
    """Exclusively hold ``.execution.lock`` under an explicit lock root."""
    root = lock_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".execution.lock"
    try:
        with lock.open("x", encoding="utf-8") as file:
            file.write(str(os.getpid()))
    except FileExistsError as error:
        try:
            recorded_pid = lock.read_text(encoding="utf-8").strip() or "unavailable"
        except OSError:
            recorded_pid = "unreadable"
        raise RuntimeError(
            f"execution lock already exists at {lock}; recorded PID={recorded_pid}. "
            "Do not delete it automatically: verify the process and active physical test "
            "read-only, then remove the stale lock manually only after confirming no run is active."
        ) from error
    try:
        yield lock
    finally:
        lock.unlink(missing_ok=False)
