from __future__ import annotations

import contextlib
import ctypes
import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import portalocker

from .io import append_jsonl, atomic_json, utc_now


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@dataclass(frozen=True)
class Lease:
    lease_id: str
    run_uuid: str
    task: str
    seed: int
    slots: int
    hostname: str
    pid: int


class CapacityManager:
    """Cross-process worker capacity with stale-owner recovery.

    A training job reserves five slots for its whole lifetime: four training
    simulators and one deterministic train-window evaluator. Two jobs therefore
    fit safely on a twelve-worker BOPTEST deployment.
    """

    def __init__(
        self,
        runtime_dir: Path,
        *,
        capacity: int = 12,
        heartbeat_timeout: float = 180.0,
    ):
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.state_path = self.runtime_dir / "worker_leases.json"
        self.lock_path = self.runtime_dir / "worker_leases.lock"
        self.audit_path = self.runtime_dir / "worker_leases.jsonl"

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with portalocker.Lock(str(self.lock_path), mode="a+", timeout=30):
            yield

    def _read(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema": "worker-leases-v1", "capacity": self.capacity, "leases": {}}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {"schema": "worker-leases-v1", "capacity": self.capacity, "leases": {}}
        if int(state.get("capacity", self.capacity)) != self.capacity:
            raise RuntimeError("Worker capacity changed while leases exist")
        state.setdefault("leases", {})
        return state

    def _stale(self, record: dict[str, Any], now: float) -> bool:
        if record.get("hostname") != socket.gethostname():
            return now - float(record.get("heartbeat_epoch", 0.0)) > self.heartbeat_timeout
        return not _pid_alive(int(record.get("pid", -1)))

    def cleanup_stale(self) -> list[str]:
        removed: list[str] = []
        with self._locked():
            state = self._read()
            now = time.time()
            for lease_id, record in list(state["leases"].items()):
                if self._stale(record, now):
                    removed.append(lease_id)
                    state["leases"].pop(lease_id, None)
                    append_jsonl(
                        self.audit_path,
                        {**record, "event": "stale_released", "timestamp": utc_now()},
                    )
            atomic_json(self.state_path, state)
        return removed

    def acquire(
        self,
        *,
        task: str,
        seed: int,
        run_uuid: str,
        slots: int = 5,
        wait_seconds: float = 0.0,
    ) -> Lease:
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        while True:
            with self._locked():
                state = self._read()
                now = time.time()
                for lease_id, record in list(state["leases"].items()):
                    if self._stale(record, now):
                        state["leases"].pop(lease_id, None)
                        append_jsonl(
                            self.audit_path,
                            {**record, "event": "stale_released", "timestamp": utc_now()},
                        )
                used = sum(int(item["slots"]) for item in state["leases"].values())
                if used + slots <= self.capacity:
                    lease = Lease(
                        lease_id=uuid.uuid4().hex,
                        run_uuid=run_uuid,
                        task=task,
                        seed=int(seed),
                        slots=int(slots),
                        hostname=socket.gethostname(),
                        pid=os.getpid(),
                    )
                    record = {
                        **lease.__dict__,
                        "created_at": utc_now(),
                        "heartbeat_at": utc_now(),
                        "heartbeat_epoch": now,
                    }
                    state["leases"][lease.lease_id] = record
                    atomic_json(self.state_path, state)
                    append_jsonl(self.audit_path, {**record, "event": "acquired"})
                    return lease
                atomic_json(self.state_path, state)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Insufficient BOPTEST capacity: requested={slots}, "
                    f"capacity={self.capacity}; another task must finish first"
                )
            time.sleep(2.0)

    def heartbeat(self, lease: Lease) -> None:
        with self._locked():
            state = self._read()
            record = state["leases"].get(lease.lease_id)
            if record is None:
                raise RuntimeError("Worker lease disappeared")
            record["heartbeat_at"] = utc_now()
            record["heartbeat_epoch"] = time.time()
            atomic_json(self.state_path, state)

    def release(self, lease: Lease) -> None:
        with self._locked():
            state = self._read()
            record = state["leases"].pop(lease.lease_id, None)
            atomic_json(self.state_path, state)
            if record:
                append_jsonl(
                    self.audit_path, {**record, "event": "released", "timestamp": utc_now()}
                )

    @contextlib.contextmanager
    def hold(self, **kwargs: Any) -> Iterator[Lease]:
        lease = self.acquire(**kwargs)
        stop = threading.Event()

        def beat() -> None:
            while not stop.wait(30.0):
                try:
                    self.heartbeat(lease)
                except Exception:
                    return

        thread = threading.Thread(target=beat, daemon=True, name=f"lease-{lease.lease_id[:8]}")
        thread.start()
        try:
            yield lease
        finally:
            stop.set()
            thread.join(timeout=2.0)
            self.release(lease)


@contextlib.contextmanager
def exclusive_run_lock(run_dir: Path) -> Iterator[None]:
    lock_path = Path(run_dir) / "runtime" / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(str(lock_path), mode="a+", timeout=0):
            yield
    except portalocker.exceptions.LockException as exc:
        raise RuntimeError(f"Another process already owns this run: {run_dir}") from exc
