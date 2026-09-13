"""Bounded process-level recovery for transient BOPTEST transport failures.

An action request must not be retried in-place because the FMU may have advanced
even when its response was lost.  The safe recovery unit is therefore the last
atomically committed training epoch.
"""

from __future__ import annotations

import http.client
import socket
import ssl
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

HTTP_RECOVERY_PROTOCOL_ID = "boptest-epoch-auto-resume-v1"

_TRANSIENT_HTTP_TYPES = {
    "http_408",
    "http_425",
    "http_429",
    "http_500",
    "http_502",
    "http_503",
    "http_504",
}
_TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
_TRANSIENT_WINERRORS = {109, 10048, 10053, 10054, 10055, 10060, 10061}
_TRANSIENT_ERRNOS = {
    32,  # broken pipe
    98,  # address already in use
    99,  # address not available
    100,  # network down
    101,  # network unreachable
    103,  # connection aborted
    104,  # connection reset
    105,  # no buffer space
    109,  # Windows: the pipe has been ended (SubprocVecEnv worker exited)
    110,  # timed out
    111,  # connection refused
    113,  # no route to host
}
_TRANSIENT_MESSAGES = (
    "winerror 109",
    "winerror 10048",
    "winerror 10053",
    "winerror 10054",
    "winerror 10055",
    "winerror 10060",
    "winerror 10061",
    "connection reset",
    "connection aborted",
    "connection refused",
    "remote end closed connection",
    "remote disconnected",
    "broken pipe",
    "pipe has been ended",
    "pipe ended",
    "timed out",
    "no buffer space",
    "queue was full",
    "temporarily unavailable",
)


def exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception plus explicit/implicit causes and URL reasons."""

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for candidate in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
            getattr(current, "reason", None),
        ):
            if isinstance(candidate, BaseException):
                pending.append(candidate)


def is_retryable_boptest_error(error: BaseException) -> bool:
    """Recognize only transient HTTP/socket failures, including worker EOFs."""

    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return False
    for current in exception_chain(error):
        # Avoid importing the complete trainer stack in the CLI supervisor. The
        # transport contract is deliberately recognized by its stable public
        # attributes so startup remains light and child workers remain isolated.
        if type(current).__name__ == "TransportError":
            if (
                bool(getattr(current, "retryable", False))
                or getattr(current, "error_type", None) in _TRANSIENT_HTTP_TYPES
            ):
                return True
        if isinstance(current, urllib.error.HTTPError):
            if int(current.code) in _TRANSIENT_HTTP_CODES:
                return True
            continue
        if isinstance(current, EOFError):
            frames = traceback.extract_tb(current.__traceback__)
            if (
                any(
                    "multiprocessing" in frame.filename.lower()
                    or "subproc_vec_env" in frame.filename.lower()
                    for frame in frames
                )
                or "worker" in str(current).lower()
            ):
                return True
            continue
        if isinstance(
            current,
            (
                ConnectionError,
                TimeoutError,
                urllib.error.URLError,
                http.client.IncompleteRead,
                http.client.RemoteDisconnected,
                socket.gaierror,
                ssl.SSLEOFError,
                ssl.SSLZeroReturnError,
            ),
        ):
            return True
        if isinstance(current, OSError):
            if getattr(current, "winerror", None) in _TRANSIENT_WINERRORS:
                return True
            if getattr(current, "errno", None) in _TRANSIENT_ERRNOS:
                return True
        text = str(current).lower()
        if any(fragment in text for fragment in _TRANSIENT_MESSAGES):
            return True
    return False


def endpoint_is_healthy(endpoint: str, *, timeout_seconds: float = 10.0) -> bool:
    """Return whether BOPTEST answers its read-only version endpoint."""

    try:
        with urllib.request.urlopen(
            f"{endpoint.rstrip('/')}/version",
            timeout=timeout_seconds,
        ) as response:
            return int(getattr(response, "status", 200)) == 200 and bool(response.read())
    except Exception:
        return False


def wait_until_healthy(
    endpoint: str,
    *,
    timeout_seconds: float,
    poll_seconds: float = 10.0,
    health_check: Callable[..., bool] = endpoint_is_healthy,
    sleep: Callable[[float], Any] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """Wait boundedly for BOPTEST to answer, without mutating any testcase."""

    deadline = monotonic() + max(0.0, float(timeout_seconds))
    while True:
        if health_check(endpoint, timeout_seconds=min(10.0, max(1.0, poll_seconds))):
            return True
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False
        sleep(min(float(poll_seconds), remaining))


def recovery_delay(attempt: int, base_seconds: float, *, cap_seconds: float = 60.0) -> float:
    """Exponential recovery delay capped to keep progress visible."""

    if attempt < 1:
        raise ValueError("Recovery attempt numbers start at one")
    return min(float(cap_seconds), float(base_seconds) * (2 ** (attempt - 1)))
