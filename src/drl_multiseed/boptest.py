"""Minimal ID-scoped HTTP client for an externally managed BOPTEST service."""

from __future__ import annotations

import http.client
import json
import math
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class TransportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        error_type: str = "transport_contract_error",
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.error_type = error_type


def _connection_failure(error: BaseException) -> bool:
    cause: BaseException | object = (
        error.reason if isinstance(error, urllib.error.URLError) else error
    )
    return isinstance(
        cause,
        (
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            TimeoutError,
            socket.gaierror,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            ssl.SSLEOFError,
            ssl.SSLZeroReturnError,
            OSError,
        ),
    )


def request_json(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout_seconds: float = 600.0,
    response_json_required: bool = True,
) -> dict[str, Any] | str:
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            status = int(getattr(response, "status", 200))
            if status != 200:
                raise TransportError(
                    f"HTTP {status} from {url}",
                    retryable=status in {408, 425, 429, 500, 502, 503, 504},
                    error_type=f"http_{status}",
                )
    except urllib.error.HTTPError as exc:
        raise TransportError(
            f"HTTP {exc.code} from {url}",
            retryable=exc.code in {408, 425, 429, 500, 502, 503, 504},
            error_type=f"http_{exc.code}",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError, http.client.IncompleteRead) as exc:
        raise TransportError(
            f"request failed for {url}: {type(exc).__name__}",
            retryable=_connection_failure(exc),
            error_type=type(exc).__name__,
        ) from exc
    if not raw:
        if response_json_required:
            return {}
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        if not response_json_required:
            return {}
        raise TransportError(f"endpoint returned invalid JSON for {url}") from exc
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise TransportError(f"endpoint returned a non-object for {url}")
    return value


LifecycleSink = Callable[[Mapping[str, Any]], None]


class BoptestHttpClient:
    """One-client/one-TestID wrapper around the BOPTEST REST API."""

    def __init__(self, endpoint: str, *, queue_poll_seconds: float = 1.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.test_id: str | None = None
        self.testcase: str | None = None
        if not math.isfinite(queue_poll_seconds) or queue_poll_seconds <= 0:
            raise ValueError("BOPTEST queue poll interval must be positive")
        self.queue_poll_seconds = float(queue_poll_seconds)
        self._lifecycle_sink: LifecycleSink | None = None

    def set_lifecycle_sink(self, sink: LifecycleSink) -> None:
        self._lifecycle_sink = sink

    def _emit(self, event: str, status: str | None = None) -> None:
        if self._lifecycle_sink is None:
            return
        row: dict[str, Any] = {
            "phase": "physical_dispatch",
            "event": event,
            "dispatch_mode": "auto",
            "test_id": self.test_id,
            "testcase": self.testcase,
        }
        if status is not None:
            row["status"] = status
        self._lifecycle_sink(row)

    def select_testcase(self, testcase: str) -> str:
        if self.test_id is not None:
            raise TransportError("BOPTEST client already owns a live TestID")
        selected = request_json("POST", f"{self.endpoint}/testcases/{testcase}/select")
        if not isinstance(selected, Mapping):
            raise TransportError("BOPTEST select response is invalid")
        test_id = selected.get("testid")
        if not isinstance(test_id, str) or not test_id:
            raise TransportError("BOPTEST select did not return a TestID")
        self.test_id = test_id
        self.testcase = testcase
        self._emit("selected")
        self._wait_until_running()
        self._configure_selected()
        return test_id

    def status(self) -> str:
        if self.test_id is None:
            raise TransportError("BOPTEST status requested before select")
        response = request_json("GET", f"{self.endpoint}/status/{self.test_id}")
        status = response if isinstance(response, str) else response.get("payload")
        if status not in {"Running", "Queued"}:
            raise TransportError("BOPTEST status payload is invalid")
        return str(status)

    def _wait_until_running(self) -> None:
        previous: str | None = None
        while True:
            status = self.status()
            if status != previous:
                self._emit("status_changed", status)
                previous = status
            if status == "Running":
                return
            time.sleep(self.queue_poll_seconds)

    def _configure_selected(self) -> None:
        if self.test_id is None:
            raise TransportError("BOPTEST configure requested without a TestID")
        request_json(
            "PUT",
            f"{self.endpoint}/scenario/{self.test_id}",
            payload={"electricity_price": "dynamic"},
        )
        request_json("PUT", f"{self.endpoint}/step/{self.test_id}", payload={"step": 900})
        self._emit("configured", "Running")

    def initialize(
        self, testcase: str, start_time_seconds: int, warmup_period_seconds: int
    ) -> dict[str, Any]:
        self.select_testcase(testcase)
        return self.initialize_selected(start_time_seconds, warmup_period_seconds)

    def initialize_selected(
        self, start_time_seconds: int, warmup_period_seconds: int
    ) -> dict[str, Any]:
        if self.test_id is None:
            raise TransportError("BOPTEST initialize requested without a TestID")
        self._wait_until_running()
        response = request_json(
            "PUT",
            f"{self.endpoint}/initialize/{self.test_id}",
            payload={
                "start_time": int(start_time_seconds),
                "warmup_period": int(warmup_period_seconds),
            },
        )
        if not isinstance(response, Mapping) or not isinstance(response.get("payload"), dict):
            raise TransportError("BOPTEST initialize payload is invalid")
        self._emit("initialized", "Running")
        return dict(response["payload"])

    def forecast(
        self, points: Sequence[str], horizon_seconds: int, interval_seconds: int
    ) -> dict[str, list[float | None]]:
        if self.test_id is None:
            raise TransportError("BOPTEST forecast requested before initialize")
        response = request_json(
            "PUT",
            f"{self.endpoint}/forecast/{self.test_id}",
            payload={
                "point_names": list(points),
                "horizon": int(horizon_seconds),
                "interval": int(interval_seconds),
            },
        )
        if not isinstance(response, Mapping) or not isinstance(response.get("payload"), dict):
            raise TransportError("BOPTEST forecast payload is invalid")
        result: dict[str, list[float | None]] = {}
        for point, raw_values in response["payload"].items():
            if not isinstance(raw_values, list):
                raise TransportError(f"BOPTEST forecast point {point} is not a list")
            values: list[float | None] = []
            for index, value in enumerate(raw_values):
                if value is None:
                    values.append(None)
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise TransportError(
                        f"BOPTEST forecast point {point} at index {index} is invalid"
                    )
                else:
                    values.append(float(value))
            result[str(point)] = values
        return result

    def advance(self, controls: Mapping[str, float]) -> dict[str, Any]:
        if self.test_id is None:
            raise TransportError("BOPTEST advance requested before initialize")
        response = request_json("POST", f"{self.endpoint}/advance/{self.test_id}", payload=controls)
        if not isinstance(response, Mapping) or not isinstance(response.get("payload"), dict):
            raise TransportError("BOPTEST advance payload is invalid")
        return dict(response["payload"])

    def stop(self) -> None:
        if self.test_id is None:
            return
        test_id = self.test_id
        request_json("PUT", f"{self.endpoint}/stop/{test_id}", response_json_required=False)
        self._emit("stopped")
        self.test_id = None
