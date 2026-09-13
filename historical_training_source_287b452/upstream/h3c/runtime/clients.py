"""HTTP clients with bounded model-only transient connection recovery."""

from __future__ import annotations

import asyncio
import hashlib
import http.client
import json
import math
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

UTC = timezone.utc  # Python 3.10 compatibility; semantically identical to datetime.UTC.
from email.utils import parsedate_to_datetime
from typing import Any, Literal, overload

from h3c.agents.prompts import Role
from h3c.agents.roles import ModelCallContext


class TransportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        error_type: str = "transport_contract_error",
        provider_response_received: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.error_type = error_type
        self.provider_response_received = provider_response_received
        self.retry_after_seconds = retry_after_seconds


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def model_request_identity(request_contract: Mapping[str, Any]) -> str:
    """Return a secret-free identity for the exact serialized model request body."""
    return hashlib.sha256(model_request_body(request_contract).encode("utf-8")).hexdigest()


def provider_neutral_request_identity(request_contract: Mapping[str, Any]) -> str:
    """Identify the complete request contract except for the provider model slug."""
    neutral = {key: value for key, value in request_contract.items() if key != "model"}
    return hashlib.sha256(model_request_body(neutral).encode("utf-8")).hexdigest()


def model_request_body(request_contract: Mapping[str, Any]) -> str:
    """Serialize the exact secret-free model request body sent on the wire."""
    return json.dumps(request_contract, allow_nan=False)


def model_request_contract(
    *,
    model: str,
    system: str,
    user: str,
    thinking_mode: str,
    response_format: str = "json_object",
    response_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the one exact OpenAI-compatible request body used on every attempt."""
    if thinking_mode not in {"low", "disabled"}:
        raise ValueError("thinking mode must be low or disabled")
    if response_format not in {"json_object", "json_schema"}:
        raise ValueError("response format must be json_object or json_schema")
    if response_format == "json_schema" and response_schema is None:
        raise ValueError("JSON-schema response format requires a schema")
    if response_format == "json_object" and response_schema is not None:
        raise ValueError("JSON-object response format cannot carry a schema")
    response_contract: dict[str, Any] = {"type": "json_object"}
    if response_schema is not None:
        response_contract = {
            "type": "json_schema",
            "json_schema": {
                "name": "h3c_agent_response",
                "strict": True,
                "schema": dict(response_schema),
            },
        }
    contract: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": response_contract,
        "thinking": {"type": "enabled" if thinking_mode == "low" else "disabled"},
    }
    if thinking_mode == "low":
        contract["reasoning_effort"] = "low"
    else:
        contract.update({"temperature": 0.0, "top_p": 1.0})
    return contract


def _request_body(payload: Mapping[str, Any]) -> bytes:
    return model_request_body(payload).encode("utf-8")


def model_logical_call_identity(
    context: Mapping[str, Any],
    role: str,
    thinking_mode: str,
    request_identity: str,
) -> str:
    """Bind one logical call to its runtime surface and immutable request body."""
    return hashlib.sha256(
        _canonical(
            {
                "context": dict(context),
                "role": role,
                "thinking_mode": thinking_mode,
                "request_identity": request_identity,
            }
        ).encode("utf-8")
    ).hexdigest()


def _connection_failure(error: BaseException) -> tuple[bool, str]:
    cause: BaseException | object = error
    if isinstance(error, urllib.error.URLError):
        cause = error.reason
    retryable = isinstance(
        cause,
        (
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            TimeoutError,
            socket.gaierror,
            http.client.IncompleteRead,
            ssl.SSLEOFError,
            ssl.SSLZeroReturnError,
        ),
    )
    return retryable, type(cause).__name__


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    try:
        seconds = float(stripped)
    except ValueError:
        try:
            target = parsedate_to_datetime(stripped)
        except (TypeError, ValueError, OverflowError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        seconds = (target - datetime.now(UTC)).total_seconds()
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


@overload
def _request_json(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 600.0,
    response_json_required: bool = True,
    retryable_status_codes: frozenset[int] | None = None,
    _boptest_status_response: Literal[False] = False,
) -> dict[str, Any]: ...


@overload
def _request_json(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 600.0,
    response_json_required: bool = True,
    retryable_status_codes: frozenset[int] | None = None,
    _boptest_status_response: Literal[True],
) -> dict[str, Any] | str: ...


def _request_json(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 600.0,
    response_json_required: bool = True,
    retryable_status_codes: frozenset[int] | None = None,
    _boptest_status_response: bool = False,
) -> dict[str, Any] | str:
    retryable_codes = retryable_status_codes or frozenset({429, 503})
    body = None if payload is None else _request_body(payload)
    request_headers = dict(headers or {})
    if payload is not None:
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            try:
                raw = response.read().decode("utf-8")
            except UnicodeDecodeError as error:
                raise TransportError(
                    f"endpoint returned non-UTF-8 text for {url}",
                    error_type="response_text_invalid",
                    provider_response_received=True,
                ) from error
            if response.status != 200:
                retryable = response.status in retryable_codes
                response_headers = getattr(response, "headers", None)
                retry_after = (
                    _retry_after_seconds(response_headers.get("Retry-After"))
                    if retryable and response_headers is not None
                    else None
                )
                raise TransportError(
                    f"HTTP {response.status} from {url}",
                    retryable=retryable,
                    error_type=f"http_{response.status}",
                    provider_response_received=True,
                    retry_after_seconds=retry_after,
                )
    except urllib.error.HTTPError as error:
        retryable = error.code in retryable_codes
        response_headers = getattr(error, "headers", None)
        raise TransportError(
            f"HTTP {error.code} from {url}",
            retryable=retryable,
            error_type=f"http_{error.code}",
            provider_response_received=True,
            retry_after_seconds=(
                _retry_after_seconds(response_headers.get("Retry-After"))
                if retryable and response_headers is not None
                else None
            ),
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError, http.client.IncompleteRead) as error:
        retryable, error_type = _connection_failure(error)
        raise TransportError(
            f"request failed for {url}: {error_type}",
            retryable=retryable,
            error_type=error_type,
        ) from error
    try:
        value = json.loads(raw) if raw else {}
    except json.JSONDecodeError as error:
        if not response_json_required:
            return {}
        raise TransportError(
            f"endpoint returned invalid JSON for {url}",
            error_type="response_json_invalid",
            provider_response_received=True,
        ) from error
    if _boptest_status_response and isinstance(value, str):
        if value in {"Running", "Queued"}:
            return value
        raise TransportError(
            f"BOPTEST status response is invalid for {url}",
            error_type="boptest_status_invalid",
            provider_response_received=True,
        )
    if not isinstance(value, dict):
        raise TransportError(
            f"endpoint returned a non-object for {url}",
            error_type="response_json_non_object",
            provider_response_received=True,
        )
    return value


LifecycleSink = Callable[[Mapping[str, Any]], None]


class BoptestHttpClient:
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

    def _emit_lifecycle(self, *, event: str, status: str | None = None) -> None:
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
        """Select one worker and freeze its test id for this client."""
        if self.test_id is not None:
            raise TransportError("BOPTEST client already owns a live test id")
        selected = _request_json("POST", f"{self.endpoint}/testcases/{testcase}/select")
        test_id = selected.get("testid")
        if not isinstance(test_id, str) or not test_id:
            raise TransportError("BOPTEST select did not return a test id")
        self.test_id = test_id
        self.testcase = testcase
        self._emit_lifecycle(event="selected")
        self._wait_until_running()
        self._configure_selected()
        return test_id

    def status(self) -> str:
        if self.test_id is None:
            raise TransportError("BOPTEST status requested before select")
        response = _request_json(
            "GET",
            f"{self.endpoint}/status/{self.test_id}",
            _boptest_status_response=True,
        )
        status = response if isinstance(response, str) else response.get("payload")
        if not isinstance(status, str) or status not in {"Running", "Queued"}:
            raise TransportError(
                "BOPTEST status payload is invalid",
                error_type="boptest_status_invalid",
            )
        return str(status)

    def _wait_until_running(self) -> None:
        previous: str | None = None
        while True:
            status = self.status()
            if status != previous:
                self._emit_lifecycle(event="status_changed", status=status)
                previous = status
            if status == "Running":
                return
            time.sleep(self.queue_poll_seconds)

    def initialize(
        self, testcase: str, start_time_seconds: int, warmup_period_seconds: int
    ) -> dict[str, Any]:
        self.select_testcase(testcase)
        return self.initialize_selected(start_time_seconds, warmup_period_seconds)

    def _configure_selected(self) -> None:
        if self.test_id is None:
            raise TransportError("BOPTEST configure requested without a selected test id")
        test_id = self.test_id
        _request_json(
            "PUT",
            f"{self.endpoint}/scenario/{test_id}",
            payload={"electricity_price": "dynamic"},
        )
        _request_json("PUT", f"{self.endpoint}/step/{test_id}", payload={"step": 900})
        self._emit_lifecycle(event="configured", status="Running")

    def initialize_selected(
        self, start_time_seconds: int, warmup_period_seconds: int
    ) -> dict[str, Any]:
        """FMU-reset one selected test and repeat its complete BOPTEST warm-up."""

        if self.test_id is None:
            raise TransportError("BOPTEST initialize requested without a selected test id")
        self._wait_until_running()
        initialized = _request_json(
            "PUT",
            f"{self.endpoint}/initialize/{self.test_id}",
            payload={
                "start_time": start_time_seconds,
                "warmup_period": warmup_period_seconds,
            },
        )
        state = initialized.get("payload")
        if not isinstance(state, dict):
            raise TransportError("BOPTEST initialize payload is invalid")
        self._emit_lifecycle(event="initialized", status="Running")
        return state

    def forecast(
        self, points: Sequence[str], horizon_seconds: int, interval_seconds: int
    ) -> dict[str, list[float | None]]:
        if self.test_id is None:
            raise TransportError("BOPTEST forecast requested before initialize")
        response = _request_json(
            "PUT",
            f"{self.endpoint}/forecast/{self.test_id}",
            payload={
                "point_names": list(points),
                "horizon": horizon_seconds,
                "interval": interval_seconds,
            },
        )
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise TransportError("BOPTEST forecast payload is invalid")
        forecast: dict[str, list[float | None]] = {}
        for raw_point, raw_values in payload.items():
            point = str(raw_point)
            if not isinstance(raw_values, list):
                raise TransportError(f"BOPTEST forecast point {point} is not a list")
            values: list[float | None] = []
            for index, value in enumerate(raw_values):
                if value is None:
                    values.append(None)
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise TransportError(
                        f"BOPTEST forecast point {point} at index {index} is not a finite number"
                    )
                values.append(float(value))
            forecast[point] = values
        return forecast

    def advance(self, controls: Mapping[str, float]) -> dict[str, Any]:
        if self.test_id is None:
            raise TransportError("BOPTEST advance requested before initialize")
        response = _request_json(
            "POST", f"{self.endpoint}/advance/{self.test_id}", payload=controls
        )
        state = response.get("payload")
        if not isinstance(state, dict):
            raise TransportError("BOPTEST advance payload is invalid")
        return state

    def get_kpis(self) -> dict[str, Any]:
        """Read native BOPTEST KPIs before stopping the active test."""
        if self.test_id is None:
            raise TransportError("BOPTEST KPI requested before initialize")
        response = _request_json("GET", f"{self.endpoint}/kpi/{self.test_id}")
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise TransportError("BOPTEST KPI payload is invalid")
        return dict(payload)

    def stop(self) -> None:
        if self.test_id is None:
            raise TransportError("BOPTEST stop requested without a live test id")
        _request_json(
            "PUT",
            f"{self.endpoint}/stop/{self.test_id}",
            response_json_required=False,
        )
        self._emit_lifecycle(event="stopped")
        self.test_id = None


CallSink = Callable[[str, Mapping[str, Any]], None]


@dataclass
class OpenAICompatibleModelClient:
    endpoint: str
    api_key: str
    model: str
    sink: CallSink
    retry_count_limit: int
    retry_backoff_seconds: tuple[float, ...]
    retry_count: int = 0
    provider_id: str = "deepseek-official"
    extra_headers: Mapping[str, str] | None = None
    retryable_status_codes: tuple[int, ...] = (429, 503)
    response_format: str = "json_object"

    async def complete(
        self,
        *,
        context: ModelCallContext,
        role: Role,
        system: str,
        user: str,
        thinking_mode: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> str:
        if self.retry_count_limit < 0 or len(self.retry_backoff_seconds) != self.retry_count_limit:
            raise ValueError("model retry configuration is invalid")
        request_contract = model_request_contract(
            model=self.model,
            system=system,
            user=user,
            thinking_mode=thinking_mode,
            response_format=self.response_format,
            response_schema=(response_schema if self.response_format == "json_schema" else None),
        )
        context_fields = context.as_mapping()
        request_identity = model_request_identity(request_contract)
        neutral_request_identity = provider_neutral_request_identity(request_contract)
        request_body = model_request_body(request_contract)
        logical_call_identity = model_logical_call_identity(
            context_fields,
            role,
            thinking_mode,
            request_identity,
        )
        logical_started = time.perf_counter()
        response: dict[str, Any] | None = None
        choice: Mapping[str, Any] | None = None
        content: str | None = None
        maximum_attempts = self.retry_count_limit + 1
        for attempt_number in range(1, maximum_attempts + 1):
            attempt_started = time.perf_counter()
            try:
                candidate = await asyncio.to_thread(
                    _request_json,
                    "POST",
                    f"{self.endpoint}/chat/completions",
                    payload=request_contract,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        **dict(self.extra_headers or {}),
                    },
                    retryable_status_codes=frozenset(self.retryable_status_codes),
                )
                raw_choice = candidate["choices"][0]
                message = raw_choice["message"]
                raw_content = message["content"] or ""
                if not isinstance(raw_choice, Mapping) or not isinstance(raw_content, str):
                    raise TransportError(
                        "model response contract is invalid",
                        error_type="model_response_contract_invalid",
                        provider_response_received=True,
                    )
                response = candidate
                choice = raw_choice
                content = raw_content
            except (KeyError, IndexError, TypeError) as error:
                failure = TransportError(
                    "model response contract is invalid",
                    error_type="model_response_contract_invalid",
                    provider_response_received=True,
                )
                self._record_attempt(
                    context=context_fields,
                    role=role,
                    thinking_mode=thinking_mode,
                    request_identity=request_identity,
                    request_body=request_body,
                    logical_call_identity=logical_call_identity,
                    attempt_number=attempt_number,
                    maximum_attempts=maximum_attempts,
                    outcome="request_failed",
                    retryable=False,
                    will_retry=False,
                    error_type=failure.error_type,
                    provider_charge_status="response_received_usage_unavailable",
                    elapsed_seconds=time.perf_counter() - attempt_started,
                    provider_retry_after_seconds=None,
                    retry_delay_seconds=None,
                )
                raise failure from error
            except TransportError as error:
                will_retry = error.retryable and attempt_number < maximum_attempts
                retry_delay = (
                    error.retry_after_seconds
                    if will_retry and error.retry_after_seconds is not None
                    else (self.retry_backoff_seconds[attempt_number - 1] if will_retry else None)
                )
                self._record_attempt(
                    context=context_fields,
                    role=role,
                    thinking_mode=thinking_mode,
                    request_identity=request_identity,
                    request_body=request_body,
                    logical_call_identity=logical_call_identity,
                    attempt_number=attempt_number,
                    maximum_attempts=maximum_attempts,
                    outcome="request_failed",
                    retryable=error.retryable,
                    will_retry=will_retry,
                    error_type=error.error_type,
                    provider_charge_status=(
                        "response_received_usage_unavailable"
                        if error.provider_response_received
                        else "unknown_after_request_failure"
                    ),
                    elapsed_seconds=time.perf_counter() - attempt_started,
                    provider_retry_after_seconds=error.retry_after_seconds,
                    retry_delay_seconds=retry_delay,
                )
                if not will_retry:
                    raise
                self.retry_count += 1
                assert retry_delay is not None
                await asyncio.sleep(retry_delay)
                continue
            self._record_attempt(
                context=context_fields,
                role=role,
                thinking_mode=thinking_mode,
                request_identity=request_identity,
                request_body=request_body,
                logical_call_identity=logical_call_identity,
                attempt_number=attempt_number,
                maximum_attempts=maximum_attempts,
                outcome="response_received",
                retryable=False,
                will_retry=False,
                error_type=None,
                provider_charge_status="confirmed_response_usage_recorded",
                elapsed_seconds=time.perf_counter() - attempt_started,
                provider_retry_after_seconds=None,
                retry_delay_seconds=None,
            )
            break
        if response is None or choice is None or content is None:
            raise AssertionError("model retry loop exited without a response or error")
        elapsed = time.perf_counter() - logical_started
        usage = normalized_usage(response.get("usage"))
        common = {
            **context_fields,
            "role": role,
            "thinking_mode": thinking_mode,
            "logical_call_identity": logical_call_identity,
            "request_identity": request_identity,
            "provider_neutral_request_identity": neutral_request_identity,
            "attempt_count": attempt_number,
            "transport_retry_count": attempt_number - 1,
            "request_model": self.model,
            "response_model": response.get("model"),
            "model_provider": self.provider_id,
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "provider_usage": response.get("usage"),
            "elapsed_seconds": elapsed,
        }
        self.sink("agent_calls.jsonl", {**common, "status": "received"})
        self.sink(
            "raw_model_io.jsonl",
            {
                **common,
                "system": system,
                "user": user,
                "output": content,
                "request_parameters": {
                    key: value for key, value in request_contract.items() if key != "messages"
                },
            },
        )
        return content

    def _record_attempt(
        self,
        *,
        context: Mapping[str, Any],
        role: Role,
        thinking_mode: str,
        request_identity: str,
        request_body: str,
        logical_call_identity: str,
        attempt_number: int,
        maximum_attempts: int,
        outcome: str,
        retryable: bool,
        will_retry: bool,
        error_type: str | None,
        provider_charge_status: str,
        elapsed_seconds: float,
        provider_retry_after_seconds: float | None,
        retry_delay_seconds: float | None,
    ) -> None:
        self.sink(
            "model_request_attempts.jsonl",
            {
                **dict(context),
                "role": role,
                "thinking_mode": thinking_mode,
                "request_model": self.model,
                "model_provider": self.provider_id,
                "logical_call_identity": logical_call_identity,
                "request_identity": request_identity,
                "provider_neutral_request_identity": provider_neutral_request_identity(
                    json.loads(request_body)
                ),
                "request_body": request_body,
                "attempt_number": attempt_number,
                "maximum_attempts": maximum_attempts,
                "outcome": outcome,
                "retryable": retryable,
                "will_retry": will_retry,
                "error_type": error_type,
                "provider_charge_status": provider_charge_status,
                "elapsed_seconds": elapsed_seconds,
                "provider_retry_after_seconds": provider_retry_after_seconds,
                "retry_delay_seconds": retry_delay_seconds,
            },
        )


def _usage_integer(value: Any) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        or int(value) != value
    ):
        return None
    return int(value)


def normalized_usage(raw: Any) -> dict[str, Any]:
    """Normalize provider usage to one explicit, internally consistent contract."""
    if not isinstance(raw, Mapping):
        return {"available": False, "reason": "provider_usage_not_an_object"}
    prompt_details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
    completion_details = (
        raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}
    )
    if not isinstance(prompt_details, Mapping):
        prompt_details = {}
    if not isinstance(completion_details, Mapping):
        completion_details = {}

    def first(*names: str) -> int | None:
        for name in names:
            if name in raw:
                return _usage_integer(raw[name])
        return None

    prompt = first("prompt_tokens", "input_tokens")
    completion = first("completion_tokens", "output_tokens")
    total = first("total_tokens")
    cache_hit = first(
        "prompt_cache_hit_tokens",
        "cache_hit_tokens",
        "cached_tokens",
        "cache_read_input_tokens",
    )
    if cache_hit is None:
        cache_hit = _usage_integer(prompt_details.get("cached_tokens"))
    cache_miss = first("prompt_cache_miss_tokens", "cache_miss_tokens", "cache_miss_input_tokens")
    if cache_miss is None and prompt is not None and cache_hit is not None:
        cache_miss = prompt - cache_hit
    reasoning = _usage_integer(completion_details.get("reasoning_tokens"))
    if reasoning is None:
        reasoning = first("reasoning_tokens")
    reasoning = 0 if reasoning is None else reasoning
    fields = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
        "reasoning_tokens": reasoning,
    }
    missing = sorted(name for name, value in fields.items() if value is None)
    if missing:
        return {"available": False, "reason": "usage_fields_unavailable", "missing": missing}
    resolved = {name: int(value) for name, value in fields.items() if value is not None}
    if resolved["cache_hit_tokens"] + resolved["cache_miss_tokens"] != resolved["prompt_tokens"]:
        return {"available": False, "reason": "cache_accounting_mismatch"}
    if resolved["prompt_tokens"] + resolved["completion_tokens"] != resolved["total_tokens"]:
        return {"available": False, "reason": "total_accounting_mismatch"}
    if resolved["reasoning_tokens"] > resolved["completion_tokens"]:
        return {"available": False, "reason": "reasoning_accounting_mismatch"}
    return {"available": True, **resolved}
