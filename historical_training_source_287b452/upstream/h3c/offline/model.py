"""Microsoft Agent Framework model boundary for offline onboarding only."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Literal, Protocol, TypedDict
from urllib.parse import urlparse

from h3c.offline.artifacts import OfflineWorkspace
from h3c.offline.contracts import OfflineContractError, OnboardingSpec, object_identity


@dataclass(frozen=True)
class ModelGeneration:
    text: str
    parsed: dict[str, Any]


class OfflineAgentPair(Protocol):
    async def generate_mapping(
        self, system_prompt: str, user_prompt: str, sequence: int, workspace: OfflineWorkspace
    ) -> ModelGeneration: ...

    async def generate_causal(
        self, system_prompt: str, user_prompt: str, sequence: int, workspace: OfflineWorkspace
    ) -> ModelGeneration: ...

    async def close(self) -> None: ...


class _ReasoningChatOptions(TypedDict, total=False):
    model: str
    reasoning_effort: Literal["low"]


def framework_versions() -> dict[str, str]:
    try:
        return {
            "agent-framework-core": version("agent-framework-core"),
            "agent-framework-openai": version("agent-framework-openai"),
            "openai": version("openai"),
        }
    except Exception as error:
        raise OfflineContractError(
            "offline dependencies are missing; run `uv sync --extra offline`"
        ) from error


def _provider_values(spec: OnboardingSpec) -> tuple[str, str, str]:
    values = tuple(
        os.environ.get(name, "").strip()
        for name in (
            spec.provider.endpoint_env,
            spec.provider.api_key_env,
            spec.provider.model_env,
        )
    )
    if any(not value for value in values):
        raise OfflineContractError(
            "provider endpoint, API key, and model environment variables must be set"
        )
    endpoint, api_key, model = values
    parsed = urlparse(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or endpoint.rstrip("/").endswith("/chat/completions")
    ):
        raise OfflineContractError(
            "provider endpoint must be a credential-free OpenAI-compatible API base URL"
        )
    return endpoint.rstrip("/"), api_key, model


def secret_identity(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def validate_wire_request(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise OfflineContractError("serialized provider request must be a JSON object")
    if body.get("reasoning_effort") != "low":
        raise OfflineContractError("serialized provider request must use thinking low")
    if "temperature" in body or "top_p" in body:
        raise OfflineContractError("serialized provider request must omit sampling fields")
    return body


def _is_network_interruption(error: BaseException) -> bool:
    """Classify only provider transport interruptions as explicitly resumable."""

    transport_names = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "WriteError",
        "WriteTimeout",
    }
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ConnectionError, TimeoutError, OSError)) or (
            type(current).__name__ in transport_names
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class MicrosoftOfflineAgents:
    """Two Agent Framework agents sharing one audited, no-retry chat client."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        http_transport: Any | None = None,
    ) -> None:
        try:
            import httpx
            from agent_framework.openai import OpenAIChatCompletionClient
            from openai import AsyncOpenAI
        except ImportError as error:
            raise OfflineContractError(
                "offline dependencies are missing; run `uv sync --extra offline`"
            ) from error
        self.endpoint = endpoint
        self.model = model
        self._framework_versions = framework_versions()
        self._wire_bodies: list[dict[str, Any]] = []

        async def capture(request: Any) -> None:
            body = json.loads((await request.aread()).decode("utf-8"))
            self._wire_bodies.append(validate_wire_request(body))

        self._http_client = httpx.AsyncClient(
            transport=http_transport,
            event_hooks={"request": [capture]},
        )
        self._openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
            max_retries=0,
            http_client=self._http_client,
        )
        chat_client = OpenAIChatCompletionClient(async_client=self._openai_client)
        options: _ReasoningChatOptions = {"model": model, "reasoning_effort": "low"}
        self._mapping_agent = chat_client.as_agent(
            name="SemanticMappingAgent", default_options=options
        )
        self._causal_agent = chat_client.as_agent(
            name="CausalDiscoveryAgent", default_options=options
        )

    @classmethod
    def from_environment(
        cls, spec: OnboardingSpec
    ) -> tuple[MicrosoftOfflineAgents, str, str, str, str]:
        endpoint, api_key, model = _provider_values(spec)
        return (
            cls(endpoint=endpoint, api_key=api_key, model=model),
            endpoint,
            model,
            api_key,
            secret_identity(api_key),
        )

    async def _generate(
        self,
        *,
        role: str,
        agent: Any,
        system_prompt: str,
        user_prompt: str,
        sequence: int,
        workspace: OfflineWorkspace,
    ) -> ModelGeneration:
        before = object_identity(
            {
                "workflow": workspace.manifest()["workflow_identity"],
                "state": workspace.state(),
                "role": role,
                "sequence": sequence,
            }
        )
        wire_index = len(self._wire_bodies)
        started = time.perf_counter()
        provider_call_completed = False
        try:
            response = await agent.run(user_prompt, options={"instructions": system_prompt})
            provider_call_completed = True
            elapsed = time.perf_counter() - started
            if len(self._wire_bodies) != wire_index + 1:
                raise OfflineContractError(
                    "one logical offline call must emit exactly one wire request"
                )
            wire = self._wire_bodies[wire_index]
            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise OfflineContractError("offline Agent returned no text")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as error:
                raise OfflineContractError("offline Agent output is not JSON") from error
            if not isinstance(parsed, dict):
                raise OfflineContractError("offline Agent output root must be an object")
            after = object_identity(
                {
                    "before": before,
                    "request": wire,
                    "response": parsed,
                }
            )
            usage = getattr(response, "usage_details", None)
            usage_serializer = getattr(usage, "to_dict", None)
            usage_value = usage_serializer() if callable(usage_serializer) else usage
            raw_row = {
                "artifact_schema": "h3c_offline_raw_model_io",
                "schema_version": 1,
                "role": role,
                "sequence": sequence,
                "wire_request": wire,
                "wire_request_identity": object_identity(wire),
                "response_text": text,
                "response_identity": object_identity(parsed),
            }
            call_row = {
                "artifact_schema": "h3c_offline_model_call",
                "schema_version": 1,
                "role": role,
                "sequence": sequence,
                "model": self.model,
                "endpoint_identity": hashlib.sha256(
                    self.endpoint.lower().rstrip("/").encode("utf-8")
                ).hexdigest(),
                "framework_versions": self._framework_versions,
                "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
                "reasoning_effort": "low",
                "temperature_absent": "temperature" not in wire,
                "top_p_absent": "top_p" not in wire,
                "wire_request_identity": raw_row["wire_request_identity"],
                "response_identity": raw_row["response_identity"],
                "usage": usage_value,
                "latency_seconds": elapsed,
                "state_checkpoint_before": before,
                "state_checkpoint_after": after,
                "status": "confirmed_response",
            }
            workspace.append("raw_model_io.jsonl", raw_row)
            workspace.append("model_calls.jsonl", call_row)
            return ModelGeneration(text=text, parsed=parsed)
        except Exception as error:
            elapsed = time.perf_counter() - started
            failed_wire = (
                self._wire_bodies[wire_index] if len(self._wire_bodies) > wire_index else None
            )
            resumable = (
                not provider_call_completed
                and failed_wire is not None
                and _is_network_interruption(error)
            )
            workspace.append(
                "model_calls.jsonl",
                {
                    "artifact_schema": "h3c_offline_model_call",
                    "schema_version": 1,
                    "role": role,
                    "sequence": sequence,
                    "model": self.model,
                    "endpoint_identity": hashlib.sha256(
                        self.endpoint.lower().rstrip("/").encode("utf-8")
                    ).hexdigest(),
                    "framework_versions": self._framework_versions,
                    "system_prompt_sha256": hashlib.sha256(
                        system_prompt.encode("utf-8")
                    ).hexdigest(),
                    "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
                    "reasoning_effort": "low",
                    "temperature_absent": not isinstance(failed_wire, dict)
                    or "temperature" not in failed_wire,
                    "top_p_absent": not isinstance(failed_wire, dict) or "top_p" not in failed_wire,
                    "wire_request_identity": (
                        object_identity(failed_wire) if failed_wire is not None else None
                    ),
                    "wire_request": failed_wire,
                    "latency_seconds": elapsed,
                    "state_checkpoint_before": before,
                    "state_checkpoint_after": None,
                    "status": "failed",
                    "failure_category": (
                        "network_interruption" if resumable else "nonresumable_model_failure"
                    ),
                    "resumable": resumable,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise

    async def generate_mapping(
        self, system_prompt: str, user_prompt: str, sequence: int, workspace: OfflineWorkspace
    ) -> ModelGeneration:
        return await self._generate(
            role="semantic_mapping",
            agent=self._mapping_agent,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            sequence=sequence,
            workspace=workspace,
        )

    async def generate_causal(
        self, system_prompt: str, user_prompt: str, sequence: int, workspace: OfflineWorkspace
    ) -> ModelGeneration:
        return await self._generate(
            role="causal_discovery",
            agent=self._causal_agent,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            sequence=sequence,
            workspace=workspace,
        )

    async def close(self) -> None:
        await self._openai_client.close()
