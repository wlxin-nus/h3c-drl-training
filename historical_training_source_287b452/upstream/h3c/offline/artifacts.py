"""Artifact ownership for recoverable offline onboarding workspaces."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

from h3c.offline.contracts import OnboardingSpec, file_identity, object_identity
from h3c.runtime.source_identity import committed_source_identity


class OfflineArtifactError(RuntimeError):
    """Raised when workspace evidence cannot be created or recovered safely."""


def source_commit(repository_root: Path) -> str:
    try:
        return committed_source_identity(repository_root)
    except RuntimeError as error:
        raise OfflineArtifactError(
            "offline execution requires a clean committed repository root"
        ) from error


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as file:
            file.write(_json_bytes(value))
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError as error:
        raise OfflineArtifactError(f"refusing to overwrite artifact: {path}") from error


def write_atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(_json_bytes(value))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    with path.open("ab") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OfflineArtifactError(f"cannot read workspace JSON: {path}") from error
    if not isinstance(value, dict):
        raise OfflineArtifactError(f"workspace JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OfflineArtifactError(f"cannot read workspace JSONL: {path}") from error
    if any(not isinstance(value, dict) for value in values):
        raise OfflineArtifactError(f"workspace JSONL rows must be objects: {path}")
    return values


def secret_occurrences(directory: Path, secret: str) -> int:
    if not secret:
        raise OfflineArtifactError("secret scan requires the actual non-empty API key")
    needle = secret.encode("utf-8")
    count = 0
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            count += path.read_bytes().count(needle)
        except OSError as error:
            raise OfflineArtifactError(f"secret scan could not read: {path}") from error
    return count


def endpoint_identity(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/").lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def source_file_records(
    spec: OnboardingSpec, *, include_onboarding_spec: bool = True
) -> list[dict[str, str]]:
    """Rebuild the exact, complete source-file inventory for one onboarding spec."""
    source_files = [
        ("building_document", spec.building_document),
        ("case_profile_template", spec.case_profile_template),
    ]
    if include_onboarding_spec:
        source_files.insert(0, ("onboarding_spec", spec.source_path))
    if spec.point_inventory is not None:
        source_files.append(("point_inventory", spec.point_inventory))
    source_files.extend(
        (f"evidence:{source.identifier}", source.path) for source in spec.evidence_sources
    )
    root = spec.repository_root.resolve()
    return [
        {
            "role": role,
            "path": file.resolve().relative_to(root).as_posix(),
            "sha256": file_identity(file),
        }
        for role, file in source_files
    ]


class OfflineWorkspace:
    """Single writer for one onboarding workspace."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    @property
    def checkpoints(self) -> Path:
        return self.path / "checkpoints"

    @property
    def state_path(self) -> Path:
        return self.checkpoints / "application_state.json"

    @classmethod
    def create(
        cls,
        spec: OnboardingSpec,
        *,
        framework_versions: dict[str, str],
        endpoint: str,
        model: str,
        secret_identity: str,
    ) -> OfflineWorkspace:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        workflow_id = f"{timestamp}-{spec.identity[:12]}"
        path = spec.repository_root / "outputs" / "offline" / spec.case_id / workflow_id
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise OfflineArtifactError(f"offline workspace already exists: {path}") from error
        workspace = cls(path)
        workspace.checkpoints.mkdir()
        commit = source_commit(spec.repository_root)
        public_spec = spec.public()
        write_new_json(path / "resolved_spec.json", public_spec)
        manifest = {
            "artifact_schema": "h3c_offline_source_manifest",
            "schema_version": 1,
            "workflow_id": workflow_id,
            "workflow_identity": object_identity(
                {
                    "workflow_id": workflow_id,
                    "spec_identity": spec.identity,
                    "source_commit": commit,
                }
            ),
            "case_id": spec.case_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_commit": commit,
            "spec_identity": spec.identity,
            "source_files": source_file_records(spec),
            "provider": {
                "kind": "openai_compatible",
                "endpoint_env": spec.provider.endpoint_env,
                "endpoint_identity": endpoint_identity(endpoint),
                "api_key_env": spec.provider.api_key_env,
                "secret_identity": secret_identity,
                "model_env": spec.provider.model_env,
                "model": model,
                "reasoning_effort": "low",
                "temperature_absent": True,
                "top_p_absent": True,
                "automatic_retries": 0,
            },
            "framework_versions": framework_versions,
            "model_call_limits": {
                "mapping": spec.mapping_model_calls,
                "causal_discovery": spec.causal_model_calls,
            },
            "secret_scan_status": "pending",
            "secret_exposure_count": None,
        }
        write_new_json(path / "source_manifest.json", manifest)
        write_atomic_json(
            workspace.state_path,
            {
                "state_schema": "h3c_offline_workflow_state",
                "schema_version": 1,
                "workflow_identity": manifest["workflow_identity"],
                "stage": "created",
                "mapping_round": 0,
                "causal_round": 0,
                "aborted": False,
                "abort_reason": None,
                "latest_framework_checkpoint": None,
            },
        )
        return workspace

    @classmethod
    def open(cls, path: Path) -> OfflineWorkspace:
        resolved = path.resolve()
        if not resolved.is_dir():
            raise OfflineArtifactError(f"offline workspace does not exist: {resolved}")
        workspace = cls(resolved)
        for required in ("resolved_spec.json", "source_manifest.json", "checkpoints"):
            if not (resolved / required).exists():
                raise OfflineArtifactError(f"offline workspace is incomplete: missing {required}")
        return workspace

    def manifest(self) -> dict[str, Any]:
        return read_json(self.path / "source_manifest.json")

    def state(self) -> dict[str, Any]:
        return read_json(self.state_path)

    def update_state(self, **changes: Any) -> dict[str, Any]:
        state = self.state()
        state.update(changes)
        write_atomic_json(self.state_path, state)
        return state

    def append(self, name: str, value: dict[str, Any]) -> None:
        append_jsonl(self.path / name, value)

    def publish(self, name: str, value: Any) -> None:
        write_new_json(self.path / name, value)

    def replace_manifest(self, value: dict[str, Any]) -> None:
        write_atomic_json(self.path / "source_manifest.json", value)
