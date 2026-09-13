"""Transactional PPO/MAPPO checkpoints and complete RNG restoration."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import uuid
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from .io import atomic_json, sha256_file, utc_now


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def sha256_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        handle.flush()
        os.fsync(handle.fileno())


class AtomicPPOCheckpointManager:
    """Commit an SB3 model, training state, RNG, and manifest as one epoch."""

    def __init__(
        self,
        run_dir: Path,
        config_hash: str,
        *,
        save_retries: int = 5,
        replace_func: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None] = os.replace,
    ) -> None:
        self.config_hash = config_hash
        self.save_retries = int(save_retries)
        self.replace_func = replace_func
        self.directory = Path(run_dir) / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / "latest.json"
        self.best_path = self.directory / "best.json"

    def _replace(self, source: Path, destination: Path) -> None:
        self.replace_func(source, destination)

    @staticmethod
    def _verify_model(path: Path) -> None:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"empty PPO checkpoint: {path}")
        with zipfile.ZipFile(path, "r") as archive:
            bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"corrupt PPO checkpoint member: {bad_member}")

    @staticmethod
    def _names(epoch: int, steps: int) -> dict[str, str]:
        prefix = f"epoch_{epoch:04d}_steps_{steps:09d}"
        return {
            "model": f"{prefix}.model.zip",
            "state": f"{prefix}.state.json",
            "rng": f"{prefix}.rng.pt",
            "manifest": f"{prefix}.manifest.json",
        }

    def save(self, model: PPO, state: Mapping[str, Any], epoch: int) -> dict[str, Any]:
        steps = int(model.num_timesteps)
        names = self._names(epoch, steps)
        final = {key: self.directory / name for key, name in names.items()}
        last_error: Exception | None = None
        for attempt in range(self.save_retries):
            token = uuid.uuid4().hex
            temporary = {
                "model": self.directory / f".model.{token}.tmp.zip",
                "state": self.directory / f".state.{token}.tmp.json",
                "rng": self.directory / f".rng.{token}.tmp.pt",
                "manifest": self.directory / f".manifest.{token}.tmp.json",
            }
            try:
                model.save(temporary["model"], exclude=["_transaction_hook"])
                self._verify_model(temporary["model"])
                _write_json(temporary["state"], state)
                with temporary["rng"].open("wb") as handle:
                    torch.save(capture_rng_state(), handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                manifest = {
                    "schema": "h3c-drl-ppo-checkpoint-v1",
                    "created_at": utc_now(),
                    "epoch": int(epoch),
                    "num_timesteps": steps,
                    "config_hash": self.config_hash,
                    "files": {
                        key: {"name": names[key], "sha256": sha256_file(temporary[key])}
                        for key in ("model", "state", "rng")
                    },
                }
                _write_json(temporary["manifest"], manifest)
                for key in ("model", "state", "rng", "manifest"):
                    self._replace(temporary[key], final[key])
                self._verify_model(final["model"])
                for key in ("model", "state", "rng"):
                    if sha256_file(final[key]) != manifest["files"][key]["sha256"]:
                        raise RuntimeError(f"checkpoint checksum mismatch: {final[key]}")
                pointer = {
                    "schema": "h3c-drl-ppo-pointer-v1",
                    "manifest": names["manifest"],
                    "manifest_sha256": sha256_file(final["manifest"]),
                    "epoch": int(epoch),
                    "num_timesteps": steps,
                    "config_hash": self.config_hash,
                    "committed_at": utc_now(),
                }
                pointer_tmp = self.directory / f".latest.{token}.tmp"
                _write_json(pointer_tmp, pointer)
                self._replace(pointer_tmp, self.latest_path)
                self._prune()
                return {**pointer, "model_path": str(final["model"])}
            except Exception as exc:
                last_error = exc
                for path in temporary.values():
                    path.unlink(missing_ok=True)
                if attempt + 1 < self.save_retries:
                    time.sleep(0.25 * 2**attempt)
        raise RuntimeError(
            f"PPO checkpoint failed after {self.save_retries} attempts; "
            "the previous latest pointer remains authoritative"
        ) from last_error

    def mark_best(self, pointer: Mapping[str, Any], score: float) -> None:
        manifest_name = str(pointer["manifest"])
        atomic_json(
            self.best_path,
            {
                "schema": "h3c-drl-ppo-best-v1",
                "manifest": manifest_name,
                "manifest_sha256": sha256_file(self.directory / manifest_name),
                "epoch": int(pointer["epoch"]),
                "num_timesteps": int(pointer["num_timesteps"]),
                "score": float(score),
                "config_hash": self.config_hash,
                "committed_at": utc_now(),
            },
        )
        self._prune()

    def _read_pointer(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        pointer = json.loads(path.read_text(encoding="utf-8"))
        if pointer.get("config_hash") != self.config_hash:
            raise RuntimeError("PPO checkpoint configuration hash mismatch")
        manifest_path = self.directory / pointer["manifest"]
        if sha256_file(manifest_path) != pointer["manifest_sha256"]:
            raise RuntimeError("PPO checkpoint manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != self.config_hash:
            raise RuntimeError("PPO checkpoint manifest configuration mismatch")
        for key in ("model", "state", "rng"):
            metadata = manifest["files"][key]
            if sha256_file(self.directory / metadata["name"]) != metadata["sha256"]:
                raise RuntimeError(f"PPO checkpoint file checksum mismatch: {key}")
        return pointer, manifest

    def load_latest_metadata(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        pointer, manifest = self._read_pointer(self.latest_path)
        state_path = self.directory / manifest["files"]["state"]["name"]
        rng_path = self.directory / manifest["files"]["rng"]["name"]
        state = json.loads(state_path.read_text(encoding="utf-8"))
        with rng_path.open("rb") as handle:
            rng = torch.load(handle, map_location="cpu", weights_only=False)
        if int(state.get("committed_epoch", -1)) != int(pointer["epoch"]):
            raise RuntimeError("PPO checkpoint state and pointer epochs differ")
        return {
            "pointer": pointer,
            "manifest": manifest,
            "state": state,
            "rng": rng,
            "model_path": self.directory / manifest["files"]["model"]["name"],
        }

    def load_best_metadata(self) -> dict[str, Any]:
        if not self.best_path.exists():
            raise FileNotFoundError("no deterministic training-window PPO best exists")
        pointer, manifest = self._read_pointer(self.best_path)
        return {
            "pointer": pointer,
            "manifest": manifest,
            "model_path": self.directory / manifest["files"]["model"]["name"],
            "state_path": self.directory / manifest["files"]["state"]["name"],
        }

    def _prune(self) -> None:
        manifests = sorted(self.directory.glob("epoch_*.manifest.json"))
        keep = {path.name for path in manifests[-2:]}
        for pointer_path in (self.latest_path, self.best_path):
            if pointer_path.exists():
                try:
                    keep.add(json.loads(pointer_path.read_text(encoding="utf-8"))["manifest"])
                except Exception:
                    pass
        for manifest_path in manifests:
            if manifest_path.name in keep:
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for metadata in manifest.get("files", {}).values():
                    (self.directory / metadata["name"]).unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
            except Exception:
                pass


class AtomicMAPPOCheckpointManager:
    """Commit a complete MAPPO bundle and checksum manifest atomically."""

    def __init__(
        self,
        run_dir: Path,
        config_hash: str,
        *,
        save_retries: int = 5,
        replace_func: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None] = os.replace,
    ) -> None:
        self.config_hash = config_hash
        self.save_retries = int(save_retries)
        self.replace_func = replace_func
        self.directory = Path(run_dir) / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / "latest.json"
        self.best_path = self.directory / "best.json"

    def _replace(self, source: Path, destination: Path) -> None:
        self.replace_func(source, destination)

    def _write_pointer(self, destination: Path, payload: Mapping[str, Any]) -> None:
        temporary = self.directory / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            _write_json(temporary, payload)
            self._replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _verify_payload(path: Path) -> Mapping[str, Any]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"empty MAPPO checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "actors" not in payload or "critic" not in payload:
            raise RuntimeError(f"incomplete MAPPO checkpoint: {path}")
        return payload

    def save(self, payload: Mapping[str, Any], *, epoch: int, global_step: int) -> dict[str, Any]:
        prefix = f"epoch_{epoch:04d}_steps_{global_step:09d}"
        bundle_name = f"{prefix}.mappo.pt"
        manifest_name = f"{prefix}.manifest.json"
        final_bundle = self.directory / bundle_name
        final_manifest = self.directory / manifest_name
        last_error: Exception | None = None
        for attempt in range(self.save_retries):
            token = uuid.uuid4().hex
            temporary_bundle = self.directory / f".bundle.{token}.tmp.pt"
            temporary_manifest = self.directory / f".manifest.{token}.tmp.json"
            try:
                with temporary_bundle.open("wb") as handle:
                    torch.save(dict(payload), handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                loaded = self._verify_payload(temporary_bundle)
                if int(loaded.get("committed_epoch", -1)) != int(epoch):
                    raise RuntimeError("serialized MAPPO epoch failed integrity check")
                manifest = {
                    "schema": "h3c-drl-mappo-checkpoint-v1",
                    "created_at": utc_now(),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "config_hash": self.config_hash,
                    "bundle": {
                        "name": bundle_name,
                        "sha256": sha256_file(temporary_bundle),
                    },
                }
                _write_json(temporary_manifest, manifest)
                self._replace(temporary_bundle, final_bundle)
                self._replace(temporary_manifest, final_manifest)
                self._verify_payload(final_bundle)
                if sha256_file(final_bundle) != manifest["bundle"]["sha256"]:
                    raise RuntimeError("MAPPO checkpoint checksum mismatch after commit")
                pointer = {
                    "schema": "h3c-drl-mappo-pointer-v1",
                    "manifest": manifest_name,
                    "manifest_sha256": sha256_file(final_manifest),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "config_hash": self.config_hash,
                    "committed_at": utc_now(),
                }
                self._write_pointer(self.latest_path, pointer)
                self._prune()
                return pointer
            except Exception as exc:
                last_error = exc
                temporary_bundle.unlink(missing_ok=True)
                temporary_manifest.unlink(missing_ok=True)
                if attempt + 1 < self.save_retries:
                    time.sleep(0.25 * 2**attempt)
        raise RuntimeError(
            f"MAPPO checkpoint failed after {self.save_retries} attempts; "
            "the previous latest pointer remains authoritative"
        ) from last_error

    def mark_best(self, pointer: Mapping[str, Any], score: float) -> None:
        manifest_path = self.directory / str(pointer["manifest"])
        self._write_pointer(
            self.best_path,
            {
                **dict(pointer),
                "schema": "h3c-drl-mappo-best-v1",
                "manifest_sha256": sha256_file(manifest_path),
                "score": float(score),
                "committed_at": utc_now(),
            },
        )
        self._prune()

    def _read(self, pointer_path: Path) -> dict[str, Any]:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if pointer.get("config_hash") != self.config_hash:
            raise RuntimeError("MAPPO checkpoint configuration hash mismatch")
        manifest_path = self.directory / str(pointer["manifest"])
        if sha256_file(manifest_path) != pointer["manifest_sha256"]:
            raise RuntimeError("MAPPO checkpoint manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != self.config_hash:
            raise RuntimeError("MAPPO checkpoint manifest configuration mismatch")
        bundle_path = self.directory / manifest["bundle"]["name"]
        if sha256_file(bundle_path) != manifest["bundle"]["sha256"]:
            raise RuntimeError("MAPPO checkpoint bundle checksum mismatch")
        payload = dict(self._verify_payload(bundle_path))
        if int(payload.get("committed_epoch", -1)) != int(pointer["epoch"]):
            raise RuntimeError("MAPPO checkpoint payload and pointer epochs differ")
        return {
            "pointer": pointer,
            "manifest": manifest,
            "payload": payload,
            "bundle_path": bundle_path,
        }

    def load_latest(self) -> dict[str, Any] | None:
        return self._read(self.latest_path) if self.latest_path.exists() else None

    def load_best(self) -> dict[str, Any]:
        if not self.best_path.exists():
            raise FileNotFoundError("no deterministic training-window MAPPO best exists")
        return self._read(self.best_path)

    def _prune(self) -> None:
        manifests = sorted(self.directory.glob("epoch_*.manifest.json"))
        keep = {path.name for path in manifests[-2:]}
        for pointer_path in (self.latest_path, self.best_path):
            if pointer_path.exists():
                try:
                    keep.add(json.loads(pointer_path.read_text(encoding="utf-8"))["manifest"])
                except Exception:
                    pass
        for manifest_path in manifests:
            if manifest_path.name in keep:
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                (self.directory / manifest["bundle"]["name"]).unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
            except Exception:
                pass


class TransactionalPPO(PPO):
    """Unmodified SB3 PPO updates with an epoch-level commit hook."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._transaction_hook: Callable[[PPO], bool] | None = None

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "_transaction_hook"]

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        tb_log_name: str = "TransactionalPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> TransactionalPPO:
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())
        assert self.env is not None
        try:
            while self.num_timesteps < total_timesteps:
                continue_training = self.collect_rollouts(
                    self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
                )
                if not continue_training:
                    break
                iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
                if log_interval is not None and iteration % log_interval == 0:
                    self.dump_logs(iteration)
                self.train()
                if self._transaction_hook is not None and not self._transaction_hook(self):
                    break
        finally:
            callback.on_training_end()
        return self


# Concise aliases used by the trainer modules.
AtomicCheckpointManager = AtomicPPOCheckpointManager
AtomicTorchCheckpointManager = AtomicMAPPOCheckpointManager
