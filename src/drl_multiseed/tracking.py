from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import append_jsonl, atomic_json, utc_now


class SafeWandb:
    def __init__(
        self,
        run_dir: Path,
        *,
        project: str,
        group: str,
        name: str,
        run_id: str,
        mode: str,
        config: Mapping[str, Any],
    ):
        self.run_dir = Path(run_dir)
        self.project = project
        self.group = group
        self.name = name
        self.run_id = run_id
        self.mode = mode
        self.config = dict(config)
        self.run: Any = None
        self.error: str | None = None

    def start(self) -> None:
        root = self.run_dir / "wandb"
        for variable, child in (
            ("WANDB_DATA_DIR", "data"),
            ("WANDB_CACHE_DIR", "cache"),
            ("WANDB_CONFIG_DIR", "config"),
            ("WANDB_ARTIFACT_DIR", "artifacts"),
        ):
            path = root / child
            path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault(variable, str(path))
        try:
            import wandb

            modes = [self.mode, "offline"] if self.mode == "online" else [self.mode]
            for candidate in dict.fromkeys(modes):
                try:
                    self.run = wandb.init(
                        project=self.project,
                        group=self.group,
                        name=self.name,
                        id=self.run_id,
                        resume="allow",
                        mode=candidate,
                        dir=str(root),
                        config=self.config,
                        reinit=True,
                        settings=wandb.Settings(init_timeout=15, login_timeout=10),
                    )
                    self.mode = candidate
                    return
                except Exception as exc:
                    self.error = repr(exc)
                    try:
                        wandb.finish()
                    except Exception:
                        pass
        except Exception as exc:
            self.error = repr(exc)
        atomic_json(
            root / "wandb_disabled.json", {"error": self.error, "requested_mode": self.mode}
        )

    def log(self, values: Mapping[str, Any], *, step: int) -> None:
        clean: dict[str, int | float | bool] = {}
        for key, value in values.items():
            if isinstance(value, bool):
                clean[key] = value
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                clean[key] = value
        row = {"timestamp": utc_now(), "step": int(step), **clean}
        if self.run is None:
            append_jsonl(self.run_dir / "wandb" / "pending_history.jsonl", row)
            return
        try:
            self.run.log(clean, step=int(step))
        except Exception as exc:
            self.error = repr(exc)
            append_jsonl(
                self.run_dir / "wandb" / "pending_history.jsonl",
                {
                    **row,
                    "wandb_error": self.error,
                },
            )
            self.run = None

    def finish(self, summary: Mapping[str, Any] | None = None) -> None:
        if self.run is None:
            if summary:
                atomic_json(
                    self.run_dir / "wandb" / "pending_summary.json",
                    {
                        "timestamp": utc_now(),
                        **dict(summary),
                        "wandb_error": self.error,
                    },
                )
            return
        try:
            for key, value in (summary or {}).items():
                self.run.summary[key] = value
            self.run.finish()
        except Exception:
            pass
