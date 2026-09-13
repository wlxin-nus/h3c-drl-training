from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass
class PlateauState:
    min_epoch: int = 100
    patience: int = 3
    min_delta_fraction: float = 0.01
    anchor: float | None = None
    best_score: float = -math.inf
    best_epoch: int | None = None
    misses: int = 0
    stopped: bool = False
    stop_epoch: int | None = None
    evaluations: int = 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> PlateauState:
        if not payload:
            return cls()
        fields = {field.name for field in dataclasses.fields(cls)}
        values = {key: value for key, value in payload.items() if key in fields}
        if values.get("best_score") is None:
            values["best_score"] = -math.inf
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def update(self, epoch: int, score: float) -> dict[str, Any]:
        if not math.isfinite(score):
            raise ValueError("Early-stop score must be finite")
        if epoch <= 0:
            raise ValueError("Epoch must be positive")
        self.evaluations += 1
        actual_best = score > self.best_score
        if actual_best:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
        patience_active = epoch > self.min_epoch
        if self.anchor is None:
            meaningful = True
            threshold = 0.0
            self.anchor = float(score)
            self.misses = 0
        else:
            threshold = self.min_delta_fraction * max(abs(self.anchor), 1.0)
            meaningful = score > self.anchor + threshold
            if meaningful:
                self.anchor = float(score)
                self.misses = 0
            elif patience_active:
                self.misses += 1
            else:
                # Warm-up evaluations may improve the best/anchor, but they do
                # not consume post-warm-up patience.
                self.misses = 0
        if patience_active and self.misses >= self.patience:
            self.stopped = True
            self.stop_epoch = int(epoch)
        return {
            "epoch": int(epoch),
            "score": float(score),
            "actual_best": actual_best,
            "meaningful_improvement": meaningful,
            "patience_active": patience_active,
            "threshold": float(threshold),
            "anchor": self.anchor,
            "misses": self.misses,
            "should_stop": self.stopped,
        }
