"""Legacy min-max normalization preserved for frozen DRL policies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def symmetric_minmax(
    values: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float32]:
    if values.shape != lower.shape or values.shape != upper.shape:
        raise ValueError("normalization arrays must have identical shapes")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(upper <= lower):
        raise ValueError("normalization bounds are invalid")
    # The frozen policies were trained behind Gym Box spaces with float32 bounds.  Preserve that
    # arithmetic order exactly: casting only the final float64 result is observably different at
    # the policy input and can perturb deterministic actions.
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    lower_f32 = lower.astype(np.float32)
    upper_f32 = upper.astype(np.float32)
    return np.asarray(2.0 * (clean - lower_f32) / (upper_f32 - lower_f32) - 1.0, dtype=np.float32)
