"""Reproducible PPO and MAPPO training for the H3C study."""

import os
import tempfile
from pathlib import Path

# pythermalcomfort imports cached Numba ufuncs. Use a short, writable default
# path so deeply nested Windows checkouts do not exceed the legacy path limit.
# Respect an operator-provided cache.
if "NUMBA_CACHE_DIR" not in os.environ:
    _numba_cache = Path(tempfile.gettempdir()) / "drl_multiseed_numba_cache"
    _numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(_numba_cache)

__version__ = "1.1.0"
