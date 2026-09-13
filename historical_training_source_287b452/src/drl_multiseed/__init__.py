"""H3C DRL multi-seed training companion."""

import os
from pathlib import Path

# pythermalcomfort imports cached Numba ufuncs. Keeping that cache inside the
# repository avoids fragile/full Windows user-temp directories and makes all
# worker processes use the same explicit location.
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[2] / "runtime" / "numba_cache")
)

__version__ = "0.1.0"
