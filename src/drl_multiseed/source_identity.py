"""Stable source identities for checkout and installed-package execution."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .config import source_checkout_root


def code_commit() -> str:
    """Return this package checkout's commit without borrowing an unrelated Git tree."""

    checkout = source_checkout_root()
    if checkout is None:
        return "installed-package-no-vcs"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "uncommitted-source-tree"


def code_fingerprint() -> str:
    """Hash every Python and JSON runtime resource in the installed package."""

    package = Path(__file__).resolve().parent
    runtime_files = sorted(
        path
        for path in package.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json"} and "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    for path in runtime_files:
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        content = path.read_text(encoding="utf-8")
        digest.update(content.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))
    return digest.hexdigest()
