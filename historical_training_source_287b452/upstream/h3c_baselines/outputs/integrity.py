"""Small evidence-integrity helpers shared by baseline run types."""

from __future__ import annotations

import os
from pathlib import Path

from h3c.experiments.settings import load_runtime_contract


def secret_occurrences(output_dir: Path) -> int:
    """Count configured provider secrets only in the newly generated output tree."""
    providers = load_runtime_contract()["model"]["providers"]
    secrets = {
        os.environ.get(str(provider["api_key_environment_variable"]), "")
        for provider in providers.values()
    }
    secrets.discard("")
    if not secrets:
        return 0
    count = 0
    for path in output_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            count += sum(text.count(secret) for secret in secrets)
    return count
