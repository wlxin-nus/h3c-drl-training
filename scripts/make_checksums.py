from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "CHECKSUMS.sha256"
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".venv",
    "__pycache__",
    "venv",
}
EXCLUDED_ROOT_DIRECTORIES = {"build", "dist", "runs", "runtime", "wandb"}
EXCLUDED_NAMES = {".coverage", ".env", "coverage.xml", OUTPUT.name}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and not any(part.endswith(".egg-info") for part in relative.parts)
        and relative.parts[0] not in EXCLUDED_ROOT_DIRECTORIES
        and not path.name.endswith((".pyc", ".pyo"))
    )


def current() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in sorted(item for item in ROOT.rglob("*") if included(item)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        values[path.relative_to(ROOT).as_posix()] = digest
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    values = current()
    if args.verify:
        recorded: dict[str, str] = {}
        for line in OUTPUT.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            recorded[name] = digest
        missing = sorted(set(recorded) - set(values))
        added = sorted(set(values) - set(recorded))
        changed = sorted(
            name for name in set(recorded) & set(values) if recorded[name] != values[name]
        )
        if missing or added or changed:
            raise SystemExit(
                f"Checksum failure: missing={missing}, added={added}, changed={changed}"
            )
        print(f"Verified {len(recorded)} entries from {OUTPUT}")
        return
    lines = [f"{digest}  {name}" for name, digest in values.items()]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
