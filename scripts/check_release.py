"""Validate publication-package scope without accessing BOPTEST."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cff", ".json", ".md", ".ps1", ".py", ".toml", ".txt", ".yml", ".yaml"}
FORBIDDEN_ROOTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "htmlcov",
    "legacy",
    "models",
    "outputs",
    "runs",
    "runtime",
    "upstream",
    "wandb",
}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
PERSONAL = re.compile(r"(?i)([A-Z]:\\Users\\[^\\]+\\|/home/[^/]+/)")
SECRET = re.compile(r"(?i)(api[_-]?key\s*[=:]\s*['\"][^'\"]+|gh[pousr]_[A-Za-z0-9]{20,})")


def main() -> None:
    present_forbidden = sorted(name for name in FORBIDDEN_ROOTS if (ROOT / name).exists())
    if present_forbidden:
        raise SystemExit(f"Forbidden release directories: {present_forbidden}")

    violations: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts or ".venv" in path.parts:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE"}:
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        if CJK.search(text):
            violations.append(f"non-English CJK text: {relative}")
        if path.resolve() != Path(__file__).resolve():
            if PERSONAL.search(text):
                violations.append(f"personal path or identifier: {relative}")
            if SECRET.search(text):
                violations.append(f"possible embedded credential: {relative}")
        if path.suffix.lower() == ".md":
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                target = target.strip().strip("<>").split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                if not (path.parent / target).resolve().is_file():
                    violations.append(f"broken local link in {relative}: {target}")

    notebook_path = ROOT / "notebooks" / "Monitor_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "".join(cell.get("source", [])).strip()
    ]
    if len(code_cells) != 3:
        violations.append("monitor notebook must contain exactly three non-empty code cells")
    for cell in code_cells:
        if cell.get("execution_count") is not None or cell.get("outputs"):
            violations.append("monitor notebook contains saved execution state")
        compile("".join(cell["source"]), str(notebook_path), "exec")

    required = {
        "CITATION.cff",
        "CHECKSUMS.sha256",
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "requirements.txt",
    }
    missing = sorted(name for name in required if not (ROOT / name).is_file())
    if missing:
        violations.append(f"missing release files: {missing}")
    if violations:
        raise SystemExit("Release validation failed:\n- " + "\n- ".join(violations))
    print("Release validation passed: English-only, scoped, credential-free source package.")


if __name__ == "__main__":
    main()
