"""Resolve historical storage names without rewriting scientific records.

Old configuration strings remain part of their original hash identities. Only
file lookup changes; callers still verify the bytes against the saved digests.
Unknown paths remain unchanged. An existing old checkout is never preferred
over the consolidated project.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
_PHASES = ("01_baselines", "02_gpfa_reliability", "03_parameter_matching", "04_model_comparison")


def _relocated(value: str | Path) -> Path | None:
    text = str(value).replace("\\", "/")
    for phase in _PHASES:
        marker = f"experiments/{phase}"
        start = text.find(marker)
        if start >= 0 and (start == 0 or text[start - 1] == "/"):
            tail = text[start + len(marker):]
            if tail and not tail.startswith("/"):
                continue
        elif text == f"../{phase}" or text.startswith(f"../{phase}/"):
            tail = text[len(f"../{phase}"):]
        else:
            continue
        suffix = Path(tail.lstrip("/"))
        if suffix.parts and suffix.parts[0] == "records":
            if phase == "01_baselines":
                return PROJECT_ROOT / "records/baselines" / Path(*suffix.parts[1:])
            if phase == "03_parameter_matching":
                return PROJECT_ROOT / "records/parameter_matching" / Path(*suffix.parts[1:])
        return PROJECT_ROOT / suffix
    return None


def resolve_path(value: str | Path, *, base: str | Path = PROJECT_ROOT) -> Path:
    """Resolve a declared path without fallback searches or identity edits."""
    relocated = _relocated(value)
    if relocated is not None:
        return relocated.resolve()
    path = Path(value)
    if not path.is_absolute():
        root = _relocated(base) or Path(base)
        path = root / path
    return path.resolve()


def repository_path(value: str | Path) -> Path:
    return resolve_path(value, base=REPOSITORY_ROOT)
