"""Provenance checks for Step 3, with an explicit documentation-only revision.

Scientific Step 1/2 files remain byte-identical. This module provides a combined
README; its historical bytes are archived and verified instead of rewriting
either earlier manifest to pretend the documentation never changed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def write_json(path, value, *, immutable=False):
    path = Path(path)
    value = json_safe(value)
    if immutable and path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen Step 3 execution inputs changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def record(path, role="source or output"):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "role": role}


def collect_records(value):
    found = {}
    def visit(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("path"), str) and isinstance(obj.get("sha256"), str):
                path = str(Path(obj["path"]).resolve())
                if path in found and found[path]["sha256"] != obj["sha256"]:
                    raise ValueError(f"Conflicting source signatures: {path}")
                found[path] = {"path": path, "sha256": obj["sha256"]}
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)
    visit(value)
    return list(found.values())


def verify_records(records):
    def check(row):
        path = Path(row["path"])
        return str(path) if not path.is_file() or sha(path) != row["sha256"] else None
    with ThreadPoolExecutor(max_workers=4) as pool:
        bad = [p for p in pool.map(check, records) if p]
    if bad:
        raise ValueError(f"Source/output checksum mismatch: {bad[:8]}")
    return len(records)


def verify_previous(project, step):
    project = Path(project).resolve()
    names = {1: "frozen_latents_manifest.json", 2: "analysis_protocol_manifest.json"}
    manifest_path = project / "artifacts" / names[step]
    manifest = read_json(manifest_path)
    if not manifest.get("completed"):
        raise ValueError(f"Step {step} is not complete")
    records = collect_records(manifest)
    revision_path = project / "notes/step3_readme_revision.json"
    substitutions = []
    verified = []
    for item in records:
        item = dict(item)
        if Path(item["path"]) == project / "README.md" and sha(item["path"]) != item["sha256"]:
            if not revision_path.is_file():
                raise ValueError("README changed without an explicit documentation revision record")
            revision = read_json(revision_path)
            if revision["original_sha256"] != item["sha256"]:
                raise ValueError("Documentation archive does not match the historical manifest")
            if sha(project / "README.md") != revision["current_sha256"]:
                raise ValueError("Current README differs from the recorded Step 3 documentation revision")
            archived = Path(revision["archived_original_path"]).resolve()
            if archived != project / "artifacts/history/README_step1.md":
                raise ValueError("Unexpected documentation archive location")
            if sha(archived) != item["sha256"]:
                raise ValueError("Historical README bytes were not preserved")
            substitutions.append({"original_path": item["path"], "original_sha256": item["sha256"],
                                  "checked_archive": str(archived), "reason": "combined three-step README; scientific files unchanged"})
            item["path"] = str(archived)
        verified.append(item)
    count = verify_records(verified)
    return {"step": step, "files_checked": count, "scientific_sources_unchanged": True,
            "documentation_revisions": substitutions, "manifest": record(manifest_path, f"unchanged Step {step} manifest"),
            "verified_source_records": verified}


def archive_readme(project):
    project = Path(project).resolve()
    dest = project / "artifacts/history/README_step1.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    current = project / "README.md"
    if not dest.exists():
        dest.write_bytes(current.read_bytes())
    expected = [r["sha256"] for r in collect_records(read_json(project / "artifacts/frozen_latents_manifest.json"))
                if Path(r["path"]) == current]
    if len(expected) != 1 or sha(dest) != expected[0]:
        raise ValueError("Cannot preserve the original Step 1 README")
    return dest


def update_readme(project, body):
    project = Path(project).resolve()
    original = archive_readme(project)
    current = project / "README.md"
    current.write_text(body, encoding="utf-8")
    write_json(project / "notes/step3_readme_revision.json", {
        "reason": "Combined execution commands and scientific narrative for the three completed steps",
        "original_sha256": sha(original), "archived_original_path": str(original),
        "current_sha256": sha(current), "current_path": str(current),
        "previous_manifests_modified": False, "scientific_configuration_or_artifact_changed": False,
        "verification_entrypoint": "run_step3.py --verify-step1 / --verify-step2 verifies archived README bytes and all unchanged scientific files",
    })
