"""Prepare an ignored, allowlisted artifact bundle; never publish payloads.

Source and destination bytes are checked against the central inventory. No
model is deserialized, trained, or evaluated here. Original source records are
copied unchanged, including their historical provenance and freeze identities.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

MODULE = Path(__file__).resolve().parents[1]
REPOSITORY = MODULE.parents[1]
DESTINATION = REPOSITORY / "models/mental_pong/current"
SCHEMA = "mental_pong_local_artifacts_v1"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_relative(value):
    text = str(value).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ":" in text or ".." in path.parts:
        raise ValueError("Artifact path must remain inside its declared root")
    if not text.startswith("trajectory_project/") or path.suffix not in (".json", ".csv", ".npz", ".joblib"):
        raise ValueError("Only inventoried final-study artifacts may be materialized")
    return path


def contained(root, relative):
    """Reject resolved children that escape through a junction or symlink."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes its declared root")
    return path


def inventory():
    source = REPOSITORY / "models/manifest.json"
    rows = [r for r in json.loads(source.read_text(encoding="utf-8"))["artifacts"]
            if r["project"] == "mental_pong"]
    if not rows or len(rows) != len({r["relative_path"] for r in rows}):
        raise ValueError("The Mental-Pong inventory is empty or ambiguous")
    for row in rows:
        safe_relative(row["relative_path"])
    return rows, source


def ignored_destination(destination):
    destination = Path(destination).resolve()
    expected = contained(REPOSITORY, DESTINATION.relative_to(REPOSITORY))
    if destination != expected:
        raise ValueError("Local preparation is restricted to models/mental_pong/current")
    probe = destination / "source_artifacts" / "inventory-probe.json"
    result = subprocess.run(["git", "check-ignore", "-q", "--", str(probe)],
                            cwd=REPOSITORY, capture_output=True)
    if result.returncode:
        raise ValueError("The complete local artifact directory must be Git-ignored first")
    return destination


def mini_core():
    spec = importlib.util.spec_from_file_location('local_mini_contract', MODULE / 'mini_replay/replay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_mini(folder):
    folder = Path(folder).resolve()
    records = json.loads(contained(folder, 'manifest.json').read_text(encoding='utf-8'))
    for relative in records['files']:
        contained(folder, relative)
    manifest, _, _ = mini_core().load_bundle(folder)
    allowed = {'manifest.json', *manifest['files']}
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual != allowed:
        raise ValueError('Mini export includes files outside its declared allowlist')
    return allowed


def file_records(rows):
    return {r['relative_path']: {'sha256': r['sha256'], 'bytes': r['bytes'],
        'purpose': r['purpose'], 'artifact_kind': r.get('artifact_kind'),
        'source_artifact_id': 'source_pilot://' + r['relative_path']} for r in rows}


def install_mini(source):
    """Install an already exported private mini after checking its entire payload."""
    destination = ignored_destination(DESTINATION)
    if not contained(destination, 'bundle_manifest.json').is_file():
        raise ValueError('Prepare the checked local artifact bundle first')
    source = Path(source).resolve()
    allowed = verify_mini(source)
    target = contained(destination, 'mini')
    if target.exists():
        raise FileExistsError('Preserve the existing installed mini package')
    for relative in sorted(allowed):
        output = contained(target, relative)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contained(source, relative), output)
        if sha(output) != sha(contained(source, relative)):
            raise ValueError('Installed mini file checksum differs')
    verify_mini(target)
    return {'status': 'installed_private_mini', 'files': len(allowed),
            'public_payload_permitted': False, 'new_model_fitting': False}


def verify(folder=DESTINATION):
    folder = Path(folder).resolve()
    manifest = json.loads(contained(folder, "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise ValueError("The local artifact bundle is incomplete or unsupported")
    source = contained(folder, "source_artifacts")
    for relative, record in manifest["files"].items():
        path = contained(source, safe_relative(relative))
        if (not path.is_file() or path.stat().st_size != record["bytes"]
                or sha(path) != record["sha256"]):
            raise ValueError(f"Missing or changed local artifact: {relative}")
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]):
        raise ValueError("Unexpected files exist outside the local artifact allowlist")
    mini_status = 'not_installed'
    mini = contained(folder, 'mini')
    if contained(mini, 'manifest.json').is_file():
        verify_mini(mini)
        mini_status = 'verified_private_package'
    return {"status": "verified", "files": len(actual), "mini": mini_status,
            "payload_bytes": sum(r["bytes"] for r in manifest["files"].values()),
            "public_payload_permitted": False, "source_pilot_accessed": False,
            "new_model_fitting": False}


def prepare(source_root, *, estimate=False):
    rows, identity = inventory()
    source = Path(source_root).resolve()
    destination = ignored_destination(DESTINATION)
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Original source and local bundle must be separate trees")
    expected_bytes = sum(r["bytes"] for r in rows)
    for row in rows:
        path = contained(source, safe_relative(row["relative_path"]))
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Source artifact missing or wrong size: {row['relative_path']}")
    result = {"files": len(rows), "payload_bytes": expected_bytes,
              "free_bytes": shutil.disk_usage(REPOSITORY).free,
              "public_payload_permitted": False}
    if estimate:
        return {"status": "estimate_only", **result}
    manifest_path = contained(destination, 'bundle_manifest.json')
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('files') != file_records(rows):
            raise ValueError('The existing bundle inventory differs; preserve it separately')
        # Completed bundles are never repaired or extended in place.
        return verify(destination)
    if result["free_bytes"] < expected_bytes * 2:
        raise ValueError("Insufficient space for a checked local copy")
    # Verify every source before creating output. Existing complete bundles are
    # read-only; a partial copy can resume only when its bytes already agree.
    target_root = contained(destination, 'source_artifacts')
    for row in rows:
        contained(target_root, safe_relative(row['relative_path']))
    for row in rows:
        if sha(contained(source, row["relative_path"])) != row["sha256"]:
            raise ValueError(f"Source checksum differs: {row['relative_path']}")
    for row in rows:
        target = contained(target_root, row["relative_path"])
        if target.exists():
            if sha(target) != row["sha256"]:
                raise ValueError(f"Existing local artifact differs: {row['relative_path']}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contained(source, row["relative_path"]), target)
        if sha(target) != row["sha256"]:
            raise ValueError(f"Copied artifact checksum differs: {row['relative_path']}")
    manifest = {"schema": SCHEMA, "complete": True,
        "inventory_sha256_at_preparation": sha(identity), "runtime_alias": "source_pilot",
        "runtime_relative_root": "source_artifacts", "public_payload_permitted": False,
        "redistribution_status": "Unresolved; ignored local preparation only",
        "scientific_status": "CLOSED_EXPLORATORY_WITH_LIMITATIONS", "raw_preprocessing": "fail",
        "files": file_records(rows)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return verify(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--install-mini", type=Path,
                        help="Install a checked private mini exported outside the repository")
    args = parser.parse_args()
    if args.install_mini:
        result = install_mini(args.install_mini)
    elif args.check:
        result = verify()
    else:
        if args.source_root is None:
            parser.error("--source-root is required for estimation or preparation")
        result = prepare(args.source_root, estimate=args.estimate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
