"""Copy the frozen cached-replay inputs without copying the original run history.

This local packaging helper does not grant redistribution rights. It preserves
all source files and copies only the dependencies of the retained replay plus
the training-side preprocessing arrays. It never fits or recalibrates a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .replay import preflight, verify_checkpoint_inputs, verify_files, write
from trajectory_model_eval.unified_contract import file_sha256


def inventory(source):
    source = Path(source).resolve()
    plan = preflight(source)
    names = set(plan["cache_checksums"])
    names.update(("analysis_protocol.json", "data/split_manifest.json", "data/preprocessing.json",
        "frozen/analysis_config.json", "frozen/response_matching_selection.csv",
        "encoders/static/frozen_identity.json", "encoders/dynamic/frozen_identity.json",
        "gpfa/frozen_gpfa_identity.json", "evaluation/summary.json",
        "evaluation/pilot/generated_manifest.json", "diagnostics/gpfa/source_identity.json",
        "diagnostics/gpfa/protocol.json",
        "diagnostics/encoder/source_snapshots/scoring_implementation_response_v2.py",
        "diagnostics/encoder/source_snapshots/scoring_implementation_zero_spatial_endpoint.py"))
    for record in plan["preprocessing"]["sessions"].values():
        name = "data/" + record["file"]
        verify_files(source, {name: record["sha256"]})
        names.add(name)
    files = {}
    for name in sorted(names):
        path = source / name
        if not path.resolve().is_relative_to(source) or not path.is_file():
            raise ValueError(f"Invalid frozen bundle input: {path}")
        files[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return {"schema": "sensorium_frozen_replay_bundle_v1", "freeze_sha256": plan["lock"]["freeze_sha256"],
        "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files.values()),
        "scope": "Retained cached replay and frozen preprocessing; no raw provider data or cancelled sensitivity fits",
        "redistribution": "Local packaging only; public distribution requires a separate rights decision",
        "files": files}


def checkpoint_inventory(source, matching_source):
    """Keep the selected model and both caches, with the original sidecar bytes."""
    plan = preflight(source)
    hashes = verify_checkpoint_inputs(plan, matching_source)
    root = Path(matching_source).resolve()
    return {"schema": "sensorium_checkpoint_control_bundle_v1",
        "selection_sha256": plan["checkpoint_selection"]["selection_sha256"],
        "source_freeze_sha256": plan["lock"]["freeze_sha256"],
        "scope": "Frozen checkpoint control replay; all candidate scores, selected weights, development and test caches",
        "file_count": len(hashes), "total_bytes": sum((root / name).stat().st_size for name in hashes),
        "files": {name: {"sha256": digest, "bytes": (root / name).stat().st_size} for name, digest in hashes.items()}}


def copy_bundle(source, destination, record, *, checkpoint_source=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if record.get("schema") == "sensorium_checkpoint_control_bundle_v1" and checkpoint_source is None:
        raise ValueError("Checkpoint bundle relocation requires the original main source root")
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Bundle destination must be separate from its source")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a bundle destination: {destination}")
    for name in record["files"]:
        relative = Path(name)
        if (relative.is_absolute() or relative.drive
                or not (source / relative).resolve().is_relative_to(source)
                or not (destination / relative).resolve().is_relative_to(destination)):
            raise ValueError(f"Bundle file must stay within source and destination: {name}")
    parent = destination.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < record["total_bytes"] + 1024**3:
        raise OSError("Insufficient free space for the complete verified bundle")
    verify_files(source, {name: item["sha256"] for name, item in record["files"].items()})
    for name, item in record["files"].items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
        verify_files(destination, {name: item["sha256"]})
    # Relocation is accepted only if the public entry point verifies the copy.
    if record.get("schema") == "sensorium_checkpoint_control_bundle_v1":
        relocated = checkpoint_inventory(checkpoint_source, destination)
        if relocated["files"] != record["files"]:
            raise ValueError("Relocated checkpoint control differs from its source")
    else:
        preflight(destination)
    write(destination / "bundle_manifest.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--copy", action="store_true", help="Copy after verification; otherwise inventory only")
    parser.add_argument("--matching-source", type=Path, help="Package this checkpoint-control root separately")
    parser.add_argument("--matching-destination", type=Path, help="Destination for the separate checkpoint-control bundle")
    args = parser.parse_args(argv)
    if args.copy and args.destination is None:
        parser.error("--copy requires --destination")
    if args.copy and args.matching_source is not None and args.matching_destination is None:
        parser.error("--matching-source with --copy requires --matching-destination")
    record = inventory(args.source)
    matching = checkpoint_inventory(args.source, args.matching_source) if args.matching_source is not None else None
    print(json.dumps({key: value for key, value in record.items() if key != "files"}, indent=2), flush=True)
    if args.copy:
        copy_bundle(args.source, args.destination, record)
        print(json.dumps({"status": "verified_bundle_copied", "destination": str(args.destination.resolve())}), flush=True)
        if matching is not None:
            copy_bundle(args.matching_source, args.matching_destination, matching, checkpoint_source=args.destination)
            print(json.dumps({"status": "verified_checkpoint_bundle_copied", "destination": str(args.matching_destination.resolve())}), flush=True)
    elif matching is not None:
        print(json.dumps({key: value for key, value in matching.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
