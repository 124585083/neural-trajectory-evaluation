"""Replay the recorded Dynamic training protocol and archive epoch candidates.

This creates a new run. Initial weights must exactly equal the recorded initial
weights; later states are labelled replayed, not recovered historical weights.
No test data or full-sequence matching scores enter this training process.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time

import torch

from .retrained_training import (
    _digest, _file_digest, _save, _seed, _write_json, assert_training_roles,
    build_encoder, configure_sources, run_training, state_digest,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


COHORT_RULE = "all_completed_epochs_through_min_119_or_native_early_stop"


def declare_recovery(path, declaration, *, resume):
    """Write the cohort before training, retaining its exact declaration on resume."""
    path = Path(path)
    if path.exists():
        saved = read_json(path)
        unsigned = {key: value for key, value in saved.items() if key != "declaration_sha256"}
        settings = {key: value for key, value in unsigned.items() if key != "created_at_utc"}
        if (not resume or settings != declaration or not saved.get("created_at_utc")
                or saved.get("declaration_sha256") != _digest(unsigned)):
            raise ValueError("Existing recovery requires explicit resume with the same declaration")
        return saved
    if resume:
        raise ValueError("Cannot resume without the original recovery declaration")
    saved = {**declaration, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    saved["declaration_sha256"] = _digest(saved)
    _write_json(path, saved)
    return saved


def validate_recovery_completion(output, candidates, summary, identity, declared_epochs):
    """Accept the full cap or native stopping, never a pause or missing epoch."""
    output = Path(output)
    final_epoch = int(summary.get("last_epoch", 0))
    native = (summary.get("status") == "training_complete"
              and summary.get("stop_reason") == "patience_cycles_exhausted")
    capped = (final_epoch == max(declared_epochs)
              and summary.get("status") == "incomplete_resumable"
              and summary.get("stop_reason") == "bounded_invocation_resumable")
    if not (native or capped) or final_epoch not in declared_epochs:
        raise ValueError("Candidate recovery needs native early stopping or the declared epoch cap")
    epochs = [row["epoch"] for row in candidates]
    if epochs != list(range(1, final_epoch + 1)):
        raise ValueError("Candidate recovery is missing completed epochs or contains duplicates")
    digest = _digest(identity)
    if summary.get("identity_digest") != digest or summary.get("test_read_by_trainer") is not False:
        raise ValueError("Candidate completion summary differs from the training identity")
    last_path = output / "last.pt"
    if _file_digest(last_path) != summary.get("last_sha256"):
        raise ValueError("Candidate completion checkpoint differs from its summary")
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    progress = last.get("progress", {})
    if (last.get("checkpoint_kind") != "part3_complete_epoch"
            or last.get("identity") != identity or last.get("identity_digest") != digest
            or progress.get("epoch") != final_epoch
            or progress.get("stop_reason") != summary["stop_reason"]
            or [row["epoch"] for row in progress.get("curve", [])] != epochs
            or (native and progress.get("cycle") != int(identity["training"]["lr_decay_steps"]))):
        raise ValueError("Candidate completion does not follow the native training policy")
    return {"completion_kind": "native_early_stop" if native else "epoch_cap_reached",
            "effective_candidate_epochs": epochs, "final_epoch": final_epoch}


def recover(args):
    original = args.original_run.resolve()
    output = args.output_dir.resolve()
    if output == original or original in output.parents:
        raise ValueError("Recovery outputs must be outside the original run")
    output.mkdir(parents=True, exist_ok=True)
    candidate_dir = output / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    source_encoder = original / "encoders/dynamic"
    source_identity = read_json(source_encoder / "training_identity.json")
    source_curve = {int(row["epoch"]): row for row in read_json(source_encoder / "training_curve.json")}
    if max(source_curve) != 119:
        raise ValueError("The declared candidate cohort requires the original 119-epoch run")
    source_initial = torch.load(source_encoder / "initial.pt", map_location="cpu", weights_only=False)
    original_initial_digest = state_digest(source_initial["state_dict"])
    config = read_json(original / "analysis_protocol.json")
    manifest = read_json(original / "data/split_manifest.json")
    prep_path = original / "data/preprocessing.json"
    preprocessing = read_json(prep_path)
    if (source_identity["split_sha256"] != manifest["split_sha256"]
            or source_identity["preprocessing_sha256"] != preprocessing["preprocessing_sha256"]
            or source_identity["protocol_sha256"] != _digest(config)):
        raise ValueError("Original training inputs do not match their recorded identity")
    declaration = {
        "schema_version": "sensorium_checkpoint_recovery_v2",
        "candidate_epochs": list(range(1, 120)),
        "cohort_rule": COHORT_RULE,
        "maximum_candidate_epoch": 119,
        "native_early_stop_reason": "patience_cycles_exhausted",
        "candidate_state": "After both validation calls, before scheduler and patience-cycle restoration",
        "original_run": str(original), "original_training_identity": source_identity,
        "original_training_identity_digest": _digest(source_identity),
        "original_initial_state_sha256": original_initial_digest,
        "original_initial_checkpoint_sha256": _file_digest(source_encoder / "initial.pt"),
        "data_root": str(args.data_root.resolve()), "seed": 42,
        "test_loaded": False, "later_historical_states_verified": False,
        "candidate_validation_scores_read": False, "checkpoint_selection_started": False,
        "scope": "New same-protocol Dynamic replay; original Static, Dynamic and GPFA remain unchanged",
    }
    protocol_path = output / "recovery_protocol.json"
    declaration = declare_recovery(protocol_path, declaration, resume=args.resume)
    _write_json(output / "recovery_status.json", {"status": "checking_training_sources", "pid": os.getpid()})
    configure_sources(args.sensorium_source, args.neuralpredictors_source)
    from .retrained_data import create_dataloaders
    from .retrained_predictions import verify_source_files
    verify_source_files(manifest, args.data_root, roles=("train", "validation"), sessions=manifest["sessions"])
    _seed(42)
    loaders = create_dataloaders(args.data_root, manifest, prep_path, batch_size=8, frames=80)
    audit = assert_training_roles(loaders, manifest)
    model, architecture, counts = build_encoder("dynamic", loaders["train"], seed=42)
    if architecture["_architecture_sha256"] != source_identity["architecture_sha256"]:
        raise ValueError("Dynamic architecture bytes differ from the recorded run")
    training = copy.deepcopy(architecture["training"])
    if training != source_identity["training"] or counts != source_identity["parameter_counts"]:
        raise ValueError("Training settings or model dimensions differ from the recorded run")
    identity = copy.deepcopy(source_identity)
    identity.update(
        run_id="checkpoint_matching_v1_dynamic_seed42_replay",
        scope="checkpoint_candidate_replay",
        source_training_identity_digest=_digest(source_identity),
        recovery_declaration_sha256=declaration["declaration_sha256"],
        training_code_sha256=_file_digest(Path(inspect.getsourcefile(run_training))),
        recovery_code_sha256=_file_digest(Path(__file__)),
    )
    from neuralpredictors.training import early_stopping
    from sensorium.training.video_training_loop import standard_trainer
    source_hashes = {name: _file_digest(Path(inspect.getsourcefile(fn)))
                     for name, fn in (("standard_trainer", standard_trainer), ("early_stopping", early_stopping))}
    original_environment = read_json(source_encoder / "environment.json")
    if source_hashes != original_environment["official_source_hashes"]:
        raise ValueError("Pinned trainer or early-stopping source differs from the original run")
    _write_json(output / "environment.json", {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
        "official_source_hashes": source_hashes, "original_environment": original_environment,
    })
    _write_json(output / "loader_role_audit.json", audit)
    _write_json(output / "training_config.json", {"protocol": config, "architecture": architecture,
                                                 "training": training, "identity": identity})
    manifest_path = output / "candidate_manifest.json"
    candidates = read_json(manifest_path)["candidates"] if manifest_path.exists() else []

    def observe(epoch, state, metrics, current_identity):
        digest = state_digest(state)
        if epoch == 0:
            equal = digest == original_initial_digest
            _write_json(output / "initial_state_check.json", {
                "exact_match": equal, "original_state_sha256": original_initial_digest,
                "replay_state_sha256": digest, "test_loaded": False,
            })
            if not equal:
                raise ValueError("Replay initialization differs from the original initial checkpoint")
            print(json.dumps({"event": "initial_state_verified", "state_sha256": digest}), flush=True)
            return
        if epoch not in declaration["candidate_epochs"]:
            raise ValueError("Epoch is outside the declared cohort")
        reference = source_curve[epoch]
        differences = {key: float(metrics[key]) - float(reference[key]) for key in metrics}
        path = candidate_dir / f"epoch_{epoch:03d}.pt"
        payload = {"checkpoint_kind": "part3_epoch_candidate", "identity": current_identity,
                   "identity_digest": _digest(current_identity), "state_dict": state,
                   "state_sha256": digest, "epoch": epoch, **metrics}
        if path.exists():
            previous = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            if previous.get("state_sha256") != digest or previous.get("identity") != current_identity:
                raise ValueError(f"Replayed epoch {epoch} conflicts with an existing candidate")
        else:
            _save(path, payload)
        record = {"epoch": epoch, "checkpoint_path": str(path), "sha256": _file_digest(path),
                  "state_sha256": digest, "training_identity_digest": _digest(current_identity),
                  "identity": current_identity, **metrics, "original_curve_difference": differences}
        candidates[:] = [row for row in candidates if row["epoch"] != epoch] + [record]
        candidates.sort(key=lambda row: row["epoch"])
        _write_json(manifest_path, {
            "schema_version": "sensorium_epoch_candidates_v1", "declared_epochs": declaration["candidate_epochs"],
            "training_identity": current_identity, "test_loaded": False,
            "complete": len(candidates) == 119, "candidates": candidates,
        })
        _write_json(output / "curve_comparison.json", {
            "original_training_identity_digest": _digest(source_identity),
            "comparison": "Numeric training and snippet-validation curves, not proof of intermediate weight equality",
            "all_observed_curve_values_exact": all(all(v == 0 for v in r["original_curve_difference"].values()) for r in candidates),
            "rows": [{"epoch": r["epoch"], **r["original_curve_difference"]} for r in candidates],
        })
        _write_json(output / "recovery_status.json", {
            "status": "training", "pid": os.getpid(), "archived_epoch": epoch,
            "declared_final_epoch": 119, "curve_difference": differences, "unix_time": time.time(),
        })
        print(json.dumps({"event": "candidate_archived", "epoch": epoch, "curve_difference": differences}), flush=True)

    summary = run_training(model, loaders, training, output, identity, device=args.device,
                           resume=args.resume, stop_after_epoch=119, state_observer=observe)
    result = {"status": "candidate_cohort_incomplete", "candidate_cohort_rule": COHORT_RULE,
              "declared_epochs": declaration["candidate_epochs"], "archived_epochs": [r["epoch"] for r in candidates],
              "training_summary": summary, "test_loaded": False}
    try:
        completion = validate_recovery_completion(output, candidates, summary, identity,
                                                  declaration["candidate_epochs"])
    except ValueError:
        _write_json(output / "recovery_status.json", result)
        raise
    result.update(status="candidate_cohort_complete", **completion)
    inventory = read_json(manifest_path)
    inventory.update(complete=True, candidate_cohort_rule=COHORT_RULE, **completion)
    _write_json(manifest_path, inventory)
    _write_json(output / "recovery_status.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sensorium-source", type=Path, required=True)
    parser.add_argument("--neuralpredictors-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        recover(args)
    except BaseException as error:
        if args.output_dir.exists():
            _write_json(args.output_dir / "recovery_error.json", {
                "type": type(error).__name__, "message": str(error), "unix_time": time.time(),
            })
        raise


if __name__ == "__main__":
    main()
