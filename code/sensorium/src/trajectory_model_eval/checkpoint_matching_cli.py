"""Explicit checkpoint scoring/export stages; never alters original main artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .checkpoint_matching import (freeze_selection, load_selection, score_candidate,
                                  validation_reference, validate_candidate_training, checkpoint_path,
                                  validate_cohort_completion, COHORT_RULE, SCHEMA, KIND)
from .unified_contract import canonical_digest, file_sha256


PROJECT = Path(__file__).resolve().parents[2]
REPOSITORY = Path(__file__).resolve().parents[4]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Immutable checkpoint artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _context(args):
    from .retrained_analysis import matching_source_identity
    from .retrained_predictions import bundle_identity, load_bundle, verify_test_lock
    root = args.source_root
    config = _read(root / "analysis_protocol.json")
    manifest = _read(root / "data/split_manifest.json")
    preprocessing_path = root / "data/preprocessing.json"
    preprocessing = _read(preprocessing_path)
    lock = verify_test_lock(config, manifest, preprocessing, root / "encoders", root / "frozen", root / "gpfa")
    source = matching_source_identity(config, lock)
    method = _read(args.config)
    if (method.get("schema_version") != SCHEMA or method.get("kind") != KIND
            or method.get("candidate_epochs") != list(range(1, 120))
            or method.get("candidate_cohort_rule") != COHORT_RULE
            or method.get("maximum_candidate_epoch") != 119
            or method.get("native_early_stop_reason") != "patience_cycles_exhausted"
            or method.get("selection_role") != "validation" or method.get("tolerance") != .005
            or method.get("selection_summary") != "mean_per_neuron_pearson_over_trial_time"
            or method.get("neuron_support") != "fixed_pilot512_common_finite_across_all_candidates"
            or method.get("test_used_for_selection") is not False
            or method.get("tie_break") != ["absolute_gap", "epoch", "sha256"]
            or method.get("source_protocol_sha256") != source["source_config_sha256"]):
        raise ValueError("Checkpoint matching config differs from the fixed source/method/cohort")
    expected = bundle_identity(config, manifest, preprocessing, lock["checkpoint_identities"], manifest["session"],
        ("train", "validation"), manifest["neurons"]["indices"])
    development = load_bundle(root / "development" / manifest["session"] / "aligned_predictions.npz", expected)
    return config, manifest, preprocessing_path, source, method, development


def _completion_evidence(args, record, method):
    """Verify native stopping without changing the running training wrapper."""
    summary_path = args.recovery_root / "training_summary.json"
    if not summary_path.exists():
        return None
    summary = _read(summary_path)
    epoch = int(summary.get("last_epoch", 0))
    native = (summary.get("status") == "training_complete"
              and summary.get("stop_reason") == "patience_cycles_exhausted" and 1 <= epoch <= 119)
    capped = (epoch == 119 and record.get("complete") is True
              and (summary.get("status"), summary.get("stop_reason")) in {
                  ("incomplete_resumable", "bounded_invocation_resumable"),
                  ("training_complete", "patience_cycles_exhausted")})
    if not native and not capped:
        return None
    epochs = sorted(row["epoch"] for row in record["candidates"])
    if epochs != list(range(1, epoch + 1)):
        raise ValueError("Native completion does not cover every completed epoch exactly once")
    training_digest = canonical_digest(record["training_identity"])
    if summary.get("identity_digest") != training_digest or summary.get("test_read_by_trainer") is not False:
        raise ValueError("Native training summary belongs to another run or used test data")
    last_path = args.recovery_root / "last.pt"
    if file_sha256(last_path) != summary.get("last_sha256"):
        raise ValueError("Native completion checkpoint differs from its summary")
    import torch
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    progress = last.get("progress", {})
    training = record["training_identity"].get("training", {})
    if (last.get("checkpoint_kind") != "part3_complete_epoch" or last.get("identity") != record["training_identity"]
            or last.get("identity_digest") != training_digest or progress.get("epoch") != epoch
            or progress.get("stop_reason") != summary["stop_reason"]
            or [row["epoch"] for row in progress.get("curve", [])] != list(range(1, epoch + 1))
            or any(training.get(key) != value for key, value in {"max_epochs": 200, "patience": 5, "lr_decay_steps": 4}.items())
            or (native and progress.get("cycle") != 4)):
        raise ValueError("Native completion does not follow the unchanged stopping policy")
    declaration_path = args.matching_root / "protocol_revision.json"
    if declaration_path.exists():
        revision = _read(declaration_path)
        if (revision.get("revision_sha256") != canonical_digest({key: value for key, value in revision.items() if key != "revision_sha256"})
                or revision.get("candidate_cohort_rule") != COHORT_RULE or revision.get("test_loaded") is not False
                or revision.get("candidate_validation_scores_read") is not False
                or revision.get("checkpoint_selection_started") is not False):
            raise ValueError("Cohort revision was not declared before checkpoint scoring")
        # Preserve the already frozen current-run proof byte-for-byte in meaning.
        declaration_identity = {"protocol_revision_sha256": revision["revision_sha256"]}
    else:
        declaration_path = args.recovery_root / "recovery_protocol.json"
        declaration = _read(declaration_path)
        digest = canonical_digest({key: value for key, value in declaration.items() if key != "declaration_sha256"})
        if (declaration.get("schema_version") != "sensorium_checkpoint_recovery_v2"
                or declaration.get("cohort_rule") != COHORT_RULE
                or declaration.get("candidate_epochs") != list(range(1, 120))
                or declaration.get("maximum_candidate_epoch") != 119
                or declaration.get("native_early_stop_reason") != "patience_cycles_exhausted"
                or declaration.get("test_loaded") is not False
                or declaration.get("candidate_validation_scores_read") is not False
                or declaration.get("checkpoint_selection_started") is not False
                or not declaration.get("created_at_utc")
                or declaration.get("declaration_sha256") != digest
                or record["training_identity"].get("recovery_declaration_sha256") != digest):
            raise ValueError("Fresh cohort declaration is missing, invalid, or not bound to the training identity")
        declaration_identity = {"protocol_declaration_kind": "recovery_protocol_v2",
                                "protocol_declaration_sha256": digest}
    snapshot_path = args.matching_root / "checkpoint_matching_config.json"
    if snapshot_path.exists():
        if _read(snapshot_path) != method:
            raise ValueError("The archived matching config differs from the requested method")
    else:
        _write(snapshot_path, method)
    evidence_paths = [summary_path, last_path, args.recovery_root / "candidate_manifest.json", declaration_path, snapshot_path]
    proof = {"schema_version": "sensorium_checkpoint_cohort_completion_v1", "candidate_cohort_rule": COHORT_RULE,
        "maximum_candidate_epoch": 119, "final_epoch": epoch, "effective_candidate_epochs": epochs,
        "completion_kind": "native_early_stop" if native else "epoch_cap_reached", "test_loaded": False,
        "training_identity_digest": training_digest, "matching_config_sha256": canonical_digest(method),
        **declaration_identity,
        "training_summary": {key: summary[key] for key in ("status", "stop_reason", "last_epoch", "identity_digest", "test_read_by_trainer")},
        "last_checkpoint": {"epoch": epoch, "cycle": progress["cycle"], "stop_reason": progress["stop_reason"],
                            "identity_digest": last["identity_digest"]},
        "evidence_files": [{"path": path.resolve().relative_to(args.matching_root.resolve()).as_posix(),
                            "sha256": file_sha256(path)} for path in evidence_paths]}
    proof["completion_sha256"] = canonical_digest(proof)
    validate_cohort_completion(proof, epochs, method)
    return proof


def _candidate_manifest(args, source, method):
    path = args.recovery_root / "candidate_manifest.json"
    while True:
        record = _read(path) if path.exists() else None
        if record is not None:
            if (record.get("schema_version") != "sensorium_epoch_candidates_v1"
                    or record.get("declared_epochs") != method["candidate_epochs"]
                    or record.get("test_loaded") is not False
                    or (record.get("training_identity", {}).get("source_training_identity_digest")
                        != source["checkpoint_identities"]["dynamic"]["identity_digest"])
                    or record.get("training_identity", {}).get("scope") != "checkpoint_candidate_replay"):
                raise ValueError("Recovery candidate manifest differs from the declared training run/cohort")
            proof = _completion_evidence(args, record, method)
            if proof is not None:
                return record, proof
        if not args.watch:
            raise ValueError("Candidate cohort needs the completed epoch cap or verified native early stopping; pauses and interruptions do not qualify")
        time.sleep(min(60., max(1., args.poll_seconds)))


def score_candidates(args):
    from .retrained_data import create_dataloaders
    from .retrained_predictions import predict_checkpoint_bundle, verify_source_files
    from .retrained_training import configure_sources
    config, manifest, prep_path, source, method, development = _context(args)
    if (args.matching_root / "checkpoint_selection.json").exists():
        return load_selection(args.matching_root, source, verify_checkpoint=True)
    inventory, completion = _candidate_manifest(args, source, method)
    candidates = sorted(inventory["candidates"], key=lambda row: row["epoch"])
    candidates = [{**row, "checkpoint_path": Path(row["checkpoint_path"]).resolve().relative_to(
        args.matching_root.resolve()).as_posix()} for row in candidates]
    if [row["epoch"] for row in candidates] != completion["effective_candidate_epochs"]:
        raise ValueError("Complete recovery manifest does not contain every declared epoch exactly once")
    training_digest = canonical_digest(inventory["training_identity"])
    for candidate in candidates:
        validate_candidate_training(candidate, source)
        if candidate["training_identity_digest"] != training_digest:
            raise ValueError("Candidate training identity differs from the declared recovery cohort")
        if file_sha256(checkpoint_path(candidate, args.matching_root)) != candidate["sha256"]:
            raise ValueError("A declared candidate checkpoint changed before scoring")
    completion_path = args.matching_root / "cohort_completion.json"
    if completion_path.exists():
        if _read(completion_path) != completion:
            raise ValueError("Frozen external cohort completion evidence changed")
    else:
        _write(completion_path, completion)
    reference = validation_reference(development)
    configure_sources(args.sensorium_source, args.neuralpredictors_source)
    verify_source_files(manifest, args.data_root, roles=("validation",), sessions=[manifest["session"]])
    train_loaders = create_dataloaders(args.data_root, manifest, prep_path)["train"]
    scores, model = [], None
    for candidate in candidates:
        path = args.matching_root / "candidate_scores" / f"epoch_{candidate['epoch']:03d}.json"
        if path.exists():
            score = _read(path)
            unsigned = {key: value for key, value in score.items() if key != "score_sha256"}
            if (score.get("score_sha256") != canonical_digest(unsigned) or score.get("checkpoint") != candidate
                    or score.get("reference_sha256") != reference["reference_sha256"]):
                raise ValueError("Stored candidate score differs from its checkpoint or validation reference")
            score = unsigned
        else:
            bundle, model = predict_checkpoint_bundle(config, manifest, prep_path, args.data_root, candidate,
                roles=("validation",), device=args.device, train_loaders=train_loaders, model=model,
                verify_sources=False, checkpoint_root=args.matching_root)
            score = score_candidate(reference, bundle, candidate)
            _write(path, {**score, "score_sha256": canonical_digest(score)})
            del bundle
        scores.append(score)
        print(json.dumps({"scored_epoch": candidate["epoch"], "candidate_count": len(scores),
                          "declared_count": len(candidates)}), flush=True)
    # A concurrent source edit invalidates the entire cohort before selection.
    verify_source_files(manifest, args.data_root, roles=("validation",), sessions=[manifest["session"]])
    selection = freeze_selection(reference, scores, source, expected_epochs=completion["effective_candidate_epochs"],
                                 tolerance=method["tolerance"], output_dir=args.matching_root,
                                 cohort_completion=completion, matching_config=method)
    return selection


def export_selected(args):
    from .retrained_predictions import (attach_matched_prediction, predict_checkpoint_bundle,
                                        load_matched_bundle, save_matched_bundle)
    from .retrained_training import configure_sources
    config, manifest, prep_path, source, _, development = _context(args)
    selection = load_selection(args.matching_root, source, verify_checkpoint=True)
    configure_sources(args.sensorium_source, args.neuralpredictors_source)
    summaries, model = {}, None
    for name, roles in (("development", ("train", "validation")), ("test", ("test",))):
        path = args.matching_root / "predictions" / name / manifest["session"] / "matched_predictions.npz"
        if path.exists():
            bundle = load_matched_bundle(path, selection, reference_bundle=development if name == "development" else None)
            summaries[name] = {"status": "verified_existing", "path": str(path)}
        else:
            bundle, model = predict_checkpoint_bundle(config, manifest, prep_path, args.data_root,
                selection["selected_checkpoint"], roles=roles, device=args.device, selection=selection, model=model,
                checkpoint_root=args.matching_root)
            if name == "development":
                attach_matched_prediction(development, bundle)
                reference = validation_reference(development)
                rescored = score_candidate(reference, bundle, selection["selected_checkpoint"])
                selected = next(row for row in selection["candidates"] if row["selected"])
                if (rescored["per_neuron_r"] != selected["per_neuron_r"]
                        or rescored["validation_prediction_sha256"] != selected["validation_prediction_sha256"]):
                    raise ValueError("Selected validation predictions changed between scoring and final export")
            summaries[name] = save_matched_bundle(path, bundle, selection)
        del bundle
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score-candidates", "export-selected", "evaluate", "run"))
    parser.add_argument("--source-root", type=Path, default=REPOSITORY / "models/sensorium/retrained_v1")
    parser.add_argument("--recovery-root", type=Path, default=PROJECT / "outputs/checkpoint_matching_v1/recovery")
    parser.add_argument("--matching-root", type=Path, default=PROJECT / "outputs/checkpoint_matching_v1")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/checkpoint_matching_v1.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sensorium-source", type=Path)
    parser.add_argument("--neuralpredictors-source", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--watch", action="store_true", help="Wait for verified native stopping or the 119-epoch cap before GPU scoring")
    parser.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args(argv)
    if args.command != "evaluate" and args.data_root is None:
        parser.error("--data-root is required for checkpoint inference")
    if args.command in ("score-candidates", "run"):
        result = score_candidates(args)
        print(json.dumps({"selection_sha256": result["selection_sha256"],
            "selected_epoch": result["selected_checkpoint"]["epoch"],
            "validation_absolute_gap": result["validation_absolute_gap"]}), flush=True)
    if args.command in ("export-selected", "run"):
        export_selected(args)
    if args.command in ("evaluate", "run"):
        from .retrained_analysis import evaluate_checkpoint_control
        result = evaluate_checkpoint_control(args.source_root, args.matching_root,
                                            args.output_dir or args.matching_root / "evaluation")
        print(json.dumps({"evaluation_status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
