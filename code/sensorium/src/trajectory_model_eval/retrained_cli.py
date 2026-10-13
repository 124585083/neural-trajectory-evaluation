"""Run the isolated retraining protocol without changing historical defaults."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .unified_contract import canonical_digest
from .unified_analysis import _clean

REPOSITORY = Path(__file__).resolve().parents[4]
PROJECT = REPOSITORY / "code/sensorium"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stage_status(root):
    """Report persisted evidence only; the presence of a directory is not success."""
    records = {
        "split": "data/split_manifest.json",
        "normalization": "data/preprocessing.json",
        "static": "encoders/static/training_summary.json",
        "dynamic": "encoders/dynamic/training_summary.json",
        "gpfa": "gpfa/frozen_gpfa_identity.json",
        "analysis_freeze": "frozen/frozen_analysis.json",
        "evaluation": "evaluation/summary.json",
    }
    result = {}
    for name, relative in records.items():
        path = root / relative
        if path.is_file():
            record = read_json(path)
            result[name] = {key: record[key] for key in (
                "status", "split_sha256", "preprocessing_sha256", "best_epoch",
                "validation_score", "last_epoch", "stop_reason", "parameter_digest") if key in record}
            result[name]["record"] = relative
            if name == "gpfa":
                result[name].update(status="gpfa_frozen", gpfa_parameter_digest=record["gpfa_parameter_digest"])
        else:
            fallback = root / (f"encoders/{name}/status.json" if name in {"static", "dynamic"} else
                               "gpfa/status.json" if name == "gpfa" else "__no_status__")
            result[name] = read_json(fallback) if fallback.is_file() else {"status": "no_completion_record"}
    return result


def main():
    parser = argparse.ArgumentParser(description="Sensorium: held-out encoding and trajectory evaluation")
    parser.add_argument("command", choices=("prepare", "smoke", "train", "fit-gpfa", "predict-development", "calibrate", "predict-test", "evaluate", "report", "status"))
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/retrained_v1.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/retrained_v1")
    parser.add_argument("--public-dir", type=Path, help="Explicit compact-report destination for a separate reproduction")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source-inventory", type=Path,
                        default=REPOSITORY / "results/sensorium/evidence/04_model_comparison/unified_v1/split_manifest.json")
    parser.add_argument("--sensorium-source", type=Path)
    parser.add_argument("--neuralpredictors-source", type=Path)
    parser.add_argument("--kind", choices=("static", "dynamic"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume this run only; never load historical training state")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = read_json(args.config)
    root = args.output_dir.resolve()
    if args.command == "status":
        print(json.dumps(_clean(stage_status(root)), indent=2, allow_nan=False))
        return
    if args.command == "report":
        from .retrained_reporting import publish_status
        primary = (PROJECT / "outputs/retrained_v1").resolve()
        canonical_public = (REPOSITORY / "results/sensorium/evidence/04_model_comparison/retrained_v1").resolve()
        public = args.public_dir.resolve() if args.public_dir else canonical_public
        if root != primary and public == canonical_public:
            parser.error("A separate reproduction requires --public-dir outside the canonical primary records")
        result = publish_status(root, public)
        print(json.dumps(_clean(result), indent=2, allow_nan=False))
        return
    if args.data_root is None:
        parser.error("This command requires --data-root")
    root.mkdir(parents=True, exist_ok=True)
    protocol = root / "analysis_protocol.json"
    if protocol.exists() and canonical_digest(read_json(protocol)) != canonical_digest(config):
        parser.error("Output directory belongs to a different frozen protocol")
    if not protocol.exists():
        protocol.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest_path, preprocessing_path = root / "data/split_manifest.json", root / "data/preprocessing.json"
    if args.command == "prepare":
        from .retrained_data import build_split, fit_preprocessing, validate_split
        if manifest_path.exists():
            if not args.resume:
                parser.error("Split already exists; --resume verifies and reuses it")
            manifest = read_json(manifest_path)
            validate_split(manifest)
        else:
            manifest = build_split(args.data_root, args.source_inventory, root / "data",
                                   seed=config["project"]["seed"], validation_fraction=config["data"]["validation_fraction"], config=config)
        if preprocessing_path.exists():
            if not args.resume:
                parser.error("Normalization already exists; use --resume to verify it")
            from .retrained_data import load_preprocessing
            for session in manifest["sessions"]:
                load_preprocessing(preprocessing_path, session)
            result = read_json(preprocessing_path)
        else:
            result = fit_preprocessing(args.data_root, manifest, root / "data")
        print(json.dumps({"status": "split_and_train_normalization_complete",
                          "split_sha256": manifest["split_sha256"],
                          "preprocessing_sha256": result["preprocessing_sha256"]}, indent=2))
        return
    manifest = read_json(manifest_path)
    if args.command in {"smoke", "train"}:
        if not args.kind:
            parser.error("smoke/train requires --kind static or dynamic")
        from .retrained_predictions import verify_source_files
        verify_source_files(manifest, args.data_root, roles=("train", "validation"), sessions=manifest["sessions"])
        from .retrained_training import configure_sources, train_encoder
        configure_sources(args.sensorium_source, args.neuralpredictors_source)
        result = train_encoder(args.kind, config, manifest, preprocessing_path, args.data_root,
                               root / ("smoke" if args.command == "smoke" else "encoders") / args.kind,
                               resume=args.resume, smoke=args.command == "smoke", device=args.device)
    else:
        result = _analysis_command(args, config, manifest, preprocessing_path, root)
    print(json.dumps(_clean(result), indent=2, default=str, allow_nan=False))


def _analysis_command(args, config, manifest, preprocessing_path, root):
    import gc
    import pandas as pd
    from .retrained_analysis import (fit_gpfa, freeze_analysis, load_neural_development,
                                     require_frozen, evaluate_pilot, score_session_response,
                                     summarize_session_responses)
    from .retrained_data import load_preprocessing
    from .retrained_predictions import (export_predictions, load_bundle, combine_pilot,
                                        select_neurons, bundle_identity, verify_source_files, verify_test_lock)

    preprocessing = read_json(preprocessing_path)
    pilot = manifest["session"]
    if args.command == "fit-gpfa":
        statistics = load_preprocessing(preprocessing_path, pilot)
        bundle = load_neural_development(args.data_root, manifest, config, statistics["response_precision"])
        fitted = fit_gpfa(bundle, manifest, config, preprocessing["preprocessing_sha256"], root / "gpfa")
        return fitted.identity
    from .retrained_training import configure_sources
    configure_sources(args.sensorium_source, args.neuralpredictors_source)
    checkpoints = {kind: read_json(root / "encoders" / kind / "frozen_identity.json")
                   for kind in ("static", "dynamic")}
    if args.command == "predict-development":
        return export_predictions(config, manifest, preprocessing_path, args.data_root, root / "encoders",
                           root / "development", roles=("train", "validation"), sessions=[pilot],
                           pilot_only=True, device=args.device)
    if args.command == "calibrate":
        verify_source_files(manifest, args.data_root, roles=("train", "validation"), sessions=[pilot])
        expected = bundle_identity(config, manifest, preprocessing, checkpoints, pilot,
                                   ("train", "validation"), manifest["neurons"]["indices"])
        bundle = load_bundle(root / "development" / pilot / "aligned_predictions.npz", expected)
        return freeze_analysis(bundle, manifest, config, preprocessing["preprocessing_sha256"],
                               root / "gpfa", checkpoints, root / "frozen")
    frozen = verify_test_lock(config, manifest, preprocessing, root / "encoders", root / "frozen", root / "gpfa")
    if frozen["checkpoint_identities"] != checkpoints:
        raise ValueError("Encoder identities changed after final analysis freezing")
    if args.command == "predict-test":
        return export_predictions(config, manifest, preprocessing_path, args.data_root, root / "encoders",
                                  root / "test", roles=("test",), sessions=manifest["sessions"], device=args.device,
                                  freeze_dir=root / "frozen", gpfa_dir=root / "gpfa")
    if args.command != "evaluate":
        raise ValueError("Unknown analysis stage")
    if (root / "evaluation").exists() and any((root / "evaluation").iterdir()):
        raise FileExistsError("Evaluation outputs already exist; preserve the completed results")
    pooled, condition, pilot_test = [], [], None
    for session in manifest["sessions"]:
        verify_source_files(manifest, args.data_root, roles=("test",), sessions=[session])
        expected = bundle_identity(config, manifest, preprocessing, checkpoints, session,
                                   ("test",), manifest["session_neurons"][session]["all_indices"], frozen_analysis=frozen)
        bundle = load_bundle(root / "test" / session / "aligned_predictions.npz", expected)
        scores, conditions = score_session_response(bundle, config)
        pooled.append(scores)
        condition.append(conditions)
        if session == pilot:
            pilot_test = select_neurons(bundle, manifest["neurons"]["indices"])
        del bundle
        gc.collect()
    response = summarize_session_responses(pd.concat(pooled, ignore_index=True),
                                           pd.concat(condition, ignore_index=True), config,
                                           root / "evaluation/response")
    expected = bundle_identity(config, manifest, preprocessing, checkpoints, pilot,
                               ("train", "validation"), manifest["neurons"]["indices"])
    development = load_bundle(root / "development" / pilot / "aligned_predictions.npz", expected)
    combined = combine_pilot(development, pilot_test, manifest)
    result = evaluate_pilot(combined, manifest, config, root / "gpfa", root / "frozen", root / "evaluation/pilot")
    summary = {"status": "held_out_evaluation_after_retraining_complete", "pilot": result,
               "response": response, "split_sha256": manifest["split_sha256"],
               "freeze_sha256": frozen["freeze_sha256"]}
    (root / "evaluation/summary.json").write_text(json.dumps(_clean(summary), indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
