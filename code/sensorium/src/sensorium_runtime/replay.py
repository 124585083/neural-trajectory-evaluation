"""Replay retained Sensorium analyses from immutable, completed caches.

This is not a provider-data rebuild. Missing inputs are errors, never triggers
for training, fitting, prediction export, or calibration. Historical training
and implementation checks remain referenced evidence, not newly run checks.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import io
import json
from pathlib import Path
from types import SimpleNamespace

from .paths import PROJECT_ROOT, REPOSITORY_ROOT
from trajectory_model_eval.unified_contract import canonical_digest, file_sha256

EVIDENCE = REPOSITORY_ROOT / "results/sensorium/evidence/04_model_comparison/retrained_v1_diagnostics"
ARTIFACTS = REPOSITORY_ROOT / "models/sensorium/retrained_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    from trajectory_model_eval.unified_analysis import _clean
    Path(path).write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_files(root, files):
    for name, expected in files.items():
        path = Path(root) / name
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input is missing: {path}")
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen input checksum differs: {path}")


def verify_cache(path, identity, shape):
    """Use the existing serialized cache contract, without decoding responses."""
    record = read(path.with_suffix(".json"))
    if (record["identity"] != identity or record["identity_sha256"] != canonical_digest(identity)
            or record["shape"] != list(shape)):
        raise ValueError(f"Frozen cache identity or shape differs: {path}")
    verify_files(path.parent, {path.name: record["payload_sha256"]})


def validate_destination(output, artifact_root, evidence_root):
    output = Path(output).resolve()
    protected = [Path(artifact_root).resolve(), Path(evidence_root).resolve(), REPOSITORY_ROOT / "results",
                 REPOSITORY_ROOT / "models"]
    protected += [PROJECT_ROOT / name for name in ("src", "configs", "records", "tests", "environments")]
    for source in protected:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"Replay output overlaps protected inputs: {output}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Replay requires a new or empty output directory: {output}")
    return output


def encoder_args(root, public, work=None):
    return SimpleNamespace(run_root=root, output_dir=root / "diagnostics/encoder",
        public_dir=public, config=PROJECT_ROOT / "configs/supplementary_encoder_v1.json", work_dir=work)


def preflight(artifact_root, evidence_root=EVIDENCE):
    """Read and verify every retained input before creating any output."""
    from trajectory_model_eval.retrained_predictions import bundle_identity, verify_test_lock
    from trajectory_model_eval.supplementary_encoder import verify_completed_scores
    from trajectory_model_eval.evidence_layout import evidence_file

    root, evidence = Path(artifact_root).resolve(), Path(evidence_root).resolve()
    config, manifest, prep = [read(root / p) for p in
        ("analysis_protocol.json", "data/split_manifest.json", "data/preprocessing.json")]
    if canonical_digest(config) != canonical_digest(read(PROJECT_ROOT / "configs/retrained_v1.json")):
        raise ValueError("Replay accepts the recorded primary protocol only")
    lock = verify_test_lock(config, manifest, prep, root / "encoders", root / "frozen", root / "gpfa")
    reversal = read(PROJECT_ROOT / "configs/supplementary_reversal_v1.json")
    if lock["freeze_sha256"] != reversal["source_freeze_sha256"]:
        raise ValueError("The input root is not the declared completed primary run")
    if read(root / "evaluation/summary.json")["status"] != "held_out_evaluation_after_retraining_complete":
        raise ValueError("Replay requires the completed source evaluation")
    pilot_record = read(root / "evaluation/pilot/generated_manifest.json")
    if pilot_record["cache_identity"]["freeze_sha256"] != lock["freeze_sha256"]:
        raise ValueError("The original pilot results belong to another freeze")
    verify_files(root / "evaluation/pilot", {"metrics.csv": pilot_record["files"]["metrics.csv"]})
    checksums = {}
    for session in manifest["sessions"]:
        neurons = manifest["session_neurons"][session]["all_indices"]
        expected = bundle_identity(config, manifest, prep, lock["checkpoint_identities"], session,
            ("test",), neurons, frozen_analysis=lock)
        path = root / "test" / session / "aligned_predictions.npz"
        verify_cache(path, expected, (len(expected["dataset_indices"]), 250, len(neurons)))
        checksums[path.relative_to(root).as_posix()] = read(path.with_suffix(".json"))["payload_sha256"]
        checksums[path.with_suffix(".json").relative_to(root).as_posix()] = file_sha256(path.with_suffix(".json"))
    pilot = manifest["session"]
    expected = bundle_identity(config, manifest, prep, lock["checkpoint_identities"], pilot,
        ("train", "validation"), manifest["neurons"]["indices"])
    path = root / "development" / pilot / "aligned_predictions.npz"
    verify_cache(path, expected, (len(expected["dataset_indices"]), 250, 512))
    checksums[path.relative_to(root).as_posix()] = read(path.with_suffix(".json"))["payload_sha256"]
    checksums[path.with_suffix(".json").relative_to(root).as_posix()] = file_sha256(path.with_suffix(".json"))
    source = read(root / "diagnostics/gpfa/source_identity.json")
    verify_files(root, source["sources"])
    checksums.update(source["sources"])
    checksums["evaluation/pilot/metrics.csv"] = pilot_record["files"]["metrics.csv"]
    gpfa_config = read(PROJECT_ROOT / "configs/supplementary_gpfa_v1.json")
    if (source["supplementary_config_sha256"] != canonical_digest(gpfa_config)
            or lock["gpfa_identity"]["gpfa_sha256"] != gpfa_config["main_gpfa_sha256"]):
        raise ValueError("Retained GPFA diagnostic sources differ")
    sampling = read(evidence / "gpfa/sampling_identity.json")
    verify_files(root, {"diagnostics/gpfa/sampling_plan.json": sampling["sampling_plan_sha256"],
        "evaluation/pilot/reliability_split_identities.csv": sampling["main_split_identity_sha256"]})
    checksums["diagnostics/gpfa/sampling_plan.json"] = sampling["sampling_plan_sha256"]
    # This verifies completed caches and the two numerical correction snapshots;
    # it cannot fall back to prediction, fitting, or strength selection.
    with redirect_stdout(io.StringIO()):
        if not verify_completed_scores(encoder_args(root, evidence / "encoder")):
            raise ValueError("Corrected, completed encoder-control evidence is required")
    encoder_manifest = read(evidence / "encoder/manifest.json")
    checksums.update(encoder_manifest["primary_sources"])
    for name, record in encoder_manifest["prediction_input_records"].items():
        checksums["diagnostics/encoder/" + name] = record["metadata_sha256"]
        checksums["diagnostics/encoder/" + str(Path(name).with_suffix(".npz")).replace("\\", "/")] = record["payload_sha256"]
    checksums.update({"diagnostics/encoder/" + name: value
                      for name, value in encoder_manifest["supplementary_source_records"].items()})
    endpoint = encoder_manifest.get("zero_spatial_endpoint_correction", {})
    if (endpoint.get("version") != 1 or endpoint.get("response_status") != "numerically_degenerate"
            or endpoint.get("matching_status") != "not_evaluable"):
        raise ValueError("The verified zero-spatial endpoint annotation is required")
    for item in endpoint["cached_endpoint_evidence"]:
        record = read(root / "diagnostics/encoder" / item["role"] / "spatial_0.00.json")
        if item["payload_sha256"] != record["payload_sha256"]:
            raise ValueError("The zero-spatial endpoint evidence targets a different cache")
    reverse_manifest = read(evidence / "reversal/manifest.json")
    if reverse_manifest["config_sha256"] != canonical_digest(reversal):
        raise ValueError("Reversal settings differ from their completed source")
    verify_files(root, reverse_manifest["source_artifacts"])
    verify_files(root, {"diagnostics/reversal_predictions.npz": reverse_manifest["reversal_cache_sha256"]})
    checksums.update(reverse_manifest["source_artifacts"])
    checksums["diagnostics/reversal_predictions.npz"] = reverse_manifest["reversal_cache_sha256"]
    for name, digest in reverse_manifest["files"].items():
        path = evidence_file(evidence / "reversal", name)
        verify_files(path.parent, {path.name: digest})
    historical = {}
    for name in ("training_summary.json", "official_evaluation.json", "architecture_audit.json"):
        path = PROJECT_ROOT / "records/baselines/dynamic" / name
        historical[path.relative_to(REPOSITORY_ROOT).as_posix()] = file_sha256(path)
    for name in ("parameter_counts.json", "static_frame_independence.json", "temporal_support.json", "weight_changes.json"):
        path = root / "diagnostics/encoder" / name
        expected_hash = encoder_manifest["files"][name]
        verify_files(path.parent, {name: expected_hash})
        checksums[path.relative_to(root).as_posix()] = expected_hash
    return {"root": root, "evidence": evidence, "config": config, "manifest": manifest,
        "preprocessing": prep, "lock": lock, "cache_checksums": checksums,
        "historical_evidence": historical, "status": "verified_frozen_inputs",
        "scope": "Cached replay; provider data and original training events are not recomputed"}


def replay_primary(plan, output):
    import pandas as pd
    from trajectory_model_eval.retrained_predictions import load_bundle, select_neurons, combine_pilot
    from trajectory_model_eval.retrained_analysis import score_session_response, summarize_session_responses, evaluate_pilot
    root, config, manifest = plan["root"], plan["config"], plan["manifest"]
    pooled, conditions, pilot_test = [], [], None
    for session in manifest["sessions"]:
        bundle = load_bundle(root / "test" / session / "aligned_predictions.npz")
        one, two = score_session_response(bundle, config)
        pooled.append(one); conditions.append(two)
        if session == manifest["session"]:
            pilot_test = select_neurons(bundle, manifest["neurons"]["indices"])
        del bundle
        gc.collect()
    response = summarize_session_responses(pd.concat(pooled), pd.concat(conditions), config, output / "response")
    development = load_bundle(root / "development" / manifest["session"] / "aligned_predictions.npz")
    combined = combine_pilot(development, pilot_test, manifest)
    pilot = evaluate_pilot(combined, manifest, config, root / "gpfa", root / "frozen", output / "pilot")
    write(output / "summary.json", {"status": "frozen_cache_replay_complete", "response": response, "pilot": pilot})


def replay_measurement(plan, output):
    import pandas as pd
    from trajectory_model_eval.supplementary_common import load_context
    from trajectory_model_eval.supplementary_gpfa import (verify_plan, _reliability, _reconstruction,
                                                         _summarize_reliability, plot_repeat_averaging)
    context = load_context(plan["root"])
    sampling = read(plan["root"] / "diagnostics/gpfa/sampling_plan.json")
    original = pd.read_csv(plan["root"] / "evaluation/pilot/reliability_split_identities.csv")
    verify_plan(sampling, context.test, original)
    a, model = context.gpfa.arrays, context.gpfa.model
    main = output / "main"
    main.mkdir(parents=True)
    _reconstruction(context.test, model, a["precision"], a["observations"], a["query_times"], "main", main)
    tables = []
    for k in (1, 2):
        target = output / f"k{k}"
        target.mkdir()
        _reliability(context.test, model, a["precision"], a["observations"], a["query_times"], sampling, k, target)
        table = pd.read_csv(target / "reliability_summary.csv"); table["repeat_level"] = str(k)
        tables.append(table)
    pilot = output.parent / "evaluation/pilot"
    table = _summarize_reliability(pd.read_csv(pilot / "reliability_observed.csv"),
                                   pd.read_csv(pilot / "reliability_nulls.csv"))
    table.to_csv(main / "reliability_summary.csv", index=False)
    table["repeat_level"] = "balanced 4/5"; tables.append(table)
    pd.concat(tables).to_csv(output / "repeat_averaging_summary.csv", index=False)
    plot_repeat_averaging(output)
    context.assert_unchanged()


def replay_reversal(plan, output):
    from trajectory_model_eval.supplementary_reversal import run
    run(plan["root"], output, PROJECT_ROOT / "configs/supplementary_reversal_v1.json",
        cache_output_path=output / "reversal_predictions.npz")


def replay_encoder(plan, output):
    from trajectory_model_eval.supplementary_encoder import (
        run_scores, refresh_response_scores, augment_source_manifest, mark_zero_spatial_endpoint)
    args = encoder_args(plan["root"], output, output.parent / "encoder_scoring_work")
    run_scores(args)
    refresh_response_scores(args)
    augment_source_manifest(args.output_dir, args.public_dir)
    mark_zero_spatial_endpoint(args)


def execute(artifact_root, output_dir=None, *, evidence_root=EVIDENCE, run=False):
    output = validate_destination(output_dir, artifact_root, evidence_root) if output_dir is not None else None
    if run and output is None:
        raise ValueError("--run requires an explicit --output-dir")
    plan = preflight(artifact_root, evidence_root)
    result = {"status": plan["status"], "scope": plan["scope"], "freeze_sha256": plan["lock"]["freeze_sha256"],
        "mode": "run" if run else "check", "historical_evidence": plan["historical_evidence"],
        "cache_checksums": plan["cache_checksums"], "gpfa_sensitivity": "excluded"}
    if not run:
        return result
    output.mkdir(parents=True, exist_ok=True)
    write(output / "replay_inputs.json", result)
    stages = (("evaluation", replay_primary), ("measurement", replay_measurement),
              ("temporal_metric_tests", replay_reversal), ("temporal_weight_controls", replay_encoder))
    for name, function in stages:
        print(json.dumps({"stage": name, "status": "running"}), flush=True)
        function(plan, output / name)
    # Full input verification repeats after scoring; source changes invalidate completion.
    checked = preflight(artifact_root, evidence_root)
    if checked["cache_checksums"] != plan["cache_checksums"]:
        raise AssertionError("Frozen source caches changed during replay")
    result.update(status="retained_cached_scoring_complete", historical_checks="verified_existing_records_only",
        recomputed_goals=[2, 3, 4, 5, 6], files={str(p.relative_to(output)).replace("\\", "/"): file_sha256(p)
            for p in output.rglob("*") if p.is_file()})
    write(output / "replay_manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify required frozen inputs without writing (default)")
    mode.add_argument("--run", action="store_true", help="Recompute retained scores from existing frozen caches")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACTS,
                        help="Read-only frozen bundle (default: models/sensorium/retrained_v1; no external fallback)")
    parser.add_argument("--output-dir", type=Path, help="New or empty destination, separate from all inputs")
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE, help="Original completed diagnostic evidence")
    args = parser.parse_args(argv)
    try:
        result = execute(args.artifact_root, args.output_dir, evidence_root=args.evidence_root, run=args.run)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as error:
        parser.exit(2, f"Frozen replay unavailable: {error}\nNo fitting or prediction fallback was attempted.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
