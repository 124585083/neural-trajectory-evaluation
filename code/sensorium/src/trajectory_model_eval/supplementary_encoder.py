"""Frozen-encoder implementation checks and bounded weight-damage diagnostics.

The prediction stage uses the encoding environment. The score stage uses the
analysis environment and the same completed primary GPFA. No training occurs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from .unified_contract import canonical_digest, file_sha256
from .evidence_layout import evidence_file


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def score_work_dir(args):
    """Separate new scoring history from immutable prediction-cache inputs."""
    return Path(getattr(args, "work_dir", None) or args.output_dir)


def response_columns(neural, prediction):
    """Center in float64 and keep exactly constant response columns undefined.

    The main analysis helper remains unchanged. In the zero spatial control,
    float32 accumulation of a constant mean otherwise creates a spurious tiny
    residual and can incorrectly make a constant prediction's r finite.
    """
    x = np.asarray(neural).reshape(-1, neural.shape[-1]).astype(np.float64)
    y = np.asarray(prediction).reshape(-1, prediction.shape[-1]).astype(np.float64)
    variable = (np.ptp(x, axis=0) > 0) & (np.ptp(y, axis=0) > 0)
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    return np.divide(np.sum(x * y, axis=0), denominator,
        out=np.full(x.shape[1], np.nan), where=variable & (denominator > 1e-12))


def mean_response_r(neural, prediction):
    """Pooled trial/time mean of jointly defined neuron correlations."""
    values = response_columns(neural, prediction)
    valid = np.isfinite(values)
    return (float(values[valid].mean()) if valid.any() else None), int(valid.sum())


def response_endpoint_status(neural, prediction, intervention, diagnostic=None, weight_changes=None):
    """Separate a verified zero-weight numerical floor from small real signals.

    This is an endpoint-specific interpretation rule, not an amplitude cutoff.
    The recorded diagnostic and zeroed first spatial convolution must both match
    the cached output before the known spatial endpoint can be marked invalid.
    Other predictions keep the existing float64 correlation calculation.
    """
    raw_r, raw_count = mean_response_r(neural, prediction)
    if intervention == "spatial_0.00":
        if diagnostic is None or weight_changes is None:
            raise ValueError("The zero spatial endpoint requires its recorded numerical and weight evidence")
        flat = np.asarray(prediction).reshape(-1, prediction.shape[-1])
        ranges = np.ptp(flat.astype(np.float64), axis=0)
        observed = {
            "exact_constant_prediction_neurons": int((ranges == 0).sum()),
            "defined_response_neurons": raw_count, "total_neurons": len(ranges),
            "prediction_range_min": float(ranges.min()),
            "prediction_range_median": float(np.median(ranges)),
            "prediction_range_max": float(ranges.max()),
            "maximum_absolute_prediction": float(np.abs(flat).max()),
        }
        if flat.dtype != np.float32 or any(observed[key] != diagnostic[key] for key in observed):
            raise ValueError("The zero spatial cache differs from the recorded floating-point-floor diagnostic")
        if (len(weight_changes) != 1
                or weight_changes[0]["tensor"] != "core.features.layer0.conv_spatial.weight"
                or weight_changes[0]["changed_l2"] != 0.0):
            raise ValueError("The endpoint is not the verified zero first-spatial-convolution control")
        return {"response_r": None, "response_status": "numerically_degenerate",
                "raw_response_r": raw_r, "raw_valid_neurons": raw_count, "valid_neurons": 0}
    return {"response_r": raw_r,
            "response_status": "defined" if raw_r is not None else "undefined_constant",
            "raw_response_r": raw_r, "raw_valid_neurons": raw_count, "valid_neurons": raw_count}


def response_pair(neural, prediction_a, prediction_b):
    a, b = response_columns(neural, prediction_a), response_columns(neural, prediction_b)
    valid = np.isfinite(a) & np.isfinite(b)
    return ((float(a[valid].mean()), float(b[valid].mean()), int(valid.sum())) if valid.any()
            else (np.nan, np.nan, 0))


def _supplement_conventional(neural, predictions, conditions, stride):
    from .unified_analysis import _conventional
    scores, support = _conventional(neural, predictions, conditions, stride)
    correlations = {name: response_columns(neural, value) for name, value in predictions.items()}
    valid = np.isfinite(np.stack(list(correlations.values()))).all(axis=0)
    for name, values in correlations.items():
        scores[name]["response_neuron_mean_r"] = float(values[valid].mean()) if valid.any() else np.nan
    support["response_r_shared_neurons"] = int(valid.sum())
    return scores, support


def _supplement_scores(neural, predictions, conditions, left, rights, dt, stride):
    from .unified_analysis import _trajectory_scores
    trajectory, support = _trajectory_scores(left, rights, dt)
    conventional, other = _supplement_conventional(neural, predictions, conditions, stride)
    return {name: {**trajectory[name], **conventional[name]} for name in predictions}, {**support, **other}


def select_spatial_matches(temporal_rows, spatial_rows, tolerance):
    """Select only from supplied validation rows; retain failed finite matches."""
    rows = list(temporal_rows) + list(spatial_rows)
    if any(row.get("role") != "validation" for row in rows):
        raise ValueError("Spatial strengths can only be selected using validation")
    candidates = [r for r in spatial_rows if r["response_r"] is not None and np.isfinite(r["response_r"])]
    result = []
    for target in temporal_rows:
        if target["response_r"] is None or not candidates:
            result.append({"temporal_retention": target["retention"], "spatial_retention": None,
                           "status": "undefined_validation_correlation"})
            continue
        chosen = min(candidates, key=lambda r: (abs(r["response_r"] - target["response_r"]), -r["retention"]))
        gap = chosen["response_r"] - target["response_r"]
        result.append({"temporal_retention": target["retention"], "spatial_retention": chosen["retention"],
            "temporal_validation_r": target["response_r"], "spatial_validation_r": chosen["response_r"],
            "spatial_minus_temporal_validation_r": gap,
            "validation_match": abs(gap) <= tolerance,
            "status": "matched" if abs(gap) <= tolerance else "outside_validation_tolerance"})
    return result


def intervention_state(state, kind, retention, spatial_names):
    """Return a new state; no source tensor or unspecified parameter changes."""
    import torch
    if kind == "temporal":
        from .extended_predictions import ablate_temporal_state
        altered, names = ablate_temporal_state(state, retention)
    elif kind == "spatial":
        if not 0 <= retention <= 1:
            raise ValueError("Retention must be in [0, 1]")
        altered, names = copy.deepcopy(state), list(spatial_names)
        for name in names:
            value = altered[name]
            if value.ndim != 5 or value.shape[2] != 1 or "conv_spatial" not in name:
                raise ValueError("The fixed control must be a spatial convolution weight")
            altered[name] = value * float(retention)
    else:
        raise ValueError(kind)
    records = []
    for name in names:
        before, after = state[name], altered[name]
        delta = after - before
        eligible = before.numel()
        if kind == "temporal":
            eligible -= before[:, :, before.shape[2] // 2].numel()
        records.append({"tensor": name, "shape": list(before.shape), "eligible_parameters": eligible,
            "actually_changed_parameters": int(torch.count_nonzero(delta).item()),
            "original_l2": float(torch.linalg.vector_norm(before)),
            "changed_l2": float(torch.linalg.vector_norm(after)),
            "delta_l2": float(torch.linalg.vector_norm(delta))})
    for name in state:
        if name not in names and not torch.equal(state[name], altered[name]):
            raise AssertionError("An unspecified parameter or buffer was changed")
    return altered, records


def temporal_support(retention, kernel_sizes):
    """Structural dependency offsets under the original valid-convolution crop."""
    width = sum(k - 1 for k in kernel_sizes) + 1
    center = sum(k // 2 for k in kernel_sizes)
    visual = [-center] if retention == 0 else list(range(-(width - 1), 1))
    return {"visual_and_appended_behavior_frame_offsets": visual,
        "pupil_shifter_frame_offsets": [0], "full_kernel_receptive_frames": width,
        "retained_visual_frames": len(visual), "time_alignment": "reported output frame is valid-convolution window end",
        "is_static_model": False}


def _cache_path(out, role, kind, retention):
    return out / role / f"{kind}_{retention:.2f}.npz"


def _load_prediction(path, identity):
    record = read_json(path.with_suffix(".json"))
    if record["identity"] != identity or record["payload_sha256"] != file_sha256(path):
        raise ValueError("Supplementary prediction cache identity differs")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}, record


def _save_prediction(path, arrays, identity, weight_records):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    record = {"identity": identity, "payload_sha256": file_sha256(path),
        "weight_changes": weight_records, "parameters_and_batchnorm_unchanged_during_inference": True}
    write_json(path.with_suffix(".json"), record)
    return record


def _negative_control(audit, tolerance):
    import torch

    class SequenceMeanCoupled(torch.nn.Module):
        # Same intentionally coupled model as test_static_frame_independence.py.
        temporal_reduction = 18

        def forward(self, inputs, **kwargs):
            values = inputs.mean(dim=(1, 3, 4)).unsqueeze(-1)
            return (values + values.mean(dim=1, keepdim=True))[:, 18:]

    values = torch.arange(27, dtype=torch.float32)[None, None, :, None, None].expand(1, 3, 27, 2, 2).clone()
    try:
        audit(SequenceMeanCoupled(), values, data_key="synthetic",
              pupil_center=torch.zeros(1, 2, 27), behavior=torch.zeros(1, 2, 27), tolerance=tolerance)
    except AssertionError as error:
        if "frame independence failed" not in str(error):
            raise
        return {"status": "coupling_detected", "observed_failure": str(error)}
    raise AssertionError("The frame-independence check missed intentional frame coupling")


def verify_completed_predictions(args):
    """Reuse completed, byte-verified outputs, with their original source record."""
    out, root = Path(args.output_dir), Path(args.run_root)
    if not (out / "prediction_status.json").exists():
        return False
    supplemental, identity = read_json(args.config), read_json(out / "prediction_identity.json")
    if read_json(out / "fixed_config.json") != supplemental:
        raise ValueError("Completed diagnostic uses a different supplementary configuration")
    if read_json(out / "prediction_status.json")["status"] != "complete":
        return False
    snapshot = out / "source_snapshots/prediction_implementation.py"
    if file_sha256(snapshot) != identity["prediction_implementation_sha256"]:
        raise ValueError("Completed predictions lack their verified implementation snapshot")
    for name, digest in identity["primary_sources"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("A completed diagnostic primary source was changed")
    selection = read_json(out / "spatial_selection.json")
    if canonical_digest({k: v for k, v in selection.items() if k != "selection_sha256"}) != selection["selection_sha256"]:
        raise ValueError("Completed spatial selection freeze was changed")
    temporal = supplemental["temporal"]["retentions"]
    selected = sorted({m["spatial_retention"] for m in selection["matches"] if m["spatial_retention"] is not None})
    for role, choices in (("validation", {"temporal": temporal, "spatial": supplemental["spatial"]["retentions"]}),
                          ("test", {"temporal": temporal, "spatial": selected})):
        for kind, retentions in choices.items():
            for retention in retentions:
                path = _cache_path(out, role, kind, retention)
                record = read_json(path.with_suffix(".json"))
                if any(record["identity"].get(k) != v for k, v in identity.items()):
                    raise ValueError("Completed prediction metadata differs")
                expected_selection = selection["selection_sha256"] if role == "test" else None
                if (record["identity"]["role"] != role or record["identity"]["kind"] != kind
                        or record["identity"]["retention"] != retention
                        or record["identity"]["spatial_selection_sha256"] != expected_selection
                        or record["payload_sha256"] != file_sha256(path)):
                    raise ValueError("Completed prediction payload or support was changed")
    print(json.dumps({"encoder_supplementary_predictions": "reused_verified_completed_outputs",
                      "prediction_implementation_sha256": identity["prediction_implementation_sha256"]}), flush=True)
    return True


def run_predictions(args):
    import torch
    from .retrained_data import create_dataloaders, create_role_loader
    from .retrained_training import configure_sources, load_frozen_encoder, _cpu_state, state_digest
    from .retrained_predictions import load_bundle, verify_test_lock, verify_source_files
    from .unified_predictions import predict_loader

    configure_sources(args.sensorium_source, args.neuralpredictors_source)
    from trajectory_eval.static_dynamic import audit_frame_independence, parameter_counts
    root, out = Path(args.run_root), Path(args.output_dir)
    config, supplemental = read_json(root / "analysis_protocol.json"), read_json(args.config)
    manifest, prep = read_json(root / "data/split_manifest.json"), read_json(root / "data/preprocessing.json")
    prep_path = root / "data/preprocessing.json"
    lock = verify_test_lock(config, manifest, prep, root / "encoders", root / "frozen", root / "gpfa")
    source_paths = ["analysis_protocol.json", "data/split_manifest.json", "data/preprocessing.json",
        "encoders/static/best.pt", "encoders/dynamic/best.pt", "gpfa/frozen_gpfa.pkl", "gpfa/frozen_gpfa_arrays.npz",
        "frozen/frozen_analysis.json", "frozen/frozen_control.npz", "evaluation/summary.json"]
    sources = {name: file_sha256(root / name) for name in source_paths}
    identity = {"supplementary_config_sha256": canonical_digest(supplemental),
        "primary_freeze_sha256": lock["freeze_sha256"], "primary_sources": sources,
        "prediction_implementation_sha256": file_sha256(__file__), "primary_response_matching_sigma": lock["response_matching"]["sigma"]}
    out.mkdir(parents=True, exist_ok=True)
    if (out / "fixed_config.json").exists():
        if read_json(out / "fixed_config.json") != supplemental:
            raise ValueError("Supplementary settings changed after starting the diagnostic")
        if read_json(out / "prediction_identity.json") != identity:
            raise ValueError("Supplementary source or implementation changed after prediction began")
    else:
        write_json(out / "fixed_config.json", supplemental)
        write_json(out / "prediction_identity.json", identity)
    snapshot = out / "source_snapshots/prediction_implementation.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    if file_sha256(snapshot) != identity["prediction_implementation_sha256"]:
        raise ValueError("Prediction source snapshot differs from the current run")
    pilot, neurons = manifest["session"], np.asarray(manifest["neurons"]["indices"])
    frames = supplemental["support"]["frames"]
    if frames != [config["data"]["frame_start"], config["data"]["frame_stop"]]:
        raise ValueError("Evaluation frame support changed")
    loaders = create_dataloaders(args.data_root, manifest, prep_path)["train"]
    static, _ = load_frozen_encoder("static", loaders, root / "encoders/static", config=config,
        manifest=manifest, preprocessing=prep, device=args.device)
    dynamic, _ = load_frozen_encoder("dynamic", loaders, root / "encoders/dynamic", config=config,
        manifest=manifest, preprocessing=prep, device=args.device)
    counts = {"static": parameter_counts(static), "dynamic": parameter_counts(dynamic)}
    ratio = abs(counts["dynamic"]["total"] - counts["static"]["total"]) / counts["static"]["total"]
    if ratio > supplemental["parameter_counts"]["maximum_relative_total_difference"]:
        raise AssertionError("Actual model totals exceed the fixed comparison tolerance")
    for kind, model in (("static", static), ("dynamic", dynamic)):
        if sum(counts[kind][part] for part in ("core", "readout", "shifter")) != counts[kind]["total"]:
            raise AssertionError("Shared parameters require explicit cross-component accounting")
    write_json(out / "parameter_counts.json", {"counts": counts, "relative_total_difference": ratio,
        "actual_instances": True, "shared_parameters_counted_once": True, "dynamic_channels": [16, 32, 64]})
    verify_source_files(manifest, args.data_root, roles=("validation",), sessions=[pilot])
    validation = create_role_loader(args.data_root, manifest, prep_path, pilot, "validation", batch_size=1)
    if len(validation.dataset) != supplemental["support"]["validation_trials"]:
        raise AssertionError("Validation support changed")
    batch = next(iter(validation))
    values = batch if isinstance(batch, dict) else batch._asdict()
    static_before = state_digest(_cpu_state(static))
    independent = audit_frame_independence(static, values["videos"].to(args.device), data_key=pilot,
        pupil_center=values["pupil_center"].to(args.device), behavior=values["behavior"].to(args.device),
        positions=tuple(supplemental["static_independence"]["input_frames"]),
        tolerance=supplemental["static_independence"]["absolute_tolerance"])
    if static_before != state_digest(_cpu_state(static)):
        raise AssertionError("Static independence inference modified state")
    independent.update(dataset_index=validation.dataset.rows[0]["dataset_index"], role="validation",
        checkpoint_sha256=lock["checkpoint_identities"]["static"]["sha256"],
        negative_control=_negative_control(audit_frame_independence, supplemental["static_independence"]["absolute_tolerance"]),
        state_unchanged=True, input_shape=list(values["videos"].shape))
    write_json(out / "static_frame_independence.json", independent)
    del static, values, batch, loaders
    torch.cuda.empty_cache()
    original = _cpu_state(dynamic)
    original_digest = state_digest(original)
    intervention = copy.deepcopy(dynamic)
    temporal = supplemental["temporal"]["retentions"]
    spatial = supplemental["spatial"]["retentions"]
    weights = {}

    def predict(role, loader, kind, retention, freeze_sha=None):
        rows = loader.dataset.rows
        record_identity = {**identity, "role": role, "kind": kind, "retention": retention,
            "session": pilot, "dataset_indices": [r["dataset_index"] for r in rows],
            "neuron_indices": neurons.tolist(), "frame_indices": list(range(*frames)),
            "source_sha256": canonical_digest([r["source_sha256"] for r in rows]), "spatial_selection_sha256": freeze_sha}
        path = _cache_path(out, role, kind, retention)
        if path.exists():
            arrays, record = _load_prediction(path, record_identity)
        else:
            altered, changes = intervention_state(original, kind, retention, supplemental["spatial"]["weight_names"])
            intervention.load_state_dict(altered, strict=True)
            start = time.monotonic()
            prediction, neural = predict_loader(intervention, loader, pilot, neurons, *frames, args.device)
            arrays = {"prediction": prediction, "neural": neural,
                "dataset_indices": np.asarray([r["dataset_index"] for r in rows]),
                "conditions": np.asarray([r["movie_id"] for r in rows])}
            record = _save_prediction(path, arrays, record_identity, changes)
            print(json.dumps({"stage": role, "kind": kind, "retention": retention,
                              "seconds": time.monotonic() - start}), flush=True)
        weights[f"{kind}_{retention:.2f}"] = record["weight_changes"]
        response, n_valid = mean_response_r(arrays["neural"], arrays["prediction"])
        return {"role": role, "kind": kind, "retention": retention, "response_r": response,
                "valid_neurons": n_valid, "total_neurons": len(neurons)}, arrays

    validation_rows = []
    baseline = None
    for kind, values in (("temporal", temporal), ("spatial", spatial)):
        for retention in values:
            row, arrays = predict("validation", validation, kind, retention)
            validation_rows.append(row)
            if retention == 1:
                if baseline is None:
                    baseline = arrays["prediction"].copy()
                elif not np.array_equal(baseline, arrays["prediction"]):
                    raise AssertionError("Identity temporal/spatial interventions differ")
    development = load_bundle(root / "development" / pilot / "aligned_predictions.npz")
    indices = [list(development["dataset_indices"]).index(r["dataset_index"]) for r in validation.dataset.rows]
    tol = supplemental["temporal"]["baseline_tolerance"]
    difference = float(np.max(np.abs(baseline - development["dynamic"][indices])))
    if not np.allclose(baseline, development["dynamic"][indices], **tol):
        raise AssertionError("Retention one failed to restore the completed Dynamic validation baseline")
    del development, baseline
    selections = select_spatial_matches([r for r in validation_rows if r["kind"] == "temporal"],
        [r for r in validation_rows if r["kind"] == "spatial"], supplemental["spatial"]["absolute_tolerance"])
    freeze = {"status": "spatial_strengths_frozen_before_test", "prediction_identity": identity,
        "validation_rows": validation_rows, "matches": selections, "selection_role": "validation",
        "baseline_validation_max_absolute_error": difference,
        "test_results_used_for_selection": False}
    freeze_sha = canonical_digest(freeze)
    freeze["selection_sha256"] = freeze_sha
    if (out / "spatial_selection.json").exists() and read_json(out / "spatial_selection.json") != freeze:
        raise ValueError("Previously frozen spatial selection differs")
    write_json(out / "spatial_selection.json", freeze)
    print(json.dumps({"status": freeze["status"], "matches": selections}), flush=True)
    # A test loader is not constructed until the validation selection is saved.
    verify_source_files(manifest, args.data_root, roles=("test",), sessions=[pilot])
    test = create_role_loader(args.data_root, manifest, prep_path, pilot, "test", batch_size=1)
    if len(test.dataset) != supplemental["support"]["test_trials"]:
        raise AssertionError("Test support changed")
    test_rows = []
    selected = sorted({r["spatial_retention"] for r in selections if r["spatial_retention"] is not None}, reverse=True)
    for kind, values in (("temporal", temporal), ("spatial", selected)):
        for retention in values:
            row, _ = predict("test", test, kind, retention, freeze_sha)
            test_rows.append(row)
    if state_digest(_cpu_state(dynamic)) != original_digest:
        raise AssertionError("The original in-memory Dynamic model was modified")
    for name, digest in sources.items():
        if file_sha256(root / name) != digest:
            raise AssertionError(f"A primary artifact changed: {name}")
    write_json(out / "weight_changes.json", weights)
    write_json(out / "temporal_support.json", {str(r): temporal_support(r, supplemental["temporal"]["kernel_sizes"]) for r in temporal})
    write_json(out / "prediction_status.json", {"status": "complete", "test_rows": test_rows,
        "main_artifacts_unchanged": True, "main_response_matching_sigma_unchanged": True,
        "primary_sources": sources, "spatial_selection_sha256": freeze_sha})
    print(json.dumps({"encoder_supplementary_predictions": "complete"}), flush=True)


def _check_test_cache(out, role, kind, retention, context, identity, selection_sha):
    path = _cache_path(out, role, kind, retention)
    record = read_json(path.with_suffix(".json"))
    if record["identity"]["spatial_selection_sha256"] != selection_sha or record["identity"]["role"] != "test":
        raise AssertionError("Test predictions lack the prior spatial selection freeze")
    expected = {"kind": kind, "retention": retention,
        "session": context.manifest["session"],
        "neuron_indices": context.test["neuron_indices"].tolist(),
        "frame_indices": context.test["frame_indices"].tolist(),
        "dataset_indices": context.test["dataset_indices"].tolist()}
    if any(record["identity"].get(k) != v for k, v in expected.items()):
        raise ValueError("Intervention name or exact trial/neuron/frame support differs")
    if any(record["identity"].get(key) != value for key, value in identity.items()):
        raise ValueError("Prediction identity differs from frozen supplementary sources")
    arrays, _ = _load_prediction(path, record["identity"])
    test = context.test
    for key in ("dataset_indices", "conditions"):
        if not np.array_equal(arrays[key], test[key]):
            raise AssertionError(f"Intervention {key} differs from the primary test")
    if not np.array_equal(arrays["neural"], test["neural"]):
        raise AssertionError("Intervention neural targets differ from the primary test")
    return arrays["prediction"]


def run_scores(args):
    import pandas as pd
    from .supplementary_common import load_context, paired_trajectory_intervals, write_manifest
    from .traditional import condition_means

    out, public = Path(args.output_dir), Path(args.public_dir)
    public.mkdir(parents=True, exist_ok=True)
    supplemental = read_json(args.config)
    if supplemental != read_json(out / "fixed_config.json"):
        raise ValueError("Supplementary configuration differs from its fixed snapshot")
    status = read_json(out / "prediction_status.json")
    if status["status"] != "complete":
        raise ValueError("Prediction stage is incomplete")
    context = load_context(args.run_root)
    identity = read_json(out / "prediction_identity.json")
    for name, digest in identity["primary_sources"].items():
        if file_sha256(context.root / name) != digest:
            raise AssertionError("A primary artifact changed between prediction and scoring")
    selection = read_json(out / "spatial_selection.json")
    unhashed = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if canonical_digest(unhashed) != selection["selection_sha256"]:
        raise AssertionError("Spatial selection freeze was changed")
    test, frozen = context.test, context.frozen
    temporal = supplemental["temporal"]["retentions"]
    spatial = sorted({m["spatial_retention"] for m in selection["matches"] if m["spatial_retention"] is not None}, reverse=True)
    predictions = {}
    for kind, values in (("temporal", temporal), ("spatial", spatial)):
        for retention in values:
            predictions[f"{kind}_{retention:.2f}"] = _check_test_cache(out, "test", kind, retention,
                context, identity, selection["selection_sha256"])
    tol = supplemental["temporal"]["baseline_tolerance"]
    baseline_error = float(np.max(np.abs(predictions["temporal_1.00"] - test["dynamic"])))
    if not np.allclose(predictions["temporal_1.00"], test["dynamic"], **tol):
        raise AssertionError("Retention one does not restore the completed Dynamic test predictions")
    # Condition means are transformed through the same fixed posterior; no fit.
    neural = test["neural"]
    conditions = test["conditions"]
    unique = np.unique(conditions)
    left = frozen.latent(condition_means(neural, conditions))
    rights = {name: frozen.latent(condition_means(value, conditions)) for name, value in predictions.items()}
    stride = supplemental["statistics"]["time_aware_rsa_stride"]
    dt = 1 / context.config["data"]["frame_rate_hz"]
    # A degenerate extra damage control must not redefine every curve's response
    # summary. Individual curves retain their own defined features; comparisons
    # below use the pair's shared support and report that support explicitly.
    scores, supports = {}, {}
    for name, prediction in predictions.items():
        one, one_support = _supplement_scores(neural, {name: prediction}, conditions,
            left, {name: rights[name]}, dt, stride)
        scores[name], supports[name] = one[name], one_support
    rows, condition_rows = [], []
    for name, values in scores.items():
        kind, retention = name.rsplit("_", 1)
        for metric, value in values.items():
            rows.append({"intervention": name, "kind": kind, "retention": float(retention), "metric": metric,
                "value": value, "estimand": "pooled_condition_trajectory" if metric in supplemental["statistics"]["primary"] + supplemental["statistics"]["secondary"] else "pooled_score",
                "n_conditions": len(unique), "n_trials": len(neural), **supports[name]})
    for i, condition in enumerate(unique):
        mask = conditions == condition
        for name, prediction in predictions.items():
            local, local_support = _supplement_scores(neural[mask], {name: prediction[mask]},
                conditions[mask], left[i:i+1], {name: rights[name][i:i+1]}, dt, stride)
            for metric, value in local[name].items():
                condition_rows.append({"condition": str(condition), "intervention": name, "metric": metric,
                    "value": value, "n_trials": int(mask.sum()), **local_support})
    pd.DataFrame(rows).to_csv(public / "intervention_scores.csv", index=False)
    condition_frame = pd.DataFrame(condition_rows)
    condition_frame.to_csv(public / "condition_scores.csv", index=False)
    pd.DataFrame(selection["validation_rows"]).to_csv(public / "validation_grid.csv", index=False)
    pairing, intervals, paired_condition_rows = [], [], []
    rng = np.random.default_rng(supplemental["statistics"]["bootstrap_seed"])
    draws = rng.integers(0, len(unique), size=(supplemental["statistics"]["bootstrap_samples"], len(unique)))
    comparisons = [("temporal_1.00", f"temporal_{r:.2f}", "temporal_minus_baseline", r) for r in temporal]
    for match in selection["matches"]:
        r, spatial_r = match["temporal_retention"], match["spatial_retention"]
        if spatial_r is None:
            pairing.append(match)
            continue
        temporal_name, spatial_name = f"temporal_{r:.2f}", f"spatial_{spatial_r:.2f}"
        pair_scores, pair_support = _supplement_conventional(neural, {temporal_name: predictions[temporal_name],
            spatial_name: predictions[spatial_name]}, conditions, stride)
        tr, sr = pair_scores[temporal_name]["response_neuron_mean_r"], pair_scores[spatial_name]["response_neuron_mean_r"]
        gap = sr - tr
        pairing.append({**match, "temporal_test_r": tr, "spatial_test_r": sr,
            "spatial_minus_temporal_test_r": gap, "test_match": bool(np.isfinite(gap) and abs(gap) <= supplemental["spatial"]["absolute_tolerance"]),
            "tolerance": supplemental["spatial"]["absolute_tolerance"], **pair_support})
        comparisons.append((temporal_name, spatial_name, "spatial_minus_temporal", r))
    for a, b, label, retention in comparisons:
        interval = paired_trajectory_intervals(left, rights[a], rights[b], dt=dt,
            seed=supplemental["statistics"]["bootstrap_seed"], n_boot=supplemental["statistics"]["bootstrap_samples"])
        interval["comparison"], interval["retention"] = label, retention
        interval["model_a"], interval["model_b"] = a, b
        intervals.extend(interval.to_dict("records"))
        conventional_pairs = []
        for condition in unique:
            mask = conditions == condition
            pair, pair_support = _supplement_conventional(neural[mask], {"a": predictions[a][mask],
                "b": predictions[b][mask]}, conditions[mask], stride)
            conventional_pairs.append(pair)
            for metric in supplemental["statistics"]["conventional"]:
                paired_condition_rows.append({"comparison": label, "retention": retention,
                    "model_a": a, "model_b": b, "condition": str(condition), "metric": metric,
                    "a": pair["a"][metric], "b": pair["b"][metric],
                    "b_minus_a": pair["b"][metric] - pair["a"][metric], **pair_support})
        for metric in supplemental["statistics"]["conventional"]:
            one = np.asarray([pair["a"][metric] for pair in conventional_pairs])
            two = np.asarray([pair["b"][metric] for pair in conventional_pairs])
            differences = two - one
            paired_valid = np.isfinite(one) & np.isfinite(two)
            # Same rule as the primary comparison: retain each whole-condition
            # draw, use its jointly defined condition scores, and preserve NA
            # when no pair is defined. Never substitute zero for an NA.
            sampled = differences[draws]
            counts = np.isfinite(sampled).sum(axis=1)
            boot = np.divide(np.nansum(sampled, axis=1), counts,
                             out=np.full(len(draws), np.nan), where=counts > 0)
            valid = np.isfinite(boot)
            lo, hi = np.quantile(boot[valid], [.025, .975]) if valid.any() else [np.nan, np.nan]
            intervals.append({"comparison": label, "retention": retention, "model_a": a, "model_b": b,
                "metric": metric, "a": np.mean(one[paired_valid]) if paired_valid.any() else np.nan,
                "b": np.mean(two[paired_valid]) if paired_valid.any() else np.nan,
                "b_minus_a": np.mean(differences[paired_valid]) if paired_valid.any() else np.nan,
                "ci_low": lo, "ci_high": hi, "n_conditions": len(unique), "valid_bootstraps": int(valid.sum()),
                "jointly_defined_conditions": int(paired_valid.sum()),
                "undefined_bootstraps": int((~valid).sum()), "resampling_unit": "whole_movie_condition",
                "estimand": "equal_weight_mean_condition_score"})
    pd.DataFrame(intervals).to_csv(public / "paired_intervals.csv", index=False)
    pd.DataFrame(paired_condition_rows).to_csv(public / "paired_condition_scores.csv", index=False)
    pd.DataFrame(pairing).to_csv(public / "spatial_response_matches.csv", index=False)
    for name in ("parameter_counts.json", "static_frame_independence.json", "temporal_support.json", "weight_changes.json"):
        write_json(public / name, read_json(out / name))
    _plot_scores(pd.DataFrame(rows), pd.DataFrame(pairing), public)
    context.assert_unchanged()
    write_manifest(public, {"status": "complete", "config_sha256": canonical_digest(supplemental),
        "primary_sources": identity["primary_sources"], "supplementary_selection_sha256": selection["selection_sha256"],
        "implementation_sha256": file_sha256(__file__), "baseline_test_max_absolute_error": baseline_error,
        "main_artifacts_unchanged": True, "main_response_matching_sigma": identity["primary_response_matching_sigma"],
        "test_conditions": unique.tolist(), "individual_curve_support": supports,
        "prediction_implementation_sha256": identity["prediction_implementation_sha256"],
        "prediction_source_snapshot": "diagnostics/encoder/source_snapshots/prediction_implementation.py",
        "scope": "Exploratory diagnostic; failed matching conditions are retained"})
    print(json.dumps({"encoder_supplementary_scores": "complete", "baseline_test_max_absolute_error": baseline_error,
        "spatial_matches": pairing}), flush=True)


def _plot_scores(scores, matches, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    for ax, metric, label in zip(axes, ["response_neuron_mean_r", "position_correlation", "velocity_direction_cosine"],
                                 ["Response correlation", "Latent position correlation", "Velocity-direction cosine"]):
        temporal = scores[(scores.kind == "temporal") & (scores.metric == metric)].sort_values("retention")
        ax.plot(temporal.retention, temporal.value, "o-", label="Temporal attenuation")
        control = []
        for row in matches.to_dict("records"):
            selected = scores[(scores.intervention == f"spatial_{row['spatial_retention']:.2f}") & (scores.metric == metric)]
            if len(selected):
                status = row.get("test_matching_status", "matched" if row.get("test_match", False) else "outside_tolerance")
                control.append((row["temporal_retention"], selected.iloc[0].value, status))
        if control:
            control.sort()
            ax.plot([r[0] for r in control], [r[1] for r in control], "s--", label="Validation-selected spatial control")
            for x, y, status in control:
                if status == "not_evaluable":
                    if metric == "response_neuron_mean_r":
                        ax.annotate("Spatial weights = 0: NA", xy=(x, .02),
                            xycoords=("data", "axes fraction"), fontsize=7,
                            color="darkred", rotation=90, va="bottom", ha="center")
                    elif np.isfinite(y):
                        ax.scatter([x], [y], marker="D", s=55, facecolors="none",
                                   edgecolors="darkred", zorder=6)
                elif status != "matched":
                    ax.scatter([x], [y], marker="x", s=70, color="black", zorder=5)
        ax.set(xlabel="Temporal off-center weight retention", ylabel=label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Frozen Dynamic diagnostics: black x = failed match; red diamond = correlation not evaluable", fontsize=9)
    fig.savefig(out / "weight_attenuation.png", dpi=180)
    plt.close(fig)


def augment_source_manifest(output_dir, public_dir):
    """Index actual supplementary inputs without rerunning numerical analyses."""
    out, public = Path(output_dir), Path(public_dir)
    identity = read_json(out / "prediction_identity.json")
    snapshot = out / "source_snapshots/prediction_implementation.py"
    if file_sha256(snapshot) != identity["prediction_implementation_sha256"]:
        raise ValueError("Prediction implementation snapshot does not match its source identity")
    manifest = read_json(public / "manifest.json")
    manifest["prediction_input_records"] = {str(path.relative_to(out)).replace("\\", "/"):
        {"metadata_sha256": file_sha256(path), "payload_sha256": read_json(path)["payload_sha256"]}
        for path in sorted(out.glob("*/*.json")) if path.parent.name in ("validation", "test")}
    manifest["supplementary_source_records"] = {name: file_sha256(out / name) for name in
        ("fixed_config.json", "prediction_identity.json", "spatial_selection.json", "prediction_status.json",
         "source_snapshots/prediction_implementation.py")}
    write_json(public / "manifest.json", manifest)


def refresh_response_scores(args):
    """Correct response-only summaries from immutable caches, without inference.

    Frozen strengths and original calibration values are retained. Existing
    trajectory and non-response conventional scores/intervals are reused.
    """
    import shutil
    import pandas as pd
    out, public, root = Path(args.output_dir), Path(args.public_dir), Path(args.run_root)
    supplemental = read_json(args.config)
    verify_completed_predictions(args)
    manifest = read_json(public / "manifest.json")
    if manifest.get("response_correlation_numerical_fix", {}).get("version") == 2:
        return
    history = score_work_dir(args) / "numerical_history/response_before_float64_fix"
    history.mkdir(parents=True, exist_ok=True)
    for path in public.iterdir():
        if path.is_file() and not (history / path.name).exists():
            shutil.copy2(path, history / path.name)
    selection_sha = file_sha256(out / "spatial_selection.json")
    selection = read_json(out / "spatial_selection.json")
    numeric_rows, corrected_grid, test = [], [], {}
    for role in ("validation", "test"):
        for path in sorted((out / role).glob("*.npz")):
            record = read_json(path.with_suffix(".json"))
            with np.load(path, allow_pickle=False) as archive:
                neural, prediction = archive["neural"], archive["prediction"]
                conditions = archive["conditions"]
            r, count = mean_response_r(neural, prediction)
            flat = prediction.reshape(-1, prediction.shape[-1]).astype(np.float64)
            ranges = np.ptp(flat, axis=0)
            numeric_rows.append({"role": role, "intervention": path.stem,
                "exact_constant_prediction_neurons": int((ranges == 0).sum()),
                "defined_response_neurons": count, "total_neurons": len(ranges),
                "prediction_range_min": float(ranges.min()), "prediction_range_median": float(np.median(ranges)),
                "prediction_range_max": float(ranges.max()),
                "maximum_absolute_prediction": float(np.abs(flat).max()),
                "interpretation": "Numerical output floor; residual correlations are not evidence of retained signal"
                    if path.stem == "spatial_0.00" else "Variable model prediction"})
            if role == "validation":
                original = next(row for row in selection["validation_rows"] if
                    row["kind"] == record["identity"]["kind"] and row["retention"] == record["identity"]["retention"])
                corrected_grid.append({**original, "original_response_r": original["response_r"],
                    "original_valid_neurons": original["valid_neurons"],
                    "response_r": r, "valid_neurons": count, "numerical_precision": "float64 centering; exact constants NA"})
            else:
                test[path.stem] = (neural, prediction, conditions)
    pd.DataFrame(numeric_rows).to_csv(public / "response_numeric_support.csv", index=False)
    pd.DataFrame(corrected_grid).to_csv(public / "validation_grid.csv", index=False)
    scores = pd.read_csv(public / "intervention_scores.csv")
    condition_scores = pd.read_csv(public / "condition_scores.csv", dtype={"condition": str})
    for name, (neural, prediction, conditions) in test.items():
        r, count = mean_response_r(neural, prediction)
        mask = scores.intervention == name
        scores.loc[mask & (scores.metric == "response_neuron_mean_r"), "value"] = r
        scores.loc[mask, "response_r_shared_neurons"] = count
        for condition in np.unique(conditions):
            selected = conditions == condition
            r, count = mean_response_r(neural[selected], prediction[selected])
            mask = (condition_scores.intervention == name) & (condition_scores.condition == str(condition))
            condition_scores.loc[mask & (condition_scores.metric == "response_neuron_mean_r"), "value"] = r
            condition_scores.loc[mask, "response_r_shared_neurons"] = count
    scores.to_csv(public / "intervention_scores.csv", index=False)
    condition_scores.to_csv(public / "condition_scores.csv", index=False)
    intervals = pd.read_csv(public / "paired_intervals.csv")
    paired_conditions = pd.read_csv(public / "paired_condition_scores.csv", dtype={"condition": str})
    for index, row in intervals[intervals.metric == "response_neuron_mean_r"].iterrows():
        neural, a, conditions = test[row.model_a]
        other_neural, b, other_conditions = test[row.model_b]
        if not np.array_equal(neural, other_neural) or not np.array_equal(conditions, other_conditions):
            raise AssertionError("Response correction lacks paired target support")
        unique = np.unique(conditions)
        one, two = [], []
        for condition in unique:
            selected = conditions == condition
            av, bv, count = response_pair(neural[selected], a[selected], b[selected])
            one.append(av)
            two.append(bv)
            mask = ((paired_conditions.model_a == row.model_a) & (paired_conditions.model_b == row.model_b)
                & (paired_conditions.condition == str(condition)))
            paired_conditions.loc[mask, "response_r_shared_neurons"] = count
            response_mask = mask & (paired_conditions.metric == "response_neuron_mean_r")
            paired_conditions.loc[response_mask, ["a", "b", "b_minus_a"]] = [av, bv, bv - av]
        one, two = np.asarray(one), np.asarray(two)
        valid = np.isfinite(one) & np.isfinite(two)
        draws = np.random.default_rng(supplemental["statistics"]["bootstrap_seed"]).integers(
            0, len(unique), size=(supplemental["statistics"]["bootstrap_samples"], len(unique)))
        sample = (two - one)[draws]
        counts = np.isfinite(sample).sum(axis=1)
        boot = np.divide(np.nansum(sample, axis=1), counts, out=np.full(len(draws), np.nan), where=counts > 0)
        defined = np.isfinite(boot)
        lo, hi = np.quantile(boot[defined], [.025, .975]) if defined.any() else [np.nan, np.nan]
        values = {"a": one[valid].mean() if valid.any() else np.nan,
            "b": two[valid].mean() if valid.any() else np.nan,
            "b_minus_a": (two - one)[valid].mean() if valid.any() else np.nan,
            "ci_low": lo, "ci_high": hi, "valid_bootstraps": int(defined.sum()),
            "undefined_bootstraps": int((~defined).sum()), "jointly_defined_conditions": int(valid.sum())}
        for key, value in values.items():
            intervals.loc[index, key] = value
    intervals.to_csv(public / "paired_intervals.csv", index=False)
    paired_conditions.to_csv(public / "paired_condition_scores.csv", index=False)
    matches = pd.read_csv(public / "spatial_response_matches.csv")
    fixed_nearest = []
    for index, match in matches.iterrows():
        tr, sr = float(match.temporal_retention), float(match.spatial_retention)
        temporal_name, spatial_name = f"temporal_{tr:.2f}", f"spatial_{sr:.2f}"
        neural, a, _ = test[temporal_name]
        _, b, _ = test[spatial_name]
        av, bv, count = response_pair(neural, a, b)
        for name in ("temporal_test_r", "spatial_test_r", "spatial_minus_temporal_test_r", "test_match"):
            matches.loc[index, "original_" + name] = match[name]
        matches.loc[index, ["temporal_test_r", "spatial_test_r", "spatial_minus_temporal_test_r"]] = [av, bv, bv - av]
        matches.loc[index, "test_match"] = bool(np.isfinite(bv - av) and abs(bv - av) <= supplemental["spatial"]["absolute_tolerance"])
        matches.loc[index, "response_r_shared_neurons"] = count
        vt = next(row for row in corrected_grid if row["kind"] == "temporal" and row["retention"] == tr)
        vs = next(row for row in corrected_grid if row["kind"] == "spatial" and row["retention"] == sr)
        nearest = min((row for row in corrected_grid if row["kind"] == "spatial" and row["response_r"] is not None),
            key=lambda row: (abs(row["response_r"] - vt["response_r"]), -row["retention"]))
        fixed_nearest.append(nearest["retention"] == sr)
        matches.loc[index, "corrected_temporal_validation_r"] = vt["response_r"]
        matches.loc[index, "corrected_spatial_validation_r"] = vs["response_r"]
        matches.loc[index, "corrected_validation_match"] = abs(vs["response_r"] - vt["response_r"]) <= supplemental["spatial"]["absolute_tolerance"]
        matches.loc[index, "fixed_choice_still_nearest_after_correction"] = nearest["retention"] == sr
        matches.loc[index, "strength_changed_after_correction"] = False
    matches.to_csv(public / "spatial_response_matches.csv", index=False)
    _plot_scores(scores, matches, public)
    if file_sha256(out / "spatial_selection.json") != selection_sha:
        raise AssertionError("The numerical correction modified frozen spatial selection")
    for name, digest in manifest["primary_sources"].items():
        if file_sha256(root / name) != digest:
            raise AssertionError("A primary artifact changed during numerical correction")
    snapshot = score_work_dir(args) / "source_snapshots/scoring_implementation_response_v2.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    manifest["response_correlation_numerical_fix"] = {"version": 2,
        "rule": "Center and accumulate in float64; exactly constant predictions remain NA; use shared finite neurons for pairs",
        "scope": "Supplementary response scores and their paired intervals only; no new predictions or GPFA inference",
        "original_calibration_preserved": True, "frozen_strengths_changed": False,
        "all_fixed_choices_remain_nearest": all(fixed_nearest),
        "spatial_selection_file_sha256": selection_sha,
        "original_score_files": {path.name: file_sha256(path) for path in history.iterdir() if path.is_file()},
        "spatial_zero_support": [row for row in numeric_rows if row["intervention"] == "spatial_0.00"],
        "implementation_sha256": file_sha256(snapshot)}
    manifest["implementation_sha256"] = file_sha256(__file__)
    manifest["files"] = {str(path.relative_to(public)).replace("\\", "/"): file_sha256(path)
        for path in sorted(public.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    write_json(public / "manifest.json", manifest)
    augment_source_manifest(out, public)
    print(json.dumps({"response_numerical_correction": "complete", "frozen_strengths_unchanged": True,
                      "all_fixed_choices_remain_nearest": all(fixed_nearest)}), flush=True)


def verify_completed_scores(args):
    """Reuse corrected scores only after verifying sources, caches and outputs."""
    public, out = Path(args.public_dir), Path(args.output_dir)
    if not (public / "manifest.json").exists():
        return False
    manifest = read_json(public / "manifest.json")
    correction = manifest.get("response_correlation_numerical_fix", {})
    if manifest.get("status") != "complete" or correction.get("version") != 2:
        return False
    if manifest["config_sha256"] != canonical_digest(read_json(args.config)):
        raise ValueError("Completed scores use a different supplementary configuration")
    if not verify_completed_predictions(args):
        raise ValueError("Completed scores lack verified completed predictions")
    snapshot = score_work_dir(args) / "source_snapshots/scoring_implementation_response_v2.py"
    if file_sha256(snapshot) != correction["implementation_sha256"]:
        raise ValueError("The actual corrected scoring source snapshot changed")
    for name, expected in manifest["files"].items():
        if file_sha256(evidence_file(public, name)) != expected:
            raise ValueError(f"A completed supplementary result changed: {name}")
    for name, expected in manifest["supplementary_source_records"].items():
        if file_sha256(out / name) != expected:
            raise ValueError("A completed supplementary source record changed")
    for name, expected in manifest["prediction_input_records"].items():
        if file_sha256(out / name) != expected["metadata_sha256"]:
            raise ValueError("A scored prediction identity record changed")
    endpoint = manifest.get("zero_spatial_endpoint_correction")
    if endpoint:
        snapshot = score_work_dir(args) / endpoint["source_snapshot"]
        if file_sha256(snapshot) != endpoint["implementation_sha256"]:
            raise ValueError("The zero spatial endpoint correction source snapshot changed")
    print(json.dumps({"encoder_supplementary_scores": "reused_verified_completed_outputs",
                      "response_correlation_numerical_fix": 2}), flush=True)
    return True


def mark_zero_spatial_endpoint(args):
    """Mark one verified endpoint NA using existing results and caches only.

    No prediction, GPFA transformation, matching selection, or bootstrap is run.
    Original response values remain diagnostic columns; the frozen selection
    record and all non-endpoint scores retain their original values.
    """
    import shutil
    import pandas as pd

    out, public, root = Path(args.output_dir), Path(args.public_dir), Path(args.run_root)
    if not verify_completed_scores(args):
        raise ValueError("Endpoint annotation requires the completed corrected response scores")
    manifest = read_json(public / "manifest.json")
    if manifest.get("zero_spatial_endpoint_correction", {}).get("version") == 1:
        print(json.dumps({"zero_spatial_endpoint": "reused_verified_annotation"}), flush=True)
        return
    selection_sha = file_sha256(out / "spatial_selection.json")
    history = score_work_dir(args) / "numerical_history/response_before_degenerate_endpoint"
    history.mkdir(parents=True, exist_ok=True)
    original_files = {path.name: file_sha256(path) for path in public.iterdir() if path.is_file()}
    for path in public.iterdir():
        if path.is_file():
            target = history / path.name
            if not target.exists():
                shutil.copy2(path, target)
            elif file_sha256(target) != original_files[path.name]:
                raise ValueError("The saved pre-annotation output differs")

    def read_table(name):
        return pd.read_csv(public / name, float_precision="round_trip")

    numeric = read_table("response_numeric_support.csv")
    evidence = []
    for role in ("validation", "test"):
        path = out / role / "spatial_0.00.npz"
        record = read_json(path.with_suffix(".json"))
        if record["identity"]["kind"] != "spatial" or record["identity"]["retention"] != 0:
            raise ValueError("The diagnostic is not a zero-spatial-weight endpoint")
        diagnostic = numeric[(numeric.role == role) & (numeric.intervention == "spatial_0.00")]
        if len(diagnostic) != 1:
            raise ValueError("A unique prior numerical diagnostic is required")
        with np.load(path, allow_pickle=False) as cache:
            endpoint = response_endpoint_status(cache["neural"], cache["prediction"], "spatial_0.00",
                diagnostic.iloc[0].to_dict(), record["weight_changes"])
        evidence.append({"role": role, "payload_sha256": record["payload_sha256"], **endpoint,
                         "exact_constant_prediction_neurons": int(diagnostic.iloc[0].exact_constant_prediction_neurons),
                         "near_constant_prediction_neurons": int(diagnostic.iloc[0].total_neurons - diagnostic.iloc[0].exact_constant_prediction_neurons),
                         "prediction_dtype": "float32",
                         "maximum_prediction_range": float(diagnostic.iloc[0].prediction_range_max)})
    numeric["response_status"] = np.where(numeric.intervention == "spatial_0.00",
                                         "numerically_degenerate", "defined")
    numeric["scientifically_defined_response_neurons"] = numeric.defined_response_neurons
    numeric.loc[numeric.intervention == "spatial_0.00", "scientifically_defined_response_neurons"] = 0
    numeric["interpretation"] = np.where(numeric.intervention == "spatial_0.00",
        "Verified zero spatial weights: 76 constant and 436 floating-point-floor outputs; raw correlations are diagnostic only",
        numeric.interpretation)

    def preserve_and_clear(frame, mask, columns):
        for name in columns:
            if name in frame:
                raw = "raw_" + name
                if frame[name].dtype == bool:
                    frame[name] = frame[name].astype(object)
                if raw not in frame:
                    frame[raw] = pd.Series(index=frame.index, dtype="object" if frame[name].dtype == object else "float64")
                frame.loc[mask, raw] = frame.loc[mask, name]
                frame.loc[mask, name] = np.nan

    grid = read_table("validation_grid.csv")
    grid["response_status"] = np.where(np.isfinite(grid.response_r), "defined", "undefined")
    zero = (grid.kind == "spatial") & (grid.retention == 0)
    preserve_and_clear(grid, zero, ["response_r", "valid_neurons"])
    grid.loc[zero, "valid_neurons"] = 0
    grid.loc[zero, "response_status"] = "numerically_degenerate"

    tables = {}
    for name in ("intervention_scores.csv", "condition_scores.csv", "paired_condition_scores.csv", "paired_intervals.csv"):
        frame = read_table(name)
        response = frame.metric == "response_neuron_mean_r"
        frame["response_status"] = "not_applicable"
        frame.loc[response, "response_status"] = "defined"
        if "intervention" in frame:
            zero = response & (frame.intervention == "spatial_0.00")
            values = ["value", "response_r_shared_neurons"]
        else:
            zero = response & ((frame.model_a == "spatial_0.00") | (frame.model_b == "spatial_0.00"))
            values = ["a", "b", "b_minus_a", "ci_low", "ci_high", "response_r_shared_neurons",
                      "valid_bootstraps", "undefined_bootstraps", "jointly_defined_conditions"]
        preserve_and_clear(frame, zero, values)
        frame.loc[zero, "response_status"] = "numerically_degenerate"
        if "response_r_shared_neurons" in frame:
            frame.loc[zero, "response_r_shared_neurons"] = 0
        if "valid_bootstraps" in frame:
            frame.loc[zero, "valid_bootstraps"] = 0
            frame.loc[zero, "undefined_bootstraps"] = (frame.loc[zero, "raw_valid_bootstraps"]
                                                       + frame.loc[zero, "raw_undefined_bootstraps"])
            frame.loc[zero, "jointly_defined_conditions"] = 0
        tables[name] = frame

    matches = read_table("spatial_response_matches.csv")
    zero = matches.spatial_retention == 0
    matches["response_status"] = "defined"
    matches["validation_matching_status"] = np.where(matches.corrected_validation_match, "matched", "outside_tolerance")
    matches["test_matching_status"] = np.where(matches.test_match, "matched", "outside_tolerance")
    invalid_columns = ["spatial_validation_r", "spatial_minus_temporal_validation_r", "validation_match", "status",
        "temporal_test_r", "spatial_test_r", "spatial_minus_temporal_test_r", "test_match",
        "response_r_shared_neurons", "corrected_spatial_validation_r", "corrected_validation_match",
        "fixed_choice_still_nearest_after_correction"]
    preserve_and_clear(matches, zero, invalid_columns)
    matches.loc[zero, "response_status"] = "numerically_degenerate"
    matches.loc[zero, ["status", "validation_matching_status", "test_matching_status"]] = "not_evaluable"
    matches.loc[zero, "response_r_shared_neurons"] = 0
    matches["temporal_test_r_support"] = "individual_original_support"
    matches["temporal_test_r_neurons"] = 512
    scores = tables["intervention_scores.csv"]
    for index, row in matches[zero].iterrows():
        temporal_name = f"temporal_{row.temporal_retention:.2f}"
        individual = scores[(scores.intervention == temporal_name) & (scores.metric == "response_neuron_mean_r")].iloc[0]
        matches.loc[index, "temporal_test_r"] = individual.value
        matches.loc[index, "temporal_test_r_neurons"] = individual.response_r_shared_neurons

    numeric.to_csv(public / "response_numeric_support.csv", index=False)
    grid.to_csv(public / "validation_grid.csv", index=False)
    for name, frame in tables.items():
        frame.to_csv(public / name, index=False)
    matches.to_csv(public / "spatial_response_matches.csv", index=False)
    _plot_scores(scores, matches, public)
    if selection_sha != file_sha256(out / "spatial_selection.json"):
        raise AssertionError("The frozen spatial selection changed")
    for name, digest in manifest["primary_sources"].items():
        if file_sha256(root / name) != digest:
            raise AssertionError("A primary artifact changed during endpoint annotation")
    snapshot = score_work_dir(args) / "source_snapshots/scoring_implementation_zero_spatial_endpoint.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    manifest["zero_spatial_endpoint_correction"] = {
        "version": 1, "status": "complete", "scope": "Existing zero-spatial-weight endpoint only",
        "rule": "Recorded float32 numerical-floor diagnostic plus verified zero first spatial-convolution weights; no amplitude cutoff for other predictions",
        "response_status": "numerically_degenerate", "matching_status": "not_evaluable",
        "raw_values": "raw_* columns retain the original calculations; earlier original_* fields retain pre-float64 values",
        "cached_endpoint_evidence": evidence, "frozen_strengths_changed": False,
        "spatial_selection_file_sha256": selection_sha, "primary_artifacts_unchanged": True,
        "prediction_or_gpfa_inference_run": False, "bootstrap_or_matching_selection_run": False,
        "original_score_files": original_files,
        "source_snapshot": "source_snapshots/scoring_implementation_zero_spatial_endpoint.py",
        "implementation_sha256": file_sha256(snapshot)}
    manifest["implementation_sha256"] = file_sha256(__file__)
    endpoint_support = manifest.get("individual_curve_support", {}).get("spatial_0.00")
    if endpoint_support is not None:
        endpoint_support["raw_response_r_shared_neurons"] = endpoint_support["response_r_shared_neurons"]
        endpoint_support["response_r_shared_neurons"] = 0
        endpoint_support["response_status"] = "numerically_degenerate"
    manifest["files"] = {str(path.relative_to(public)).replace("\\", "/"): file_sha256(path)
        for path in sorted(public.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    write_json(public / "manifest.json", manifest)
    print(json.dumps({"zero_spatial_endpoint": "numerically_degenerate", "matching_status": "not_evaluable",
                      "frozen_strengths_unchanged": True, "evidence": evidence}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "score", "mark-degenerate"))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--public-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--sensorium-source")
    parser.add_argument("--neuralpredictors-source")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage == "predict":
        if not all((args.data_root, args.sensorium_source, args.neuralpredictors_source)):
            parser.error("predict requires data and both pinned source directories")
        if not verify_completed_predictions(args):
            run_predictions(args)
    elif args.stage == "score":
        if not verify_completed_scores(args):
            run_scores(args)
            refresh_response_scores(args)
            augment_source_manifest(args.output_dir, args.public_dir)
        mark_zero_spatial_endpoint(args)
    else:
        mark_zero_spatial_endpoint(args)


if __name__ == "__main__":
    main()
