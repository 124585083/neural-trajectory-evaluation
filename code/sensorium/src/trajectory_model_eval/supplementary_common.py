"""Read-only access and paired summaries for diagnostics of a completed run."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .retrained_analysis import load_gpfa, validate_retrained_manifest
from .retrained_predictions import bundle_identity, load_bundle, select_neurons, verify_test_lock
from .unified_analysis import FrozenAnalysis, TRAJECTORY, _digest, _trajectory_scores, _json
from .unified_contract import file_sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class SupplementContext:
    root: Path
    config: dict
    manifest: dict
    preprocessing: dict
    lock: dict
    gpfa: object
    frozen: FrozenAnalysis
    test: dict
    development: dict | None
    sources: dict

    def assert_unchanged(self):
        if self.gpfa.model.parameter_digest() != self.lock["gpfa_identity"]["gpfa_parameter_digest"]:
            raise AssertionError("The primary GPFA was modified")
        for relative, digest in self.sources.items():
            if file_sha256(self.root / relative) != digest:
                raise AssertionError(f"A primary source artifact changed: {relative}")


def frozen_from_gpfa(gpfa, lock, config):
    """Construct the existing inference adapter without refitting any object."""
    a, model = gpfa.arrays, gpfa.model
    transform = _digest(a["precision"], a["encoder_precision"], a["observations"],
                        a["query_times"], a["neuron_ids"], a["frame_indices"],
                        model.times_seconds, model.jitter, model.min_noise)
    return FrozenAnalysis(model, a["precision"], a["encoder_precision"], a["observations"],
        a["query_times"], a["neuron_ids"], a["frame_indices"], model.parameter_digest(),
        lock["fit_digest"], lock["response_matching"]["sigma"], np.empty(0),
        lock["response_matching"], _digest(config), "supplementary_read_only",
        transform, lock["control_digest"], {"freeze_sha256": lock["freeze_sha256"]})


def load_context(run_root, include_development=False):
    """Validate only the frozen artifacts and prediction caches used here.

    No raw-data audit, model selection, fitting or source response reload occurs.
    The original main evaluation and all of its choices remain unchanged.
    """
    root = Path(run_root).resolve()
    config = read_json(root / "analysis_protocol.json")
    manifest = read_json(root / "data/split_manifest.json")
    preprocessing = read_json(root / "data/preprocessing.json")
    validate_retrained_manifest(manifest, config)
    lock = verify_test_lock(config, manifest, preprocessing, root / "encoders",
                            root / "frozen", root / "gpfa")
    if read_json(root / "evaluation/summary.json")["status"] != "held_out_evaluation_after_retraining_complete":
        raise ValueError("Supplementary diagnostics require the completed main comparison")
    pilot = manifest["session"]
    expected = bundle_identity(config, manifest, preprocessing, lock["checkpoint_identities"],
        pilot, ("test",), manifest["session_neurons"][pilot]["all_indices"], frozen_analysis=lock)
    test_path = root / "test" / pilot / "aligned_predictions.npz"
    test = select_neurons(load_bundle(test_path, expected), manifest["neurons"]["indices"])
    gpfa = load_gpfa(root / "gpfa", manifest, config, preprocessing["preprocessing_sha256"])
    frozen = frozen_from_gpfa(gpfa, lock, config)
    if not np.array_equal(test["encoder_precision"], frozen.encoder_precision):
        raise ValueError("Prediction and GPFA target scales differ")
    development = None
    if include_development:
        expected = bundle_identity(config, manifest, preprocessing, lock["checkpoint_identities"],
            pilot, ("train", "validation"), manifest["neurons"]["indices"])
        development = load_bundle(root / "development" / pilot / "aligned_predictions.npz", expected)
    files = ["analysis_protocol.json", "data/split_manifest.json", "data/preprocessing.json",
             "gpfa/frozen_gpfa.pkl", "gpfa/frozen_gpfa_arrays.npz", "frozen/frozen_analysis.json",
             "frozen/frozen_control.npz", "encoders/static/best.pt", "encoders/dynamic/best.pt",
             "evaluation/summary.json", "evaluation/pilot/paired_bootstrap.csv"]
    sources = {name: file_sha256(root / name) for name in files}
    sources[str(test_path.relative_to(root)).replace("\\", "/")] = file_sha256(test_path)
    return SupplementContext(root, config, manifest, preprocessing, lock, gpfa, frozen, test,
                             development, sources)


def paired_trajectory_intervals(left, right_a, right_b, *, dt=1/30, seed=42, n_boot=2000):
    """Whole-condition paired bootstrap with a pooled-trajectory estimand.

    Memoization only reuses repeated draws with identical condition multiplicity.
    It does not select or omit draws. Every metric retains NA rather than zero.
    """
    point, support = _trajectory_scores(left, {"a": right_a, "b": right_b}, dt)
    indices = np.random.default_rng(seed).integers(0, len(left), size=(n_boot, len(left)))
    cached, rows = {}, []
    draws = {m: [] for m in TRAJECTORY}
    for chosen in indices:
        key = tuple(sorted(chosen.tolist()))
        if key not in cached:
            sample = np.asarray(key)
            result, _ = _trajectory_scores(left[sample], {"a": right_a[sample], "b": right_b[sample]}, dt)
            cached[key] = {m: result["b"][m] - result["a"][m] for m in TRAJECTORY}
        for metric in TRAJECTORY:
            draws[metric].append(cached[key][metric])
    for metric, values in draws.items():
        values = np.asarray(values)
        finite = np.isfinite(values)
        low, high = np.quantile(values[finite], [.025, .975]) if finite.any() and len(left) > 1 else [np.nan, np.nan]
        rows.append({"metric": metric, "a": point["a"][metric], "b": point["b"][metric],
            "b_minus_a": point["b"][metric] - point["a"][metric], "ci_low": low, "ci_high": high,
            "n_conditions": len(left), "valid_bootstraps": int(finite.sum()),
            "undefined_bootstraps": int((~finite).sum()), "resampling_unit": "whole_movie_condition",
            "estimand": "pooled_condition_trajectory", **support})
    return pd.DataFrame(rows)


def write_manifest(output_dir, record):
    out = Path(output_dir)
    _json(out / "manifest.json", {**record, "files": {
        str(p.relative_to(out)).replace("\\", "/"): file_sha256(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}})
