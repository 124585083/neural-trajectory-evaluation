"""Unweighted OLS on one condition-mean row per original condition/time bin.

Representations are supplied from the existing per-round train-only FA50/GPFA50
cache. This module fits only the two linear xy readouts. Behavioral trial counts
are provenance, never repetitions, regression weights, or independent neural N.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits


MODE = "condition_mean_neural_and_behavior"
REPRESENTATIONS = ("FA50", "GPFA50")
HEADS = ("D_obj", "D_beh")
TARGETS = ("objective", "behavior")
EPOCHS = ("full", "visible", "hidden", "bounce", "no_bounce", "post_bounce")
SCORE_FIELDS = ("r", "RMSE", "bias", "target_sd", "prediction_sd", "amplitude_ratio", "target_min", "target_max")


def _array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + np.asarray(a.shape, dtype=np.int64).tobytes() + a.tobytes()).hexdigest()


def source_causal_audit(rep):
    """Read the existing audit, rather than infer a pass from a model name."""
    directory = rep.get("model_directory")
    path = Path(directory) / "causal_audit.json" if directory else None
    if path is not None and path.is_file():
        raw = path.read_bytes()
        audit = json.loads(raw)
        source = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    else:
        audit = rep.get("causal_audit", rep.get("provenance", {}))
        source = {"path": None, "sha256": None, "scope": "caller-supplied audit for synthetic tests or explicit in-memory reuse"}
    for key in ("provided_input_filtering", "fit_provenance"):
        if audit.get(key) != "pass":
            raise ValueError(f"Reused representation requires its existing {key}=pass audit")
    if "raw_preprocessing" not in audit:
        raise ValueError("The reused raw preprocessing status must remain explicit")
    return {**source, **{key: audit[key] for key in ("provided_input_filtering", "fit_provenance", "raw_preprocessing")}}


def validate_inputs(payload, representations, split):
    """Build one common support before any representation/head is fitted."""
    if payload.get("data_mode") != MODE:
        raise ValueError("This version requires condition-mean neural AND behavior labels")
    ids = np.asarray(payload["condition_ids"], int)
    times = np.asarray(payload["times_ms"], float)
    if ids.shape != (79,) or len(np.unique(ids)) != 79:
        raise ValueError("Exactly the original 79 distinct physical identities are required")
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Original bin availability times must be finite and increasing")
    nt = len(times)
    objective = np.asarray(payload["objective_xy"], float)
    behavior = np.asarray(payload["behavior_xy"], float)
    if objective.shape != (79, nt, 2) or behavior.shape != objective.shape:
        raise ValueError("Labels must contain exactly one xy value per condition/original bin")
    n_behavior = np.asarray(payload["n_behavior_trials"])
    if n_behavior.shape != (79,) or not np.isfinite(n_behavior).all() or np.any(n_behavior < 0) or np.any(n_behavior != np.floor(n_behavior)):
        raise ValueError("n_behavior_trials must be 79 nonnegative provenance counts")
    common = np.asarray(payload["common_mask"], bool).copy()
    if common.shape != (79, nt):
        raise ValueError("Common label mask must be condition x original time")
    common &= np.isfinite(objective).all(-1) & np.isfinite(behavior).all(-1)
    common &= n_behavior[:, None] > 0
    if set(representations) != set(REPRESENTATIONS):
        raise ValueError("Exactly the reused FA50 and GPFA50 pair is required")
    for name in REPRESENTATIONS:
        rep = representations[name]
        latent = np.asarray(rep["latent"])
        if latent.shape != (79, nt, 50):
            raise ValueError(f"{name}: the fixed comparison requires 50 neural features")
        common &= np.isfinite(latent).all(-1)
        if "common_mask" in rep:
            measured = np.asarray(rep["common_mask"], bool)
            if measured.shape != common.shape:
                raise ValueError("Representation measurement support shape changed")
            common &= measured
        for key, expected in (("condition_ids", ids), ("times_ms", times)):
            if key in rep:
                np.testing.assert_array_equal(rep[key], expected)
        if not rep.get("causal_status"):
            raise ValueError("Preserve the actual causal status of the reused representation")
        source_causal_audit(rep)
        provenance = rep.get("provenance", {})
        for status in ("fit_provenance", "provided_input_filtering"):
            if provenance.get(status) in ("fail", "unknown"):
                raise ValueError(f"Reused representation has unsupported {status}")
        # A known raw preprocessing failure is retained, never promoted to pass.
    if not np.array_equal(objective[common, 0], behavior[common, 0]):
        raise ValueError("Objective and behavior labels must have exactly the same x")
    source_epochs = dict(payload["epoch_masks"])
    for alias, name in (("nobounce", "no_bounce"), ("postbounce", "post_bounce")):
        if alias in source_epochs and name not in source_epochs:
            source_epochs[name] = source_epochs[alias]
    if set(EPOCHS) - set(source_epochs):
        raise ValueError(f"All six fixed evaluation masks required: {EPOCHS}")
    epochs = {}
    for name in EPOCHS:
        mask = np.asarray(source_epochs[name], bool)
        if mask.shape != common.shape:
            raise ValueError(f"{name}: expected condition x original time support")
        epochs[name] = common & mask
    common &= epochs["full"]
    epochs = {name: value & common for name, value in epochs.items()}
    train = np.asarray(split["train_indices"], int)
    test = np.asarray(split["test_indices"], int)
    if len(train) != 39 or len(test) != 40 or set(train) & set(test) or set(train) | set(test) != set(range(79)):
        raise ValueError("Keep the original complete 39/40 condition split; never redraw missing conditions")
    for key, index in (("train_condition_ids", train), ("test_condition_ids", test)):
        if key in split:
            np.testing.assert_array_equal(split[key], ids[index])
    for rep in representations.values():
        provenance = rep.get("provenance", {})
        for key, index in (("fit_conditions", train), ("test_conditions", test)):
            if key in provenance and set(map(int, provenance[key])) != set(ids[index]):
                raise ValueError("Reused representation was fitted on another condition split")
        if "round_id" in provenance and int(provenance["round_id"]) != int(split["iteration"]):
            raise ValueError("Reused representation round identity changed")
    return ids, times, objective, behavior, n_behavior.astype(np.int64), common, epochs, train, test


def fit_condition_ols(X, objective, behavior, *, threads=2):
    """Actually fit original condition-time rows, without behavioral repetition."""
    X, objective, behavior = np.asarray(X, float), np.asarray(objective, float), np.asarray(behavior, float)
    if X.ndim != 2 or X.shape[1] != 50 or len(X) == 0 or objective.shape != (len(X), 2) or behavior.shape != objective.shape:
        raise ValueError("Nonempty [condition-time,50] features and paired xy labels required")
    if not all(np.isfinite(a).all() for a in (X, objective, behavior)):
        raise ValueError("OLS accepts only the shared finite support")
    if not np.array_equal(objective[:, 0], behavior[:, 0]):
        raise ValueError("The paired heads must share x labels")
    with threadpool_limits(limits=threads):
        model = LinearRegression(fit_intercept=True, positive=False).fit(X, np.c_[objective, behavior])
    if not np.allclose(model.coef_[0], model.coef_[2], rtol=1e-11, atol=1e-11) or not np.isclose(model.intercept_[0], model.intercept_[2], rtol=1e-11, atol=1e-11):
        raise AssertionError("Identical x targets gave different fitted mappings")
    return model


def score_rows(target, prediction):
    """Pearson r/RMSE on original valid rows, using the prior metric conventions."""
    y, p = np.asarray(target, float), np.asarray(prediction, float)
    if y.ndim != 2 or y.shape[-1] != 2 or p.shape != y.shape:
        raise ValueError("Scoring expects [valid original row,xy] target/prediction pairs")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Scoring support must be jointly finite")
    if not len(y):
        return {key: np.full(2, np.nan) for key in SCORE_FIELDS}
    yc, pc = y - y.mean(0), p - p.mean(0)
    norm = np.sqrt(np.sum(yc * yc, axis=0) * np.sum(pc * pc, axis=0))
    usable = (len(y) >= 3) & (np.ptp(y, axis=0) > 0) & (np.ptp(p, axis=0) > 0) & (norm > 0)
    r = np.divide(np.sum(yc * pc, axis=0), norm, out=np.full(2, np.nan), where=usable)
    target_sd, prediction_sd = np.std(y, axis=0, ddof=0), np.std(p, axis=0, ddof=0)
    return {"r": np.clip(r, -1., 1.), "RMSE": np.sqrt(np.mean((p - y) ** 2, axis=0)),
            "bias": np.mean(p - y, axis=0), "target_sd": target_sd, "prediction_sd": prediction_sd,
            "amplitude_ratio": np.divide(prediction_sd, target_sd, out=np.full(2, np.nan), where=target_sd > 0),
            "target_min": y.min(0), "target_max": y.max(0)}


def _count_rows(mask, n_behavior):
    contributes = mask.any(1)
    n = int(mask.sum())
    return {"n_conditions": int(contributes.sum()), "n_bins": n, "n_condition_time_rows": n,
            "n_unique_neural_input_ids": n, "n_behavior_trials_in_means": int(n_behavior[contributes].sum()),
            "n_neural_trials": 0}


def score_predictions(ids, objective, behavior, predictions, epochs, test, n_behavior, identity):
    self_rows, cross_rows, condition_rows = [], [], []
    is_test = np.zeros(len(ids), bool); is_test[test] = True
    for epoch in EPOCHS:
        mask = epochs[epoch] & is_test[:, None]
        counts = _count_rows(mask, n_behavior)
        scores = {}
        for h in range(2):
            for yi, target in enumerate((objective, behavior)):
                scored = score_rows(target[mask], predictions[h][mask])
                scores[h, yi] = scored
                for j, axis in enumerate(("x", "y")):
                    cross_rows.append({**identity, "epoch": epoch, "coordinate": axis, "head": HEADS[h], "target": TARGETS[yi],
                                       "is_self_target": h == yi, **counts, **{name: float(value[j]) for name, value in scored.items()}})
                for c in test:
                    condition_mask = mask[c]
                    result = score_rows(target[c, condition_mask], predictions[h, c, condition_mask])
                    for j, axis in enumerate(("x", "y")):
                        condition_rows.append({**identity, "epoch": epoch, "coordinate": axis, "head": HEADS[h], "target": TARGETS[yi],
                                               "condition_id": int(ids[c]), "condition_index": int(c), "is_self_target": h == yi,
                                               "n_bins": int(condition_mask.sum()), "n_condition_time_rows": int(condition_mask.sum()),
                                               "n_behavior_trials_in_mean": int(n_behavior[c]), "n_neural_trials": 0,
                                               **{name: float(value[j]) for name, value in result.items()}})
        for j, axis in enumerate(("x", "y")):
            oo, bb = scores[0, 0], scores[1, 1]
            self_rows.append({**identity, "epoch": epoch, "coordinate": axis, **counts,
                              "r_obj": float(oo["r"][j]), "r_beh": float(bb["r"][j]),
                              "RMSE_obj": float(oo["RMSE"][j]), "RMSE_beh": float(bb["RMSE"][j]),
                              "Delta_r": float(bb["r"][j] - oo["r"][j]),
                              "Delta_RMSE": float(oo["RMSE"][j] - bb["RMSE"][j])})
    return pd.DataFrame(self_rows), pd.DataFrame(cross_rows), pd.DataFrame(condition_rows)


def run_condition_round(payload, representations, split, output_dir, *, threads=2):
    """Refit only OLS for one specified original split; never launch a batch."""
    ids, times, objective, behavior, n_behavior, common, epochs, train, test = validate_inputs(payload, representations, split)
    root = Path(output_dir)
    for directory in ("models", "predictions", "round_results"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    train_mask = common.copy(); train_mask[test] = False
    test_mask = common.copy(); test_mask[train] = False
    training_indices = np.argwhere(train_mask)
    if not len(training_indices):
        raise ValueError("This original split has no fitting rows; do not redraw or fill missing labels")
    unique_inputs = training_indices[:, 0] * len(times) + training_indices[:, 1]
    if len(unique_inputs) != len(np.unique(unique_inputs)):
        raise AssertionError("A condition-time row was duplicated")
    common_hash = _array_sha(common)
    files, prediction_paths = [], {}
    summary_frames, cross_frames, condition_frames, fit_rows = [], [], [], []
    animal, iteration = str(payload["animal"]), int(split["iteration"])
    for representation in REPRESENTATIONS:
        rep = representations[representation]
        identity = {"animal": animal, "iteration": iteration, "representation": representation,
                    "data_mode": MODE, "causal_status": str(rep["causal_status"])}
        prefix = f"{animal}_{representation}_r{iteration:03d}"
        latent = np.asarray(rep["latent"], float)
        X = latent[train_mask]  # Exactly one original row per condition and bin.
        model = fit_condition_ols(X, objective[train_mask], behavior[train_mask], threads=threads)
        coefficients, intercepts = model.coef_.reshape(2, 2, 50), model.intercept_.reshape(2, 2)
        predictions = np.full((2, 79, len(times), 2), np.nan)
        with threadpool_limits(limits=threads):
            if test_mask.any():
                predicted = model.predict(latent[test_mask]).reshape(-1, 2, 2)
                for h in range(2):
                    predictions[h, test_mask] = predicted[:, h]
        if np.isfinite(predictions[:, train]).any() or test_mask[train].any():
            raise AssertionError("Saved predictions must contain only held-out conditions")
        if not np.allclose(predictions[0, ..., 0], predictions[1, ..., 0], rtol=1e-11, atol=1e-11, equal_nan=True):
            raise AssertionError("Same x heads gave different test predictions")
        model_path = root / "models" / f"{prefix}_ols.npz"
        prediction_path = root / "predictions" / f"{prefix}_test_predictions.npz"
        provenance = json.dumps(rep.get("provenance", {}), ensure_ascii=False, default=str)
        source_directory = str(rep.get("model_directory", ""))
        causal_audit = source_causal_audit(rep)
        causal_audit_json = json.dumps(causal_audit, ensure_ascii=False)
        np.savez_compressed(model_path, coefficients=coefficients, intercepts=intercepts, singular_values=model.singular_,
                            animal=np.asarray(animal), iteration=np.asarray(iteration), representation=np.asarray(representation),
                            design_rank=np.asarray(model.rank_), condition_time_design_rank=np.asarray(model.rank_),
                            n_train_rows=np.asarray(len(X)), training_condition_bin_indices=training_indices,
                            unique_neural_input_ids=unique_inputs, train_condition_indices=train, test_condition_indices=test,
                            condition_ids=ids, head_names=np.asarray(HEADS), data_mode=np.asarray(MODE),
                            causal_status=np.asarray(rep["causal_status"]), common_mask_sha256=np.asarray(common_hash),
                            source_representation_directory=np.asarray(source_directory), provenance_json=np.asarray(provenance),
                            source_causal_audit_json=np.asarray(causal_audit_json),
                            fit_method=np.asarray("LinearRegression(fit_intercept=True,positive=False); unweighted original condition-time rows; four independent outputs"))
        np.savez_compressed(prediction_path, predictions=predictions, condition_ids=ids, times_ms=times,
                            test_condition_indices=test, test_input_support=test_mask, common_mask=common,
                            common_mask_sha256=np.asarray(common_hash), n_behavior_trials_in_means=n_behavior,
                            head_names=np.asarray(HEADS), source_representation_directory=np.asarray(source_directory),
                            source_causal_audit_json=np.asarray(causal_audit_json),
                            **{key: np.asarray(value) for key, value in identity.items()})
        summary, cross, conditions = score_predictions(ids, objective, behavior, predictions, epochs, test, n_behavior, identity)
        for kind, frame in (("self", summary), ("cross_2x2", cross), ("condition", conditions)):
            path = root / "round_results" / f"{prefix}_{kind}.csv"
            frame.to_csv(path, index=False, float_format="%.17g")
            files.append(str(path.resolve()))
        fit_rows.append({**identity, "n_train_conditions_assigned": 39, "n_test_conditions_assigned": 40,
                         "n_train_conditions_contributing": int(train_mask.any(1).sum()), "n_test_conditions_contributing": int(test_mask.any(1).sum()),
                         "n_training_rows": len(X), "n_unique_training_neural_input_ids": len(unique_inputs),
                         "condition_time_design_rank": int(model.rank_), "n_neural_trials": 0,
                         "n_behavior_trials_in_training_means": int(n_behavior[train_mask.any(1)].sum()),
                         "behavior_counts_used_as_weights": False, "duplicate_condition_time_rows": 0,
                         "representation_refitted": False, "sample_weights": "none", "scaler": "none", "dtype": "float64",
                         "coefficients_per_coordinate": 51, "coefficients_per_xy_head": 102,
                         "x_coefficient_max_abs_difference": float(np.max(abs(coefficients[0, 0] - coefficients[1, 0]))),
                         "x_intercept_abs_difference": float(abs(intercepts[0, 0] - intercepts[1, 0])),
                         "common_mask_sha256": common_hash, "training_rows_sha256": _array_sha(training_indices),
                         "numpy_version": np.__version__, "scipy_version": scipy.__version__, "sklearn_version": sklearn.__version__,
                         "blas_threads": threads, "source_representation_directory": source_directory, "provenance_json": provenance,
                         "provided_input_filtering": causal_audit["provided_input_filtering"],
                         "fit_provenance": causal_audit["fit_provenance"], "raw_preprocessing": causal_audit["raw_preprocessing"],
                         "source_causal_audit_json": causal_audit_json,
                         "model_path": str(model_path.resolve()), "prediction_path": str(prediction_path.resolve())})
        files.extend([str(model_path.resolve()), str(prediction_path.resolve())])
        prediction_paths[representation] = str(prediction_path.resolve())
        summary_frames.append(summary); cross_frames.append(cross); condition_frames.append(conditions)
    audits = pd.DataFrame(fit_rows)
    audit_path = root / "round_results" / f"{animal}_r{iteration:03d}_fit_audit.csv"
    audits.to_csv(audit_path, index=False, float_format="%.17g")
    files.append(str(audit_path.resolve()))
    return {"summary": pd.concat(summary_frames, ignore_index=True), "cross_2x2": pd.concat(cross_frames, ignore_index=True),
            "condition_metrics": pd.concat(condition_frames, ignore_index=True), "fit_audit": audits,
            "files": files, "prediction_paths": prediction_paths, "common_row_count": int(common.sum()),
            "common_mask_sha256": common_hash}
