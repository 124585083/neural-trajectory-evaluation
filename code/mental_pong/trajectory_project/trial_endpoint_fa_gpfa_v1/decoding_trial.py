"""Actual expanded-trial OLS and streamed trial scoring, without trial pairing claims.

Route B repeats each condition's neural feature vector for its real behavioral
records. It fits those expanded rows directly. Unique-condition predictions plus
the real trial index are a lossless storage format, not a mean-label fit.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits


HEADS = ("D_obj", "D_beh")
TARGETS = ("objective", "behavior")
EPOCHS = ("full", "visible", "hidden", "bounce", "no_bounce", "post_bounce")
MODE = "trial_labels_with_mean_neural"
STAT_FIELDS = ("sy", "sp", "sy2", "sp2", "syp", "sse", "se")
SCORE_FIELDS = ("r", "RMSE", "bias", "target_sd", "prediction_sd", "amplitude_ratio", "target_min", "target_max")


def generate_splits(condition_ids, n_splits=100):
    ids = np.asarray(condition_ids, int)
    if len(ids) != 79 or len(np.unique(ids)) != 79:
        raise ValueError("Exactly 79 original physical condition IDs required")
    groups = ids
    splitter = GroupShuffleSplit(n_splits=n_splits, train_size=39, test_size=40, random_state=0)
    return [{"iteration": k, "train_indices": tr.tolist(), "test_indices": te.tolist(),
             "train_condition_ids": ids[tr].tolist(), "test_condition_ids": ids[te].tolist()}
            for k, (tr, te) in enumerate(splitter.split(groups[:, None], groups=groups))]


def _trial_table(payload):
    table = payload["trial_table"]
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)
    required = {"trial_id", "condition_index", "condition_id", "session_id"}
    if not required.issubset(table):
        raise ValueError(f"trial_table requires {sorted(required)}")
    if table.trial_id.isna().any() or table.trial_id.astype(str).duplicated().any():
        raise ValueError("Real trial identifiers must be unique, traceable, and nonmissing")
    return table.reset_index(drop=True)


def _objective(payload, trial_indices, ci):
    values = np.asarray(payload["objective_xy"])
    axis = payload.get("objective_axis", "condition")
    if axis == "condition":
        return np.asarray(values[ci[trial_indices]], float)
    if axis == "trial":
        return np.asarray(values[trial_indices], float)
    raise ValueError("objective_axis must explicitly be condition or trial")


def _validate(payload, representations, split):
    if payload.get("data_mode") != MODE:
        raise ValueError("This implementation is Route B, never real trial neural decoding")
    table = _trial_table(payload)
    ids = np.asarray(payload["condition_ids"], int)
    times = np.asarray(payload["times_ms"], float)
    if len(ids) != 79 or len(set(ids.tolist())) != 79:
        raise ValueError("Retain all 79 physical identities before row filtering")
    ci = table.condition_index.to_numpy(int)
    if np.any((ci < 0) | (ci >= 79)) or not np.array_equal(ids[ci], table.condition_id.to_numpy(int)):
        raise ValueError("Trial condition indices/IDs disagree with frozen condition order")
    nt, nb = len(table), len(times)
    if times.ndim != 1 or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
        raise ValueError("Original bin availability times must be ordered and finite")
    beh = np.asarray(payload["behavior_xy"])
    if beh.shape != (nt, nb, 2):
        raise ValueError("behavior_xy must retain each real trial: [trial,time,xy]")
    objective_shape = (79 if payload.get("objective_axis", "condition") == "condition" else nt, nb, 2)
    if np.asarray(payload["objective_xy"]).shape != objective_shape:
        raise ValueError("objective_xy shape disagrees with explicit objective_axis")
    common = np.asarray(payload["common_mask"], bool).copy()
    if common.shape != (nt, nb):
        raise ValueError("A real trial x original-time common mask is required")
    if len(representations) != 2:
        raise ValueError("Both FA50 and causal GPFA50 representations are required together")
    neural_common = np.ones((79, nb), bool)
    for name, rep in representations.items():
        z = np.asarray(rep["latent"])
        if z.shape != (79, nb, 50):
            raise ValueError(f"{name}: main comparison requires exactly 50 features")
        neural_common &= np.isfinite(z).all(-1)
        if "common_mask" in rep:
            measured = np.asarray(rep["common_mask"], bool)
            if measured.shape != (79, nb):
                raise ValueError("Representation measurement support must retain condition x time")
            neural_common &= measured
        for key, expected in (("condition_ids", ids), ("times_ms", times)):
            if key in rep:
                np.testing.assert_array_equal(rep[key], expected)
        if not rep.get("causal_status"):
            raise ValueError("Explicit causal_status is required; never infer a pass")
        if rep.get("provenance", {}).get("fit_provenance") == "fail" or rep.get("provenance", {}).get("provided_input_filtering") == "fail":
            raise ValueError("Failed filtering or fit provenance cannot enter primary comparison")
    # Construct one common row definition for all representations and targets.
    for start in range(0, nt, 4096):
        ix = np.arange(start, min(start + 4096, nt))
        obj = _objective(payload, ix, ci)
        use = common[ix] & neural_common[ci[ix]] & np.isfinite(obj).all(-1) & np.isfinite(beh[ix]).all(-1)
        if not np.array_equal(obj[use, 0], beh[ix][use, 0]):
            raise ValueError("Both target families must share exactly the same x at every paired row")
        common[ix] = use
    epoch_input = dict(payload["epoch_masks"])
    for alias, canonical in (("nobounce", "no_bounce"), ("postbounce", "post_bounce")):
        if alias in epoch_input and canonical not in epoch_input:
            epoch_input[canonical] = epoch_input[alias]
    if set(EPOCHS) - set(epoch_input):
        raise ValueError(f"epoch_masks require {EPOCHS}")
    epochs = {}
    for epoch in EPOCHS:
        mask = np.asarray(epoch_input[epoch], bool)
        if mask.shape != (nt, nb):
            raise ValueError(f"{epoch}: expected trial x time mask")
        epochs[epoch] = common & mask
    common &= epochs["full"]
    epochs = {key: value & common for key, value in epochs.items()}
    tr, te = np.asarray(split["train_indices"], int), np.asarray(split["test_indices"], int)
    if len(tr) != 39 or len(te) != 40 or set(tr) & set(te) or set(tr) | set(te) != set(range(79)):
        raise ValueError("Each round must assign every condition to exactly one 39/40 side")
    for key, indices in (("train_condition_ids", tr), ("test_condition_ids", te)):
        if key in split:
            np.testing.assert_array_equal(ids[indices], split[key])
    for rep in representations.values():
        provenance = rep.get("provenance", {})
        for key, indices in (("fit_conditions", tr), ("test_conditions", te)):
            if key in provenance and set(map(int, provenance[key])) != set(ids[indices].tolist()):
                raise ValueError("Round-specific representation provenance does not match this split")
    return table, ci, common, epochs, tr, te


def fit_expanded_ols(X, objective, behavior, *, threads=4):
    """One four-output solve equals two separately parameterized two-output OLS heads."""
    X, objective, behavior = np.asarray(X, float), np.asarray(objective, float), np.asarray(behavior, float)
    if X.ndim != 2 or X.shape[1] != 50 or objective.shape != (len(X), 2) or behavior.shape != objective.shape or not len(X):
        raise ValueError("Nonempty actually expanded [row,50] and paired [row,2] labels required")
    if not all(np.isfinite(value).all() for value in (X, objective, behavior)):
        raise ValueError("Nonfinite rows must be excluded identically before the fit")
    if not np.array_equal(objective[:, 0], behavior[:, 0]):
        raise ValueError("Same-x labels are required")
    with threadpool_limits(limits=threads):
        estimator = LinearRegression(fit_intercept=True, positive=False).fit(X, np.c_[objective, behavior])
    if not np.allclose(estimator.coef_[0], estimator.coef_[2], rtol=1e-11, atol=1e-11) or not np.isclose(estimator.intercept_[0], estimator.intercept_[2], rtol=1e-11, atol=1e-11):
        raise AssertionError("Two identical x targets did not produce identical x mappings")
    return estimator


def squared_error_decomposition(target, prediction, unique_input_ids):
    """Numerical identity audit only; never supplies labels to the main fit."""
    y, p = np.asarray(target, float), np.asarray(prediction, float)
    if y.ndim == 1:
        y, p = y[:, None], p[:, None]
    if y.shape != p.shape:
        raise ValueError("Targets and repeated-input predictions must have matching shapes")
    ids = np.asarray(unique_input_ids)
    total = np.sum((y-p)**2, axis=0)
    between, within = np.zeros(y.shape[1]), np.zeros(y.shape[1])
    max_prediction_spread = 0.
    for uid in np.unique(ids):
        mask = ids == uid
        prediction_at_z = p[mask]
        spread = float(np.max(np.ptp(prediction_at_z, axis=0)))
        max_prediction_spread = max(max_prediction_spread, spread)
        if spread > 1e-10:
            raise AssertionError("Identical neural inputs have different predictions")
        ym = y[mask].mean(0)
        between += mask.sum() * (ym-prediction_at_z[0])**2
        within += np.sum((y[mask]-ym)**2, axis=0)
    return {"sse_total": total, "n_times_mean_error_squared": between, "within_trial_label_sse": within,
            "maximum_identity_residual": float(np.max(abs(total-between-within))),
            "same_input_prediction_max_spread": max_prediction_spread}


def _raw_stats(target, prediction, mask):
    """One statistics row per real trial, computed before any label averaging."""
    valid = np.asarray(mask, bool)
    if not np.isfinite(target[valid]).all() or not np.isfinite(prediction[valid]).all():
        raise ValueError("Common-valid trial prediction or target is missing")
    y, p = np.where(valid[..., None], target, 0.), np.where(valid[..., None], prediction, 0.)
    e = p-y
    return {"n": valid.sum(1).astype(np.int64), "sy": y.sum(1), "sp": p.sum(1),
            "sy2": (y*y).sum(1), "sp2": (p*p).sum(1), "syp": (y*p).sum(1),
            "sse": (e*e).sum(1), "se": e.sum(1),
            "ymin": np.min(np.where(valid[..., None], target, np.inf), axis=1),
            "ymax": np.max(np.where(valid[..., None], target, -np.inf), axis=1),
            "pmin": np.min(np.where(valid[..., None], prediction, np.inf), axis=1),
            "pmax": np.max(np.where(valid[..., None], prediction, -np.inf), axis=1)}


def _final_stats(stats):
    n = np.asarray(stats["n"], float)
    denominator = np.maximum(n[..., None], 1.)
    variance_y = np.maximum(stats["sy2"] - stats["sy"]**2 / denominator, 0.)
    variance_p = np.maximum(stats["sp2"] - stats["sp"]**2 / denominator, 0.)
    covariance = stats["syp"] - stats["sy"] * stats["sp"] / denominator
    r_den = np.sqrt(variance_y * variance_p)
    usable_r = ((n[..., None] >= 3) & (stats["ymax"] > stats["ymin"])
                & (stats["pmax"] > stats["pmin"]) & (r_den > 0))
    r = np.divide(covariance, r_den, out=np.full_like(covariance, np.nan), where=usable_r)
    rmse = np.sqrt(stats["sse"] / denominator)
    bias = stats["se"] / denominator
    sd_y, sd_p = np.sqrt(variance_y / denominator), np.sqrt(variance_p / denominator)
    for value in (rmse, bias, sd_y, sd_p):
        value[n == 0] = np.nan
    amplitude = np.divide(sd_p, sd_y, out=np.full_like(sd_p, np.nan), where=sd_y > 0)
    return {"r": np.clip(r, -1., 1.), "RMSE": rmse, "bias": bias, "target_sd": sd_y,
            "prediction_sd": sd_p, "amplitude_ratio": amplitude,
            "target_min": np.where(n[..., None] > 0, stats["ymin"], np.nan),
            "target_max": np.where(n[..., None] > 0, stats["ymax"], np.nan)}


class _ByCondition:
    def __init__(self, n_conditions=79):
        self.n_groups = n_conditions
        self.values = {"n": np.zeros(n_conditions, np.int64)}
        self.values.update({name: np.zeros((n_conditions, 2)) for name in STAT_FIELDS})
        self.values.update({name: np.full((n_conditions, 2), np.inf if name.endswith("min") else -np.inf)
                            for name in ("ymin", "ymax", "pmin", "pmax")})
        self.n_trials = np.zeros(n_conditions, np.int64)

    def update(self, condition_indices, stats):
        c = np.asarray(condition_indices, int)
        self.values["n"] += np.bincount(c, weights=stats["n"], minlength=self.n_groups).astype(np.int64)
        self.n_trials += np.bincount(c, weights=(stats["n"] > 0), minlength=self.n_groups).astype(np.int64)
        for name in STAT_FIELDS:
            for coordinate in range(2):
                self.values[name][:, coordinate] += np.bincount(c, weights=stats[name][:, coordinate], minlength=self.n_groups)
        for name in ("ymin", "ymax", "pmin", "pmax"):
            operation = np.minimum.at if name.endswith("min") else np.maximum.at
            operation(self.values[name], c, stats[name])

    def total(self):
        output = {name: np.sum(value, axis=0) for name, value in self.values.items() if name not in ("ymin", "ymax", "pmin", "pmax")}
        output.update({name: (np.min(value, axis=0) if name.endswith("min") else np.max(value, axis=0))
                       for name, value in self.values.items() if name in ("ymin", "ymax", "pmin", "pmax")})
        return output


def _score_saved_predictions(payload, table, ci, common, epochs, test_indices, predictions,
                             identity, score_path, *, chunk_trials):
    accumulators = {(epoch, h, y): _ByCondition() for epoch in EPOCHS for h in range(2) for y in range(2)}
    test_trials = np.flatnonzero(np.isin(ci, test_indices))
    ntime = common.shape[1]
    seen = {epoch: np.zeros((79, ntime), bool) for epoch in EPOCHS}
    session_sets = {epoch: set() for epoch in EPOCHS}
    session_codes, session_names = pd.factorize(table.session_id.astype(str), sort=True)
    session_acc = {(epoch, h): _ByCondition(len(session_names)) for epoch in EPOCHS[:3] for h in range(2)}
    session_seen = {epoch: np.zeros((len(session_names), 79, ntime), bool) for epoch in EPOCHS[:3]}
    trial_scores = np.full((len(test_trials), len(EPOCHS), 2, 2, len(SCORE_FIELDS)), np.nan)
    trial_nbins = np.zeros((len(test_trials), len(EPOCHS)), np.int32)
    for start in range(0, len(test_trials), chunk_trials):
            trials = test_trials[start:start + chunk_trials]
            c = ci[trials]
            target_arrays = (_objective(payload, trials, ci), np.asarray(payload["behavior_xy"][trials], float))
            prediction_arrays = (predictions[0, c], predictions[1, c])
            for ei, epoch in enumerate(EPOCHS):
                mask = epochs[epoch][trials]
                np.logical_or.at(seen[epoch], c, mask)
                good_trials = np.flatnonzero(mask.any(1))
                session_sets[epoch].update(table.iloc[trials[good_trials]].session_id.astype(str))
                if epoch in session_seen:
                    np.logical_or.at(session_seen[epoch], (session_codes[trials], c), mask)
                trial_nbins[start:start + len(trials), ei] = mask.sum(1)
                for h in range(2):
                    for y in range(2):
                        stats = _raw_stats(target_arrays[y], prediction_arrays[h], mask)
                        accumulators[(epoch, h, y)].update(c, stats)
                        if h != y:
                            continue  # Complete cross matrix retained per condition and pooled.
                        scores = _final_stats(stats)
                        trial_scores[start:start + len(trials), ei, h] = np.stack([scores[key] for key in SCORE_FIELDS], axis=-1)
                        if epoch in session_seen:
                            session_acc[(epoch, h)].update(session_codes[trials], stats)
    np.savez_compressed(score_path, scores=trial_scores, n_bins=trial_nbins,
                        test_trial_indices=test_trials, trial_ids=table.iloc[test_trials].trial_id.astype(str).to_numpy(dtype=str),
                        session_ids=table.iloc[test_trials].session_id.astype(str).to_numpy(dtype=str),
                        condition_indices=ci[test_trials], condition_ids=np.asarray(payload["condition_ids"])[ci[test_trials]],
                        epoch_names=np.asarray(EPOCHS), head_names=np.asarray(HEADS), target_names=np.asarray(TARGETS),
                        coordinate_names=np.asarray(["x", "y"]), metric_names=np.asarray(SCORE_FIELDS),
                        axis_order=np.asarray("scores[test_trial,epoch,self_head,coordinate,metric]; self_head matches its own target"),
                        **{key: np.asarray(value) for key, value in identity.items()})
    del trial_scores
    cross_rows, condition_rows, summary_rows = [], [], []
    for epoch in EPOCHS:
        overall = {}
        for h in range(2):
            for y in range(2):
                aggregate = accumulators[(epoch, h, y)]
                total = aggregate.total()
                final = _final_stats(total)
                overall[(h, y)] = final
                n_conditions = int(np.sum(aggregate.values["n"] > 0))
                for coordinate, axis in enumerate(("x", "y")):
                    row = {**identity, "epoch": epoch, "head": HEADS[h], "target": TARGETS[y],
                           "coordinate": axis, "n_conditions": n_conditions, "n_real_trials": int(aggregate.n_trials.sum()),
                           "n_neural_trials": 0, "n_sessions": len(session_sets[epoch]), "n_bins": int(total["n"]),
                           "n_unique_neural_input_ids": int(seen[epoch].sum()), "is_self_target": h == y}
                    row.update({key: float(value[coordinate]) for key, value in final.items()})
                    cross_rows.append(row)
                by_condition = _final_stats(aggregate.values)
                for c in test_indices:
                    for coordinate, axis in enumerate(("x", "y")):
                        row = {**identity, "condition_index": int(c), "condition_id": int(payload["condition_ids"][c]),
                               "epoch": epoch, "head": HEADS[h], "target": TARGETS[y], "coordinate": axis,
                               "n_real_trials": int(aggregate.n_trials[c]), "n_neural_trials": 0,
                               "n_bins": int(aggregate.values["n"][c]), "is_self_target": h == y}
                        row.update({key: float(value[c, coordinate]) for key, value in by_condition.items()})
                        condition_rows.append(row)
        reference = accumulators[(epoch, 0, 0)]
        for coordinate, axis in enumerate(("x", "y")):
            own_obj, own_beh = overall[(0, 0)], overall[(1, 1)]
            summary_rows.append({**identity, "epoch": epoch, "coordinate": axis,
                                 "n_conditions": int(np.sum(reference.values["n"] > 0)),
                                 "n_real_trials": int(reference.n_trials.sum()), "n_neural_trials": 0,
                                 "n_sessions": len(session_sets[epoch]), "n_bins": int(reference.values["n"].sum()),
                                 "n_unique_neural_input_ids": int(seen[epoch].sum()),
                                 "r_obj": float(own_obj["r"][coordinate]), "r_beh": float(own_beh["r"][coordinate]),
                                 "RMSE_obj": float(own_obj["RMSE"][coordinate]), "RMSE_beh": float(own_beh["RMSE"][coordinate]),
                                 "Delta_r": float(own_beh["r"][coordinate] - own_obj["r"][coordinate]),
                                 "Delta_RMSE": float(own_obj["RMSE"][coordinate] - own_beh["RMSE"][coordinate])})
    session_rows = []
    for epoch in EPOCHS[:3]:
        for h in range(2):
            aggregate = session_acc[(epoch, h)]
            scores = _final_stats(aggregate.values)
            for si, session in enumerate(session_names):
                for coordinate, axis in enumerate(("x", "y")):
                    row = {**identity, "session_id": str(session), "epoch": epoch, "head": HEADS[h],
                           "target": TARGETS[h], "coordinate": axis, "is_self_target": True,
                           "n_conditions": int(session_seen[epoch][si].any(1).sum()),
                           "n_unique_neural_input_ids": int(session_seen[epoch][si].sum()),
                           "n_real_trials": int(aggregate.n_trials[si]), "n_neural_trials": 0,
                           "n_bins": int(aggregate.values["n"][si])}
                    row.update({key: float(value[si, coordinate]) for key, value in scores.items()})
                    session_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(cross_rows), pd.DataFrame(condition_rows), pd.DataFrame(session_rows)


def estimate_cost(payload, n_splits=100):
    nrows = int(np.asarray(payload["common_mask"], bool).sum())
    ntrials = len(_trial_table(payload))
    expected_train = nrows * 39 / 79
    return {"n_real_trials": ntrials, "n_trial_time_rows_before_round_masks": nrows,
            "expected_training_rows": expected_train,
            "expanded_X_float64_bytes_per_representation": int(expected_train * 50 * 8),
            "four_output_Y_float64_bytes": int(expected_train * 4 * 8),
            "minimum_two_X_working_arrays_bytes": int(expected_train * 50 * 8 * 2),
            "condition_prediction_uncompressed_bytes_all_rounds_two_representations": int(n_splits * 2 * 2 * 79 * len(payload["times_ms"]) * 2 * 8),
            "expanded_prediction_uncompressed_bytes_all_rounds_two_representations": int(n_splits * 2 * nrows * 40 / 79 * 4 * 8),
            "storage": "lossless normalized condition prediction NPZ + real trial/sample index; expanded gzip exporter provided",
            "fitting": "one representation at a time; actually expanded real trial rows; no mean-label shortcut"}


def run_trial_round(payload, representations, split, output_dir, *, chunk_trials=2048, threads=4):
    """Fit one explicitly specified round; never launches a formal 100-round run."""
    table, ci, common, epochs, train_conditions, test_conditions = _validate(payload, representations, split)
    root = Path(output_dir)
    for name in ("models", "predictions", "trial_scores", "round_results"):
        (root / name).mkdir(parents=True, exist_ok=True)
    iteration, animal = int(split["iteration"]), str(payload["animal"])
    tr_rows, tr_bins = np.nonzero(common & np.isin(ci, train_conditions)[:, None])
    if len(tr_rows) == 0:
        raise ValueError("No valid training rows; do not substitute or redraw conditions")
    objective_rows = tr_rows if payload.get("objective_axis", "condition") == "trial" else ci[tr_rows]
    train_obj = np.asarray(payload["objective_xy"])[objective_rows, tr_bins].astype(float)
    train_beh = np.asarray(payload["behavior_xy"])[tr_rows, tr_bins].astype(float)
    unique_train_ids = np.unique(ci[tr_rows] * len(payload["times_ms"]) + tr_bins)
    summary_frames, cross_frames, condition_frames, session_frames, audits, files, prediction_paths = [], [], [], [], [], [], {}
    test_trials = np.flatnonzero(np.isin(ci, test_conditions))
    common_mask_sha256 = hashlib.sha256(np.asarray(common.shape, dtype=np.int64).tobytes() + common.tobytes()).hexdigest()
    trial_identity_sha256 = hashlib.sha256(table[["trial_id", "condition_index", "condition_id", "session_id"]].to_csv(index=False).encode("utf-8")).hexdigest()
    test_input_support = np.zeros((79, len(payload["times_ms"])), bool)
    for start in range(0, len(table), chunk_trials):
        chunk = np.arange(start, min(start + chunk_trials, len(table)))
        np.logical_or.at(test_input_support, ci[chunk], common[chunk] & np.isin(ci[chunk], test_conditions)[:, None])
    for representation, rep in representations.items():
        identity = {"animal": animal, "iteration": iteration, "representation": str(representation),
                    "data_mode": MODE, "causal_status": str(rep["causal_status"])}
        prefix = f"{animal}_{representation}_r{iteration:03d}"
        latent = np.asarray(rep["latent"], float)
        X = latent[ci[tr_rows], tr_bins]  # Real expanded design is passed to sklearn.fit.
        model = fit_expanded_ols(X, train_obj, train_beh, threads=threads)
        unique_X = latent.reshape(-1, 50)[unique_train_ids]
        with threadpool_limits(limits=threads):
            unique_rank = int(np.linalg.matrix_rank(unique_X - unique_X.mean(0)))
        if model.rank_ > unique_rank:
            raise AssertionError("Repeating neural inputs increased the numerical design rank")
        coefficients, intercepts = model.coef_.reshape(2, 2, 50), model.intercept_.reshape(2, 2)
        del X, unique_X
        gc.collect()
        predictions = np.full((2, 79, len(payload["times_ms"]), 2), np.nan)
        with threadpool_limits(limits=threads):
            if test_input_support.any():
                predicted = model.predict(latent[test_input_support]).reshape(-1, 2, 2)
                for h in range(2):
                    predictions[h, test_input_support] = predicted[:, h]
        if np.isfinite(predictions[:, train_conditions]).any():
            raise AssertionError("Training predictions cannot enter saved held-out output")
        model_path = root / "models" / f"{prefix}_ols.npz"
        prediction_path = root / "predictions" / f"{prefix}_test_predictions.npz"
        score_path = root / "trial_scores" / f"{prefix}_self_scores.npz"
        provenance_json = json.dumps(rep.get("provenance", {}), ensure_ascii=False, default=str)
        np.savez_compressed(model_path, coefficients=coefficients, intercepts=intercepts,
                            singular_values=model.singular_, expanded_design_rank=np.asarray(model.rank_),
                            unique_design_rank=np.asarray(unique_rank), n_train_rows=np.asarray(len(tr_rows)),
                            n_train_real_trials=np.asarray(len(np.unique(tr_rows))), unique_neural_input_ids=unique_train_ids,
                            train_condition_indices=train_conditions, test_condition_indices=test_conditions,
                            condition_ids=payload["condition_ids"], head_names=np.asarray(HEADS), data_mode=np.asarray(MODE),
                            causal_status=np.asarray(rep["causal_status"]), provenance_json=np.asarray(provenance_json),
                            fit_method=np.asarray("LinearRegression(fit_intercept=True, positive=False), four separately parameterized outputs; actual expanded trial rows"))
        np.savez_compressed(prediction_path, predictions=predictions, condition_ids=payload["condition_ids"],
                            times_ms=payload["times_ms"], test_condition_indices=test_conditions,
                            test_input_support=test_input_support, head_names=np.asarray(HEADS),
                            test_trial_indices=test_trials, test_common_mask_packed=np.packbits(common[test_trials], axis=1),
                            common_mask_sha256=np.asarray(common_mask_sha256), trial_identity_sha256=np.asarray(trial_identity_sha256),
                            sample_index_source_json=np.asarray(json.dumps(payload.get("sample_index_source", {}), default=str)),
                            indexing=np.asarray("predictions[head, trial_table.condition_index[test_trial_indices[j]], bin, xy]; row valid iff unpackbits(test_common_mask_packed[j])[:len(times_ms)][bin]"),
                            animal=np.asarray(animal), iteration=np.asarray(iteration), representation=np.asarray(representation),
                            data_mode=np.asarray(MODE), causal_status=np.asarray(rep["causal_status"]))
        summary, cross, conditions, sessions = _score_saved_predictions(payload, table, ci, common, epochs, test_conditions,
                                                              predictions, identity, score_path, chunk_trials=chunk_trials)
        for name, frame in (("self", summary), ("cross_2x2", cross), ("condition", conditions), ("session", sessions)):
            path = root / "round_results" / f"{prefix}_{name}.csv"
            frame.to_csv(path, index=False, float_format="%.17g")
            files.append(str(path.resolve()))
        audit = {**identity, "n_train_conditions_assigned": 39, "n_test_conditions_assigned": 40,
                 "n_train_conditions_contributing": len(np.unique(ci[tr_rows])),
                 "n_train_real_trials": len(np.unique(tr_rows)), "n_neural_trials": 0, "n_training_rows": len(tr_rows),
                 "n_unique_training_neural_input_ids": len(unique_train_ids), "expanded_design_rank": int(model.rank_),
                 "unique_design_rank": unique_rank, "same_input_predictions": "one unique condition-time prediction linked to every real trial; exact identity by construction",
                 "x_coefficient_max_abs_difference": float(np.max(abs(coefficients[0, 0] - coefficients[1, 0]))),
                 "x_intercept_abs_difference": float(abs(intercepts[0, 0] - intercepts[1, 0])),
                 "target_averaging_before_fit": False, "sample_weights": "none", "scaler": "none",
                 "common_mask_sha256": common_mask_sha256, "trial_identity_sha256": trial_identity_sha256,
                 "dtype": "float64", "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                 "sklearn_version": sklearn.__version__, "blas_threads": threads,
                 "provenance_json": provenance_json, "model_path": str(model_path.resolve()),
                 "prediction_path": str(prediction_path.resolve()), "per_trial_scores_path": str(score_path.resolve())}
        audits.append(audit)
        summary_frames.append(summary); cross_frames.append(cross); condition_frames.append(conditions); session_frames.append(sessions)
        files.extend(str(path.resolve()) for path in (model_path, prediction_path, score_path))
        prediction_paths[representation] = str(prediction_path.resolve())
        del model, predictions
        gc.collect()
    audit_frame = pd.DataFrame(audits)
    audit_path = root / "round_results" / f"{animal}_r{iteration:03d}_fit_audit.csv"
    audit_frame.to_csv(audit_path, index=False)
    files.append(str(audit_path.resolve()))
    return {"summary": pd.concat(summary_frames, ignore_index=True), "cross_2x2": pd.concat(cross_frames, ignore_index=True),
            "condition_metrics": pd.concat(condition_frames, ignore_index=True), "fit_audit": audit_frame,
            "session_metrics": pd.concat(session_frames, ignore_index=True),
            "files": files, "prediction_paths": prediction_paths, "common_row_count": int(common.sum())}


def export_trial_predictions(payload, prediction_path, output_gz, *, trial_ids=None, chunk_trials=2048):
    """Materialize losslessly linked real-trial held-out rows without refitting."""
    table = _trial_table(payload)
    ci = table.condition_index.to_numpy(int)
    with np.load(prediction_path, allow_pickle=False) as saved:
        predictions = saved["predictions"].copy()
        support = saved["test_input_support"].copy()
        test_conditions = saved["test_condition_indices"].copy()
        saved_trials = saved["test_trial_indices"].copy()
        saved_mask = np.unpackbits(saved["test_common_mask_packed"], axis=1)[:, :len(payload["times_ms"])].astype(bool)
        trial_sha = hashlib.sha256(table[["trial_id", "condition_index", "condition_id", "session_id"]].to_csv(index=False).encode("utf-8")).hexdigest()
        if trial_sha != saved["trial_identity_sha256"].item():
            raise ValueError("Prediction lookup requires the identical real-trial index table")
        np.testing.assert_array_equal(saved["condition_ids"], payload["condition_ids"])
        np.testing.assert_array_equal(saved["times_ms"], payload["times_ms"])
        identity = {key: saved[key].item() for key in ("animal", "iteration", "representation", "data_mode", "causal_status")}
    selected = np.isin(np.arange(len(table)), saved_trials)
    if trial_ids is not None:
        selected &= table.trial_id.astype(str).isin(set(map(str, trial_ids))).to_numpy()
    trials = np.flatnonzero(selected)
    common_saved = np.zeros((len(table), len(payload["times_ms"])), bool)
    common_saved[saved_trials] = saved_mask
    output_gz = Path(output_gz)
    output_gz.parent.mkdir(parents=True, exist_ok=True)
    fields = [*identity, "sample_id", "trial_index", "trial_id", "session_id", "condition_id", "original_bin_index", "time_ms", "unique_neural_input_id", "x_obj", "y_obj", "x_beh", "y_beh", "D_obj_x", "D_obj_y", "D_beh_x", "D_beh_y"]
    count = 0
    trial_names = table.trial_id.astype(str).to_numpy()
    session_names = table.session_id.astype(str).to_numpy()
    condition_names = table.condition_id.to_numpy(int)
    with gzip.open(output_gz, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for start in range(0, len(trials), chunk_trials):
            chunk = trials[start:start + chunk_trials]
            objective = _objective(payload, chunk, ci)
            behavior = np.asarray(payload["behavior_xy"])[chunk]
            mask = common_saved[chunk]
            if not np.isfinite(objective[mask]).all() or not np.isfinite(behavior[mask]).all():
                raise ValueError("Saved valid labels are now missing; do not silently change scored support")
            for local, bi in zip(*np.nonzero(mask)):
                ti, c = int(chunk[local]), int(ci[chunk[local]])
                values = (*objective[local, bi], *behavior[local, bi], *predictions[0, c, bi], *predictions[1, c, bi])
                row = {**identity, "sample_id": ti * len(payload["times_ms"]) + int(bi), "trial_index": ti,
                       "trial_id": trial_names[ti], "session_id": session_names[ti], "condition_id": int(condition_names[ti]),
                       "original_bin_index": int(bi), "time_ms": float(payload["times_ms"][bi]),
                       "unique_neural_input_id": c * len(payload["times_ms"]) + int(bi)}
                row.update(dict(zip(fields[-8:], map(float, values))))
                writer.writerow(row); count += 1
    return {"path": str(output_gz.resolve()), "n_real_trial_time_predictions": count,
            "refit": False, "storage_expansion_only": True}


def export_trial_scores(score_path, output_gz, *, trial_ids=None):
    """Export complete saved per-trial self scores; n=0 epochs remain explicit."""
    with np.load(score_path, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    wanted = None if trial_ids is None else set(map(str, trial_ids))
    identity = {key: data[key].item() for key in ("animal", "iteration", "representation", "data_mode", "causal_status")}
    output_gz = Path(output_gz)
    output_gz.parent.mkdir(parents=True, exist_ok=True)
    fields = [*identity, "trial_index", "trial_id", "session_id", "condition_id", "condition_index", "epoch", "head", "target", "coordinate", "n_bins", *data["metric_names"]]
    count = 0
    with gzip.open(output_gz, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, trial in enumerate(data["trial_ids"]):
            if wanted is not None and str(trial) not in wanted:
                continue
            for ei, epoch in enumerate(data["epoch_names"]):
                for h, head in enumerate(data["head_names"]):
                    for coordinate, axis in enumerate(data["coordinate_names"]):
                        row = {**identity, "trial_index": int(data["test_trial_indices"][i]), "trial_id": str(trial),
                               "session_id": str(data["session_ids"][i]), "condition_id": int(data["condition_ids"][i]),
                               "condition_index": int(data["condition_indices"][i]), "epoch": str(epoch), "head": str(head),
                               "target": str(data["target_names"][h]), "coordinate": str(axis), "n_bins": int(data["n_bins"][i, ei])}
                        row.update(zip(data["metric_names"], data["scores"][i, ei, h, coordinate]))
                        writer.writerow(row)
                        count += 1
    return {"path": str(output_gz.resolve()), "n_per_trial_score_rows": count, "refit": False}


def main():
    parser = argparse.ArgumentParser(description="Export saved held-out predictions to real-trial rows; never fit models")
    parser.add_argument("--payload-npz", help="Compact labels: objective_xy,behavior_xy,condition_ids,times_ms,common_mask,epoch_* arrays")
    parser.add_argument("--trials-csv")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prediction-npz")
    inputs.add_argument("--scores-npz")
    parser.add_argument("--output-gz", required=True)
    parser.add_argument("--objective-axis", choices=("condition", "trial"), default="condition")
    parser.add_argument("--trial-id", action="append")
    args = parser.parse_args()
    if args.scores_npz:
        print(json.dumps(export_trial_scores(args.scores_npz, args.output_gz, trial_ids=args.trial_id), indent=2))
        return
    if not args.payload_npz or not args.trials_csv:
        parser.error("Prediction export requires --payload-npz and --trials-csv")
    with np.load(args.payload_npz, allow_pickle=False) as saved:
        payload = {key: saved[key].copy() for key in ("objective_xy", "behavior_xy", "condition_ids", "times_ms", "common_mask")}
        payload["epoch_masks"] = {name: saved[f"epoch_{name}"].copy() for name in EPOCHS if f"epoch_{name}" in saved}
    payload.update(trial_table=pd.read_csv(args.trials_csv, dtype={"trial_id": str, "session_id": str}), objective_axis=args.objective_axis, data_mode=MODE)
    print(json.dumps(export_trial_predictions(payload, args.prediction_npz, args.output_gz, trial_ids=args.trial_id), indent=2))


if __name__ == "__main__":
    main()
