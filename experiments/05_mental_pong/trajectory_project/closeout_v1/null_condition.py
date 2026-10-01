"""B: fixed-head test-condition permutations on exactly paired time support.

No model fitting occurs. One fixed permutation is shared by representations,
heads and targets. Every shuffled score has a matched score on the identical
source/target/phase intersection. Randomization summaries have 1000 units, not
100000 independent experiments, and support-dependent tails are descriptive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "condition_endpoint_fa_gpfa_v1"
B = 1000
SEEDS = {"mahler": 314159, "perle": 314160}
REPS = ("FA50", "GPFA50")
EPOCHS = ("full", "visible", "hidden", "bounce", "no_bounce", "post_bounce")
HEADS = ("D_obj", "D_beh")
TARGETS = ("objective", "behavior")
METRICS = ("matched_r", "null_r", "paired_r", "matched_RMSE", "null_RMSE", "paired_RMSE",
           "matched_bias", "null_bias", "matched_amplitude_ratio", "null_amplitude_ratio")
MOMENTS = ("sy", "sp", "sy2", "sp2", "syp", "sse", "ymin", "ymax", "pmin", "pmax")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _record(path, role):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size, "role": role}


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def generate_mappings(valid_test_indices, n_conditions, n_randomizations, seed):
    """q outermost, original split innermost, ascending original source index."""
    rng = np.random.default_rng(seed)  # PCG64, explicitly checked and recorded.
    if type(rng.bit_generator).__name__ != "PCG64":
        raise AssertionError("The frozen generator is PCG64")
    mapping = np.full((n_randomizations, len(valid_test_indices), n_conditions), -1, np.int16)
    for q in range(n_randomizations):
        for r, valid in enumerate(valid_test_indices):
            valid = np.asarray(valid, int)
            if len(np.unique(valid)) != len(valid) or np.any(np.diff(valid) <= 0):
                raise ValueError("Permutation sources must be distinct ascending original indices")
            mapping[q, r, valid] = rng.permutation(valid)
    return mapping


def permutation_audit(mapping, valid_test_indices, animal):
    rows = []
    for r, valid in enumerate(valid_test_indices):
        valid = np.asarray(valid, int)
        observed = mapping[:, r, valid]
        if not all(np.array_equal(np.sort(row), valid) for row in observed):
            raise AssertionError("Permutation escaped its valid test-condition set")
        unique = len(np.unique(observed, axis=0))
        fixed = np.sum(observed == valid[None], axis=1)
        rows.append({"animal": animal, "split": r, "n_valid_test_conditions": len(valid),
                     "n_randomizations": len(mapping), "n_unique_permutations": unique,
                     "n_repeated_permutation_draws": len(mapping) - unique,
                     "mean_fixed_conditions": float(fixed.mean()), "minimum_fixed_conditions": int(fixed.min()),
                     "maximum_fixed_conditions": int(fixed.max()), "n_identity_permutations": int(np.sum(fixed == len(valid)))})
    return pd.DataFrame(rows)


def pair_moments(predictions, targets, phase_mask):
    """Precompute sufficient statistics for every source i and target j.

    predictions: [head,condition,time,xy], targets: [target,condition,time,xy].
    Returned modes are matched (label i) and shuffled (label j). Both use the
    same phase_mask[i,t] & phase_mask[j,t], without shifting the time index.
    """
    p, y = np.asarray(predictions, float), np.asarray(targets, float)
    mask = np.asarray(phase_mask, bool)
    n, nt = mask.shape
    if p.shape != (2, n, nt, 2) or y.shape != p.shape:
        raise ValueError("Pair moments require two heads, two targets, original common time, and xy")
    if not all(np.isfinite(a[:, mask]).all() for a in (p, y)):
        raise ValueError("A paired phase-valid prediction/target is missing")
    shared = mask[:, None, :] & mask[None, :, :]
    pair_counts = shared.sum(-1).astype(np.int16)
    mf = mask.astype(float)
    moments = np.empty((2, n, n, 2, 2, 2, len(MOMENTS)), float)
    for h in range(2):
        for coordinate in range(2):
            pv = np.where(mask, p[h, ..., coordinate], 0.)
            sp, sp2 = pv @ mf.T, (pv * pv) @ mf.T
            pmin = np.min(np.where(shared, pv[:, None, :], np.inf), axis=-1)
            pmax = np.max(np.where(shared, pv[:, None, :], -np.inf), axis=-1)
            for k in range(2):
                yv = np.where(mask, y[k, ..., coordinate], 0.)
                sy_matched, sy2_matched = yv @ mf.T, (yv * yv) @ mf.T
                yp_matched = (yv * pv) @ mf.T
                sy_null, sy2_null, yp_null = mf @ yv.T, mf @ (yv * yv).T, pv @ yv.T
                for mode, (sy, sy2, syp) in enumerate(((sy_matched, sy2_matched, yp_matched), (sy_null, sy2_null, yp_null))):
                    reference = yv[:, None, :] if mode == 0 else yv[None, :, :]
                    ymin = np.min(np.where(shared, reference, np.inf), axis=-1)
                    ymax = np.max(np.where(shared, reference, -np.inf), axis=-1)
                    # Direct squared errors avoid cancellation near a perfect fit.
                    sse = np.sum(np.where(shared, (pv[:, None, :] - reference) ** 2, 0.), axis=-1)
                    moments[mode, :, :, h, k, coordinate] = np.stack((sy, sp, sy2, sp2, syp, sse, ymin, ymax, pmin, pmax), axis=-1)
    return moments, pair_counts


def _finish_moments(sums, ymin, ymax, pmin, pmax, n):
    denominator = np.maximum(n, 1.)[:, None, None, None]
    sy, sp, sy2, sp2, syp, sse = np.moveaxis(sums, -1, 0)
    vy = np.maximum(sy2 - sy * sy / denominator, 0.)
    vp = np.maximum(sp2 - sp * sp / denominator, 0.)
    covariance = syp - sy * sp / denominator
    norm = np.sqrt(vy * vp)
    usable = (n[:, None, None, None] >= 3) & (ymax > ymin) & (pmax > pmin) & (norm > 0)
    r = np.divide(covariance, norm, out=np.full_like(norm, np.nan), where=usable)
    rmse = np.sqrt(sse / denominator)
    bias = (sp - sy) / denominator
    amplitude = np.sqrt(np.divide(vp, vy, out=np.full_like(vy, np.nan), where=vy > 0))
    for a in (rmse, bias, amplitude):
        a[n == 0] = np.nan
    return np.stack((np.clip(r, -1, 1), rmse, bias, amplitude), axis=-1)


def evaluate_pair_moments(moments, pair_counts, permutations):
    """permutations[q,i] is target j in local test-condition coordinates."""
    permutations = np.asarray(permutations, int)
    nq, ncond = permutations.shape
    if moments.shape[1:3] != (ncond, ncond) or pair_counts.shape != (ncond, ncond):
        raise ValueError("Permutation and pair table sizes disagree")
    flat = np.arange(ncond)[None] * ncond + permutations
    per_condition_counts = pair_counts.reshape(-1)[flat]
    n = per_condition_counts.sum(1)
    values = []
    for mode in range(2):
        selected = moments[mode].reshape(ncond * ncond, 2, 2, 2, len(MOMENTS))[flat]
        sums = selected[..., :6].sum(axis=1)
        values.append(_finish_moments(sums, selected[..., 6].min(1), selected[..., 7].max(1),
                                      selected[..., 8].min(1), selected[..., 9].max(1), n))
    matched, null = values
    scores = np.stack((matched[..., 0], null[..., 0], matched[..., 0] - null[..., 0],
                       matched[..., 1], null[..., 1], null[..., 1] - matched[..., 1],
                       matched[..., 2], null[..., 2], matched[..., 3], null[..., 3]), axis=-1)
    return scores, per_condition_counts


def _direct_score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not len(y):
        return np.full(4, np.nan)
    ym, pm = y - y.mean(), p - p.mean()
    norm = np.sqrt(np.dot(ym, ym) * np.dot(pm, pm))
    r = float(np.dot(ym, pm) / norm) if len(y) >= 3 and np.ptp(y) > 0 and np.ptp(p) > 0 and norm > 0 else np.nan
    return np.array([r, np.sqrt(np.mean((p - y) ** 2)), np.mean(p - y), np.std(p) / np.std(y) if np.std(y) > 0 else np.nan])


def direct_permutation_score(predictions, targets, mask, permutation):
    """Independent explicit bin loop for validation, never used for formal speed."""
    n = len(permutation)
    counts = np.zeros(n, np.int16)
    output = np.full((2, 2, 2, len(METRICS)), np.nan)
    shared_masks = []
    for i, j in enumerate(permutation):
        same_time = mask[i] & mask[j]
        shared_masks.append(same_time)
        counts[i] = same_time.sum()
    for h in range(2):
        for k in range(2):
            for coordinate in range(2):
                p, ym, yn = [], [], []
                for i, j in enumerate(permutation):
                    for t in np.flatnonzero(shared_masks[i]):
                        p.append(predictions[h, i, t, coordinate])
                        ym.append(targets[k, i, t, coordinate])
                        yn.append(targets[k, j, t, coordinate])
                matched, null = _direct_score(ym, p), _direct_score(yn, p)
                output[h, k, coordinate] = (matched[0], null[0], matched[0] - null[0], matched[1], null[1],
                                            null[1] - matched[1], matched[2], null[2], matched[3], null[3])
    return output, counts


def load_sources(source):
    source = Path(source)
    splits_path = source / "configs" / "condition_splits_100.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if len(splits) != 100 or [r["iteration"] for r in splits] != list(range(100)):
        raise ValueError("Use all original 100 condition splits")
    sources = [_record(splits_path, "original fixed 39/40 splits")]
    animals = {}
    for animal in SEEDS:
        label_path = source / "artifacts" / f"{animal}_condition_labels.npz"
        with np.load(label_path, allow_pickle=False) as saved:
            labels = {key: saved[key].copy() for key in saved.files}
        sources.append(_record(label_path, "unchanged current condition-average labels"))
        ids, times = labels["condition_ids"], labels["times_ms"]
        if ids.shape != (79,) or len(np.unique(ids)) != 79 or np.any(np.diff(times) <= 0):
            raise ValueError("Original identities/time support changed")
        targets = np.stack((labels["objective_xy"], labels["behavior_xy"]))
        predictions = {rep: np.full((100, 2, 79, len(times), 2), np.nan) for rep in REPS}
        support = np.zeros((100, 79, len(times)), bool)
        statuses = set()
        for r, split in enumerate(splits):
            train, test = np.array(split["train_indices"]), np.array(split["test_indices"])
            if len(train) != 39 or len(test) != 40 or set(train) & set(test) or len(set(train) | set(test)) != 79:
                raise ValueError("Source split no longer 39/40")
            np.testing.assert_array_equal(ids[test], split["test_condition_ids"])
            common = labels["common_mask"].copy() & labels["epoch_full"]
            common &= np.isfinite(targets).all(axis=(0, 3))
            common[train] = False
            supplied_support = None
            for rep in REPS:
                path = source / "readouts" / "predictions" / f"{animal}_{rep}_r{r:03d}_test_predictions.npz"
                model = source / "readouts" / "models" / f"{animal}_{rep}_r{r:03d}_ols.npz"
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["condition_ids"], ids)
                    np.testing.assert_array_equal(saved["times_ms"], times)
                    np.testing.assert_array_equal(saved["test_condition_indices"], test)
                    current_support = saved["test_input_support"].copy()
                    p = saved["predictions"].copy()
                    if np.isfinite(p[:, train]).any() or current_support[train].any():
                        raise ValueError("Training predictions entered source test output")
                    if supplied_support is not None:
                        np.testing.assert_array_equal(current_support, supplied_support)
                    supplied_support = current_support
                    common &= current_support & np.isfinite(p).all(axis=(0, 3))
                    statuses.add(str(saved["causal_status"].item()))
                    predictions[rep][r] = p
                sources.extend([_record(path, "fixed existing test prediction; no null fit"), _record(model, "unchanged original OLS head")])
            support[r] = common
        if len(statuses) != 1 or "raw_preprocessing_fail" not in next(iter(statuses)):
            raise ValueError("Preserve the actual published-input/raw-failure causal status")
        animals[animal] = {"labels": labels, "targets": targets, "predictions": predictions, "support": support,
                           "valid_test_indices": [np.flatnonzero(mask.any(1)) for mask in support], "causal_status": next(iter(statuses))}
    return splits, animals, sources


def freeze_mappings(root, splits, animals, sources):
    root = Path(root)
    for folder in ("configs", "artifacts/B", "results/B", "logs/B"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    code = _record(Path(__file__), "B implementation frozen before score generation")
    protocol = {"analysis": "B_fixed_head_condition_permutation", "B": B, "n_original_splits": 100,
                "seeds": SEEDS, "generator": "numpy.random.default_rng / PCG64", "numpy_version": np.__version__,
                "draw_order": "q outermost, split iteration 0..99 innermost; ascending original source condition index",
                "mapping": "Uniform permutation within valid test conditions; fixed points allowed; no retries or seed search",
                "support": "same original timestamp; source AND permuted target common validity AND both same evaluation phase",
                "paired_matched": "For each q/split/phase matched is recomputed on exactly its shuffled support",
                "primary_effects": {"paired_r": "matched_r-null_r", "paired_RMSE": "null_RMSE-matched_RMSE"},
                "aggregation": "each q assesses all 100 original splits; average finite split scores, preserve all missing counts; 1000 q summaries",
                "tail_interpretation": "empirical nonpositive paired-effect proportion; not an exact fixed-support randomization p-value",
                "heads_refitted": False, "representations_refitted": False, "behavior_trial_weights": False,
                "epochs": EPOCHS, "heads": HEADS, "targets": TARGETS, "coordinates": ["x", "y"], "score_metrics": METRICS,
                "x_interpretation": "Condition shuffling also changes reference x; x remains an implementation check, not behavioral-representation evidence",
                "raw_preprocessing": "fail", "provided_input_filtering": "source-audited pass on published inputs", "fit_provenance": "source-audited pass scope unchanged",
                "code": code, "sources": sources}
    protocol_path = root / "configs" / "b_fixed_head_protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != json.loads(json.dumps(protocol)):
            raise ValueError("Frozen B protocol/source/code changed; do not overwrite")
    else:
        _json(protocol_path, protocol)
    all_audits, map_records, outputs = [], [], {}
    for animal, data in animals.items():
        mapping = generate_mappings(data["valid_test_indices"], 79, B, SEEDS[animal])
        path = root / "artifacts/B" / f"{animal}_mappings.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(mapping, saved["mapping"])
        else:
            np.savez_compressed(path, mapping=mapping, condition_ids=data["labels"]["condition_ids"], seed=SEEDS[animal],
                                q=np.arange(B), split_iterations=np.arange(100), missing_value=-1,
                                axis_order="mapping[q,split,source_condition_index] -> target_condition_index; -1 outside valid test set")
        audit = permutation_audit(mapping, data["valid_test_indices"], animal)
        audit.to_csv(root / "results/B" / f"{animal}_mapping_audit.csv", index=False)
        all_audits.append(audit)
        map_records.append({**_record(path, "random mappings saved before all scoring"), "animal": animal,
                            "n_unique_q_mapping_bundles": int(len(np.unique(mapping.reshape(B, -1), axis=0))),
                            "n_repeated_q_mapping_bundles": int(B - len(np.unique(mapping.reshape(B, -1), axis=0)))})
        outputs[animal] = mapping
    pd.concat(all_audits, ignore_index=True).to_csv(root / "results/B/B_mapping_audit.csv", index=False)
    _json(root / "configs/b_mapping_lock.json", {"protocol": _record(protocol_path, "pre-score frozen protocol"), "mappings": map_records,
                                               "all_mappings_exist_before_scoring": True})
    return outputs


def _finite_mean(values, axis):
    finite = np.isfinite(values); count = finite.sum(axis)
    summed = np.where(finite, values, 0.).sum(axis)
    return np.divide(summed, count, out=np.full_like(summed, np.nan), where=count > 0), count


def _summaries(scores, observed, animal, rep, causal_status):
    means, finite = _finite_mean(scores, 1)
    index = pd.MultiIndex.from_product([range(B), EPOCHS, HEADS, TARGETS, ("x", "y")], names=["q", "epoch", "head", "target", "coordinate"])
    frame = pd.DataFrame(means.reshape(-1, len(METRICS)), index=index, columns=METRICS).reset_index()
    for i, metric in enumerate(METRICS):
        frame[metric + "_n_finite_splits"] = finite[..., i].reshape(-1)
    frame.insert(0, "representation", rep); frame.insert(0, "animal", animal)
    frame["causal_status"] = causal_status
    observed_mean, _ = _finite_mean(observed, 0)
    rows = []
    for ei, epoch in enumerate(EPOCHS):
        for h, head in enumerate(HEADS):
            for k, target in enumerate(TARGETS):
                for coordinate, axis in enumerate(("x", "y")):
                    identity = {"animal": animal, "representation": rep, "epoch": epoch, "head": head, "target": target,
                                "coordinate": axis, "is_self_target": h == k, "causal_status": causal_status,
                                "n_original_splits_assessed": 100, "B": B, "raw_preprocessing": "fail"}
                    for mi, metric in enumerate(METRICS):
                        a = means[:, ei, h, k, coordinate, mi]; a = a[np.isfinite(a)]
                        row = {**identity, "metric": metric, "n_finite_randomizations": len(a),
                               "mean": float(a.mean()) if len(a) else np.nan, "sd": float(a.std(ddof=0)) if len(a) else np.nan,
                               "median": float(np.median(a)) if len(a) else np.nan,
                               "q025": float(np.quantile(a, .025)) if len(a) else np.nan,
                               "q975": float(np.quantile(a, .975)) if len(a) else np.nan,
                               "minimum_finite_splits_per_q": int(finite[:, ei, h, k, coordinate, mi].min()),
                               "maximum_finite_splits_per_q": int(finite[:, ei, h, k, coordinate, mi].max()),
                               "n_nonpositive_paired_effect": int(np.sum(a <= 0)) if metric.startswith("paired_") else np.nan,
                               "empirical_nonpositive_effect_fraction": float(np.mean(a <= 0)) if metric.startswith("paired_") and len(a) else np.nan,
                               "tail_label": "empirical paired comparison, changing support; not exact permutation p"}
                        rows.append(row)
    return frame, pd.DataFrame(rows), observed_mean


def run_B(root=HERE, source=SOURCE):
    root, source = Path(root), Path(source)
    started = time.perf_counter()
    splits, animals, sources = load_sources(source)
    mappings = freeze_mappings(root, splits, animals, sources)  # All seeds/maps fixed first.
    completed = root / "results/B/B_validation.json"
    inventory = root / "results/B/B_outputs.json"
    if completed.exists() and inventory.exists():
        prior = json.loads(completed.read_text(encoding="utf-8"))
        if prior.get("status") != "B_COMPLETE" or prior.get("B") != B or prior.get("n_splits_per_q") != 100:
            raise ValueError("Existing B completion record has a different scope")
        records = json.loads(inventory.read_text(encoding="utf-8"))["outputs"]
        for item in records:
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Existing B output failed its saved checksum")
        print("B already complete: all mappings, sources, and output checksums verified; no scoring repeated", flush=True)
        return {"summary_path": str((root / "results/B/B_summary.csv").resolve()),
                "coverage_path": str((root / "results/B/B_coverage_summary.csv").resolve()),
                "validation_path": str(completed.resolve()), "output_records": records, "verified_reuse": True}
    summary_frames, observed_rows, validations, coverage_rows, output_records = [], [], [], [], []
    with threadpool_limits(limits=2):
        for animal, data in animals.items():
            mapping = mappings[animal]
            pair_counts_all = np.zeros((B, 100, len(EPOCHS), 79), np.int16)
            original_bins = np.zeros((100, len(EPOCHS)), np.int16)
            original_conditions = np.zeros_like(original_bins)
            for rep in REPS:
                scores = np.full((B, 100, len(EPOCHS), 2, 2, 2, len(METRICS)), np.nan)
                observed = np.full((100, len(EPOCHS), 2, 2, 2, 4), np.nan)
                for r in range(100):
                    valid = data["valid_test_indices"][r]
                    local_index = np.full(79, -1, int); local_index[valid] = np.arange(len(valid))
                    permutations = local_index[mapping[:, r, valid]]
                    p = data["predictions"][rep][r][:, valid]
                    y = data["targets"][:, valid]
                    for ei, epoch in enumerate(EPOCHS):
                        mask = data["support"][r, valid] & data["labels"]["epoch_" + epoch][valid]
                        moments, pair_counts = pair_moments(p, y, mask)
                        scored, per_condition = evaluate_pair_moments(moments, pair_counts, permutations)
                        scores[:, r, ei] = scored
                        if rep == REPS[0]:
                            pair_counts_all[:, r, ei, valid] = per_condition
                            original_bins[r, ei] = mask.sum()
                            original_conditions[r, ei] = mask.any(1).sum()
                        else:
                            np.testing.assert_array_equal(pair_counts_all[:, r, ei, valid], per_condition)
                        # Original whole-support observation is a separate context, not the null comparator.
                        for h in range(2):
                            for k in range(2):
                                for co in range(2):
                                    observed[r, ei, h, k, co] = _direct_score(y[k, ..., co][mask], p[h, ..., co][mask])
                        if r in (0, 37, 99):
                            for q in (0, 17, 999):
                                direct, counts = direct_permutation_score(p, y, mask, permutations[q])
                                np.testing.assert_array_equal(counts, per_condition[q])
                                np.testing.assert_allclose(scored[q], direct, atol=2e-10, rtol=2e-10, equal_nan=True)
                                difference = np.abs(scored[q] - direct)
                                validations.append({"animal": animal, "representation": rep, "split": r, "q": q, "epoch": epoch,
                                                    "maximum_finite_abs_difference": float(np.nanmax(difference)) if np.isfinite(difference).any() else 0.,
                                                    "matched_null_bins_identical": True})
                    if r % 20 == 0:
                        print(f"B {animal}/{rep}: split {r + 1}/100, all {B} permutations", flush=True)
                score_path = root / "artifacts/B" / f"{animal}_{rep}_scores.npz"
                np.savez_compressed(score_path, scores=scores, observed_full_support=observed,
                                    epoch_names=EPOCHS, head_names=HEADS, target_names=TARGETS, coordinate_names=["x", "y"],
                                    metric_names=METRICS, observed_metric_names=["r", "RMSE", "bias", "amplitude_ratio"],
                                    animal=animal, representation=rep, causal_status=data["causal_status"],
                                    axis_order="scores[q,split,epoch,head,target,coordinate,metric]",
                                    B=B, heads_refitted=False, mapping_path=str((root / "artifacts/B" / f"{animal}_mappings.npz").resolve()))
                qframe, summary, _ = _summaries(scores, observed, animal, rep, data["causal_status"])
                q_path = root / "results/B" / f"{animal}_{rep}_q_aggregate.csv.gz"
                qframe.to_csv(q_path, index=False, float_format="%.17g")
                summary_frames.append(summary)
                for r in range(100):
                    for ei, epoch in enumerate(EPOCHS):
                        for h, head in enumerate(HEADS):
                            for k, target in enumerate(TARGETS):
                                for co, coordinate in enumerate(("x", "y")):
                                    observed_rows.append({"animal": animal, "representation": rep, "split": r, "epoch": epoch,
                                                          "head": head, "target": target, "coordinate": coordinate,
                                                          "n_bins": int(original_bins[r, ei]), "n_conditions": int(original_conditions[r, ei]),
                                                          **dict(zip(("r", "RMSE", "bias", "amplitude_ratio"), observed[r, ei, h, k, co]))})
                output_records.extend([_record(score_path, "all 1000 x 100 paired-support matched/null scores"), _record(q_path, "1000 randomization units, each aggregating original splits")])
                del scores, observed, qframe
            n_bins = pair_counts_all.sum(-1, dtype=np.int32)
            n_conditions = (pair_counts_all > 0).sum(-1, dtype=np.int16)
            coverage_path = root / "artifacts/B" / f"{animal}_coverage.npz"
            np.savez_compressed(coverage_path, pair_bin_counts=pair_counts_all, n_bins=n_bins, n_conditions=n_conditions,
                                original_n_bins=original_bins, original_n_conditions=original_conditions,
                                condition_ids=data["labels"]["condition_ids"], epoch_names=EPOCHS,
                                axis_order="pair_bin_counts[q,split,epoch,source_condition]; all methods share this support")
            qcover = []
            for ei, epoch in enumerate(EPOCHS):
                fraction = np.divide(n_bins[:, :, ei], original_bins[None, :, ei],
                                     out=np.full((B, 100), np.nan), where=original_bins[None, :, ei] > 0)
                frac_mean, _ = _finite_mean(fraction, 1)
                for q in range(B):
                    qcover.append({"animal": animal, "q": q, "epoch": epoch,
                                   "shared_n_bins_mean": float(n_bins[q, :, ei].mean()),
                                   "shared_n_conditions_mean": float(n_conditions[q, :, ei].mean()),
                                   "shared_bins_fraction_mean": float(frac_mean[q]),
                                   "n_zero_bin_splits": int(np.sum(n_bins[q, :, ei] == 0)),
                                   "minimum_shared_bins_in_split": int(n_bins[q, :, ei].min()),
                                   "maximum_shared_bins_in_split": int(n_bins[q, :, ei].max())})
                for name, values in (("shared_n_bins", n_bins[..., ei]), ("shared_n_conditions", n_conditions[..., ei]), ("shared_bins_fraction", fraction)):
                    qmean, count = _finite_mean(values, 1)
                    finite = qmean[np.isfinite(qmean)]
                    coverage_rows.append({"animal": animal, "epoch": epoch, "metric": name,
                                          "mean_of_q_means": float(finite.mean()), "q025": float(np.quantile(finite, .025)),
                                          "q975": float(np.quantile(finite, .975)), "minimum_over_all_q_splits": float(np.nanmin(values)),
                                          "maximum_over_all_q_splits": float(np.nanmax(values)),
                                          "n_zero_support_q_splits": int(np.sum(n_bins[..., ei] == 0)),
                                          "interpretation": "support varies by mapping; not a fixed-support exact permutation test"})
            qcover_path = root / "results/B" / f"{animal}_q_coverage.csv.gz"
            pd.DataFrame(qcover).to_csv(qcover_path, index=False, float_format="%.17g")
            output_records.extend([_record(coverage_path, "every source condition shared-bin count for every q/split/phase"), _record(qcover_path, "randomization-level coverage distributions")])
            del pair_counts_all
    summary_path = root / "results/B/B_summary.csv"
    pd.concat(summary_frames, ignore_index=True).to_csv(summary_path, index=False, float_format="%.17g")
    observed_path = root / "results/B/B_observed_full_support.csv"
    pd.DataFrame(observed_rows).to_csv(observed_path, index=False, float_format="%.17g")
    coverage_summary_path = root / "results/B/B_coverage_summary.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_summary_path, index=False, float_format="%.17g")
    # Verify originals were neither refitted nor changed during B.
    for item in sources:
        if sha(item["path"]) != item["sha256"]:
            raise AssertionError("A read-only source changed during fixed-head null")
    validation = {"status": "B_COMPLETE", "B": B, "n_splits_per_q": 100, "animal_representation_pairs": 4,
                  "all_q_split_phase_combinations_saved": True, "all_mappings_frozen_before_scores": True,
                  "heads_refitted": False, "representations_refitted": False, "source_hashes_unchanged": True,
                  "shared_support_for_all_methods": True, "raw_preprocessing": "fail",
                  "direct_validation_cases": validations, "n_direct_validation_cases": len(validations),
                  "max_direct_score_abs_difference": max(row["maximum_finite_abs_difference"] for row in validations),
                  "elapsed_seconds": time.perf_counter() - started,
                  "tail_interpretation": "empirical reference proportions only; support changes and split repetitions share conditions"}
    validation_path = root / "results/B/B_validation.json"
    _json(validation_path, validation)
    output_records.extend(_record(path, "B result") for path in (summary_path, observed_path, coverage_summary_path, validation_path))
    _json(root / "results/B/B_outputs.json", {"outputs": output_records, "B_status": "B_COMPLETE", "overall_project_status_not_set_here": True})
    print(json.dumps({key: value for key, value in validation.items() if key != "direct_validation_cases"}, indent=2), flush=True)
    return {"summary_path": str(summary_path.resolve()), "coverage_path": str(coverage_summary_path.resolve()),
            "validation_path": str(validation_path.resolve()), "output_records": output_records}


def run(root=HERE, source=SOURCE):
    """Orchestrator-compatible alias; always uses the frozen B=1000 design."""
    return run_B(root, source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B fixed-head test condition null; all 1000 x 100, no fitting")
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    run_B(args.root, args.source)
