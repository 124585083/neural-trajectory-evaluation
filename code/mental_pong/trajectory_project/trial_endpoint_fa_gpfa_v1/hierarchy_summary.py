"""Describe between-unit heterogeneity of saved held-out own-target scores.

No prediction, label, model, bootstrap, confidence interval, or significance test
is generated here. Within-unit split means precede between-unit SD/quantiles.
These descriptive distributions do not replace the raw-trial-row main scores.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


METRICS = ("r_obj", "r_beh", "RMSE_obj", "RMSE_beh", "Delta_r", "Delta_RMSE")
REPS = ("FA50", "GPFA50")
EPOCHS = ("full", "visible", "hidden", "bounce", "no_bounce", "post_bounce")
MODE = "trial_labels_with_mean_neural"
GROUP = ("animal", "representation", "data_mode", "causal_status", "epoch", "coordinate")
INTERPRETATION = "between-unit descriptive heterogeneity of within-unit heldout-split means; not a confidence interval or independent neural replication"


def _iteration_filter(frame, iterations):
    if iterations is None:
        return frame
    return frame[frame.iteration.isin(iterations)].copy()


def _require_rounds(iterations, expected_rounds, animal, representation):
    actual = sorted(set(map(int, iterations)))
    if len(actual) != expected_rounds:
        raise ValueError(f"{animal}/{representation}: expected {expected_rounds} rounds, found {len(actual)}")
    if expected_rounds == 100 and actual != list(range(100)):
        raise ValueError("The formal summary must contain exactly original rounds 0..99")


def _heterogeneity(frame, unit_level):
    rows = []
    for key, group in frame.groupby(list(GROUP), sort=True, dropna=False):
        identity = dict(zip(GROUP, key))
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            row = {**identity, "unit_level": unit_level, "metric": metric,
                   "n_units_total": len(group), "n_units_with_support": int(group.n_supported_test_splits.gt(0).sum()),
                   "n_units_finite": len(values), "interpretation": INTERPRETATION,
                   "within_unit_aggregation": "arithmetic mean of saved heldout-split score; Delta paired before averaging",
                   "across_unit_weighting": "one unit one value; does not replace primary raw trial-time weighting",
                   "independent_neural_trials": 0}
            row.update({"mean": float(values.mean()), "sd": float(values.std(ddof=0)),
                        "median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
                        "q75": float(np.quantile(values, .75)), "min": float(values.min()), "max": float(values.max())}
                       if len(values) else {name: np.nan for name in ("mean", "sd", "median", "q25", "q75", "min", "max")})
            rows.append(row)
    return pd.DataFrame(rows)


def _trial_mean_scores(payload, root, representation, iterations, expected_rounds):
    table = payload["trial_table"].reset_index(drop=True)
    ntrial = len(table)
    shape = (ntrial, len(EPOCHS), 2, len(METRICS))
    sums = np.zeros(shape)
    counts = np.zeros(shape, np.uint16)
    tested = np.zeros(ntrial, np.uint16)
    supported = np.zeros((ntrial, len(EPOCHS)), np.uint16)
    nbins_sum = np.zeros((ntrial, len(EPOCHS)), np.uint32)
    found = []
    causal_status = None
    animal = str(payload["animal"])
    sources = []
    for path in sorted((root / "readouts" / "trial_scores").glob(f"{animal}_{representation}_r*_self_scores.npz")):
        # Filename filtering avoids decompressing excluded rounds in smoke tests.
        iteration = int(path.stem.split("_r")[-1].split("_")[0])
        if iterations is not None and iteration not in iterations:
            continue
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["iteration"]) != iteration or saved["animal"].item() != animal or saved["representation"].item() != representation:
                raise ValueError("Saved trial-score identity disagrees with its file")
            if saved["data_mode"].item() != MODE:
                raise ValueError("Do not mix actual-trial-neural and repeated-mean-neural scores")
            current_status = str(saved["causal_status"].item())
            if causal_status is not None and causal_status != current_status:
                raise ValueError("Causal status changed between rounds")
            causal_status = current_status
            np.testing.assert_array_equal(saved["epoch_names"], EPOCHS)
            np.testing.assert_array_equal(saved["head_names"], ["D_obj", "D_beh"])
            np.testing.assert_array_equal(saved["target_names"], ["objective", "behavior"])
            np.testing.assert_array_equal(saved["coordinate_names"], ["x", "y"])
            ix = saved["test_trial_indices"]
            if len(np.unique(ix)) != len(ix):
                raise ValueError("A real trial appears twice in one saved test split")
            np.testing.assert_array_equal(table.iloc[ix].trial_id.astype(str).to_numpy(), saved["trial_ids"])
            np.testing.assert_array_equal(table.iloc[ix].condition_id.to_numpy(int), saved["condition_ids"])
            np.testing.assert_array_equal(table.iloc[ix].session_id.astype(str).to_numpy(), saved["session_ids"])
            scores = saved["scores"]
            fields = list(saved["metric_names"])
            r = scores[..., fields.index("r")]
            rmse = scores[..., fields.index("RMSE")]
            values = np.stack([r[:, :, 0], r[:, :, 1], rmse[:, :, 0], rmse[:, :, 1],
                               r[:, :, 1] - r[:, :, 0], rmse[:, :, 0] - rmse[:, :, 1]], axis=-1)
            # Missing one member of a pair necessarily produces missing Delta.
            finite = np.isfinite(values)
            sums[ix] += np.where(finite, values, 0.)
            counts[ix] += finite.astype(np.uint16)
            tested[ix] += 1
            nbins = saved["n_bins"]
            supported[ix] += (nbins > 0).astype(np.uint16)
            nbins_sum[ix] += nbins.astype(np.uint32)
        found.append(iteration)
        sources.append(str(path.resolve()))
    _require_rounds(found, expected_rounds, animal, representation)
    if len(found) != len(set(found)):
        raise ValueError("Duplicate round score files")
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    del sums
    # Long rows retain every original trial identity, even no-support records.
    chunks = []
    identity_columns = [name for name in ("trial_id", "session_id", "condition_id", "condition_index", "source_row", "endpoint_valid", "candidate_valid") if name in table]
    identity = table[identity_columns].copy()
    identity.insert(0, "trial_index", np.arange(ntrial))
    for ei, epoch in enumerate(EPOCHS):
        for coordinate, axis in enumerate(("x", "y")):
            chunk = identity.copy()
            chunk.insert(0, "animal", animal)
            chunk["representation"], chunk["data_mode"], chunk["causal_status"] = representation, MODE, causal_status
            chunk["epoch"], chunk["coordinate"] = epoch, axis
            chunk["n_test_splits"] = tested
            chunk["n_supported_test_splits"] = supported[:, ei]
            chunk["n_bins_per_test_split_mean"] = np.divide(nbins_sum[:, ei], tested, out=np.full(ntrial, np.nan), where=tested > 0)
            for mi, metric in enumerate(METRICS):
                chunk[metric] = means[:, ei, coordinate, mi]
                chunk[metric + "_n_finite"] = counts[:, ei, coordinate, mi]
            chunks.append(chunk)
    return chunks, sources


def _paired_unit_means(frame, unit_level, iterations, expected_rounds):
    frame = _iteration_filter(frame, iterations)
    for (animal, representation), group in frame.groupby(["animal", "representation"]):
        _require_rounds(group.iteration, expected_rounds, animal, representation)
    key = "condition_id" if unit_level == "condition" else "session_id"
    join = [*GROUP, key, "iteration"]
    obj = frame[frame["head"].eq("D_obj") & frame.target.eq("objective")]
    beh = frame[frame["head"].eq("D_beh") & frame.target.eq("behavior")]
    kept = [*join, "r", "RMSE", "n_bins"]
    paired = obj[kept].merge(beh[kept], on=join, suffixes=("_obj", "_beh"), validate="one_to_one", how="outer", indicator=True)
    if not paired["_merge"].eq("both").all() or not paired.n_bins_obj.eq(paired.n_bins_beh).all():
        raise ValueError("Self heads do not have matching unit-round support")
    paired["Delta_r"] = paired.r_beh - paired.r_obj
    paired["Delta_RMSE"] = paired.RMSE_obj - paired.RMSE_beh
    rows = []
    for values, group in paired.groupby([*GROUP, key], sort=True, dropna=False):
        row = dict(zip([*GROUP, key], values))
        row.update(n_test_splits=int(group.iteration.nunique()), n_supported_test_splits=int(group.n_bins_obj.gt(0).sum()),
                   n_bins_per_test_split_mean=float(group.n_bins_obj.mean()))
        for metric in METRICS:
            v = group[metric].to_numpy(float); v = v[np.isfinite(v)]
            row[metric] = float(v.mean()) if len(v) else np.nan
            row[metric + "_n_finite"] = len(v)
        rows.append(row)
    return pd.DataFrame(rows)


def write_unit_heterogeneity(data, root, frames=None, *, iterations=None, expected_rounds=100, output_dir=None):
    """Read frozen scores; default requires all original 100 rounds.

    For a read-only real round-0 smoke test, use iterations=[0],
    expected_rounds=1, output_dir=<separate temporary output folder>.
    """
    root = Path(root)
    output = root / "results" if output_dir is None else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    iterations = None if iterations is None else set(map(int, iterations))
    hierarchy = []
    files, score_sources = [], []
    for animal, payload in sorted(data.items()):
        for rep in REPS:
            chunks, sources = _trial_mean_scores(payload, root, rep, iterations, expected_rounds)
            score_sources.extend(sources)
            path = output / f"{animal}_{rep}_trial_oof_score_summary.csv.gz"
            # One gzip stream prevents appending ambiguous headers/members.
            import gzip
            with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
                for i, chunk in enumerate(chunks):
                    chunk.to_csv(stream, index=False, header=i == 0, float_format="%.17g")
                    hierarchy.append(_heterogeneity(chunk, "trial"))
            files.append(str(path.resolve()))
            del chunks
    for unit in ("condition", "session"):
        if frames is not None:
            source = frames[unit]
        else:
            paths = sorted((root / "readouts" / "round_results").glob(f"*_{unit}.csv"))
            if iterations is not None:
                paths = [path for path in paths if int(path.stem.split("_r")[-1].split("_")[0]) in iterations]
            if not paths:
                raise ValueError(f"No saved {unit} scores")
            source = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        source = source[source.animal.isin(data.keys())].copy()
        if set(zip(source.animal, source.representation)) != {(a, r) for a in data for r in REPS}:
            raise ValueError(f"Missing animal/representation in {unit} source tables")
        means = _paired_unit_means(source, unit, iterations, expected_rounds)
        hierarchy.append(_heterogeneity(means, unit))
        path = output / f"{unit}_oof_self_score_summary.csv.gz"
        means.to_csv(path, index=False, float_format="%.17g")
        files.append(str(path.resolve()))
    summary = pd.concat(hierarchy, ignore_index=True).sort_values([*GROUP, "unit_level", "metric"])
    path = output / "descriptive_unit_heterogeneity.csv"
    summary.to_csv(path, index=False, float_format="%.17g")
    files.append(str(path.resolve()))
    metadata = {"interpretation": INTERPRETATION, "expected_rounds": expected_rounds,
                "included_iterations": sorted(iterations) if iterations is not None else list(range(100)),
                "source": "existing real-trial and condition/session heldout scores; no models or predictions recomputed",
                "paired_differences": "computed within each split before finite-pair mean",
                "unit_sd_ddof": 0, "confidence_intervals": False, "significance_tests": False,
                "missing_scores": "remain NaN; finite counts explicit; zero support retains identity",
                "primary_main_scores_replaced": False, "data_mode": MODE, "independent_neural_trials": 0,
                "trial_score_sources": score_sources, "files": files}
    metadata_path = output / "unit_heterogeneity_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary, "files": [*files, str(metadata_path.resolve())], "metadata": metadata}
