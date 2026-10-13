"""One fixed prediction reversal, evaluated across matched metric families."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .supplementary_common import load_context, read_json, paired_trajectory_intervals, write_manifest
from .traditional import condition_means, linear_cka, correlation_rdm
from .unified_analysis import _all_scores, _conventional, _json
from .unified_contract import canonical_digest, file_sha256


def condition_pattern_scores(neural, predictions, conditions):
    """Time-averaged condition patterns in the declared response scale."""
    left = condition_means(np.asarray(neural, dtype=np.float64), conditions).mean(axis=1)
    upper = np.triu_indices(len(left), 1)
    a = correlation_rdm(left)[upper]
    scores, patterns = {}, {}
    for name, prediction in predictions.items():
        right = condition_means(np.asarray(prediction, dtype=np.float64), conditions).mean(axis=1)
        patterns[name] = right
        b = correlation_rdm(right)[upper]
        common = np.isfinite(a) & np.isfinite(b)
        rsa = (float(spearmanr(a[common], b[common]).statistic)
               if common.sum() >= 3 and np.std(a[common]) > 1e-12 and np.std(b[common]) > 1e-12 else np.nan)
        cka = (linear_cka(left, right) if np.linalg.norm(left-left.mean(0)) > 1e-12
               and np.linalg.norm(right-right.mean(0)) > 1e-12 else np.nan)
        scores[name] = {"cka_time_averaged_condition_pattern": cka,
                       "rsa_time_averaged_condition_pattern": rsa}
    return scores, patterns


def run(run_root, output_dir, config_path, *, cache_output_path=None):
    config = read_json(config_path)
    out = Path(output_dir)
    cache_path = Path(cache_output_path) if cache_output_path is not None else Path(run_root) / "diagnostics/reversal_predictions.npz"
    if (out / "manifest.json").exists():
        saved = read_json(out / "manifest.json")
        if saved["config_sha256"] != canonical_digest(config):
            raise ValueError("Existing reversal diagnostic belongs to a different configuration")
        for name, digest in saved["files"].items():
            if file_sha256(out / name) != digest:
                raise ValueError("Completed reversal output changed")
        root = Path(run_root)
        for name, digest in saved["source_artifacts"].items():
            if file_sha256(root / name) != digest:
                raise ValueError(f"Completed reversal source changed: {name}")
        if file_sha256(cache_path) != saved["reversal_cache_sha256"]:
            raise ValueError("Completed reversal prediction cache changed")
        return saved
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve incomplete reversal output for diagnosis")
    out.mkdir(parents=True, exist_ok=True)
    _json(out / "configuration.json", config)
    context = load_context(run_root)
    if context.lock["freeze_sha256"] != config["source_freeze_sha256"]:
        raise ValueError("Reversal must use the declared completed main run")
    test, frozen = context.test, context.frozen
    neural, conditions = test["neural"], test["conditions"]
    unique = np.unique(conditions)
    left = frozen.latent(condition_means(neural, conditions))
    main = pd.read_csv(context.root / "evaluation/pilot/metrics.csv")
    rows, local_rows, intervals, checks = [], [], [], []
    reversed_cache = {}
    for model in config["objects"]:
        original = test[model]
        reverse = original[:, ::-1].copy()
        reversed_cache[model] = reverse
        predictions = {"original": original, "reversed": reverse}
        rights = {key: frozen.latent(condition_means(value, conditions)) for key, value in predictions.items()}
        scores, support = _all_scores(neural, predictions, conditions, left, rights,
            1/context.config["data"]["frame_rate_hz"], context.config["traditional"]["temporal_rsa_stride"])
        patterns_scores, patterns = condition_pattern_scores(neural, predictions, conditions)
        pattern_error = float(np.max(np.abs(patterns["original"]-patterns["reversed"])))
        assert pattern_error <= config["pattern_absolute_tolerance"]
        for metric in patterns_scores["original"]:
            delta = abs(patterns_scores["original"][metric]-patterns_scores["reversed"][metric])
            assert np.isfinite(delta) and delta <= config["metric_absolute_tolerance"]
            checks.append({"model": model, "check": metric+"_invariance", "absolute_error": delta,
                           "tolerance": config["metric_absolute_tolerance"], "passed": True})
        checks.append({"model": model, "check": "condition_pattern_invariance", "absolute_error": pattern_error,
                       "tolerance": config["pattern_absolute_tolerance"], "passed": True})
        for metric, value in scores["original"].items():
            expected = main[(main.model == model) & (main.metric == metric)].value.item()
            error = abs(value-expected)
            assert error <= config["baseline_replay_absolute_tolerance"], (model, metric, error)
            checks.append({"model": model, "check": "main_replay_"+metric, "absolute_error": error,
                           "tolerance": config["baseline_replay_absolute_tolerance"], "passed": True})
        for operation in predictions:
            for metric, value in {**scores[operation], **patterns_scores[operation]}.items():
                rows.append({"model": model, "operation": operation, "metric": metric, "value": value,
                    "n_conditions": len(unique), "n_trials": len(neural), "role": "test",
                    "estimand": "pooled_score", **support})
        paired = paired_trajectory_intervals(left, rights["original"], rights["reversed"],
            seed=config["seed"], n_boot=config["bootstrap_samples"])
        paired.insert(0, "model", model)
        paired["contrast"] = "reversed_minus_original"
        intervals.append(paired)
        local_by_op = {op: [] for op in predictions}
        for condition in unique:
            mask = conditions == condition
            metrics, local_support = _conventional(neural[mask], {k: v[mask] for k,v in predictions.items()},
                conditions[mask], context.config["traditional"]["temporal_rsa_stride"])
            for operation, values in metrics.items():
                local_by_op[operation].append(values)
                for metric, value in values.items():
                    local_rows.append({"model": model, "operation": operation, "condition": str(condition),
                        "metric": metric, "value": value, "role": "test", **local_support})
        chosen = np.random.default_rng(config["seed"]).integers(0, len(unique),
                    size=(config["bootstrap_samples"], len(unique)))
        conventional_intervals = []
        for metric in local_by_op["original"][0]:
            a = np.array([v[metric] for v in local_by_op["original"]])
            b = np.array([v[metric] for v in local_by_op["reversed"]])
            valid = np.isfinite(a)&np.isfinite(b)
            deltas = np.where(valid,b-a,np.nan)
            counts = np.isfinite(deltas[chosen]).sum(axis=1)
            draws = np.divide(np.nansum(deltas[chosen],axis=1), counts,
                              out=np.full(len(chosen),np.nan), where=counts>0)
            finite = np.isfinite(draws)
            low,high = np.quantile(draws[finite],[.025,.975]) if finite.any() else [np.nan,np.nan]
            conventional_intervals.append({"model":model,"metric":metric,"a":float(a[valid].mean()) if valid.any() else np.nan,
                "b":float(b[valid].mean()) if valid.any() else np.nan,
                "b_minus_a":float(deltas[valid].mean()) if valid.any() else np.nan,
                "ci_low":low,"ci_high":high,"valid_bootstraps":int(finite.sum()),
                "undefined_bootstraps":int((~finite).sum()),"n_conditions":len(unique),
                "contrast":"reversed_minus_original","resampling_unit":"whole_movie_condition",
                "estimand":"equal_weight_mean_paired_condition_score"})
        intervals.append(pd.DataFrame(conventional_intervals))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        raise FileExistsError("The deterministic reversal cache already exists")
    np.savez_compressed(cache_path, **reversed_cache, conditions=conditions, frame_indices=test["frame_indices"])
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(local_rows).to_csv(out / "condition_metrics.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(out / "paired_intervals.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "numerical_checks.csv", index=False)
    context.assert_unchanged()
    result = {"status":"complete","protocol":config["protocol"],"config_sha256":canonical_digest(config),
        "source_freeze_sha256":context.lock["freeze_sha256"],"source_artifacts":context.sources,
        "reversal_cache_sha256":file_sha256(cache_path),"inference":"unchanged offline primary GPFA",
        "deterministic_transformations_per_model":1,"main_artifacts_unchanged":True,
        "scope":"Supplementary prediction-order diagnostic; no independent-information claim"}
    _json(out / "summary.json",result)
    write_manifest(out,result)
    return result
