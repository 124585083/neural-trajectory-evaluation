"""Locked supplementary diagnostics of the completed Sensorium GPFA assay.

Auxiliary fits never replace the main model. The sampling plan reproduces the
main repeat/null RNG stream and is fixed before new test scores are calculated.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from .supplementary_common import load_context, paired_trajectory_intervals, write_manifest
from .traditional import condition_means
from .unified_analysis import PRIMARY, TRAJECTORY, LOWER, _digest, _json, _mean, _trajectory_scores
from .unified_contract import canonical_digest, file_sha256

NULL_SPECS = [(s, s, None) for s in
              ("condition_shuffle", "circular_shift", "time_reversal", "independent_neuron_shift")]
NULL_SPECS += [(f"block_shuffle_{k}", "block_shuffle", k) for k in (4, 8, 16, 32)]


def sampling_plan(counts, *, n_times=250, n_neurons=512, n_splits=200, seed=42):
    """Save reproducible disjoint repeat memberships and existing null RNG states.

    The dummy array advances only the randomization stream. Its contents never
    enter a score. Prefixes before sorting provide uniform nested k=1/2 sides.
    """
    from trajectory_reliability.reliability import _null_response
    rng = np.random.default_rng(seed)
    dummy = np.zeros((len(counts), n_times, n_neurons), dtype=np.int8)
    plan = []
    for split in range(n_splits):
        halves = []
        for count in counts:
            order = rng.permutation(count)
            k = count // 2
            if k < 2:
                raise ValueError("At least four repeats per condition are needed for the locked comparison")
            halves.append({"a": order[:k].tolist(), "b": order[k:2*k].tolist(),
                           "unused": order[2*k:].tolist()})
        states = {}
        for label, name, block in NULL_SPECS:
            states[label] = copy.deepcopy(rng.bit_generator.state)
            _null_response(dummy, name, rng, block)
        plan.append({"split": split, "halves": halves, "null_rng_states": states})
    return plan


def verify_plan(plan, bundle, original):
    """Reject a sampling plan that fails to reproduce any main balanced split."""
    conditions = np.unique(bundle["conditions"])
    for item in plan:
        rows = original[original.split == item["split"]]
        for condition, half in zip(conditions, item["halves"]):
            ids = np.asarray(bundle["dataset_indices"])[bundle["conditions"] == condition]
            for side, key in (("half1", "a"), ("half2", "b"), ("unused_odd_repeat", "unused")):
                expected = sorted(ids[np.asarray(half[key], dtype=int)].tolist())
                actual = sorted(rows[(rows.condition == condition) & (rows.half == side)].dataset_index.tolist())
                if actual != expected:
                    raise AssertionError("Supplementary sampling does not reproduce the frozen main split identities")


def training_indices(bundle, fraction, seed=42):
    train = np.flatnonzero(np.asarray(bundle["roles"]).astype(str) == "train")
    validation = np.flatnonzero(np.asarray(bundle["roles"]).astype(str) == "validation")
    if fraction < 1:
        train = train[np.sort(np.random.default_rng(seed).permutation(len(train))[:int(len(train)*fraction)])]
    return train, validation


def convergence_record(model):
    h = model.history
    converged = len(h) >= 2 and abs(h[-1]["improvement"]) <= model.tolerance * (1 + abs(h[-2]["log_likelihood"]))
    return {"em_iterations": len(h), "em_limit": model.max_em_iterations,
            "em_converged": bool(converged), "em_limit_reached": len(h) == model.max_em_iterations,
            "last_log_likelihood": h[-1]["log_likelihood"] if h else np.nan,
            "last_improvement": h[-1]["improvement"] if h else np.nan}


def _extra_source_hashes(context):
    pilot = context.manifest["session"]
    names = [f"development/{pilot}/aligned_predictions.npz",
             f"development/{pilot}/aligned_predictions.json",
             f"test/{pilot}/aligned_predictions.json",
             "gpfa/frozen_gpfa_identity.json", "gpfa/gpfa_selection.csv", "gpfa/gpfa_fit_history.csv"]
    names += ["evaluation/pilot/" + filename for filename in (
        "reliability_observed.csv", "reliability_nulls.csv", "reliability_summary.csv",
        "reliability_split_identities.csv", "condition_metrics.csv")]
    return {name: file_sha256(context.root / name) for name in names}


def verify_additive_guard(saved, current_sources, protocol):
    """A self-consistent replacement cache is still a different source."""
    if saved["supplementary_config_sha256"] != canonical_digest(protocol) or saved["sources"] != current_sources:
        raise ValueError("Supplementary development cache or copied source tables changed")


def _source_guard(context, protocol, out, work):
    """Bind additional sources without rewriting a completed stage identity.

    A guard introduced after the initial live computation is accepted only
    after actual numeric fit-input checks and byte comparisons of copied main
    tables. This performs no fitting, posterior inference, or test scoring.
    """
    sources = _extra_source_hashes(context)
    path = out / "source_identity.json"
    local = work / "source_identity.json"
    if path.exists():
        saved = json.loads(path.read_text())
        verify_additive_guard(saved, sources, protocol)
        if not local.exists() or json.loads(local.read_text()) != saved:
            raise ValueError("Public and local supplementary source guards differ")
        return saved
    bundle = context.development
    if _digest(np.asarray(bundle["neural"], dtype=np.float32)) != context.gpfa.identity["development_neural_digest"]:
        raise ValueError("Development responses no longer match the frozen main GPFA source")
    if not np.array_equal(bundle["encoder_precision"], context.gpfa.arrays["encoder_precision"]):
        raise ValueError("Development precision no longer matches the frozen main GPFA")
    main = out / "main"
    copied = {"reliability_observed.csv": "evaluation/pilot/reliability_observed.csv",
              "reliability_nulls.csv": "evaluation/pilot/reliability_nulls.csv",
              "condition_metrics.csv": "evaluation/pilot/condition_metrics.csv",
              "selection.csv": "gpfa/gpfa_selection.csv", "fit_history.csv": "gpfa/gpfa_fit_history.csv"}
    checked_tables = []
    for dest, source in copied.items():
        if (main / dest).exists():
            if file_sha256(main / dest) != sources[source]:
                raise ValueError(f"Copied main table changed: {dest}")
            checked_tables.append(source)
    raw = np.asarray(bundle["neural"], dtype=float) / bundle["encoder_precision"]
    checked_profiles = []
    for profile in protocol["profiles"]:
        if profile.get("reuse_main"):
            continue
        directory = work / profile["name"]
        identity_path = directory / "identity.json"
        if not identity_path.exists():
            if (out / profile["name"] / "completion.json").exists():
                raise ValueError("A completed profile lacks its fit provenance")
            continue
        saved = json.loads(identity_path.read_text())
        n = profile["neurons"]
        train, validation = training_indices(bundle, profile["train_fraction"], protocol["seed"])
        selected = raw[..., :n]
        std = selected[train].reshape(-1, n).std(axis=0, ddof=1)
        precision = 1 / np.maximum(std, .01*std.mean())
        obs = np.arange(0, raw.shape[1], profile["observation_step"])
        checks = {"fit_values_sha256": _digest(selected[train][:, obs]*precision),
                  "validation_values_sha256": _digest(selected[validation][:, obs]*precision),
                  "precision_sha256": _digest(precision),
                  "train_dataset_indices": bundle["dataset_indices"][train].tolist(),
                  "validation_dataset_indices": bundle["dataset_indices"][validation].tolist(),
                  "neuron_ids": bundle["neuron_ids"][:n].tolist(), "profile": profile,
                  "supplementary_config_sha256": canonical_digest(protocol)}
        if any(saved["input_identity"].get(k) != value for k,value in checks.items()):
            raise ValueError(f"Completed {profile['name']} fit has different development inputs")
        for filename, key in (("model.pkl", "model_sha256"), ("arrays.npz", "arrays_sha256")):
            if file_sha256(directory / filename) != saved[key]:
                raise ValueError(f"Auxiliary artifact changed: {profile['name']}/{filename}")
        checked_profiles.append(profile["name"])
    record = {"supplementary_config_sha256": canonical_digest(protocol), "sources": sources,
        "development_matches_main_fitted_neural_digest": True,
        "existing_auxiliary_numeric_fit_inputs_verified": checked_profiles,
        "existing_copied_main_tables_byte_verified": checked_tables,
        "verification": "Additive source binding; no stage identity rewrite, fitting, inference, or score recomputation"}
    _json(path, record); _json(local, record)
    return record


def _fit_profile(context, profile, protocol, work):
    """Training and selection depend only on the development bundle."""
    from trajectory_reliability.gpfa import GaussianProcessFactorAnalysis
    from trajectory_reliability.selection import _fit_and_score
    bundle = context.development
    n = profile["neurons"]
    train, validation = training_indices(bundle, profile["train_fraction"], protocol["seed"])
    raw = np.asarray(bundle["neural"], dtype=np.float64) / bundle["encoder_precision"]
    raw = raw[..., :n]
    std = raw[train].reshape(-1, n).std(axis=0, ddof=1)
    floor = .01 * float(std.mean())
    precision = 1 / np.maximum(std, floor)
    query = np.arange(raw.shape[1]) / float(context.config["data"]["frame_rate_hz"])
    obs = np.arange(0, len(query), profile["observation_step"])
    values, held = raw[train][:, obs] * precision, raw[validation][:, obs] * precision
    identity = {"supplementary_config_sha256": canonical_digest(protocol), "profile": profile,
        "main_sources": context.sources, "train_dataset_indices": bundle["dataset_indices"][train].tolist(),
        "validation_dataset_indices": bundle["dataset_indices"][validation].tolist(),
        "neuron_ids": bundle["neuron_ids"][:n].tolist(), "fit_values_sha256": _digest(values),
        "validation_values_sha256": _digest(held), "precision_sha256": _digest(precision),
        "fit_role": "train", "selection_role": "validation", "test_used_for_fit_or_selection": False}
    work.mkdir(parents=True, exist_ok=True)
    identity_file = work / "identity.json"
    if identity_file.exists():
        saved = json.loads(identity_file.read_text())
        if (saved["input_identity"] != identity or file_sha256(work / "model.pkl") != saved["model_sha256"]
                or file_sha256(work / "arrays.npz") != saved["arrays_sha256"]):
            raise ValueError("Auxiliary fit cannot be reused after a source or configuration change")
        model = GaussianProcessFactorAnalysis.load(work / "model.pkl")
        if model.parameter_digest() != saved["parameter_digest"]:
            raise ValueError("Auxiliary GPFA parameters changed")
        return model, precision, obs, query, saved
    cfg = protocol["fitting"]
    rows, histories = [], []
    models = {}
    for initial in cfg["selection_initial_lengthscales_seconds"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, scores = _fit_and_score(values, held, query[obs], profile["latent_dim"], initial,
                cfg["selection_em_iterations"], cfg["tolerance"], tuple(cfg["lengthscale_bounds_seconds"]),
                profile["fit_seed"])
        models[initial] = model
        rows.append({"initial_lengthscale_seconds": initial, "validation_nll_per_observation": float(scores.mean()),
            "validation_standard_error": float(scores.std(ddof=1)/np.sqrt(len(scores))),
            "warnings": "; ".join(str(w.message) for w in caught), **convergence_record(model)})
        histories.extend({"stage": "initialization", "initial_lengthscale_seconds": initial, **h} for h in model.history)
        pd.DataFrame(rows).to_csv(work / "selection.csv", index=False)
        pd.DataFrame(histories).to_csv(work / "fit_history.csv", index=False)
        print(f"{profile['name']}: initialization {initial} completed", flush=True)
    selected = min(rows, key=lambda r: r["validation_nll_per_observation"])["initial_lengthscale_seconds"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = GaussianProcessFactorAnalysis(latent_dim=profile["latent_dim"], times_seconds=query[obs],
            max_em_iterations=cfg["final_em_iterations"], tolerance=cfg["tolerance"],
            initial_lengthscale_seconds=selected, lengthscale_bounds_seconds=tuple(cfg["lengthscale_bounds_seconds"]),
            random_seed=profile["fit_seed"]).fit(values, split_name="train")
    model.save(work / "model.pkl")
    histories.extend({"stage": "final", "initial_lengthscale_seconds": selected, **h} for h in model.history)
    pd.DataFrame(histories).to_csv(work / "fit_history.csv", index=False)
    np.savez_compressed(work / "arrays.npz", precision=precision, observations=obs, query_times=query,
                        train_dataset_indices=bundle["dataset_indices"][train], neuron_ids=bundle["neuron_ids"][:n])
    record = {"input_identity": identity, "model_sha256": file_sha256(work / "model.pkl"),
        "arrays_sha256": file_sha256(work / "arrays.npz"), "parameter_digest": model.parameter_digest(),
        "selected_initial_lengthscale_seconds": selected, "precision_floor": floor,
        "warnings": "; ".join(str(w.message) for w in caught), **convergence_record(model),
        "lengthscales_seconds": {k: v.tolist() for k, v in model.lengthscales.items()}}
    _json(identity_file, record)
    return model, precision, obs, query, record


def reconstruction_scores(target, reconstructed, train_mean):
    """Observation reconstruction on one explicitly supplied response scale."""
    x, y = np.asarray(target, dtype=float), np.asarray(reconstructed, dtype=float)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Reconstruction requires identical finite support")
    axes = tuple(range(x.ndim-1))
    sse = np.sum((x-y)**2, axis=axes)
    sst = np.sum((x-x.mean(axis=axes, keepdims=True))**2, axis=axes)
    train_sst = np.sum((x-train_mean)**2, axis=axes)
    r2 = np.divide(sse, sst, out=np.full_like(sse, np.nan), where=sst > 1e-12)
    r2 = 1-r2
    valid = np.isfinite(r2)
    return {"population_r2_evaluation_mean": 1-float(sse.sum()/sst.sum()) if sst.sum() > 1e-12 else np.nan,
        "population_r2_train_intercept": 1-float(sse.sum()/train_sst.sum()) if train_sst.sum() > 1e-12 else np.nan,
        "mean_neuron_r2": _mean(r2), "median_neuron_r2": float(np.nanmedian(r2)) if valid.any() else np.nan,
        "p90_neuron_r2": float(np.nanquantile(r2, .9)) if valid.any() else np.nan,
        "positive_neuron_fraction": float(np.mean(r2[valid] > 0)) if valid.any() else np.nan,
        "defined_neurons": int(valid.sum()), "n_neurons": len(r2), "sse": float(sse.sum()),
        "sst_evaluation_mean": float(sst.sum()), "sst_train_intercept": float(train_sst.sum()),
        "rmse_gpfa_input_scale": float(np.sqrt(np.mean((x-y)**2))),
        "n_arrays": x.shape[0], "n_frames": x.shape[1], "n_values": x.size}, r2


def _reconstruction(bundle, model, precision, obs, query, profile, out):
    n = len(precision)
    rows, neurons = [], []
    for level, normalized in (("single_trial", bundle["neural"]),
            ("condition_average", condition_means(bundle["neural"], bundle["conditions"]))):
        scaled = np.asarray(normalized, dtype=float)[..., :n] / bundle["encoder_precision"][:, :n] * precision
        latent = model.transform_query(scaled[:, obs], query)
        restored = latent @ model.C.T + model.d
        for grid, indices in (("full_query", np.arange(len(query))), ("observation", obs)):
            scores, neuron_r2 = reconstruction_scores(scaled[:, indices], restored[:, indices], model.d)
            rows.append({"profile": profile, "level": level, "grid": grid, **scores})
            neurons.extend({"profile": profile, "level": level, "grid": grid,
                "neuron_id": str(neuron), "r2_evaluation_mean": r2}
                for neuron, r2 in zip(bundle["neuron_ids"][:n], neuron_r2))
    pd.DataFrame(rows).to_csv(out / "reconstruction.csv", index=False)
    pd.DataFrame(neurons).to_csv(out / "reconstruction_neurons.csv", index=False)


def _summarize_reliability(observed, nulls):
    rows = []
    for (metric, label), group in nulls.groupby(["metric", "null"]):
        a, b = group.observed.to_numpy(), group.value.to_numpy()
        valid = np.isfinite(a) & np.isfinite(b)
        delta = (a-b) * (-1 if metric in LOWER else 1)
        interval = np.quantile(a[valid], [.025, .975]) if valid.any() else [np.nan, np.nan]
        rows.append({"metric": metric, "null": label, "observed_mean": _mean(a[valid]),
            "observed_sd": float(np.std(a[valid], ddof=1)) if valid.sum()>1 else np.nan,
            "split_interval_low": interval[0], "split_interval_high": interval[1],
            "null_mean": _mean(b[valid]), "paired_superiority": float(np.mean(delta[valid]>0)) if valid.any() else np.nan,
            "split_null_failures": int(np.sum(delta[valid]<=0)), "undefined_splits": int((~valid).sum()),
            "splits": len(group), "interval_unit": "repeat_half_randomization_not_biological_samples"})
    return pd.DataFrame(rows)


def _reliability(bundle, model, precision, obs, query, plan, repeats, out):
    from trajectory_reliability.reliability import _null_response
    unique = np.unique(bundle["conditions"])
    groups = [bundle["neural"][bundle["conditions"] == c] for c in unique]
    n = len(precision)
    observed_rows, null_rows = [], []
    digest = model.parameter_digest()
    for item in plan:
        halves = [{side: np.sort(h[side] if repeats == "balanced_half" else h[side][:repeats])
                   for side in ("a", "b")} for h in item["halves"]]
        a, b = [np.stack([g[h[side]].mean(0) for g,h in zip(groups, halves)]) for side in ("a", "b")]
        # Keep 512 columns through the randomization so independent-neuron null
        # RNG consumption is identical for the nested 256-neuron profile.
        a = np.asarray(a, dtype=float) / bundle["encoder_precision"]
        b = np.asarray(b, dtype=float) / bundle["encoder_precision"]
        scale = np.ones(a.shape[-1]); scale[:n] = precision
        a *= scale; b *= scale
        inputs = [a[..., :n], b[..., :n]]
        for label, name, block in NULL_SPECS:
            rng = np.random.default_rng()
            rng.bit_generator.state = copy.deepcopy(item["null_rng_states"][label])
            inputs.append(_null_response(b, name, rng, block)[..., :n])
        batch = np.concatenate(inputs)[:, obs]
        values = model.transform_query(batch, query)
        chunks = np.split(values, len(inputs))
        rights = {"observed": chunks[1], **{spec[0]: value for spec,value in zip(NULL_SPECS, chunks[2:])}}
        metrics, support = _trajectory_scores(chunks[0], rights, 1/30)
        observed_rows.extend({"split": item["split"], "metric": m, "value": v, **support}
                             for m,v in metrics["observed"].items())
        null_rows.extend({"split": item["split"], "null": label, "metric": m, "value": v,
            "observed": metrics["observed"][m], **support} for label,_,_ in NULL_SPECS for m,v in metrics[label].items())
        if (item["split"]+1) % 25 == 0:
            print(f"{out.parent.name}/{out.name}: repeat split {item['split']+1}/{len(plan)}", flush=True)
    if model.parameter_digest() != digest:
        raise AssertionError("Supplementary inference mutated fitted GPFA parameters")
    observed, nulls = pd.DataFrame(observed_rows), pd.DataFrame(null_rows)
    observed.to_csv(out / "reliability_observed.csv", index=False)
    nulls.to_csv(out / "reliability_nulls.csv", index=False)
    _summarize_reliability(observed, nulls).to_csv(out / "reliability_summary.csv", index=False)


def _comparison(bundle, model, precision, obs, query, protocol, out):
    n = len(precision)
    latents = {}
    for name in ("neural", "static", "dynamic"):
        means = condition_means(bundle[name], bundle["conditions"])
        scaled = np.asarray(means, dtype=float)[..., :n] / bundle["encoder_precision"][:, :n] * precision
        latents[name] = model.transform_query(scaled[:, obs], query)
    result = paired_trajectory_intervals(latents["neural"], latents["static"], latents["dynamic"],
        seed=protocol["model_comparison"]["bootstrap_seed"],
        n_boot=protocol["model_comparison"]["bootstrap_samples"])
    result = result.rename(columns={"a": "static", "b": "dynamic", "b_minus_a": "dynamic_minus_static"})
    result["primary"] = result.metric.isin(PRIMARY)
    result.to_csv(out / "model_comparison.csv", index=False)
    rows = []
    for index, condition in enumerate(np.unique(bundle["conditions"])):
        scores, support = _trajectory_scores(latents["neural"][index:index+1],
            {k: latents[k][index:index+1] for k in ("static", "dynamic")}, 1/30)
        rows.extend({"condition": condition, "model": name, "metric": m, "value": v, **support}
            for name, values in scores.items() for m,v in values.items())
    pd.DataFrame(rows).to_csv(out / "condition_metrics.csv", index=False)


def _copy_main(context, out):
    source = context.root / "evaluation/pilot"
    for name in ("reliability_observed.csv", "reliability_nulls.csv", "condition_metrics.csv"):
        (out / name).write_bytes((source / name).read_bytes())
    observed, nulls = [pd.read_csv(out / name) for name in ("reliability_observed.csv", "reliability_nulls.csv")]
    _summarize_reliability(observed, nulls).to_csv(out / "reliability_summary.csv", index=False)
    table = pd.read_csv(source / "paired_bootstrap.csv")
    table = table[table.metric.isin(TRAJECTORY)].copy()
    table["primary"] = table.metric.isin(PRIMARY)
    table.to_csv(out / "model_comparison.csv", index=False)


def plot_repeat_averaging(out):
    """Render the retained repeat comparison without loading sensitivity data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    repeat = pd.read_csv(out / "repeat_averaging_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric in zip(axes, PRIMARY):
        rows = repeat[(repeat.metric == metric) & (repeat["null"] == "condition_shuffle")]
        ax.errorbar(np.arange(len(rows)), rows.observed_mean,
            yerr=[rows.observed_mean-rows.split_interval_low, rows.split_interval_high-rows.observed_mean], fmt="o", capsize=3)
        ax.set_xticks(np.arange(len(rows)), rows.repeat_level)
        ax.set_title(metric.replace("_", " ")); ax.set_ylabel("Split reliability (95% split interval)")
        ax.set_xlabel("Repeats per disjoint side")
    fig.savefig(out / "repeat_averaging.png", dpi=150); plt.close(fig)


def _figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_repeat_averaging(out)
    sensitivity = pd.read_csv(out / "profile_model_comparison.csv")
    profile_labels = {"main": "Main (4D)", "q8": "8 dimensions", "q16": "16 dimensions",
        "neurons256": "256 neurons", "observation_step1": "Observation step 1",
        "train50": "Half training set", "seed43": "Seed 43"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric in zip(axes, PRIMARY):
        rows = sensitivity[sensitivity.metric == metric]
        y = np.arange(len(rows))
        ax.errorbar(rows.dynamic_minus_static, y,
            xerr=[rows.dynamic_minus_static-rows.ci_low, rows.ci_high-rows.dynamic_minus_static], fmt="o", capsize=3)
        ax.axvline(0, color="gray", lw=1); ax.set_yticks(y, rows.profile.map(profile_labels))
        ax.set_title(metric.replace("_", " ")); ax.set_xlabel("Dynamic minus Static (95% condition interval)")
    fig.savefig(out / "profile_sensitivity.png", dpi=150); plt.close(fig)


def run(run_root, output_dir, config_path):
    """Run the fixed seven-profile supplement, retaining failures and provenance."""
    protocol = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if len(protocol["profiles"]) != 7 or protocol["reliability"]["splits"] != 200:
        raise ValueError("Use the fixed seven-profile, 200-split supplementary specification")
    out, work = Path(output_dir), Path(run_root) / "diagnostics/gpfa"
    out.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    frozen_config = work / "protocol.json"
    if frozen_config.exists() and json.loads(frozen_config.read_text()) != protocol:
        raise ValueError("The supplementary protocol is immutable once execution starts")
    _json(frozen_config, protocol)
    _json(out / "protocol.json", protocol)
    context = load_context(run_root, include_development=True)
    if canonical_digest(context.config) != protocol["main_config_sha256"]:
        raise ValueError("Main configuration differs from the fixed supplementary source")
    if context.gpfa.identity["gpfa_sha256"] != protocol["main_gpfa_sha256"]:
        raise ValueError("The supplementary protocol targets a different main GPFA")
    _source_guard(context, protocol, out, work)
    unique, counts = np.unique(context.test["conditions"], return_counts=True)
    plan = sampling_plan(counts, seed=protocol["seed"])
    original = pd.read_csv(context.root / "evaluation/pilot/reliability_split_identities.csv")
    verify_plan(plan, context.test, original)
    _json(work / "sampling_plan.json", plan)
    _json(out / "sampling_identity.json", {"seed": protocol["seed"], "splits": len(plan),
        "condition_ids": unique.tolist(), "repeat_counts": counts.tolist(),
        "sampling_plan_sha256": file_sha256(work / "sampling_plan.json"),
        "main_split_memberships_verified": True,
        "main_split_identity_sha256": file_sha256(context.root / "evaluation/pilot/reliability_split_identities.csv"),
        "array_order": "Same audited test-bundle trial order within each sorted condition"})
    records = []
    for profile in protocol["profiles"]:
        name = profile["name"]
        dest = out / name; dest.mkdir(exist_ok=True)
        complete = dest / "completion.json"
        stage_identity = _digest(protocol, context.sources, profile)
        if complete.exists():
            previous = json.loads(complete.read_text())
            if previous["stage_identity"] != stage_identity:
                raise ValueError("Completed diagnostic has different sources or settings")
            if any(file_sha256(dest / p) != h for p,h in previous["files"].items()):
                raise ValueError("Completed diagnostic results changed")
            records.append(previous["record"])
            continue
        start = time.monotonic()
        _json(work / "status.json", {"status": "running", "profile": name})
        print(f"Starting GPFA diagnostic profile {name}", flush=True)
        try:
            if profile.get("reuse_main"):
                model = context.gpfa.model
                a = context.gpfa.arrays
                precision, obs, query = a["precision"], a["observations"], a["query_times"]
                fit_record = {**context.gpfa.identity, **convergence_record(model), "main_reused": True}
                _copy_main(context, dest)
                (dest / "selection.csv").write_bytes((context.root / "gpfa/gpfa_selection.csv").read_bytes())
                (dest / "fit_history.csv").write_bytes((context.root / "gpfa/gpfa_fit_history.csv").read_bytes())
                for k in (1, 2):
                    repeat_dest = dest / f"k{k}"; repeat_dest.mkdir(exist_ok=True)
                    _reliability(context.test, model, precision, obs, query, plan, k, repeat_dest)
            else:
                model, precision, obs, query, fit_record = _fit_profile(context, profile, protocol, work / name)
                for filename in ("selection.csv", "fit_history.csv"):
                    (dest / filename).write_bytes((work / name / filename).read_bytes())
                _comparison(context.test, model, precision, obs, query, protocol, dest)
                _reliability(context.test, model, precision, obs, query, plan, "balanced_half", dest)
            _reconstruction(context.test, model, precision, obs, query, name, dest)
            _json(dest / "fit_identity.json", fit_record)
            record = {"profile": name, "status": "complete", "latent_dim": profile["latent_dim"],
                "neurons": profile["neurons"], "observation_step": profile["observation_step"],
                "train_fraction": profile["train_fraction"], "fit_seed": profile["fit_seed"],
                "parameter_digest": model.parameter_digest(), "elapsed_seconds": time.monotonic()-start,
                **convergence_record(model)}
            _json(complete, {"stage_identity": stage_identity, "record": record, "files": {
                str(p.relative_to(dest)).replace("\\", "/"): file_sha256(p)
                for p in dest.rglob("*") if p.is_file() and p.name != "completion.json"}})
        except Exception as error:
            record = {"profile": name, "status": "failed", "error": f"{type(error).__name__}: {error}",
                      "elapsed_seconds": time.monotonic()-start}
            _json(dest / "failure.json", record)
            print(f"{name}: {record['error']}", flush=True)
        records.append(record)
        pd.DataFrame(records).to_csv(out / "profile_status.csv", index=False)
        context.assert_unchanged()
    summaries, comparisons, reconstructions = [], [], []
    for record in records:
        name = record["profile"]
        if record["status"] != "complete":
            continue
        for filename, target in (("reliability_summary.csv", summaries),
                ("model_comparison.csv", comparisons), ("reconstruction.csv", reconstructions)):
            table = pd.read_csv(out / name / filename); table["profile"] = name; target.append(table)
    for tables, filename in ((summaries, "profile_reliability_summary.csv"),
            (comparisons, "profile_model_comparison.csv"), (reconstructions, "reconstruction_summary.csv")):
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(out / filename, index=False)
    repeats = []
    for label, relative in (("1", "main/k1"), ("2", "main/k2"), ("balanced 4/5", "main")):
        path = out / relative / "reliability_summary.csv"
        if path.exists():
            table = pd.read_csv(path); table["repeat_level"] = label; repeats.append(table)
    if repeats:
        pd.concat(repeats, ignore_index=True).to_csv(out / "repeat_averaging_summary.csv", index=False)
    pd.DataFrame(records).to_csv(out / "profile_status.csv", index=False)
    if comparisons and repeats:
        _figures(out)
    context.assert_unchanged()
    result = {"status": "complete" if all(r["status"] == "complete" for r in records) else "completed_with_failed_profiles",
        "main_sources": context.sources, "supplementary_config_sha256": canonical_digest(protocol),
        "profiles": records, "main_unchanged": True, "test_conditions": unique.tolist(),
        "inference": "offline full-window posterior", "scope": protocol["interpretation"],
        "reconstruction_denominator_equivalence": protocol["reconstruction"]["historical_equivalence"]}
    _json(out / "summary.json", result)
    _json(work / "status.json", {"status": result["status"]})
    write_manifest(out, {"analysis": "supplementary_gpfa_v1", "config_sha256": canonical_digest(protocol),
                         "implementation_sha256": file_sha256(Path(__file__)), "main_sources": context.sources})
    return result
