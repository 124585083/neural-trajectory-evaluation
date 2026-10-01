"""Read-only descriptive closeout of fixed condition-mean labels and saved OOF results.

No model fitting, neural re-averaging, hypothesis selection, or old-file writes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
CURRENT = PROJECT / "condition_endpoint_fa_gpfa_v1"
OUT = ROOT / "results" / "descriptive"
FIG = ROOT / "figures" / "descriptive"
ANIMALS = ("mahler", "perle")
REPS = ("FA50", "GPFA50")
EPOCHS = ("full", "visible", "hidden", "bounce", "no_bounce", "post_bounce")
CASES = (55062, 241919)  # User-specified posthoc cases; never score-selected.
SOURCES = {}


def digest(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    SOURCES[str(p)] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
    return h.hexdigest()


def read_csv(path, **kwargs):
    digest(path)
    return pd.read_csv(path, **kwargs)


def read_npz(path):
    digest(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def save_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer, np.bool_)):
            return x.item()
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        return x
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8")


def endpoint_statistics(e):
    """Population variance: exact sign rule is separate from numeric validation."""
    e = np.asarray(e, dtype=np.float64)
    assert e.ndim == 1 and len(e) and np.isfinite(e).all()
    mean = float(e.mean())
    mabs = float(np.abs(e).mean())
    variance = float(np.mean((e - mean) ** 2))
    msq = float(np.mean(e ** 2))
    stats = {
        "n_members": len(e), "n_up": int((e > 0).sum()),
        "n_down": int((e < 0).sum()), "n_exact_zero": int((e == 0).sum()),
        "both_signs_present": bool((e > 0).any() and (e < 0).any()),
        "mean_error": mean, "abs_mean_error": abs(mean), "mean_abs_error": mabs,
        "cancellation_fraction": 1 - abs(mean) / mabs if mabs > 0 else np.nan,
        "mean_square_error": msq, "square_mean_error": mean ** 2,
        "within_variance_ddof0": variance, "within_sd_ddof0": np.sqrt(variance),
        "variance_identity_residual": msq - mean ** 2 - variance,
        "error_min": float(e.min()), "error_max": float(e.max()),
        "error_q25": float(np.quantile(e, .25)), "error_median": float(np.median(e)),
        "error_q75": float(np.quantile(e, .75)),
    }
    for name, group in (("up", e[e > 0]), ("down", e[e < 0]), ("exact_zero", e[e == 0])):
        stats[f"{name}_mean_error"] = float(group.mean()) if len(group) else np.nan
        stats[f"{name}_variance_ddof0"] = float(group.var(ddof=0)) if len(group) else np.nan
        stats[f"{name}_sd_ddof0"] = float(group.std(ddof=0)) if len(group) else np.nan
    return stats


def markdown_table(frame, digits=4):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}f}" if np.isfinite(v) else "NA"
        return str(v).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def mean_audit():
    membership_path = CURRENT / "sources/source_trial_membership.json"
    digest(membership_path)
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    geometry = read_csv(CURRENT / "results/condition_geometry.csv")
    coverage = read_csv(CURRENT / "results/condition_label_coverage.csv")
    current_scores = read_csv(CURRENT / "results/condition_self_comparison.csv")
    main_scores = read_csv(CURRENT / "results/self_reconstruction_main_table.csv")
    rows, members, paths = [], [], []
    labels = {}
    mean_differences = []
    for animal in ANIMALS:
        item = membership[animal]
        record_path = Path(item["source_records"]["path"])
        assert digest(record_path) == item["source_records"]["sha256"]
        records = pd.read_csv(record_path)
        z = labels[animal] = read_npz(CURRENT / f"artifacts/{animal}_condition_labels.npz")
        used = []
        for ci, cid in enumerate(z["condition_ids"]):
            cid = int(cid)
            g = geometry.query("animal == @animal and condition_id == @cid").iloc[0]
            cov = coverage.query("animal == @animal and condition_id == @cid").iloc[0]
            ix = np.asarray(item["trial_indices_by_condition"][str(cid)], dtype=int)
            assert len(ix) == len(set(ix.tolist())) == int(z["n_behavior_trials"][ci])
            assert len(ix) == int(cov.n_behavior_trials_in_mean)
            row = dict(animal=animal, condition_id=cid, condition_index=ci,
                       bounce_class=g.bounce_class, geometry_valid=bool(g.geometry_valid),
                       invalid_reason=g.invalid_reason, n_members=len(ix),
                       n_up=0, n_down=0, n_exact_zero=0, both_signs_present=False,
                       objective_end_y=g.objective_end_y, mean_endpoint=z["mean_endpoint"][ci])
            if len(ix):
                r = records.iloc[ix].copy()
                assert (r.condition_id.to_numpy() == cid).all()
                assert r.endpoint_valid.all() and r.candidate_valid.all()
                assert r.trial_id.is_unique
                endpoint = r.paddle_y.to_numpy(dtype=float)
                assert np.isfinite(endpoint).all() and np.isfinite(g.objective_end_y)
                err = endpoint - float(g.objective_end_y)
                row.update(endpoint_statistics(err))
                row["condition_mean_endpoint_error"] = float(z["mean_endpoint"][ci] - g.objective_end_y)
                row["mean_endpoint_replay_abs_difference"] = abs(endpoint.mean() - z["mean_endpoint"][ci])
                mean_differences.append(row["mean_endpoint_replay_abs_difference"])
                assert row["mean_endpoint_replay_abs_difference"] < 1e-12
                assert abs(row["variance_identity_residual"]) < 1e-10
                m = r[["animal", "condition_id", "trial_id", "session_id", "source_row", "success", "failure", "endpoint_time_status"]].copy()
                m["source_records_iloc"] = ix
                m["endpoint"] = endpoint
                m["objective_end_y"] = g.objective_end_y
                m["endpoint_error"] = err
                m["sign_exact"] = np.sign(err).astype(int)
                members.append(m)
                used.extend(ix.tolist())
            else:
                assert cid == 59920 and not z["common_mask"][ci].any()
            rows.append(row)
            delta = z["behavior_xy"][ci, :, 1] - z["objective_xy"][ci, :, 1]
            for epoch in EPOCHS:
                mask = z["common_mask"][ci] & z[f"epoch_{epoch}"][ci]
                values = delta[mask]
                paths.append(dict(animal=animal, condition_id=cid, condition_index=ci, epoch=epoch,
                                  n_bins=int(mask.sum()), path_separation_RMS=float(np.sqrt(np.mean(values ** 2))) if len(values) else np.nan,
                                  path_separation_MAE=float(np.mean(abs(values))) if len(values) else np.nan,
                                  path_separation_max_abs=float(np.max(abs(values))) if len(values) else np.nan))
        assert len(used) == len(set(used)) == int(z["n_behavior_trials"].sum())
    detail = pd.DataFrame(rows)
    mem = pd.concat(members, ignore_index=True)
    path = pd.DataFrame(paths)
    joined = current_scores[current_scores.coordinate == "y"].merge(path, on=["animal", "condition_id", "epoch"], how="left", validate="many_to_one")
    for head in ("obj", "beh"):
        joined[f"path_RMS_over_RMSE_{head}"] = joined.path_separation_RMS / joined[f"RMSE_{head}"]
    detail.to_csv(OUT / "endpoint_condition_audit.csv", index=False)
    mem.to_csv(OUT / "endpoint_member_errors.csv.gz", index=False, compression="gzip")
    joined.to_csv(OUT / "path_separation_vs_error.csv", index=False)
    summary = {
        "sign_zero_rule": "exact floating zero; no near-correct threshold inferred",
        "variance": "population variance ddof=0; numerical identity tolerance 1e-10 only",
        "n_condition_identities_per_animal": 79,
        "unknown_condition_id": 59920,
        "membership_replay_max_abs_difference": max(mean_differences),
        "variance_identity_max_abs_residual": float(detail.variance_identity_residual.abs().max()),
        "raw_preprocessing": "fail", "neural_behavior_trial_membership_match": "unknown",
        "animals": {}, "path_separation": [],
    }
    for animal in ANIMALS:
        d = detail[(detail.animal == animal) & (detail.n_members > 0)]
        m = mem[mem.animal == animal]
        summary["animals"][animal] = {
            "n_members": len(m), "n_valid_conditions": len(d),
            "n_conditions_with_both_error_signs": int(d.both_signs_present.sum()),
            "n_up_records": int((m.endpoint_error > 0).sum()),
            "n_down_records": int((m.endpoint_error < 0).sum()),
            "n_exact_zero_records": int((m.endpoint_error == 0).sum()),
            "condition_equal_mean_abs_trial_error": float(d.mean_abs_error.mean()),
            "condition_equal_abs_mean_error": float(d.abs_mean_error.mean()),
            "condition_equal_mean_within_variance": float(d.within_variance_ddof0.mean()),
            "condition_equal_mean_squared_mean_error": float(d.square_mean_error.mean()),
            "condition_equal_mean_square_trial_error": float(d.mean_square_error.mean()),
            "condition_equal_mean_cancellation_fraction": float(d.cancellation_fraction.mean()),
            "median_condition_cancellation_fraction": float(d.cancellation_fraction.median()),
            "mean_endpoint_error_quantiles": {str(q): float(d.condition_mean_endpoint_error.quantile(q)) for q in [0, .25, .5, .75, 1]},
            "trial_weighted_mean_abs_trial_error": float(m.endpoint_error.abs().mean()),
            "trial_weighted_abs_condition_mean_error": float(np.average(d.abs_mean_error, weights=d.n_members)),
            "conditions_exact_mean_zero": int((d.condition_mean_endpoint_error == 0).sum()),
        }
        z = labels[animal]
        delta = z["behavior_xy"][..., 1] - z["objective_xy"][..., 1]
        for epoch in EPOCHS:
            mask = z["common_mask"] & z[f"epoch_{epoch}"]
            base = dict(animal=animal, epoch=epoch, n_bins=int(mask.sum()),
                        path_RMS_pooled_bins=float(np.sqrt(np.mean(delta[mask] ** 2))),
                        path_MAE_pooled_bins=float(np.mean(abs(delta[mask]))))
            for rep in REPS:
                score = main_scores.query("animal == @animal and representation == @rep and epoch == @epoch and coordinate == 'y'").iloc[0]
                summary["path_separation"].append({**base, "representation": rep,
                    "RMSE_obj_mean_of_100_splits": float(score.RMSE_obj),
                    "RMSE_beh_mean_of_100_splits": float(score.RMSE_beh)})
    save_json(OUT / "mean_cancellation_summary.json", summary)
    pd.DataFrame(summary["path_separation"]).to_csv(OUT / "path_separation_summary.csv", index=False)
    return labels, geometry, detail, mem, current_scores, main_scores, summary


def create_figures(labels, geometry, detail, scores):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ai, animal in enumerate(ANIMALS):
        d = detail[(detail.animal == animal) & (detail.n_members > 0)]
        ax = axes[ai, 0]
        ax.hist(d.condition_mean_endpoint_error, bins=16, color="#466995", alpha=.85)
        ax.axvline(0, color="black", lw=.7)
        ax.set(title=f"{animal.title()}: 78 condition-mean endpoint errors", xlabel="mean endpoint - objective endpoint (position units)", ylabel="Conditions")
        ax = axes[ai, 1]
        ax.scatter(d.mean_abs_error, d.abs_mean_error, color="#466995", s=25)
        lim = max(d.mean_abs_error.max(), d.abs_mean_error.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color=".5", lw=1)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="mean |individual endpoint error|", ylabel="|mean endpoint error|", title="Below identity: cancellation by averaging")
    fig.suptitle("Behavior members only: neural mean membership remains unknown\nExact zero sign rule; no behavior-group threshold or new neural analysis")
    fig.savefig(FIG / "endpoint_mean_cancellation.png", dpi=170)
    plt.close(fig)
    all_ids = labels[ANIMALS[0]]["condition_ids"]
    fig, axes = plt.subplots(4, 2, figsize=(19, 10), sharex=True, constrained_layout=True)
    for ri, (animal, rep) in enumerate((a, r) for a in ANIMALS for r in REPS):
        for ei, epoch in enumerate(("full", "hidden")):
            d = scores.query("animal == @animal and representation == @rep and epoch == @epoch and coordinate == 'y'").set_index("condition_id").reindex(all_ids)
            ax = axes[ri, ei]
            x = np.arange(len(all_ids))
            ax.bar(x, d.Delta_RMSE, width=.8, color=np.where(d.Delta_RMSE >= 0, "#007f75", "#a35279"))
            ax.axhline(0, color="black", lw=.7)
            ax.axvspan(int(np.where(all_ids == 59920)[0][0]) - .5, int(np.where(all_ids == 59920)[0][0]) + .5, color=".75")
            ax.set(title=f"{animal.title()} / {rep} / {epoch}", ylabel="RMSE obj - beh")
            ax2 = ax.twinx()
            ax2.plot(x, d.Delta_r, color="#20324b", marker=".", ms=2, lw=.5)
            ax2.set_ylabel("r beh - obj", color="#20324b")
            # Both zero levels must agree despite the two units.
            scale_rmse = max(float(d.Delta_RMSE.abs().max()) * 1.1, .01)
            scale_r = max(float(d.Delta_r.abs().max()) * 1.1, .001)
            ax.set_ylim(-scale_rmse, scale_rmse)
            ax2.set_ylim(-scale_r, scale_r)
            ax.set_xticks(x)
            ax.set_xticklabels(all_ids, rotation=90, fontsize=5.5)
    fig.suptitle("All 79 fixed condition identities: mean of held-out split scores, not scores of averaged predictions\nBars: positive favors candidate RMSE; navy line: positive favors candidate r. Gray = unresolved 59920.")
    fig.savefig(FIG / "all79_condition_heterogeneity.png", dpi=180)
    plt.close(fig)
    fixed = scores[(scores.condition_id.isin(CASES)) & (scores.coordinate == "y")].copy()
    fixed["selection_status"] = "user-specified posthoc; not confirmatory"
    fixed.to_csv(OUT / "fixed_posthoc_case_scores.csv", index=False)
    cross = read_csv(CURRENT / "results/condition_cross_summary.csv")
    cross[(cross.condition_id.isin(CASES)) & (cross.coordinate == "y")].to_csv(OUT / "fixed_posthoc_case_2x2.csv", index=False)
    case_shapes = []
    for animal in ANIMALS:
        z = labels[animal]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for ci_plot, cid in enumerate(CASES):
            ci = int(np.where(z["condition_ids"] == cid)[0][0])
            g = geometry.query("animal == @animal and condition_id == @cid").iloc[0]
            mask = z["common_mask"][ci]
            time = z["times_ms"] / 1000
            labels_y = [z["objective_xy"][ci, :, 1], z["behavior_xy"][ci, :, 1]]
            allcurves = [v[mask] for v in labels_y]
            caches = {}
            for rep in REPS:
                caches[rep] = read_npz(CURRENT / f"artifacts/{animal}_{rep}_condition_oof.npz")
                assert np.array_equal(caches[rep]["condition_ids"], z["condition_ids"])
                for h in range(2):
                    allcurves += [(caches[rep]["mean"][h, ci, :, 1] + sign * caches[rep]["sd"][h, ci, :, 1])[mask] for sign in [-1, 1]]
            values = np.concatenate(allcurves)
            lo, hi = np.nanmin(values), np.nanmax(values)
            pad = max(.6, (hi-lo) * .1)
            for ri, rep in enumerate(REPS):
                ax = axes[ci_plot, ri]
                cache = caches[rep]
                for epoch in EPOCHS:
                    support = mask & z[f"epoch_{epoch}"][ci]
                    pred_delta = cache["mean"][1, ci, :, 1] - cache["mean"][0, ci, :, 1]
                    label_delta = labels_y[1] - labels_y[0]
                    case_shapes.append(dict(animal=animal, representation=rep, condition_id=cid,
                        epoch=epoch, n_bins=int(support.sum()),
                        label_path_RMS=float(np.sqrt(np.mean(label_delta[support] ** 2))) if support.any() else np.nan,
                        OOF_mean_head_separation_RMS=float(np.sqrt(np.mean(pred_delta[support] ** 2))) if support.any() else np.nan,
                        interpretation="descriptive curve separation only; not per-split reconstruction score"))
                for h, color in enumerate(("#3069a5", "#d15c2f")):
                    mu, sd = cache["mean"][h, ci, :, 1], cache["sd"][h, ci, :, 1]
                    ax.fill_between(time, np.where(mask, mu-sd, np.nan), np.where(mask, mu+sd, np.nan), color=color, alpha=.13)
                    ax.plot(time, np.where(mask, mu, np.nan), color=color, lw=1.6)
                ax.plot(time, np.where(mask, labels_y[0], np.nan), color="black", lw=1.7)
                ax.plot(time, np.where(mask, labels_y[1], np.nan), color="#c23b74", lw=1.7, ls="--")
                ax.axvspan(g.occ_design_ms/1000, g.T_ms/1000, color=".7", alpha=.14)
                if np.isfinite(g.collision_time_ms):
                    ax.axvspan(g.collision_low_ms/1000, g.collision_high_ms/1000, color="#ddbb4f", alpha=.22)
                    ax.axvline(g.collision_time_ms/1000, color="#8a6c09", lw=.8, ls=":")
                ax.axvline(g.T_ms/1000, color=".5", lw=.8, ls=":")
                row = fixed.query("animal == @animal and representation == @rep and condition_id == @cid and epoch == 'full'").iloc[0]
                ax.set(title=f"Condition {cid} / {rep} / test memberships {cache['test_membership_count'][ci]}\nfull RMSE obj {row.RMSE_obj:.3f}, beh {row.RMSE_beh:.3f}; r {row.r_obj:.3f}, {row.r_beh:.3f}", xlabel="Time from trial-aligned origin (s; completed bin end)", ylabel="y (position units)", ylim=(lo-pad, hi+pad), xlim=(time[mask].min()-.05, g.T_ms/1000+.05))
        handles = [Line2D([], [], color="black", label="Objective ball path"), Line2D([], [], color="#3069a5", label="Objective reconstruction"), Line2D([], [], color="#c23b74", ls="--", label="Behavior-constrained candidate path"), Line2D([], [], color="#d15c2f", label="Candidate reconstruction")]
        fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
        fig.suptitle(f"{animal.title()} | fixed posthoc cases, not confirmatory\nShading on predictions: readout-split SD; gray: hidden; gold: estimated collision interval. T is estimated, exact feedback time unknown.", fontsize=11)
        fig.savefig(FIG / f"{animal}_fixed_posthoc_four_curves.png", dpi=180)
        plt.close(fig)
    pd.DataFrame(case_shapes).to_csv(OUT / "fixed_posthoc_case_curve_separation.csv", index=False)
    (OUT / "fixed_posthoc_case_interpretation.md").write_text(
        "# Current post hoc cases\n\n"
        "Conditions 55062 and 241919 are user-specified post hoc illustrations. They provide no independent confirmation. Scores average original per-split held-out results; the four curves show test-prediction means and readout-split SD for visualization.\n\n"
        + markdown_table(fixed[fixed.epoch.isin(["full", "hidden"])][["animal", "representation", "condition_id", "epoch", "r_obj", "r_beh", "RMSE_obj", "RMSE_beh", "Delta_r", "Delta_RMSE"]])
        + "\n\nCandidate own-target RMSE for 55062 improves in both animals and representations. Full-epoch correlation decreases in Mahler and increases in Perle. For 241919, candidate RMSE worsens in all four groups; Mahler correlation rises while Perle has no corresponding improvement. Preserve these metric differences rather than importing the old stopping-proxy result.\n\n"
        "The two readout curves are close relative to the overall error scale. Readout separation is smaller than label separation for both Mahler cases and Perle 55062. Perle 241919 has nearly coincident labels and a larger readout-mean separation. The phase-specific label_path_RMS and OOF_mean_head_separation_RMS comparison is in fixed_posthoc_case_curve_separation.csv; it does not replace per-split own-target error. Complete cross-scores are in fixed_posthoc_case_2x2.csv.\n\n"
        "raw_preprocessing=fail remains. Neural mean membership, strictly pre-feedback endpoint timing and collision-estimation limits remain unresolved.\n",
        encoding="utf-8")
    save_json(FIG / "descriptive_figure_manifest.json", {
        "posthoc_ids_fixed_by_request": list(CASES), "condition_order": [int(x) for x in all_ids],
        "prediction_source": "Existing test-only OOF mean/SD; current per-split scores unchanged",
        "raw_preprocessing": "fail", "shadow_interpretation": "readout split stability, not animal trial variation",
        "all79_atlases": [str(CURRENT / f"figures/{a}_all79_condition_atlas.pdf") for a in ANIMALS],
        "files": [str(p.relative_to(ROOT)) for p in FIG.glob("*.png")],
    })


def closeout_ledger_status(kind):
    """Require actual completed markers and compatible saved summaries, not a template flag."""
    protocol = json.loads((ROOT / "configs/closeout_protocol.json").read_text(encoding="utf-8"))
    repeats = protocol[kind]["repeats"]
    splits = protocol["splits"]["n_splits"]
    marker_path = ROOT / ("results/B/B_validation.json" if kind == "B" else "results/C/completed.json")
    summary_path = ROOT / ("results/B/B_summary.csv" if kind == "B" else "results/C/random_endpoint_summary.csv")
    if not marker_path.exists() or not summary_path.exists():
        return {"status": "PENDING", "actual_results": "Formal completion marker and complete summary are unavailable; a smoke test cannot substitute.", "support": "Awaiting complete formal outputs", "sources": "configs/closeout_protocol.json"}
    digest(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    summary = read_csv(summary_path)
    pair_count = summary[["animal", "representation"]].drop_duplicates().shape[0]
    expected_pairs = len(protocol["animals"]) * len(protocol["representations"])
    if kind == "B":
        complete = (marker.get("status") == "B_COMPLETE" and marker.get("B") == repeats
                    and marker.get("n_splits_per_q") == splits
                    and marker.get("animal_representation_pairs") == expected_pairs
                    and marker.get("heads_refitted") is False
                    and marker.get("representations_refitted") is False
                    and marker.get("shared_support_for_all_methods") is True
                    and marker.get("all_q_split_phase_combinations_saved") is True
                    and (summary.B == repeats).all()
                    and (summary.n_original_splits_assessed == splits).all()
                    and pair_count == expected_pairs)
        full = summary[(summary.epoch == "full") & (summary.coordinate == "y") & summary.is_self_target]
        parts = []
        for (animal, rep, head), d in full.groupby(["animal", "representation", "head"]):
            m = d.set_index("metric")["mean"]
            parts.append(f"{animal}/{rep}/{head}: matched→null r {m.matched_r:.4f}→{m.null_r:.4f}, RMSE {m.matched_RMSE:.4f}→{m.null_RMSE:.4f}")
        text = f"{repeats} random correspondences x {splits} original splits; all four groups/two heads/two targets saved. Full-epoch own-target cells on matched support: " + "; ".join(parts)
        support = "Correct correspondence exceeds mismatches on shared support, supporting task/condition specificity without isolating behavior."
    else:
        complete = (marker.get("completed") is True and marker.get("repeats_per_animal") == repeats
                    and marker.get("splits_per_repeat") == splits
                    and len(marker.get("branches", [])) == expected_pairs
                    and (summary.n_random_statistics == repeats).all()
                    and (summary.n_splits == splits).all() and pair_count == expected_pairs)
        for branch in marker.get("branches", []):
            p = Path(branch["path"])
            if not p.exists():
                complete = False
                continue
            complete = bool(complete and digest(p) == branch["sha256"])
            b = json.loads(p.read_text(encoding="utf-8"))
            complete = bool(complete and b.get("completed") is True
                            and b.get("random_repeats") == repeats and b.get("condition_splits") == splits
                            and b.get("OLS_calls") == splits and b.get("independent_random_y_refits") == repeats * splits
                            and b.get("new_representation_fit") is False and b.get("new_inference") is False)
        complete = bool(complete and digest(summary_path) == marker.get("summary", {}).get("sha256"))
        parts = []
        for (animal, rep), d in summary[summary.epoch == "full"].groupby(["animal", "representation"]):
            m = d.set_index("metric")
            parts.append(f"{animal}/{rep}: actual/random candidate means r {m.loc['r','behavior_mean']:.4f}/{m.loc['r','null_mean']:.4f}, RMSE {m.loc['RMSE','behavior_mean']:.4f}/{m.loc['RMSE','null_mean']:.4f}, candidate geometry-baseline skill {m.loc['skill','behavior_mean']:.4f}")
        positive = summary[(summary.metric == "skill") & (summary.behavior_mean > 0)]
        exceptions = "; ".join(f"{r.animal}/{r.representation}/{r.epoch}={r.behavior_mean:.4f}" for r in positive.itertuples()) or "none"
        text = (f"{repeats} random endpoint allocations x {splits} original splits with actual new OLS fits and fixed representations. Full epoch: "
                + "; ".join(parts) + ". Positive candidate geometry-baseline skill exceptions: " + exceptions + "; complete objective/candidate/random skill for all other phases is retained in the source table.")
        full_skill = summary[(summary.metric == "skill") & (summary.epoch == "full")]
        support = ("Actual task/candidate labels exceed this random-endpoint reference; that result does not establish candidate superiority over objective paths. "
                   + ("All four full-epoch objective and candidate skills are negative relative to the mean-endpoint geometry baseline. Phase-specific exceptions remain."
                      if ((full_skill.objective_mean < 0) & (full_skill.behavior_mean < 0)).all()
                      else "Geometry-baseline skill differs by group and phase; consult the complete table."))
    if marker.get("raw_preprocessing") != "fail" or not (summary.raw_preprocessing == "fail").all():
        complete = False
    return {"status": "EXECUTED" if complete else "PARTIAL",
            "actual_results": text if complete else "Completion markers, summaries or branch counts disagree; formal completion cannot be claimed.",
            "support": support if complete else "Formal output completeness requires checking",
            "sources": str(marker_path.relative_to(ROOT)).replace("\\", "/") + ";" + str(summary_path.relative_to(ROOT)).replace("\\", "/")}


def create_ledger(main_scores):
    sources = {
        "FA reference": PROJECT / "results/official_fa50_reference.csv",
        "representation selection": PROJECT / "results/representation_comparison.csv",
        "early preview": PROJECT / "condition_mean_behavior_preview_v1/preview_report.md",
        "early proxy": PROJECT / "condition_mean_behavior_stageB_v1/results/primary_summary.csv",
        "old segmented OLS": PROJECT / "all79_official_readout_v1/results/self_reconstruction_main_table.csv",
        "trial comparison": CURRENT / "results/previous_trial_version_comparison.csv",
    }
    for p in sources.values():
        assert p.exists(), p
        digest(p)
    ref = pd.read_csv(sources["FA reference"])
    rep = pd.read_csv(sources["representation selection"])
    proxy = pd.read_csv(sources["early proxy"])
    old = pd.read_csv(sources["old segmented OLS"])
    trial = pd.read_csv(sources["trial comparison"])
    def records(d, cols):
        return json.dumps(json.loads(d[cols].to_json(orient="records")), ensure_ascii=False)
    core = ["animal", "representation", "r_obj", "r_beh", "RMSE_obj", "RMSE_beh", "Delta_r", "Delta_RMSE"]
    rows = [
        dict(category=1, stage="Original reproduction and representation selection", status="EXECUTED", question="Assess the published reproduction and a usable representation tool", input_level="Separate animals; condition-mean neural halves; original FA and later GPFA protocols differ", labels="Objective x/y and cross-half neural responses", representation="Original FA50; tested AE/GPFA 32/64; historical frozen causal GPFA64", readout="Original OLS; validation-selected Ridge for tool evaluation", splits="Original 100 condition splits with 50% held out; tool selection 47/16/16", actual_results=records(ref, ["animal", "position_x_r", "position_y_r"])+"; GPFA64="+records(rep[(rep.latent_dim==64)&rep.method.str.contains("GPFA",case=False)], ["animal","representation_parameters","cross_response_r","position_x_r","position_y_r"]), support="Supports a tool choice for these released data, without a universal optimum or identical protocol", limitations="One million is a parameter cap, not parameter matching; causal filtering does not undo upstream cross-condition/time preprocessing", sources="../reports/01_representation_selection.md;../results/official_fa50_reference.csv;../results/representation_comparison.csv"),
        dict(category=2, stage="Early stopping/average-behavior proxy exploration", status="EXECUTED", question="Construct continuous candidates from stable mean segments and read out deviations", input_level="Condition-mean behavior and neural responses; exact shared membership unknown", labels="Stable-window targets and continuous segment updates; no individual stopped-paddle observation", representation="Frozen GPFA64 seed42", readout="Small-set Ridge with shared lambda and equal condition weighting", splits="Original 47/16/16; Mahler 3 and Perle 2 test conditions", actual_results=records(proxy[(proxy.split=="test")&(proxy.half=="half1")], ["animal","n_conditions","n_bins","mean_delta_r","mean_delta_mse","mean_zero_delta_mse","D_obj_to_y_obj_mse","D_beh_to_y_beh_mse"]), support="Completed physical preview and small-sample deviation analysis; not a substitute for current full-path own-target reconstruction", limitations="Low-ambiguity selection, small sample and short proxy windows; historical MSE/delta scores do not substitute for current E_OO/E_BB", sources="../condition_mean_behavior_preview_v1/preview_report.md;../condition_mean_behavior_stageB_v1/REPORT.md;../condition_mean_behavior_stageB_v1/results/primary_summary.csv"),
        dict(category=3, stage="Old all 79 segmented-candidate OLS", status="EXECUTED", question="Reconstruct each complete objective and segmented candidate path", input_level="Condition-mean half1; half2 uses the same readout as a supplement", labels="Physical reference until qualifying stable-proxy activation, then continuous segment updates", representation="Frozen GPFA64 fitted on the earlier 47 training conditions", readout="Intercept OLS, 65 coefficients per coordinate, one full-time mapping", splits="100 random 39/40 condition splits", actual_results=records(old[(old.half=="half1")&(old.epoch=="full")&(old.coordinate=="y")], ["animal","r_obj","r_beh","RMSE_obj","RMSE_beh","Delta_r","Delta_RMSE"]), support="Completed direct own-target evaluation with small full-path candidate improvements", limitations="Only 4.5%/5.6% of bins have active proxies; representation-fit conditions overlap some readout-test conditions; this is not the current endpoint-label result", sources="../all79_official_readout_v1/REPORT.md;../all79_official_readout_v1/results/self_reconstruction_main_table.csv"),
        dict(category=4, stage="Trial-endpoint label version", status="EXECUTED", question="Assess the readability of individual endpoint-constrained labels", input_level="Route B: each behavioral record reuses the same condition-mean neural input", labels="Individual endpoint connection after a single-collision anchor or from the no-collision starting point", representation="Per-split train39 FA50 and shared-single-learnable-RBF GPFA50", readout="Intercept OLS without explicit weights; record expansion increases the contribution of conditions with more trials", splits="Same 100 nominal 39/40 splits ; 78 valid conditions ; 59920 retained as unknown", actual_results=records(trial[(trial.epoch=="full")&(trial.coordinate=="y")], ["animal","representation","r_obj_trial_version","r_beh_trial_version","RMSE_obj_trial_version","RMSE_beh_trial_version"]), support="Evaluates repeated condition inputs against individual label sets; not single-trial neural validation", limitations="No independent paired neural trials; repeated inputs add neither neural information nor rank; raw_preprocessing=fail", sources="../trial_endpoint_fa_gpfa_v1/REPORT.md;../condition_endpoint_fa_gpfa_v1/results/previous_trial_version_comparison.csv"),
        dict(category=5, stage="Current condition-mean endpoint FA50/GPFA50", status="EXECUTED", question="Compare own-target reconstruction of two condition-mean paths", input_level="One condition-by-time row, half1 ; 7407/84873 behavioral members", labels="Fixed-membership mean endpoints, original anchors and shared time support", representation="Existing per-split FA50/GPFA50; no further representation fitting", readout="OLS 51 coefficients per coordinate, 102 per xy head; no trial-count weighting", splits="Same 100 nominal 39/40 splits ; 78 valid conditions / 3369 rows ; all 79 identities retained", actual_results=records(main_scores[(main_scores.epoch=="full")&(main_scores.coordinate=="y")],core), support="All four full-epoch candidate correlations are lower and RMSEs slightly higher; no overall candidate advantage", limitations="Within-condition target variance and extra trial-count weighting are removed together; not a one-factor ablation; raw_preprocessing=fail", sources="../condition_endpoint_fa_gpfa_v1/REPORT.md;../condition_endpoint_fa_gpfa_v1/results/self_reconstruction_main_table.csv;../condition_endpoint_fa_gpfa_v1/results/previous_trial_version_comparison.csv"),
        dict(category=6, stage="Closeout A: own-target and 2x2 replay", status="EXECUTED", question="Verify own-target and cross-target column identities", input_level="Existing current test-only predictions", labels="Current fixed objective and candidate labels", representation="Current per-split FA50/GPFA50 unchanged", readout="Rescore existing fixed OLS readouts without fitting", splits="All 100 fixed splits", actual_results="See results/A; A is cross-scoring without a randomized null distribution", support="Source and prediction replay checks", limitations="Current shared-input evidence limits remain", sources="results/A"),
        dict(category=6, stage="Closeout B: fixed-readout condition mismatch", status="PENDING", question="Assess task specificity of correct condition correspondence", input_level="Permute current test-condition correspondence on shared valid time support", labels="Both fixed target types, matched-support rescoring for each q/split", representation="No representation fitting", readout="Existing OLS readouts remain fixed", splits="Fixed 1000 q x 100 original splits", actual_results="Formal completion remains to be verified; smoke results are insufficient", support="Awaiting executed outputs", limitations="Mapping-dependent support requires empirical comparison proportions; not separate evidence for behavioral representation", sources="configs/closeout_protocol.json"),
        dict(category=6, stage="Closeout C: random endpoints and new OLS fits", status="PENDING", question="Assess real endpoint correspondence against random endpoints with retained geometry", input_level="Original neural inputs and geometry; reallocate 78 mean endpoints preserving their marginal distribution", labels="Each q allocation fixed across 100 splits and the complete sequence", representation="Reuse representations without FA/GPFA fitting", readout="Actual new random-target OLS fits and a mean-endpoint geometry baseline", splits="Fixed 1000 q x 100 original splits", actual_results="Formal completion remains to be verified; randomization results are not assumed", support="Awaiting executed outputs", limitations="Exceeding random endpoints does not establish candidate superiority, an internal path or trial-level behavior correspondence", sources="configs/closeout_protocol.json"),
    ]
    rows[-2].update(closeout_ledger_status("B"))
    rows[-1].update(closeout_ledger_status("C"))
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results/experiment_ledger.csv", index=False)
    text = "# Executed experiment ledger\n\nEntries are organized by scientific question and retain historical protocol identities. B/C status comes from actual completion markers, branches and score summaries; incomplete records remain PENDING/PARTIAL. **raw_preprocessing = fail** remains the upstream boundary.\n"
    for row in rows:
        text += f"\n## {row['category']}. {row['stage']}（{row['status']}）\n\n"
        for label, key in [("Question", "question"),("Input level","input_level"),("Target definition","labels"),("Representation","representation"),("Readout","readout"),("Splits","splits"),("Observed result","actual_results"),("Supported scope","support"),("Limitations","limitations")]:
            text += f"**{label}**：{row[key]}\n\n"
        text += "**Sources**: " + "; ".join(f"[{Path(s).name}]({s})" for s in row["sources"].split(";")) + ".\n"
    text += "\nThe current version changes target averaging and condition weighting together, so score changes cannot be assigned entirely to averaging. Historical tables use saved results; no procedure was adjusted to approach paper values.\n"
    (ROOT / "experiment_ledger.md").write_text(text, encoding="utf-8")


def write_interpretation(summary):
    text = "# Behavioral averaging and path separation\n\nThe frozen membership JSON and verified source-record SHA256 values define each recomputed mean. Endpoint errors use **objective_end_y from the current released-trajectory geometry**, rather than trial yf_mworks. Signs use exact floating zero without an inferred near-correct threshold. Variance uses ddof=0. endpoint_condition_audit.csv retains means, variances and SDs for each condition and its positive, negative and exact-zero subgroups; empty subgroups are NA.\n\n"
    for animal, s in summary["animals"].items():
        q=s["mean_endpoint_error_quantiles"]
        text += (f"{animal.title()}: 78 valid conditions and {s['n_members']} behavioral records; positive {s['n_up_records']}, negative {s['n_down_records']}, exact zero {s['n_exact_zero_records']}. Both signs occur in {s['n_conditions_with_both_error_signs']} conditions. Equal-condition mean_i|e_ci| is {s['condition_equal_mean_abs_trial_error']:.4f}, and |mean_i e_ci| is {s['condition_equal_abs_mean_error']:.4f}. Median within-condition cancellation is {s['median_condition_cancellation_fraction']:.1%}. Mean endpoint error interquartile range {q['0.25']:.4f}–{q['0.75']:.4f}, median {q['0.5']:.4f}, range {q['0']}–{q['1']} .\n\n"
                 f"Equal-condition mean_i(e_ci^2)={s['condition_equal_mean_square_trial_error']:.6f} decomposes into mean_i(e_ci)^2={s['condition_equal_mean_squared_mean_error']:.6f} and within-condition variance={s['condition_equal_mean_within_variance']:.6f}.\n\n")
    text += f"Maximum variance-identity residual across conditions {summary['variance_identity_max_abs_residual']:.3g}; maximum difference between recomputed behavioral means and existing labels {summary['membership_replay_max_abs_difference']:.3g}. This provides descriptive evidence that averaging cancels opposite endpoint errors.\n\n"
    paths=pd.DataFrame(summary["path_separation"])
    text += markdown_table(paths[paths.epoch.isin(["full","hidden"])][["animal","representation","epoch","path_RMS_pooled_bins","RMSE_obj_mean_of_100_splits","RMSE_beh_mean_of_100_splits"]])+"\n\n"
    text += "Path separation is descriptive RMS over all valid label rows. Reconstruction error is scored per held-out split and then averaged over 100 splits. Their magnitude comparison uses different aggregation. path_separation_vs_error.csv retains every condition and phase; scores of averaged prediction curves do not replace the original per-split scores.\n\n"
    text += "Behavioral cancellation does not establish that neural means contain exactly the same behavioral members. The comparison with trial labels removes within-condition target variation and extra record-count weighting together; it is not a one-factor ablation. A lack of overall candidate advantage cannot be assigned entirely to averaging, and weak separation of condition means does not rule out trial-level behavioral representation. **raw_preprocessing=fail**, unverified strictly pre-feedback terminal timing and estimated collision anchors remain.\n\n"
    text += "## Fixed post hoc cases and all conditions\n\nConditions 55062 and 241919 are user-specified post hoc illustrations, without independent confirmation. fixed_posthoc_case_scores.csv and fixed_posthoc_case_2x2.csv retain all phases for both animals and representations. Four-curve PNGs show existing test-prediction means; bands describe readout-split stability rather than animal trial variation. The all-79 heterogeneity figure retains invalid 59920 without favorable selection.\n\n"
    text += "Complete current atlases are listed in the [artifact registry](../../../../integration/artifact_registry.csv). Local improvements from the old stopping-proxy version do not transfer to current labels.\n"
    (OUT / "mean_cancellation_interpretation.md").write_text(text, encoding="utf-8")


def write_future_design():
    (ROOT / "future_design.md").write_text("""# Historical closeout design: behavioral groups within a condition

This is the design recorded at closeout. It was not executed. The [current cross-study research plan](../../../../docs/FUTURE_DIRECTIONS.md) is the single maintained plan for future work.

The proposed test asks whether neural representations differ with the direction of behavioral endpoint errors while the physical condition remains fixed. Define signed error as final paddle position minus objective endpoint. Form above-target, below-target and near-correct groups whose endpoints are reasonably similar within each group. Set thresholds from measurement precision and task tolerance before examining neural results. Exact numerical zero in the descriptive closeout audit is a counting rule, not a validated future grouping threshold. Failure alone does not identify an internal judgment error.

Match absolute error, session, genuine trial count and available neural units where feasible. Record unmatched data without selecting groups by decoding performance. The prerequisite is recovery of unit/session/trial neural responses, corresponding endpoint behavior, timestamps, event and feedback boundaries, and original averaging membership. Strictly pre-feedback sampling of the terminal behavioral measurement must also be checked.

With those paired records, compute each group's neural mean and evaluate it using a shared representation and readout protocol with independent splits. Ask whether reconstructed positions shift with above-target versus below-target behavior while the objective path stays fixed, and whether group-specific candidates explain structured departures from objective reconstruction. Fit representations, unit selection and data-dependent preprocessing on training data. Reliability must use genuine within-group trial splits.

Group-average analysis conditions on known behavioral membership. Blind prediction for a new single trial requires its own paired observations and evaluation. Cross-session grouped pseudopopulations require correct member identities and do not establish one simultaneously observed population decision.

The currently obtained data contain already mixed condition-mean neural responses. Their original members have not been recovered. A single mean cannot be decomposed into the proposed behavioral-group means; repeating it for separate behavioral records does not create new neural observations. This closeout therefore performed no grouped neural reconstruction, new download or representation search. The limitation applies to the obtained and audited data, without asserting that every unpublished record from the original experiment has the same limitation.

`raw_preprocessing=fail` remains unresolved. Released-input filtering/provenance checks cannot repair upstream cross-condition/time filling. Any future recovery of raw paired data must establish a training-side preprocessing boundary at its source.
""", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    digest(ROOT / "configs/closeout_protocol.json")
    labels, geometry, detail, mem, scores, main_scores, summary = mean_audit()
    create_figures(labels, geometry, detail, scores)
    create_ledger(main_scores)
    write_interpretation(summary)
    write_future_design()
    save_json(OUT / "source_hashes.json", SOURCES)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
