"""Render publication prose and tables from completed scientific results.

Templates contain the maintained English narrative. Tables are rebuilt from saved
full-precision values. This route does not collect predictions, refit models, or
write numerical results. An explicit output directory keeps inspection separate
from the source study.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
TEMPLATES = PROJECT / "report_templates"
METRICS = ["r_obj", "r_beh", "RMSE_obj", "RMSE_beh", "Delta_r", "Delta_RMSE"]
IDENTITY = ["animal", "representation"]
OWN = IDENTITY + METRICS
PHASE = IDENTITY + ["epoch"] + METRICS


def markdown(frame):
    """Use the saved values before rounding each displayed cell to four decimals."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}" if np.isfinite(value) else "NA"
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(cell(x) for x in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def cancellation_table(summary):
    keys = {
        "n_up": "n_up_records", "n_down": "n_down_records",
        "both_sign_conditions": "n_conditions_with_both_error_signs",
        "mean_abs": "condition_equal_mean_abs_trial_error",
        "abs_mean": "condition_equal_abs_mean_error",
        "mean_square": "condition_equal_mean_square_trial_error",
        "squared_mean": "condition_equal_mean_squared_mean_error",
        "within_variance": "condition_equal_mean_within_variance",
    }
    return pd.DataFrame([{"animal": animal, **{k: row[v] for k, v in keys.items()}}
                         for animal, row in summary["animals"].items()])


def closeout_tables(root):
    root = Path(root)
    acceptance = json.loads((root / "results/final_acceptance_audit.json").read_text(encoding="utf-8"))
    if not acceptance["completed"]:
        raise ValueError("Completed scientific results are required for this narrative.")
    a = pd.read_csv(root / "results/A/self_reconstruction_main_table.csv")
    y = a[a.coordinate.eq("y")]
    full = y[y.epoch.eq("full")]
    if len(full) != 4 or not full.n_rounds.eq(100).all():
        raise ValueError("Expected four full-interval y summaries from 100 splits.")
    # These completed-study conclusions must fail visibly if different data are supplied.
    if not (full.Delta_r.lt(0).all() and full.Delta_RMSE.lt(0).all()):
        raise ValueError("Saved values no longer support the published own-target conclusion.")
    cross = pd.read_csv(root / "results/A/cross_2x2_summary.csv")
    cross = cross[cross.coordinate.eq("y") & cross.epoch.isin(["full", "hidden"])]
    matrix = []
    for (animal, rep, epoch), group in cross.groupby(IDENTITY + ["epoch"], sort=False):
        row = dict(animal=animal, representation=rep, epoch=epoch)
        for score in group.itertuples():
            row[score.cell] = f"{score.r:.4f} / {score.RMSE:.4f}"
        matrix.append(row)
    b = pd.read_csv(root / "results/B/B_summary.csv")
    b = b[b.coordinate.eq("y") & b.epoch.isin(["full", "hidden"]) & b.is_self_target]
    b = b.pivot(index=IDENTITY + ["epoch", "head", "target"],
                columns="metric", values="mean").reset_index()
    cov = pd.read_csv(root / "results/B/B_coverage_summary.csv")
    cov = cov[cov.metric.eq("shared_bins_fraction")]
    c = pd.read_csv(root / "results/C/random_endpoint_summary.csv")
    c = c[c.metric.isin(["r", "RMSE", "skill"])].pivot(
        index=IDENTITY + ["epoch"], columns="metric",
        values=["objective_mean", "behavior_mean", "null_mean",
                "behavior_empirical_random_at_least_as_good_fraction"])
    c.columns = ["_".join(x) for x in c.columns]
    c = c.reset_index()
    short = c[c.epoch.isin(["full", "hidden", "endpoint_influence"])]
    post = c[c.epoch.eq("post_bounce")]
    # Preserve the existing phase order in the geometry-baseline table.
    phase_order = ["full", "hidden", "no_bounce", "endpoint_influence"]
    skill = c[c.epoch.isin(phase_order)].copy()
    skill["epoch"] = pd.Categorical(skill.epoch, phase_order, ordered=True)
    skill = skill.sort_values(IDENTITY + ["epoch"])
    skill = skill[IDENTITY + ["epoch", "objective_mean_skill", "behavior_mean_skill", "null_mean_skill"]]
    skill.columns = IDENTITY + ["epoch", "objective_mean", "behavior_mean", "null_mean"]
    summary = json.loads((root / "results/descriptive/mean_cancellation_summary.json").read_text(encoding="utf-8"))
    return [full[OWN], y[y.epoch.isin(["hidden", "no_bounce", "post_bounce"])][PHASE],
            pd.DataFrame(matrix)[IDENTITY + ["epoch", "OO", "OB", "BO", "BB"]],
            b[IDENTITY + ["epoch", "head", "matched_r", "null_r", "paired_r",
                          "matched_RMSE", "null_RMSE", "paired_RMSE"]],
            cov[["animal", "epoch", "mean_of_q_means", "q025", "q975",
                 "minimum_over_all_q_splits", "maximum_over_all_q_splits", "n_zero_support_q_splits"]],
            short[IDENTITY + ["epoch", "objective_mean_r", "behavior_mean_r", "null_mean_r",
                              "objective_mean_RMSE", "behavior_mean_RMSE", "null_mean_RMSE"]],
            post[IDENTITY + ["behavior_mean_r", "null_mean_r", "behavior_mean_RMSE", "null_mean_RMSE",
                             "behavior_empirical_random_at_least_as_good_fraction_RMSE"]],
            skill, cancellation_table(summary)]


def condition_tables(root):
    root = Path(root)
    main = pd.read_csv(root / "results/self_reconstruction_main_table.csv")
    y = main[main.coordinate.eq("y")]
    full = y[y.epoch.eq("full")]
    stats = full[IDENTITY].copy()
    for name in METRICS:
        stats[name + " (mean +/- SD)"] = [f"{m:.4f} +/- {s:.4f}" for m, s in zip(full[name], full[name + "_sd"])]
    coverage = pd.read_csv(root / "results/condition_label_coverage.csv").groupby("animal").agg(
        source=("n_all_source_behavior_records", "sum"), members=("n_behavior_trials_in_mean", "sum"),
        conditions=("n_condition_time_rows", lambda x: int((x > 0).sum())), rows=("n_condition_time_rows", "sum")).reset_index()
    coverage.animal = coverage.animal.str.title()
    coverage.columns = ["Animal", "Source behavioral records", "Records in means", "Valid conditions", "Condition-by-time rows"]
    counts = pd.read_csv(root / "results/condition_difference_counts.csv")
    counts = counts[counts.epoch.isin(["full", "hidden", "post_bounce"])]
    cross = pd.read_csv(root / "results/cross_2x2_summary.csv")
    own = cross[cross.coordinate.eq("y") & cross.epoch.eq("full") & (
        (cross["head"].eq("D_obj") & cross.target.eq("objective")) |
        (cross["head"].eq("D_beh") & cross.target.eq("behavior")))]
    comparison = pd.read_csv(root / "results/previous_trial_version_comparison.csv")
    comparison = comparison[comparison.epoch.eq("full") & comparison.coordinate.eq("y")]
    return [stats, coverage, y[PHASE], counts,
            own[IDENTITY + ["head", "target_sd", "prediction_sd", "bias", "amplitude_ratio"]],
            comparison[IDENTITY + ["r_obj_trial_version", "r_obj_condition_mean", "r_beh_trial_version",
                                   "r_beh_condition_mean", "RMSE_obj_trial_version", "RMSE_obj_condition_mean",
                                   "RMSE_beh_trial_version", "RMSE_beh_condition_mean"]]]


def render_template(name, output_path, tables=()):
    template = (TEMPLATES / (name + ".md.in")).read_text(encoding="utf-8")
    expected = set(re.findall(r"\{\{TABLE_(\d+)\}\}", template))
    if expected != {str(i) for i in range(len(tables))}:
        raise ValueError(f"Template/table mismatch for {name}: {expected}")
    for i, frame in enumerate(tables):
        template = template.replace("{{TABLE_" + str(i) + "}}", markdown(frame))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding="utf-8", newline="\n")
    return path


def render_closeout(source_root, output_dir):
    tables = closeout_tables(source_root)
    out = Path(output_dir)
    return [render_template("closeout_report", out / "FINAL_REPORT.md", tables),
            render_template("closeout_summary", out / "FINAL_SUMMARY.md", [tables[0]])]


def render_condition(source_root, output_dir):
    out = Path(output_dir)
    return [render_template("condition_report", out / "REPORT.md", condition_tables(source_root)),
            render_template("condition_readme", out / "README.md"),
            render_template("condition_protocol_diff", out / "protocol_diff.md")]


def render_descriptive(source_root, output_dir):
    root, out = Path(source_root), Path(output_dir)
    summary = json.loads((root / "results/descriptive/mean_cancellation_summary.json").read_text(encoding="utf-8"))
    paths = pd.DataFrame(summary["path_separation"])
    paths = paths[paths.epoch.isin(["full", "hidden"])][IDENTITY + ["epoch", "path_RMS_pooled_bins",
                   "RMSE_obj_mean_of_100_splits", "RMSE_beh_mean_of_100_splits"]]
    cases = pd.read_csv(root / "results/descriptive/fixed_posthoc_case_scores.csv")
    cases = cases[cases.epoch.isin(["full", "hidden"]) & cases.coordinate.eq("y")]
    return [render_template("mean_cancellation", out / "results/descriptive/mean_cancellation_interpretation.md",
                            [cancellation_table(summary), paths]),
            render_template("posthoc_cases", out / "results/descriptive/fixed_posthoc_case_interpretation.md",
                            [cases[IDENTITY + ["condition_id", "epoch"] + METRICS]]),
            render_template("future_design", out / "future_design.md"),
            render_template("experiment_ledger", out / "experiment_ledger.md")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Separate destination for rendered reports; no numeric outputs are written.")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.is_relative_to(PROJECT) or PROJECT.is_relative_to(destination):
        parser.error("Choose an output directory outside the scientific source tree.")
    render_closeout(PROJECT / "closeout_v1", destination / "closeout_v1")
    render_condition(PROJECT / "condition_endpoint_fa_gpfa_v1", destination / "condition_endpoint_fa_gpfa_v1")
    render_descriptive(PROJECT / "closeout_v1", destination / "closeout_v1")


if __name__ == "__main__":
    main()
