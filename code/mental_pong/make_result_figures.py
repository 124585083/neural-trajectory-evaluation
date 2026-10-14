"""Render documentation figures from published summary tables only.

No neural inputs, model fitting, inference, resampling, or new statistics are
used. Means, split SDs, and randomization quantiles are read as saved. Run this
file from any working directory; --output-dir redirects all generated files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "results/mental_pong/evidence/closeout_v1"
DEFAULT_OUTPUT = ROOT / "results/mental_pong/figures/documentation"
ANIMALS = ("mahler", "perle")
REPS = ("FA50", "GPFA50")
PHASES = ("full", "visible", "hidden", "bounce", "no_bounce", "post_bounce")
LABELS = ("Full", "Visible", "Hidden", "Collision conditions", "No-collision conditions", "Post-collision")
COLORS = ("#2878a5", "#b75d23")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_grid(frame: pd.DataFrame, metrics: tuple[str, ...] = ()) -> None:
    keys = ["animal", "representation", "epoch"] + (["metric"] if metrics else [])
    assert not frame.duplicated(keys).any(), "Duplicate summary rows"
    assert set(frame.animal) == set(ANIMALS)
    assert set(frame.representation) == set(REPS)
    for animal in ANIMALS:
        for rep in REPS:
            subset = frame[(frame.animal == animal) & (frame.representation == rep)]
            assert set(subset.epoch) == set(PHASES), "All six phases are required"
            for metric in metrics:
                assert set(subset[subset.metric == metric].epoch) == set(PHASES)
    assert len(frame) == len(ANIMALS) * len(REPS) * len(PHASES) * (len(metrics) or 1)


def axes_grid(title: str, subtitle: str):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    fig.subplots_adjust(left=.205, right=.975, bottom=.075, top=.82, wspace=.50, hspace=.34)
    fig.suptitle(title, x=.5, y=.985, fontsize=17, fontweight="bold")
    fig.text(.5, .937, subtitle, ha="center", va="top", fontsize=10.5, color="#414851")
    fig.legend(
        handles=[Line2D([], [], color=color, marker=marker, linestyle="none", label=rep)
                 for rep, color, marker in zip(REPS, COLORS, ("o", "s"))],
        loc="upper center", bbox_to_anchor=(.5, .90), ncol=2, frameon=False,
    )
    for row, animal in enumerate(ANIMALS):
        for ax in axes[row]:
            ax.set_yticks(np.arange(len(PHASES)), LABELS)
            ax.set_ylim(len(PHASES) - .45, -.55)
            ax.set_title(animal.title(), loc="left", fontsize=12, fontweight="bold", pad=10)
            ax.grid(axis="x", alpha=.20)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    return fig, axes


def save(fig, stem: str, output: Path) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white", metadata={"Creator": "make_result_figures.py", "Date": None} if suffix == "svg" else None)
        paths.append(path.name)
    plt.close(fig)
    return paths


def objective_reconstruction(frame: pd.DataFrame, output: Path) -> list[str]:
    data = frame[frame.coordinate.eq("y")].copy()
    assert set(data.coordinate) == {"y"}
    validate_grid(data)
    assert data.n_rounds.eq(100).all()
    for metric in ("r_obj", "RMSE_obj"):
        assert data[f"{metric}_n_finite"].eq(100).all()
        assert np.isfinite(data[[metric, f"{metric}_sd"]].to_numpy()).all()
        assert data[f"{metric}_sd"].ge(0).all()
    fig, axes = axes_grid(
        "Objective position is recoverable across all six task phases",
        "Objective-trained head / objective y target; saved mean ± split SD across 100 overlapping condition splits",
    )
    for row, animal in enumerate(ANIMALS):
        for col, metric in enumerate(("r_obj", "RMSE_obj")):
            ax = axes[row, col]
            for rep, color, marker, shift in zip(REPS, COLORS, ("o", "s"), (-.13, .13)):
                d = data[(data.animal == animal) & (data.representation == rep)].set_index("epoch").loc[list(PHASES)]
                ax.errorbar(d[metric], np.arange(6) + shift, xerr=d[f"{metric}_sd"],
                            fmt=marker, color=color, capsize=3, markersize=5, linewidth=1.4)
            ax.set_xlabel("Pearson r (higher is better)" if col == 0 else 'RMSE, MWorks position units (lower is better)')
            ax.set_xlim((.30, .96) if col == 0 else (2.4, 6.65))
    return save(fig, "results_objective_six_phases", output)


def objective_mismatch(frame: pd.DataFrame, output: Path) -> list[str]:
    metrics = ("paired_r", "paired_RMSE")
    data = frame[frame["head"].eq("D_obj") & frame.target.eq("objective") &
                 frame.coordinate.eq("y") & frame.metric.isin(metrics)].copy()
    assert set(data["head"]) == {"D_obj"}, "Never substitute candidate-head mismatch results"
    assert set(data.target) == {"objective"}
    assert set(data.coordinate) == {"y"}
    validate_grid(data, metrics)
    assert data.n_original_splits_assessed.eq(100).all()
    assert data.B.eq(1000).all() and data.n_finite_randomizations.eq(1000).all()
    assert np.isfinite(data[["mean", "q025", "q975"]].to_numpy()).all()
    assert (data.q025 <= data["mean"]).all() and (data["mean"] <= data.q975).all()
    fig, axes = axes_grid(
        "Objective readouts preserve physical-condition correspondence",
        "B: fixed D_obj head / objective y target; saved mean and 2.5th–97.5th random-allocation percentiles\n"
        "Each allocation combines the same 100 splits; paired support can change between allocations",
    )
    for row, animal in enumerate(ANIMALS):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for rep, color, marker, shift in zip(REPS, COLORS, ("o", "s"), (-.13, .13)):
                d = data[(data.animal == animal) & (data.representation == rep) &
                         (data.metric == metric)].set_index("epoch").loc[list(PHASES)]
                # Endpoint-to-mean distances only; the quantiles are not recalculated.
                xerr = np.vstack((d["mean"] - d.q025, d.q975 - d["mean"]))
                ax.errorbar(d["mean"], np.arange(6) + shift, xerr=xerr,
                            fmt=marker, color=color, capsize=3, markersize=5, linewidth=1.4)
            ax.axvline(0, color="#555555", linewidth=.9, linestyle="--")
            ax.set_xlabel("Matched − mismatched r" if col == 0 else 'Mismatched − matched RMSE (MWorks position units)')
            ax.set_xlim((-.035, .97) if col == 0 else (-.15, 3.55))
    return save(fig, "results_objective_mismatch", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory for PNG/SVG figures and source manifest (default: published documentation figures)")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "svg.fonttype": "none", "svg.hashsalt": "mental-pong-results"})
    a_source = EVIDENCE / "A/self_reconstruction_main_table.csv"
    b_source = EVIDENCE / "B/B_summary.csv"
    files = objective_reconstruction(pd.read_csv(a_source), output)
    files += objective_mismatch(pd.read_csv(b_source), output)
    manifest = {
        "operation": "Plot saved summaries only; no model fit, inference, bootstrap, or new summary statistics",
        "script": str(Path(__file__).relative_to(ROOT).as_posix()),
        "script_sha256": sha256(Path(__file__)),
        "sources": {p.relative_to(ROOT).as_posix(): sha256(p) for p in (a_source, b_source)},
        "figures": {
            "results_objective_six_phases": {"rows": 24, "filter": {"coordinate": "y"},
                "fields": ["r_obj", "r_obj_sd", "RMSE_obj", "RMSE_obj_sd"],
                "interval": "Saved population SD across 100 overlapping condition splits"},
            "results_objective_mismatch": {"rows": 48, "filter": {"head": "D_obj", "target": "objective", "coordinate": "y", "metric": ["paired_r", "paired_RMSE"]},
                "fields": ["mean", "q025", "q975"],
                "interval": "Saved 2.5th–97.5th percentiles of 1000 random-allocation summary statistics; changing paired support"},
        },
        "animals": list(ANIMALS), "representations": list(REPS), "phases": list(PHASES),
        "outputs": files,
    }
    manifest_path = output / "results_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name in [*files, manifest_path.name]:
        print(output / name)


if __name__ == "__main__":
    main()
