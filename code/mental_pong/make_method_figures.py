"""Render the Mental-Pong documentation schematics without loading research data.

The path coordinates and future response shapes below are illustrative, not
observations or model predictions. This script fits no model, calculates no
scientific score, and changes none of the retained analysis figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "mental_pong" / "figures" / "documentation"
INK = "#20364D"
MUTED = "#526476"
BLUE = "#24659B"
ORANGE = "#C35D21"
TEAL = "#227969"
EDGE = "#BAC7D1"
PALE = "#F1F5F8"
WARM = "#FFF4E9"
GREEN = "#EDF6F2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": EDGE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "svg.fonttype": "none",
    "svg.hashsalt": "mentalpong-methods-v1",
})


def canvas(title: str, subtitle: str, size=(14, 10)):
    fig = plt.figure(figsize=size, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.05, .964, title, fontsize=21, weight="bold", va="top")
    ax.text(.05, .925, subtitle, fontsize=12, color=MUTED, va="top")
    return fig, ax


def box(ax, x, y, w, h, title, body, *, face=PALE, title_color=INK,
        title_size=12.5, body_size=11.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1, edgecolor=EDGE, facecolor=face,
    ))
    ax.text(x + .013, y + h - .022, title, fontsize=title_size,
            weight="bold", color=title_color, va="top", linespacing=1.25)
    ax.text(x + .013, y + h - .057, body, fontsize=body_size,
            va="top", linespacing=1.4)


def arrow(ax, start, end, *, color=MUTED, style="-", bend=0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.5,
        color=color, linestyle=style, connectionstyle=f"arc3,rad={bend}",
        shrinkA=2, shrinkB=2,
    ))


def save(fig, output_dir: Path, name: str):
    for extension in ("png", "svg"):
        path = output_dir / f"{name}.{extension}"
        metadata = {"Title": name, "Description": "Documentation schematic; not research data."}
        if extension == "svg":
            metadata["Date"] = None
        fig.savefig(path, dpi=170, facecolor="white", metadata=metadata)
        print(path)
    plt.close(fig)


def workflow(output_dir: Path):
    fig, ax = canvas(
        "Condition-held-out position reconstruction",
        "Completed protocol  |  Mahler and Perle analyzed separately  |  half1 readouts",
        size=(14, 9.5),
    )
    ax.add_patch(FancyBboxPatch((.05, .825), .90, .061,
        boxstyle="round,pad=0.008,rounding_size=0.008", facecolor=WARM,
        edgecolor="#E7C39E", linewidth=1))
    ax.text(.065, .856,
        "Upstream limitation: released missing responses were filled across conditions and times.\n"
        "Training-only fitting and GPFA prefix inference cannot undo this: raw_preprocessing = fail.",
        fontsize=11.5, va="center", color="#77502B", linespacing=1.4)

    box(ax, .05, .64, .24, .135, "Released neural means",
        "79 conditions × 100 bins\n50 ms/bin; N aligned units\nNo paired neural–behavior trials")
    box(ax, .365, .64, .25, .135, "100 fixed condition splits",
        "39 train / 40 test identities\nWhole conditions held out\nSame splits for all comparisons")
    box(ax, .69, .64, .26, .135, "Fit on training half1 only",
        "Neuron eligibility + mean / SD\nFA50 or shared-timescale GPFA50\nNo task labels in latent fitting",
        face=GREEN, title_color=TEAL)
    arrow(ax, (.295, .707), (.358, .707))
    arrow(ax, (.622, .707), (.683, .707))

    box(ax, .365, .405, .25, .145, "Infer held-out features",
        "Freeze preprocessing + model\nFA50 or prefix GPFA50\n50 features per condition/bin",
        face=GREEN, title_color=TEAL)
    box(ax, .69, .405, .26, .145, "Fit two OLS XY heads",
        "Same training features + rows\nObjective head / candidate head\nOne fixed head across phases",
        face=GREEN, title_color=TEAL)
    arrow(ax, (.49, .636), (.49, .557))
    ax.text(.502, .597, "test responses", fontsize=10.5, va="center")
    arrow(ax, (.815, .636), (.815, .557), color=TEAL)
    ax.text(.827, .597, "train features", fontsize=10.5, va="center")
    arrow(ax, (.71, .636), (.605, .554), color=TEAL)
    ax.text(.645, .592, "freeze", fontsize=10.5, ha="center", color=TEAL,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})

    box(ax, .365, .17, .25, .145, "Score test rows only",
        "Both heads × both targets\nr and RMSE within each phase\nThen summarize split scores",
        face="#EEF3FA", title_color=BLUE)
    box(ax, .69, .17, .26, .145, "Separate target branch",
        "Physical geometry → objective\nMean paddle endpoint + geometry\n→ behavior-constrained candidate",
        face=WARM, title_color=ORANGE)
    arrow(ax, (.49, .401), (.49, .322), color=BLUE)
    ax.text(.502, .365, "test features", fontsize=10.5, va="center")
    arrow(ax, (.705, .401), (.605, .322), color=BLUE)
    ax.text(.655, .368, "frozen heads", fontsize=10.5, ha="center", color=BLUE,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    arrow(ax, (.815, .318), (.815, .398), color=ORANGE)
    ax.text(.827, .365, "train labels", fontsize=10.5, va="center", color=ORANGE)
    arrow(ax, (.683, .24), (.622, .24), color=ORANGE)
    ax.text(.652, .258, "test labels", fontsize=10, ha="center", color=ORANGE)

    box(ax, .05, .17, .24, .38, "Fixed within a split",
        "• Animal analyzed separately\n\n"
        "• Split identities and masks\n\n"
        "• Unlabeled neural representation\n\n"
        "• OLS head within every phase\n\n"
        "Only the downstream target\nand control operation change.",
        face="white", body_size=11)
    ax.text(.05, .095,
        "78 geometry-valid conditions; 59920 stays in the 79-ID split file but has no scored labels.\n"
        "100 overlapping splits measure partition sensitivity, not 100 independent neural experiments.",
        fontsize=11.5, color=MUTED, va="top", linespacing=1.5)
    save(fig, output_dir, "methods_condition_workflow")


def target_panel(fig, rect, collision: bool):
    ax = fig.add_axes(rect)
    ax.set(xlim=(0, 10.7), ylim=(-2.9, 3.5), xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(10, color=EDGE, lw=1.5)
    if collision:
        xs, obj, candidate = [0, 5, 10], [2.2, -1.9, 2.2], [2.2, -1.9, 1.0]
        ax.plot([0, 10], [-1.9, -1.9], color=EDGE, lw=1, ls=":")
        ax.text(5, -2.5, "Physical collision anchor", ha="center", fontsize=11)
        ax.scatter([5], [-1.9], color=INK, s=28, zorder=4)
        ax.text(.4, -.1, "Same path\nbefore anchor", fontsize=10.5, color=MUTED)
        title = "Collision condition"
    else:
        xs, obj, candidate = [0, 10], [1.9, -.6], [1.9, 1.2]
        ax.scatter([0], [1.9], color=INK, s=28, zorder=4)
        ax.text(.2, 2.45, "Metadata initial anchor", fontsize=11)
        ax.text(1.3, -.5, "Candidate can already differ\nduring visible motion", fontsize=10.5, color=MUTED)
        title = "No-collision condition"
    ax.plot(xs, obj, color=BLUE, lw=2.5)
    ax.plot(xs, candidate, color=ORANGE, lw=2.5, ls="--")
    ax.scatter([10], [candidate[-1]], color=ORANGE, s=38, zorder=4)
    ax.scatter([10], [obj[-1]], color=BLUE, s=34, zorder=4)
    ax.text(10, 3.0, "Paddle plane x = 10", fontsize=10.5, ha="right", color=MUTED)
    ax.text(10.2, candidate[-1], "b", color=ORANGE, fontsize=12, va="center")
    ax.set_title(title, fontsize=13, weight="bold", loc="left", pad=4)


def targets_controls(output_dir: Path):
    fig, ax = canvas(
        "Targets, fitted heads and randomized controls",
        "Completed protocol  |  Geometry sketches illustrate the rule; they are not recorded trajectories",
        size=(14, 11),
    )
    target_panel(fig, [.06, .60, .40, .265], collision=True)
    target_panel(fig, [.55, .60, .40, .265], collision=False)
    ax.plot([.06, .09], [.565, .565], color=BLUE, lw=2.5)
    ax.text(.10, .565, "Objective physical path", va="center", fontsize=11.5)
    ax.plot([.34, .37], [.565, .565], color=ORANGE, lw=2.5, ls="--")
    ax.text(.38, .565, "Endpoint-constrained candidate", va="center", fontsize=11.5)
    ax.text(.735, .565, "Shared x, arrival time and masks", va="center", fontsize=11.5, color=MUTED)
    ax.text(.06, .523,
        r"After the anchor:  $\alpha(t) = [x_{obj}(t)-x_a]/[10-x_a]$;  "
        r"$y_{cand}(t) = (1-\alpha)y_a + \alpha b$.", fontsize=13, va="center")
    ax.text(.06, .49,
        "b is a retrospective condition-mean terminal paddle label; its exact pre-feedback timing is unresolved.",
        fontsize=11.5, color=MUTED, va="center")

    # Rows identify the trained head; columns identify the scoring target.
    ax.text(.06, .441, "A  |  Same test predictions, both references", weight="bold", fontsize=13)
    x0, y0, widths, row_h = .06, .285, [.22, .135, .135], .042
    cells = [
        ["Trained head / scored target", "Objective", "Candidate"],
        ["Objective head", "OO", "OB"],
        ["Candidate head", "BO", "BB"],
    ]
    for r, row in enumerate(cells):
        left = x0
        for c, label in enumerate(row):
            bottom = y0 + (2-r)*row_h
            face = PALE if r == 0 or c == 0 else ("#EEF3FA" if c == 1 else WARM)
            ax.add_patch(Rectangle((left, bottom), widths[c], row_h, facecolor=face,
                                   edgecolor="white", linewidth=2))
            ax.text(left + widths[c]/2, bottom + row_h/2, label,
                    ha="center", va="center", fontsize=10.5 if r == 0 else 11.5,
                    weight="bold" if r == 0 else "normal")
            left += widths[c]
    ax.text(.575, .389,
        "OO vs BB: change both head and target\n"
        "OO vs BO: keep objective reference fixed\n"
        "OB vs BB: keep candidate reference fixed",
        fontsize=11.5, va="top", linespacing=1.65)

    box(ax, .06, .09, .42, .143, "B  |  Shuffle test-condition references",
        "Fitted head + predictions stay FIXED.\n"
        "Re-score matched / shuffled labels on the same intersection.\n"
        "Also disrupts physical condition correspondence.",
        face="#EEF3FA", title_color=BLUE, body_size=11)
    box(ax, .535, .09, .415, .143, "C  |  Shuffle mean endpoints, then REFIT",
        "Keep destination geometry, x/time and valid masks.\n"
        "Rebuild the candidate; fit a new training-side OLS head.\n"
        "Tests real assignments against fitted random targets.",
        face=WARM, title_color=ORANGE, body_size=11)
    ax.text(.06, .045,
        "Both controls: 1,000 allocations × the same 100 splits; representations stay fixed.  "
        "Geometry-only baseline: a separate, unmatched-input reference.", fontsize=10.7, color=MUTED)
    save(fig, output_dir, "methods_targets_controls")


def proposal_panel(fig, rect, event_change: bool):
    ax = fig.add_axes(rect)
    ax.set(xlim=(0, 1), ylim=(-.07, 1.15), xticks=[], yticks=[])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(EDGE)
    ax.axvspan(.32, .98, color=PALE)
    ax.axvline(.32, color=EDGE, ls=":", lw=1.3)
    ax.axvline(.64, color=MUTED, ls=":", lw=1.3)
    ax.axvline(.98, color=MUTED, ls=":", lw=1.3)
    if event_change:
        xs, ys = [0, .32, .64, .74, .98], [.32, .32, .32, .73, .73]
        title = "Event-related change"
    else:
        xs, ys = [0, .98], [.63, .63]
        title = "Persistent behavior-related offset"
    ax.plot(xs, ys, color=TEAL, lw=2.7, ls="--")
    ax.text(.32, -.12, "Occlusion", fontsize=10, ha="center")
    ax.text(.64, -.12, "Task event", fontsize=10, ha="center")
    ax.text(.98, -.12, "Before\nfeedback", fontsize=10, ha="center")
    ax.text(.06, 1.04, "Illustrative signed group difference", fontsize=10.5, color=MUTED)
    ax.set_title(title, loc="left", weight="bold", fontsize=12.5, pad=9)
    ax.set_ylabel("Schematic difference\n(no measured scale)", fontsize=10.5)


def paired_trial_proposal(output_dir: Path):
    fig, ax = canvas(
        "Proposed next study: behavior within the same physical condition",
        "NOT EXECUTED  |  Requires paired neural–behavior trials  |  All curves below are schematic",
        size=(14, 10),
    )
    box(ax, .05, .705, .255, .15, "1  |  Verify paired observations",
        "Same animal, session and trial ID\n"
        "Known stimulus + neural + action records\n"
        "Exact event / feedback timing",
        face=PALE, body_size=11)
    box(ax, .367, .705, .255, .15, "2  |  Group within condition",
        "Signed error: e = endpoint − objective\n"
        "Positive: e > δ; negative: e < −δ\n"
        "Choose δ before inspecting neural results",
        face=WARM, title_color=ORANGE, body_size=11)
    box(ax, .684, .705, .266, .15, "3  |  Establish a reliable contrast",
        "Disjoint paired-trial train / test sets\n"
        "Frozen representation + readout\n"
        "Reliability and within-condition shuffles",
        face=GREEN, title_color=TEAL, body_size=11)
    arrow(ax, (.311, .78), (.36, .78))
    arrow(ax, (.628, .78), (.677, .78))

    ax.text(.05, .649, "Primary test: pre-feedback signed-error decoding on held-out paired trials",
        fontsize=14, weight="bold")
    ax.text(.05, .611,
        "Compare with condition/time-only and movement-informed references. Report unsupported groups and failed reliability.\n"
        "Behavioral groups denote action outcomes; a positive neural result could reflect estimate, goal or motor preparation.",
        fontsize=11.5, color=MUTED, va="top", linespacing=1.55)
    ax.text(.05, .533, "Only after reliable behavioral correspondence: examine when the difference changes",
        fontsize=13, weight="bold", color=TEAL)
    proposal_panel(fig, [.085, .263, .37, .205], event_change=False)
    proposal_panel(fig, [.585, .263, .37, .205], event_change=True)
    ax.text(.05, .166,
        "These patterns may coexist. An event-related claim needs change from a pre-event baseline, reliable support\n"
        "and timing/movement controls; post-event decoding alone is insufficient. A state difference does not identify its mechanism.",
        fontsize=11.5, color=MUTED, va="top", linespacing=1.5)
    ax.add_patch(FancyBboxPatch((.05, .054), .9, .063,
        boxstyle="round,pad=0.008,rounding_size=0.008", facecolor=WARM,
        edgecolor="#E7C39E", linewidth=1))
    ax.text(.065, .085,
        "Decision gate: without trial pairing, pre-feedback timing and adequate group reliability, stop at the data limitation.\n"
        "Released condition means cannot be relabeled to recover above/below neural trial groups.",
        fontsize=11.5, va="center", color="#77502B", linespacing=1.4)
    save(fig, output_dir, "methods_paired_trial_proposal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory for six documentation-only PNG/SVG files.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workflow(args.output_dir)
    targets_controls(args.output_dir)
    paired_trial_proposal(args.output_dir)


if __name__ == "__main__":
    main()
