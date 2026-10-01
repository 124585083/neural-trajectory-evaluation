"""All79 condition-average trajectories from held-out readout predictions only."""
from __future__ import annotations

import json
import hashlib
import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

from trajectory_project.runtime_paths import pdf_python, pdftoppm
BUNDLE_PYTHON = pdf_python()
PDFTOPPM = pdftoppm()
REPRESENTATIONS = ("FA50", "GPFA50")
COLORS = {"objective": "#263445", "D_obj": "#1674B8", "behavior": "#CF7514", "D_beh": "#AB327B"}


def condition_figure_order(data):
    """Retain all79 identities, sorted by ID; no trial or score selection."""
    rows = []
    for animal, a in sorted(data.items()):
        geometry = a["geometry"].set_index("condition_index")
        ids = np.asarray(a["condition_ids"], int)
        if len(ids) != 79 or len(set(ids.tolist())) != 79:
            raise ValueError("Every atlas must retain the same 79 physical identities")
        for ci, cid in enumerate(ids):
            supported = bool(np.asarray(a["common_mask"][ci], bool).any())
            rows.append({"animal": animal, "condition_index": ci, "condition_id": int(cid),
                         "n_behavior_trials_in_mean": int(a["n_behavior_trials"][ci]),
                         "bounce_class": str(geometry.loc[ci, "bounce_class"]),
                         "label_available": supported,
                         "display_rule": "every physical condition, sorted by condition_id; no example selection",
                         "no_label_reason": "" if supported else str(geometry.loc[ci, "invalid_reason"])})
    return rows


def collect_test_predictions(data, root, expected_rounds=100):
    """Only test-condition predictions contribute to mean or split SD."""
    root = Path(root)
    output, audits = {}, []
    for animal, a in sorted(data.items()):
        output[animal] = {}
        for rep in REPRESENTATIONS:
            files = sorted((root/"readouts/predictions").glob(f"{animal}_{rep}_r*_test_predictions.npz"))
            if not files:
                # Prefix is deliberately read from the verified decoder naming.
                files = sorted((root/"readouts/predictions").glob(f"{animal}_r*_{rep}_test_predictions.npz"))
            if len(files) != expected_rounds:
                raise ValueError(f"{animal}/{rep}: expected {expected_rounds} saved test-only rounds, found {len(files)}")
            shape = (2, len(a["condition_ids"]), len(a["times_ms"]), 2)
            total, square = np.zeros(shape), np.zeros(shape)
            count = np.zeros(shape, np.int32)
            membership = np.zeros(79, np.int32)
            iterations, sources = [], []
            for path in files:
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["condition_ids"], a["condition_ids"])
                    np.testing.assert_array_equal(z["times_ms"], a["times_ms"])
                    pred = z["predictions"].copy()
                    test = z["test_condition_indices"].copy()
                    support = z["test_input_support"].copy()
                    iteration = int(z["iteration"])
                    if len(test) != 40 or len(np.unique(test)) != 40:
                        raise AssertionError("Plot source must preserve all 40 test identities")
                    train = np.setdiff1d(np.arange(79), test)
                    if np.isfinite(pred[:, train]).any() or support[train].any():
                        raise AssertionError("Training predictions cannot enter an OOF figure")
                    valid = np.isfinite(pred) & support[None, :, :, None]
                    total += np.where(valid, pred, 0.)
                    square += np.where(valid, pred*pred, 0.)
                    count += valid
                    membership[test] += 1
                    iterations.append(iteration)
                sources.append({"path": str(path.resolve()), "sha256": sha(path)})
            if len(set(iterations)) != expected_rounds:
                raise AssertionError("Duplicate or missing iterations")
            mean = np.divide(total, count, out=np.full(shape, np.nan), where=count > 0)
            variance = np.divide(square, count, out=np.full(shape, np.nan), where=count > 0)-mean*mean
            sd = np.sqrt(np.maximum(variance, 0.))
            output[animal][rep] = {"mean": mean, "sd": sd, "count": count, "test_membership_count": membership}
            np.savez_compressed(root/"artifacts"/f"{animal}_{rep}_condition_oof.npz",
                                **output[animal][rep], condition_ids=a["condition_ids"], times_ms=a["times_ms"],
                                interpretation=np.asarray("test split stability, not behavioral or neural trial variation"))
            audits.append({"animal": animal, "representation": rep, "n_rounds": len(files),
                           "minimum_condition_test_memberships": int(membership.min()),
                           "maximum_condition_test_memberships": int(membership.max()),
                           "training_prediction_used": False, "sources": sources})
    return output, audits


def _global_limits(data, summaries, selection):
    low, high, last = -10., 10., 0.
    for row in selection:
        a = data[row["animal"]]
        ci = row["condition_index"]
        g = a["geometry"].set_index("condition_index").loc[ci]
        last = max(last, float(g.T_ms))
        if not row["label_available"]:
            continue
        mask = a["common_mask"][ci]
        values = [a["objective_xy"][ci, mask, 1], a["behavior_xy"][ci, mask, 1]]
        for rep in REPRESENTATIONS:
            z = summaries[row["animal"]][rep]
            for head in range(2):
                values.extend([z["mean"][head, ci, mask, 1]-z["sd"][head, ci, mask, 1],
                               z["mean"][head, ci, mask, 1]+z["sd"][head, ci, mask, 1]])
        values = np.concatenate(values)
        finite = values[np.isfinite(values)]
        if len(finite):
            low, high = min(low, finite.min()), max(high, finite.max())
    margin = .035*(high-low)
    return (math.floor(low-margin), math.ceil(high+margin)), (.25, math.ceil((last+100)/100)/10)


def _panel(ax, a, item, representation, summary, ylimits, xlimits):
    ci = item["condition_index"]
    g = a["geometry"].set_index("condition_index").loc[ci]
    t = a["times_ms"]/1000.
    occ, arrival = float(g.occ_design_ms)/1000., float(g.T_ms)/1000.
    ax.axvspan(occ, arrival, color="#CAD3DC", alpha=.3, lw=0, zorder=0)
    ax.axvline(occ, color="#6E7A86", lw=.8, linestyle=":", zorder=1)
    ax.axvline(arrival, color="#B94A48", lw=.9, linestyle=":", zorder=1)
    if np.isfinite(g.collision_time_ms):
        ax.axvspan(float(g.collision_low_ms)/1000., float(g.collision_high_ms)/1000.,
                   color="#F0C866", alpha=.22, lw=0, zorder=0)
        ax.axvline(float(g.collision_time_ms)/1000., color="#B28624", lw=.8, linestyle="--", zorder=1)
    if not item["label_available"]:
        ax.text(.5, .53, "No valid mean candidate\nUnresolved terminal collision", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#6F4852")
        ntested = int(summary["test_membership_count"][ci])
    else:
        mask = a["common_mask"][ci]
        obj = np.where(mask, a["objective_xy"][ci, :, 1], np.nan)
        beh = np.where(mask, a["behavior_xy"][ci, :, 1], np.nan)
        ax.plot(t, obj, color=COLORS["objective"], lw=1.55, zorder=4)
        ax.plot(t, beh, color=COLORS["behavior"], lw=1.55, linestyle="--", zorder=5)
        for head, name in enumerate(("D_obj", "D_beh")):
            y = np.where(mask, summary["mean"][head, ci, :, 1], np.nan)
            sd = np.where(mask, summary["sd"][head, ci, :, 1], np.nan)
            ax.fill_between(t, y-sd, y+sd, color=COLORS[name], alpha=.11, lw=0, zorder=2)
            ax.plot(t, y, color=COLORS[name], lw=1.45, zorder=3)
        ntested = int(summary["test_membership_count"][ci])
    ax.set_title(f"{representation} | condition {item['condition_id']} | {ntested} test splits", loc="left", fontsize=10.5, pad=5)
    ax.set_xlim(*xlimits)
    ax.set_ylim(*ylimits)
    ax.set_xlabel("Time from aligned trial reference (s)", fontsize=9)
    ax.set_ylabel("Y (MWorks units)", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.grid(axis="y", lw=.4, alpha=.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _legend(fig, y):
    handles = [Line2D([0], [0], color=COLORS["objective"], lw=1.8, label="Objective ball path"),
               Line2D([0], [0], color=COLORS["D_obj"], lw=1.8, label="Objective-head reconstructed position"),
               Line2D([0], [0], color=COLORS["behavior"], lw=1.8, linestyle="--", label="Mean-endpoint candidate path"),
               Line2D([0], [0], color=COLORS["D_beh"], lw=1.8, label="Candidate-head reconstructed position"),
               Patch(facecolor="#CAD3DC", alpha=.4, label="Hidden interval"),
               Patch(facecolor="#F0C866", alpha=.4, label="Collision uncertainty"),
               Line2D([0], [0], color="#B94A48", linestyle=":", label="Estimated interception T")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, y), ncol=4, frameon=False,
               fontsize=9, handlelength=2.6, columnspacing=1.5, labelspacing=.65)


def _pdf_from_images(images, target, root):
    """ReportLab creates final PDFs; source PNG pages remain inspectable."""
    job = Path(root)/"figures"/(target.stem+"_pdf_job.json")
    save_json(job, {"images": [str(p.resolve()) for p in images], "target": str(target.resolve())})
    code = """import json,sys
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
d=json.load(open(sys.argv[1],encoding='utf-8'))
c=canvas.Canvas(d['target'],pagesize=A4,pageCompression=1)
c.setTitle('Mental-Pong all79 condition-average endpoint candidates: FA50 and GPFA50')
c.setAuthor('Mental-Pong local analysis')
W,H=A4
for p in d['images']:
 im=ImageReader(p);iw,ih=im.getSize();scale=min(W/iw,H/ih)
 c.drawImage(im,(W-iw*scale)/2,(H-ih*scale)/2,width=iw*scale,height=ih*scale)
 c.showPage()
c.save()
"""
    subprocess.run([str(BUNDLE_PYTHON), "-c", code, str(job)], check=True, capture_output=True, text=True)


def _atlases(data, summaries, selection, root, ylimits, xlimits):
    generated = []
    for animal, a in sorted(data.items()):
        rows = sorted([r for r in selection if r["animal"] == animal], key=lambda r: r["condition_id"])
        pages = []
        folder = root/"figures"/f"{animal}_condition_atlas_pages"
        folder.mkdir(parents=True, exist_ok=True)
        for start in range(0, 79, 4):
            fig, axes = plt.subplots(4, 2, figsize=(12, 16.1), squeeze=False)
            fig.subplots_adjust(left=.075, right=.982, top=.837, bottom=.093, hspace=.51, wspace=.17)
            page_rows = rows[start:start+4]
            for ri in range(4):
                if ri >= len(page_rows):
                    for ax in axes[ri]: ax.axis("off")
                    continue
                item = page_rows[ri]
                for column, rep in enumerate(REPRESENTATIONS):
                    _panel(axes[ri, column], a, item, rep, summaries[animal][rep], ylimits, xlimits)
                label_count = item["n_behavior_trials_in_mean"]
                bounds = axes[ri, 0].get_position()
                fig.text(.075, bounds.y1+.031, f"{item['bounce_class']} | behavior mean from {label_count} terminal records", fontsize=10.3, fontweight="bold", color="#34404C")
            fig.suptitle(f"{animal.title()} | 79 condition-average endpoint candidates", x=.075, ha="left", y=.978,
                         fontsize=18, fontweight="bold", color="#243344")
            fig.text(.075, .944, "All 79 physical identities. Same condition-average labels and y range for FA50 and GPFA50.", fontsize=10.5)
            _legend(fig, .927)
            fig.text(.075, .026, "Condition means on both sides: neural input and endpoint-constrained behavior labels. Shading = test-split stability, not trial variability.\n"
                     "Raw preprocessing causality failed; released-input filtering and training/test isolation are audited separately. T is not a verified feedback log.",
                     fontsize=8.4, color="#52606D", linespacing=1.6)
            fig.text(.982, .013, f"{start//4+1}/20", ha="right", fontsize=9, color="#52606D")
            page = folder/f"page_{start//4+1:02d}.png"
            fig.savefig(page, dpi=150, facecolor="white")
            plt.close(fig)
            pages.append(page)
        target = root/"figures"/f"{animal}_all79_condition_atlas.pdf"
        _pdf_from_images(pages, target, root)
        generated.append({"animal": animal, "path": str(target.resolve()), "sha256": sha(target),
                          "n_pages": len(pages), "n_conditions": len(rows),
                          "condition_ids": [r["condition_id"] for r in rows],
                          "page_images": [str(p.resolve()) for p in pages]})
    return generated


def _paired_distribution(round_summary, root):
    score = pd.DataFrame(round_summary)
    if "coordinate" in score:
        score = score[score.coordinate.eq("y")]
    epochs = ("full", "visible", "hidden", "post_bounce")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    for row, animal in enumerate(("mahler", "perle")):
        for col, metric in enumerate(("Delta_RMSE", "Delta_r")):
            ax = axes[row, col]
            for ri, rep in enumerate(REPRESENTATIONS):
                values = [score[(score.animal == animal) & (score.representation == rep) & (score.epoch == epoch)][metric].dropna().to_numpy()
                          for epoch in epochs]
                positions = np.arange(len(epochs))*3 + ri*.8
                bp = ax.boxplot(values, positions=positions, widths=.64, showfliers=True, patch_artist=True,
                                medianprops={"color": "#172332", "linewidth": 1.1},
                                flierprops={"markersize": 2., "markerfacecolor": "#789", "markeredgecolor": "none", "alpha": .35})
                color = "#87B7D5" if ri == 0 else "#D3A2BF"
                for box in bp["boxes"]: box.set_facecolor(color)
            ax.axhline(0, color="#596B7A", lw=.9, linestyle="--")
            ax.set_xticks(np.arange(len(epochs))*3+.4, ["Full", "Visible", "Hidden", "Post-collision"])
            ax.set_title(f"{animal.title()} | {'RMSE_obj - RMSE_beh' if col == 0 else 'r_beh - r_obj'}", loc="left", fontsize=13, fontweight="bold")
            ax.set_ylabel("MWorks coordinate units" if col == 0 else "Pearson r difference")
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paired own-target reconstruction differences", x=.07, y=.97, ha="left", fontsize=20, fontweight="bold")
    fig.text(.07, .92, "Positive favors the mean-endpoint candidate on its own target. Scores use held-out condition x time rows.", fontsize=10.5)
    fig.legend(handles=[Patch(facecolor="#87B7D5", label="FA50"), Patch(facecolor="#D3A2BF", label="GPFA50")],
               loc="upper right", bbox_to_anchor=(.96, .96), ncol=2, frameon=False)
    fig.text(.07, .026, "Overlapping condition splits describe readout stability; they are not independent experiments.\n"
             "Neural input and behavioral labels are condition averages. Results do not measure trial-specific neural prediction.", fontsize=9.5, color="#52606D")
    fig.subplots_adjust(left=.07, right=.975, top=.84, bottom=.13, hspace=.35, wspace=.2)
    path = root/"figures/paired_self_reconstruction_differences.png"
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return {"path": str(path.resolve()), "sha256": sha(path)}


def make_plots(data, round_summary, root, *, expected_rounds=100):
    root = Path(root)
    (root/"figures").mkdir(parents=True, exist_ok=True)
    order_path = root/"configs/figure_condition_order.json"
    selected = condition_figure_order(data)
    if order_path.exists():
        if json.loads(order_path.read_text(encoding="utf-8")) != selected:
            raise ValueError("All79 condition figure order changed")
    else:
        save_json(order_path, selected)
    summaries, source_audit = collect_test_predictions(data, root, expected_rounds=expected_rounds)
    ylimits, xlimits = _global_limits(data, summaries, selected)
    atlases = _atlases(data, summaries, selected, root, ylimits, xlimits)
    paired = _paired_distribution(round_summary, root)
    manifest = {"n_readout_rounds_per_animal_representation": expected_rounds,
                "condition_order_source": str(order_path.resolve()), "condition_order_sha256": sha(order_path),
                "condition_selection": "all79, none excluded by scores", "global_y_limits": list(ylimits), "global_time_limits_seconds": list(xlimits),
                "condition_order": selected, "atlases": atlases, "paired_distribution": paired,
                "prediction_sources": source_audit, "shadow_interpretation": "held-out split stability, not animal trial variability",
                "data_mode": "condition_mean_neural_and_behavior", "raw_preprocessing_causality": "fail"}
    save_json(root/"figures/figure_manifest.json", manifest)
    return manifest


def render_pdf_checks(root):
    """Render the actual two PDF files; human/model visual review follows."""
    from PIL import Image, ImageDraw
    root = Path(root)
    manifest = json.loads((root/"figures/figure_manifest.json").read_text(encoding="utf-8"))
    qa = root/"tmp/pdfs/final_pdf_qa"
    qa.mkdir(parents=True, exist_ok=True)
    checks = []
    for atlas in manifest["atlases"]:
        path = Path(atlas["path"])
        animal = atlas["animal"]
        script = "from pypdf import PdfReader;import json,sys;r=PdfReader(sys.argv[1]);print(json.dumps({'n_pages':len(r.pages),'page_sizes':[[float(p.mediabox.width),float(p.mediabox.height)] for p in r.pages]}))"
        metadata = json.loads(subprocess.check_output([str(BUNDLE_PYTHON), "-c", script, str(path)], text=True))
        assert metadata["n_pages"] == 20
        assert len(atlas["condition_ids"]) == len(set(atlas["condition_ids"])) == 79
        prefix = qa/f"{animal}_contact"
        subprocess.run([str(PDFTOPPM), "-r", "35", "-png", str(path), str(prefix)], check=True, capture_output=True)
        pagefiles = sorted(qa.glob(f"{animal}_contact-*.png"))
        assert len(pagefiles) == 20
        canvas = Image.new("RGB", (1000, 5*370), "#E6EBF0")
        drawer = ImageDraw.Draw(canvas)
        for i, page in enumerate(pagefiles):
            with Image.open(page) as image:
                image = image.convert("RGB")
                values = np.asarray(image)
                assert np.any(values < 100), "Rendered PDF page appears blank"
                image.thumbnail((246, 346))
                x, y = (i % 4)*250, (i//4)*370
                canvas.paste(image, (x+(250-image.width)//2, y))
                drawer.text((x+8, y+348), f"{animal} page {i+1}", fill="#243344")
        sheet = qa/f"{animal}_all_pages_contact.png"
        canvas.save(sheet)
        representative = []
        for page in (1, 10, 20):
            prefix = qa/f"{animal}_page_{page:02d}"
            subprocess.run([str(PDFTOPPM), "-f", str(page), "-l", str(page), "-r", "100", "-singlefile", "-png", str(path), str(prefix)],
                           check=True, capture_output=True)
            representative.append(str(prefix.with_suffix(".png").resolve()))
        checks.append({"animal": animal, "pdf_path": str(path.resolve()), "pdf_sha256": sha(path),
                       "n_pages": metadata["n_pages"], "n_conditions": 79, "all_pages_nonblank": True,
                       "contact_sheet": str(sheet.resolve()), "representative_pdf_renders": representative,
                       "visual_review_status": "pending_visual_inspection"})
    save_json(root/"results/figure_structural_checks.json", checks)
    return checks
