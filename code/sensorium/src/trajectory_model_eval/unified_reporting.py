"""Publish input availability separately from unexecuted scientific results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .unified_contract import validate_manifest


def write_inventory_status(manifest: dict, output_dir: Path) -> dict:
    """Derive coverage and question status from an audited inventory."""
    output_dir = Path(output_dir)
    validation = validate_manifest(manifest, require_test=False)
    status = {
        "schema_version": "sensorium_part3_status_v1",
        "split_sha256": manifest["split_sha256"], "config_sha256": manifest["config_sha256"],
        "inventory": "ready" if validation["ready"] else "blocked",
        "scientific_analysis": "not_executed", "new_gpfa_fit": "not_executed",
        "test_reliability_and_nulls": "not_executed", "response_and_trajectory_comparison": "not_executed",
        "reason_codes": manifest["eligibility"]["reasons"],
        "encoder_training": "not_performed", "historical_artifacts_overwritten": False,
        "counts": validation["counts"],
        "questions": {
            "Q1": "Not evaluated: no qualified common test comparison of response summaries and trajectories.",
            "Q2": "Not evaluated: no qualified common test scores for the two frozen encoders.",
            "Q3": "Not evaluated: qualified frozen-model test evaluation has not been executed; consult the inventory reason codes."
        }
    }
    if validation["ready"]:
        status["questions"] = {question: "Not executed: this command inventories inputs and does not evaluate models."
                               for question in ("Q1", "Q2", "Q3")}
    (output_dir / "analysis_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "question_status.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question", "status", "interpretation", "split_sha256"])
        writer.writeheader()
        for question, interpretation in status["questions"].items():
            writer.writerow({"question": question, "status": "not_executed", "interpretation": interpretation,
                             "split_sha256": manifest["split_sha256"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sessions = manifest.get("sessions", [manifest["session"]])
    train, validation_count, candidates, labeled = [], [], [], []
    for session in sessions:
        rows = [row for row in manifest["trials"] if row["session"] == session]
        train.append(sum(row["role"] == "train" for row in rows))
        validation_count.append(sum(row["role"] == "validation" for row in rows))
        candidates.append(sum(row["role"] == "test" for row in rows))
        labeled.append(sum(row["role"] == "test" and row.get("eligible_labeled_trial", False) for row in rows))
    fig, ax = plt.subplots(figsize=(9, 4.7), layout="constrained")
    y = np.arange(len(sessions))
    ax.barh(y, train, label="Train identities", color="#377eb8")
    ax.barh(y, validation_count, left=train, label="Validation identities (oracle)", color="#4daf4a")
    ax.barh(y, candidates, left=np.array(train) + validation_count, label="Test candidates", color="white", edgecolor="#777777", hatch="///")
    for index, total in enumerate(np.array(train) + validation_count + candidates):
        ax.text(total + 7, index, f"{labeled[index]} labeled test", va="center", fontsize=9)
    ax.set_yticks(y, [s.replace("dynamic", "").split("-Video")[0] for s in sessions])
    ax.invert_yaxis()
    ax.set_xlim(0, max(np.array(train) + validation_count + candidates) * 1.24)
    ax.set_xlabel("Trials, not independent animals")
    ax.set_title("Current Sensorium input coverage" + (": candidate tests lack neural labels" if not any(labeled) else ""))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "data_coverage.png", dpi=150)
    plt.close(fig)
    return status
