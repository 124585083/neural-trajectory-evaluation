"""Publish compact scientific provenance and actual stage status from run files."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .unified_contract import file_sha256


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _plot_training(run_root, public_root):
    """Plot recorded decisions without smoothing, extrapolation or test scores."""
    histories = {}
    for kind in ("static", "dynamic"):
        path = run_root / "encoders" / kind / "training_curve.json"
        if path.is_file() and _read(path):
            histories[kind] = _read(path)
    if not histories:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    for column, kind in enumerate(("static", "dynamic")):
        top, bottom = axes[0, column], axes[1, column]
        title = "Static" if kind == "static" else "Reduced Dynamic"
        if kind not in histories:
            top.text(.5, .5, "Not started", ha="center", va="center", transform=top.transAxes)
            bottom.text(.5, .5, "No validation scores", ha="center", va="center", transform=bottom.transAxes)
        else:
            rows = histories[kind]
            epochs = [row["epoch"] for row in rows]
            top.plot(epochs, [row["train_objective_mean"] for row in rows], color="#315f94")
            bottom.plot(epochs, [row["validation_logged_r"] for row in rows], color="#999999", label="Logging draw")
            bottom.plot(epochs, [row["validation_selection_r"] for row in rows], color="#315f94", label="Selection draw")
            selected = [row for row in rows if row["selected"]]
            bottom.scatter([row["epoch"] for row in selected], [row["validation_selection_r"] for row in selected],
                           marker="o", s=18, color="#a6492a", label="Checkpoint update", zorder=3)
            bottom.legend(frameon=False, fontsize=8)
        complete = (run_root / "encoders" / kind / "frozen_identity.json").is_file()
        top.set_title(f"{title} — {'completed' if complete else 'incomplete'}")
        top.set_ylabel("Mean training objective")
        bottom.set_ylabel("Validation correlation")
        for axis in (top, bottom):
            axis.set_xlabel("Completed epoch")
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("New train / validation split; no test scores enter training", fontsize=12)
    figure.savefig(public_root / "training_curves.png", dpi=160)
    plt.close(figure)


def publish_status(run_root, public_root):
    """Export tables and hashes, excluding raw arrays, checkpoints and private paths."""
    run_root, public_root = Path(run_root), Path(public_root)
    public_root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[4]
    try:
        artifact_root = run_root.resolve().relative_to(repository).as_posix()
    except ValueError:
        artifact_root = "external_run"
    status = {"protocol": "retrained_v1", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Held-out evaluation after retraining; test movies were explored in historical work",
              "stages": {}, "artifact_root": artifact_root,
              "large_artifacts": "Local or external only; raw data, normalization arrays, predictions and weights are not published"}
    copied = []

    def copy(source, target=None):
        source = run_root / source
        if source.is_file():
            destination = public_root / (target or source.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append(str(destination.relative_to(public_root)).replace("\\", "/"))

    copy("analysis_protocol.json", "configuration.json")
    if (run_root / "data/split_manifest.json").is_file():
        manifest = _read(run_root / "data/split_manifest.json")
        status["stages"]["split"] = {"status": "complete", "split_sha256": manifest["split_sha256"]}
        copy("data/split_manifest.json")
        copy("data/split_summary.csv")
    else:
        status["stages"]["split"] = {"status": "not_completed"}
    if (run_root / "data/preprocessing.json").is_file():
        preprocessing = _read(run_root / "data/preprocessing.json")
        status["stages"]["normalization"] = {"status": "complete", "fit_role": "train",
                                              "preprocessing_sha256": preprocessing["preprocessing_sha256"]}
        copy("data/preprocessing.json", "normalization_provenance.json")
    else:
        status["stages"]["normalization"] = {"status": "not_completed"}
    for kind in ("static", "dynamic"):
        summary_file = run_root / "encoders" / kind / "training_summary.json"
        progress_file = summary_file.with_name("status.json")
        if summary_file.is_file() or progress_file.is_file():
            record = _read(summary_file if summary_file.is_file() else progress_file)
            kept = {key: record[key] for key in (
                "status", "run_id", "identity_digest", "best_epoch", "validation_score", "best_score",
                "last_epoch", "epoch", "last_completed_epoch", "best_validation_r", "stop_reason", "elapsed_seconds", "parameter_counts", "best_sha256",
                "last_sha256", "best_state_sha256", "test_read_by_trainer") if key in record}
            status["stages"][kind] = kept
            _write(public_root / f"{kind}_training_status.json", kept)
            copy(f"encoders/{kind}/training_curve.json", f"{kind}_training_curve.json")
            copy(f"encoders/{kind}/selection_events.json", f"{kind}_selection_events.json")
            # Identity records contain portable configuration hashes and trial provenance.
            copy(f"encoders/{kind}/frozen_identity.json", f"{kind}_frozen_identity.json")
        else:
            status["stages"][kind] = {"status": "not_started"}
        smoke = run_root / "smoke" / kind / "training_summary.json"
        status["stages"][f"{kind}_real_smoke"] = {
            "status": _read(smoke)["status"] if smoke.is_file() else "not_completed"}
    gpfa = run_root / "gpfa/frozen_gpfa_identity.json"
    if gpfa.is_file():
        record = _read(gpfa)
        status["stages"]["gpfa"] = {"status": "complete", "gpfa_sha256": record["gpfa_sha256"],
                                      "fit_role": "train", "n_train_trials": record["n_train_trials"],
                                      "n_validation_trials": record["n_validation_trials"]}
        for filename in ("frozen_gpfa_identity.json", "gpfa_selection.csv", "gpfa_fit_history.csv"):
            copy(f"gpfa/{filename}")
    elif (run_root / "gpfa/status.json").is_file():
        status["stages"]["gpfa"] = _read(run_root / "gpfa/status.json")
    else:
        status["stages"]["gpfa"] = {"status": "not_started"}
    frozen = run_root / "frozen/frozen_analysis.json"
    if frozen.is_file():
        record = _read(frozen)
        status["stages"]["validation_control"] = {"status": "complete", "freeze_sha256": record["freeze_sha256"],
                                                   "response_matching": record["response_matching"]}
        copy("frozen/frozen_analysis.json")
        copy("frozen/response_matching_selection.csv")
    else:
        status["stages"]["validation_control"] = {"status": "pending_completed_encoders_and_gpfa"}
    evaluation = run_root / "evaluation/summary.json"
    status["stages"]["test_evaluation"] = {
        "status": _read(evaluation)["status"] if evaluation.is_file() else "not_completed"}
    if evaluation.is_file():
        evaluated = _read(evaluation)
        pilot = evaluated["pilot"]
        status["stages"]["test_reliability_null"] = pilot["reliability"]
        status["stages"]["test_main_comparison"] = {
            "status": "complete", "response_sessions": evaluated["response"]["n_sessions"],
            "trajectory_sessions": 1, "Q2": pilot["Q2"], "Q3": pilot["Q3"]}
        status["stages"]["test_response_matching_control"] = {
            **pilot["response_matching"], "status": "computed", "Q1": pilot["Q1"]}
        copy("evaluation/summary.json", "evaluation_summary.json")
        for source in sorted((run_root / "evaluation").rglob("*")):
            if source.is_file() and source.suffix in (".csv", ".png"):
                copy(source.relative_to(run_root), source.relative_to(run_root / "evaluation"))
    else:
        for name in ("test_reliability_null", "test_main_comparison", "test_response_matching_control"):
            status["stages"][name] = {"status": "not_completed"}
    _plot_training(run_root, public_root)
    _write(public_root / "status.json", status)
    files = {str(path.relative_to(public_root)).replace("\\", "/"): file_sha256(path)
             for path in sorted(public_root.rglob("*"))
             if path.is_file() and path.name != "generated_manifest.json"}
    _write(public_root / "generated_manifest.json", {"protocol": "retrained_v1",
                                                     "observed_at_utc": status["observed_at_utc"], "files": files})
    return status
