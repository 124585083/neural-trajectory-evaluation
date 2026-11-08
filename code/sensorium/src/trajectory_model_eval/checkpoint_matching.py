"""Validation-only selection of an unmodified Dynamic model checkpoint.

Only per-neuron validation correlations are retained for every candidate.  This
allows a fixed common neuron cohort without keeping every prediction tensor.
The source model/GPFA records are provenance, not a newly hashed training run.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .traditional import column_correlation
from .unified_contract import canonical_digest, file_sha256


SCHEMA = "sensorium_checkpoint_matching_v1"
KIND = "validation_selected_dynamic_checkpoint"
COHORT_RULE = "all_completed_epochs_through_min_119_or_native_early_stop"
IDENTITY_KEYS = ("session", "dataset_indices", "trial_ids", "conditions", "movie_ids",
                 "roles", "neuron_indices", "neuron_ids", "frame_indices", "encoder_precision")
TRIAL_KEYS = ("dataset_indices", "trial_ids", "conditions", "movie_ids", "roles")
REQUIRED_SOURCE_KEYS = ("source_config_sha256", "source_freeze_sha256", "split_sha256",
                        "preprocessing_sha256", "gpfa_identity", "checkpoint_identities")


def array_digest(value):
    array = np.asarray(value)
    if array.dtype.kind in "OUS":
        return canonical_digest(array.astype(str).tolist())
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validation(bundle):
    roles = np.asarray(bundle["roles"]).astype(str)
    if not set(roles).issubset({"train", "validation"}) or not np.any(roles == "validation"):
        raise ValueError("Checkpoint selection accepts validation/development only; test is forbidden")
    mask = roles == "validation"
    result = {key: np.asarray(bundle[key])[mask] if key in TRIAL_KEYS else np.asarray(bundle[key])
              for key in IDENTITY_KEYS}
    for key in ("neural", "static", "matched_dynamic"):
        if key in bundle:
            result[key] = np.asarray(bundle[key])[mask]
    return result


def validation_reference(development):
    """Extract the fixed Static target and trial/time/neuron support, never test."""
    reference = _validation(development)
    neural, static = reference["neural"], reference["static"]
    if neural.ndim != 3 or static.shape != neural.shape or not np.isfinite([neural, static]).all():
        raise ValueError("Validation target and Static predictions need complete finite support")
    if neural.shape != (len(reference["roles"]), len(reference["frame_indices"]), len(reference["neuron_ids"])):
        raise ValueError("Validation identity axes differ from the response tensors")
    if len(np.unique(reference["dataset_indices"])) != len(neural):
        raise ValueError("Validation contains duplicate trial identities")
    precision = reference["encoder_precision"]
    if precision.shape not in ((neural.shape[-1],), neural.shape[1:]) or not np.isfinite(precision).all() or np.any(precision <= 0):
        raise ValueError("Validation response scale must be finite and positive")
    reference["static_per_neuron_r"] = column_correlation(neural.reshape(-1, neural.shape[-1]), static.reshape(-1, neural.shape[-1]))
    reference["reference_sha256"] = canonical_digest({key: array_digest(reference[key])
        for key in (*IDENTITY_KEYS, "neural", "static")})
    return reference


def _checkpoint_identity(identity):
    value = dict(identity)
    if not isinstance(value.get("epoch"), int) or value["epoch"] < 1 or not value.get("checkpoint_path"):
        raise ValueError("Every candidate needs a positive epoch and checkpoint path")
    for key in ("sha256", "state_sha256", "training_identity_digest"):
        digest = value.get(key, "")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Candidate {key} must be a SHA-256")
    return value


def validate_candidate_training(identity, source_identity):
    """Distinguish a new replay run from the immutable original training run."""
    original = source_identity["checkpoint_identities"]["dynamic"]
    training = identity.get("identity")
    if training is None:
        if identity["training_identity_digest"] != original["identity_digest"]:
            raise ValueError("Replay candidate requires its own complete training identity")
        return
    if canonical_digest(training) != identity["training_identity_digest"]:
        raise ValueError("Candidate training identity digest changed")
    if identity["training_identity_digest"] != original["identity_digest"]:
        if (training.get("source_training_identity_digest") != original["identity_digest"]
                or training.get("scope") != "checkpoint_candidate_replay"):
            raise ValueError("Candidate belongs to a different Dynamic training identity")
        for key, value in original.get("identity", {}).items():
            if key not in {"run_id", "scope", "training_code_sha256"} and training.get(key) != value:
                raise ValueError(f"Replay candidate changed source training setting {key}")


def score_candidate(reference, candidate_bundle, checkpoint_identity):
    """Reduce one aligned checkpoint prediction to neuron correlations."""
    candidate = _validation(candidate_bundle)
    for key in (*IDENTITY_KEYS, "neural"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Candidate validation {key} differs from the fixed reference")
    prediction = np.asarray(candidate["matched_dynamic"])
    if prediction.shape != reference["neural"].shape or not np.isfinite(prediction).all():
        raise ValueError("Candidate prediction must be finite on the full validation support")
    neural = reference["neural"]
    scores = column_correlation(neural.reshape(-1, neural.shape[-1]), prediction.reshape(-1, neural.shape[-1]))
    return {"checkpoint": _checkpoint_identity(checkpoint_identity),
            "reference_sha256": reference["reference_sha256"],
            "validation_prediction_sha256": array_digest(prediction),
            "per_neuron_r": [float(score) if np.isfinite(score) else None for score in scores]}


def _rank_candidates(candidates, static_scores, expected_epochs):
    epochs = [row["checkpoint"]["epoch"] for row in candidates]
    if len(epochs) != len(set(epochs)) or sorted(epochs) != list(expected_epochs):
        raise ValueError("Candidate cohort is incomplete, duplicated, or differs from the fixed epochs")
    static = np.asarray(static_scores, dtype=float)
    correlations = np.asarray([row["per_neuron_r"] for row in candidates], dtype=float)
    if correlations.shape != (len(candidates), len(static)):
        raise ValueError("Candidate neuron support differs")
    common = np.isfinite(static) & np.isfinite(correlations).all(axis=0)
    if not common.any():
        raise ValueError("No common finite neuron correlations across the fixed candidate cohort")
    target = float(static[common].mean())
    rows = []
    for row, values in zip(candidates, correlations):
        score = float(values[common].mean())
        rows.append({**row, "validation_static_response_r": target,
                     "validation_matched_dynamic_response_r": score, "absolute_gap": abs(score - target),
                     "n_shared_neurons": int(common.sum()), "n_total_neurons": len(static)})
    rows.sort(key=lambda row: (row["checkpoint"]["epoch"], row["checkpoint"]["sha256"]))
    selected = min(rows, key=lambda row: (row["absolute_gap"], row["checkpoint"]["epoch"], row["checkpoint"]["sha256"]))
    for row in rows:
        row["selected"] = row is selected
    return rows, selected, common


def validate_cohort_completion(proof, epochs, matching_config):
    unsigned = {key: value for key, value in proof.items() if key != "completion_sha256"}
    if proof.get("completion_sha256") != canonical_digest(unsigned):
        raise ValueError("Candidate cohort completion evidence changed")
    final = proof.get("final_epoch", 0)
    if (proof.get("candidate_cohort_rule") != COHORT_RULE or proof.get("maximum_candidate_epoch") != 119
            or proof.get("test_loaded") is not False or not 1 <= final <= 119
            or list(epochs) != list(range(1, final + 1))
            or proof.get("effective_candidate_epochs") != list(epochs)
            or matching_config.get("candidate_cohort_rule") != COHORT_RULE
            or proof.get("matching_config_sha256") != canonical_digest(matching_config)):
        raise ValueError("Candidate cohort rule, epoch support, or matching config differs")
    summary, checkpoint = proof["training_summary"], proof["last_checkpoint"]
    if (summary["last_epoch"] != final or checkpoint["epoch"] != final
            or summary.get("test_read_by_trainer") is not False
            or summary["identity_digest"] != proof["training_identity_digest"]
            or checkpoint["identity_digest"] != proof["training_identity_digest"]
            or summary["stop_reason"] != checkpoint["stop_reason"]):
        raise ValueError("Native training completion identities differ")
    if proof["completion_kind"] == "native_early_stop":
        if (summary["status"] != "training_complete" or summary["stop_reason"] != "patience_cycles_exhausted"
                or checkpoint["cycle"] != 4):
            raise ValueError("A shorter cohort requires completed native early stopping")
    elif proof["completion_kind"] == "epoch_cap_reached":
        if final != 119 or (summary["status"], summary["stop_reason"]) not in {
                ("incomplete_resumable", "bounded_invocation_resumable"),
                ("training_complete", "patience_cycles_exhausted")}:
            raise ValueError("Candidate epoch cap was not reached by a completed epoch")
    else:
        raise ValueError("Interrupted or arbitrary partial cohorts cannot complete selection")
    if final < 119 and proof["completion_kind"] != "native_early_stop":
        raise ValueError("Short candidate cohort lacks native stopping evidence")


def freeze_selection(reference, candidate_scores, source_identity, *, expected_epochs=range(1, 120),
                     tolerance=.005, output_dir=None, cohort_completion=None, matching_config=None):
    """Freeze the complete candidate cohort and deterministic validation decision."""
    if float(tolerance) != .005:
        raise ValueError("The checkpoint matching tolerance is fixed at 0.005")
    missing = set(REQUIRED_SOURCE_KEYS) - set(source_identity)
    if missing:
        raise ValueError(f"Missing source provenance: {sorted(missing)}")
    expected_epochs = sorted(map(int, expected_epochs))
    if not expected_epochs or len(set(expected_epochs)) != len(expected_epochs) or min(expected_epochs) < 1:
        raise ValueError("Expected candidate epochs must be unique positive integers")
    candidates = list(candidate_scores)
    for row in candidates:
        _checkpoint_identity(row["checkpoint"])
        if row["reference_sha256"] != reference["reference_sha256"]:
            raise ValueError("Candidate was scored on a different validation reference")
        validate_candidate_training(row["checkpoint"], source_identity)
    if len({row["checkpoint"]["training_identity_digest"] for row in candidates}) != 1:
        raise ValueError("All candidates must belong to one declared Dynamic training run")
    static = [float(value) if np.isfinite(value) else None for value in reference["static_per_neuron_r"]]
    rows, selected, common = _rank_candidates(candidates, static, expected_epochs)
    if cohort_completion is not None:
        validate_cohort_completion(cohort_completion, expected_epochs, matching_config)
        if cohort_completion["training_identity_digest"] != candidates[0]["checkpoint"]["training_identity_digest"]:
            raise ValueError("Candidate completion evidence belongs to another training run")
    record = {"schema_version": SCHEMA, "kind": KIND, "status": "selected", "selection_role": "validation",
        "test_used_for_selection": False, "test_loaded": False, "source_identity": source_identity,
        "selection_summary": "mean_per_neuron_pearson_over_trial_time",
        "neuron_support": "fixed_pilot512_common_finite_across_all_candidates",
        "tie_break": ["absolute_gap", "epoch", "sha256"], "candidate_epochs": expected_epochs,
        "reference_sha256": reference["reference_sha256"],
        "validation_support": {key: np.asarray(reference[key]).tolist() for key in IDENTITY_KEYS if key != "encoder_precision"},
        "encoder_precision_sha256": array_digest(reference["encoder_precision"]),
        "static_per_neuron_r": static, "shared_neuron_positions": np.flatnonzero(common).tolist(),
        "candidates": rows, "candidate_cohort_sha256": canonical_digest([row["checkpoint"] for row in rows]),
        "candidate_training_identity_digest": candidates[0]["checkpoint"]["training_identity_digest"],
        "selected_checkpoint": selected["checkpoint"], "tolerance": .005,
        "validation_static_response_r": selected["validation_static_response_r"],
        "validation_matched_dynamic_response_r": selected["validation_matched_dynamic_response_r"],
        "validation_absolute_gap": selected["absolute_gap"],
        "validation_shared_neurons": int(common.sum()), "total_gpfa_neurons_unchanged": len(static),
        "validation_matching_achieved": bool(selected["absolute_gap"] <= tolerance)}
    if cohort_completion is not None:
        record.update(candidate_cohort_rule=COHORT_RULE, candidate_final_epoch=expected_epochs[-1],
            cohort_completion=cohort_completion, matching_config=matching_config,
            matching_config_sha256=canonical_digest(matching_config),
            matching_config_source="code/sensorium/configs/checkpoint_matching_v1.json")
    record["selection_sha256"] = canonical_digest(record)
    if output_dir is not None:
        save_selection(output_dir, record)
    return record


def checkpoint_path(identity, checkpoint_root=None):
    path = Path(identity["checkpoint_path"])
    if path.is_absolute():
        return path
    if checkpoint_root is None:
        raise ValueError("Relative candidate checkpoint requires its matching artifact root")
    root = Path(checkpoint_root).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Candidate checkpoint path escapes the matching artifact root")
    return resolved


def validate_selection(record, source_identity=None, *, verify_checkpoint=False, checkpoint_root=None):
    """Check the decision, source identities and optional selected model bytes."""
    unsigned = {key: value for key, value in record.items() if key != "selection_sha256"}
    if record.get("selection_sha256") != canonical_digest(unsigned):
        raise ValueError("Checkpoint selection identity changed")
    if (record.get("schema_version") != SCHEMA or record.get("kind") != KIND
            or record.get("selection_role") != "validation" or record.get("test_used_for_selection") is not False
            or record.get("test_loaded") is not False or record.get("tolerance") != .005):
        raise ValueError("Invalid checkpoint selection method or test-dependent selection")
    if source_identity is not None and record["source_identity"] != source_identity:
        raise ValueError("Checkpoint control belongs to different source artifacts")
    source = record["source_identity"]
    if set(REQUIRED_SOURCE_KEYS) - set(source):
        raise ValueError("Checkpoint control is missing source provenance")
    if "cohort_completion" in record:
        validate_cohort_completion(record["cohort_completion"], record["candidate_epochs"], record["matching_config"])
        if (record["matching_config_sha256"] != canonical_digest(record["matching_config"])
                or record["candidate_final_epoch"] != record["candidate_epochs"][-1]
                or record["candidate_cohort_rule"] != COHORT_RULE
                or record["cohort_completion"]["training_identity_digest"] != record["candidate_training_identity_digest"]):
            raise ValueError("Frozen cohort completion or method provenance changed")
    for row in record["candidates"]:
        _checkpoint_identity(row["checkpoint"])
        if row["reference_sha256"] != record["reference_sha256"]:
            raise ValueError("Candidate validation reference changed")
        validate_candidate_training(row["checkpoint"], source)
        if row["checkpoint"]["training_identity_digest"] != record["candidate_training_identity_digest"]:
            raise ValueError("Candidate Dynamic training cohort changed")
    rows, selected, common = _rank_candidates(record["candidates"], record["static_per_neuron_r"], record["candidate_epochs"])
    if (rows != record["candidates"] or selected["checkpoint"] != record["selected_checkpoint"]
            or np.flatnonzero(common).tolist() != record["shared_neuron_positions"]
            or canonical_digest([row["checkpoint"] for row in rows]) != record["candidate_cohort_sha256"]
            or selected["absolute_gap"] != record["validation_absolute_gap"]
            or bool(selected["absolute_gap"] <= .005) != record["validation_matching_achieved"]):
        raise ValueError("Checkpoint selection decision or candidate cohort changed")
    if verify_checkpoint:
        selected_path = checkpoint_path(record["selected_checkpoint"], checkpoint_root)
        if file_sha256(selected_path) != record["selected_checkpoint"]["sha256"]:
            raise ValueError("Selected checkpoint payload changed")
        for evidence in record.get("cohort_completion", {}).get("evidence_files", []):
            path = checkpoint_path({"checkpoint_path": evidence["path"]}, checkpoint_root)
            if file_sha256(path) != evidence["sha256"]:
                raise ValueError("Candidate native-stop or protocol evidence changed")
    return record


def _candidate_csv(record):
    stream = io.StringIO(newline="")
    fields = ["epoch", "checkpoint_path", "sha256", "state_sha256", "training_identity_digest",
              "validation_static_response_r", "validation_matched_dynamic_response_r", "absolute_gap",
              "n_shared_neurons", "n_total_neurons", "validation_prediction_sha256", "selected"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in record["candidates"]:
        flat = {**row, **row["checkpoint"]}
        writer.writerow({key: flat[key] for key in fields})
    return stream.getvalue()


def save_selection(directory, record):
    validate_selection(record)
    directory = Path(directory)
    paths = [directory / name for name in ("checkpoint_selection.json", "checkpoint_candidates.csv")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Checkpoint selection artifacts are immutable")
    directory.mkdir(parents=True, exist_ok=True)
    if "cohort_completion" in record:
        for name, value in (("cohort_completion.json", record["cohort_completion"]),
                            ("checkpoint_matching_config.json", record["matching_config"])):
            path = directory / name
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != value:
                    raise ValueError("Adjacent cohort evidence or matching config differs from selection")
            else:
                path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    paths[1].write_text(_candidate_csv(record), encoding="utf-8", newline="")
    paths[0].write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def load_selection(directory, source_identity=None, *, verify_checkpoint=False):
    directory = Path(directory)
    record = json.loads((directory / "checkpoint_selection.json").read_text(encoding="utf-8"))
    validate_selection(record, source_identity, verify_checkpoint=verify_checkpoint, checkpoint_root=directory)
    if "cohort_completion" in record:
        for name, expected in (("cohort_completion.json", record["cohort_completion"]),
                               ("checkpoint_matching_config.json", record["matching_config"])):
            actual = json.loads((directory / name).read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError(f"Adjacent checkpoint evidence changed: {name}")
    if (directory / "checkpoint_candidates.csv").read_text(encoding="utf-8") != _candidate_csv(record):
        raise ValueError("Checkpoint candidate score table changed")
    return record
