"""Inventory the data eligible for the unified frozen-model comparison.

This command reads the local data and model identities. It never trains a model,
relabels a used validation trial as a test, or treats zero-filled responses as
observed neural activity. Paths in the saved record are relative source keys.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from sensorium_runtime.paths import repository_path

from trajectory_reliability.data import _stimulus_signature

from .unified_contract import canonical_digest, manifest_digest


MODALITIES = ("videos", "responses", "behavior", "pupil_center")
TEST_TIERS = ("live_test_main", "final_test_main", "live_test_bonus", "final_test_bonus")
LOGGER = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _response_inventory(path: Path, n_neurons: int, frame_start: int, frame_stop: int) -> dict:
    if not path.is_file():
        return {"response_status": "missing", "response_shape": None, "response_values": 0,
                "response_nonzero_values": 0, "response_finite_values": 0, "response_support_valid": False,
                "evaluation_response_values": 0, "evaluation_finite_values": 0, "evaluation_nonzero_values": 0}
    response = np.load(path, mmap_mode="r", allow_pickle=False)
    # Every value is checked, including frames outside the evaluation window.
    # Chunking limits temporary boolean allocations for large neural populations.
    finite = 0
    nonzero = 0
    for first in range(0, response.shape[0] if response.ndim else 1, 256):
        block = response[first:first + 256] if response.ndim else response
        finite += int(np.count_nonzero(np.isfinite(block)))
        nonzero += int(np.count_nonzero(block))
    valid = response.ndim == 2 and response.shape[0] == n_neurons and response.shape[-1] >= frame_stop
    evaluated = response[:, frame_start:frame_stop] if response.ndim == 2 else np.asarray([])
    evaluation_finite = int(np.count_nonzero(np.isfinite(evaluated)))
    evaluation_nonzero = int(np.count_nonzero(evaluated))
    if not response.size:
        status = "missing"
    elif evaluation_finite != evaluated.size:
        status = "nonfinite"
    elif not evaluation_nonzero:
        status = "all_zero_placeholder"
    else:
        status = "real_nonzero"
    return {"response_status": status, "response_shape": list(response.shape),
            "response_values": int(response.size), "response_nonzero_values": nonzero,
            "response_finite_values": finite, "response_support_valid": bool(valid),
            "evaluation_response_values": int(evaluated.size), "evaluation_finite_values": evaluation_finite,
            "evaluation_nonzero_values": evaluation_nonzero}


def _assign_movies(rows: list[dict], signatures: list[np.ndarray | None], threshold: float) -> list[dict]:
    """Use the existing grayscale similarity definition across all data roles.

    Components may cross sessions. They track matching provided clips, not an
    unavailable upstream movie identifier. Behavior never enters this grouping.
    """
    good = [i for i, signature in enumerate(signatures) if signature is not None]
    parent = np.arange(len(rows))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    if good:
        values = np.stack([signatures[i] for i in good])
        # A block avoids retaining the complete all-trial similarity matrix.
        for first in range(0, len(good), 128):
            similarities = values[first:first + 128] @ values.T
            local, peers = np.where(similarities >= threshold)
            for left, right in zip(local + first, peers):
                if right <= left:
                    continue
                a, b = find(good[int(left)]), find(good[int(right)])
                if a != b:
                    parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in good:
        groups[find(index)].append(index)
    conditions = []
    for number, indices in enumerate(sorted(groups.values(), key=lambda values: values[0])):
        movie_id = f"movie_{number:05d}"
        for index in indices:
            rows[index]["movie_id"] = movie_id
        conditions.append({"movie_id": movie_id,
                           "members": [{"session": rows[i]["session"], "dataset_index": rows[i]["dataset_index"]}
                                       for i in indices],
                           "roles": sorted({rows[i]["role"] for i in indices})})
    return conditions


def _checkpoint_identities(config: dict, checkpoint_paths: dict | None) -> dict:
    identities = {}
    for name in ("static", "dynamic"):
        source = dict(config.get("models", {}).get(name, {}))
        # The record contains logical identifiers, never local model paths.
        allowed = ("trained_sessions", "training_tiers", "selection_tiers", "configuration_sha256",
                   "expected_sha256", "sha256", "source", "scientific_label", "checkpoint_kind")
        record = {key: source[key] for key in allowed if key in source}
        record.setdefault("trained_sessions", [])
        record.setdefault("training_tiers", ["train"])
        record.setdefault("selection_tiers", ["oracle"])
        if source.get("configuration"):
            configuration_path = repository_path(source["configuration"])
            configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
            record["configuration"] = str(source["configuration"])
            record["configuration_sha256"] = file_sha256(configuration_path)
            recorded_sessions = configuration["data"]["sessions"]
            if record["trained_sessions"] and record["trained_sessions"] != recorded_sessions:
                raise ValueError(f"{name} configured training sessions differ from the source record")
            record["trained_sessions"] = recorded_sessions
        if source.get("source_record"):
            source_path = repository_path(source["source_record"])
            record["source_record"] = str(source["source_record"])
            record["source_record_sha256"] = file_sha256(source_path)
        supplied = (checkpoint_paths or {}).get(name)
        if supplied is not None:
            path = Path(supplied)
            record["path"] = f"{name}/{path.name}"
            record["sha256"] = file_sha256(path)
            record["present"] = True
            expected = source.get("expected_sha256", source.get("sha256"))
            if expected and expected != record["sha256"]:
                raise ValueError(f"{name} checkpoint SHA256 does not match the configured identity")
        else:
            record["path"] = f"{name}/checkpoint"
            record["present"] = False
            record.setdefault("sha256", source.get("expected_sha256"))
        identities[name] = record
    return identities


def audit_dataset(data_root: Path, config: dict, output_dir: Path,
                  checkpoint_paths: dict | None = None) -> dict:
    """Write a traceable split inventory without executing a scientific fit."""
    data_root, output_dir = Path(data_root), Path(output_dir)
    data = config["data"]
    pilot = data["session"]
    sessions = list(data.get("sessions", [pilot]))
    if pilot not in sessions:
        raise ValueError("The GPFA session must be included in the audited sessions")
    frame_start, frame_stop = int(data["frame_start"]), int(data["frame_stop"])
    threshold = float(data.get("stimulus_similarity", data.get("oracle_stimulus_similarity", .999)))
    neurons_requested, seed = int(data["neurons"]), int(data.get("neuron_seed", 42))
    rows, signatures, metadata_files, statistics_files, session_neurons = [], [], [], [], {}
    for session in sessions:
        LOGGER.info("Checking response values, source hashes and input support for %s", session)
        path = data_root / session
        relative_metadata = ("meta/trials/tiers.npy", "meta/trials/trial_idx.npy", "meta/neurons/unit_ids.npy")
        for relative in relative_metadata:
            metadata_files.append({"path": f"{session}/{relative}", "sha256": file_sha256(path / relative)})
        tiers = np.load(path / relative_metadata[0], allow_pickle=False).astype(str)
        trial_ids = np.load(path / relative_metadata[1], allow_pickle=False).astype(str)
        unit_ids = np.load(path / relative_metadata[2], allow_pickle=False)
        if len(tiers) != len(trial_ids):
            raise ValueError(f"Trial metadata length differs in {session}")
        if len(unit_ids) < neurons_requested:
            raise ValueError(f"Too few neurons in {session} for the fixed neuron count")
        order = sorted(range(len(unit_ids)), key=lambda i: (hashlib.sha256(f"{seed}:{unit_ids[i]}".encode()).digest(), i))
        selected = order[:neurons_requested]
        session_neurons[session] = {"count_total": len(unit_ids), "indices": selected,
                                    "ids": unit_ids[selected].astype(str).tolist()}
        for stats_path in sorted((path / "meta/statistics").rglob("*.npy")):
            statistics_files.append({"path": f"{session}/{stats_path.relative_to(path).as_posix()}",
                                     "sha256": file_sha256(stats_path)})
        for index, tier in enumerate(tiers):
            if tier not in ("train", "oracle", *TEST_TIERS):
                raise ValueError(f"Unexpected source tier {tier!r} in the configured session {session}")
            files = {modality: path / "data" / modality / f"{index}.npy" for modality in MODALITIES}
            present = all(file.is_file() for file in files.values())
            response = _response_inventory(files["responses"], len(unit_ids), frame_start, frame_stop)
            row = {"session": session, "dataset_index": index, "trial_id": str(trial_ids[index]),
                   "movie_id": f"unresolved:{session}:{index}",
                   "role": "validation" if tier == "oracle" else "train" if tier == "train" else "test",
                   "source_tier": str(tier), "modalities_complete": present,
                   "source_files": {key: f"{session}/data/{key}/{index}.npy" for key in MODALITIES},
                   "source_sha256": {key: file_sha256(file) if file.is_file() else None for key, file in files.items()},
                   **response}
            support = bool(present and response["response_support_valid"])
            for modality in ("videos", "behavior", "pupil_center"):
                if files[modality].is_file():
                    value = np.load(files[modality], mmap_mode="r", allow_pickle=False)
                    support = support and value.ndim >= 2 and value.shape[-1] >= frame_stop
                    # The encoder needs the complete prefix, not only scored frames.
                    # Padding after frame_stop is excluded from the input contract.
                    support = support and bool(np.isfinite(value[..., :frame_stop]).all())
            row["time_support_valid"] = bool(support)
            row["eligible_labeled_trial"] = bool(support and response["response_status"] == "real_nonzero")
            try:
                signature = _stimulus_signature(path, index) if files["videos"].is_file() else None
                if signature is not None and not np.isfinite(signature).all():
                    raise ValueError("Nonfinite grayscale signature")
            except (ValueError, OSError) as error:
                signature = None
                # Shape/value failures are public scientific input failures, not local path messages.
                row["movie_identity_status"] = type(error).__name__
            else:
                row["movie_identity_status"] = "matched_by_grayscale_signature" if signature is not None else "missing"
            rows.append(row)
            signatures.append(signature)
            if (index + 1) % 100 == 0:
                LOGGER.info("Checked %d/%d trials in %s", index + 1, len(tiers), session)
    LOGGER.info("Grouping movie signatures across %d trials", len(rows))
    conditions = _assign_movies(rows, signatures, threshold)
    checkpoints = _checkpoint_identities(config, checkpoint_paths)
    preprocessing = {"status": "unverified",
                     "provenance": "Dataset-supplied meta/statistics/*/all files. The upstream fitting trial set is not documented in the available files.",
                     "sha256": canonical_digest(statistics_files), "statistics_files": statistics_files,
                     "response_formula": "normalized[n,t] = raw[n,t] * p_off[n,t]; p_off = 1/std when std > 0.01*nanmean(std), otherwise 1/(0.01*nanmean(std)); no response mean subtraction",
                     "fit_performed": False}
    reasons = ["preprocessing_provenance_unverified"]
    pilot_test = [row for row in rows if row["session"] == pilot and row["role"] == "test"]
    labeled_test = [row for row in pilot_test if row["eligible_labeled_trial"]]
    if not labeled_test:
        reasons.append("no_eligible_labeled_test_for_gpfa_session")
    if any(not record.get("present") for record in checkpoints.values()):
        reasons.append("checkpoint_file_not_verified")
    for name, record in checkpoints.items():
        if pilot not in record["trained_sessions"]:
            reasons.append(f"{name}_has_no_verified_readout_for_gpfa_session")
    if any(row["movie_identity_status"] != "matched_by_grayscale_signature" for row in rows):
        reasons.append("movie_identity_incomplete")
    crossing = [condition["movie_id"] for condition in conditions if "test" in condition["roles"]
                and any(role in condition["roles"] for role in ("train", "validation"))]
    generalization = data.get("scope", config.get("split", {}).get("generalization", "unseen_movies"))
    if crossing and generalization in ("unseen_movie", "unseen_movies", "unseen_movie_conditions"):
        reasons.append("test_movies_overlap_development_movies")
    manifest = {
        "schema_version": "sensorium_part3_v1", "config_sha256": canonical_digest(config),
        "session": pilot, "sessions": sessions, "evaluation_scope": generalization,
        "checkpoint_identities": checkpoints,
        "preprocessing": preprocessing, "neurons": session_neurons[pilot], "session_neurons": session_neurons,
        "time": {"frame_start": frame_start, "frame_stop": frame_stop,
                 "frame_rate_hz": float(data.get("frame_rate_hz", 30.0)),
                 "observation_step": int(config.get("gpfa", {}).get("effective_observation_step", 4))},
        "split_rule": {"train": "original train tier used by the selected encoders",
                       "validation": "entire original oracle tier used by checkpoint selection",
                       "test": "original live/final main/bonus candidate tiers; only observed nonzero responses can contribute",
                       "random_seed": int(config.get("project", {}).get("seed", 42)),
                       "generalization": generalization, "test_development_movie_overlaps": crossing},
        "movie_identity": {"method": "existing sampled grayscale signature, centered unit norm, connected components",
                           "similarity_threshold": threshold, "sample": "video[::4, ::4, :300:4]",
                           "source": "code/sensorium/src/trajectory_reliability/data.py::_stimulus_signature",
                           "not_source_clip_ids": True},
        "metadata_files": metadata_files, "conditions": conditions, "trials": rows,
        "eligibility": {"status": "blocked" if reasons else "ready", "reasons": reasons,
                        "scientific_analysis_executed": False,
                        "candidate_test_trials": len(pilot_test), "eligible_labeled_test_trials": len(labeled_test)},
    }
    manifest["split_sha256"] = manifest_digest(manifest)
    coverage = []
    for session in sessions:
        for tier in ("train", "oracle", *TEST_TIERS):
            selected_rows = [row for row in rows if row["session"] == session and row["source_tier"] == tier]
            counts = Counter(row["response_status"] for row in selected_rows)
            coverage.append({"session": session, "source_tier": tier,
                             "role": "validation" if tier == "oracle" else "train" if tier == "train" else "test",
                             "n_trials": len(selected_rows), "n_movies": len({row["movie_id"] for row in selected_rows
                                                                            if row["movie_identity_status"] == "matched_by_grayscale_signature"}),
                             "modalities_complete": sum(row["modalities_complete"] for row in selected_rows),
                             "time_support_valid": sum(row["time_support_valid"] for row in selected_rows),
                             "real_nonzero": counts["real_nonzero"], "all_zero_placeholder": counts["all_zero_placeholder"],
                             "nonfinite": counts["nonfinite"], "missing": counts["missing"],
                             "eligible_labeled_trials": sum(row["eligible_labeled_trial"] for row in selected_rows),
                             "response_values_scanned": sum(row["response_values"] for row in selected_rows),
                             "response_nonzero_values": sum(row["response_nonzero_values"] for row in selected_rows)})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (output_dir / "data_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    return manifest
