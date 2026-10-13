"""Identity and alignment checks for the separate Sensorium part-three run.

This module does not change historical loaders or normalize data. Runtime
eligibility is stricter than an inventory audit: an audit may document missing
labels, while an evaluation must reject them.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False, default=_json_value)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scientific_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _scientific_identity(item) for key, item in value.items()
                if key not in {"split_sha256", "path", "paths", "local_paths"}
                and not key.endswith("_path") and not key.endswith("_paths")}
    if isinstance(value, (list, tuple)):
        return [_scientific_identity(item) for item in value]
    return value


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Hash scientific identities without machine-local file locations."""
    return canonical_digest(_scientific_identity(manifest))


def _sha(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc


def validate_manifest(manifest: Mapping[str, Any], require_test: bool = True) -> dict[str, Any]:
    """Validate identities; report inventory blockers or reject them at runtime.

    Checkpoint histories remain authoritative even in an audit. Movie overlap
    is an explicit blocker only when the declared scope is unseen movies.
    """
    if manifest.get("schema_version") != "sensorium_part3_v1":
        raise ValueError("Unsupported or historical manifest schema")
    for key in ("config_sha256", "split_sha256"):
        _sha(manifest.get(key), key)
    if manifest["split_sha256"] != manifest_digest(manifest):
        raise ValueError("Split manifest digest mismatch")
    session = manifest.get("session")
    if not isinstance(session, str) or not session:
        raise ValueError("A GPFA session is required")
    checkpoints = manifest.get("checkpoint_identities", {})
    if set(checkpoints) != {"static", "dynamic"}:
        raise ValueError("Both frozen checkpoint identities are required")
    for name, checkpoint in checkpoints.items():
        _sha(checkpoint.get("sha256"), f"{name} checkpoint")
        if not checkpoint.get("trained_sessions"):
            raise ValueError(f"{name} has no recorded training sessions")
    neurons = manifest.get("neurons", {})
    indices, ids = neurons.get("indices", []), neurons.get("ids", [])
    if not indices or len(indices) != len(ids):
        raise ValueError("Neuron indices and IDs must have the same nonzero length")
    if (len(set(indices)) != len(indices) or len(set(map(str, ids))) != len(ids)
            or any(not isinstance(i, int) or i < 0 for i in indices)):
        raise ValueError("Neuron identities must be unique ordered indices and IDs")
    time = manifest.get("time", {})
    start, stop = time.get("frame_start"), time.get("frame_stop")
    if not isinstance(start, int) or not isinstance(stop, int) or start < 0 or stop <= start:
        raise ValueError("Invalid evaluation frame interval")
    if time.get("frame_rate_hz", 0) <= 0 or time.get("observation_step", 0) < 1:
        raise ValueError("Invalid time grid")
    prep = manifest.get("preprocessing", {})
    _sha(prep.get("sha256"), "preprocessing")
    eligibility = manifest.get("eligibility", {})
    if eligibility.get("status") not in {"ready", "blocked"}:
        raise ValueError("Eligibility must be ready or blocked")
    if not isinstance(eligibility.get("reasons"), list):
        raise ValueError("Eligibility reasons must be recorded")
    rows = manifest.get("trials", [])
    if not rows:
        raise ValueError("No trial identities were recorded")
    seen_indices, seen_ids, movies = set(), set(), {}
    blockers, counts = [], {role: 0 for role in ("train", "validation", "test")}
    selected_rows = []
    for row in rows:
        if not all(key in row for key in ("session", "dataset_index", "trial_id", "movie_id", "role",
                                          "source_tier", "response_status", "modalities_complete")):
            raise ValueError("Trial identity or data status is incomplete")
        index_key = (row["session"], row["dataset_index"])
        id_key = (row["session"], str(row["trial_id"]))
        if index_key in seen_indices or id_key in seen_ids:
            raise ValueError("A trial appears more than once or in multiple roles")
        seen_indices.add(index_key)
        seen_ids.add(id_key)
        if not isinstance(row["dataset_index"], int) or row["dataset_index"] < 0:
            raise ValueError("Invalid dataset index")
        role = row["role"]
        if role not in counts:
            raise ValueError("Invalid trial role")
        if not row["movie_id"]:
            blockers.append(f"Unknown movie identity {row['session']}:{row['dataset_index']}")
        else:
            movies.setdefault(str(row["movie_id"]), set()).add(role)
        counts[role] += 1
        for name, checkpoint in checkpoints.items():
            if row["session"] in checkpoint["trained_sessions"]:
                fit_tiers = checkpoint.get("training_tiers", ["train"])
                select_tiers = checkpoint.get("selection_tiers", ["oracle"])
                if row["source_tier"] in fit_tiers and role != "train":
                    raise ValueError(f"{name}: a historical fitting trial was relabeled {role}")
                if row["source_tier"] in select_tiers and role != "validation":
                    raise ValueError(f"{name}: a historical selection trial was relabeled {role}")
        if row["session"] == session:
            selected_rows.append(row)
            accepted = {"real", "verified_real", "real_nonzero"}
            if manifest.get("synthetic") is True:
                accepted.add("synthetic")
            if (row["response_status"] not in accepted or not row["modalities_complete"]
                    or not row.get("time_support_valid", True)
                    or not row.get("response_support_valid", True)):
                blockers.append(f"Unusable {role} trial {session}:{row['dataset_index']}")
    if not selected_rows:
        raise ValueError("No trials belong to the declared GPFA session")
    overlaps = [{"movie_id": movie, "roles": sorted(roles)}
                for movie, roles in sorted(movies.items()) if len(roles) > 1]
    scope = manifest.get("evaluation_scope", manifest.get("split_rule", {}).get("scope", "unseen_movie"))
    if overlaps and scope in {"unseen_movie", "unseen_movies", "unseen_movie_conditions"}:
        blockers.append("Movie conditions cross data roles under unseen-movie scope")
    for role in counts:
        if not any(row["role"] == role for row in selected_rows):
            blockers.append(f"The GPFA session has no {role} trials")
    if prep.get("status") != "verified" or not prep.get("provenance"):
        blockers.append("Encoder preprocessing provenance is unverified")
    for name, checkpoint in checkpoints.items():
        if session not in checkpoint["trained_sessions"]:
            blockers.append(f"{name} has no fitted readout for the GPFA session")
    blockers.extend(eligibility["reasons"])
    blockers = list(dict.fromkeys(blockers))
    if require_test and (blockers or eligibility["status"] != "ready"):
        raise ValueError("Scientific evaluation is blocked: " + "; ".join(blockers or ["inventory status is blocked"]))
    return {"counts": counts, "movie_role_overlaps": overlaps, "blockers": blockers,
            "ready": not blockers and eligibility["status"] == "ready"}


def validate_bundle(bundle: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Require complete pilot-session rows in manifest order and common axes."""
    validate_manifest(manifest)
    rows = [row for row in manifest["trials"] if row["session"] == manifest["session"]]
    if str(np.asarray(bundle.get("session", "")).item()) != manifest["session"]:
        raise ValueError("Prediction bundle session mismatch")
    columns = {"dataset_indices": "dataset_index", "trial_ids": "trial_id",
               "movie_ids": "movie_id", "roles": "role"}
    for array_key, row_key in columns.items():
        actual = np.asarray(bundle.get(array_key, []))
        expected = np.asarray([row[row_key] for row in rows])
        if actual.shape != expected.shape or not np.array_equal(actual.astype(str), expected.astype(str)):
            raise ValueError(f"Prediction bundle {array_key} order or identity mismatch")
    for key, expected in (("neuron_indices", manifest["neurons"]["indices"]),
                          ("neuron_ids", manifest["neurons"]["ids"]),
                          ("frame_indices", range(manifest["time"]["frame_start"], manifest["time"]["frame_stop"]))):
        actual, expected = np.asarray(bundle.get(key, [])), np.asarray(list(expected))
        if actual.shape != expected.shape or not np.array_equal(actual.astype(str), expected.astype(str)):
            raise ValueError(f"Prediction bundle {key} mismatch")
    shape = (len(rows), len(bundle["frame_indices"]), len(bundle["neuron_ids"]))
    precision = np.asarray(bundle.get("encoder_precision", []))
    if (precision.shape not in {(shape[2],), (shape[1], shape[2])}
            or not np.isfinite(precision).all() or np.any(precision <= 0)):
        raise ValueError("Prediction bundle encoder_precision must be positive finite [neuron] or [time, neuron]")
    for key in ("neural", "static", "dynamic"):
        values = np.asarray(bundle.get(key, []))
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Prediction bundle {key} must be finite and aligned to {shape}")


def cache_identity(manifest: Mapping[str, Any], gpfa_sha256: str) -> dict[str, Any]:
    _sha(gpfa_sha256, "GPFA identity")
    return {"schema_version": "sensorium_part3_cache_v1", "split_sha256": manifest["split_sha256"],
            "config_sha256": manifest["config_sha256"],
            "checkpoints": {key: value["sha256"] for key, value in manifest["checkpoint_identities"].items()},
            "preprocessing_sha256": manifest["preprocessing"]["sha256"], "gpfa_sha256": gpfa_sha256}


def _payload_digest(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(payload.items()):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError("Cache arrays cannot contain Python objects")
        digest.update(canonical_digest({"key": name, "shape": array.shape, "dtype": array.dtype.str}).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_cache(path: str | Path, payload: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    if "__identity__" in payload or "__payload_sha256__" in payload:
        raise ValueError("Payload contains a reserved cache key")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        np.savez_compressed(handle, **payload,
                            __identity__=json.dumps(identity, sort_keys=True, default=_json_value),
                            __payload_sha256__=_payload_digest(payload))


def load_cache(path: str | Path, expected_identity: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not {"__identity__", "__payload_sha256__"}.issubset(archive.files):
            raise ValueError("Unstamped historical prediction cache is not eligible")
        actual = json.loads(str(archive["__identity__"].item()))
        if canonical_digest(actual) != canonical_digest(expected_identity):
            raise ValueError("Cache identity does not match split, models, preprocessing, GPFA, and configuration")
        payload = {key: archive[key] for key in archive.files if not key.startswith("__")}
        if _payload_digest(payload) != str(archive["__payload_sha256__"].item()):
            raise ValueError("Prediction cache payload digest mismatch")
    return payload


@contextmanager
def frozen_torch_inference(model: Any):
    """Use evaluation mode and reject parameter or buffer changes, including BN.

    Original training flags are restored. A detected mutation is rolled back
    before raising, so a failed evaluation cannot contaminate later evaluations.
    """
    import torch

    flags = [(module, module.training) for module in model.modules()]
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.eval()
    try:
        with torch.inference_mode():
            yield model
    finally:
        after = model.state_dict()
        changed = set(after) != set(before) or any(
            key not in after or not torch.equal(value, after[key]) for key, value in before.items())
        if changed:
            model.load_state_dict(before)
        for module, training in flags:
            module.training = training
        if changed:
            raise RuntimeError("Frozen encoder parameters or buffers changed during inference")
