from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .common import array_digest, json_dump, output_dir, resolve


def _phase2_import(config: dict[str, Any]) -> None:
    src = resolve(config, config["references"]["phase2_root"]) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


class ProtocolMismatchError(ValueError):
    """A saved protocol does not describe the requested scientific inputs."""


_OPERATIONAL_KEYS = {
    "output_dir", "checkpoint_dir", "log_dir", "prediction_dir", "timestamp",
    "created_at", "generated_at", "sensorium_root", "sensorium_source",
    "published_checkpoint", "comparison_checkpoint", "data_root", "root",
}


def _canonical(value: Any) -> Any:
    """Exclude named storage/provenance fields, retaining numerical settings."""
    if isinstance(value, dict):
        return {
            key: _canonical(item) for key, item in sorted(value.items())
            if not key.startswith("_") and key not in _OPERATIONAL_KEYS
            and key not in {"references", "artifacts", "protocol_guard"}
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"protocol identity requires an available file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Identify checkpoint bytes and scientific model configs, not their paths."""
    identities = {}
    for name, value in sorted(config["references"].items()):
        if name.endswith("checkpoint"):
            identities[name] = _sha256(resolve(config, value))
        elif name.endswith("config"):
            identities[name] = _canonical(yaml.safe_load(resolve(config, value).read_text(encoding="utf-8")))
    phase2 = resolve(config, config["references"]["phase2_root"]) / "src/trajectory_reliability"
    # Keep the exact GPFA implementation as part of new lock identity.
    identities["gpfa_implementation_sha256"] = _sha256(phase2 / "gpfa.py")
    identities["gpfa_selection_sha256"] = _sha256(phase2 / "selection.py")
    return identities


def _current_protocol(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read current inputs and derive the deterministic lock without writing."""
    _phase2_import(config)
    from trajectory_reliability.data import (
        deterministic_neuron_order,
        discover_oracle_conditions,
        load_session_metadata,
    )

    data = config["data"]
    session_path = resolve(config, data["sensorium_root"]) / data["session"]
    metadata = load_session_metadata(session_path)
    neurons = deterministic_neuron_order(metadata, int(data["neuron_seed"]))[
        : int(data["neurons"])
    ]
    train = metadata.indices("train")
    rng = np.random.default_rng(int(config["project"]["seed"]))
    shuffled = train.copy()
    rng.shuffle(shuffled)
    selected_count = max(2, int(round(len(train) * float(data["gpfa_train_fraction"]))))
    selected = np.sort(shuffled[:selected_count])
    calibration_count = max(
        1, int(round(len(selected) * float(data["gpfa_calibration_fraction"])))
    )
    split_order = selected.copy()
    rng.shuffle(split_order)
    calibration = np.sort(split_order[:calibration_count])
    fit = np.sort(split_order[calibration_count:])
    conditions = discover_oracle_conditions(
        metadata, float(data["oracle_stimulus_similarity"])
    )
    oracle_indices = np.asarray(
        [index for condition in conditions for index in condition.dataset_indices], dtype=np.int64
    )
    condition_by_index = {
        int(index): condition_index
        for condition_index, condition in enumerate(conditions)
        for index in condition.dataset_indices
    }
    oracle_conditions = np.asarray(
        [condition_by_index[int(index)] for index in oracle_indices], dtype=np.int64
    )
    result = {
        "status": "locked",
        "session": metadata.session_id,
        "total_session_train_trials": int(len(train)),
        "selected_gpfa_train_trials": int(len(selected)),
        "gpfa_train_fraction": float(len(selected) / len(train)),
        "fit_trials": int(len(fit)),
        "calibration_trials": int(len(calibration)),
        "neurons": int(len(neurons)),
        "oracle_trials": int(len(oracle_indices)),
        "oracle_conditions": int(len(conditions)),
        "oracle_repeat_counts": [int(len(condition.dataset_indices)) for condition in conditions],
        "frame_interval": [int(data["frame_start"]), int(data["frame_stop"] - 1)],
        "leakage_rule": "GPFA fit/calibration and scaling use selected official train trials only; encoding checkpoints were selected on oracle observations also used for scoring",
        "fingerprints": {
            "neuron_ids": array_digest(metadata.unit_ids[neurons]),
            "selected_train_indices": array_digest(selected),
            "oracle_indices_and_conditions": array_digest(oracle_indices, oracle_conditions),
        },
    }
    arrays = dict(
        neuron_indices=neurons, neuron_ids=metadata.unit_ids[neurons],
        selected_train_indices=selected, fit_indices=fit, calibration_indices=calibration,
        oracle_indices=oracle_indices, oracle_conditions=oracle_conditions,
    )
    files = [path.relative_to(session_path).as_posix() for path in (session_path / "meta").rglob("*.npy")]
    for index in np.union1d(selected, oracle_indices):
        files.append(f"data/responses/{int(index)}.npy")
    for index in oracle_indices:
        for group in ("videos", "behavior", "pupil_center"):
            files.append(f"data/{group}/{int(index)}.npy")
    # Check raw bytes as well as identities; changed responses must not retain a valid lock.
    result["validation"] = {
        "schema_version": 2,
        "scientific_config": _canonical(config),
        "source_identity": _source_identity(config),
        "data_files_sha256": {name: _sha256(session_path / name) for name in sorted(files)},
        "array_digests": {key: array_digest(value) for key, value in sorted(arrays.items())},
    }
    return result, arrays


def _differences(saved: Any, requested: Any, prefix: str = "validation") -> list[str]:
    if isinstance(saved, dict) and isinstance(requested, dict):
        differences = []
        for key in sorted(set(saved) | set(requested)):
            label = f"{prefix}.{key}"
            if key not in saved or key not in requested:
                differences.append(f"{label}: field added or removed")
            else:
                differences.extend(_differences(saved[key], requested[key], label))
        return differences
    if saved != requested:
        return [f"{prefix}: saved={saved!r}; requested={requested!r}"]
    return []


def prepare_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Create a new lock, or validate an existing one without overwriting it."""
    out = output_dir(config)
    if (out / "protocol_lock.npz").exists() or (out / "protocol_lock.json").exists():
        load_protocol(config)
        return json.loads((out / "protocol_lock.json").read_text(encoding="utf-8"))
    if any(out.iterdir()):
        raise FileExistsError("A new protocol requires an empty output directory. Keep existing results and use a new --output-dir.")
    result, arrays = _current_protocol(config)
    np.savez_compressed(out / "protocol_lock.npz", **arrays)
    json_dump(out / "protocol_lock.json", result)
    return result


def load_protocol(config: dict[str, Any]) -> dict[str, np.ndarray]:
    out = output_dir(config)
    path, metadata_path = out / "protocol_lock.npz", out / "protocol_lock.json"
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError("A complete protocol lock is required. Run 'lock' explicitly in a new output directory; existing locks are never regenerated.")
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=True) as values:
        arrays = {key: values[key] for key in values.files}
    validation = saved.get("validation", {})
    if validation.get("schema_version") != 2:
        if not config.get("protocol_guard", {}).get("allow_legacy_unverified", False):
            raise ProtocolMismatchError("legacy-unverified protocol: scientific config and source identities were not recorded. Use --allow-legacy-unverified only for an explicit diagnostic run; no validation metadata will be invented.")
        warnings.warn("Using legacy-unverified protocol for an explicitly requested diagnostic run. Scientific compatibility has not been verified.", RuntimeWarning, stacklevel=2)
        return arrays
    requested, _ = _current_protocol(config)
    differences = _differences(validation, requested["validation"])
    actual_digests = {key: array_digest(value) for key, value in sorted(arrays.items())}
    differences.extend(_differences(validation["array_digests"], actual_digests, "lock_arrays"))
    if differences:
        raise ProtocolMismatchError("Protocol mismatch; no files were changed:\n" + "\n".join(differences) + "\nIntentional changes require an explicit 'lock' command with a new --output-dir.")
    return arrays

