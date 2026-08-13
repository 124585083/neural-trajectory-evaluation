from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from .common import array_digest, json_dump, output_dir, resolve


def _phase2_import(config: dict[str, Any]) -> None:
    src = resolve(config, config["references"]["phase2_root"]) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def prepare_protocol(config: dict[str, Any]) -> dict[str, Any]:
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
        "leakage_rule": "GPFA fit/calibration and scaling use selected official train trials only; oracle is evaluation only",
        "fingerprints": {
            "neuron_ids": array_digest(metadata.unit_ids[neurons]),
            "selected_train_indices": array_digest(selected),
            "oracle_indices_and_conditions": array_digest(oracle_indices, oracle_conditions),
        },
    }
    out = output_dir(config)
    np.savez_compressed(
        out / "protocol_lock.npz",
        neuron_indices=neurons,
        neuron_ids=metadata.unit_ids[neurons],
        selected_train_indices=selected,
        fit_indices=fit,
        calibration_indices=calibration,
        oracle_indices=oracle_indices,
        oracle_conditions=oracle_conditions,
    )
    json_dump(out / "protocol_lock.json", result)
    return result


def load_protocol(config: dict[str, Any]) -> dict[str, np.ndarray]:
    path = output_dir(config) / "protocol_lock.npz"
    if not path.exists():
        prepare_protocol(config)
    with np.load(path, allow_pickle=True) as values:
        return {key: values[key] for key in values.files}

