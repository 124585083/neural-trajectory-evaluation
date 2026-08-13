from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import resolve_project_path
from .data import DynamicSensoriumDataModule, LoaderBundle
from .models import SensoriumReferenceModel
from .training import collect_aligned_predictions


def stimulus_hash(session_path: Path, trial_index: int) -> str:
    digest = hashlib.sha1()
    with (session_path / "data" / "videos" / f"{trial_index}.npy").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_aligned_trials(
    model: SensoriumReferenceModel,
    bundle: LoaderBundle,
    data: DynamicSensoriumDataModule,
    config: dict,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray], list[str]]:
    burn_in = int(config["training"]["burn_in_frames"])
    amp = bool(config["training"]["amp"])
    targets, predictions, trial_meta = collect_aligned_predictions(model, bundle, device, burn_in, amp)
    output_root = resolve_project_path(config, Path("predictions") / model.name / "oracle")
    brain_root = resolve_project_path(config, Path("predictions") / "brain" / "oracle")
    output_root.mkdir(parents=True, exist_ok=True)
    brain_root.mkdir(parents=True, exist_ok=True)
    rate = float(config["data"]["sampling_rate_hz"])
    stimulus_ids: list[str] = []
    manifest_trials: list[dict[str, Any]] = []
    for target, prediction, meta in zip(targets, predictions, trial_meta):
        trial_index = int(meta["trial_index"])
        frame_index = np.asarray(meta["frame_index"], dtype=np.int64)
        timestamps = frame_index.astype(np.float64) / rate
        stim_id = stimulus_hash(data.session_path, trial_index)
        stimulus_ids.append(stim_id)
        common = {
            "timestamps": timestamps,
            "mask": np.ones(len(frame_index), dtype=np.bool_),
            "neuron_ids": data.metadata.neuron_ids,
            "session": np.asarray(data.session_key),
            "sampling_rate_hz": np.asarray(rate),
            "trial_index": np.asarray(trial_index),
            "stimulus_id": np.asarray(stim_id),
        }
        pred_path = output_root / f"trial_{trial_index:04d}.npz"
        np.savez(
            pred_path,
            predictions=prediction.astype(np.float32),
            model=np.asarray(model.name),
            checkpoint=np.asarray(checkpoint_path),
            **common,
        )
        brain_path = brain_root / f"trial_{trial_index:04d}.npz"
        if not brain_path.exists():
            np.savez(brain_path, responses=target.astype(np.float32), **common)
        manifest_trials.append(
            {
                "trial_index": trial_index,
                "prediction_path": str(pred_path),
                "brain_path": str(brain_path),
                "shape": list(prediction.shape),
                "stimulus_id": stim_id,
            }
        )
    manifest = {
        "schema": "one file per trial; predictions/responses [T,N]",
        "model": model.name,
        "session": data.session_key,
        "checkpoint": checkpoint_path,
        "neuron_count": data.n_neurons,
        "sampling_rate_hz": rate,
        "trials": manifest_trials,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, targets, predictions, stimulus_ids

