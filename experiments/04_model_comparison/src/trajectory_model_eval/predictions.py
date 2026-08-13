from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import json_dump, output_dir, resolve
from .protocol import load_protocol, prepare_protocol


def _checkpoint_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Accept both training checkpoints and public state-dict-only weights."""
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    return state if isinstance(state, dict) else payload


def _add_sources(config: dict[str, Any]) -> None:
    for root_key in ("phase1_root", "phase3_root"):
        source = resolve(config, config["references"][root_key]) / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))


def _predict_model(
    model: torch.nn.Module,
    loader: Any,
    data_key: str,
    neuron_indices: np.ndarray,
    frame_start: int,
    frame_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cuda")
    model = model.to(device).eval()
    predictions, targets, indices = [], [], []
    sampler_indices = np.asarray(list(loader.sampler), dtype=np.int64)
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            values = batch._asdict() if not isinstance(batch, dict) else batch
            images = values["videos"].to(device, non_blocking=True)
            kwargs = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in values.items()
                if key not in {"videos", "responses"}
            }
            output = model(images, data_key=data_key, **kwargs)
            retained = frame_stop - frame_start
            if output.shape[1] < retained:
                raise AssertionError(f"prediction has {output.shape[1]} frames, needs {retained}")
            prediction = output[:, -retained:, neuron_indices].float().cpu().numpy()
            response = (
                values["responses"][:, neuron_indices, frame_start:frame_stop]
                .permute(0, 2, 1)
                .float()
                .cpu()
                .numpy()
            )
            predictions.append(prediction)
            targets.append(response)
            count = len(prediction)
            indices.append(sampler_indices[cursor : cursor + count])
            cursor += count
    if cursor != len(sampler_indices):
        raise AssertionError("dataloader sampler and prediction count differ")
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(indices)


def generate_predictions(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("real checkpoint prediction requires CUDA")
    _add_sources(config)
    from trajectory_eval.official_dynamic import make_official_loaders
    from trajectory_eval.static_dynamic import build_model as build_static
    from trajectory_eval.static_dynamic import load_config as load_static_config
    from trajectory_param_match.experiment import build_official_model, load_config as load_dynamic_config

    protocol = load_protocol(config)
    neuron_indices = protocol["neuron_indices"].astype(np.int64)
    data = config["data"]
    frame_start, frame_stop = int(data["frame_start"]), int(data["frame_stop"])
    static_config = load_static_config(resolve(config, config["references"]["static_config"]))
    dynamic_config = load_dynamic_config(resolve(config, config["references"]["dynamic_config"]))
    loaders = make_official_loaders(
        static_config, cuda=False, batch_size=1, to_cut=False, offset=0
    )
    session = data["session"]
    loader = loaders["oracle"][session]

    static = build_static(static_config, loaders)
    static_payload = torch.load(
        resolve(config, config["references"]["static_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    static.load_state_dict(_checkpoint_state(static_payload), strict=True)
    static_prediction, neural, dataset_indices = _predict_model(
        static, loader, session, neuron_indices, frame_start, frame_stop
    )
    del static
    torch.cuda.empty_cache()
    gc.collect()

    dynamic = build_official_model(dynamic_config, loaders)
    dynamic_payload = torch.load(
        resolve(config, config["references"]["dynamic_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    dynamic.load_state_dict(_checkpoint_state(dynamic_payload), strict=True)
    dynamic_prediction, neural_check, dynamic_indices = _predict_model(
        dynamic, loader, session, neuron_indices, frame_start, frame_stop
    )
    if not np.array_equal(dataset_indices, dynamic_indices):
        raise AssertionError("Static and Dynamic trial order differs")
    if not np.allclose(neural, neural_check, rtol=0, atol=0):
        raise AssertionError("Static and Dynamic target tensors differ")
    del dynamic
    torch.cuda.empty_cache()

    condition_by_index = {
        int(index): int(condition)
        for index, condition in zip(protocol["oracle_indices"], protocol["oracle_conditions"])
    }
    conditions = np.asarray([condition_by_index[int(index)] for index in dataset_indices], dtype=np.int64)
    if np.any(np.asarray([index not in condition_by_index for index in dataset_indices])):
        raise AssertionError("oracle loader contains an unlocked trial")
    out = output_dir(config)
    np.savez_compressed(
        out / "oracle_predictions.npz",
        neural=neural.astype(np.float32),
        static=static_prediction.astype(np.float32),
        dynamic=dynamic_prediction.astype(np.float32),
        dataset_indices=dataset_indices,
        conditions=conditions,
        neuron_indices=neuron_indices,
        neuron_ids=protocol["neuron_ids"],
        frame_indices=np.arange(frame_start, frame_stop, dtype=np.int64),
    )
    summary = {
        "status": "oracle_predictions_complete",
        "session": session,
        "shape": list(neural.shape),
        "trials": int(len(neural)),
        "conditions": int(len(np.unique(conditions))),
        "neurons": int(len(neuron_indices)),
        "frames": int(frame_stop - frame_start),
        "static_checkpoint_validation": float(config["checkpoint_metadata"]["static_validation"]),
        "dynamic_checkpoint_validation": float(config["checkpoint_metadata"]["dynamic_validation"]),
        "normalization": "official Sensorium NeuroNormalizer (all response standard deviation)",
    }
    json_dump(out / "prediction_summary.json", summary)
    return summary
