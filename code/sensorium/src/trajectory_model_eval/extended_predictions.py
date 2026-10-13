from __future__ import annotations

import copy
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import json_dump, output_dir, resolve
from .predictions import _checkpoint_state, _predict_model
from .protocol import load_protocol


def _add_sources(config: dict[str, Any]) -> None:
    for root_key in ("phase1_root", "phase3_root"):
        source = resolve(config, config["references"][root_key]) / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))


def _dynamic_validation_by_epoch(config: dict[str, Any]) -> dict[int, float]:
    matching = config["validation_matching"]
    return {int(matching["dynamic_epoch"]): float(matching["dynamic_validation"])}


def _select_validation_matched_epoch(config: dict[str, Any]) -> dict[str, Any]:
    matching = config["validation_matching"]
    target = float(matching["static_validation"])
    candidates = _dynamic_validation_by_epoch(config)
    epoch, score = min(candidates.items(), key=lambda item: (abs(item[1] - target), item[0]))
    checkpoint = resolve(config, config["references"]["validation_matched_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "selection_split": matching["selection_split"],
        "static_validation": target,
        "dynamic_epoch": epoch,
        "dynamic_validation": score,
        "absolute_validation_gap": abs(score - target),
        "checkpoint": str(checkpoint),
    }


def ablate_temporal_state(
    state_dict: dict[str, torch.Tensor], retention: float
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must lie in [0, 1]")
    result = copy.deepcopy(state_dict)
    changed = []
    for name, value in result.items():
        if "conv_temporal" not in name or not name.endswith("weight"):
            continue
        if value.ndim != 5 or value.shape[2] <= 1:
            raise AssertionError(f"unexpected temporal kernel {name}: {tuple(value.shape)}")
        center = value.shape[2] // 2
        mask = torch.full_like(value, float(retention))
        mask[:, :, center] = 1.0
        result[name] = value * mask
        changed.append(name)
    if not changed:
        raise AssertionError("no temporal convolution weights were ablated")
    return result, changed


def generate_extended_predictions(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("extended checkpoint inference requires CUDA")
    _add_sources(config)
    from trajectory_eval.official_dynamic import make_official_loaders
    from trajectory_eval.static_dynamic import load_config as load_static_config
    from trajectory_param_match.experiment import build_official_model, load_config as load_dynamic_config

    out = output_dir(config)
    base_path = out / "oracle_predictions.npz"
    if not base_path.is_file():
        raise FileNotFoundError("run the Phase 4 primary prediction step first")
    with np.load(base_path, allow_pickle=True) as base_values:
        base = {key: base_values[key] for key in base_values.files}
    lock = load_protocol(config)
    neuron_indices = lock["neuron_indices"].astype(np.int64)
    data = config["data"]
    frame_start, frame_stop = int(data["frame_start"]), int(data["frame_stop"])
    static_config = load_static_config(resolve(config, config["references"]["static_config"]))
    dynamic_config = load_dynamic_config(resolve(config, config["references"]["dynamic_config"]))
    loaders = make_official_loaders(static_config, cuda=False, batch_size=1, to_cut=False, offset=0)
    session = data["session"]
    loader = loaders["oracle"][session]
    model = build_official_model(dynamic_config, loaders)

    selection = _select_validation_matched_epoch(config)
    validation_matched_state = torch.load(selection["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(validation_matched_state, strict=True)
    validation_matched, neural, indices = _predict_model(
        model, loader, session, neuron_indices, frame_start, frame_stop
    )
    if not np.array_equal(indices, base["dataset_indices"]) or not np.allclose(neural, base["neural"], rtol=0, atol=0):
        raise AssertionError("validation-matched checkpoint prediction is not aligned to the frozen oracle tensor")

    best_payload = torch.load(
        resolve(config, config["references"]["dynamic_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    best_state = _checkpoint_state(best_payload)
    retentions = (1.0, 0.75, 0.5, 0.25, 0.0)
    ablations: dict[str, np.ndarray] = {}
    changed_names: list[str] | None = None
    for retention in retentions:
        state, changed = ablate_temporal_state(best_state, retention)
        changed_names = changed if changed_names is None else changed_names
        if changed != changed_names:
            raise AssertionError("temporal kernel inventory changed across ablation levels")
        model.load_state_dict(state, strict=True)
        prediction, target, current_indices = _predict_model(
            model, loader, session, neuron_indices, frame_start, frame_stop
        )
        if not np.array_equal(current_indices, base["dataset_indices"]) or not np.allclose(target, base["neural"], rtol=0, atol=0):
            raise AssertionError("temporal ablation prediction is not aligned")
        ablations[f"retention_{retention:.2f}"] = prediction.astype(np.float32)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    np.savez_compressed(
        out / "extended_predictions.npz",
        validation_matched_dynamic=validation_matched.astype(np.float32),
        **ablations,
    )
    summary = {
        "status": "extended_predictions_complete",
        "validation_matched_checkpoint": selection,
        "temporal_ablation": {
            "definition": "multiply every off-center temporal-convolution weight by retention; preserve center slice and biases",
            "retentions": list(retentions),
            "changed_weight_tensors": changed_names,
            "strongest_level": "retention=0 removes off-center temporal-kernel weights but preserves spatial core/readout/shifter",
        },
        "shape": list(validation_matched.shape),
    }
    json_dump(out / "extended_prediction_summary.json", summary)
    return summary
