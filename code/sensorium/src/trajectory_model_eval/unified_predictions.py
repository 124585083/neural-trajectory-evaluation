"""Export aligned predictions from the two existing frozen encoders.

Only the caller's audited trial list is used. This adapter never reads an old
oracle prediction archive and never updates encoder weights or normalization.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

import numpy as np
from sensorium_runtime.paths import SOURCE_ROOT, repository_path

from .unified_contract import (cache_identity, file_sha256, frozen_torch_inference,
                               save_cache, validate_bundle, validate_manifest)


def aligned_prediction(output, input_frames: int, frame_start: int, frame_stop: int):
    """Map valid convolution outputs to original frames, not the trial tail."""
    reduction = input_frames - output.shape[1]
    if reduction != 18:
        raise ValueError(f"Unexpected frozen encoder temporal reduction: {reduction}")
    if not reduction <= frame_start < frame_stop <= input_frames:
        raise ValueError("The frozen evaluation frames are not supported by this trial")
    return output[:, frame_start - reduction:frame_stop - reduction]


def predict_loader(model, loader, session, neurons, frame_start, frame_stop, device):
    import torch

    predictions, targets = [], []
    model = model.to(device)
    with frozen_torch_inference(model):
        for batch in loader:
            values = batch if isinstance(batch, dict) else batch._asdict()
            images = values["videos"].to(device)
            arguments = {key: value.to(device) if torch.is_tensor(value) else value
                         for key, value in values.items() if key not in {"videos", "responses"}}
            output = model(images, data_key=session, **arguments)
            output = aligned_prediction(output, images.shape[2], frame_start, frame_stop)
            predictions.append(output[:, :, neurons].detach().cpu().numpy())
            targets.append(values["responses"][:, neurons, frame_start:frame_stop]
                           .permute(0, 2, 1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def _verify_files(data_root, manifest, selected_rows):
    for group in (manifest["metadata_files"], manifest["preprocessing"]["statistics_files"]):
        for entry in group:
            if file_sha256(data_root / entry["path"]) != entry["sha256"]:
                raise ValueError(f"Audited data metadata changed: {entry['path']}")
    for row in selected_rows:
        for modality, source in row["source_files"].items():
            expected = row.get("source_sha256", {}).get(modality)
            if expected is None or file_sha256(data_root / source) != expected:
                raise ValueError(f"Trial content is missing an identity or changed: {source}")


def export_predictions(config, manifest, data_root, checkpoint_paths, output_dir,
                       device="cuda", sensorium_source=None, neuralpredictors_source=None):
    """Use official transforms and existing builders, with immutable state checks."""
    validate_manifest(manifest)
    if manifest.get("synthetic"):
        raise ValueError("Real encoder export cannot use an artificial manifest")
    import torch
    from torch.utils.data import DataLoader, Subset

    for source in (sensorium_source, neuralpredictors_source, SOURCE_ROOT):
        if source is not None:
            sys.path.insert(0, str(Path(source).resolve()))
    from trajectory_eval.official_dynamic import make_official_loaders
    from trajectory_eval.static_dynamic import build_model as build_static, load_config as static_config
    from trajectory_param_match.experiment import build_official_model as build_dynamic, load_config as dynamic_config
    from .unified_cli import UNFITTED_GPFA

    data_root, output_dir = Path(data_root), Path(output_dir)
    session = manifest["session"]
    rows = [row for row in manifest["trials"] if row["session"] == session]
    _verify_files(data_root, manifest, rows)
    configs = {}
    for name, load in (("static", static_config), ("dynamic", dynamic_config)):
        identity = manifest["checkpoint_identities"][name]
        if file_sha256(checkpoint_paths[name]) != identity["sha256"]:
            raise ValueError(f"The {name} checkpoint differs from the audited model")
        config_path = repository_path(identity["configuration"])
        if file_sha256(config_path) != identity["configuration_sha256"]:
            raise ValueError(f"The {name} model configuration changed after the audit")
        configs[name] = load(config_path)
        configs[name]["data"]["root"] = str(data_root.resolve())
    loaders = make_official_loaders(configs["static"], cuda=False, batch_size=1, to_cut=False, offset=0)
    dataset = loaders["train"][session].dataset
    # Subset fixes manifest order even though the historical train loader shuffles.
    indices = [row["dataset_index"] for row in rows]
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False)
    neurons = np.asarray(manifest["neurons"]["indices"], dtype=np.int64)
    start, stop = manifest["time"]["frame_start"], manifest["time"]["frame_stop"]
    bundle = {}
    for name, build in (("static", build_static), ("dynamic", build_dynamic)):
        model = build(configs[name], loaders)
        payload = torch.load(checkpoint_paths[name], map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        model.load_state_dict(state, strict=True)
        prediction, target = predict_loader(model, loader, session, neurons, start, stop, device)
        bundle[name] = prediction
        if "neural" in bundle and not np.array_equal(bundle["neural"], target):
            raise ValueError("Neural targets differ between encoder passes")
        bundle["neural"] = target
        del model, payload, state
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    std = np.asarray(np.load(data_root / session / "meta/statistics/responses/all/std.npy"), dtype=np.float64)
    floor = 0.01 * np.nanmean(std)
    precision = np.full_like(std, 1.0 / floor)
    valid = std > floor
    precision[valid] = 1.0 / std[valid]
    precision = precision[neurons, start:stop].T if precision.ndim == 2 else precision[neurons]
    for trial, row in enumerate(rows):
        raw = np.load(data_root / row["source_files"]["responses"], mmap_mode="r")[neurons, start:stop].T
        if not np.allclose(bundle["neural"][trial], raw * precision, rtol=3e-6, atol=3e-6):
            raise ValueError("Official transformed target does not match the locked response scale and frames")
    bundle.update(session=np.asarray(session), encoder_precision=precision,
                  dataset_indices=np.asarray(indices), trial_ids=np.asarray([row["trial_id"] for row in rows]),
                  movie_ids=np.asarray([row["movie_id"] for row in rows]),
                  conditions=np.asarray([row["movie_id"] for row in rows]), roles=np.asarray([row["role"] for row in rows]),
                  neuron_indices=neurons, neuron_ids=np.asarray(manifest["neurons"]["ids"]), frame_indices=np.arange(start, stop))
    validate_bundle(bundle, manifest)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a new empty prediction directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_cache(output_dir / "aligned_predictions.npz", bundle, cache_identity(manifest, UNFITTED_GPFA))
    summary = {"status": "frozen_predictions_exported", "shape": list(bundle["neural"].shape),
               "split_sha256": manifest["split_sha256"], "checkpoint_identities": manifest["checkpoint_identities"],
               "gpfa_stage": "not_fitted", "parameters_and_batchnorm_unchanged": True,
               "scale": "original frozen NeuroNormalizer; neural and both predictions use the same time-dependent precision"}
    (output_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
