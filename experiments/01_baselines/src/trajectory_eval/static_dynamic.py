"""Full official static Sensorium CNN retrained on Dynamic Sensorium 2023.

The scientific model is the complete SENSORIUM 2022 Sensorium+ CNN. The only
video adapter reshapes frames into independent 2D samples and removes the first
18 outputs so that training targets exactly match the valid-time output of the
locked Factorized3D reference. The adapter has no trainable parameters.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import types
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml

from .config import PROJECT_ROOT
from .official_dynamic import (
    EXPECTED_NEURONS,
    EXPECTED_SESSIONS,
    FUNCTIONAL_NEURALPREDICTORS_COMMIT,
    GPUMonitor,
    OFFICIAL_SENSORIUM_COMMIT,
    Tee,
    make_official_loaders,
)

from neuralpredictors.measures import modules
from neuralpredictors.training import LongCycler
from nnfabrik.utility.nn_helpers import set_random_seed
from sensorium.models.make_model import make_video_model
from sensorium.training import video_training_loop
from sensorium.utility.scores import get_correlations


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "static_dynamic_sensorium2023.yaml"
STATIC_SENSORIUM_COMMIT = "c433fed25f234724fd9adf0cef3c260a2068b1fa"
EXPECTED_CORE_PARAMETERS = 50_624
EXPECTED_PARAMETER_COUNTS = {
    "core": 50_624,
    "readout": 2_763_106,
    "shifter": 285,
    "total": 2_814_015,
}
EXPECTED_CORE_OUTPUT = (64, 28, 56)
TEMPORAL_REDUCTION = 18

LOCKED_CORE = {
    "input_channels": 3,
    "hidden_channels": 64,
    "input_kern": 9,
    "hidden_kern": 7,
    "layers": 4,
    "gamma_input": 6.3831,
    "gamma_hidden": 0.0,
    "skip": 0,
    "final_nonlinearity": True,
    "bias": True,
    "momentum": 0.9,
    "pad_input": False,
    "batch_norm": True,
    "hidden_dilation": 1,
    "laplace_padding": None,
    "input_regularizer": "LaplaceL2norm",
    "stack": -1,
    "depth_separable": True,
    "linear": False,
    "attention_conv": False,
    "hidden_padding": None,
    "use_avg_reg": False,
}
LOCKED_READOUT = {
    "bias": True,
    "init_mu_range": 0.3,
    "init_sigma": 0.1,
    "gamma_readout": 0.0076,
    "gauss_type": "full",
    "grid_mean_predictor": {
        "type": "cortex",
        "input_dimensions": 2,
        "hidden_layers": 1,
        "hidden_features": 30,
        "final_tanh": True,
    },
    "share_features": False,
    "share_grid": False,
    "shared_match_ids": None,
    "gamma_grid_dispersion": 0.0,
}
LOCKED_SHIFTER = {
    "gamma_shifter": 0.0,
    "shift_layers": 3,
    "input_channels_shifter": 2,
    "hidden_channels_shifter": 5,
}


def _json(value: Any, *, indent: int | None = None) -> str:
    def default(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"cannot serialize {type(item).__name__}")

    return json.dumps(value, ensure_ascii=False, indent=indent, default=default)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def _assert_mapping(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise AssertionError(f"{label}.{key} is missing")
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                raise AssertionError(f"{label}.{key} must be a mapping")
            _assert_mapping(actual_value, expected_value, f"{label}.{key}")
        elif actual_value != expected_value:
            raise AssertionError(f"{label}.{key}: expected {expected_value!r}, got {actual_value!r}")


def audit_locked_config(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    if tuple(data["sessions"]) != EXPECTED_SESSIONS:
        raise AssertionError("the five-session Dynamic Sensorium scope changed")
    if int(data["frames"]) != 80 or float(data["scale"]) != 1.0:
        raise AssertionError("temporal window or native 36x64 resolution scale changed")
    if not data["include_behavior"] or not data["include_pupil_centers"]:
        raise AssertionError("behavior or pupil input was removed")
    if int(data["physical_batch_size_per_session"]) != 8 or int(data["effective_batch_size"]) != 40:
        raise AssertionError("physical/effective batch lock changed")
    model = config["model"]
    _assert_mapping(model["core"], LOCKED_CORE, "model.core")
    _assert_mapping(model["readout"], LOCKED_READOUT, "model.readout")
    _assert_mapping(model["shifter"], LOCKED_SHIFTER, "model.shifter")
    if model["core_type"] != "2D" or model["use_gru"] or not model["use_shifter"]:
        raise AssertionError("static/GRU/shifter model lock changed")
    adapter = model["temporal_adapter"]
    if adapter != {"type": "framewise_reshape_and_crop", "trainable_parameters": 0, "crop_leading_frames": 18}:
        raise AssertionError("zero-parameter temporal adapter changed")
    training = config["training"]
    expected_training = {
        "optimizer": "AdamW",
        "learning_rate": 0.005,
        "max_epochs": 200,
        "loss": "PoissonLoss",
        "average_loss": False,
        "scale_loss": True,
        "gradient_accumulation_sessions": 5,
        "patience": 5,
        "interval": 1,
        "tolerance": 1e-6,
        "restore_best": True,
        "lr_decay_steps": 4,
        "lr_decay_factor": 0.3,
        "min_lr": 1e-4,
        "mixed_precision": False,
    }
    _assert_mapping(training, expected_training, "training")
    if int(config["evaluation"]["burn_in_frames"]) != 50:
        raise AssertionError("official burn-in changed")
    return {
        "status": "pass",
        "scientific_label": config["project"]["scientific_label"],
        "sessions": list(EXPECTED_SESSIONS),
        "neuron_counts": list(EXPECTED_NEURONS),
        "core_parameters": EXPECTED_CORE_PARAMETERS,
        "core_output_per_frame": list(EXPECTED_CORE_OUTPUT),
        "temporal_adapter_parameters": 0,
        "temporal_reduction": TEMPORAL_REDUCTION,
    }


class FramewiseStaticVideoEncoder(nn.Module):
    """Apply an official 2D Sensorium encoder independently to every frame."""

    def __init__(self, source_model: nn.Module, temporal_reduction: int = TEMPORAL_REDUCTION) -> None:
        super().__init__()
        self.core = source_model.core
        self.readout = source_model.readout
        self.shifter = source_model.shifter
        self.temporal_reduction = int(temporal_reduction)
        self.output_nonlinearity = nn.ELU()

    def predict_all_frames(
        self,
        inputs: torch.Tensor,
        *,
        data_key: str,
        pupil_center: torch.Tensor,
        trial_idx: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
        detach_core: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(f"inputs must be [B,C,T,H,W], got {tuple(inputs.shape)}")
        if pupil_center is None:
            raise ValueError("pupil_center is required by the locked MLP shifter")
        batch, channels, time_points, height, width = inputs.shape
        frames = inputs.transpose(1, 2).reshape(batch * time_points, channels, height, width)
        features = self.core(frames)
        if detach_core:
            features = features.detach()
        flat_pupil = pupil_center.transpose(1, 2).reshape(batch * time_points, pupil_center.shape[1])
        if shift is None:
            shift = self.shifter[data_key](flat_pupil, trial_idx)
        rates = self.readout(features, data_key=data_key, shift=shift, **kwargs)
        rates = self.output_nonlinearity(rates) + 1.0
        return rates.reshape(batch, time_points, -1)

    def forward(
        self,
        inputs: torch.Tensor,
        *args: Any,
        data_key: str | None = None,
        behavior: torch.Tensor | None = None,
        pupil_center: torch.Tensor | None = None,
        trial_idx: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
        detach_core: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if data_key is None:
            raise ValueError("data_key is required for the multi-session readout")
        rates = self.predict_all_frames(
            inputs,
            data_key=data_key,
            pupil_center=pupil_center,
            trial_idx=trial_idx,
            shift=shift,
            detach_core=detach_core,
        )
        if rates.shape[1] <= self.temporal_reduction:
            raise ValueError("input video is too short for the locked comparison crop")
        return rates[:, self.temporal_reduction :, :]

    def regularizer(self, data_key: str | None = None) -> torch.Tensor:
        core_regularizer = self.core.regularizer()
        if isinstance(core_regularizer, tuple):
            core_regularizer = sum(core_regularizer)
        readout_regularizer = self.readout.regularizer(data_key) if data_key is not None else 0
        shifter_regularizer = self.shifter.regularizer(data_key) if data_key is not None else 0
        return core_regularizer + readout_regularizer + shifter_regularizer


def build_model(config: dict[str, Any], dataloaders: dict[str, dict[str, Any]]) -> FramewiseStaticVideoEncoder:
    model_config = config["model"]
    source = make_video_model(
        dataloaders,
        int(config["project"]["seed"]),
        core_dict=copy.deepcopy(model_config["core"]),
        core_type="2D",
        readout_dict=copy.deepcopy(model_config["readout"]),
        readout_type="gaussian",
        use_gru=False,
        gru_dict=None,
        use_shifter=True,
        shifter_dict=copy.deepcopy(model_config["shifter"]),
        shifter_type="MLP",
        deeplake_ds=False,
    )
    return FramewiseStaticVideoEncoder(source, temporal_reduction=TEMPORAL_REDUCTION)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "core": sum(parameter.numel() for parameter in model.core.parameters()),
        "readout": sum(parameter.numel() for parameter in model.readout.parameters()),
        "shifter": sum(parameter.numel() for parameter in model.shifter.parameters()),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }


def audit_architecture(config: dict[str, Any], *, save: bool = True) -> dict[str, Any]:
    locked = audit_locked_config(config)
    dataloaders = make_official_loaders(config, cuda=False, batch_size=1)
    model = build_model(config, dataloaders)
    counts = parameter_counts(model)
    if counts != EXPECTED_PARAMETER_COUNTS:
        raise AssertionError(f"parameter lock failed: {counts} != {EXPECTED_PARAMETER_COUNTS}")
    if len(model.core.features) != 4 or list(model.core.stack) != [3]:
        raise AssertionError("static core depth/readout-stack lock failed")
    named_parameter_roots = {name.split(".", 1)[0] for name, _ in model.named_parameters()}
    if named_parameter_roots != {"core", "readout", "shifter"}:
        raise AssertionError(f"the temporal adapter acquired parameters: {named_parameter_roots}")
    data_keys = tuple(dataloaders["train"].keys())
    outdims = tuple(model.readout[key].outdims for key in data_keys)
    if data_keys != EXPECTED_SESSIONS or outdims != EXPECTED_NEURONS:
        raise AssertionError("session/neuron readout lock failed")

    first_key = data_keys[0]
    batch = next(iter(dataloaders["oracle"][first_key]))
    model.eval()
    with torch.inference_mode():
        all_frames = model.predict_all_frames(batch.videos, data_key=first_key, pupil_center=batch.pupil_center)
        output = model(batch.videos, data_key=first_key, behavior=batch.behavior, pupil_center=batch.pupil_center)
        features = model.core(batch.videos[:, :, 0])
    if tuple(batch.videos.shape[1:]) != (3, 80, 36, 64):
        raise AssertionError(f"input lock failed: {tuple(batch.videos.shape)}")
    if tuple(features.shape[1:]) != EXPECTED_CORE_OUTPUT:
        raise AssertionError(f"core output lock failed: {tuple(features.shape)}")
    if tuple(all_frames.shape[1:]) != (80, EXPECTED_NEURONS[0]):
        raise AssertionError(f"all-frame output mismatch: {tuple(all_frames.shape)}")
    if tuple(output.shape[1:]) != (62, EXPECTED_NEURONS[0]):
        raise AssertionError(f"aligned output mismatch: {tuple(output.shape)}")
    if not torch.equal(output, all_frames[:, TEMPORAL_REDUCTION:]):
        raise AssertionError("the static comparison crop is not an exact view of frames 18..79")

    # In evaluation mode, permuting frames and undoing the permutation must
    # preserve every prediction. This is the explicit no-temporal-leakage test.
    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(batch.videos.shape[2], generator=generator)
    inverse = torch.argsort(permutation)
    with torch.inference_mode():
        permuted = model.predict_all_frames(
            batch.videos[:, :, permutation],
            data_key=first_key,
            pupil_center=batch.pupil_center[:, :, permutation],
        )[:, inverse]
    max_permutation_error = float((permuted - all_frames).abs().max())
    if max_permutation_error > 2e-6:
        raise AssertionError(f"static frame permutation invariance failed: {max_permutation_error}")

    audit = {
        **locked,
        "status": "pass",
        "static_sensorium_commit": STATIC_SENSORIUM_COMMIT,
        "dynamic_sensorium_commit": OFFICIAL_SENSORIUM_COMMIT,
        "neuralpredictors_commit": FUNCTIONAL_NEURALPREDICTORS_COMMIT,
        "parameter_counts": counts,
        "sessions": list(data_keys),
        "neuron_counts": list(outdims),
        "input_shape": list(batch.videos.shape[1:]),
        "all_frame_output_shape": list(all_frames.shape[1:]),
        "comparison_output_shape": list(output.shape[1:]),
        "core_output_per_frame": list(features.shape[1:]),
        "compared_source_frames": [TEMPORAL_REDUCTION, 79],
        "max_frame_permutation_error": max_permutation_error,
        "model": str(model),
    }
    if save:
        artifact_dir = resolve_path(config, "artifacts/static_dynamic_sensorium2023")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "architecture_audit.json").write_text(_json(audit, indent=2), encoding="utf-8")
        (artifact_dir / "model_architecture.txt").write_text(str(model) + "\n", encoding="utf-8")
    return audit


def training_objective(
    model: FramewiseStaticVideoEncoder,
    train_loaders: dict[str, Any],
    data_key: str,
    data: Any,
    criterion: nn.Module,
    device: str,
) -> torch.Tensor:
    batch_args = list(data)
    batch_kwargs = data._asdict() if not isinstance(data, dict) else data
    loss_scale = np.sqrt(len(train_loaders[data_key].dataset) / batch_args[0].shape[0])
    core_regularizer = model.core.regularizer()
    regularizers = (sum(core_regularizer) if isinstance(core_regularizer, tuple) else core_regularizer) + model.readout.regularizer(data_key)
    prediction = model(batch_args[0].to(device), data_key=data_key, **batch_kwargs)
    targets = batch_args[1].transpose(2, 1)[:, -prediction.shape[1] :, :].to(device)
    if prediction.shape[1] != 62 or targets.shape != prediction.shape:
        raise AssertionError(f"static/dynamic target alignment failed: {tuple(prediction.shape)} vs {tuple(targets.shape)}")
    objective = loss_scale * criterion(prediction, targets) + regularizers
    if not torch.isfinite(objective):
        raise FloatingPointError(f"non-finite objective for {data_key}")
    return objective


class LimitedLoader:
    def __init__(self, loader: Any, batches: int = 1) -> None:
        self.loader = loader
        self.dataset = loader.dataset
        self.batches = min(int(batches), len(loader))

    def __iter__(self):
        from itertools import islice

        return iter(islice(iter(self.loader), self.batches))

    def __len__(self) -> int:
        return self.batches


def smoke_test(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("static-on-dynamic smoke test requires CUDA")
    audit = audit_architecture(config)
    set_random_seed(int(config["project"]["seed"]))
    dataloaders = make_official_loaders(config, cuda=True)
    model = build_model(config, dataloaders).to("cuda").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    criterion = modules.PoissonLoss(avg=False)
    optimizer.zero_grad(set_to_none=True)
    losses: dict[str, float] = {}
    for data_key, loader in dataloaders["train"].items():
        data = next(iter(loader))
        objective = training_objective(model, dataloaders["train"], data_key, data, criterion, "cuda")
        objective.backward()
        losses[data_key] = float(objective.detach().cpu())
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).cpu())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise AssertionError(f"invalid gradient norm: {gradient_norm}")
    optimizer.step()

    artifact_dir = resolve_path(config, "artifacts/static_dynamic_sensorium2023")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "smoke_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    reloaded = build_model(config, dataloaders).to("cuda")
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cuda", weights_only=True))
    limited_oracle = {key: LimitedLoader(loader) for key, loader in dataloaders["oracle"].items()}
    reload_score = float(get_correlations(reloaded, limited_oracle, device="cuda", as_dict=False, per_neuron=False))
    result = {
        "status": "pass_smoke_only_not_a_training_result",
        "sessions_checked": list(losses),
        "losses": losses,
        "gradient_norm": gradient_norm,
        "reloaded_official_metric": reload_score,
        "checkpoint": str(checkpoint_path),
        "parameter_counts": audit["parameter_counts"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (artifact_dir / "smoke_test.json").write_text(_json(result, indent=2), encoding="utf-8")
    return result


def benchmark(config: dict[str, Any], *, measured_microbatches: int = 200, warmup_microbatches: int = 20) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("static-on-dynamic benchmark requires CUDA")
    audit_architecture(config)
    set_random_seed(int(config["project"]["seed"]))
    dataloaders = make_official_loaders(config, cuda=True)
    model = build_model(config, dataloaders).to("cuda").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    criterion = modules.PoissonLoss(avg=False)
    accumulation = len(dataloaders["train"])
    cycler = iter(LongCycler(dataloaders["train"]))
    total = warmup_microbatches + measured_microbatches
    timings: list[float] = []
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with GPUMonitor() as monitor:
        for index in range(total):
            try:
                data_key, data = next(cycler)
            except StopIteration:
                cycler = iter(LongCycler(dataloaders["train"]))
                data_key, data = next(cycler)
            torch.cuda.synchronize()
            started = time.perf_counter()
            objective = training_objective(model, dataloaders["train"], data_key, data, criterion, "cuda")
            objective.backward()
            if (index + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if index >= warmup_microbatches:
                timings.append(elapsed)
                losses.append(float(objective.detach().cpu()))
    seconds_per_microbatch = float(np.mean(timings))
    microbatches_per_epoch = len(LongCycler(dataloaders["train"]))
    gpu_utilization = [sample.utilization for sample in monitor.samples]
    gpu_memory = [sample.memory_mib for sample in monitor.samples]
    result = {
        "status": "pass_benchmark_only_not_a_training_result",
        "architecture": "full_static_sensorium_plus_framewise_on_dynamic_sensorium_2023",
        "precision": "fp32",
        "physical_batch_size_per_session": int(config["data"]["physical_batch_size_per_session"]),
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": int(config["data"]["effective_batch_size"]),
        "warmup_microbatches": warmup_microbatches,
        "measured_microbatches": measured_microbatches,
        "seconds_per_session_microbatch": seconds_per_microbatch,
        "seconds_per_optimizer_step": seconds_per_microbatch * accumulation,
        "microbatches_per_epoch": microbatches_per_epoch,
        "optimizer_steps_per_epoch": microbatches_per_epoch // accumulation,
        "estimated_epoch_seconds_excluding_validation": seconds_per_microbatch * microbatches_per_epoch,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "nvidia_smi_peak_memory_mib": max(gpu_memory) if gpu_memory else None,
        "gpu_utilization_mean_percent": float(np.mean(gpu_utilization)) if gpu_utilization else None,
        "gpu_utilization_peak_percent": max(gpu_utilization) if gpu_utilization else None,
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }
    artifact = resolve_path(config, config["artifacts"]["benchmark"])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(_json(result, indent=2), encoding="utf-8")
    return result


def _git_identity() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except subprocess.SubprocessError:
            return None

    return {"commit": command("rev-parse", "HEAD"), "branch": command("branch", "--show-current"), "status": command("status", "--short")}


def _environment_snapshot() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "static_sensorium_commit": STATIC_SENSORIUM_COMMIT,
        "dynamic_sensorium_commit": OFFICIAL_SENSORIUM_COMMIT,
        "neuralpredictors_commit": FUNCTIONAL_NEURALPREDICTORS_COMMIT,
        "git": _git_identity(),
    }


def train_full(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("static-on-dynamic full training requires CUDA")
    audit = audit_architecture(config)
    checkpoint_dir = resolve_path(config, config["artifacts"]["checkpoint_dir"])
    log_dir = resolve_path(config, config["artifacts"]["log_dir"])
    raw_dir = checkpoint_dir / "official_raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (checkpoint_dir / "best.pt").exists():
        raise FileExistsError(f"formal best checkpoint already exists; refusing to overwrite: {checkpoint_dir / 'best.pt'}")
    if list(raw_dir.glob("epoch_*.pth")) or (raw_dir / "final.pth").exists():
        raise FileExistsError(f"formal raw checkpoint directory is not empty: {raw_dir}")

    config_snapshot = copy.deepcopy(config)
    config_snapshot.pop("_config_path", None)
    config_snapshot.pop("_project_root", None)
    (log_dir / "training_config.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8")
    (log_dir / "environment.json").write_text(_json(_environment_snapshot(), indent=2), encoding="utf-8")
    (log_dir / "architecture_audit.json").write_text(_json(audit, indent=2), encoding="utf-8")
    (log_dir / "training_started.json").write_text(
        _json({"status": "running", "pid": os.getpid(), "started_unix": time.time()}, indent=2), encoding="utf-8"
    )

    set_random_seed(int(config["project"]["seed"]))
    dataloaders = make_official_loaders(config, cuda=True)
    model = build_model(config, dataloaders)
    evaluation_events: list[dict[str, Any]] = []
    original_scores_get = video_training_loop.scores.get_correlations
    original_direct_get = video_training_loop.get_correlations

    def logged_correlations(*args: Any, **kwargs: Any) -> Any:
        started = time.time()
        value = original_scores_get(*args, **kwargs)
        event = {"event_index": len(evaluation_events), "unix_time": started, "value": float(np.mean(value))}
        evaluation_events.append(event)
        with (log_dir / "validation_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(_json(event) + "\n")
        return value

    video_training_loop.scores.get_correlations = logged_correlations
    video_training_loop.get_correlations = logged_correlations
    original_os = video_training_loop.os
    video_training_loop.os = types.SimpleNamespace(listdir=os.listdir, remove=lambda _path: None)
    start = time.time()
    text_log = log_dir / "official_trainer.log"
    try:
        with text_log.open("w", encoding="utf-8", buffering=1) as handle:
            tee = Tee(sys.__stdout__, handle)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                score, trainer_output, best_state = video_training_loop.standard_trainer(
                    model=model,
                    dataloaders=dataloaders,
                    seed=int(config["project"]["seed"]),
                    use_wandb=False,
                    verbose=True,
                    lr_decay_steps=int(config["training"]["lr_decay_steps"]),
                    lr_init=float(config["training"]["learning_rate"]),
                    max_iter=int(config["training"]["max_epochs"]),
                    patience=int(config["training"]["patience"]),
                    interval=int(config["training"]["interval"]),
                    tolerance=float(config["training"]["tolerance"]),
                    lr_decay_factor=float(config["training"]["lr_decay_factor"]),
                    min_lr=float(config["training"]["min_lr"]),
                    restore_best=bool(config["training"]["restore_best"]),
                    device="cuda",
                    detach_core=False,
                    deeplake_ds=False,
                    save_checkpoints=True,
                    checkpoint_save_path=str(raw_dir) + os.sep,
                    chpt_save_step=1,
                )
    except BaseException as error:
        (log_dir / "training_interrupted.json").write_text(
            _json({"status": "interrupted", "type": type(error).__name__, "message": str(error), "unix_time": time.time()}, indent=2),
            encoding="utf-8",
        )
        raise
    finally:
        video_training_loop.scores.get_correlations = original_scores_get
        video_training_loop.get_correlations = original_direct_get
        video_training_loop.os = original_os

    elapsed = time.time() - start
    epoch_files = sorted(raw_dir.glob("epoch_*.pth"), key=lambda path: int(re.search(r"epoch_(\d+)", path.name).group(1)))
    if not epoch_files:
        raise RuntimeError("official trainer completed without a retained epoch checkpoint")
    last_raw = epoch_files[-1]
    last_epoch = int(re.search(r"epoch_(\d+)", last_raw.name).group(1))
    last_state = torch.load(last_raw, map_location="cpu", weights_only=True)
    metadata = {
        "run_name": config["project"]["run_name"],
        "scientific_label": config["project"]["scientific_label"],
        "seed": int(config["project"]["seed"]),
        "config": config_snapshot,
        "parameter_counts": EXPECTED_PARAMETER_COUNTS,
        "environment": _environment_snapshot(),
        "official_validation_score": float(score),
        "trainer_output": trainer_output,
        "last_epoch": last_epoch,
        "elapsed_seconds": elapsed,
    }
    torch.save({**metadata, "checkpoint_kind": "best_restored", "state_dict": best_state}, checkpoint_dir / "best.pt")
    torch.save({**metadata, "checkpoint_kind": "last_pre_restore", "state_dict": last_state}, checkpoint_dir / "last.pt")
    official_last = raw_dir / "last.pth"
    if official_last.exists():
        official_last.unlink()
    last_raw.replace(official_last)
    for path in raw_dir.glob("epoch_*.pth"):
        path.unlink()
    summary = {
        "status": "training_complete",
        "seed": int(config["project"]["seed"]),
        "validation_score": float(score),
        "last_epoch": last_epoch,
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "validation_events": len(evaluation_events),
    }
    (log_dir / "training_summary.json").write_text(_json(summary, indent=2), encoding="utf-8")
    (log_dir / "training_started.json").write_text(
        _json({"status": "complete", "pid": os.getpid(), "completed_unix": time.time()}, indent=2), encoding="utf-8"
    )
    return summary


def evaluate_best(config: dict[str, Any]) -> dict[str, Any]:
    """Reload the frozen best checkpoint and evaluate complete oracle trials."""
    if not torch.cuda.is_available():
        raise RuntimeError("static-on-dynamic full-sequence evaluation requires CUDA")
    dataloaders = make_official_loaders(config, cuda=True, batch_size=1, to_cut=False, offset=0)
    model = build_model(config, dataloaders).cuda()
    checkpoint_path = resolve_path(config, config["evaluation"].get("published_checkpoint", Path(config["artifacts"]["checkpoint_dir"]) / "best.pt"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    model.load_state_dict(state if isinstance(state, dict) else checkpoint, strict=True)
    model.eval()
    by_neuron = get_correlations(
        model,
        dataloaders["oracle"],
        device="cuda",
        as_dict=True,
        per_neuron=True,
    )
    by_session = {key: float(np.mean(value)) for key, value in by_neuron.items()}
    result = {
        "status": "full_sequence_oracle_evaluated",
        "implementation": "sensorium.utility.scores.get_correlations",
        "checkpoint": str(checkpoint_path),
        "full_sequence_oracle_single_trial": float(np.mean(np.hstack(list(by_neuron.values())))),
        "full_sequence_oracle_by_session": by_session,
        "burn_in_frames": int(config["evaluation"]["burn_in_frames"]),
        "retained_source_frames": [50, 299],
    }
    artifact_dir = resolve_path(config, "artifacts/static_dynamic_sensorium2023")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "official_evaluation.json").write_text(_json(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the full static Sensorium CNN on Dynamic Sensorium 2023")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit-config")
    subparsers.add_parser("audit")
    subparsers.add_parser("smoke")
    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--iterations", type=int, default=200)
    benchmark_parser.add_argument("--warmup", type=int, default=20)
    subparsers.add_parser("train")
    subparsers.add_parser("evaluate")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "audit-config":
        result = audit_locked_config(config)
    elif args.command == "audit":
        result = audit_architecture(config)
    elif args.command == "smoke":
        result = smoke_test(config)
    elif args.command == "benchmark":
        result = benchmark(config, measured_microbatches=args.iterations, warmup_microbatches=args.warmup)
    elif args.command == "train":
        result = train_full(config)
    else:
        result = evaluate_best(config)
    print(_json(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
