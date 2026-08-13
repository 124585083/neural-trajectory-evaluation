from __future__ import annotations

import copy
import json
import math
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import resolve_project_path
from .data import DynamicSensoriumDataModule, LoaderBundle
from .evaluation.response import correlation_summary
from .models import ModelName, SensoriumReferenceModel, build_model


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def aligned_target(batch: dict[str, Any], prediction: torch.Tensor) -> torch.Tensor:
    target = batch["responses"].transpose(1, 2)
    return target[:, -prediction.shape[1] :, :]


def poisson_objective(
    model: SensoriumReferenceModel,
    batch: dict[str, Any],
    dataset_size: int,
    regularizer_divisor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model(batch["video"], batch["behavior"], batch["pupil_center"])
    target = aligned_target(batch, prediction)
    data_loss = nn.functional.poisson_nll_loss(
        prediction.float(), target.float(), log_input=False, full=False, eps=1e-8, reduction="sum"
    )
    loss_scale = math.sqrt(dataset_size / batch["video"].shape[0])
    objective = loss_scale * data_loss + model.regularizer() / regularizer_divisor
    if not torch.isfinite(objective):
        raise FloatingPointError("non-finite Poisson objective")
    return objective, prediction


def collect_aligned_predictions(
    model: SensoriumReferenceModel,
    bundle: LoaderBundle,
    device: torch.device,
    burn_in_frames: int,
    amp: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in bundle.loader:
            batch = _device_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                prediction = model(batch["video"], batch["behavior"], batch["pupil_center"])
            target = aligned_target(batch, prediction)
            original_time = batch["video"].shape[2]
            lag = original_time - prediction.shape[1]
            aligned_skip = max(0, burn_in_frames - lag)
            prediction = prediction[:, aligned_skip:, :].float().cpu().numpy()
            target = target[:, aligned_skip:, :].float().cpu().numpy()
            frame_index = batch["frame_index"][:, -model_output_length(model, original_time) :]
            frame_index = frame_index[:, aligned_skip:].cpu().numpy()
            trial_indices = batch["trial_index"].cpu().numpy()
            window_offsets = batch["window_offset"].cpu().numpy()
            for index in range(prediction.shape[0]):
                predictions.append(prediction[index])
                targets.append(target[index])
                metadata.append(
                    {
                        "trial_index": int(trial_indices[index]),
                        "window_offset": int(window_offsets[index]),
                        "frame_index": frame_index[index],
                    }
                )
    return targets, predictions, metadata


def model_output_length(model: SensoriumReferenceModel, input_time: int) -> int:
    return input_time - model.temporal_reduction


def validation_correlation(
    model: SensoriumReferenceModel,
    bundle: LoaderBundle,
    device: torch.device,
    burn_in_frames: int,
    amp: bool,
) -> float:
    targets, predictions, _ = collect_aligned_predictions(model, bundle, device, burn_in_frames, amp)
    return float(correlation_summary(targets, predictions)["single_trial_mean"])


def _make_optimizer(model: SensoriumReferenceModel, name: ModelName, config: dict) -> torch.optim.Optimizer:
    training = config["training"]
    if name == "dynamic_reference":
        return torch.optim.AdamW(model.parameters(), lr=float(training["dynamic_lr"]))
    return torch.optim.Adam(model.parameters(), lr=float(training["static_lr"]))


def _optimizer_step(
    model: SensoriumReferenceModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: dict[str, Any],
    dataset_size: int,
    accumulation_steps: int,
    amp: bool,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        objective, prediction = poisson_objective(model, batch, dataset_size, accumulation_steps)
    scaler.scale(objective).backward()
    return float(objective.detach().cpu()), prediction


def train_reference(
    name: ModelName,
    config: dict,
    data: DynamicSensoriumDataModule | None = None,
) -> dict[str, Any]:
    data = data or DynamicSensoriumDataModule(config)
    device = torch.device(config["project"]["device"] if torch.cuda.is_available() else "cpu")
    model, shapes = build_model(name, config, data, device)
    train_bundle = data.make_loader("train", frames=int(config["data"]["frames"]), shuffle=True)
    oracle_bundle = data.make_loader("oracle", frames=int(config["data"]["frames"]), shuffle=False)
    effective_batch = int(config["data"]["effective_batch_size"])
    physical_batch = int(config["data"]["physical_batch_size"])
    accumulation_steps = math.ceil(effective_batch / physical_batch)
    optimizer = _make_optimizer(model, name, config)
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["patience"])
    max_decays = int(config["training"]["lr_decay_steps"])
    decay_factor = float(config["training"]["lr_decay_factor"])
    min_lr = float(config["training"]["min_lr"])
    tolerance = float(config["training"]["tolerance"])
    burn_in = int(config["training"]["burn_in_frames"])

    checkpoint_dir = resolve_project_path(config, Path("checkpoints") / name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolve_project_path(config, Path("logs") / f"{name}.jsonl")
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    bad_epochs = 0
    decay_count = 0
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(max_epochs):
            train_bundle.set_epoch(epoch)
            oracle_bundle.set_epoch(0)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_objective = 0.0
            microbatches = 0
            for microbatch_index, raw_batch in enumerate(train_bundle.loader):
                batch = _device_batch(raw_batch, device)
                objective, _ = _optimizer_step(
                    model,
                    optimizer,
                    scaler,
                    batch,
                    len(train_bundle.dataset),
                    accumulation_steps,
                    amp,
                    device,
                )
                epoch_objective += objective
                microbatches += 1
                should_step = (microbatch_index + 1) % accumulation_steps == 0
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            if microbatches % accumulation_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            score = validation_correlation(model, oracle_bundle, device, burn_in, amp)
            improved = score > best_score + tolerance
            if improved:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                torch.save(
                    {
                        "model_name": name,
                        "epoch": epoch,
                        "validation_correlation": score,
                        "model_shapes": asdict(shapes),
                        "session_key": data.session_key,
                        "state_dict": best_state,
                        "config": config,
                    },
                    checkpoint_dir / "best.pt",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    old_lr = optimizer.param_groups[0]["lr"]
                    new_lr = max(min_lr, old_lr * decay_factor)
                    for group in optimizer.param_groups:
                        group["lr"] = new_lr
                    bad_epochs = 0
                    if new_lr < old_lr:
                        decay_count += 1

            record = {
                "epoch": epoch,
                "train_objective_sum": epoch_objective,
                "validation_correlation": score,
                "best_validation_correlation": best_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "decay_count": decay_count,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            history.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps({"model": name, **record}), flush=True)
            if decay_count >= max_decays or (optimizer.param_groups[0]["lr"] <= min_lr and not improved):
                break

    if best_state is None:
        raise RuntimeError("training completed without a finite validation checkpoint")
    model.load_state_dict(best_state)
    final_path = checkpoint_dir / "final.pt"
    torch.save(
        {
            "model_name": name,
            "best_epoch": best_epoch,
            "validation_correlation": best_score,
            "model_shapes": asdict(shapes),
            "session_key": data.session_key,
            "state_dict": best_state,
            "history": history,
            "config": config,
        },
        final_path,
    )
    return {
        "model_name": name,
        "checkpoint": str(final_path),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "best_epoch": best_epoch,
        "validation_correlation": best_score,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - start_time,
    }


class _GpuUtilizationMonitor:
    def __init__(self) -> None:
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                value = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=5,
                ).splitlines()[0]
                self.samples.append(float(value.strip()))
            except Exception:
                pass
            self._stop.wait(0.5)

    def __enter__(self) -> "_GpuUtilizationMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=6)


def benchmark_reference(
    name: ModelName,
    config: dict,
    data: DynamicSensoriumDataModule | None = None,
) -> dict[str, Any]:
    data = data or DynamicSensoriumDataModule(config)
    device = torch.device(config["project"]["device"] if torch.cuda.is_available() else "cpu")
    model, shapes = build_model(name, config, data, device)
    bundle = data.make_loader("train", frames=int(config["data"]["frames"]), shuffle=True)
    optimizer = _make_optimizer(model, name, config)
    effective_batch = int(config["data"]["effective_batch_size"])
    physical_batch = int(config["data"]["physical_batch_size"])
    accumulation_steps = math.ceil(effective_batch / physical_batch)
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    iterations = int(config["benchmark"]["iterations"])
    warmup = int(config["benchmark"]["warmup_iterations"])
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(bundle.loader)
    epoch = 0
    measured_start: float | None = None

    with _GpuUtilizationMonitor() as monitor:
        for index in range(iterations):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                epoch += 1
                bundle.set_epoch(epoch)
                iterator = iter(bundle.loader)
                raw_batch = next(iterator)
            batch = _device_batch(raw_batch, device)
            _optimizer_step(
                model,
                optimizer,
                scaler,
                batch,
                len(bundle.dataset),
                accumulation_steps,
                amp,
                device,
            )
            if (index + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if index + 1 == warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                measured_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if measured_start is None:
        raise ValueError("warmup_iterations must be lower than iterations")
    measured_iterations = iterations - warmup
    elapsed = time.perf_counter() - measured_start
    seconds_per_iteration = elapsed / measured_iterations
    iterations_per_epoch = math.ceil(len(bundle.dataset) / physical_batch)
    result = {
        "model_name": name,
        "iterations": iterations,
        "warmup_iterations": warmup,
        "seconds_per_iteration": seconds_per_iteration,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "mean_gpu_utilization_percent": float(np.mean(monitor.samples)) if monitor.samples else None,
        "gpu_utilization_samples": len(monitor.samples),
        "physical_batch_size": physical_batch,
        "effective_batch_size": effective_batch,
        "gradient_accumulation_steps": accumulation_steps,
        "iterations_per_epoch": iterations_per_epoch,
        "estimated_seconds_per_epoch": seconds_per_iteration * iterations_per_epoch,
        "estimated_seconds_for_max_epochs": seconds_per_iteration
        * iterations_per_epoch
        * int(config["training"]["max_epochs"]),
        "model_shapes": asdict(shapes),
    }
    output = resolve_project_path(config, Path("benchmarks") / f"{name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_checkpoint_model(
    name: ModelName,
    config: dict,
    data: DynamicSensoriumDataModule,
    device: torch.device,
) -> tuple[SensoriumReferenceModel, dict[str, Any]]:
    checkpoint_path = resolve_project_path(config, Path("checkpoints") / name / "final.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, _ = build_model(name, config, data, device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint

