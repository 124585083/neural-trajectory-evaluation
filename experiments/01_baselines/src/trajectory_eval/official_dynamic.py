from __future__ import annotations

import argparse
import collections
import collections.abc
import contextlib
import copy
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml

from .config import PROJECT_ROOT


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase1A_dynamic_official.yaml"
SENSORIUM_SOURCE = PROJECT_ROOT / "third_party" / "sensorium_2023"
OFFICIAL_SENSORIUM_COMMIT = "0e02656220e84a50f3be1b92d6f66c2f9ccd51ef"
FUNCTIONAL_NEURALPREDICTORS_COMMIT = "efdda679596517fad95d71f36d0385d7450dd207"
HISTORICAL_NEURALPREDICTORS_COMMIT = "43faededa2d2e76bb904f38a49b9d8b81d287a0a"
EXPECTED_SESSIONS = (
    "dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20",
)
EXPECTED_NEURONS = (7863, 7908, 8202, 7939, 8122)
EXPECTED_PARAMETER_COUNTS = {"core": 382_176, "readout": 5_325_282, "shifter": 285, "total": 5_707_743}


def _json(value: Any, *, indent: int | None = None) -> str:
    def default(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"cannot serialize {type(item).__name__}")

    return json.dumps(value, indent=indent, default=default)


def _install_python311_compatibility() -> None:
    """Restore aliases required by the official 2023 nnfabrik dependency."""

    for name in ("Iterable", "Mapping", "MutableMapping"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


_install_python311_compatibility()
if str(SENSORIUM_SOURCE) not in sys.path:
    sys.path.insert(0, str(SENSORIUM_SOURCE))

# These imports deliberately resolve to the retained official Sensorium source.
from neuralpredictors.measures import modules  # noqa: E402
from neuralpredictors.training import LongCycler  # noqa: E402
from nnfabrik.utility.nn_helpers import set_random_seed  # noqa: E402
from sensorium.datasets.mouse_video_loaders import mouse_video_loader  # noqa: E402
from sensorium.models.make_model import make_video_model  # noqa: E402
from sensorium.training import video_training_loop  # noqa: E402
from sensorium.utility.scores import get_correlations  # noqa: E402
from sensorium.utility.submission import generate_submission  # noqa: E402


def load_official_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def official_session_paths(config: dict[str, Any]) -> list[str]:
    sessions = tuple(config["data"]["sessions"])
    if sessions != EXPECTED_SESSIONS:
        raise AssertionError(f"official session lock failed: {sessions!r} != {EXPECTED_SESSIONS!r}")
    root = resolve_path(config, config["data"]["root"])
    paths: list[str] = []
    for session in sessions:
        session_path = root / session
        if not session_path.is_dir():
            raise FileNotFoundError(f"missing official session: {session_path}")
        # The official loader derives data keys by splitting on '/'.
        paths.append(session_path.as_posix().rstrip("/") + "/")
    return paths


def make_official_loaders(
    config: dict[str, Any],
    *,
    cuda: bool,
    batch_size: int | None = None,
    to_cut: bool | None = None,
    offset: int | None = None,
) -> dict[str, dict[str, Any]]:
    data = config["data"]
    dataloaders = mouse_video_loader(
        paths=official_session_paths(config),
        batch_size=batch_size or int(data["physical_batch_size_per_session"]),
        scale=float(data["scale"]),
        max_frame=data["max_frame"],
        frames=int(data["frames"]),
        offset=int(data["offset"] if offset is None else offset),
        include_behavior=bool(data["include_behavior"]),
        include_pupil_centers=bool(data["include_pupil_centers"]),
        cuda=cuda,
        to_cut=bool(data["to_cut"] if to_cut is None else to_cut),
    )
    # The official FileTreeDataset cache is unbounded and would retain every
    # raw response array. Five sessions exceed the host's 31 GiB RAM. Disabling
    # this I/O cache changes neither samples nor transforms nor their order.
    use_cache = bool(data.get("use_file_cache", True))
    seen: set[int] = set()
    for tier_loaders in dataloaders.values():
        for loader in tier_loaders.values():
            dataset = loader.dataset
            if id(dataset) in seen:
                continue
            seen.add(id(dataset))
            dataset.use_cache = use_cache
            if not use_cache and hasattr(dataset, "_cache"):
                for values in dataset._cache.values():
                    values.clear()
    return dataloaders


def build_official_model(config: dict[str, Any], dataloaders: dict[str, dict[str, Any]]) -> torch.nn.Module:
    model_config = config["model"]
    core_config = copy.deepcopy(model_config["core"])
    # Preserve the tuple types used verbatim by the official notebook. YAML has
    # no native tuple and otherwise decodes these kernels as lists.
    core_config["spatial_input_kernel"] = tuple(core_config["spatial_input_kernel"])
    core_config["spatial_hidden_kernel"] = tuple(core_config["spatial_hidden_kernel"])
    set_random_seed(int(config["project"]["seed"]))
    return make_video_model(
        dataloaders,
        int(config["project"]["seed"]),
        core_dict=core_config,
        core_type=model_config["core_type"],
        readout_dict=copy.deepcopy(model_config["readout"]),
        readout_type=model_config["readout_type"],
        use_gru=bool(model_config["use_gru"]),
        gru_dict=None,
        use_shifter=bool(model_config["use_shifter"]),
        shifter_dict=copy.deepcopy(model_config["shifter"]),
        shifter_type=model_config["shifter_type"],
        deeplake_ds=False,
    )


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "core": sum(parameter.numel() for parameter in model.core.parameters()),
        "readout": sum(parameter.numel() for parameter in model.readout.parameters()),
        "shifter": sum(parameter.numel() for parameter in model.shifter.parameters()),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }


def _conv_signature(layer: torch.nn.Module) -> list[dict[str, Any]]:
    signature = []
    for name, module in layer.named_children():
        if isinstance(module, torch.nn.Conv3d):
            signature.append(
                {
                    "name": name,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": list(module.kernel_size),
                    "stride": list(module.stride),
                    "padding": list(module.padding),
                }
            )
    return signature


def audit_architecture(config: dict[str, Any], *, save: bool = True) -> dict[str, Any]:
    dataloaders = make_official_loaders(config, cuda=False)
    model = build_official_model(config, dataloaders)
    counts = parameter_counts(model)
    if counts != EXPECTED_PARAMETER_COUNTS:
        raise AssertionError(f"parameter lock failed: {counts} != {EXPECTED_PARAMETER_COUNTS}")

    if len(model.core.features) != 3:
        raise AssertionError(f"core layer lock failed: {len(model.core.features)} != 3")
    signatures = [_conv_signature(layer) for layer in model.core.features]
    expected = [
        [
            {"name": "conv_spatial", "in_channels": 3, "out_channels": 32, "kernel_size": [1, 11, 11], "stride": [1, 1, 1], "padding": [0, 0, 0]},
            {"name": "conv_temporal", "in_channels": 32, "out_channels": 32, "kernel_size": [11, 1, 1], "stride": [1, 1, 1], "padding": [0, 0, 0]},
        ],
        [
            {"name": "conv_spatial_1", "in_channels": 32, "out_channels": 64, "kernel_size": [1, 5, 5], "stride": [1, 1, 1], "padding": [0, 0, 0]},
            {"name": "conv_temporal_1", "in_channels": 64, "out_channels": 64, "kernel_size": [5, 1, 1], "stride": [1, 1, 1], "padding": [0, 0, 0]},
        ],
        [
            {"name": "conv_spatial_2", "in_channels": 64, "out_channels": 128, "kernel_size": [1, 5, 5], "stride": [1, 1, 1], "padding": [0, 0, 0]},
            {"name": "conv_temporal_2", "in_channels": 128, "out_channels": 128, "kernel_size": [5, 1, 1], "stride": [1, 1, 1], "padding": [0, 0, 0]},
        ],
    ]
    if signatures != expected:
        raise AssertionError(f"convolution lock failed: {signatures!r}")

    data_keys = tuple(dataloaders["train"].keys())
    if data_keys != EXPECTED_SESSIONS:
        raise AssertionError(f"readout/session lock failed: {data_keys!r}")
    outdims = tuple(model.readout[key].outdims for key in data_keys)
    if outdims != EXPECTED_NEURONS:
        raise AssertionError(f"neuron/readout lock failed: {outdims!r} != {EXPECTED_NEURONS!r}")

    for layer, channels in zip(model.core.features, (32, 64, 128)):
        norm = layer.norm
        if not isinstance(norm, torch.nn.BatchNorm3d) or norm.num_features != channels or norm.momentum != 0.7:
            raise AssertionError(f"batch-norm lock failed for {channels} channels: {norm}")
        if not isinstance(layer.nonlin, torch.nn.ELU):
            raise AssertionError(f"activation lock failed: {layer.nonlin}")

    first_key = data_keys[0]
    batch = next(iter(dataloaders["oracle"][first_key]))
    model.eval()
    with torch.inference_mode():
        output = model(batch.videos, data_key=first_key, behavior=batch.behavior, pupil_center=batch.pupil_center)
    if tuple(batch.videos.shape[1:]) != (3, 80, 36, 64):
        raise AssertionError(f"input lock failed: {tuple(batch.videos.shape)}")
    if tuple(output.shape[1:]) != (62, EXPECTED_NEURONS[0]):
        raise AssertionError(f"output lock failed: {tuple(output.shape)}")

    audit = {
        "status": "pass",
        "sensorium_commit": OFFICIAL_SENSORIUM_COMMIT,
        "neuralpredictors_commit": FUNCTIONAL_NEURALPREDICTORS_COMMIT,
        "historical_neuralpredictors_pin": HISTORICAL_NEURALPREDICTORS_COMMIT,
        "sessions": list(data_keys),
        "neuron_counts": list(outdims),
        "train_loader_batches": {key: len(value) for key, value in dataloaders["train"].items()},
        "input_shape": list(batch.videos.shape[1:]),
        "output_shape": list(output.shape[1:]),
        "convolutions": signatures,
        "parameter_counts": counts,
        "model": str(model),
    }
    if save:
        artifact_dir = resolve_path(config, "artifacts/phase1A")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "architecture_audit.json").write_text(_json(audit, indent=2), encoding="utf-8")
        (artifact_dir / "dynamic_official_model.txt").write_text(str(model) + "\n", encoding="utf-8")
    return audit


class LimitedLoader:
    def __init__(self, loader: Any, batches: int) -> None:
        self.loader = loader
        self.dataset = loader.dataset
        self.batches = min(int(batches), len(loader))

    def __iter__(self) -> Iterator[Any]:
        return iter(islice(iter(self.loader), self.batches))

    def __len__(self) -> int:
        return self.batches


def _limited_tiers(dataloaders: dict[str, dict[str, Any]], batches: int = 1) -> dict[str, dict[str, LimitedLoader]]:
    return {
        tier: {key: LimitedLoader(loader, batches) for key, loader in loaders.items()}
        for tier, loaders in dataloaders.items()
    }


def smoke_test(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("official smoke test requires CUDA")
    audit_architecture(config)
    set_random_seed(int(config["project"]["seed"]))
    dataloaders = make_official_loaders(config, cuda=True)
    model = build_official_model(config, dataloaders)
    limited = _limited_tiers(dataloaders, batches=1)
    output_dir = resolve_path(config, "checkpoints/dynamic_official_reproduction_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.pth"):
        existing.unlink()

    start = time.perf_counter()
    score, trainer_output, state_dict = video_training_loop.standard_trainer(
        model=model,
        dataloaders=limited,
        seed=int(config["project"]["seed"]),
        use_wandb=False,
        verbose=False,
        lr_decay_steps=1,
        lr_init=float(config["training"]["learning_rate"]),
        max_iter=1,
        patience=int(config["training"]["patience"]),
        device="cuda",
        detach_core=False,
        deeplake_ds=False,
        save_checkpoints=True,
        checkpoint_save_path=str(output_dir) + os.sep,
        chpt_save_step=1,
    )
    elapsed = time.perf_counter() - start
    checkpoint_path = output_dir / "final.pth"
    if not checkpoint_path.is_file():
        raise AssertionError("official trainer did not save the smoke checkpoint")
    if not all(torch.isfinite(value).all() for value in state_dict.values() if torch.is_tensor(value)):
        raise AssertionError("smoke checkpoint contains non-finite tensors")

    # Independent construction and reload, followed by official validation.
    reloaded = build_official_model(config, dataloaders).to("cuda")
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cuda", weights_only=True))
    reload_score = float(
        get_correlations(reloaded, limited["oracle"], device="cuda", as_dict=False, per_neuron=False, deeplake_ds=False)
    )
    result = {
        "status": "pass",
        "scope": "smoke_test_only_not_a_reproduction_result",
        "train_batches_per_session": 1,
        "optimizer_steps": 1,
        "trainer_score": float(score),
        "trainer_output": trainer_output,
        "reloaded_validation_score": reload_score,
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    artifact = resolve_path(config, "artifacts/phase1A/smoke_test.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(_json(result, indent=2), encoding="utf-8")
    return result


def official_training_objective(
    model: torch.nn.Module,
    train_loaders: dict[str, Any],
    data_key: str,
    data: Any,
    criterion: torch.nn.Module,
    device: str,
) -> torch.Tensor:
    """Exact public trainer objective, exposed only for throughput measurement."""

    batch_args = list(data)
    batch_kwargs = data._asdict() if not isinstance(data, dict) else data
    loss_scale = np.sqrt(len(train_loaders[data_key].dataset) / batch_args[0].shape[0])
    core_regularizer = model.core.regularizer()
    regularizers = (sum(core_regularizer) if isinstance(core_regularizer, tuple) else core_regularizer) + model.readout.regularizer(data_key)
    model_output = model(batch_args[0].to(device), data_key=data_key, **batch_kwargs)
    time_left = model_output.shape[1]
    targets = batch_args[1].transpose(2, 1)[:, -time_left:, :].to(device)
    return loss_scale * criterion(model_output, targets) + regularizers


@dataclass
class GPUSample:
    utilization: float
    memory_mib: float
    power_w: float


class GPUMonitor:
    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.samples: list[GPUSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip().splitlines()[0]
                util, memory, power = (float(value.strip()) for value in output.split(","))
                self.samples.append(GPUSample(util, memory, power))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "GPUMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def benchmark(config: dict[str, Any], *, measured_microbatches: int = 200, warmup_microbatches: int = 20) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("official benchmark requires CUDA")
    audit_architecture(config)
    set_random_seed(int(config["project"]["seed"]))
    dataloaders = make_official_loaders(config, cuda=True)
    model = build_official_model(config, dataloaders).to("cuda")
    model.train()
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
    criterion = modules.PoissonLoss(avg=False)
    accumulation = len(dataloaders["train"])
    total_microbatches = warmup_microbatches + measured_microbatches
    cycler = iter(LongCycler(dataloaders["train"]))
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    measured_start = 0.0
    measured_end = 0.0
    measured_losses: list[float] = []
    optimizer_steps = 0

    with GPUMonitor() as monitor:
        for index in range(total_microbatches):
            if index == warmup_microbatches:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                measured_start = time.perf_counter()
            try:
                data_key, data = next(cycler)
            except StopIteration:
                cycler = iter(LongCycler(dataloaders["train"]))
                data_key, data = next(cycler)
            loss = official_training_objective(model, dataloaders["train"], data_key, data, criterion, "cuda")
            loss.backward()
            if index >= warmup_microbatches:
                measured_losses.append(float(loss.detach().cpu()))
            if (index + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if index >= warmup_microbatches:
                    optimizer_steps += 1
        torch.cuda.synchronize()
        measured_end = time.perf_counter()

    elapsed = measured_end - measured_start
    seconds_per_microbatch = elapsed / measured_microbatches
    seconds_per_optimizer_step = elapsed / optimizer_steps
    microbatches_per_epoch = len(LongCycler(dataloaders["train"]))
    optimizer_steps_per_epoch = microbatches_per_epoch // accumulation
    estimated_epoch = seconds_per_microbatch * microbatches_per_epoch
    utilization = [sample.utilization for sample in monitor.samples]
    memory = [sample.memory_mib for sample in monitor.samples]
    power = [sample.power_w for sample in monitor.samples]
    result = {
        "status": "pass",
        "architecture": "full_official_factorized3d",
        "precision": "fp32",
        "physical_batch_size_per_session": int(config["data"]["physical_batch_size_per_session"]),
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": int(config["data"]["effective_batch_size"]),
        "warmup_microbatches": warmup_microbatches,
        "measured_microbatches": measured_microbatches,
        "measured_optimizer_steps": optimizer_steps,
        "seconds_per_session_microbatch": seconds_per_microbatch,
        "seconds_per_optimizer_step": seconds_per_optimizer_step,
        "microbatches_per_epoch": microbatches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "estimated_epoch_seconds_excluding_validation": estimated_epoch,
        "estimated_200_epoch_upper_bound_hours_excluding_early_stop_and_validation": estimated_epoch * 200 / 3600,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "nvidia_smi_peak_memory_mib": max(memory) if memory else None,
        "gpu_utilization_mean_percent": statistics.mean(utilization) if utilization else None,
        "gpu_utilization_median_percent": statistics.median(utilization) if utilization else None,
        "gpu_utilization_peak_percent": max(utilization) if utilization else None,
        "power_mean_w": statistics.mean(power) if power else None,
        "power_peak_w": max(power) if power else None,
        "loss_first": measured_losses[0],
        "loss_last": measured_losses[-1],
    }
    artifact = resolve_path(config, "benchmarks/phase1A_dynamic_official.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(_json(result, indent=2), encoding="utf-8")
    report = resolve_path(config, "reports/phase1A_runtime_benchmark.md")
    report.write_text(
        "# Phase 1A runtime benchmark\n\n"
        "This benchmark uses the complete five-session official architecture and loader in FP32. "
        "Smoke-test or benchmark scores are not reproduction results.\n\n"
        f"- Physical batch per session: `{result['physical_batch_size_per_session']}`\n"
        f"- Session-gradient accumulation: `{result['gradient_accumulation_steps']}`\n"
        f"- Effective batch: `{result['effective_batch_size']}`\n"
        f"- Stable measured session microbatches: `{measured_microbatches}` after `{warmup_microbatches}` warmups\n"
        f"- Seconds / session microbatch: `{seconds_per_microbatch:.4f}`\n"
        f"- Seconds / optimizer step: `{seconds_per_optimizer_step:.4f}`\n"
        f"- Iterations / epoch: `{microbatches_per_epoch}` session microbatches / `{optimizer_steps_per_epoch}` optimizer steps\n"
        f"- Estimated epoch time excluding validation: `{estimated_epoch:.2f}` s\n"
        f"- 200-epoch upper bound excluding validation: `{result['estimated_200_epoch_upper_bound_hours_excluding_early_stop_and_validation']:.2f}` h\n"
        f"- PyTorch peak allocated VRAM: `{result['peak_allocated_bytes'] / 2**30:.2f}` GiB\n"
        f"- PyTorch peak reserved VRAM: `{result['peak_reserved_bytes'] / 2**30:.2f}` GiB\n"
        f"- nvidia-smi peak memory: `{result['nvidia_smi_peak_memory_mib']}` MiB\n"
        f"- GPU utilization mean / median / peak: `{result['gpu_utilization_mean_percent']:.2f}%` / `{result['gpu_utilization_median_percent']:.2f}%` / `{result['gpu_utilization_peak_percent']:.2f}%`\n"
        f"- GPU power mean / peak: `{result['power_mean_w']:.2f} W` / `{result['power_peak_w']:.2f} W`\n\n"
        "No width, depth, kernel, resolution, session, neuron, temporal-context, loss, or effective-batch change was made.\n",
        encoding="utf-8",
    )
    return result


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


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
        "sensorium_commit": OFFICIAL_SENSORIUM_COMMIT,
        "neuralpredictors_commit": FUNCTIONAL_NEURALPREDICTORS_COMMIT,
        "git": _git_identity(),
    }


def train_full(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("full official training requires CUDA")
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

    set_random_seed(int(config["project"]["seed"]))
    dataloaders = make_official_loaders(config, cuda=True)
    model = build_official_model(config, dataloaders)
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
    # Preserve official per-epoch states long enough to identify the true last state.
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
        "seed": int(config["project"]["seed"]),
        "single_seed_reproduction": True,
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

    # Keep the official raw final and true last states, remove only redundant epoch files created by this run.
    official_last = raw_dir / "last.pth"
    if official_last.exists():
        official_last.unlink()
    last_raw.replace(official_last)
    for path in raw_dir.glob("epoch_*.pth"):
        path.unlink()

    summary = {
        "status": "training_complete",
        "single_seed": 42,
        "validation_score": float(score),
        "last_epoch": last_epoch,
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "official_raw_final": str(raw_dir / "final.pth"),
        "official_raw_last": str(official_last),
        "validation_events": len(evaluation_events),
    }
    (log_dir / "training_summary.json").write_text(_json(summary, indent=2), encoding="utf-8")
    return summary


def recover_user_stopped_training(
    config: dict[str, Any],
    *,
    best_epoch: int,
    last_epoch: int,
    best_score: float,
    best_post_epoch_score: float,
) -> dict[str, Any]:
    """Package complete epoch states after an intentional user stop."""

    audit = audit_architecture(config)
    checkpoint_dir = resolve_path(config, config["artifacts"]["checkpoint_dir"])
    log_dir = resolve_path(config, config["artifacts"]["log_dir"])
    raw_dir = checkpoint_dir / "official_raw"
    best_raw = raw_dir / f"epoch_{best_epoch}.pth"
    last_raw = raw_dir / f"epoch_{last_epoch}.pth"
    for path in (best_raw, last_raw):
        if not path.is_file():
            raise FileNotFoundError(f"cannot recover stopped training; missing complete checkpoint: {path}")
    if (raw_dir / f"epoch_{last_epoch + 1}.pth").exists():
        raise AssertionError("last_epoch is not the final complete saved epoch")

    validation_path = log_dir / "validation_events.jsonl"
    validation_events = [json.loads(line) for line in validation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observed_scores = [float(event["value"]) for event in validation_events]
    if not any(abs(value - best_score) <= 1e-9 for value in observed_scores):
        raise AssertionError(f"best closure score {best_score} is absent from the official validation log")
    if not any(abs(value - best_post_epoch_score) <= 1e-9 for value in observed_scores):
        raise AssertionError(f"best post-epoch score {best_post_epoch_score} is absent from the official validation log")

    config_snapshot = copy.deepcopy(config)
    config_snapshot.pop("_config_path", None)
    config_snapshot.pop("_project_root", None)
    common_metadata = {
        "run_name": config["project"]["run_name"],
        "seed": int(config["project"]["seed"]),
        "single_seed_reproduction": True,
        "training_completion": f"stopped_by_user_after_epoch_{last_epoch}_validation",
        "stop_reason": "User requested no further training and selection of the current official best model.",
        "config": config_snapshot,
        "parameter_counts": EXPECTED_PARAMETER_COUNTS,
        "environment": _environment_snapshot(),
        "architecture_audit": audit,
        "official_validation_score": float(best_score),
        "best_post_epoch_validation_score": float(best_post_epoch_score),
        "best_epoch": int(best_epoch),
        "last_complete_epoch": int(last_epoch),
        "partial_next_epoch_saved": False,
        "validation_events": len(validation_events),
    }
    best_state = torch.load(best_raw, map_location="cpu", weights_only=True)
    last_state = torch.load(last_raw, map_location="cpu", weights_only=True)
    torch.save({**common_metadata, "checkpoint_kind": "official_early_stopping_best", "state_dict": best_state}, checkpoint_dir / "best.pt")
    torch.save({**common_metadata, "checkpoint_kind": "last_complete_before_user_stop", "state_dict": last_state}, checkpoint_dir / "last.pt")

    optimizer_configuration = {
        "optimizer": config["training"]["optimizer"],
        "initial_learning_rate": float(config["training"]["learning_rate"]),
        **copy.deepcopy(config["training"]["optimizer_defaults"]),
        "optimizer_state_saved": False,
        "reason": "The unmodified official trainer saves model state_dict checkpoints, not optimizer state.",
    }
    scheduler_configuration = {
        "scheduler": config["training"]["scheduler"],
        "factor": float(config["training"]["lr_decay_factor"]),
        "patience": int(config["training"]["patience"]),
        "tolerance": float(config["training"]["tolerance"]),
        "min_lr": float(config["training"]["min_lr"]),
        "lr_decay_steps": int(config["training"]["lr_decay_steps"]),
        "observed_trained_learning_rates": [0.005, 0.0015, 0.00045, 0.000135],
        "restore_best": bool(config["training"]["restore_best"]),
    }
    (log_dir / "optimizer_configuration.json").write_text(_json(optimizer_configuration, indent=2), encoding="utf-8")
    (log_dir / "scheduler_configuration.json").write_text(_json(scheduler_configuration, indent=2), encoding="utf-8")
    (log_dir / "random_seed.json").write_text(_json({"seed": int(config["project"]["seed"])}, indent=2), encoding="utf-8")
    (log_dir / "git_commit.txt").write_text(str(_git_identity().get("commit", "unknown")) + "\n", encoding="utf-8")

    epoch_files = list(raw_dir.glob("epoch_*.pth"))
    summary = {
        "status": "training_stopped_by_user_best_recovered",
        "single_seed": int(config["project"]["seed"]),
        "best_epoch": int(best_epoch),
        "last_complete_epoch": int(last_epoch),
        "best_official_early_stopping_closure": float(best_score),
        "best_post_epoch_validation_score": float(best_post_epoch_score),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "raw_epoch_checkpoint_count": len(epoch_files),
        "raw_epoch_checkpoints_retained": True,
        "validation_events": len(validation_events),
        "full_official_early_stopping_completed": False,
        "next_action": "independent best-checkpoint evaluation and prediction export",
    }
    (log_dir / "training_summary.json").write_text(_json(summary, indent=2), encoding="utf-8")
    return summary


def load_best_model(config: dict[str, Any], dataloaders: dict[str, dict[str, Any]], device: str = "cuda") -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = resolve_path(config, config["evaluation"].get("published_checkpoint", Path(config["artifacts"]["checkpoint_dir"]) / "best.pt"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_official_model(config, dataloaders)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    model.load_state_dict(state if isinstance(state, dict) else checkpoint)
    model.to(device).eval()
    metadata = checkpoint if isinstance(state, dict) else {
        "official_validation_score": config["evaluation"]["checkpoint_validation_score"],
        "best_epoch": config["evaluation"]["checkpoint_best_epoch"],
    }
    return model, metadata


def evaluate_best(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("official evaluation requires CUDA")
    set_random_seed(int(config["project"]["seed"]))
    full_loaders = make_official_loaders(config, cuda=True, batch_size=1, to_cut=False, offset=0)
    model, checkpoint = load_best_model(config, full_loaders, "cuda")
    scalar = float(get_correlations(model, full_loaders["oracle"], device="cuda", as_dict=False, per_neuron=False))
    by_session_raw = get_correlations(model, full_loaders["oracle"], device="cuda", as_dict=True, per_neuron=True)
    by_session = {key: float(np.asarray(value).mean()) for key, value in by_session_raw.items()}
    target = float(config["evaluation"]["official_seed_42_final_test_main"])
    result = {
        "status": "oracle_evaluated_hidden_final_pending",
        "implementation": "sensorium.utility.scores.get_correlations",
        "checkpoint": str(resolve_path(config, Path(config["artifacts"]["checkpoint_dir"]) / "best.pt")),
        "checkpoint_training_validation_score": float(checkpoint["official_validation_score"]),
        "full_sequence_oracle_single_trial": scalar,
        "full_sequence_oracle_by_session": by_session,
        "official_seed_42_hidden_final_single_trial": target,
        "official_minus_local": None,
        "acceptance": "pending_hidden_final_server_evaluation",
        "reason": "competition-mouse final-test response arrays are zero placeholders locally",
    }
    artifact = resolve_path(config, "artifacts/phase1A/official_evaluation.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(_json(result, indent=2), encoding="utf-8")
    return result


def export_oracle_predictions(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("prediction export requires CUDA")
    set_random_seed(int(config["project"]["seed"]))
    loaders = make_official_loaders(config, cuda=True, batch_size=1, to_cut=False, offset=0)
    model, _checkpoint = load_best_model(config, loaders, "cuda")
    output_root = resolve_path(config, config["artifacts"]["prediction_dir"]) / "oracle"
    output_root.mkdir(parents=True, exist_ok=True)
    burn_in = int(config["evaluation"]["burn_in_frames"])
    sampling_rate = 30.0
    manifest: dict[str, Any] = {"split": "oracle", "burn_in_frames": burn_in, "sampling_rate_hz": sampling_rate, "sessions": {}}
    alignment_records: list[dict[str, Any]] = []

    with torch.inference_mode():
        for session_id, loader in loaders["oracle"].items():
            session_dir = output_root / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            trial_indices = list(iter(loader.sampler))
            neuron_ids = np.asarray(loader.dataset.neurons.unit_ids)
            files: list[str] = []
            for local_index, (trial_index, batch) in enumerate(zip(trial_indices, loader)):
                prediction = model(
                    batch.videos,
                    data_key=session_id,
                    behavior=batch.behavior,
                    pupil_center=batch.pupil_center,
                )
                response = batch.responses[:, :, burn_in:]
                prediction = prediction[:, -response.shape[-1] :, :]
                predicted = prediction[0].float().cpu().numpy()
                ground_truth = response[0].transpose(0, 1).float().cpu().numpy()
                if predicted.shape != ground_truth.shape:
                    raise AssertionError(f"alignment mismatch for {session_id} trial {trial_index}: {predicted.shape} != {ground_truth.shape}")
                frame_index = np.arange(burn_in, burn_in + predicted.shape[0], dtype=np.int64)
                timestamps = frame_index.astype(np.float64) / sampling_rate
                valid_mask = np.ones(predicted.shape[0], dtype=bool)
                filename = f"trial_{int(trial_index):04d}.npz"
                np.savez_compressed(
                    session_dir / filename,
                    predicted_response=predicted,
                    ground_truth_response=ground_truth,
                    timestamps=timestamps,
                    valid_mask=valid_mask,
                    neuron_ids=neuron_ids,
                    session_id=np.asarray(session_id),
                    trial_index=np.asarray(int(trial_index), dtype=np.int64),
                    model_architecture=np.asarray("official Factorized3dCore [32,64,128]; kernels spatial 11/5/5 temporal 11/5/5"),
                    checkpoint=np.asarray(str(resolve_path(config, Path(config["artifacts"]["checkpoint_dir"]) / "best.pt"))),
                    sampling_rate_hz=np.asarray(sampling_rate),
                    input_temporal_context=np.asarray(int(config["data"]["frames"])),
                    dataset_split=np.asarray("oracle"),
                )
                files.append(filename)
                alignment_records.append(
                    {
                        "session_id": session_id,
                        "trial_index": int(trial_index),
                        "source_frames": int(batch.videos.shape[2]),
                        "raw_prediction_frames": int(batch.videos.shape[2] - 18),
                        "burn_in": burn_in,
                        "exported_frames": int(predicted.shape[0]),
                        "first_frame_index": int(frame_index[0]),
                        "last_frame_index": int(frame_index[-1]),
                    }
                )
            manifest["sessions"][session_id] = {"neuron_count": int(neuron_ids.size), "trial_count": len(files), "files": files}
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(_json(manifest, indent=2), encoding="utf-8")

    expected_trials = sum(len(loader) for loader in loaders["oracle"].values())
    if len(alignment_records) != expected_trials:
        raise AssertionError(f"exported {len(alignment_records)} trials, expected {expected_trials}")
    alignment_path = resolve_path(config, "artifacts/phase1A/temporal_alignment.json")
    alignment_path.write_text(_json(alignment_records, indent=2), encoding="utf-8")
    report_path = resolve_path(config, "reports/phase1A_temporal_alignment.md")
    report_path.write_text(
        "# Phase 1A temporal alignment validation\n\n"
        "Status: **PASS for the exported oracle predictions.**\n\n"
        "- Source stimulus, response, behavior, and pupil arrays are cut by the official loader at the same trial boundaries.\n"
        "- Valid temporal convolutions reduce predictions by 18 frames (`11,5,5` kernels).\n"
        "- The official metric removes response frames 0–49 and selects the last matching model predictions.\n"
        "- Therefore exported prediction row 0, response row 0, and timestamp row 0 all refer to original frame index 50.\n"
        "- Every export retains original trial order and strictly increasing original frame indices.\n"
        "- Window reconstruction is not used for this export: each oracle trial is inferred as one complete continuous sequence.\n"
        "- Boundary handling and burn-in are session-local; no state or frame is carried between trials.\n\n"
        f"Machine-readable alignment records: `{alignment_path}`\n"
        f"Prediction manifest: `{manifest_path}`\n\n"
        "No trajectory metric was computed.\n",
        encoding="utf-8",
    )
    return {"status": "pass", "trial_count": len(alignment_records), "manifest": str(manifest_path), "alignment": str(alignment_path)}


def generate_final_test_submission(config: dict[str, Any]) -> dict[str, Any]:
    """Generate, but do not upload, the official main-track final-test file."""

    if not torch.cuda.is_available():
        raise RuntimeError("official submission generation requires CUDA")
    set_random_seed(int(config["project"]["seed"]))
    loaders = make_official_loaders(config, cuda=True, batch_size=1, to_cut=False, offset=0)
    model, checkpoint = load_best_model(config, loaders, "cuda")
    output_dir = resolve_path(config, "submissions/dynamic_official_reproduction")
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_submission(
        loaders,
        model,
        deeplake_ds=False,
        path=str(output_dir),
        tier="final_test",
        track="main",
        skip=int(config["evaluation"]["burn_in_frames"]),
        device="cuda",
    )
    submission_path = output_dir / "predictions_file_final_test_main_main_track.parquet.brotli"
    if not submission_path.is_file():
        raise FileNotFoundError(f"official generator did not create the expected submission: {submission_path}")

    import pandas as pd

    frame = pd.read_parquet(submission_path, engine="pyarrow")
    sessions = sorted(frame["mouse"].unique().tolist())
    if sessions != sorted(EXPECTED_SESSIONS):
        raise AssertionError(f"submission session mismatch: {sessions}")
    if frame.empty or frame["prediction"].isna().any() or frame["neuron_ids"].isna().any():
        raise AssertionError("submission contains missing rows or arrays")
    per_session_rows = {key: int(value) for key, value in frame.groupby("mouse").size().to_dict().items()}
    digest_builder = hashlib.sha256()
    with submission_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    manifest = {
        "status": "ready_for_user_authorized_upload",
        "uploaded": False,
        "generator": "sensorium.utility.submission.generate_submission",
        "tier": "final_test",
        "track": "main",
        "checkpoint": str(resolve_path(config, Path(config["artifacts"]["checkpoint_dir"]) / "best.pt")),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_validation_closure": float(checkpoint["official_validation_score"]),
        "submission": str(submission_path),
        "rows": int(len(frame)),
        "rows_by_session": per_session_rows,
        "bytes": int(submission_path.stat().st_size),
        "sha256": digest,
        "note": "External upload was not performed without explicit user authorization.",
    }
    manifest_path = output_dir / "submission_manifest.json"
    manifest_path.write_text(_json(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Full official Dynamic Sensorium Factorized baseline reproduction")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("smoke")
    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--iterations", type=int, default=200)
    benchmark_parser.add_argument("--warmup", type=int, default=20)
    subparsers.add_parser("train")
    recover_parser = subparsers.add_parser("recover-stopped")
    recover_parser.add_argument("--best-epoch", type=int, required=True)
    recover_parser.add_argument("--last-epoch", type=int, required=True)
    recover_parser.add_argument("--best-score", type=float, required=True)
    recover_parser.add_argument("--best-post-score", type=float, required=True)
    subparsers.add_parser("evaluate")
    subparsers.add_parser("export")
    subparsers.add_parser("submission")
    args = parser.parse_args()
    config = load_official_config(args.config)
    if args.command == "audit":
        result = audit_architecture(config)
    elif args.command == "smoke":
        result = smoke_test(config)
    elif args.command == "benchmark":
        result = benchmark(config, measured_microbatches=args.iterations, warmup_microbatches=args.warmup)
    elif args.command == "train":
        result = train_full(config)
    elif args.command == "recover-stopped":
        result = recover_user_stopped_training(
            config,
            best_epoch=args.best_epoch,
            last_epoch=args.last_epoch,
            best_score=args.best_score,
            best_post_epoch_score=args.best_post_score,
        )
    elif args.command == "evaluate":
        result = evaluate_best(config)
    elif args.command == "export":
        result = export_oracle_predictions(config)
    else:
        result = generate_final_test_submission(config)
    print(_json(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
