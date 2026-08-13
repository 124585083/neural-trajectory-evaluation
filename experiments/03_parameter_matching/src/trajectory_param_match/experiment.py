from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import types
from typing import Any, Iterator

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT.parent
PHASE1_ROOT = EXPERIMENTS_ROOT / "01_baselines"
PHASE1_SRC = PHASE1_ROOT / "src"
if str(PHASE1_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE1_SRC))

from trajectory_eval.official_dynamic import (  # noqa: E402
    EXPECTED_NEURONS,
    EXPECTED_SESSIONS,
    FUNCTIONAL_NEURALPREDICTORS_COMMIT,
    OFFICIAL_SENSORIUM_COMMIT,
    Tee,
    build_official_model,
    get_correlations,
    make_official_loaders,
    parameter_counts,
    set_random_seed,
    video_training_loop,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dynamic_parameter_matched.yaml"
EXPECTED_CHANNELS = (16, 32, 64)
EXPECTED_PARAMETER_COUNTS = {
    "core": 98_672,
    "readout": 2_763_106,
    "shifter": 285,
    "total": 2_862_063,
}
STATIC_PARAMETER_COUNTS = {
    "core": 50_624,
    "readout": 2_763_106,
    "shifter": 285,
    "total": 2_814_015,
}
EXPECTED_TRAIN_COUNTS = (348, 329, 354, 359, 354)
EXPECTED_ORACLE_COUNTS = (58, 56, 60, 60, 59)
DATA_LOCK_KEYS = (
    "sessions",
    "physical_batch_size_per_session",
    "effective_batch_size",
    "frames",
    "scale",
    "max_frame",
    "offset",
    "include_behavior",
    "include_pupil_centers",
    "to_cut",
    "use_file_cache",
)


def _json(value: Any, *, indent: int = 2) -> str:
    def default(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

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


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _fingerprint(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        if array.dtype.kind in "OUS":
            digest.update(json.dumps(array.tolist(), ensure_ascii=False).encode("utf-8"))
        else:
            digest.update(array.tobytes())
    return digest.hexdigest()


def audit_config(config: dict[str, Any]) -> dict[str, Any]:
    dynamic_reference = _load_yaml(resolve_path(config, config["references"]["dynamic_config"]))
    static_reference = _load_yaml(resolve_path(config, config["references"]["static_training_snapshot"]))

    for key in DATA_LOCK_KEYS:
        if key == "sessions":
            candidate = tuple(config["data"][key])
            expected_dynamic = tuple(dynamic_reference["data"][key])
            expected_static = tuple(static_reference["data"][key])
        else:
            candidate = config["data"][key]
            expected_dynamic = dynamic_reference["data"][key]
            expected_static = static_reference["data"][key]
        if candidate != expected_dynamic or candidate != expected_static:
            raise AssertionError(
                f"data.{key} differs: candidate={candidate!r}, dynamic={expected_dynamic!r}, static={expected_static!r}"
            )

    expected_model = copy.deepcopy(dynamic_reference["model"])
    expected_model["core"]["hidden_channels"] = list(EXPECTED_CHANNELS)
    if config["model"] != expected_model:
        raise AssertionError("model differs from the full Dynamic baseline beyond the locked channel reduction")
    if tuple(config["matching"]["selected_channels"]) != EXPECTED_CHANNELS:
        raise AssertionError("matching.selected_channels changed")
    if float(config["matching"]["selected_width_multiplier"]) != 0.5:
        raise AssertionError("the predeclared half-width multiplier changed")
    if config["training"] != dynamic_reference["training"]:
        raise AssertionError("training protocol differs from the full Dynamic baseline")

    common_static_training = set(config["training"]) & set(static_reference["training"])
    ignored = {"protocol"}
    for key in sorted(common_static_training - ignored):
        if config["training"][key] != static_reference["training"][key]:
            raise AssertionError(f"training.{key} differs from Static: {config['training'][key]!r}")
    if int(config["evaluation"]["burn_in_frames"]) != int(static_reference["evaluation"]["burn_in_frames"]):
        raise AssertionError("evaluation burn-in differs from Static")

    target = int(config["matching"]["target_total_parameters"])
    if target != STATIC_PARAMETER_COUNTS["total"]:
        raise AssertionError("Static parameter target changed")
    relative = (EXPECTED_PARAMETER_COUNTS["total"] - target) / target
    preferred = float(config["matching"]["preferred_relative_tolerance"])
    maximum = float(config["matching"]["maximum_relative_tolerance"])
    if abs(relative) > preferred or abs(relative) > maximum:
        raise AssertionError(f"parameter match outside tolerance: {relative:.3%}")

    return {
        "status": "pass",
        "channels": list(EXPECTED_CHANNELS),
        "width_multiplier": 0.5,
        "spatial_kernels": [11, 5, 5],
        "temporal_kernels": [11, 5, 5],
        "expected_parameter_counts": EXPECTED_PARAMETER_COUNTS,
        "static_parameter_counts": STATIC_PARAMETER_COUNTS,
        "absolute_total_difference": EXPECTED_PARAMETER_COUNTS["total"] - target,
        "relative_total_difference": relative,
        "readout_parameter_difference": EXPECTED_PARAMETER_COUNTS["readout"] - STATIC_PARAMETER_COUNTS["readout"],
        "preferred_tolerance": preferred,
        "maximum_tolerance": maximum,
    }


def audit_data_contract(config: dict[str, Any], *, save: bool = True) -> dict[str, Any]:
    audit_config(config)
    candidate_root = resolve_path(config, config["data"]["root"])
    static_snapshot = _load_yaml(resolve_path(config, config["references"]["static_training_snapshot"]))
    static_snapshot_root = (PHASE1_ROOT / static_snapshot["data"]["root"]).resolve()
    same_root = os.path.samefile(candidate_root, static_snapshot_root)
    if not same_root:
        raise AssertionError(f"candidate and Static resolve to different data roots: {candidate_root} vs {static_snapshot_root}")

    checkpoint_data = static_snapshot["data"]
    for key in DATA_LOCK_KEYS:
        left = tuple(config["data"][key]) if key == "sessions" else config["data"][key]
        right = tuple(checkpoint_data[key]) if key == "sessions" else checkpoint_data[key]
        if left != right:
            raise AssertionError(f"candidate data.{key} differs from the trained Static checkpoint")

    sessions: list[dict[str, Any]] = []
    for index, session in enumerate(EXPECTED_SESSIONS):
        session_path = candidate_root / session
        tiers = np.load(session_path / "meta/trials/tiers.npy", allow_pickle=True).astype(str)
        trial_ids = np.load(session_path / "meta/trials/trial_idx.npy", allow_pickle=True).astype(str)
        unit_ids = np.load(session_path / "meta/neurons/unit_ids.npy", allow_pickle=True)
        if len(unit_ids) != EXPECTED_NEURONS[index]:
            raise AssertionError(f"neuron count changed for {session}")
        entry: dict[str, Any] = {
            "session": session,
            "neurons": int(len(unit_ids)),
            "neuron_id_fingerprint": _fingerprint(unit_ids),
            "tiers": {},
        }
        for tier, expected_count in (("train", EXPECTED_TRAIN_COUNTS[index]), ("oracle", EXPECTED_ORACLE_COUNTS[index])):
            indices = np.flatnonzero(tiers == tier)
            ids = trial_ids[indices]
            if len(indices) != expected_count:
                raise AssertionError(f"{session} {tier} count changed: {len(indices)} != {expected_count}")
            entry["tiers"][tier] = {
                "count": int(len(indices)),
                "dataset_indices": indices.tolist(),
                "trial_ids": ids.tolist(),
                "fingerprint": _fingerprint(indices, ids),
            }
        sessions.append(entry)

    result = {
        "status": "pass",
        "candidate_data_root": str(candidate_root),
        "static_data_root": str(static_snapshot_root),
        "same_physical_data_root": same_root,
        "static_reference_checkpoint": str(resolve_path(config, config["references"]["static_checkpoint"])),
        "static_reference_total_parameters": int(STATIC_PARAMETER_COUNTS["total"]),
        "total_train_trials": int(sum(item["tiers"]["train"]["count"] for item in sessions)),
        "total_oracle_trials": int(sum(item["tiers"]["oracle"]["count"] for item in sessions)),
        "total_neurons": int(sum(item["neurons"] for item in sessions)),
        "loader_contract": {key: config["data"][key] for key in DATA_LOCK_KEYS if key != "root"},
        "sessions": sessions,
    }
    if save:
        output = resolve_path(config, Path(config["artifacts"]["audit_dir"]) / "data_split_lock.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_json(result), encoding="utf-8")
    return result


def _conv_signature(model: torch.nn.Module) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for layer in model.core.features:
        current = []
        for name, module in layer.named_children():
            if isinstance(module, torch.nn.Conv3d):
                current.append(
                    {
                        "name": name,
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "kernel_size": list(module.kernel_size),
                        "stride": list(module.stride),
                        "padding": list(module.padding),
                    }
                )
        result.append(current)
    return result


def audit_architecture(config: dict[str, Any], *, save: bool = True) -> dict[str, Any]:
    config_result = audit_config(config)
    dataloaders = make_official_loaders(config, cuda=False, batch_size=1)
    model = build_official_model(config, dataloaders)
    counts = parameter_counts(model)
    if counts != EXPECTED_PARAMETER_COUNTS:
        raise AssertionError(f"parameter counts changed: {counts} != {EXPECTED_PARAMETER_COUNTS}")
    if len(model.core.features) != 3:
        raise AssertionError("Factorized3D depth changed")

    expected = [
        [
            {"name": "conv_spatial", "in_channels": 3, "out_channels": 16, "kernel_size": [1, 11, 11], "stride": [1, 1, 1], "padding": [0, 0, 0]},
            {"name": "conv_temporal", "in_channels": 16, "out_channels": 16, "kernel_size": [11, 1, 1], "stride": [1, 1, 1], "padding": [0, 0, 0]},
        ],
        [
            {"name": "conv_spatial_1", "in_channels": 16, "out_channels": 32, "kernel_size": [1, 5, 5], "stride": [1, 1, 1], "padding": [0, 0, 0]},
            {"name": "conv_temporal_1", "in_channels": 32, "out_channels": 32, "kernel_size": [5, 1, 1], "stride": [1, 1, 1], "padding": [0, 0, 0]},
        ],
        [
            {"name": "conv_spatial_2", "in_channels": 32, "out_channels": 64, "kernel_size": [1, 5, 5], "stride": [1, 1, 1], "padding": [0, 0, 0]},
            {"name": "conv_temporal_2", "in_channels": 64, "out_channels": 64, "kernel_size": [5, 1, 1], "stride": [1, 1, 1], "padding": [0, 0, 0]},
        ],
    ]
    signatures = _conv_signature(model)
    if signatures != expected:
        raise AssertionError(f"kernel/channel lock failed: {signatures!r}")

    data_keys = tuple(dataloaders["train"])
    outdims = tuple(model.readout[key].outdims for key in data_keys)
    if data_keys != EXPECTED_SESSIONS or outdims != EXPECTED_NEURONS:
        raise AssertionError("session/readout/neuron lock failed")
    batch = next(iter(dataloaders["oracle"][data_keys[0]]))
    model.eval()
    with torch.inference_mode():
        core_output = model.core(batch.videos)
        output = model(batch.videos, data_key=data_keys[0], behavior=batch.behavior, pupil_center=batch.pupil_center)
    if tuple(batch.videos.shape[1:]) != (3, 80, 36, 64):
        raise AssertionError(f"input shape changed: {tuple(batch.videos.shape)}")
    if tuple(core_output.shape[1:]) != (64, 62, 18, 46):
        raise AssertionError(f"core output changed: {tuple(core_output.shape)}")
    if tuple(output.shape[1:]) != (62, EXPECTED_NEURONS[0]):
        raise AssertionError(f"response output changed: {tuple(output.shape)}")

    result = {
        "status": "pass",
        "scientific_label": config["project"]["scientific_label"],
        "sensorium_commit": OFFICIAL_SENSORIUM_COMMIT,
        "neuralpredictors_commit": FUNCTIONAL_NEURALPREDICTORS_COMMIT,
        "sessions": list(data_keys),
        "neuron_counts": list(outdims),
        "channels": list(EXPECTED_CHANNELS),
        "convolutions": signatures,
        "input_shape": list(batch.videos.shape[1:]),
        "core_output_shape": list(core_output.shape[1:]),
        "response_output_shape": list(output.shape[1:]),
        "parameter_counts": counts,
        "static_parameter_counts": STATIC_PARAMETER_COUNTS,
        "absolute_total_difference": counts["total"] - STATIC_PARAMETER_COUNTS["total"],
        "relative_total_difference": config_result["relative_total_difference"],
        "model": str(model),
    }
    if save:
        output_path = resolve_path(config, Path(config["artifacts"]["audit_dir"]) / "architecture_audit.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_json(result), encoding="utf-8")
    return result


def audit_all(config: dict[str, Any]) -> dict[str, Any]:
    config_result = audit_config(config)
    data_result = audit_data_contract(config)
    architecture_result = audit_architecture(config)
    result = {
        "status": "pass",
        "config": config_result,
        "data": {
            key: data_result[key]
            for key in ("same_physical_data_root", "total_train_trials", "total_oracle_trials", "total_neurons")
        },
        "architecture": {
            key: architecture_result[key]
            for key in ("channels", "parameter_counts", "relative_total_difference", "input_shape", "core_output_shape", "response_output_shape")
        },
    }
    output = resolve_path(config, Path(config["artifacts"]["audit_dir"]) / "audit_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_json(result), encoding="utf-8")
    return result


class LimitedLoader:
    def __init__(self, loader: Any, batches: int = 1) -> None:
        self.loader = loader
        self.dataset = loader.dataset
        self.batches = min(batches, len(loader))

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self.loader)
        for _ in range(self.batches):
            yield next(iterator)

    def __len__(self) -> int:
        return self.batches


def _limited_tiers(dataloaders: dict[str, dict[str, Any]]) -> dict[str, dict[str, LimitedLoader]]:
    return {tier: {key: LimitedLoader(loader) for key, loader in loaders.items()} for tier, loaders in dataloaders.items()}


def smoke_test(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the parameter-matched smoke test")
    audit_all(config)
    set_random_seed(int(config["project"]["seed"]))
    dataloaders = make_official_loaders(config, cuda=True)
    limited = _limited_tiers(dataloaders)
    model = build_official_model(config, dataloaders)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_path(config, Path("checkpoints") / "smoke" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
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
        interval=1,
        tolerance=float(config["training"]["tolerance"]),
        lr_decay_factor=float(config["training"]["lr_decay_factor"]),
        min_lr=float(config["training"]["min_lr"]),
        restore_best=True,
        device="cuda",
        detach_core=False,
        deeplake_ds=False,
        save_checkpoints=True,
        checkpoint_save_path=str(output_dir) + os.sep,
        chpt_save_step=1,
    )
    checkpoint = output_dir / "final.pth"
    if not checkpoint.is_file():
        raise AssertionError("smoke trainer did not save final.pth")
    if not all(torch.isfinite(value).all() for value in state_dict.values() if torch.is_tensor(value)):
        raise AssertionError("smoke state contains NaN or Inf")
    reloaded = build_official_model(config, dataloaders).cuda()
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True)
    reload_score = float(get_correlations(reloaded, limited["oracle"], device="cuda", as_dict=False, per_neuron=False))
    result = {
        "status": "pass",
        "scope": "smoke_only_not_a_training_result",
        "parameter_counts": EXPECTED_PARAMETER_COUNTS,
        "optimizer_steps": 1,
        "trainer_score": float(score),
        "trainer_output": trainer_output,
        "reloaded_validation_score": reload_score,
        "checkpoint": str(checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    output = resolve_path(config, Path(config["artifacts"]["audit_dir"]) / "smoke_test.json")
    output.write_text(_json(result), encoding="utf-8")
    return result


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
    }


def train_full(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal training")
    audit = audit_all(config)
    checkpoint_dir = resolve_path(config, config["artifacts"]["checkpoint_dir"])
    log_dir = resolve_path(config, config["artifacts"]["log_dir"])
    raw_dir = checkpoint_dir / "official_raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (checkpoint_dir / "best.pt").exists() or (checkpoint_dir / "last.pt").exists():
        raise FileExistsError("formal matched checkpoints already exist; refusing to overwrite")
    if any(raw_dir.iterdir()):
        raise FileExistsError("formal raw checkpoint directory is not empty")

    snapshot = copy.deepcopy(config)
    snapshot.pop("_config_path", None)
    snapshot.pop("_project_root", None)
    (log_dir / "training_config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    (log_dir / "environment.json").write_text(_json(_environment_snapshot()), encoding="utf-8")
    (log_dir / "audit_summary.json").write_text(_json(audit), encoding="utf-8")
    (log_dir / "training_started.json").write_text(
        _json({"status": "running", "pid": os.getpid(), "started_unix": time.time()}), encoding="utf-8"
    )

    set_random_seed(int(config["project"]["seed"]))
    dataloaders = make_official_loaders(config, cuda=True)
    model = build_official_model(config, dataloaders)
    events: list[dict[str, Any]] = []
    original_scores_get = video_training_loop.scores.get_correlations
    original_direct_get = video_training_loop.get_correlations

    def logged_correlations(*args: Any, **kwargs: Any) -> Any:
        value = original_scores_get(*args, **kwargs)
        event = {"event_index": len(events), "unix_time": time.time(), "value": float(np.mean(value))}
        events.append(event)
        with (log_dir / "validation_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(_json(event, indent=None) + "\n")
        return value

    video_training_loop.scores.get_correlations = logged_correlations
    video_training_loop.get_correlations = logged_correlations
    original_os = video_training_loop.os
    video_training_loop.os = types.SimpleNamespace(listdir=os.listdir, remove=lambda _path: None)
    started = time.time()
    try:
        with (log_dir / "official_trainer.log").open("w", encoding="utf-8", buffering=1) as handle:
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
            _json({"status": "interrupted", "type": type(error).__name__, "message": str(error), "unix_time": time.time()}),
            encoding="utf-8",
        )
        raise
    finally:
        video_training_loop.scores.get_correlations = original_scores_get
        video_training_loop.get_correlations = original_direct_get
        video_training_loop.os = original_os

    epoch_files = sorted(
        raw_dir.glob("epoch_*.pth"),
        key=lambda path: int(re.search(r"epoch_(\d+)", path.name).group(1)),
    )
    if not epoch_files:
        raise RuntimeError("formal trainer completed without epoch checkpoints")
    last_raw = epoch_files[-1]
    last_epoch = int(re.search(r"epoch_(\d+)", last_raw.name).group(1))
    last_state = torch.load(last_raw, map_location="cpu", weights_only=True)
    metadata = {
        "run_name": config["project"]["run_name"],
        "scientific_label": config["project"]["scientific_label"],
        "seed": int(config["project"]["seed"]),
        "config": snapshot,
        "parameter_counts": EXPECTED_PARAMETER_COUNTS,
        "static_parameter_counts": STATIC_PARAMETER_COUNTS,
        "relative_parameter_difference": (EXPECTED_PARAMETER_COUNTS["total"] - STATIC_PARAMETER_COUNTS["total"]) / STATIC_PARAMETER_COUNTS["total"],
        "environment": _environment_snapshot(),
        "official_validation_score": float(score),
        "trainer_output": trainer_output,
        "last_epoch": last_epoch,
        "elapsed_seconds": time.time() - started,
    }
    torch.save({**metadata, "checkpoint_kind": "best_restored", "state_dict": best_state}, checkpoint_dir / "best.pt")
    torch.save({**metadata, "checkpoint_kind": "last_pre_restore", "state_dict": last_state}, checkpoint_dir / "last.pt")
    summary = {
        "status": "training_complete",
        "seed": int(config["project"]["seed"]),
        "validation_score": float(score),
        "last_epoch": last_epoch,
        "elapsed_seconds": time.time() - started,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "raw_epoch_checkpoints": len(epoch_files),
        "validation_events": len(events),
    }
    (log_dir / "training_summary.json").write_text(_json(summary), encoding="utf-8")
    (log_dir / "training_started.json").write_text(
        _json({"status": "complete", "pid": os.getpid(), "completed_unix": time.time()}), encoding="utf-8"
    )
    return summary


def evaluate_best(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    audit_all(config)
    dataloaders = make_official_loaders(config, cuda=True, batch_size=1, to_cut=False, offset=0)
    model = build_official_model(config, dataloaders).cuda()
    checkpoint = resolve_path(config, config["evaluation"]["published_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    model.load_state_dict(state if isinstance(state, dict) else payload, strict=True)
    model.eval()
    by_neuron = get_correlations(model, dataloaders["oracle"], device="cuda", as_dict=True, per_neuron=True)
    result = {
        "status": "full_sequence_oracle_evaluated",
        "checkpoint": str(checkpoint),
        "implementation": "sensorium.utility.scores.get_correlations",
        "full_sequence_oracle_single_trial": float(np.mean(np.hstack(list(by_neuron.values())))),
        "full_sequence_oracle_by_session": {key: float(np.mean(value)) for key, value in by_neuron.items()},
        "burn_in_frames": 50,
        "retained_source_frames": [50, 299],
        "static_oracle_reference": float(config["evaluation"]["static_oracle_reference"]),
        "full_dynamic_oracle_reference": float(config["evaluation"]["dynamic_full_oracle_reference"]),
    }
    output = resolve_path(config, Path(config["artifacts"]["audit_dir"]) / "official_evaluation.json")
    output.write_text(_json(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter-matched Dynamic Sensorium experiment")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("command", choices=("audit-config", "audit-data", "audit", "smoke", "train", "evaluate"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "audit-config":
        result = audit_config(config)
    elif args.command == "audit-data":
        result = audit_data_contract(config)
    elif args.command == "audit":
        result = audit_all(config)
    elif args.command == "smoke":
        result = smoke_test(config)
    elif args.command == "train":
        result = train_full(config)
    else:
        result = evaluate_best(config)
    print(_json(result), flush=True)


if __name__ == "__main__":
    main()
