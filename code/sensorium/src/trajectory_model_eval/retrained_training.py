"""From-scratch Part 3 encoders with validation-only, resumable training.

The objective, cycling, optimizer, scheduler and patience cycles follow the
retained Sensorium trainer. Its generator state is made explicit so that an
interrupted run can resume at a completed epoch, including its RNG state.
Historical training entry points and learned parameters are not used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

from .unified_contract import canonical_digest
from sensorium_runtime.paths import SOURCE_ROOT, repository_path


REPOSITORY = Path(__file__).resolve().parents[4]
ARCHITECTURE_CONFIGS = {
    "static": "experiments/01_baselines/records/static/training_config.yaml",
    "dynamic": "experiments/03_parameter_matching/configs/dynamic_parameter_matched.yaml",
}
# Retain frozen provenance keys; repository_path maps storage without changing
# the recorded configuration names or their byte hashes.
EXPECTED_PARAMETERS = {
    "static": {"core": 50624, "readout": 2763106, "shifter": 285, "total": 2814015},
    "dynamic": {"core": 98672, "readout": 2763106, "shifter": 285, "total": 2862063},
}


def _digest(value: Any) -> str:
    return canonical_digest(value)


def _file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def configure_sources(sensorium_source=None, neuralpredictors_source=None) -> None:
    """Use supplied official checkouts and this repository's unchanged builders."""
    for path in (sensorium_source, neuralpredictors_source, SOURCE_ROOT):
        if path is not None:
            sys.path.insert(0, str(Path(path).resolve()))


def load_architecture_config(kind: str) -> dict:
    import yaml

    if kind not in ARCHITECTURE_CONFIGS:
        raise ValueError("Only static and reduced dynamic are part of this protocol")
    source = repository_path(ARCHITECTURE_CONFIGS[kind])
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["_architecture_source"] = ARCHITECTURE_CONFIGS[kind]
    config["_architecture_sha256"] = _file_digest(source)
    return config


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cpu_state(model) -> dict:
    return {key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in model.state_dict().items()}


def state_digest(state: dict) -> str:
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode())
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            result.update(str((array.shape, array.dtype)).encode())
            result.update(array.tobytes())
        else:
            result.update(repr(value).encode())
    return result.hexdigest()


def build_encoder(kind: str, train_loaders: dict, seed: int = 42):
    """Initialize the original architecture using only Part 3 train batches."""
    configure_sources()
    config = load_architecture_config(kind)
    config["project"]["seed"] = int(seed)
    _seed(seed)
    if any(getattr(loader.dataset, "role", None) != "train" for loader in train_loaders.values()):
        raise ValueError("Model initialization requires audited Part 3 train datasets")
    if tuple(train_loaders) != tuple(config["data"]["sessions"]):
        raise ValueError("The five-session order differs from the locked architecture")
    # The official builder obtains dimensions and readout bias from train batches.
    # It receives no validation/test loader and no pretrained state dictionary.
    if kind == "static":
        from trajectory_eval.static_dynamic import build_model
        model = build_model(config, {"train": train_loaders})
    else:
        from trajectory_eval.official_dynamic import build_official_model
        model = build_official_model(config, {"train": train_loaders})
    counts = {part: sum(p.numel() for p in getattr(model, part).parameters())
              for part in ("core", "readout", "shifter")}
    counts["total"] = sum(p.numel() for p in model.parameters())
    if counts != EXPECTED_PARAMETERS[kind]:
        raise ValueError(f"The {kind} architecture changed: {counts}")
    return model, config, counts


def assert_training_roles(loaders: dict, manifest: dict) -> dict:
    """Check actual parent trial IDs, not the trainer's historical dictionary key."""
    if set(loaders) != {"train", "oracle"}:
        raise ValueError("The trainer accepts train and validation only; no test loader")
    audit = {}
    for key, role in (("train", "train"), ("oracle", "validation")):
        audit[role] = {}
        for session, loader in loaders[key].items():
            dataset = loader.dataset
            actual = list(map(int, dataset.dataset_indices))
            expected = [int(row["dataset_index"]) for row in manifest["trials"]
                        if row["session"] == session and row["role"] == role]
            if getattr(dataset, "role", None) != role or sorted(actual) != sorted(expected):
                raise ValueError(f"Actual {session}/{key} parent trial IDs differ from {role}")
            if len(actual) != len(set(actual)) or len(dataset) != len(actual):
                raise ValueError("The dataset length must equal its unique parent trials")
            audit[role][session] = actual
    if set(loaders["train"]) != set(loaders["oracle"]):
        raise ValueError("Train and validation sessions differ")
    for session in loaders["train"]:
        if set(audit["train"][session]) & set(audit["validation"][session]):
            raise ValueError("A parent trial crosses train and validation")
    return audit


class LimitedLoader:
    """Bound an implementation smoke check without changing dataset identities."""

    def __init__(self, loader, batches=1):
        self.loader, self.dataset = loader, loader.dataset
        self.batches = min(int(batches), len(loader))

    def __iter__(self):
        from itertools import islice
        return iter(islice(iter(self.loader), self.batches))

    def __len__(self):
        return self.batches


def _long_cycle(loaders):
    """Match neuralpredictors.training.LongCycler, including session ordering."""
    def cycle(loader):
        while True:
            yield from loader
    iterators = {key: cycle(loader) for key, loader in loaders.items()}
    for _ in range(max(map(len, loaders.values()))):
        for key, iterator in iterators.items():
            yield key, next(iterator)


def _rng_state(loaders):
    generators = {}
    for role, session_loaders in loaders.items():
        for session, loader in session_loaders.items():
            loader = getattr(loader, "loader", loader)
            generator = getattr(loader, "generator", None)
            if generator is not None:
                generators[(role, session)] = generator.get_state()
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_generators": generators}


def _restore_rng(state, loaders):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    for (role, session), value in state["loader_generators"].items():
        loader = getattr(loaders[role][session], "loader", loaders[role][session])
        loader.generator.set_state(value)


def sensorium_objective(model, loaders, session, data, device):
    from neuralpredictors.measures import modules

    values = data._asdict() if hasattr(data, "_asdict") else data
    values = {key: value.to(device) if torch.is_tensor(value) else value
              for key, value in values.items()}
    prediction = model(values["videos"], data_key=session, **values)
    target = values["responses"].transpose(2, 1)[:, -prediction.shape[1]:, :]
    if prediction.shape[1] != 62 or prediction.shape != target.shape:
        raise ValueError("The 80-frame training snippet must align to 62 target frames")
    regularizer = model.core.regularizer()
    if isinstance(regularizer, tuple):
        regularizer = sum(regularizer)
    regularizer = regularizer + model.readout.regularizer(session)
    # The retained loader samples a subset from a full source dataset. Preserve
    # its metadata-only loss multiplier and therefore its regularization weight.
    source_length = loaders[session].dataset.original_dataset_length
    scale = np.sqrt(source_length / values["videos"].shape[0])
    return scale * modules.PoissonLoss(avg=False)(prediction, target) + regularizer


def sensorium_score(model, validation_loaders, device):
    from sensorium.utility.scores import get_correlations

    class DeviceLoader:
        def __init__(self, loader):
            self.loader, self.dataset = loader, loader.dataset

        def __len__(self):
            return len(self.loader)

        def __iter__(self):
            for batch in self.loader:
                if hasattr(batch, "_fields"):
                    yield type(batch)(*(value.to(device) if torch.is_tensor(value) else value
                                        for value in batch))
                else:
                    yield {key: value.to(device) if torch.is_tensor(value) else value
                           for key, value in batch.items()}

    moved = {session: DeviceLoader(loader) for session, loader in validation_loaders.items()}
    return float(get_correlations(model, moved, device=device,
                                  as_dict=False, per_neuron=False, deeplake_ds=False))


def run_training(model, loaders, training, output_dir, identity, *, device="cuda",
                 objective=sensorium_objective, score=sensorium_score, resume=False,
                 stop_after_epoch=None):
    """Run the checked trainer with checkpointed patience-cycle and RNG state.

    A bounded invocation leaves a resumable state and is never marked complete.
    Resume restores only the same Part 3 run's last completed epoch. A process
    interrupted within an epoch restarts that epoch from the preceding boundary.
    """
    if set(loaders) != {"train", "oracle"}:
        raise ValueError("Test loaders must not enter the training process")
    if int(training["interval"]) != 1 or not training["restore_best"]:
        raise ValueError("Only the locked interval=1, restore_best protocol is supported")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                 **training.get("optimizer_defaults", {}))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(training["lr_decay_factor"]),
        patience=int(training["patience"]), threshold=float(training["tolerance"]),
        min_lr=float(training["min_lr"]), threshold_mode="abs")
    identity_digest = _digest(identity)
    progress = {"epoch": 0, "best_epoch": 0, "best_score": None,
                "patience_counter": 0, "cycle": 0, "elapsed_seconds": 0.0,
                "selection_events": [], "curve": [], "stop_reason": None}

    def evaluate(event, epoch):
        model.eval()
        with torch.no_grad():
            value = float(score(model, loaders["oracle"], device))
        model.train()
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite validation score at epoch {epoch}")
        with (output_dir / "validation_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, "epoch": epoch, "score": value}) + "\n")
        return value

    if resume:
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        if payload.get("identity_digest") != identity_digest or payload.get("identity") != identity:
            raise ValueError("Resume rejected: protocol, split, normalization, architecture or seed differs")
        if payload.get("checkpoint_kind") != "part3_complete_epoch":
            raise ValueError("Resume requires this run's complete optimizer/selection checkpoint")
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        progress = payload["progress"]
        best_state = payload["best_state_dict"]
        _restore_rng(payload["rng"], loaders)
        if progress["stop_reason"] in {"patience_cycles_exhausted", "maximum_epochs"}:
            if (output_dir / "training_summary.json").is_file():
                return json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
        else:
            progress["stop_reason"] = None
    else:
        if last_path.exists() or best_path.exists():
            raise FileExistsError("Existing run state requires explicit resume; it will not be overwritten")
        _seed(int(identity["seed"]))
        progress["best_score"] = evaluate("initial_selection", 0)
        best_state = _cpu_state(model)
        progress["selection_events"].append({"epoch": 0, "score": progress["best_score"]})
        _write_json(output_dir / "training_identity.json", identity)
        _save(output_dir / "initial.pt", {"identity": identity, "state_dict": best_state,
                                          "checkpoint_kind": "new_initialization"})

    def checkpoint():
        _save(last_path, {"checkpoint_kind": "part3_complete_epoch", "identity": identity,
                         "identity_digest": identity_digest, "state_dict": _cpu_state(model),
                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "best_state_dict": best_state, "progress": copy.deepcopy(progress),
                         "rng": _rng_state(loaders)})
        _save(best_path, {"checkpoint_kind": "part3_validation_selected", "identity": identity,
                         "identity_digest": identity_digest, "state_dict": best_state,
                         "best_epoch": progress["best_epoch"], "best_score": progress["best_score"]})
        _write_json(output_dir / "training_curve.json", progress["curve"])
        _write_json(output_dir / "selection_events.json", progress["selection_events"])
        _write_json(output_dir / "status.json", {
            "status": "running", "pid": os.getpid(), "run_id": identity.get("run_id"),
            "last_completed_epoch": progress["epoch"], "best_epoch": progress["best_epoch"],
            "best_validation_r": progress["best_score"], "unix_time": time.time(),
            "resume_checkpoint": "last.pt"})

    checkpoint()
    start = time.monotonic()
    previous_elapsed = float(progress["elapsed_seconds"])
    accumulation = int(training["gradient_accumulation_sessions"])
    max_epochs, patience = int(training["max_epochs"]), int(training["patience"])
    max_cycles = int(training["lr_decay_steps"])
    try:
        while progress["epoch"] < max_epochs and progress["cycle"] < max_cycles:
            epoch = progress["epoch"] + 1
            epoch_start = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_losses = []
            for batch_no, (session, batch) in enumerate(_long_cycle(loaders["train"])):
                loss = objective(model, loaders["train"], session, batch, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite objective: epoch {epoch}, session {session}")
                loss.backward()
                epoch_losses.append(float(loss.detach().cpu()))
                if (batch_no + 1) % accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            if len(epoch_losses) % accumulation:
                raise ValueError("Session cycling did not complete an accumulation block")
            # The retained trainer evaluates once for logging, then once for
            # early stopping. Keep both calls and their random snippet draws.
            logged_score = evaluate("epoch_log", epoch)
            selection_score = evaluate("selection", epoch)
            scheduler.step(selection_score)
            selected = selection_score > progress["best_score"] + float(training["tolerance"])
            if selected:
                best_state = _cpu_state(model)
                progress.update(best_epoch=epoch, best_score=selection_score, patience_counter=0)
                progress["selection_events"].append({"epoch": epoch, "score": selection_score})
            else:
                progress["patience_counter"] += 1
            if progress["patience_counter"] >= patience:
                progress["cycle"] += 1
                # The official early_stopping restores the best model at each
                # exhausted patience cycle while retaining AdamW state.
                if epoch < max_epochs and max_cycles > 1:
                    evaluate("before_cycle_restore", epoch)
                    model.load_state_dict(best_state)
                    evaluate("after_cycle_restore", epoch)
                progress["patience_counter"] = 0
            progress["epoch"] = epoch
            progress["elapsed_seconds"] = previous_elapsed + time.monotonic() - start
            row = {"epoch": epoch, "train_objective_mean": float(np.mean(epoch_losses)),
                   "validation_logged_r": logged_score, "validation_selection_r": selection_score,
                   "selected": selected, "learning_rate": optimizer.param_groups[0]["lr"],
                   "patience_counter": progress["patience_counter"], "cycle": progress["cycle"],
                   "epoch_seconds": time.monotonic() - epoch_start}
            progress["curve"].append(row)
            if epoch >= max_epochs:
                progress["stop_reason"] = "maximum_epochs"
            elif progress["cycle"] >= max_cycles:
                progress["stop_reason"] = "patience_cycles_exhausted"
            elif stop_after_epoch is not None and epoch >= int(stop_after_epoch):
                progress["stop_reason"] = "bounded_invocation_resumable"
            checkpoint()
            print(json.dumps({"run": identity.get("run_id"), **row}), flush=True)
            if progress["stop_reason"]:
                break
    except BaseException as error:
        interruption = {
            "status": "interrupted", "type": type(error).__name__, "message": str(error),
            "last_completed_epoch": progress["epoch"], "resume_checkpoint": str(last_path)}
        _write_json(output_dir / "interruption.json", interruption)
        _write_json(output_dir / "status.json", interruption)
        raise
    complete = progress["stop_reason"] in {"maximum_epochs", "patience_cycles_exhausted"}
    if complete:
        model.load_state_dict(best_state)
        model.eval()
    summary = {"status": "training_complete" if complete else "incomplete_resumable",
               "run_id": identity.get("run_id"), "identity_digest": identity_digest,
               "best_epoch": progress["best_epoch"], "validation_score": progress["best_score"],
               "last_epoch": progress["epoch"], "stop_reason": progress["stop_reason"],
               "elapsed_seconds": progress["elapsed_seconds"],
               "selection_events": len(progress["selection_events"]),
               "parameter_counts": identity.get("parameter_counts"),
               "best_checkpoint": str(best_path), "last_checkpoint": str(last_path),
               "best_sha256": _file_digest(best_path), "last_sha256": _file_digest(last_path),
               "best_state_sha256": state_digest(best_state),
               "test_read_by_trainer": False}
    _write_json(output_dir / "training_summary.json", summary)
    _write_json(output_dir / "status.json", {
        "status": summary["status"], "run_id": identity.get("run_id"),
        "last_completed_epoch": progress["epoch"], "stop_reason": progress["stop_reason"],
        "unix_time": time.time()})
    if complete and identity.get("scope") == "formal_training":
        _write_json(output_dir / "frozen_identity.json", {
            "status": "complete", "initialization": "from_scratch", "identity": identity,
            "identity_digest": identity_digest, "checkpoint": "best.pt",
            "checkpoint_sha256": summary["best_sha256"],
            "sha256": summary["best_sha256"],
            "split_sha256": identity["split_sha256"],
            "preprocessing_sha256": identity["preprocessing_sha256"],
            "config_sha256": identity["protocol_sha256"],
            "state_sha256": summary["best_state_sha256"],
            "best_epoch": progress["best_epoch"], "selection_role": "validation",
            "test_used_for_selection": False})
    return summary


def train_encoder(kind, config, manifest, preprocessing, data_root, output_dir, *,
                  resume=False, smoke=False, device="cuda", stop_after_epoch=None,
                  artifact_root=None):
    """Build the audited Part 3 datasets, then train one fresh frozen-scope model."""
    from .retrained_data import create_dataloaders

    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if isinstance(preprocessing, (str, Path)):
        artifact_root = Path(preprocessing).parent
        preprocessing = json.loads(Path(preprocessing).read_text(encoding="utf-8"))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The locked full encoder training requires CUDA")
    seed = int(config.get("project", {}).get("seed", config.get("seed", 42)))
    if seed != 42:
        raise ValueError("This protocol fixes the single training seed at 42")
    _seed(seed)
    loaders = create_dataloaders(data_root, manifest, preprocessing, artifact_root=artifact_root,
                               batch_size=8, frames=80)
    role_audit = assert_training_roles(loaders, manifest)
    model, architecture, counts = build_encoder(kind, loaders["train"], seed=seed)
    expected_source = config.get("models", {}).get(kind, {})
    if expected_source and (expected_source.get("configuration") != architecture["_architecture_source"]
                            or expected_source.get("configuration_sha256") != architecture["_architecture_sha256"]):
        raise ValueError("The protocol architecture source differs from its immutable configuration")
    training = copy.deepcopy(architecture["training"])
    if smoke:
        loaders = {role: {key: LimitedLoader(value) for key, value in values.items()}
                   for role, values in loaders.items()}
        training.update(max_epochs=1, lr_decay_steps=1)
    identity = {"run_id": f"retrained_v1_{kind}_seed{seed}" + ("_smoke" if smoke else ""),
                "protocol_sha256": _digest(config), "split_sha256": manifest["split_sha256"],
                "preprocessing_sha256": preprocessing["preprocessing_sha256"], "seed": seed,
                "architecture": kind, "architecture_sha256": architecture["_architecture_sha256"],
                "training_code_sha256": _file_digest(Path(__file__)),
                "training": training, "parameter_counts": counts,
                "initialization": "fresh_parameters_and_batchnorm_train_batches_only",
                "loss_scale_dataset_lengths": {key: loader.dataset.original_dataset_length
                                                for key, loader in loaders["train"].items()},
                "scope": "implementation_smoke" if smoke else "formal_training"}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "last.pt").is_file():
        if not resume:
            raise FileExistsError("Existing run state requires explicit resume")
        saved = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("Resume rejected before writing: run identity differs")
    _write_json(output_dir / "loader_role_audit.json", role_audit)
    _write_json(output_dir / "training_config.json", {"protocol": config, "architecture": architecture,
                                                    "training": training, "identity": identity})
    (output_dir / "model_architecture.txt").write_text(str(model) + "\n", encoding="utf-8")
    import inspect
    import platform
    from neuralpredictors.training import early_stopping
    from sensorium.training.video_training_loop import standard_trainer
    _write_json(output_dir / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "official_source_hashes": {
            name: _file_digest(Path(inspect.getsourcefile(function)))
            for name, function in (("standard_trainer", standard_trainer), ("early_stopping", early_stopping))},
        "training_adapter_sha256": identity["training_code_sha256"],
        "loss_scale_note": "Historical whole-source dataset size retained as a metadata-only scalar",
        "validation_note": "Two random snippet evaluations per epoch: log first, then selection",
        "omitted_log": "The historical trainer's mislabeled validation loss reused its last training batch; it is not reported here"})
    result = run_training(model, loaders, training, output_dir, identity, device=device,
                          resume=resume, stop_after_epoch=stop_after_epoch)
    if smoke:
        result["status"] = "smoke_complete_not_a_scientific_training_result"
        _write_json(output_dir / "training_summary.json", result)
    return result


def load_frozen_encoder(kind, train_loaders, run_dir, *, config, manifest, preprocessing,
                        device="cuda"):
    """Load a completed new run, rejecting historical or incomplete checkpoints."""
    run_dir = Path(run_dir)
    frozen = json.loads((run_dir / "frozen_identity.json").read_text(encoding="utf-8"))
    identity = frozen["identity"]
    expected = {"protocol_sha256": _digest(config), "split_sha256": manifest["split_sha256"],
                "preprocessing_sha256": preprocessing["preprocessing_sha256"], "architecture": kind,
                "seed": 42, "scope": "formal_training"}
    if frozen.get("status") != "complete":
        raise ValueError("Encoder training has not completed")
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("The frozen encoder belongs to a different Part 3 run")
    checkpoint = run_dir / frozen["checkpoint"]
    if _file_digest(checkpoint) != frozen["checkpoint_sha256"]:
        raise ValueError("The frozen encoder checkpoint changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (payload.get("checkpoint_kind") != "part3_validation_selected"
            or payload.get("identity") != identity
            or payload.get("identity_digest") != _digest(identity)):
        raise ValueError("A historical or incompatible encoder checkpoint was supplied")
    model, architecture, counts = build_encoder(kind, train_loaders)
    if identity["architecture_sha256"] != architecture["_architecture_sha256"]:
        raise ValueError("Encoder architecture configuration changed")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if state_digest(model.state_dict()) != frozen["state_sha256"]:
        raise ValueError("Loaded encoder state does not match the frozen identity")
    return model, frozen
