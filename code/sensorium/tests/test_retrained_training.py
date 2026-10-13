"""Training dependency, role, checkpoint, and official decision-rule checks."""
from collections import namedtuple
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trajectory_model_eval.retrained_training import (
    _cpu_state, _seed, assert_training_roles, run_training, state_digest,
)


Batch = namedtuple("Batch", "x y")


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, role, indices, sign=1.):
        self.role, self.dataset_indices = role, list(indices)
        self.x = torch.arange(1, len(indices) + 1, dtype=torch.float32)[:, None]
        self.y = sign * self.x

    def __len__(self):
        return len(self.dataset_indices)

    def __getitem__(self, index):
        return Batch(self.x[index], self.y[index])


def setup_loaders(validation_sign=1.):
    return {"train": {"s": torch.utils.data.DataLoader(TinyDataset("train", [0, 1, 2, 3]),
                                                       batch_size=2, shuffle=True)},
            "oracle": {"s": torch.utils.data.DataLoader(TinyDataset("validation", [4, 5], validation_sign),
                                                        batch_size=2)}}


def settings(max_epochs=4, patience=5, cycles=4):
    return {"learning_rate": .05, "optimizer_defaults": {}, "lr_decay_factor": .3,
            "patience": patience, "tolerance": 1e-6, "min_lr": 1e-4,
            "interval": 1, "restore_best": True, "gradient_accumulation_sessions": 1,
            "max_epochs": max_epochs, "lr_decay_steps": cycles}


def tiny_objective(model, loaders, session, batch, device):
    return ((model(batch.x.to(device)) - batch.y.to(device)) ** 2).mean()


def tiny_score(model, loaders, device):
    batch = next(iter(loaders["s"]))
    return -float(((model(batch.x.to(device)) - batch.y.to(device)) ** 2).mean())


def identity():
    return {"seed": 42, "run_id": "dependency_test", "scope": "synthetic",
            "protocol_sha256": "p", "split_sha256": "s", "preprocessing_sha256": "n"}


def model():
    _seed(42)
    result = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        result.weight.fill_(0.)
    return result


def run(path, *, validation_sign=1., resume=False, stop=None):
    return run_training(model(), setup_loaders(validation_sign), settings(), path, identity(),
                        objective=tiny_objective, score=tiny_score, device="cpu",
                        resume=resume, stop_after_epoch=stop)


def test_only_actual_manifest_roles_enter_trainer():
    loaders = setup_loaders()
    manifest = {"trials": [{"session": "s", "role": role, "dataset_index": i}
                            for role, indices in (("train", range(4)), ("validation", (4, 5)))
                            for i in indices]}
    assert assert_training_roles(loaders, manifest)["validation"]["s"] == [4, 5]
    loaders["oracle"]["s"].dataset.role = "test"
    with pytest.raises(ValueError, match="parent trial IDs"):
        assert_training_roles(loaders, manifest)


def test_test_loader_rejected_before_optimization(tmp_path):
    loaders = setup_loaders()
    loaders["test"] = loaders["oracle"]
    with pytest.raises(ValueError, match="Test loaders"):
        run_training(model(), loaders, settings(), tmp_path, identity(), device="cpu")


def test_test_responses_cannot_affect_training_or_selection(tmp_path):
    # Test labels are evaluated only after both identical training runs finish.
    first, second = run(tmp_path / "a"), run(tmp_path / "b")
    assert first["best_state_sha256"] == second["best_state_sha256"]
    assert first["best_epoch"] == second["best_epoch"]
    assert first["validation_score"] == second["validation_score"]
    frozen = model()
    state = torch.load(tmp_path / "a/best.pt", weights_only=False)["state_dict"]
    frozen.load_state_dict(state)
    with torch.no_grad():
        prediction = frozen(torch.tensor([[1.], [2.]]))
    response_a = torch.tensor([[1.], [2.]])
    response_b = torch.tensor([[7.], [14.]])
    assert ((prediction - response_a) ** 2).mean() != ((prediction - response_b) ** 2).mean()


def test_validation_can_change_checkpoint_selection(tmp_path):
    positive, negative = run(tmp_path / "positive"), run(tmp_path / "negative", validation_sign=-1.)
    assert positive["best_epoch"] > 0
    assert negative["best_epoch"] == 0
    assert positive["best_state_sha256"] != negative["best_state_sha256"]
    assert not positive["test_read_by_trainer"]


def test_resume_matches_uninterrupted_rng_optimizer_and_selection(tmp_path):
    uninterrupted = run(tmp_path / "full")
    interrupted = run(tmp_path / "resume", stop=2)
    assert interrupted["status"] == "incomplete_resumable"
    resumed = run(tmp_path / "resume", resume=True)
    assert resumed["best_state_sha256"] == uninterrupted["best_state_sha256"]
    assert resumed["best_epoch"] == uninterrupted["best_epoch"]
    assert resumed["validation_score"] == uninterrupted["validation_score"]
    for folder in ("full", "resume"):
        state = torch.load(tmp_path / folder / "last.pt", weights_only=False)
        assert set(state) >= {"optimizer", "scheduler", "rng", "progress", "best_state_dict"}


def test_resume_rejects_incompatible_identity(tmp_path):
    run(tmp_path, stop=1)
    changed = identity()
    changed["preprocessing_sha256"] = "changed"
    with pytest.raises(ValueError, match="Resume rejected"):
        run_training(model(), setup_loaders(), settings(), tmp_path, changed, device="cpu",
                     objective=tiny_objective, score=tiny_score, resume=True)


def test_existing_run_is_not_silently_overwritten(tmp_path):
    run(tmp_path)
    with pytest.raises(FileExistsError, match="explicit resume"):
        run(tmp_path)


def test_patience_and_restore_match_official_generator(tmp_path):
    """Compare optimizer/scheduler/state decisions with the pinned generator."""
    training_module = pytest.importorskip("neuralpredictors.training")
    from trajectory_model_eval.retrained_training import _long_cycle

    configuration = settings(max_epochs=20, patience=2, cycles=2)
    # A first improvement followed by plateaus exercises both patience cycles.
    numbers = [0., .2, .3] + [.25] * 100
    custom_numbers, reference_numbers = iter(numbers), iter(numbers)
    custom = model()
    custom_result = run_training(custom, setup_loaders(), configuration, tmp_path, identity(),
                                 device="cpu", objective=tiny_objective,
                                 score=lambda *args: next(custom_numbers))
    reference = model()
    reference_loaders = setup_loaders()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=.3, patience=2, threshold=1e-6,
        min_lr=1e-4, threshold_mode="abs")
    _seed(42)
    reference.train()
    epochs = []
    for epoch, _ in training_module.early_stopping(
            reference, lambda model: next(reference_numbers), interval=1, patience=2,
            max_iter=20, maximize=True, tolerance=1e-6, restore_best=True,
            scheduler=scheduler, lr_decay_steps=2):
        optimizer.zero_grad(set_to_none=True)
        for key, batch in _long_cycle(reference_loaders["train"]):
            loss = tiny_objective(reference, reference_loaders["train"], key, batch, "cpu")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        next(reference_numbers)  # The official per-epoch logging evaluation.
        epochs.append(epoch)
    assert custom_result["last_epoch"] == epochs[-1]
    assert custom_result["best_state_sha256"] == state_digest(_cpu_state(reference))
    saved = torch.load(tmp_path / "last.pt", weights_only=False)
    assert saved["scheduler"] == scheduler.state_dict()
