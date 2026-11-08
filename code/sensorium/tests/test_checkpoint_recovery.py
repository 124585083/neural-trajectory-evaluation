"""Replay completion must distinguish native stopping from interrupted cohorts."""
import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trajectory_model_eval.checkpoint_recovery import (COHORT_RULE, declare_recovery,
                                                      validate_recovery_completion)
from trajectory_model_eval.retrained_training import _digest, _file_digest, _write_json, run_training
from test_retrained_training import (identity, model, settings, setup_loaders,
                                     tiny_objective, tiny_score)


def declaration():
    return {"schema_version": "sensorium_checkpoint_recovery_v2", "cohort_rule": COHORT_RULE,
            "candidate_epochs": list(range(1, 120)), "maximum_candidate_epoch": 119,
            "native_early_stop_reason": "patience_cycles_exhausted", "test_loaded": False,
            "candidate_validation_scores_read": False, "checkpoint_selection_started": False}


def test_fresh_declaration_is_hashed_and_resume_preserves_exact_bytes(tmp_path):
    path = tmp_path / "recovery_protocol.json"
    first = declare_recovery(path, declaration(), resume=False)
    original_bytes = path.read_bytes()
    assert first["created_at_utc"]
    assert first["declaration_sha256"] == _digest({key: value for key, value in first.items()
                                                if key != "declaration_sha256"})
    assert declare_recovery(path, declaration(), resume=True) == first
    assert path.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="explicit resume"):
        declare_recovery(path, declaration(), resume=False)


@pytest.mark.parametrize("field,value", [
    ("created_at_utc", ""), ("declaration_sha256", "changed"),
    ("checkpoint_selection_started", True),
])
def test_resume_rejects_tampered_pretraining_declaration(tmp_path, field, value):
    path = tmp_path / "recovery_protocol.json"
    saved = declare_recovery(path, declaration(), resume=False)
    saved[field] = value
    if field != "declaration_sha256":
        saved["declaration_sha256"] = _digest({key: entry for key, entry in saved.items()
                                             if key != "declaration_sha256"})
    _write_json(path, saved)
    with pytest.raises(ValueError, match="same declaration"):
        declare_recovery(path, declaration(), resume=True)


def test_resume_requires_preexisting_declaration(tmp_path):
    with pytest.raises(ValueError, match="without the original recovery declaration"):
        declare_recovery(tmp_path / "recovery_protocol.json", declaration(), resume=True)


def completed_run(path, *, native=True):
    training = settings(max_epochs=30, patience=1 if native else 50, cycles=2)
    run_identity = {**identity(), "training": training}
    candidates = []

    def observe(epoch, state, metrics, current_identity):
        if epoch:
            candidates.append({"epoch": epoch})

    cap = 8 if native else 4
    summary = run_training(model(), setup_loaders(), training, path, run_identity,
                           objective=tiny_objective,
                           score=(lambda *args: 0.) if native else tiny_score,
                           device="cpu", stop_after_epoch=cap, state_observer=observe)
    return candidates, summary, run_identity, list(range(1, cap + 1))


@pytest.mark.parametrize("native,expected_epoch,kind", [
    (True, 2, "native_early_stop"), (False, 4, "epoch_cap_reached"),
])
def test_complete_native_stop_and_declared_cap(tmp_path, native, expected_epoch, kind):
    candidates, summary, run_identity, declared = completed_run(tmp_path, native=native)
    result = validate_recovery_completion(tmp_path, candidates, summary, run_identity, declared)
    assert result == {"completion_kind": kind, "final_epoch": expected_epoch,
                      "effective_candidate_epochs": list(range(1, expected_epoch + 1))}


def test_partial_bounded_run_is_not_native_completion(tmp_path):
    candidates, summary, run_identity, declared = completed_run(tmp_path, native=False)
    with pytest.raises(ValueError, match="native early stopping or the declared epoch cap"):
        validate_recovery_completion(tmp_path, candidates, summary, run_identity, list(range(1, 9)))


@pytest.mark.parametrize("bad_epochs", [[1], [1, 1, 2], [2, 1]])
def test_missing_duplicate_or_reordered_candidates_rejected(tmp_path, bad_epochs):
    _, summary, run_identity, declared = completed_run(tmp_path)
    with pytest.raises(ValueError, match="missing completed epochs or contains duplicates"):
        validate_recovery_completion(tmp_path, [{"epoch": e} for e in bad_epochs],
                                     summary, run_identity, declared)


def test_native_summary_requires_checkpoint_policy_evidence(tmp_path):
    candidates, summary, run_identity, declared = completed_run(tmp_path)
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    last["progress"]["cycle"] -= 1
    torch.save(last, tmp_path / "last.pt")
    summary["last_sha256"] = _file_digest(tmp_path / "last.pt")
    with pytest.raises(ValueError, match="native training policy"):
        validate_recovery_completion(tmp_path, candidates, summary, run_identity, declared)


def test_completion_rejects_changed_checkpoint_and_test_access(tmp_path):
    candidates, summary, run_identity, declared = completed_run(tmp_path)
    changed = {**summary, "test_read_by_trainer": True}
    with pytest.raises(ValueError, match="training identity"):
        validate_recovery_completion(tmp_path, candidates, changed, run_identity, declared)
    with (tmp_path / "last.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint differs"):
        validate_recovery_completion(tmp_path, candidates, summary, run_identity, declared)


def test_observer_cannot_change_training_rng_or_model_state(tmp_path):
    arguments = dict(objective=tiny_objective, score=tiny_score, device="cpu")
    baseline = run_training(model(), setup_loaders(), settings(), tmp_path / "baseline",
                            identity(), **arguments)
    captured = []

    def consume_rng_and_mutate_snapshot(epoch, state, metrics, current_identity):
        captured.append(epoch)
        random.random()
        np.random.random(13)
        torch.rand(17)
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()

    observed = run_training(model(), setup_loaders(), settings(), tmp_path / "observed",
                            identity(), state_observer=consume_rng_and_mutate_snapshot, **arguments)
    assert captured == [0, 1, 2, 3, 4]
    assert baseline["best_state_sha256"] == observed["best_state_sha256"]
    first = torch.load(tmp_path / "baseline/last.pt", weights_only=False)
    second = torch.load(tmp_path / "observed/last.pt", weights_only=False)
    for key in first["state_dict"]:
        assert torch.equal(first["state_dict"][key], second["state_dict"][key])
    curve_a, curve_b = copy.deepcopy(first["progress"]["curve"]), copy.deepcopy(second["progress"]["curve"])
    for row in curve_a + curve_b:
        row.pop("epoch_seconds")
    assert curve_a == curve_b
