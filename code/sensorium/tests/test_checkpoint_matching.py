"""Checkpoint controls select on validation and retain original artifact identities."""
from copy import deepcopy
import json

import numpy as np
import pytest

from trajectory_model_eval.checkpoint_matching import (
    freeze_selection, load_selection, save_selection, score_candidate, validate_selection,
    validation_reference,
)
from trajectory_model_eval.retrained_predictions import (
    attach_matched_prediction, combine_pilot, load_candidate_encoder, load_matched_bundle, save_matched_bundle, select_neurons,
)
from trajectory_model_eval.unified_contract import canonical_digest, file_sha256


def fixture(tmp_path):
    rng = np.random.default_rng(63)
    neural = rng.normal(size=(6, 11, 5))
    static = neural + rng.normal(scale=.5, size=neural.shape)
    roles = np.array(["train"] * 3 + ["validation"] * 3)
    bundle = {"session": np.array("pilot"), "roles": roles, "conditions": np.arange(6).astype(str),
        "movie_ids": np.arange(6).astype(str), "dataset_indices": np.arange(6), "trial_ids": np.arange(6).astype(str),
        "neuron_ids": np.array(list("abcde")), "neuron_indices": np.array([9, 7, 5, 3, 1]),
        "frame_indices": np.arange(50, 61), "encoder_precision": np.ones((11, 5)),
        "neural": neural, "static": static, "dynamic": neural + .1}
    original = {"run_id": "original", "scope": "formal_training", "seed": 42,
                "architecture": "dynamic", "split_sha256": "s" * 64, "training_code_sha256": "a" * 64}
    original_digest = canonical_digest(original)
    replay = {**original, "run_id": "replay", "scope": "checkpoint_candidate_replay",
              "source_training_identity_digest": original_digest, "training_code_sha256": "b" * 64}
    source = {"source_config_sha256": "c" * 64, "source_freeze_sha256": "f" * 64,
              "split_sha256": "s" * 64, "preprocessing_sha256": "p" * 64,
              "gpfa_identity": {"gpfa_sha256": "g" * 64},
              "checkpoint_identities": {"static": {"identity_digest": "a" * 64},
                  "dynamic": {"identity_digest": original_digest, "identity": original}}}
    identities = []
    for epoch in (1, 2):
        path = tmp_path / "recovery" / f"epoch_{epoch:03d}.pt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"synthetic candidate {epoch}".encode())
        identities.append({"epoch": epoch, "checkpoint_path": path.relative_to(tmp_path).as_posix(),
            "sha256": file_sha256(path), "state_sha256": str(epoch) * 64,
            "training_identity_digest": canonical_digest(replay), "identity": deepcopy(replay)})
    return bundle, identities, source


def candidate(bundle, prediction):
    return {**{key: value for key, value in bundle.items() if key not in {"static", "dynamic"}},
            "matched_dynamic": prediction}


def selection(tmp_path):
    bundle, identities, source = fixture(tmp_path)
    reference = validation_reference(bundle)
    rows = [score_candidate(reference, candidate(bundle, prediction), identity)
            for prediction, identity in zip((bundle["static"], bundle["dynamic"]), identities)]
    record = freeze_selection(reference, rows, source, expected_epochs=[1, 2])
    return bundle, identities, source, reference, rows, record


def test_validation_only_choice_uses_intact_checkpoint_and_ties_use_epoch(tmp_path):
    bundle, ids, source, reference, rows, record = selection(tmp_path)
    assert record["selected_checkpoint"] == ids[0]
    assert record["validation_absolute_gap"] == 0
    assert record["validation_matching_achieved"]
    assert record["source_identity"] == source
    assert record["candidate_training_identity_digest"] != source["checkpoint_identities"]["dynamic"]["identity_digest"]
    reversed_tie = [score_candidate(reference, candidate(bundle, bundle["static"]), identity) for identity in ids[::-1]]
    tied = freeze_selection(reference, reversed_tie, source, expected_epochs=[1, 2])
    assert tied["selected_checkpoint"]["epoch"] == 1
    assert [row["checkpoint"]["epoch"] for row in tied["candidates"]] == [1, 2]


def test_test_input_cannot_enter_reference_or_candidate_selection(tmp_path):
    bundle, ids, _, reference, _, _ = selection(tmp_path)
    changed = deepcopy(bundle)
    changed["roles"][0] = "test"
    with pytest.raises(ValueError, match="test is forbidden"):
        validation_reference(changed)
    with pytest.raises(ValueError, match="test is forbidden"):
        score_candidate(reference, candidate(changed, changed["dynamic"]), ids[0])
    # Training predictions/targets never enter the validation reference.
    changed = deepcopy(bundle)
    changed["static"][:3] *= 1000
    changed["neural"][:3] *= -300
    assert validation_reference(changed)["reference_sha256"] == reference["reference_sha256"]


@pytest.mark.parametrize("key", ["trial_ids", "neuron_ids", "frame_indices", "encoder_precision", "neural"])
def test_candidate_axis_target_and_scale_mismatch_is_rejected(tmp_path, key):
    bundle, ids, _, reference, _, _ = selection(tmp_path)
    changed = candidate(bundle, bundle["static"])
    changed = deepcopy(changed)
    if changed[key].dtype.kind in "US":
        changed[key][-1] = "x"
    else:
        changed[key].flat[-1] += 1
    with pytest.raises(ValueError, match=key):
        score_candidate(reference, changed, ids[0])


def test_common_neuron_cohort_is_fixed_across_every_candidate(tmp_path):
    bundle, ids, source = fixture(tmp_path)
    reference = validation_reference(bundle)
    first, second = bundle["static"].copy(), bundle["dynamic"].copy()
    first[..., 0] = 3
    second[..., 1] = 4
    rows = [score_candidate(reference, candidate(bundle, values), identity)
            for values, identity in zip((first, second), ids)]
    result = freeze_selection(reference, rows, source, expected_epochs=[1, 2])
    assert result["shared_neuron_positions"] == [2, 3, 4]
    assert all(row["n_shared_neurons"] == 3 for row in result["candidates"])
    assert result["total_gpfa_neurons_unchanged"] == 5
    with pytest.raises(ValueError, match="incomplete"):
        freeze_selection(reference, rows[:1], source, expected_epochs=[1, 2])


def test_replay_cannot_change_source_training_settings_or_mix_runs(tmp_path):
    _, _, source, reference, rows, _ = selection(tmp_path)
    changed = deepcopy(rows)
    identity = changed[0]["checkpoint"]
    identity["identity"]["seed"] = 90
    identity["training_identity_digest"] = canonical_digest(identity["identity"])
    with pytest.raises(ValueError, match="source training setting seed"):
        freeze_selection(reference, changed, source, expected_epochs=[1, 2])
    changed = deepcopy(rows)
    identity = changed[0]["checkpoint"]
    identity["identity"]["run_id"] = "another allowed replay"
    identity["training_identity_digest"] = canonical_digest(identity["identity"])
    with pytest.raises(ValueError, match="one declared"):
        freeze_selection(reference, changed, source, expected_epochs=[1, 2])


def test_saved_selection_verifies_csv_source_and_checkpoint_and_is_portable(tmp_path):
    _, ids, source, _, _, record = selection(tmp_path)
    save_selection(tmp_path, record)
    assert load_selection(tmp_path, source, verify_checkpoint=True) == record
    with pytest.raises(FileExistsError):
        save_selection(tmp_path, record)
    changed_source = deepcopy(source)
    changed_source["source_freeze_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="different source"):
        load_selection(tmp_path, changed_source)
    (tmp_path / ids[0]["checkpoint_path"]).write_bytes(b"modified checkpoint")
    with pytest.raises(ValueError, match="payload changed"):
        load_selection(tmp_path, verify_checkpoint=True)
    (tmp_path / "checkpoint_candidates.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="score table changed"):
        load_selection(tmp_path)


def test_rehashed_wrong_selection_is_rejected(tmp_path):
    _, ids, _, _, _, record = selection(tmp_path)
    record["selected_checkpoint"] = ids[1]
    record["selection_sha256"] = canonical_digest({key: value for key, value in record.items() if key != "selection_sha256"})
    with pytest.raises(ValueError, match="decision"):
        validate_selection(record)


def test_sidecar_preserves_predictions_and_propagates_trial_neuron_axes(tmp_path):
    bundle, _, _, _, _, record = selection(tmp_path)
    matched = candidate(bundle, bundle["static"].copy())
    path = tmp_path / "predictions.npz"
    save_matched_bundle(path, matched, record)
    loaded = load_matched_bundle(path, record, reference_bundle=bundle)
    combined = attach_matched_prediction(bundle, loaded)
    np.testing.assert_array_equal(combined["dynamic"], bundle["dynamic"])
    np.testing.assert_array_equal(combined["matched_dynamic"], bundle["static"])
    selected = select_neurons(combined, [1, 9])
    np.testing.assert_array_equal(selected["matched_dynamic"], bundle["static"][..., [4, 0]])
    test = deepcopy(combined)
    test["roles"] = np.array(["test"] * 6)
    test["dataset_indices"] += 6
    test["trial_ids"] = test["dataset_indices"].astype(str)
    manifest = {"session": "pilot", "trials": [{"session": "pilot", "dataset_index": i, "role": "test" if i >= 6 else "validation"}
                                                 for i in range(12)]}
    joined = combine_pilot(combined, test, manifest)
    np.testing.assert_array_equal(joined["matched_dynamic"], np.concatenate([bundle["static"], bundle["static"]]))
    changed = deepcopy(loaded)
    changed["dataset_indices"] = changed["dataset_indices"][::-1]
    with pytest.raises(ValueError, match="dataset_indices"):
        attach_matched_prediction(bundle, changed)


def test_candidate_loader_restores_exact_replay_state_and_rejects_changed_payload(tmp_path, monkeypatch):
    import torch
    from trajectory_model_eval import retrained_training
    model = torch.nn.Linear(3, 2)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    config, manifest, prep = {}, {"split_sha256": "s" * 64}, {"preprocessing_sha256": "p" * 64}
    training = {"protocol_sha256": canonical_digest(config), "split_sha256": manifest["split_sha256"],
        "preprocessing_sha256": prep["preprocessing_sha256"], "architecture": "dynamic", "seed": 42,
        "architecture_sha256": "a" * 64, "scope": "checkpoint_candidate_replay",
        "source_training_identity_digest": "b" * 64}
    path = tmp_path / "candidate.pt"
    payload = {"checkpoint_kind": "part3_epoch_candidate", "epoch": 1, "identity": training,
               "identity_digest": canonical_digest(training), "state_dict": state}
    torch.save(payload, path)
    identity = {"epoch": 1, "checkpoint_path": "candidate.pt", "sha256": file_sha256(path),
        "state_sha256": retrained_training.state_digest(state), "training_identity_digest": canonical_digest(training),
        "identity": training}
    monkeypatch.setattr(retrained_training, "load_architecture_config", lambda kind: {"_architecture_sha256": "a" * 64})
    monkeypatch.setattr(retrained_training, "build_encoder", lambda kind, loaders: (torch.nn.Linear(3, 2), {}, {}))
    loaded = load_candidate_encoder(identity, {}, config=config, manifest=manifest, preprocessing=prep,
                                     device="cpu", checkpoint_root=tmp_path)
    assert not loaded.training and all(not value.requires_grad for value in loaded.parameters())
    assert retrained_training.state_digest(loaded.state_dict()) == identity["state_sha256"]
    with torch.no_grad():
        payload["state_dict"]["weight"].add_(1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="payload changed"):
        load_candidate_encoder(identity, {}, config=config, manifest=manifest, preprocessing=prep,
                                 device="cpu", checkpoint_root=tmp_path)
    identity["sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="state differs"):
        load_candidate_encoder(identity, {}, config=config, manifest=manifest, preprocessing=prep,
                                 device="cpu", checkpoint_root=tmp_path)


def native_fixture(tmp_path, *, epoch=25, native=True):
    from types import SimpleNamespace
    import torch
    from trajectory_model_eval.checkpoint_matching import COHORT_RULE
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    training = {"scope": "checkpoint_candidate_replay", "source_training_identity_digest": "a" * 64,
                "training": {"max_epochs": 200, "patience": 5, "lr_decay_steps": 4}}
    digest = canonical_digest(training)
    reason = "patience_cycles_exhausted" if native else "bounded_invocation_resumable"
    progress = {"epoch": epoch, "cycle": 4 if native else 3, "stop_reason": reason,
                "curve": [{"epoch": value} for value in range(1, epoch + 1)]}
    torch.save({"checkpoint_kind": "part3_complete_epoch", "identity": training, "identity_digest": digest,
                "progress": progress}, recovery / "last.pt")
    summary = {"status": "training_complete" if native else "incomplete_resumable", "stop_reason": reason,
               "last_epoch": epoch, "identity_digest": digest, "test_read_by_trainer": False,
               "last_sha256": file_sha256(recovery / "last.pt")}
    (recovery / "training_summary.json").write_text(json.dumps(summary))
    inventory = {"schema_version": "sensorium_epoch_candidates_v1", "declared_epochs": list(range(1, 120)),
        "training_identity": training, "test_loaded": False, "complete": epoch == 119,
        "candidates": [{"epoch": value} for value in range(1, epoch + 1)]}
    (recovery / "candidate_manifest.json").write_text(json.dumps(inventory))
    revision = {"candidate_cohort_rule": COHORT_RULE, "test_loaded": False,
                "candidate_validation_scores_read": False, "checkpoint_selection_started": False}
    revision["revision_sha256"] = canonical_digest(revision)
    (tmp_path / "protocol_revision.json").write_text(json.dumps(revision))
    method = {"candidate_cohort_rule": COHORT_RULE, "candidate_epochs": list(range(1, 120))}
    args = SimpleNamespace(recovery_root=recovery, matching_root=tmp_path, watch=False)
    source = {"checkpoint_identities": {"dynamic": {"identity_digest": "a" * 64}}}
    return args, inventory, method, summary, source


@pytest.mark.parametrize("epoch,native,kind", [(25, True, "native_early_stop"), (119, False, "epoch_cap_reached")])
def test_native_completion_or_epoch_cap_defines_complete_contiguous_cohort(tmp_path, epoch, native, kind):
    from trajectory_model_eval.checkpoint_matching_cli import _candidate_manifest
    from trajectory_model_eval.checkpoint_matching import validate_cohort_completion
    args, inventory, method, _, source = native_fixture(tmp_path, epoch=epoch, native=native)
    observed, proof = _candidate_manifest(args, source, method)
    assert observed == inventory
    assert proof["completion_kind"] == kind
    assert proof["effective_candidate_epochs"] == list(range(1, epoch + 1))
    assert proof["final_epoch"] == epoch
    validate_cohort_completion(proof, proof["effective_candidate_epochs"], method)
    assert {row["path"] for row in proof["evidence_files"]} == {
        "recovery/training_summary.json", "recovery/last.pt", "recovery/candidate_manifest.json",
        "protocol_revision.json", "checkpoint_matching_config.json"}


@pytest.mark.parametrize("status,reason", [("incomplete_resumable", "bounded_invocation_resumable"),
    ("interrupted", "interrupted"), ("paused_by_user", "paused_by_user"), ("training_complete", "maximum_epochs")])
def test_partial_pause_interruption_or_non_native_stopping_never_qualifies(tmp_path, status, reason):
    from trajectory_model_eval.checkpoint_matching_cli import _candidate_manifest
    args, _, method, summary, source = native_fixture(tmp_path)
    summary.update(status=status, stop_reason=reason)
    (args.recovery_root / "training_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="pauses and interruptions"):
        _candidate_manifest(args, source, method)


def test_native_completion_requires_exact_checkpoint_contiguity_and_preselection_revision(tmp_path):
    from trajectory_model_eval.checkpoint_matching_cli import _completion_evidence
    args, inventory, method, _, _ = native_fixture(tmp_path)
    changed = deepcopy(inventory)
    changed["candidates"].pop(3)
    with pytest.raises(ValueError, match="every completed epoch"):
        _completion_evidence(args, changed, method)
    revision_path = tmp_path / "protocol_revision.json"
    revision = json.loads(revision_path.read_text())
    revision["candidate_validation_scores_read"] = True
    revision["revision_sha256"] = canonical_digest({key: value for key, value in revision.items() if key != "revision_sha256"})
    revision_path.write_text(json.dumps(revision))
    with pytest.raises(ValueError, match="before checkpoint scoring"):
        _completion_evidence(args, inventory, method)
    (args.recovery_root / "last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint differs"):
        _completion_evidence(args, inventory, method)


@pytest.mark.parametrize("changed", [None, "test_loaded", "cohort_rule", "unbound_identity"])
def test_fresh_run_uses_pretraining_declaration_bound_to_native_checkpoint(tmp_path, changed):
    import torch
    from trajectory_model_eval.checkpoint_matching import COHORT_RULE
    from trajectory_model_eval.checkpoint_matching_cli import _completion_evidence
    args, inventory, method, summary, _ = native_fixture(tmp_path)
    (tmp_path / "protocol_revision.json").unlink()
    declaration = {"schema_version": "sensorium_checkpoint_recovery_v2", "cohort_rule": COHORT_RULE,
        "candidate_epochs": list(range(1, 120)), "maximum_candidate_epoch": 119,
        "native_early_stop_reason": "patience_cycles_exhausted", "test_loaded": False,
        "candidate_validation_scores_read": False, "checkpoint_selection_started": False,
        "created_at_utc": "2026-11-08T00:00:00+00:00"}
    if changed == "test_loaded":
        declaration["test_loaded"] = True
    elif changed == "cohort_rule":
        declaration["cohort_rule"] = "arbitrary_partial"
    declaration["declaration_sha256"] = canonical_digest(declaration)
    path = args.recovery_root / "recovery_protocol.json"
    path.write_text(json.dumps(declaration))
    inventory["training_identity"]["recovery_declaration_sha256"] = (
        "f" * 64 if changed == "unbound_identity" else declaration["declaration_sha256"])
    digest = canonical_digest(inventory["training_identity"])
    last_path = args.recovery_root / "last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    last.update(identity=inventory["training_identity"], identity_digest=digest)
    torch.save(last, last_path)
    summary.update(identity_digest=digest, last_sha256=file_sha256(last_path))
    (args.recovery_root / "training_summary.json").write_text(json.dumps(summary))
    (args.recovery_root / "candidate_manifest.json").write_text(json.dumps(inventory))
    if changed is not None:
        with pytest.raises(ValueError, match="Fresh cohort declaration"):
            _completion_evidence(args, inventory, method)
    else:
        proof = _completion_evidence(args, inventory, method)
        assert proof["protocol_declaration_kind"] == "recovery_protocol_v2"
        assert proof["protocol_declaration_sha256"] == declaration["declaration_sha256"]
        assert "protocol_revision_sha256" not in proof
        assert "recovery/recovery_protocol.json" in {row["path"] for row in proof["evidence_files"]}
