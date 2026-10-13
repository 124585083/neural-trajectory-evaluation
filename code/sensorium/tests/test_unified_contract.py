from copy import deepcopy

import numpy as np
import pytest

from trajectory_model_eval.unified_contract import (
    cache_identity, canonical_digest, frozen_torch_inference, load_cache,
    manifest_digest, save_cache, validate_bundle, validate_manifest,
)


def manifest_fixture():
    manifest = {
        "schema_version": "sensorium_part3_v1", "synthetic": True,
        "config_sha256": "a" * 64, "session": "pilot",
        "checkpoint_identities": {name: {"sha256": letter * 64, "trained_sessions": ["pilot"]}
                                  for name, letter in [("static", "b"), ("dynamic", "c")]},
        "preprocessing": {"status": "verified", "provenance": "synthetic training fixture", "sha256": "d" * 64},
        "neurons": {"indices": [0, 2], "ids": ["n0", "n2"]},
        "time": {"frame_start": 50, "frame_stop": 54, "frame_rate_hz": 30, "observation_step": 1},
        "eligibility": {"status": "ready", "reasons": []},
        "trials": [{"session": "pilot", "dataset_index": i, "trial_id": str(10 + i),
                    "movie_id": f"movie{i}", "role": role, "source_tier": tier,
                    "response_status": "synthetic", "modalities_complete": True}
                   for i, (role, tier) in enumerate([("train", "train"), ("validation", "oracle"), ("test", "test")])],
    }
    manifest["split_sha256"] = manifest_digest(manifest)
    return manifest


def seal(manifest):
    manifest["split_sha256"] = manifest_digest(manifest)
    return manifest


def bundle_fixture(manifest):
    return {"session": "pilot", "dataset_indices": np.arange(3), "trial_ids": np.array(["10", "11", "12"]),
            "movie_ids": np.array(["movie0", "movie1", "movie2"]),
            "roles": np.array(["train", "validation", "test"]),
            "neuron_indices": np.array([0, 2]), "neuron_ids": np.array(["n0", "n2"]),
            "frame_indices": np.arange(50, 54),
            "encoder_precision": np.ones(2),
            **{name: np.ones((3, 4, 2)) for name in ("neural", "static", "dynamic")}}


def test_portable_identity_does_not_erase_scientific_changes():
    manifest = manifest_fixture()
    moved = deepcopy(manifest)
    moved["local_paths"] = {"root": "different/computer"}
    assert manifest_digest(moved) == manifest["split_sha256"]
    moved["trials"][0]["movie_id"] = "different_movie"
    assert manifest_digest(moved) != manifest["split_sha256"]
    with pytest.raises(ValueError, match="digest"):
        validate_manifest(moved)


def test_historical_selection_cannot_be_renamed_test():
    manifest = manifest_fixture()
    manifest["trials"][1]["role"] = "test"
    with pytest.raises(ValueError, match="historical selection"):
        validate_manifest(seal(manifest), require_test=False)


def test_blocked_inventory_is_not_a_scientific_run():
    manifest = manifest_fixture()
    manifest["preprocessing"]["status"] = "unverified"
    manifest["trials"][2]["response_status"] = "zero_placeholder"
    manifest["eligibility"] = {"status": "blocked", "reasons": ["No qualified labels"]}
    audit = validate_manifest(seal(manifest), require_test=False)
    assert not audit["ready"]
    with pytest.raises(ValueError, match="blocked"):
        validate_manifest(manifest)


def test_global_movie_overlap_is_reported_then_rejected_for_unseen_scope():
    manifest = manifest_fixture()
    manifest["trials"][2]["movie_id"] = "movie0"
    assert validate_manifest(seal(manifest), require_test=False)["movie_role_overlaps"]
    with pytest.raises(ValueError, match="Movie conditions"):
        validate_manifest(manifest)
    manifest["evaluation_scope"] = "new_repeats_of_seen_movies"
    validate_manifest(seal(manifest))


def test_bundle_identity_checks_order_and_axes():
    manifest = manifest_fixture()
    bundle = bundle_fixture(manifest)
    validate_bundle(bundle, manifest)
    bundle["neuron_ids"] = bundle["neuron_ids"][::-1]
    with pytest.raises(ValueError, match="neuron_ids"):
        validate_bundle(bundle, manifest)


def test_cache_rejects_old_identity_or_modified_payload(tmp_path):
    manifest = manifest_fixture()
    identity = cache_identity(manifest, "e" * 64)
    path = tmp_path / "predictions.npz"
    payload = {"neural": np.ones((2, 3, 4))}
    save_cache(path, payload, identity)
    assert np.array_equal(load_cache(path, identity)["neural"], payload["neural"])
    wrong = {**identity, "config_sha256": "f" * 64}
    with pytest.raises(ValueError, match="identity"):
        load_cache(path, wrong)
    with np.load(path, allow_pickle=False) as archive:
        changed = {key: archive[key] for key in archive.files}
    changed["neural"] = np.zeros((2, 3, 4))
    np.savez(path, **changed)
    with pytest.raises(ValueError, match="payload"):
        load_cache(path, identity)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="historical"):
        load_cache(path, identity)


def test_frozen_inference_preserves_batchnorm_and_rejects_mutation():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    model.train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with frozen_torch_inference(model):
        model(torch.randn(10, 3))
        assert not model.training
    assert model.training
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    with pytest.raises(RuntimeError, match="Frozen encoder"):
        with frozen_torch_inference(model):
            model[1].running_mean.add_(1)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_duplicate_trial_and_nonfinite_prediction_fail():
    manifest = manifest_fixture()
    manifest["trials"].append(deepcopy(manifest["trials"][0]))
    with pytest.raises(ValueError, match="more than once"):
        validate_manifest(seal(manifest))
    manifest = manifest_fixture()
    bundle = bundle_fixture(manifest)
    bundle["dynamic"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="dynamic"):
        validate_bundle(bundle, manifest)


@pytest.mark.parametrize("precision", [np.zeros(2), np.array([1., np.inf]), np.array([1., np.nan]),
                                      -np.ones(2), np.ones((2, 4)), np.ones((1, 2)), np.array([])])
def test_invalid_encoder_precision_is_rejected(precision):
    manifest = manifest_fixture()
    bundle = bundle_fixture(manifest)
    bundle["encoder_precision"] = precision
    with pytest.raises(ValueError, match="encoder_precision"):
        validate_bundle(bundle, manifest)


def test_time_resolved_encoder_precision_is_accepted():
    manifest = manifest_fixture()
    bundle = bundle_fixture(manifest)
    bundle["encoder_precision"] = np.arange(1, 9).reshape(4, 2)
    validate_bundle(bundle, manifest)


def test_missing_movie_and_invalid_time_are_audit_blockers():
    manifest = manifest_fixture()
    manifest["trials"][2]["movie_id"] = None
    manifest["trials"][2]["time_support_valid"] = False
    audit = validate_manifest(seal(manifest), require_test=False)
    assert len(audit["blockers"]) == 2
    with pytest.raises(ValueError, match="Unknown movie"):
        validate_manifest(manifest)
