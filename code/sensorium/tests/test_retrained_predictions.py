"""Identity, support and cache checks for the new prediction export."""
import copy
import json
import numpy as np
import pytest

from trajectory_model_eval.retrained_predictions import (combine_pilot, load_bundle,
    save_bundle, select_neurons, bundle_identity, verify_source_files, verify_test_lock)
from trajectory_model_eval.unified_contract import canonical_digest, file_sha256


def tiny_bundle(indices, roles):
    array = np.arange(len(indices) * 5 * 3, dtype=np.float32).reshape(len(indices), 5, 3)
    return {"session": np.asarray("pilot"), "encoder_precision": np.ones((5, 3)),
            "neuron_indices": np.asarray([7, 2, 4]), "neuron_ids": np.asarray(["a", "b", "c"]),
            "frame_indices": np.arange(50, 55), "dataset_indices": np.asarray(indices),
            "trial_ids": np.asarray(indices), "conditions": np.asarray([str(i) for i in indices]),
            "movie_ids": np.asarray([str(i) for i in indices]), "roles": np.asarray(roles),
            "neural": array, "static": array + 1, "dynamic": array + 2}


def test_disjoint_role_exports_restore_manifest_order():
    development = tiny_bundle([3, 1], ["train", "validation"])
    test = tiny_bundle([2], ["test"])
    manifest = {"session": "pilot", "trials": [
        {"session": "pilot", "dataset_index": i, "role": r}
        for i, r in [(1, "validation"), (2, "test"), (3, "train")]]}
    combined = combine_pilot(development, test, manifest)
    assert combined["dataset_indices"].tolist() == [1, 2, 3]
    np.testing.assert_array_equal(combined["neural"][0], development["neural"][1])
    changed = copy.deepcopy(test)
    changed["roles"] = np.asarray(["train"])
    with pytest.raises(ValueError, match="development row"):
        combine_pilot(development, changed, manifest)


def test_neuron_selection_retains_declared_order_and_scale():
    bundle = tiny_bundle([1], ["test"])
    selected = select_neurons(bundle, [4, 7])
    assert selected["neuron_ids"].tolist() == ["c", "a"]
    np.testing.assert_array_equal(selected["neural"], bundle["neural"][..., [2, 0]])
    assert selected["encoder_precision"].shape == (5, 2)


def test_cache_rejects_changed_payload_or_sources(tmp_path):
    bundle = tiny_bundle([1], ["test"])
    path = tmp_path / "predictions.npz"
    identity = {"source_sha256": "new_run", "roles": ["test"]}
    save_bundle(path, bundle, identity)
    np.testing.assert_array_equal(load_bundle(path, identity)["neural"], bundle["neural"])
    with pytest.raises(ValueError, match="different sources"):
        load_bundle(path, {**identity, "source_sha256": "old_run"})
    bundle["neural"][0, 0, 0] += 1
    np.savez_compressed(path, **bundle)
    with pytest.raises(ValueError, match="payload changed"):
        load_bundle(path, identity)


def test_test_cache_requires_freeze_and_gpfa_identities():
    config = {"data": {"frame_start": 50, "frame_stop": 55}}
    manifest = {"split_sha256": "s" * 64, "trials": [
        {"session": "pilot", "role": "test", "dataset_index": 2, "source_sha256": {"responses": "r" * 64}}]}
    preprocessing = {"preprocessing_sha256": "p" * 64}
    checkpoints = {name: {"sha256": name * 8} for name in ("static", "dynamic")}
    with pytest.raises(ValueError, match="completed analysis freeze"):
        bundle_identity(config, manifest, preprocessing, checkpoints, "pilot", ("test",), [0, 1])
    frozen = {"status": "all_choices_frozen", "split_sha256": manifest["split_sha256"],
        "config_sha256": canonical_digest(config), "preprocessing_sha256": preprocessing["preprocessing_sha256"],
        "checkpoint_identities": checkpoints, "freeze_sha256": "f" * 64, "gpfa_identity": {"gpfa_sha256": "g" * 64}}
    identity = bundle_identity(config, manifest, preprocessing, checkpoints, "pilot", ("test",), [0, 1], frozen_analysis=frozen)
    assert identity["freeze_sha256"] == "f" * 64 and identity["gpfa_sha256"] == "g" * 64


def test_changed_source_responses_reject_stale_cache_before_scoring(tmp_path):
    path = tmp_path / "response.npy"
    np.save(path, np.arange(5))
    manifest = {"trials": [{"session": "pilot", "role": "test", "dataset_index": 3,
        "source_files": {"responses": "response.npy"}, "source_sha256": {"responses": file_sha256(path)}}]}
    verify_source_files(manifest, tmp_path, roles=("test",), sessions=["pilot"])
    np.save(path, np.arange(5) * 2)
    with pytest.raises(ValueError, match="cached targets are stale"):
        verify_source_files(manifest, tmp_path, roles=("test",), sessions=["pilot"])


def test_direct_test_gate_requires_actual_frozen_models(tmp_path):
    from test_retrained_analysis import fixture
    from trajectory_model_eval.retrained_analysis import fit_gpfa, freeze_analysis

    _, dev, manifest, config, identities = fixture()
    preprocessing = {"preprocessing_sha256": "c" * 64}
    with pytest.raises(ValueError, match="completed analysis freeze"):
        verify_test_lock(config, manifest, preprocessing, tmp_path, None)
    fit_gpfa(dev, manifest, config, "c" * 64, tmp_path / "gpfa")
    for name, identity in identities.items():
        directory = tmp_path / "encoders" / name
        directory.mkdir(parents=True)
        checkpoint = directory / "best.pt"
        checkpoint.write_bytes(name.encode())
        identity.update(checkpoint="best.pt", sha256=file_sha256(checkpoint))
        (directory / "frozen_identity.json").write_text(json.dumps(identity))
    freeze_analysis(dev, manifest, config, "c" * 64, tmp_path / "gpfa", identities, tmp_path / "frozen")
    frozen = verify_test_lock(config, manifest, preprocessing, tmp_path / "encoders", tmp_path / "frozen")
    assert frozen["status"] == "all_choices_frozen"
    (tmp_path / "encoders/static/best.pt").write_bytes(b"modified checkpoint")
    with pytest.raises(ValueError, match="Actual encoder checkpoint changed"):
        verify_test_lock(config, manifest, preprocessing, tmp_path / "encoders", tmp_path / "frozen")
