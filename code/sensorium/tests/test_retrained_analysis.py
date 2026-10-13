"""Causal dependency checks for new training, calibration and held-out scoring."""
from copy import deepcopy
import json

import numpy as np
import pytest

from test_unified_analysis import synthetic_fixture
from trajectory_model_eval.retrained_analysis import (
    evaluate_pilot, fit_gpfa, freeze_analysis, load_gpfa, require_frozen,
    score_session_response, summarize_session_responses, validate_development_bundle,
)
from trajectory_model_eval.unified_contract import manifest_digest


def fixture():
    bundle, manifest, config = synthetic_fixture()
    manifest["schema_version"] = "sensorium_part3_retrained_v1"
    manifest["protocol"] = "retrained_v1"
    for row in manifest["trials"]:
        row["source_tier"] = "oracle" if row["role"] == "test" else "train"
        row["group_id"] = row["movie_id"]
    manifest["split_sha256"] = manifest_digest(manifest)
    mask = bundle["roles"] != "test"
    development = subset(bundle, mask)
    identities = {name: {"status": "complete", "initialization": "from_scratch", "selection_role": "validation",
        "split_sha256": manifest["split_sha256"], "preprocessing_sha256": "c" * 64, "sha256": letter * 64}
        for name, letter in (("static", "a"), ("dynamic", "b"))}
    return bundle, development, manifest, config, identities


def subset(bundle, mask):
    trial_keys = {"roles", "conditions", "movie_ids", "dataset_indices", "trial_ids", "neural", "static", "dynamic"}
    return {key: value[mask] if key in trial_keys else value.copy() for key, value in bundle.items()}


def test_fit_rejects_test_arrays_and_freeze_requires_complete_encoders(tmp_path):
    bundle, dev, manifest, config, identities = fixture()
    with pytest.raises(ValueError, match="declared rows"):
        validate_development_bundle(bundle, manifest, config)
    with pytest.raises(ValueError, match="pending"):
        require_frozen(manifest, config, tmp_path / "missing")
    fitted = fit_gpfa(dev, manifest, config, "c" * 64, tmp_path / "gpfa")
    assert fitted.identity["test_loaded"] is False
    assert fitted.identity["validation_in_final_fit"] is False
    assert fitted.identity["n_train_trials"] == 8
    incomplete = deepcopy(identities)
    incomplete["dynamic"]["status"] = "training"
    with pytest.raises(ValueError, match="pending"):
        freeze_analysis(dev, manifest, config, "c" * 64, tmp_path / "gpfa", incomplete, tmp_path / "lock")


def test_test_mutation_cannot_enter_fit_and_changes_current_scores(tmp_path):
    bundle, dev, manifest, config, identities = fixture()
    first = fit_gpfa(dev, manifest, config, "c" * 64, tmp_path / "gpfa")
    modified = deepcopy(bundle)
    modified["neural"][modified["roles"] == "test"] *= 1.3
    second_dev = subset(modified, modified["roles"] != "test")
    second = fit_gpfa(second_dev, manifest, config, "c" * 64, tmp_path / "gpfa2")
    assert first.identity["gpfa_parameter_digest"] == second.identity["gpfa_parameter_digest"]
    assert first.identity["fit_digest"] == second.identity["fit_digest"]
    np.testing.assert_array_equal(first.arrays["precision"], second.arrays["precision"])
    lock = freeze_analysis(dev, manifest, config, "c" * 64, tmp_path / "gpfa", identities, tmp_path / "lock")
    assert lock["test_loaded"] is False
    before = evaluate_pilot(bundle, manifest, config, tmp_path / "gpfa", tmp_path / "lock", tmp_path / "score")
    after = evaluate_pilot(modified, manifest, config, tmp_path / "gpfa", tmp_path / "lock", tmp_path / "changed")
    assert before["test_content_digest"] != after["test_content_digest"]
    assert before["scores"]["dynamic"]["response_nmse"] != after["scores"]["dynamic"]["response_nmse"]
    assert before["gpfa_parameter_digest"] == after["gpfa_parameter_digest"]
    assert before["reliability"]["repeat_counts"] == [4, 4]
    assert (tmp_path / "score/generated_manifest.json").exists()


def test_gpfa_scale_uses_train_and_frozen_artifacts_reject_mismatch(tmp_path):
    _, dev, manifest, config, identities = fixture()
    fitted = fit_gpfa(dev, manifest, config, "c" * 64, tmp_path / "gpfa")
    raw_train = dev["neural"][dev["roles"] == "train"] / dev["encoder_precision"]
    std = raw_train.reshape(-1, raw_train.shape[-1]).std(0, ddof=1)
    np.testing.assert_allclose(fitted.arrays["precision"], 1 / np.maximum(std, .01 * std.mean()))
    changed = deepcopy(config)
    changed["traditional"]["paired_bootstrap_samples"] += 1
    with pytest.raises(ValueError, match="config_sha256"):
        load_gpfa(tmp_path / "gpfa", manifest, changed, "c" * 64)
    freeze_analysis(dev, manifest, config, "c" * 64, tmp_path / "gpfa", identities, tmp_path / "lock")
    record = tmp_path / "lock/frozen_analysis.json"
    identity = json.loads(record.read_text())
    identity["response_matching"]["sigma"] = 900
    record.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="lock changed"):
        require_frozen(manifest, config, tmp_path / "lock")


def test_validation_changes_selection_input_not_training_precision(tmp_path):
    _, dev, manifest, config, identities = fixture()
    first = fit_gpfa(dev, manifest, config, "c" * 64, tmp_path / "first")
    changed = deepcopy(dev)
    changed["neural"][changed["roles"] == "validation"] *= 1.5
    second = fit_gpfa(changed, manifest, config, "c" * 64, tmp_path / "second")
    np.testing.assert_array_equal(first.arrays["precision"], second.arrays["precision"])
    assert first.identity["fit_digest"] != second.identity["fit_digest"]
    with pytest.raises(ValueError, match="responses changed"):
        freeze_analysis(changed, manifest, config, "c" * 64, tmp_path / "first", identities, tmp_path / "lock")


def test_movie_groups_cannot_cross_roles():
    _, dev, manifest, config, _ = fixture()
    manifest["trials"][8]["movie_id"] = manifest["trials"][0]["movie_id"]
    manifest["split_sha256"] = manifest_digest(manifest)
    with pytest.raises(ValueError, match="movie group"):
        validate_development_bundle(dev, manifest, config)


def test_recognized_subclip_groups_cannot_cross_roles():
    _, dev, manifest, config, _ = fixture()
    manifest["trials"][8]["group_id"] = manifest["trials"][0]["group_id"]
    manifest["split_sha256"] = manifest_digest(manifest)
    with pytest.raises(ValueError, match="movie group"):
        validate_development_bundle(dev, manifest, config)


def test_all_session_scores_use_test_only_and_keep_undefined_correlations(tmp_path):
    bundle, _, _, config, _ = fixture()
    test = subset(bundle, bundle["roles"] == "test")
    pooled, local = score_session_response(test, config)
    assert len(pooled) == 4 and len(local) == 8
    assert set(pooled.model) == {"static", "dynamic"}
    assert set(pooled.n_trials) == {8}
    result = summarize_session_responses(pooled, local, config, tmp_path / "responses")
    assert result["n_sessions"] == 1
    assert result["interval_unit"] == "movie_condition_within_session"
    with pytest.raises(ValueError, match="test rows only"):
        score_session_response(bundle, config)
    test["neural"][:] = 1
    constant, _ = score_session_response(test, config)
    assert constant[constant.metric == "response_neuron_mean_r"].value.isna().all()
