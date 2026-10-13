"""Small numerical runs test data roles, not Sensorium scientific claims."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from trajectory_model_eval.unified_analysis import (
    _conventional, _trajectory_scores, fit_frozen, run_analysis, score_frozen,
)
from trajectory_model_eval.unified_contract import canonical_digest, manifest_digest


def synthetic_fixture():
    rng = np.random.default_rng(91)
    time, neurons = 12, 7
    roles = np.array(["train"] * 8 + ["validation"] * 4 + ["test"] * 8)
    conditions = np.array([f"train{i}" for i in range(8)] + [f"val{i}" for i in range(4)] +
                          ["test0"] * 4 + ["test1"] * 4)
    phase = np.linspace(0, 2 * np.pi, time)
    latent = np.stack([np.sin(phase), np.cos(phase), np.sin(phase / 2), np.cos(phase / 2)], axis=-1)
    loading = rng.normal(size=(4, neurons))
    base = latent @ loading
    neural = np.stack([base + rng.normal(scale=.1, size=base.shape) + .08 * i for i in range(len(roles))]) + 8
    static = neural + rng.normal(scale=.9, size=neural.shape)
    dynamic = neural + rng.normal(scale=.3, size=neural.shape)
    precision = np.linspace(.8, 1.2, time)[:, None] * np.linspace(.9, 1.1, neurons)[None, :]
    bundle = {"session": np.array("synthetic"), "roles": roles, "conditions": conditions, "movie_ids": conditions.copy(),
              "dataset_indices": np.arange(len(roles)), "trial_ids": np.arange(200, 200 + len(roles)).astype(str),
              "neuron_indices": np.arange(neurons), "neuron_ids": np.array([f"n{i}" for i in range(neurons)]),
              "frame_indices": np.arange(50, 50 + time), "encoder_precision": precision,
              "neural": neural * precision, "static": static * precision, "dynamic": dynamic * precision}
    config = {"project": {"seed": 42}, "data": {"session": "synthetic", "frame_rate_hz": 30,
              "frame_start": 50, "frame_stop": 50 + time, "neurons": neurons},
              "gpfa": {"latent_dim": 4, "effective_observation_step": 2,
                       "initial_lengthscales_seconds": [.125, .25], "selection_em_iterations": 2,
                       "final_em_iterations": 2, "tolerance": 1e-6, "lengthscale_bounds_seconds": [.1, 3.]},
              "response_matching": {"tolerance": .005, "noise_seed": 123, "sigma_grid": [0., .1, .5, 1.]},
              "reliability": {"splits": 2, "block_sizes_frames": [4]},
              "model_nulls": {"permutations": 2, "block_size_frames": 4},
              "traditional": {"paired_bootstrap_samples": 4, "temporal_rsa_stride": 2}}
    manifest = {"schema_version": "sensorium_part3_v1", "synthetic": True,
        "config_sha256": canonical_digest(config), "session": "synthetic",
        "checkpoint_identities": {name: {"sha256": letter * 64, "trained_sessions": ["synthetic"]}
                                  for name, letter in (("static", "a"), ("dynamic", "b"))},
        "preprocessing": {"status": "verified", "provenance": "Known synthetic scale", "sha256": "c" * 64},
        "neurons": {"indices": bundle["neuron_indices"].tolist(), "ids": bundle["neuron_ids"].tolist()},
        "time": {"frame_start": 50, "frame_stop": 50 + time, "frame_rate_hz": 30, "observation_step": 2},
        "eligibility": {"status": "ready", "reasons": []},
        "trials": [{"session": "synthetic", "dataset_index": i, "trial_id": str(bundle["trial_ids"][i]),
                    "movie_id": str(conditions[i]), "role": str(role),
                    "source_tier": {"train": "train", "validation": "oracle", "test": "test"}[str(role)],
                    "response_status": "synthetic", "modalities_complete": True} for i, role in enumerate(roles)]}
    manifest["split_sha256"] = manifest_digest(manifest)
    return bundle, manifest, config


def test_synthetic_complete_flow_preserves_gpfa_and_labels_results(tmp_path):
    bundle, manifest, config = synthetic_fixture()
    result = run_analysis(bundle, manifest, config, tmp_path / "run", synthetic=True)
    assert result["status"] == "synthetic_verified"
    assert result["reliability"]["status"] == "computed"
    assert result["reliability"]["repeat_counts"] == [4, 4]
    assert result["primary_metrics"] == ["position_correlation", "velocity_direction_cosine"]
    frozen = json.loads((tmp_path / "run/frozen_identity.json").read_text())
    assert frozen["gpfa_parameter_digest"] == result["gpfa_parameter_digest"]
    assert frozen["validation_in_final_fit"] is False
    assert (tmp_path / "run/generated_manifest.json").exists()
    assert len(list((tmp_path / "run").glob("*.png"))) <= 3
    with pytest.raises(FileExistsError):
        run_analysis(bundle, manifest, config, tmp_path / "run", synthetic=True)


def test_test_mutation_changes_scores_but_not_fit_or_selection(tmp_path):
    bundle, manifest, config = synthetic_fixture()
    first = fit_frozen(bundle, manifest, config, tmp_path / "first")
    modified = deepcopy(bundle)
    mask = modified["roles"] == "test"
    modified["neural"][mask] = modified["neural"][mask, ::-1] + 1.5
    second = fit_frozen(modified, manifest, config, tmp_path / "second")
    assert first.parameter_digest == second.parameter_digest
    assert first.fit_digest == second.fit_digest
    assert first.cache_identity["gpfa_sha256"] == second.cache_identity["gpfa_sha256"]
    assert first.sigma == second.sigma
    np.testing.assert_array_equal(first.precision, second.precision)
    before = score_frozen(bundle, first, config, tmp_path / "scores_first")
    after = score_frozen(modified, first, config, tmp_path / "scores_changed")
    assert before["test_content_digest"] != after["test_content_digest"]
    assert before["scores"]["dynamic"]["position_correlation"] != after["scores"]["dynamic"]["position_correlation"]
    assert first.model.parameter_digest() == first.parameter_digest


def test_scale_is_train_only_and_time_resolved_conversion_is_exact(tmp_path):
    bundle, manifest, config = synthetic_fixture()
    frozen = fit_frozen(bundle, manifest, config, tmp_path)
    raw_train = bundle["neural"][bundle["roles"] == "train"] / bundle["encoder_precision"]
    std = raw_train.reshape(-1, raw_train.shape[-1]).std(0, ddof=1)
    expected = 1 / np.maximum(std, .01 * std.mean())
    np.testing.assert_allclose(frozen.precision, expected)
    value = bundle["neural"][:2]
    direct = frozen.model.transform_query((value / bundle["encoder_precision"] * expected)[:, frozen.observations], frozen.query_times)
    np.testing.assert_allclose(frozen.latent(value), direct)


def test_constant_and_zero_displacement_scores_are_na():
    value = np.ones((2, 12, 4))
    scores, support = _trajectory_scores(value, {"static": value.copy(), "dynamic": value.copy()}, 1/30)
    assert all(np.isnan(score) for score in scores["static"].values())
    assert support["velocity_direction_cosine_bins"] == 0
    conventional, support = _conventional(value, {"static": value.copy(), "dynamic": value.copy()}, np.array([0, 1]), 2)
    assert all(np.isnan(score) for score in conventional["static"].values())
    assert support["response_r_shared_neurons"] == 0


def test_missing_repeats_are_recorded_without_condition_removal(tmp_path):
    bundle, manifest, config = synthetic_fixture()
    test = np.flatnonzero(bundle["roles"] == "test")
    bundle["conditions"] = bundle["conditions"].astype("U64")
    bundle["conditions"][test] = np.array([f"unique{i}" for i in range(len(test))])
    bundle["movie_ids"] = bundle["conditions"].copy()
    for i in test:
        manifest["trials"][i]["movie_id"] = str(bundle["conditions"][i])
    manifest["split_sha256"] = manifest_digest(manifest)
    frozen = fit_frozen(bundle, manifest, config, tmp_path / "fit")
    result = score_frozen(bundle, frozen, config, tmp_path / "scores")
    assert result["reliability"]["status"] == "insufficient_test_repeats"
    assert result["reliability"]["conditions_removed"] == 0
    assert result["n_test_conditions"] == len(test)


def test_gpfa_parameter_and_inference_identity_mutation_is_rejected(tmp_path):
    bundle, manifest, config = synthetic_fixture()
    frozen = fit_frozen(bundle, manifest, config, tmp_path / "fit")
    changed = deepcopy(bundle)
    changed["encoder_precision"] *= 2
    with pytest.raises(ValueError, match="encoder_precision"):
        score_frozen(changed, frozen, config, tmp_path / "scores")
    frozen.model.C[0, 0] += 1
    with pytest.raises(AssertionError, match="parameters changed"):
        frozen.latent(bundle["neural"][:2])


def test_config_or_trial_role_cannot_change_after_freeze(tmp_path):
    bundle, manifest, config = synthetic_fixture()
    wrong = deepcopy(config)
    wrong["gpfa"]["initial_lengthscales_seconds"] = [.5]
    with pytest.raises(ValueError, match="Configuration digest"):
        fit_frozen(bundle, manifest, wrong, tmp_path / "wrong")
    frozen = fit_frozen(bundle, manifest, config, tmp_path / "fit")
    altered = deepcopy(bundle)
    altered["roles"][-1] = "validation"
    with pytest.raises(ValueError, match="identity changed"):
        score_frozen(altered, frozen, config, tmp_path / "roles")
    frozen.model.times_seconds += .01
    with pytest.raises(AssertionError, match="time grid changed"):
        frozen.latent(bundle["neural"][:2])


def test_time_resolved_encoder_scale_does_not_change_gpfa_or_nulls(tmp_path):
    bundle, manifest, config = synthetic_fixture()
    raw = {name: bundle[name] / bundle["encoder_precision"] for name in ("neural", "static", "dynamic")}
    scalar = deepcopy(bundle)
    scalar.update(raw)
    scalar["encoder_precision"] = np.ones(bundle["neural"].shape[-1])
    varying = deepcopy(scalar)
    varying["encoder_precision"] = np.ones(bundle["neural"].shape[1:])
    varying["encoder_precision"][1::2] = 2.
    for name in raw:
        varying[name] = raw[name] * varying["encoder_precision"]
    first = fit_frozen(scalar, manifest, config, tmp_path / "scalar_fit")
    second = fit_frozen(varying, manifest, config, tmp_path / "varying_fit")
    assert first.parameter_digest == second.parameter_digest
    score_frozen(scalar, first, config, tmp_path / "scalar_scores")
    score_frozen(varying, second, config, tmp_path / "varying_scores")
    for filename in ("model_nulls.csv", "reliability_nulls.csv", "reliability_observed.csv"):
        a = pd.read_csv(tmp_path / "scalar_scores" / filename)
        b = pd.read_csv(tmp_path / "varying_scores" / filename)
        np.testing.assert_allclose(a.value, b.value, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_output_control_parameters_cannot_change_after_freeze(tmp_path):
    bundle, manifest, config = synthetic_fixture()
    frozen = fit_frozen(bundle, manifest, config, tmp_path / "fit")
    frozen.noise_scale *= 2
    with pytest.raises(AssertionError, match="perturbation settings changed"):
        score_frozen(bundle, frozen, config, tmp_path / "scores")


def test_matching_uses_shared_neurons_without_changing_gpfa_neurons(tmp_path):
    bundle, manifest, config = synthetic_fixture()
    bundle["static"][:, :, 0] = 5.
    bundle["dynamic"][:, :, 1] = 5.
    frozen = fit_frozen(bundle, manifest, config, tmp_path / "fit")
    selection = pd.read_csv(tmp_path / "fit/response_matching_selection.csv")
    assert (selection.n_shared_neurons == len(bundle["neuron_ids"]) - 2).all()
    np.testing.assert_array_equal(frozen.neuron_ids, bundle["neuron_ids"])
    assert frozen.model.C.shape[0] == len(bundle["neuron_ids"])
