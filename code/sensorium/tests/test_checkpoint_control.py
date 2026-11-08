"""Control integration checks use artificial arrays, never scientific evidence."""
from copy import deepcopy

import numpy as np

from test_checkpoint_matching import fixture as identity_fixture, candidate
from test_retrained_analysis import subset
from test_unified_analysis import synthetic_fixture
from trajectory_model_eval.checkpoint_matching import validation_reference, score_candidate, freeze_selection
from trajectory_model_eval.unified_analysis import fit_frozen, score_checkpoint_control, score_frozen, _digest


def setup_control(tmp_path):
    bundle, manifest, config = synthetic_fixture()
    _, identities, source = identity_fixture(tmp_path)
    dev = subset(bundle, bundle["roles"] != "test")
    reference = validation_reference(dev)
    rows = [score_candidate(reference, candidate(dev, dev[name]), identity)
            for name, identity in zip(("static", "dynamic"), identities)]
    record = freeze_selection(reference, rows, source, expected_epochs=[1, 2])
    frozen = fit_frozen(bundle, manifest, config, tmp_path / "fit")
    frozen.matching, frozen.control_digest = record, _digest(record)
    bundle["matched_dynamic"] = bundle["static"].copy()
    return bundle, frozen, config


def test_control_scores_test_only_actual_predictions_and_keeps_selected_checkpoint(tmp_path):
    bundle, frozen, config = setup_control(tmp_path)
    mask = bundle["roles"] == "test"
    test = subset({k: v for k, v in bundle.items() if k != "matched_dynamic"}, mask)
    test["matched_dynamic"] = bundle["matched_dynamic"][mask]
    record = deepcopy(frozen.matching)
    result = score_checkpoint_control(test, frozen, config, tmp_path / "same")
    assert result["test_absolute_gap"] == 0
    assert result["test_matching_retained"]
    assert result["prediction_transform"] == "none"
    assert result["scores"]["static"] == result["scores"]["matched_dynamic"]
    test["matched_dynamic"] = test["matched_dynamic"][:, ::-1].copy()
    failed = score_checkpoint_control(test, frozen, config, tmp_path / "failed")
    assert not failed["test_matching_retained"]
    assert not failed["test_retuned"]
    assert failed["Q1"] == "response_matching_not_achieved_or_not_retained"
    assert frozen.matching == record


def test_undefined_test_neuron_cannot_shrink_frozen_cohort_and_claim_matching(tmp_path):
    bundle, frozen, config = setup_control(tmp_path)
    bundle["neural"][bundle["roles"] == "test", :, 0] = 1.
    result = score_checkpoint_control(bundle, frozen, config, tmp_path / "undefined")
    assert np.isnan(result["test_absolute_gap"])
    assert result["test_matching_status"] == "undefined_response_r_on_frozen_neurons"
    assert not result["test_matching_evaluable"]
    assert not result["test_matching_retained"]
    assert result["support"]["response_r_shared_neurons"] == result["support"]["response_r_frozen_neurons"] - 1


def test_control_cannot_change_main_model_scores_or_gpfa(tmp_path):
    bundle, frozen, config = setup_control(tmp_path)
    before = score_frozen(bundle, frozen, config, tmp_path / "first")
    altered = deepcopy(bundle)
    altered["matched_dynamic"] = altered["matched_dynamic"][:, ::-1].copy()
    after = score_frozen(altered, frozen, config, tmp_path / "second")
    assert before["scores"] == after["scores"]
    assert before["gpfa_parameter_digest"] == after["gpfa_parameter_digest"]
    assert before["response_matching"]["test_content_digest"] != after["response_matching"]["test_content_digest"]
