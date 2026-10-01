"""Synthetic tests only: no data fitting or representation changes."""
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("decoding_condition", HERE / "decoding_condition.py")
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


def synthetic():
    rng = np.random.default_rng(33)
    ids = np.arange(1000, 1079)
    nt = 8
    z = rng.normal(size=(79, nt, 50))
    obj = z @ rng.normal(size=(50, 2)) / 4
    beh = obj.copy()
    beh[..., 1] += np.linspace(0., 1., nt)[None] * rng.normal(size=(79, 1))
    mask = np.ones((79, nt), bool)
    mask[-1] = False
    mask[4, 3] = False
    visible = np.broadcast_to(np.arange(nt) < 4, (79, nt)).copy()
    bounce = np.broadcast_to((ids % 2 == 0)[:, None], (79, nt)).copy()
    payload = dict(animal="synthetic", condition_ids=ids, times_ms=np.arange(1, nt + 1) * 50.,
                   objective_xy=obj, behavior_xy=beh, common_mask=mask, n_behavior_trials=rng.integers(2, 100, 79),
                   epoch_masks=dict(full=mask, visible=visible, hidden=~visible, bounce=bounce,
                                    no_bounce=~bounce, post_bounce=bounce & ~visible), data_mode=d.MODE)
    tr = np.arange(39); te = np.arange(39, 79)
    split = dict(iteration=0, train_indices=tr, test_indices=te, train_condition_ids=ids[tr], test_condition_ids=ids[te])
    reps = {rep: dict(latent=value, condition_ids=ids, times_ms=payload["times_ms"], common_mask=np.ones_like(mask),
                      causal_status="published_input_causal_train_only;raw_preprocessing_fail",
                      provenance=dict(fit_conditions=ids[tr].tolist(), test_conditions=ids[te].tolist(), round_id=0,
                                      fit_provenance="pass", provided_input_filtering="pass", raw_preprocessing="fail"))
            for rep, value in (("FA50", z), ("GPFA50", rng.normal(size=z.shape)))}
    return payload, reps, split


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    payload, reps, split = synthetic()
    root = tmp_path_factory.mktemp("condition_decode")
    output = d.run_condition_round(payload, reps, split, root, threads=1)
    return payload, reps, split, root, output


def test_multitarget_solve_equals_independent_unweighted_heads():
    payload, reps, split = synthetic()
    mask = payload["common_mask"].copy();mask[split["test_indices"]] = False
    x = reps["FA50"]["latent"][mask]
    obj, beh = payload["objective_xy"][mask], payload["behavior_xy"][mask]
    model = d.fit_condition_ols(x, obj, beh, threads=1)
    with threadpool_limits(1):
        for head, y in enumerate((obj, beh)):
            separate = LinearRegression(fit_intercept=True, positive=False).fit(x, y)
            np.testing.assert_allclose(model.coef_[head * 2:head * 2 + 2], separate.coef_, atol=1e-12)
            np.testing.assert_allclose(model.intercept_[head * 2:head * 2 + 2], separate.intercept_, atol=1e-12)


def test_fit_contains_one_condition_time_row_not_trial_repetition(result):
    payload, _, split, _, output = result
    expected = int(payload["common_mask"][split["train_indices"]].sum())
    assert expected == 39 * 8 - 1
    for row in output["fit_audit"].itertuples():
        assert row.n_training_rows == expected == row.n_unique_training_neural_input_ids
        assert row.duplicate_condition_time_rows == 0
        assert not row.behavior_counts_used_as_weights and not row.representation_refitted
        assert row.coefficients_per_xy_head == 102
        with np.load(row.model_path) as saved:
            indices = saved["training_condition_bin_indices"]
            assert len(indices) == len(np.unique(indices, axis=0)) == expected
            assert not np.isin(indices[:, 0], split["test_indices"]).any()


def test_behavior_trial_counts_do_not_change_weights(result, tmp_path):
    payload, reps, split, _, output = result
    modified = dict(payload, n_behavior_trials=payload["n_behavior_trials"] * np.arange(1, 80))
    second = d.run_condition_round(modified, reps, split, tmp_path, threads=1)
    for old, new in zip(output["fit_audit"].model_path, second["fit_audit"].model_path):
        with np.load(old) as a, np.load(new) as b:
            np.testing.assert_array_equal(a["coefficients"], b["coefficients"])
            np.testing.assert_array_equal(a["intercepts"], b["intercepts"])
    for key in ("r_obj", "r_beh", "RMSE_obj", "RMSE_beh", "Delta_r", "Delta_RMSE"):
        np.testing.assert_array_equal(output["summary"][key], second["summary"][key])


def test_pure_test_prediction_mask_x_and_exact_identities(result):
    payload, _, split, _, output = result
    for path in output["prediction_paths"].values():
        with np.load(path) as saved:
            p = saved["predictions"]
            assert p.shape == (2, 79, 8, 2)
            assert np.isnan(p[:, split["train_indices"]]).all()
            assert not saved["test_input_support"][split["train_indices"]].any()
            np.testing.assert_array_equal(saved["condition_ids"], payload["condition_ids"])
            np.testing.assert_array_equal(saved["test_condition_indices"], split["test_indices"])
            np.testing.assert_allclose(p[0, ..., 0], p[1, ..., 0], atol=1e-12, equal_nan=True)
            assert saved["data_mode"].item() == d.MODE
            assert "raw_preprocessing_fail" in saved["causal_status"].item()


def test_primary_scores_direct_original_timepoint_pooling(result):
    payload, _, _, _, output = result
    for representation, path in output["prediction_paths"].items():
        with np.load(path) as saved:
            p = saved["predictions"].copy();mask = saved["test_input_support"].copy()
        for h, target in enumerate((payload["objective_xy"], payload["behavior_xy"])):
            suffix = "obj" if h == 0 else "beh"
            y, pred = target[mask], p[h][mask]
            for axis, coord in enumerate(("x", "y")):
                row = output["summary"].query('representation==@representation and epoch=="full" and coordinate==@coord').iloc[0]
                assert row["r_" + suffix] == pytest.approx(pearsonr(y[:, axis], pred[:, axis]).statistic, abs=1e-12)
                assert row["RMSE_" + suffix] == pytest.approx(np.sqrt(np.mean((y[:, axis] - pred[:, axis]) ** 2)), abs=1e-12)
                assert row.n_bins == mask.sum() == row.n_condition_time_rows == row.n_unique_neural_input_ids
                assert row.n_conditions == 39
                assert row.n_neural_trials == 0
                assert row.Delta_r == pytest.approx(row.r_beh - row.r_obj)
                assert row.Delta_RMSE == pytest.approx(row.RMSE_obj - row.RMSE_beh)


def test_own_cross_columns_not_switched(result):
    *_, output = result
    for row in output["summary"].itertuples():
        cross = output["cross_2x2"]
        part = cross[(cross.representation == row.representation) & (cross.epoch == row.epoch) & (cross.coordinate == row.coordinate)]
        assert len(part) == 4 and part.n_bins.nunique() == 1
        oo = part[(part["head"] == "D_obj") & (part.target == "objective")].iloc[0]
        bb = part[(part["head"] == "D_beh") & (part.target == "behavior")].iloc[0]
        assert row.r_obj == oo.r and row.r_beh == bb.r
        assert row.RMSE_obj == oo.RMSE and row.RMSE_beh == bb.RMSE


def test_missing_condition_keeps_test_identity_and_nan_scores(result):
    payload, _, split, _, output = result
    condition = output["condition_metrics"]
    assert set(condition.condition_id) == set(payload["condition_ids"][split["test_indices"]])
    missing = condition[condition.condition_index == 78]
    assert len(missing) == 2 * 6 * 4 * 2
    assert missing.n_bins.eq(0).all()
    assert missing[list(d.SCORE_FIELDS)].isna().all().all()


def test_metrics_match_prior_scoring_conventions():
    spec = importlib.util.spec_from_file_location("prior_decoding", HERE.parent / "trial_endpoint_fa_gpfa_v1" / "decoding_trial.py")
    prior = importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    rng = np.random.default_rng(1)
    y, p = rng.normal(size=(75, 2)), rng.normal(size=(75, 2))
    old = prior._final_stats(prior._raw_stats(y[None], p[None], np.ones((1, 75), bool)))
    new = d.score_rows(y, p)
    for key in d.SCORE_FIELDS:
        np.testing.assert_allclose(new[key], old[key][0], atol=1e-12)


def test_causal_name_cannot_replace_existing_audit():
    payload, reps, split = synthetic()
    reps["GPFA50"]["provenance"].pop("provided_input_filtering")
    with pytest.raises(ValueError, match="existing provided_input_filtering"):
        d.validate_inputs(payload, reps, split)


def test_empty_and_constant_correlation_not_zero_or_fake_valid():
    zero = d.score_rows(np.zeros((0, 2)), np.zeros((0, 2)))
    assert all(np.isnan(value).all() for value in zero.values())
    constant = d.score_rows(np.ones((5, 2)), np.ones((5, 2)))
    assert np.isnan(constant["r"]).all()
    np.testing.assert_array_equal(constant["RMSE"], [0., 0.])
    assert np.isnan(d.score_rows(np.ones((2, 2)), np.zeros((2, 2)))["r"]).all()


def test_four_group_support_intersection_and_no_trial_mean_substitution():
    payload, reps, split = synthetic()
    payload["behavior_xy"][1, 1, 1] = np.nan
    reps["FA50"]["latent"][2, 2, 0] = np.nan
    reps["GPFA50"]["common_mask"][3, 3] = False
    payload["n_behavior_trials"][4] = 0
    *_, common, epochs, train, test = d.validate_inputs(payload, reps, split)
    assert not common[1, 1] and not common[2, 2] and not common[3, 3] and not common[4].any()
    assert all(not phase[3, 3] for phase in epochs.values())
    assert len(train) == 39 and len(test) == 40


@pytest.mark.parametrize("issue", ["split", "dimension", "mode", "x", "representation_split", "round", "filtering"])
def test_primary_protocol_mismatches_rejected(issue):
    payload, reps, split = synthetic()
    if issue == "split": split["test_indices"][0] = split["train_indices"][0]
    if issue == "dimension": reps["FA50"]["latent"] = reps["FA50"]["latent"][..., :49]
    if issue == "mode": payload["data_mode"] = "trial_labels_with_mean_neural"
    if issue == "x": payload["behavior_xy"][1, 1, 0] += 1.
    if issue == "representation_split": reps["FA50"]["provenance"]["fit_conditions"] = split["test_condition_ids"].tolist()
    if issue == "round": reps["GPFA50"]["provenance"]["round_id"] = 2
    if issue == "filtering": reps["FA50"]["provenance"]["provided_input_filtering"] = "fail"
    with pytest.raises(ValueError):d.validate_inputs(payload, reps, split)


def test_changing_heldout_behavior_does_not_change_fitted_weights(result, tmp_path):
    payload, reps, split, _, output = result
    modified = dict(payload, behavior_xy=payload["behavior_xy"].copy())
    modified["behavior_xy"][split["test_indices"], :, 1] += 8.
    second = d.run_condition_round(modified, reps, split, tmp_path, threads=1)
    for old, new in zip(output["fit_audit"].model_path, second["fit_audit"].model_path):
        with np.load(old) as a, np.load(new) as b:
            np.testing.assert_array_equal(a["coefficients"], b["coefficients"])
            np.testing.assert_array_equal(a["intercepts"], b["intercepts"])
    for rep in d.REPRESENTATIONS:
        with np.load(output["prediction_paths"][rep]) as a, np.load(second["prediction_paths"][rep]) as b:
            np.testing.assert_array_equal(a["predictions"], b["predictions"])
