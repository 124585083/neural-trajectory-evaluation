"""Synthetic only: no scientific data or model selection is performed."""
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits


MODULE = Path(__file__).resolve().parents[1] / "decoding_trial.py"
SPEC = importlib.util.spec_from_file_location("decoding_trial", MODULE)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


def synthetic():
    rng = np.random.default_rng(8)
    ids = rng.permutation(np.arange(2000, 2079))
    nt = 8
    ci = np.repeat(np.arange(79), 3)
    z = rng.normal(size=(79, nt, 50))
    w = rng.normal(size=(50, 2)) / 5
    obj = z @ w
    beh = obj[ci].copy()
    # Real-trial labels disagree for the same repeated neural input.
    offsets = np.tile([-1.1, .1, 1.7], 79)
    beh[..., 1] += offsets[:, None] * np.linspace(.1, 1, nt)
    full = np.ones((len(ci), nt), bool)
    full[0, 1] = False
    # One identity exists but has no usable row; do not silently remove it.
    full[-3:] = False
    visible = np.broadcast_to(np.arange(nt) < 4, full.shape).copy()
    hidden = ~visible
    bounce = np.broadcast_to((ci % 2 == 0)[:, None], full.shape).copy()
    epochs = dict(full=full, visible=visible, hidden=hidden, bounce=bounce,
                  no_bounce=~bounce, post_bounce=bounce & hidden)
    payload = dict(animal="synthetic", data_mode=d.MODE, trial_table=pd.DataFrame({
        "trial_id": [f"trial_{i}" for i in range(len(ci))], "session_id": [f"s{i % 3}" for i in range(len(ci))],
        "condition_index": ci, "condition_id": ids[ci], "source_row": np.arange(len(ci))}),
        condition_ids=ids, times_ms=np.arange(1, nt + 1) * 50., objective_xy=obj,
        behavior_xy=beh, common_mask=full, epoch_masks=epochs)
    reps = {name: dict(latent=latent, causal_status="published_input_causal_train_only;raw_preprocessing_fail",
                       provenance=dict(fit_provenance="pass", provided_input_filtering="pass", raw_preprocessing="fail"))
            for name, latent in (("FA50", z), ("GPFA50", rng.normal(size=z.shape)))}
    return payload, reps, d.generate_splits(ids, 1)[0]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    payload, reps, split = synthetic()
    root = tmp_path_factory.mktemp("trial_decoder")
    out = d.run_trial_round(payload, reps, split, root, chunk_trials=17, threads=1)
    return payload, reps, split, root, out


def test_group_split_is_exactly_actual_physical_id_gss():
    payload, _, _ = synthetic()
    ids = payload["condition_ids"]
    ours = d.generate_splits(ids, 100)
    expected = GroupShuffleSplit(n_splits=100, train_size=39, test_size=40, random_state=0)
    for row, (tr, te) in zip(ours, expected.split(ids[:, None], groups=ids)):
        np.testing.assert_array_equal(row["train_indices"], tr)
        np.testing.assert_array_equal(row["test_indices"], te)
        assert len(set(tr) | set(te)) == 79 and not set(tr) & set(te)


def test_four_output_fit_equals_two_independent_actual_expanded_fits():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(260, 50))
    obj = rng.normal(size=(260, 2)); beh = rng.normal(size=(260, 2)); beh[:, 0] = obj[:, 0]
    joint = d.fit_expanded_ols(x, obj, beh, threads=1)
    with threadpool_limits(1):
        for h, target in enumerate((obj, beh)):
            separate = LinearRegression(fit_intercept=True, positive=False).fit(x, target)
            np.testing.assert_allclose(joint.coef_[h * 2:h * 2 + 2], separate.coef_, atol=1e-12)
            np.testing.assert_allclose(joint.intercept_[h * 2:h * 2 + 2], separate.intercept_, atol=1e-12)


def test_weighted_mean_equivalence_is_only_synthetic_audit():
    rng = np.random.default_rng(3)
    unique = rng.normal(size=(80, 50)); counts = rng.integers(2, 7, size=80)
    groups = np.repeat(np.arange(80), counts)
    obj = rng.normal(size=(len(groups), 2)); beh = rng.normal(size=(len(groups), 2)); beh[:, 0] = obj[:, 0]
    fitted = d.fit_expanded_ols(unique[groups], obj, beh, threads=1)
    targets = np.c_[obj, beh]
    means = np.array([targets[groups == i].mean(0) for i in range(80)])
    with threadpool_limits(1):
        equivalent = LinearRegression().fit(unique, means, sample_weight=counts)
    np.testing.assert_allclose(fitted.predict(unique), equivalent.predict(unique), atol=3e-12)
    predicted = fitted.predict(unique)[groups]
    audit = d.squared_error_decomposition(targets, predicted, groups)
    assert audit["maximum_identity_residual"] < 1e-10
    assert audit["within_trial_label_sse"].min() > 0
    assert audit["same_input_prediction_max_spread"] == 0
    assert fitted.rank_ <= np.linalg.matrix_rank(unique - unique.mean(0))


def test_repeated_input_predictions_must_be_identical():
    with pytest.raises(AssertionError, match="Identical"):
        d.squared_error_decomposition([1., 2.], [1., 1.1], [0, 0])


def test_constant_or_missing_scores_are_not_spurious_correlations():
    y = np.ones((2, 5, 2)); p = y.copy()
    mask = np.array([[True] * 5, [False] * 5])
    scores = d._final_stats(d._raw_stats(y, p, mask))
    assert np.isnan(scores["r"]).all()
    np.testing.assert_array_equal(scores["RMSE"][0], [0., 0.])
    assert np.isnan(scores["RMSE"][1]).all()


def test_shared_masks_expanded_counts_and_heldout_only(run):
    payload, reps, split, _, out = run
    ci = payload["trial_table"].condition_index.to_numpy()
    test = np.isin(ci, split["test_indices"])
    expected = int(payload["common_mask"][test].sum())
    for path in out["prediction_paths"].values():
        with np.load(path) as z:
            assert np.isnan(z["predictions"][:, split["train_indices"]]).all()
            np.testing.assert_array_equal(z["test_trial_indices"], np.flatnonzero(test))
            restored = np.unpackbits(z["test_common_mask_packed"], axis=1)[:, :8].astype(bool)
            np.testing.assert_array_equal(restored, payload["common_mask"][test])
            assert "predictions[head" in z["indexing"].item()
    rows = out["cross_2x2"].query("epoch == 'full'")
    assert set(rows.n_bins) == {expected}
    assert set(rows.n_neural_trials) == {0}
    assert rows.n_real_trials.min() <= test.sum()
    audit = out["fit_audit"]
    assert audit.common_mask_sha256.nunique() == 1
    assert not audit.target_averaging_before_fit.any()
    assert set(audit.sample_weights) == {"none"}
    assert (audit.expanded_design_rank <= audit.unique_design_rank).all()
    assert set(out["condition_metrics"].condition_id) == set(np.asarray(payload["condition_ids"])[split["test_indices"]])


def test_main_self_metrics_are_direct_trial_label_errors_not_cross_columns(run):
    payload, reps, split, _, out = run
    ci = payload["trial_table"].condition_index.to_numpy()
    for representation, path in out["prediction_paths"].items():
        with np.load(path) as z:
            predictions = z["predictions"].copy()
        mask = payload["common_mask"] & np.isin(ci, split["test_indices"])[:, None]
        targets = [payload["objective_xy"][ci], payload["behavior_xy"]]
        for h in range(2):
            predicted = predictions[h, ci][mask]
            true = targets[h][mask]
            for j, coordinate in enumerate(("x", "y")):
                row = out["summary"].query("representation == @representation and epoch == 'full' and coordinate == @coordinate").iloc[0]
                suffix = "obj" if h == 0 else "beh"
                assert row[f"r_{suffix}"] == pytest.approx(pearsonr(true[:, j], predicted[:, j]).statistic, abs=1e-12)
                assert row[f"RMSE_{suffix}"] == pytest.approx(np.sqrt(np.mean((true[:, j] - predicted[:, j]) ** 2)), abs=1e-12)
                assert row.Delta_r == pytest.approx(row.r_beh - row.r_obj)
                assert row.Delta_RMSE == pytest.approx(row.RMSE_obj - row.RMSE_beh)
        np.testing.assert_allclose(predictions[0, ..., 0], predictions[1, ..., 0], atol=1e-12, equal_nan=True)


def test_trial_scores_saved_before_aggregation_and_export_lossless(run, tmp_path):
    payload, _, split, _, out = run
    row = out["fit_audit"].iloc[0]
    with np.load(row.per_trial_scores_path) as saved:
        assert saved["scores"].shape[1:] == (6, 2, 2, 8)
        trial = str(saved["trial_ids"][0])
        ti = int(saved["test_trial_indices"][0])
        metrics = saved["scores"][0].copy()
    result = d.export_trial_scores(row.per_trial_scores_path, tmp_path / "scores.csv.gz", trial_ids=[trial])
    frame = pd.read_csv(result["path"])
    assert len(frame) == 24
    assert set(frame.trial_id) == {trial}
    r = frame.query("epoch == 'full' and head == 'D_beh' and coordinate == 'y'").iloc[0]
    assert r.RMSE == pytest.approx(metrics[0, 1, 1, 1])
    expanded = d.export_trial_predictions(payload, row.prediction_path, tmp_path / "pred.csv.gz", trial_ids=[trial])
    frame = pd.read_csv(expanded["path"])
    assert len(frame) == payload["common_mask"][ti].sum()
    assert set(frame.trial_index) == {ti}
    for sample in frame.itertuples():
        assert sample.y_beh == pytest.approx(payload["behavior_xy"][ti, sample.original_bin_index, 1])
        assert sample.sample_id == ti * 8 + sample.original_bin_index


def test_session_scores_use_trial_rows_on_same_mask(run):
    payload, _, split, _, out = run
    table = payload["trial_table"]
    ci = table.condition_index.to_numpy()
    path = out["prediction_paths"]["FA50"]
    with np.load(path) as z:
        pred = z["predictions"][1, ci]
    for session in table.session_id.unique():
        mask = payload["common_mask"] & (table.session_id.to_numpy() == session)[:, None] & np.isin(ci, split["test_indices"])[:, None]
        row = out["session_metrics"].query("representation == 'FA50' and session_id == @session and epoch == 'full' and head == 'D_beh' and coordinate == 'y'").iloc[0]
        assert row.n_bins == mask.sum()
        assert row.RMSE == pytest.approx(np.sqrt(np.mean((pred[mask, 1] - payload["behavior_xy"][mask, 1]) ** 2)))


def test_finite_support_is_shared_across_both_representations_and_heads():
    payload, reps, split = synthetic()
    reps["FA50"]["latent"][0, 0, 0] = np.nan
    payload["behavior_xy"][4, 2, 1] = np.nan
    _, ci, common, epochs, _, _ = d._validate(payload, reps, split)
    assert not common[ci == 0, 0].any()
    assert not common[4, 2]
    assert all(not mask[4, 2] for mask in epochs.values())


def test_explicit_representation_measurement_support_overrides_finite_latent():
    payload, reps, split = synthetic()
    reps["GPFA50"]["common_mask"] = np.ones((79, 8), bool)
    reps["GPFA50"]["common_mask"][1, 3] = False
    _, ci, common, epochs, _, _ = d._validate(payload, reps, split)
    assert not common[ci == 1, 3].any()
    reps["GPFA50"]["provenance"]["fit_conditions"] = split["test_condition_ids"]
    with pytest.raises(ValueError, match="provenance"):
        d._validate(payload, reps, split)


@pytest.mark.parametrize("problem", ["mode", "dimension", "provenance", "x", "split"])
def test_invalid_primary_contract_rejected(problem):
    payload, reps, split = synthetic()
    if problem == "mode": payload["data_mode"] = "single_trial_neural"
    if problem == "dimension": reps["FA50"]["latent"] = reps["FA50"]["latent"][..., :49]
    if problem == "provenance": reps["GPFA50"]["provenance"]["fit_provenance"] = "fail"
    if problem == "x": payload["behavior_xy"][3, 3, 0] += 1
    if problem == "split": split["test_indices"][0] = split["train_indices"][0]
    with pytest.raises(ValueError): d._validate(payload, reps, split)


def test_trial_objective_axis_supported_without_full_time_row_explosion(tmp_path):
    payload, reps, split = synthetic()
    ci = payload["trial_table"].condition_index.to_numpy()
    payload["objective_xy"] = payload["objective_xy"][ci]
    payload["objective_axis"] = "trial"
    out = d.run_trial_round(payload, reps, split, tmp_path, chunk_trials=50, threads=1)
    assert len(out["summary"]) == 24


def test_export_rejects_different_trial_identity(run, tmp_path):
    payload, _, _, _, out = run
    copied = dict(payload, trial_table=payload["trial_table"].copy())
    copied["trial_table"].loc[0, "trial_id"] = "substituted_trial"
    with pytest.raises(ValueError, match="identical real-trial"):
        d.export_trial_predictions(copied, out["prediction_paths"]["FA50"], tmp_path / "invalid.csv.gz")


def test_cost_estimate_is_expanded_real_trial_memory():
    payload, _, _ = synthetic()
    cost = d.estimate_cost(payload)
    assert cost["n_real_trials"] == 237
    assert cost["expanded_X_float64_bytes_per_representation"] == int(payload["common_mask"].sum() * 39 / 79 * 50 * 8)
