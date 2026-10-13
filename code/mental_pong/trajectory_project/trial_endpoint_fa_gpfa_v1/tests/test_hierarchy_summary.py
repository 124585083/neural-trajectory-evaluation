from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("hierarchy_summary", Path(__file__).resolve().parents[1] / "hierarchy_summary.py")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)


def test_pair_difference_is_averaged_only_with_finite_pair():
    rows = []
    for iteration, values in enumerate(((.1, .5), (np.nan, .9))):
        for head, target, r in (("D_obj", "objective", values[0]), ("D_beh", "behavior", values[1])):
            rows.append(dict(animal="test", representation="FA50", data_mode=h.MODE, causal_status="raw_fail",
                             epoch="full", coordinate="y", condition_id=100, iteration=iteration, head=head,
                             target=target, r=r, RMSE=2. if head == "D_obj" else 1., n_bins=10))
    means = h._paired_unit_means(pd.DataFrame(rows), "condition", None, 2)
    row = means.iloc[0]
    assert row.r_obj == .1 and row.r_beh == .7
    assert row.Delta_r == pytest.approx(.4)  # Not the difference of unequal-availability means (.6).
    assert row.Delta_r_n_finite == 1
    assert row.Delta_RMSE == 1.
    assert row.n_test_splits == 2


def test_between_unit_spread_ignores_missing_not_zero():
    rows = []
    for value in (1., 3., np.nan):
        rows.append(dict(animal="test", representation="FA50", data_mode=h.MODE, causal_status="raw_fail",
                         epoch="full", coordinate="y", n_supported_test_splits=0 if np.isnan(value) else 2,
                         **{metric: value for metric in h.METRICS}))
    result = h._heterogeneity(pd.DataFrame(rows), "trial")
    assert set(result.n_units_total) == {3}
    assert set(result.n_units_finite) == {2}
    assert set(result.n_units_with_support) == {2}
    assert set(result["mean"]) == {2.}
    assert set(result.sd) == {1.}
    assert set(result.q25) == {1.5}
    assert set(result.q75) == {2.5}


def test_formal_requires_actual_100_rounds():
    with pytest.raises(ValueError, match="expected 100"):
        h._require_rounds([0, 1], 100, "test", "FA50")
    with pytest.raises(ValueError, match="original rounds"):
        h._require_rounds(range(1, 101), 100, "test", "FA50")


def test_missing_one_self_head_rejected():
    row = dict(animal="test", representation="FA50", data_mode=h.MODE, causal_status="raw_fail",
               epoch="full", coordinate="y", condition_id=100, iteration=0, head="D_obj", target="objective", r=.1, RMSE=2., n_bins=10)
    with pytest.raises(ValueError, match="matching"):
        h._paired_unit_means(pd.DataFrame([row]), "condition", None, 1)


def test_trial_oof_reduction_preserves_identity_and_missing(tmp_path):
    folder = tmp_path / "readouts" / "trial_scores"
    folder.mkdir(parents=True)
    table = pd.DataFrame(dict(trial_id=["a", "b", "c"], session_id=["s", "s", "s"], condition_id=[1, 1, 2], condition_index=[0, 0, 1]))
    for iteration, selected in ((0, [0, 1]), (1, [0, 2])):
        scores = np.full((2, 6, 2, 2, 2), np.nan)
        scores[:, 0, 0, :, :] = [.2, 2.]
        scores[:, 0, 1, :, :] = [.5, 1.]
        scores[1] = np.nan
        n = np.zeros((2, 6), int);n[0, 0] = 5
        np.savez_compressed(folder / f"test_FA50_r{iteration:03d}_self_scores.npz", animal="test", representation="FA50", iteration=iteration,
                            data_mode=h.MODE, causal_status="raw_fail", epoch_names=h.EPOCHS, head_names=["D_obj", "D_beh"],
                            target_names=["objective", "behavior"], coordinate_names=["x", "y"], metric_names=["r", "RMSE"],
                            test_trial_indices=selected, trial_ids=table.iloc[selected].trial_id.to_numpy(str),
                            condition_ids=table.iloc[selected].condition_id.to_numpy(), session_ids=table.iloc[selected].session_id.to_numpy(str),
                            scores=scores, n_bins=n)
    chunks, _ = h._trial_mean_scores(dict(animal="test", trial_table=table), tmp_path, "FA50", None, 2)
    first = chunks[0]
    assert first.n_test_splits.tolist() == [2, 1, 1]
    assert first.n_supported_test_splits.tolist() == [2, 0, 0]
    assert first.iloc[0].Delta_r == pytest.approx(.3)
    assert first.iloc[0].Delta_RMSE == 1.
    assert first.Delta_r_n_finite.tolist() == [2, 0, 0]
    assert first.iloc[1:][list(h.METRICS)].isna().all().all()
