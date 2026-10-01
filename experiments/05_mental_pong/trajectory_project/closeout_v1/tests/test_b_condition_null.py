"""Synthetic checks for paired-support fixed-head condition permutation."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("null_condition", Path(__file__).resolve().parents[1] / "null_condition.py")
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


def fixture():
    rng = np.random.default_rng(13)
    n, nt = 5, 9
    y = rng.normal(size=(2, n, nt, 2))
    p = .35 * y + rng.normal(size=y.shape)
    y[1, ..., 0] = y[0, ..., 0]
    p[1, ..., 0] = p[0, ..., 0]
    mask = np.ones((n, nt), bool)
    mask[0, 5:] = False;mask[1, :3] = False;mask[2, 1::2] = False;mask[4, 0] = False
    y[:, ~mask] = np.nan;p[:, ~mask] = np.nan
    permutations = np.array([np.arange(n), [1, 0, 4, 2, 3], [4, 3, 2, 1, 0]])
    return p, y, mask, permutations


def test_pair_moments_equal_explicit_original_bin_loop():
    p, y, mask, permutations = fixture()
    moments, counts = b.pair_moments(p, y, mask)
    vectorized, covered = b.evaluate_pair_moments(moments, counts, permutations)
    for q, permutation in enumerate(permutations):
        direct, direct_covered = b.direct_permutation_score(p, y, mask, permutation)
        np.testing.assert_array_equal(covered[q], direct_covered)
        np.testing.assert_allclose(vectorized[q], direct, atol=1e-12, rtol=1e-12, equal_nan=True)


def test_identity_permutation_has_zero_paired_effect():
    p, y, mask, permutations = fixture()
    moments, counts = b.pair_moments(p, y, mask)
    scores, _ = b.evaluate_pair_moments(moments, counts, permutations[:1])
    np.testing.assert_allclose(scores[..., 2], 0., atol=1e-14)
    np.testing.assert_allclose(scores[..., 5], 0., atol=1e-14)


def test_time_and_phase_intersections_do_not_stretch_or_use_one_side_only():
    p, y, mask, permutations = fixture()
    phase = mask.copy();phase[0, :4] = False;phase[1, 4:] = False
    moments, counts = b.pair_moments(p, y, phase)
    scores, covered = b.evaluate_pair_moments(moments, counts, permutations[1:2])
    assert covered[0, 0] == 0 and covered[0, 1] == 0
    for i, j in enumerate(permutations[1]):
        assert covered[0, i] == np.sum(phase[i] & phase[j])
    direct, _ = b.direct_permutation_score(p, y, phase, permutations[1])
    np.testing.assert_allclose(scores[0], direct, atol=1e-12)


def test_matched_must_be_recomputed_on_null_support():
    p, y, mask, permutations = fixture()
    moments, counts = b.pair_moments(p, y, mask)
    scores, covered = b.evaluate_pair_moments(moments, counts, permutations[:2])
    assert covered[1].sum() < covered[0].sum()
    assert abs(scores[1, 0, 0, 1, 0] - scores[0, 0, 0, 1, 0]) > 1e-5
    direct, _ = b.direct_permutation_score(p, y, mask, permutations[1])
    assert scores[1, 0, 0, 1, 0] == pytest.approx(direct[0, 0, 1, 0])


def test_permutation_only_valid_test_and_deterministic_q_split_order():
    valid = [np.array([1, 4, 7]), np.array([0, 3])]
    mapping = b.generate_mappings(valid, 8, 20, 314159)
    np.testing.assert_array_equal(mapping, b.generate_mappings(valid, 8, 20, 314159))
    assert not np.array_equal(mapping, b.generate_mappings(valid, 8, 20, 314160))
    rng = np.random.default_rng(314159)
    for q in range(20):
        for r, v in enumerate(valid):
            np.testing.assert_array_equal(mapping[q, r, v], rng.permutation(v))
            assert (mapping[q, r, np.setdiff1d(np.arange(8), v)] == -1).all()
    audit = b.permutation_audit(mapping, valid, "synthetic")
    assert audit.n_unique_permutations.max() <= 6
    assert (audit.n_repeated_permutation_draws > 0).all()


def test_mapping_subset_does_not_silently_allow_duplicate_condition():
    with pytest.raises(ValueError):
        b.generate_mappings([np.array([1, 1, 2])], 3, 2, 3)


@pytest.mark.parametrize("empty", [False, True])
def test_constant_or_empty_phase_r_is_nan_without_zero_fill(empty):
    n, nt = 3, 5
    p = np.ones((2, n, nt, 2));y = p.copy();mask = np.ones((n, nt), bool)
    if empty:mask[:] = False
    moments, counts = b.pair_moments(p, y, mask)
    scores, covered = b.evaluate_pair_moments(moments, counts, np.array([[2, 1, 0]]))
    assert np.isnan(scores[..., :3]).all()
    if empty:
        assert np.isnan(scores).all() and covered.sum() == 0
    else:
        np.testing.assert_array_equal(scores[..., 3:6], 0.)


def test_x_is_same_for_all_heads_targets_after_shared_mapping():
    p, y, mask, permutations = fixture()
    moments, counts = b.pair_moments(p, y, mask)
    scores, _ = b.evaluate_pair_moments(moments, counts, permutations)
    for h in range(2):
        for k in range(2):
            np.testing.assert_allclose(scores[:, h, k, 0], scores[:, 0, 0, 0], atol=1e-13, equal_nan=True)


def test_primary_difference_signs_are_matched_better_positive():
    p, y, mask, permutations = fixture()
    moments, counts = b.pair_moments(p, y, mask)
    scores, _ = b.evaluate_pair_moments(moments, counts, permutations)
    np.testing.assert_allclose(scores[..., 2], scores[..., 0] - scores[..., 1])
    np.testing.assert_allclose(scores[..., 5], scores[..., 4] - scores[..., 3])


def test_full_coverage_advanced_assignment_keeps_q_and_source_axes():
    output = np.zeros((7, 4, 6, 79), np.int16)
    valid = np.array([1, 3, 9])
    values = np.arange(21).reshape(7, 3)
    output[:, 2, 1, valid] = values
    np.testing.assert_array_equal(output[:, 2, 1, valid], values)


def test_only_finite_splits_are_averaged_and_counted_not_zero_filled():
    values = np.array([[1., np.nan, 3.], [np.nan, np.nan, np.nan]])
    mean, count = b._finite_mean(values, axis=1)
    assert mean[0] == 2. and np.isnan(mean[1])
    np.testing.assert_array_equal(count, [2, 0])
