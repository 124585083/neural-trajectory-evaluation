import numpy as np

from trajectory_model_eval.question_analysis import _balanced_repeat_split, _sign_flip_p


def test_balanced_repeat_split_is_disjoint_and_balanced():
    conditions = np.repeat(np.arange(3), [10, 9, 8])
    left, right = _balanced_repeat_split(conditions, 3)
    assert not set(left) & set(right)
    for condition in np.unique(conditions):
        assert np.sum(conditions[left] == condition) == np.sum(conditions[right] == condition)


def test_exact_sign_flip_detects_all_positive_six_condition_effect():
    assert _sign_flip_p(np.ones(6)) == 1.0 / 64.0

