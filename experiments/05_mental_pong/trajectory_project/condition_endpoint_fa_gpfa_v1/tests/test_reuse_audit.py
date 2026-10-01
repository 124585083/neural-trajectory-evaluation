from pathlib import Path
import copy
import sys

import numpy as np
import pandas as pd
import pytest

from trajectory_project.condition_endpoint_fa_gpfa_v1.reuse_audit import validate_condition_averaging, validate_reused_representations


def fixture():
    ids = np.array([10, 20, 30])
    obj = np.array([[[0., 1.], [5., 3.], [10., 5.]],
                    [[0., 2.], [5., 4.], [10., 1.]],
                    [[np.nan, np.nan]] * 3])
    candidate = np.array([[[0., 1.], [5., 1.5], [10., 2.]],
                          [[0., 1.], [5., 2.5], [10., 4.]],
                          [[0., 2.], [5., 4.], [10., 1.]],
                          [[0., 2.], [5., 4.], [10., 3.]],
                          [[np.nan, np.nan]] * 3])
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]], bool)
    geometry = pd.DataFrame({'condition_index': [0, 1, 2], 'condition_id': ids,
                             'metadata_bounce_count': [0, 1, 1], 'x0': [0., 0., 0.],
                             'y0': [1., 2., np.nan], 'x_end': [10., 10., 10.],
                             'collision_x': [np.nan, 5., np.nan], 'collision_y': [np.nan, 4., np.nan]})
    source = {'animal': 'test', 'condition_ids': ids, 'times_ms': np.array([50, 100, 150]),
              'objective_xy': obj, 'condition_index': np.array([0, 0, 1, 1, 2]),
              'endpoint': np.array([2., 4., 1., 3., np.nan]), 'behavior_xy': candidate,
              'common_mask': mask, 'epoch_masks': {'full': mask}, 'geometry': geometry}
    average = {'animal': 'test', 'condition_ids': ids, 'times_ms': source['times_ms'],
               'objective_xy': obj.copy(), 'behavior_xy': np.array([candidate[:2].mean(0), candidate[2:4].mean(0), candidate[4]]),
               'common_mask': mask[[0, 2, 4]], 'epoch_masks': {'full': mask[[0, 2, 4]]},
               'mean_endpoint': np.array([3., 2., np.nan]), 'n_behavior_trials': np.array([2, 2, 0]),
               'source_trial_indices_by_condition': {'10': [0, 1], '20': [2, 3], '30': []},
               'geometry': geometry.copy()}
    return source, average


def test_same_trial_average_commutes_with_endpoint_formula():
    source, average = fixture()
    result = validate_condition_averaging(source, average)
    assert result['n_source_behavior_trials'] == 4
    assert result['n_valid_conditions'] == 2
    assert result['n_condition_time_rows'] == 6
    assert result['maximum_mean_endpoint_formula_error'] == 0
    assert result['unknown_candidates_not_filled']


def test_variable_trial_per_bin_support_is_rejected():
    source, average = fixture()
    source['common_mask'][1, 1] = False
    with pytest.raises(AssertionError, match='Variable trial membership'):
        validate_condition_averaging(source, average)


def test_changing_trial_membership_is_rejected_even_if_mean_happens_to_match():
    source, average = fixture()
    average['source_trial_indices_by_condition']['10'] = [0]
    with pytest.raises(AssertionError):
        validate_condition_averaging(source, average)


def test_missing_candidates_cannot_be_filled_as_physics_or_zero():
    source, average = fixture()
    average['behavior_xy'][2] = 0
    with pytest.raises(AssertionError, match='must remain unknown'):
        validate_condition_averaging(source, average)


def test_same_x_is_exact_not_approximately_equal():
    source, average = fixture()
    average['behavior_xy'][0, 1, 0] += 1e-13
    with pytest.raises(AssertionError, match='bit-identical'):
        validate_condition_averaging(source, average)


def test_condition_average_cannot_change_physical_collision_anchor():
    source, average = fixture()
    average['geometry'].loc[1, 'collision_x'] = 5.1
    with pytest.raises(AssertionError, match='Physical anchor/time changed'):
        validate_condition_averaging(source, average)


def test_reuse_never_writes_into_historical_analysis(tmp_path):
    with pytest.raises(ValueError, match='must not write'):
        validate_reused_representations(tmp_path, tmp_path, [])
