from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.decomposition import FactorAnalysis

from trajectory_project.trial_endpoint_fa_gpfa_v1 import representations as rep


@pytest.fixture(autouse=True)
def require_original_configuration_provenance():
    # These six synthetic checks read the historical source configuration and
    # its code hashes. They do not read the large released neural dataset.
    if not (rep.PILOT/'gpfa_config.json').is_file():
        pytest.skip('Configure source_pilot for the original configuration/code provenance checks.')


def data_fixture():
    rng = np.random.default_rng(25)
    return {'half1': rng.normal(size=(79, 7, 60)),
            'half2': rng.normal(size=(79, 7, 60)),
            'condition_ids': np.arange(79), 'valid': np.ones((79, 7), dtype=bool),
            'neuron_ids': np.arange(60)}


def test_preprocessing_statistics_ignore_test_conditions_and_half2():
    data = data_fixture()
    cfg = rep.configuration()
    first = rep.fit_preprocessing(data, np.arange(39), cfg)
    data['half1'][39:] = 5000
    data['half2'][:] = 9000
    second = rep.fit_preprocessing(data, np.arange(39), cfg)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_preprocessing_does_not_read_future_for_current_point():
    data = data_fixture()
    prep = rep.fit_preprocessing(data, np.arange(39), rep.configuration())
    seq = data['half1'][50].copy()
    a = rep.preprocess(seq, prep)
    seq[3:] = np.nan
    b = rep.preprocess(seq, prep)
    np.testing.assert_array_equal(a[:3], b[:3])
    np.testing.assert_array_equal(b[3:], 0)


def test_training_neuron_selection_not_old47_selection():
    data = data_fixture()
    data['half1'][:39, :, 3] = 1
    data['half1'][39:, :, 4] = 0
    prep = rep.fit_preprocessing(data, np.arange(39), rep.configuration())
    assert not prep['keep'][3]
    assert prep['keep'][4]


def test_overlap_or_incomplete_splits_refused_before_any_fit(tmp_path):
    data = data_fixture()
    data.update(animal='test', metadata={'source_sha256': 'synthetic'})
    with pytest.raises(ValueError, match='disjoint'):
        rep.fit_round(data, np.arange(39), np.arange(38, 78), 0, tmp_path)
    with pytest.raises(ValueError, match='79 conditions'):
        rep.fit_round(data, np.arange(39), np.arange(40, 80), 0, tmp_path)


def test_actual_optimized_gpfa_matches_prefix_and_resets():
    data = data_fixture()
    cfg = rep.configuration()
    prep = rep.fit_preprocessing(data, np.arange(39), cfg)
    train = [rep.preprocess(x, prep) for x in data['half1'][:39]]
    fa = FactorAnalysis(n_components=3, random_state=42, max_iter=20).fit(np.concatenate(train))
    gpfa = rep.SharedTimescaleGPFA(3, max_iterations=3, seed=42)
    gpfa.fit(train, init_parameters={'C': fa.components_.T, 'd': fa.mean_, 'R': fa.noise_variance_})
    audit = rep.audit_prefix(fa, gpfa, data, prep, np.arange(39), np.arange(39, 79))
    assert audit['provided_input_filtering'] == 'pass'
    assert audit['fit_provenance'] == 'pass'
    assert audit['raw_preprocessing'] == 'fail'
    assert audit['max_error'] < 1e-8


def test_configuration_is_fixed_and_label_free():
    cfg = rep.configuration()
    assert cfg['latent_dim'] == 50
    assert cfg['fa_options'] == {'n_components': 50, 'random_state': 42, 'max_iter': 300,
                                 'tol': .01, 'svd_method': 'lapack', 'iterated_power': 7, 'rotation': None}
    assert cfg['gpfa_options']['max_iterations'] == 400
    assert cfg['gpfa_options']['tolerance'] == 1e-6
    import inspect
    assert 'behavior' not in inspect.signature(rep.fit_round).parameters
    assert 'endpoint' not in inspect.signature(rep.fit_round).parameters
