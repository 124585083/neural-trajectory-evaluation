from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

from trajectory_project.trial_endpoint_fa_gpfa_v1.representations import SharedTimescaleGPFA, save_json
from trajectory_project.trial_endpoint_fa_gpfa_v1.causal_readout_audit import audit_saved_round, predict_saved, _snapshot_endpoint_copy


def test_saved_predict_has_exact_two_independent_linear_heads():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(7, 50))
    coef = rng.normal(size=(2, 2, 50))
    intercept = rng.normal(size=(2, 2))
    pred = predict_saved(x, coef, intercept)
    for h in range(2):
        np.testing.assert_allclose(pred[h], x @ coef[h].T + intercept[h], atol=1e-12)


def test_endpoint_copy_changes_test_only_and_preserves_original():
    payload = {'endpoint': np.array([1., 2., np.nan]),
               'trial_table': pd.DataFrame({'condition_id': [4, 3, 4]})}
    copy, n = _snapshot_endpoint_copy(payload, [4])
    assert n == 1
    np.testing.assert_equal(payload['endpoint'], [1., 2., np.nan])
    assert copy['endpoint'][0] != 1.
    assert copy['endpoint'][1] == 2.


def test_actual_saved_models_and_readouts_pass_three_layer_audit(tmp_path):
    rng = np.random.default_rng(28)
    values = rng.normal(size=(79, 7, 60))
    fa = FactorAnalysis(n_components=50, max_iter=5, random_state=42).fit(values[:39].reshape(-1, 60))
    gpfa = SharedTimescaleGPFA(50)
    gpfa._set_parameters(fa.components_.T, fa.mean_, fa.noise_variance_, .15)
    joblib.dump(fa, tmp_path / 'fa50.joblib')
    gpfa.save(tmp_path / 'gpfa50.npz')
    np.savez(tmp_path / 'preprocessing.npz', keep=np.ones(60, bool), mean=np.zeros(60),
             std=np.ones(60), neuron_ids=np.arange(60), train_ids=np.arange(39))
    save_json(tmp_path / 'training.json', {'fit_conditions': list(range(39)), 'test_conditions': list(range(39, 79)),
             'round_id': 0, 'fingerprint': {'source_sha256': 'synthetic'}})
    save_json(tmp_path / 'causal_audit.json', {'provided_input_filtering': 'pass'})
    paths = {}
    for name in ('FA50', 'GPFA50'):
        paths[name] = tmp_path / f'{name}_ols.npz'
        np.savez(paths[name], coefficients=rng.normal(size=(2, 2, 50)), intercepts=rng.normal(size=(2, 2)),
                 condition_ids=np.arange(79), train_condition_indices=np.arange(39),
                 test_condition_indices=np.arange(39, 79))
    data = {'animal': 'synthetic', 'condition_ids': np.arange(79), 'half1': values,
            'half2': values * 1.1, 'valid': np.ones((79, 7), bool),
            'metadata': {'source_sha256': 'synthetic', 'raw_preprocessing_reason': 'known original all-data imputation'}}
    payload = {'endpoint': np.array([1., 2.]), 'trial_table': pd.DataFrame({'condition_id': [40, 1]})}
    result = audit_saved_round(payload, data, tmp_path, paths, tmp_path / 'audit.json')
    assert result['provided_input_filtering'] == 'pass'
    assert result['fit_provenance'] == 'pass'
    assert result['raw_preprocessing'] == 'fail'
    assert result['max_prediction_error'] < 1e-8
    assert result['n_changed_test_trial_endpoints'] == 1
    assert result['parameter_files_unchanged']
