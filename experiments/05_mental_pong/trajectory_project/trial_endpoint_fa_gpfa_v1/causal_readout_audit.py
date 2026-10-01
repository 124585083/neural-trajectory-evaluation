"""Future/endpoint-independence checks using actual saved OLS weights.

The failed raw-preprocessing layer is never overwritten by a downstream pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

try:
    from .representations import SharedTimescaleGPFA, preprocess, save_json, sha256
except ImportError:
    from trajectory_project.trial_endpoint_fa_gpfa_v1.representations import SharedTimescaleGPFA, preprocess, save_json, sha256


def predict_saved(latent, coefficients, intercepts):
    """Frozen two-head OLS.  Targets, endpoints, time and IDs are not inputs."""
    z = np.asarray(latent, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    intercepts = np.asarray(intercepts, dtype=np.float64)
    if z.ndim != 2 or coefficients.shape != (2, 2, z.shape[-1]) or intercepts.shape != (2, 2):
        raise ValueError('Expected z[time,q], coefficients[2heads,2coords,q], intercepts[2heads,2coords]')
    return np.einsum('tq,hcq->htc', z, coefficients, optimize=True) + intercepts[:, None, :]


def _snapshot_endpoint_copy(payload, test_ids):
    """Change only real test-trial terminal labels; no model or neural edits."""
    altered = dict(payload)
    table = payload['trial_table']
    condition_ids = np.asarray(table['condition_id'], dtype=int)
    test = np.isin(condition_ids, test_ids)
    endpoint = np.asarray(payload['endpoint'] if 'endpoint' in payload else table['paddle_y'], dtype=np.float64)
    altered['endpoint'] = endpoint.copy()
    finite_test = test & np.isfinite(endpoint)
    altered['endpoint'][finite_test] = -endpoint[finite_test] + 123.456
    return altered, int(finite_test.sum())


def audit_saved_round(payload, data, representations_folder, readout_model_paths,
                      output_path, n_per_split=3, tolerance=1e-8):
    folder = Path(representations_folder)
    meta = json.loads((folder / 'training.json').read_text(encoding='utf-8'))
    source_audit = json.loads((folder / 'causal_audit.json').read_text(encoding='utf-8'))
    train_ids, test_ids = meta['fit_conditions'], meta['test_conditions']
    fit_pass = (len(train_ids) == 39 and len(test_ids) == 40 and
                not (set(train_ids) & set(test_ids)) and
                set(train_ids + test_ids) == set(map(int, data['condition_ids'])) and
                meta['fingerprint']['source_sha256'] == data['metadata']['source_sha256'])
    if not fit_pass:
        raise ValueError('Saved parameter provenance does not match39/40 source split')
    files = {'fa': folder / 'fa50.joblib', 'gpfa': folder / 'gpfa50.npz',
             'preprocessing': folder / 'preprocessing.npz'}
    files.update({f'readout_{k}': Path(v) for k, v in readout_model_paths.items()})
    before = {key: sha256(path) for key, path in files.items()}
    fa = joblib.load(files['fa'])
    gpfa = SharedTimescaleGPFA.load(files['gpfa'])
    with np.load(files['preprocessing'], allow_pickle=False) as stored:
        prep = {k: stored[k] for k in ('keep', 'mean', 'std', 'neuron_ids')}
        if set(map(int, stored['train_ids'])) != set(train_ids):
            raise ValueError('Scaler fitted-condition identity differs')
    readouts = {}
    for name in ('FA50', 'GPFA50'):
        with np.load(readout_model_paths[name], allow_pickle=False) as stored:
            saved_ids = stored['condition_ids']
            if (set(map(int, saved_ids[stored['train_condition_indices']])) != set(train_ids)
                    or set(map(int, saved_ids[stored['test_condition_indices']])) != set(test_ids)
                    or not np.array_equal(saved_ids, data['condition_ids'])):
                raise ValueError('Actual OLS condition provenance differs from representation split')
            readouts[name] = (stored['coefficients'].copy(), stored['intercepts'].copy())
    lookup = {int(c): i for i, c in enumerate(data['condition_ids'])}
    chosen = [(side, c) for side, ids in [('train', train_ids), ('test', test_ids)]
              for c in sorted(map(int, ids))[:n_per_split]]
    changed_payload, endpoint_changes = _snapshot_endpoint_copy(payload, test_ids)
    checks = []
    rng = np.random.default_rng(77031)
    with threadpool_limits(limits=4):
        for split, cid in chosen:
            ci = lookup[cid]
            valid = data['valid'][ci]
            for half in ('half1', 'half2'):
                raw = data[half][ci, valid]
                x = preprocess(raw, prep)
                k = len(x) // 2
                changed_raw = raw.copy()
                changed_raw[k + 1:] = rng.normal(size=changed_raw[k + 1:].shape) * 17
                changed_x = preprocess(changed_raw, prep)
                fa_z = fa.transform(x)
                gpfa_z = gpfa.infer_sequence(x, 'causal')[1]
                changed_z = {'FA50': fa.transform(changed_x),
                             'GPFA50': gpfa.infer_sequence(changed_x, 'causal')[1]}
                prefix_z = {'FA50': fa.transform(x[:k + 1])[-1],
                            'GPFA50': gpfa.infer_sequence(x[:k + 1], 'smooth')[1][-1]}
                other_i = (ci + 1) % len(data['condition_ids'])
                other_x = preprocess(data[half][other_i, data['valid'][other_i]], prep)
                gpfa.infer_sequence(other_x, 'causal')
                fa.transform(other_x)
                after_other = {'FA50': fa.transform(x), 'GPFA50': gpfa.infer_sequence(x, 'causal')[1]}
                # Final endpoints have been changed in the payload copy.  The
                # actual inference inputs remain the published neural arrays;
                # neither inference nor saved OLS APIs accept that payload.
                assert changed_payload['endpoint'] is not payload.get('endpoint')
                endpoint_changed_z = {'FA50': fa.transform(preprocess(raw, prep)),
                                      'GPFA50': gpfa.infer_sequence(preprocess(raw, prep), 'causal')[1]}
                for name, z in [('FA50', fa_z), ('GPFA50', gpfa_z)]:
                    coef, intercept = readouts[name]
                    prediction = predict_saved(z, coef, intercept)
                    pred_future = predict_saved(changed_z[name], coef, intercept)
                    pred_prefix = predict_saved(prefix_z[name][None], coef, intercept)[:, 0]
                    pred_other = predict_saved(after_other[name], coef, intercept)
                    pred_endpoint = predict_saved(endpoint_changed_z[name], coef, intercept)
                    row = {
                        'animal': data['animal'], 'round_id': meta['round_id'], 'representation': name,
                        'split': split, 'condition_id': cid, 'half': half, 'cutoff_index': k,
                        'preprocessing_future_error': float(np.max(np.abs(x[:k + 1] - changed_x[:k + 1]))),
                        'latent_future_error': float(np.max(np.abs(z[:k + 1] - changed_z[name][:k + 1]))),
                        'prediction_future_error': float(np.max(np.abs(prediction[:, :k + 1] - pred_future[:, :k + 1]))),
                        'latent_prefix_reference_error': float(np.max(np.abs(z[k] - prefix_z[name]))),
                        'prediction_prefix_reference_error': float(np.max(np.abs(prediction[:, k] - pred_prefix))),
                        'latent_next_trial_error': float(np.max(np.abs(z - after_other[name]))),
                        'prediction_next_trial_error': float(np.max(np.abs(prediction - pred_other))),
                        'latent_endpoint_change_error': float(np.max(np.abs(z - endpoint_changed_z[name]))),
                        'prediction_endpoint_change_error': float(np.max(np.abs(prediction - pred_endpoint))),
                    }
                    checks.append(row)
    max_latent = max(v for row in checks for key, v in row.items() if key.startswith('latent_'))
    max_prediction = max(v for row in checks for key, v in row.items() if key.startswith('prediction_'))
    max_preprocessing = max(row['preprocessing_future_error'] for row in checks)
    files_unchanged = all(sha256(path) == before[key] for key, path in files.items())
    passed = (max(max_latent, max_prediction, max_preprocessing) <= tolerance and files_unchanged
              and endpoint_changes > 0 and source_audit['provided_input_filtering'] == 'pass')
    result = {
        'animal': data['animal'], 'round_id': meta['round_id'],
        'raw_preprocessing': 'fail', 'provided_input_filtering': 'pass' if passed else 'fail',
        'fit_provenance': 'pass' if fit_pass else 'fail',
        'raw_preprocessing_reason': data['metadata']['raw_preprocessing_reason'],
        'max_latent_error': max_latent, 'max_prediction_error': max_prediction,
        'max_preprocessing_error': max_preprocessing, 'tolerance': tolerance,
        'n_changed_test_trial_endpoints': endpoint_changes,
        'endpoint_probe_scope': 'Actual endpoint payload copy changed; fixed inference/OLS APIs consume only released neural response, frozen training statistics and model weights. No readout refitting, and no endpoint is an input feature.',
        'parameter_files_unchanged': files_unchanged, 'files': {key: {'path': str(p), 'sha256': before[key]} for key, p in files.items()},
        'checks': checks, 'strict_raw_causality_passed': False,
    }
    save_json(output_path, result)
    if not passed:
        raise AssertionError(f'Provided-input fixed-readout causal audit failed: {output_path}')
    return result
