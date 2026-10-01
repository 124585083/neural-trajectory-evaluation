"""Validate every frozen split/model/prediction before sealing the delivery.

This checks identities, hashes, masks and algebra, never chooses scientific
settings from any reconstruction score.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from trajectory_project.trial_endpoint_fa_gpfa_v1.representations import load_neural_data, sha256, save_json
from trajectory_project.trial_endpoint_fa_gpfa_v1.causal_readout_audit import audit_saved_round, predict_saved


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _epoch_counts(payload):
    result = {}
    ci = np.asarray(payload['condition_index'], int)
    for epoch, mask in payload['epoch_masks'].items():
        m = np.asarray(mask, bool)
        seen = np.zeros((79, m.shape[1]), bool)
        np.logical_or.at(seen, ci, m)
        result[epoch] = {
            'bins': np.bincount(ci, weights=m.sum(1), minlength=79).astype(np.int64),
            'trials': np.bincount(ci, weights=m.any(1), minlength=79).astype(np.int64),
            'unique': seen.sum(1),
        }
    return result


def validate_delivery(data, root):
    root = Path(root)
    protocol_path = root / 'configs/analysis_protocol.json'
    protocol_hash = sha256(protocol_path)
    assert protocol_hash == _json(root / 'configs/protocol_lock.json')['sha256']
    protocol = _json(protocol_path)
    assert protocol['n_splits'] == 100
    splits = _json(root / 'configs/condition_splits_100.json')
    assert len(splits) == 100
    decoder_hash = sha256(root / 'decoding_trial.py')
    labels_hash = {animal: sha256(p['labels_path']) for animal, p in data.items()}
    counts = {animal: _epoch_counts(p) for animal, p in data.items()}
    file_digests = {}

    def verify(path, expected):
        path = Path(path).resolve()
        name = str(path)
        if name not in file_digests:
            file_digests[name] = sha256(path)
        if file_digests[name] != expected:
            raise AssertionError(f'Integrity mismatch: {path}')

    all_models, rows = set(), []
    maximum_x_error = maximum_replay_error = 0.
    for animal in ('mahler', 'perle'):
        payload = data[animal]
        ids = np.asarray(payload['condition_ids'], int)
        times = np.asarray(payload['times_ms'])
        ci = np.asarray(payload['condition_index'], int)
        common = np.asarray(payload['common_mask'], bool)
        expected_common_hash = hashlib.sha256(np.asarray(common.shape, dtype=np.int64).tobytes() + common.tobytes()).hexdigest()
        for iteration, split in enumerate(splits):
            folder = root / 'representations' / animal / f'round_{iteration:03d}'
            marker = _json(root / 'readouts/completed' / f'{animal}_round_{iteration:03d}.json')
            expected = {'protocol': protocol_hash, 'decoder': decoder_hash, 'labels': labels_hash[animal],
                        'representation': sha256(folder / 'training.json'), 'split': split}
            if marker['fingerprint'] != expected:
                raise AssertionError(f'Completed marker fingerprint changed: {animal} round{iteration}')
            assert marker['animal'] == animal and marker['iteration'] == iteration
            assert marker['common_row_count'] == int(common.sum())
            for item in marker['outputs']:
                verify(item['path'], item['sha256'])
            meta = _json(folder / 'training.json')
            assert set(meta['fit_conditions']) == set(split['train_condition_ids'])
            assert set(meta['test_conditions']) == set(split['test_condition_ids'])
            for name, digest in meta['file_hashes'].items():
                verify(folder / name, digest)
            latent_file = np.load(folder / 'latents.npz', allow_pickle=False)
            np.testing.assert_array_equal(latent_file['condition_ids'], ids)
            np.testing.assert_array_equal(latent_file['times_ms'], times)
            # Coverage and per-bin moments use label support. This proves it is
            # exactly the common representation/label support in every round.
            assert not np.any(common & ~latent_file['common_mask'][ci])
            tr, te = np.asarray(split['train_indices'], int), np.asarray(split['test_indices'], int)
            assert len(tr) == 39 and len(te) == 40 and not set(tr) & set(te)
            np.testing.assert_array_equal(ids[tr], split['train_condition_ids'])
            np.testing.assert_array_equal(ids[te], split['test_condition_ids'])
            test_trials = np.flatnonzero(np.isin(ci, te))
            train_rows = int(common[np.isin(ci, tr)].sum())
            for representation in ('FA50', 'GPFA50'):
                prefix = f'{animal}_{representation}_r{iteration:03d}'
                model_path = root / 'readouts/models' / f'{prefix}_ols.npz'
                prediction_path = root / 'readouts/predictions' / f'{prefix}_test_predictions.npz'
                all_models.add(str(model_path.resolve()))
                with np.load(model_path, allow_pickle=False) as model:
                    assert model['coefficients'].shape == (2, 2, 50)
                    assert model['intercepts'].shape == (2, 2)
                    np.testing.assert_array_equal(model['condition_ids'], ids)
                    np.testing.assert_array_equal(model['train_condition_indices'], tr)
                    np.testing.assert_array_equal(model['test_condition_indices'], te)
                    assert int(model['n_train_rows']) == train_rows
                    assert int(model['expanded_design_rank']) == int(model['unique_design_rank'])
                    coefficient = model['coefficients'].copy()
                    intercept = model['intercepts'].copy()
                x_error = max(float(np.max(np.abs(coefficient[0, 0] - coefficient[1, 0]))),
                              float(abs(intercept[0, 0] - intercept[1, 0])))
                assert x_error < 1e-10
                with np.load(prediction_path, allow_pickle=False) as pred:
                    np.testing.assert_array_equal(pred['condition_ids'], ids)
                    np.testing.assert_array_equal(pred['times_ms'], times)
                    np.testing.assert_array_equal(pred['test_condition_indices'], te)
                    np.testing.assert_array_equal(pred['test_trial_indices'], test_trials)
                    stored_common = np.unpackbits(pred['test_common_mask_packed'], axis=1)[:, :len(times)].astype(bool)
                    np.testing.assert_array_equal(stored_common, common[test_trials])
                    assert str(pred['common_mask_sha256']) == expected_common_hash
                    support = pred['test_input_support']
                    expected_support = np.zeros((79, len(times)), bool)
                    np.logical_or.at(expected_support, ci[test_trials], common[test_trials])
                    np.testing.assert_array_equal(support, expected_support)
                    prediction = pred['predictions']
                    assert prediction.shape == (2, 79, len(times), 2)
                    assert not np.isfinite(prediction[:, tr]).any()
                    assert not np.isfinite(prediction[:, ~support]).any()
                    assert np.isfinite(prediction[:, support]).all()
                    z = latent_file[representation][0]
                    assert np.isfinite(z[support]).all()
                    with threadpool_limits(limits=2):
                        replay = predict_saved(z[support], coefficient, intercept)
                    replay_error = float(np.max(np.abs(replay - prediction[:, support])))
                    x_prediction_error = float(np.max(np.abs(prediction[0, support, 0] - prediction[1, support, 0])))
                    assert replay_error < 1e-8 and x_prediction_error < 1e-10
                # Verify phase identities/counts without inspecting r/RMSE.
                identity_columns = ['epoch', 'coordinate', 'n_conditions', 'n_real_trials',
                                    'n_neural_trials', 'n_bins', 'n_unique_neural_input_ids']
                score_identity = pd.read_csv(root / 'readouts/round_results' / f'{prefix}_self.csv', usecols=identity_columns)
                for row in score_identity.itertuples(index=False):
                    count = counts[animal][row.epoch]
                    assert row.n_bins == int(count['bins'][te].sum())
                    assert row.n_real_trials == int(count['trials'][te].sum())
                    assert row.n_conditions == int(np.sum(count['bins'][te] > 0))
                    assert row.n_unique_neural_input_ids == int(count['unique'][te].sum())
                    assert row.n_neural_trials == 0
                maximum_x_error = max(maximum_x_error, x_error, x_prediction_error)
                maximum_replay_error = max(maximum_replay_error, replay_error)
            latent_file.close()
            rows.append({'animal': animal, 'iteration': iteration, 'marker_fingerprint_verified': True,
                         'source_and_output_hashes_verified': True, 'same_common_support_all_four_groups': True,
                         'all_epoch_trial_bin_counts_verified': True, 'training_predictions_excluded': True})
        print(f'Final integrity: {animal}100 rounds checked', flush=True)
    actual_models = {str(p.resolve()) for p in (root / 'readouts/models').glob('*_ols.npz')}
    assert actual_models == all_models and len(all_models) == 400
    causal_results = []
    for animal in ('mahler', 'perle'):
        neural = load_neural_data(animal, cache_root=root / 'artifacts')
        for iteration in (0, 99):
            paths = {rep: root / 'readouts/models' / f'{animal}_{rep}_r{iteration:03d}_ols.npz'
                     for rep in ('FA50', 'GPFA50')}
            output = root / 'results' / f'{animal}_r{iteration:03d}_causal_readout_audit.json'
            audit = audit_saved_round(data[animal], neural,
                                      root / 'representations' / animal / f'round_{iteration:03d}', paths, output)
            causal_results.append({'animal': animal, 'iteration': iteration, 'path': str(output.resolve()),
                                   'sha256': sha256(output), 'max_latent_error': audit['max_latent_error'],
                                   'max_prediction_error': audit['max_prediction_error'],
                                   'provided_input_filtering': audit['provided_input_filtering'],
                                   'fit_provenance': audit['fit_provenance'], 'raw_preprocessing': audit['raw_preprocessing']})
        del neural
    assert sha256(protocol_path) == protocol_hash
    assert sha256(root / 'decoding_trial.py') == decoder_hash
    for animal, payload in data.items():
        assert sha256(payload['labels_path']) == labels_hash[animal]
    output = {'completed_animal_rounds': len(rows), 'saved_dual_head_OLS_models': len(all_models),
              'xy_OLS_heads': len(all_models) * 2, 'coordinates_per_head': 2, 'features_per_head': 50,
              'hash_verified_file_count': len(file_digests), 'all200_marker_fingerprints_verified': True,
              'all_four_groups_share_same_trial_time_support': True,
              'all_epochs_trial_counts_and_bin_counts_verified': True,
              'all_training_predictions_excluded': True,
              'maximum_same_representation_x_difference': maximum_x_error,
              'maximum_saved_weight_prediction_replay_error': maximum_replay_error,
              'round0_and_99_actual_weight_causal_audits': causal_results,
              'raw_preprocessing': 'fail', 'strict_raw_future_free_analysis_completed': False,
              'data_mode': 'trial_labels_with_mean_neural', 'round_checks': rows}
    save_json(root / 'results/final_integrity_and_causal_audit.json', output)
    return output
