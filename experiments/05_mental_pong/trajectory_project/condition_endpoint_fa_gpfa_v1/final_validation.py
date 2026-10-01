"""Read-only checks of new condition-time OLS output and unchanged provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .reuse_audit import read_json, sha256, write_json


def _array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + np.asarray(a.shape, dtype=np.int64).tobytes() + a.tobytes()).hexdigest()


def _self_metrics(y, prediction):
    if len(y) == 0:
        return np.full(2, np.nan), np.full(2, np.nan)
    rmse = np.sqrt(np.mean(np.square(prediction - y), axis=0))
    r = np.full(2, np.nan)
    for j in range(2):
        if len(y) >= 3 and np.ptp(y[:, j]) > 0 and np.ptp(prediction[:, j]) > 0:
            centered_y = y[:, j] - y[:, j].mean()
            centered_p = prediction[:, j] - prediction[:, j].mean()
            norm = np.linalg.norm(centered_y) * np.linalg.norm(centered_p)
            if norm > 0:
                r[j] = np.clip(np.dot(centered_y, centered_p) / norm, -1., 1.)
    return r, rmse


def validate_delivery(data, root):
    root = Path(root)
    protocol_path = root / 'configs/analysis_protocol.json'
    protocol_hash = sha256(protocol_path)
    assert protocol_hash == read_json(root / 'configs/protocol_lock.json')['sha256']
    splits = read_json(root / 'configs/condition_splits_100.json')
    assert len(splits) == 100
    reuse_path = root / 'results/reused_representation_audit.json'
    reuse = read_json(reuse_path)
    source = Path(reuse['source_root'])
    assert reuse['representation_animal_rounds_verified'] == 200
    assert reuse['new_representation_fits'] == 0
    assert reuse['raw_preprocessing'] == 'fail'
    assert splits == read_json(source / 'configs/condition_splits_100.json')
    old_records = {str(Path(r['path']).resolve()): r['sha256'] for r in reuse['source_records']}
    decoder_hash = sha256(root / 'decoding_condition.py')
    label_hashes = {animal: sha256(p['labels_path']) for animal, p in data.items()}
    validated_outputs = {}

    def check_file(path, expected):
        name = str(Path(path).resolve())
        if name not in validated_outputs:
            validated_outputs[name] = sha256(name)
        assert validated_outputs[name] == expected, f'File hash differs:{name}'

    models, rows = set(), []
    maximum_prediction_error = maximum_x_error = maximum_score_error = 0.
    for animal in ('mahler', 'perle'):
        payload = data[animal]
        ids, times = payload['condition_ids'], payload['times_ms']
        common = np.asarray(payload['common_mask'], bool)
        common_hash = _array_sha(common)
        objective, behavior = payload['objective_xy'], payload['behavior_xy']
        nbehavior = payload['n_behavior_trials']
        assert np.array_equal(objective[common, 0], behavior[common, 0])
        for iteration, split in enumerate(splits):
            source_folder = source / 'representations' / animal / f'round_{iteration:03d}'
            training_path = source_folder / 'training.json'
            source_training_hash = sha256(training_path)
            assert source_training_hash == old_records[str(training_path.resolve())]
            marker = read_json(root / 'readouts/completed' / f'{animal}_round_{iteration:03d}.json')
            expected_fingerprint = {'protocol': protocol_hash, 'decoder': decoder_hash,
                                    'labels': label_hashes[animal], 'representation': source_training_hash,
                                    'split': split}
            assert marker['fingerprint'] == expected_fingerprint
            assert marker['animal'] == animal and marker['iteration'] == iteration
            for record in marker['outputs']:
                check_file(record['path'], record['sha256'])
            train, test = np.asarray(split['train_indices'], int), np.asarray(split['test_indices'], int)
            assert len(train) == 39 and len(test) == 40 and not (set(train) & set(test))
            np.testing.assert_array_equal(ids[train], split['train_condition_ids'])
            np.testing.assert_array_equal(ids[test], split['test_condition_ids'])
            train_mask = common.copy(); train_mask[test] = False
            test_mask = common.copy(); test_mask[train] = False
            training_indices = np.argwhere(train_mask)
            with np.load(source_folder / 'latents.npz', allow_pickle=False) as latent:
                np.testing.assert_array_equal(latent['condition_ids'], ids)
                np.testing.assert_array_equal(latent['times_ms'], times)
                assert not np.any(common & ~latent['common_mask'])
                for representation in ('FA50', 'GPFA50'):
                    prefix = f'{animal}_{representation}_r{iteration:03d}'
                    model_path = root / 'readouts/models' / f'{prefix}_ols.npz'
                    prediction_path = root / 'readouts/predictions' / f'{prefix}_test_predictions.npz'
                    models.add(str(model_path.resolve()))
                    with np.load(model_path, allow_pickle=False) as model:
                        coefficient, intercept = model['coefficients'], model['intercepts']
                        assert coefficient.shape == (2, 2, 50) and intercept.shape == (2, 2)
                        assert int(model['n_train_rows']) == int(train_mask.sum())
                        np.testing.assert_array_equal(model['training_condition_bin_indices'], training_indices)
                        unique = model['unique_neural_input_ids']
                        assert len(unique) == len(np.unique(unique)) == len(training_indices)
                        np.testing.assert_array_equal(unique, training_indices[:, 0] * len(times) + training_indices[:, 1])
                        np.testing.assert_array_equal(model['condition_ids'], ids)
                        np.testing.assert_array_equal(model['train_condition_indices'], train)
                        np.testing.assert_array_equal(model['test_condition_indices'], test)
                        assert str(model['common_mask_sha256']) == common_hash
                        assert Path(str(model['source_representation_directory'])).resolve() == source_folder.resolve()
                    x_error = max(float(np.max(np.abs(coefficient[0, 0] - coefficient[1, 0]))),
                                  float(abs(intercept[0, 0] - intercept[1, 0])))
                    assert x_error < 1e-10
                    with np.load(prediction_path, allow_pickle=False) as predicted:
                        np.testing.assert_array_equal(predicted['common_mask'], common)
                        np.testing.assert_array_equal(predicted['test_input_support'], test_mask)
                        np.testing.assert_array_equal(predicted['condition_ids'], ids)
                        np.testing.assert_array_equal(predicted['times_ms'], times)
                        np.testing.assert_array_equal(predicted['test_condition_indices'], test)
                        np.testing.assert_array_equal(predicted['n_behavior_trials_in_means'], nbehavior)
                        prediction = predicted['predictions']
                        assert prediction.shape == (2, 79, len(times), 2)
                        assert not np.isfinite(prediction[:, train]).any()
                        assert not np.isfinite(prediction[:, ~test_mask]).any()
                        assert np.isfinite(prediction[:, test_mask]).all()
                    z = latent[representation][0]
                    assert z.shape == (79, len(times), 50)
                    assert np.isfinite(z[common]).all()
                    with threadpool_limits(limits=2):
                        replay = np.stack([z[test_mask] @ coefficient[h].T + intercept[h] for h in range(2)])
                    replay_error = float(np.max(np.abs(replay - prediction[:, test_mask])))
                    x_error = max(x_error, float(np.max(np.abs(prediction[0, test_mask, 0] - prediction[1, test_mask, 0]))))
                    assert replay_error < 1e-8 and x_error < 1e-10
                    own = pd.read_csv(root / 'readouts/round_results' / f'{prefix}_self.csv').set_index(['epoch', 'coordinate'])
                    cross = pd.read_csv(root / 'readouts/round_results' / f'{prefix}_cross_2x2.csv').set_index(['epoch', 'coordinate', 'head', 'target'])
                    assert len(own) == 12 and len(cross) == 48 and own.index.is_unique and cross.index.is_unique
                    for epoch, epoch_mask in payload['epoch_masks'].items():
                        selected = np.asarray(epoch_mask, bool) & test_mask
                        contributes = selected.any(axis=1)
                        n_bins, n_conditions = int(selected.sum()), int(contributes.sum())
                        n_source_trials = int(nbehavior[contributes].sum())
                        checks = {}
                        for h, head in enumerate(('D_obj', 'D_beh')):
                            for yi, (target_name, y) in enumerate((('objective', objective), ('behavior', behavior))):
                                pearson, rmse = _self_metrics(y[selected], prediction[h, selected])
                                checks[h, yi] = (pearson, rmse)
                                for coordinate, axis in enumerate(('x', 'y')):
                                    row = cross.loc[(epoch, axis, head, target_name)]
                                    expected_scores = np.asarray([pearson[coordinate], rmse[coordinate]])
                                    actual_scores = row[['r', 'RMSE']].to_numpy(float)
                                    np.testing.assert_allclose(actual_scores, expected_scores, atol=1e-10, rtol=1e-12, equal_nan=True)
                                    finite = np.isfinite(expected_scores)
                                    if finite.any():
                                        maximum_score_error = max(maximum_score_error, float(np.max(np.abs(actual_scores[finite] - expected_scores[finite]))))
                                    assert row.n_bins == n_bins and row.n_conditions == n_conditions
                                    assert row.n_behavior_trials_in_means == n_source_trials and row.n_neural_trials == 0
                        for coordinate, axis in enumerate(('x', 'y')):
                            row = own.loc[(epoch, axis)]
                            oo, bb = checks[0, 0], checks[1, 1]
                            wanted = np.asarray([oo[0][coordinate], bb[0][coordinate], oo[1][coordinate], bb[1][coordinate],
                                                 bb[0][coordinate] - oo[0][coordinate], oo[1][coordinate] - bb[1][coordinate]])
                            np.testing.assert_allclose(row[['r_obj', 'r_beh', 'RMSE_obj', 'RMSE_beh', 'Delta_r', 'Delta_RMSE']].to_numpy(float),
                                                       wanted, atol=1e-10, rtol=1e-12, equal_nan=True)
                            assert row.n_bins == row.n_condition_time_rows == row.n_unique_neural_input_ids == n_bins
                            assert row.n_conditions == n_conditions and row.n_neural_trials == 0
                            assert row.n_behavior_trials_in_means == n_source_trials
                    maximum_prediction_error = max(maximum_prediction_error, replay_error)
                    maximum_x_error = max(maximum_x_error, x_error)
            fit = pd.read_csv(root / 'readouts/round_results' / f'{animal}_r{iteration:03d}_fit_audit.csv')
            assert len(fit) == 2 and set(fit.representation) == {'FA50', 'GPFA50'}
            assert fit.representation_refitted.eq(False).all()
            assert fit.behavior_counts_used_as_weights.eq(False).all()
            assert fit.duplicate_condition_time_rows.eq(0).all()
            assert fit.n_training_rows.eq(int(train_mask.sum())).all()
            assert fit.sample_weights.eq('none').all() and fit.scaler.eq('none').all()
            assert fit.provided_input_filtering.eq('pass').all() and fit.fit_provenance.eq('pass').all()
            assert fit.raw_preprocessing.eq('fail').all()
            rows.append({'animal': animal, 'iteration': iteration, 'fingerprint_and_output_hashes': 'pass',
                         'source_training_record_unchanged': True, 'original_condition_time_rows_no_repetition': True,
                         'same_four_group_support': True, 'heldout_predictions_only': True,
                         'all_phase_counts': 'pass', 'all_2x2_and_own_scores_independently_recomputed': True})
        print(f'Condition final validation: {animal}100 rounds passed', flush=True)
    assert len(rows) == 200 and len(models) == 400
    assert models == {str(p.resolve()) for p in (root / 'readouts/models').glob('*_ols.npz')}
    assert sha256(protocol_path) == protocol_hash and sha256(root / 'decoding_condition.py') == decoder_hash
    for animal, payload in data.items():
        assert sha256(payload['labels_path']) == label_hashes[animal]
    result = {'completed_animal_rounds': 200, 'dual_head_OLS_files': 400, 'xy_OLS_heads': 800,
              'retrained_representation_models': 0, 'verified_new_output_hashes': len(validated_outputs),
              'source_representation_audit': {'path': str(reuse_path.resolve()), 'sha256': sha256(reuse_path)},
              'source_representation_weights_and_latents': 'unchanged; complete reuse audit cited, whole prior manifest rechecked by final seal',
              'max_saved_prediction_replay_error': maximum_prediction_error,
              'max_same_representation_x_error': maximum_x_error,
              'max_independent_metric_recomputation_error': maximum_score_error,
              'own_scores_match_OO_BB_not_cross_columns': True, 'delta_signs_verified': True,
              'all_100_splits_same39_40': True, 'original_condition_time_rows_no_repetition_or_trial_weights': True,
              'raw_preprocessing': 'fail', 'provided_input_filtering': 'pass_from_byte_identical_source',
              'fit_provenance': 'pass', 'strict_raw_future_free': False,
              'data_mode': 'condition_mean_neural_and_behavior', 'round_checks': rows}
    write_json(root / 'results/final_integrity_and_score_audit.json', result)
    return result
