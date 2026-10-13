"""Independent audits of unchanged representations and exact trial averaging.

No estimator is fitted here.  Readout split identities remain the representation
training split identities; a new supervision aggregation cannot repair known
upstream raw-response imputation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def _member_lookup(mapping, condition_id):
    if condition_id in mapping:
        return np.asarray(mapping[condition_id], dtype=int)
    if str(condition_id) in mapping:
        return np.asarray(mapping[str(condition_id)], dtype=int)
    raise AssertionError(f'Missing source trial membership for condition{condition_id}')


def validate_condition_averaging(trial_payload, average_payload, output_path=None,
                                tolerance=1e-10):
    """Mean the SAME valid trials; reject silent variable-membership averaging."""
    np.testing.assert_array_equal(average_payload['condition_ids'], trial_payload['condition_ids'])
    np.testing.assert_array_equal(average_payload['times_ms'], trial_payload['times_ms'])
    source_obj = np.asarray(trial_payload['objective_xy'], float)
    obj = np.asarray(average_payload['objective_xy'], float)
    behavior = np.asarray(average_payload['behavior_xy'], float)
    np.testing.assert_array_equal(obj, source_obj)
    if obj.shape != behavior.shape:
        raise AssertionError('Average objective/behavior arrays must share condition×time×xy shape')
    source_mask = np.asarray(trial_payload['common_mask'], bool)
    source_ci = np.asarray(trial_payload['condition_index'], int)
    source_endpoint = np.asarray(trial_payload['endpoint'], float)
    candidate = np.asarray(trial_payload['behavior_xy'], float)
    common = np.asarray(average_payload['common_mask'], bool)
    mean_endpoint = np.asarray(average_payload['mean_endpoint'], float)
    if common.shape != obj.shape[:2]:
        raise AssertionError('Average support is condition×time, not repeated trial×time')
    if set(trial_payload['epoch_masks']) != set(average_payload['epoch_masks']):
        raise AssertionError('Epoch definitions changed while averaging')
    geometry = average_payload['geometry'].set_index('condition_index')
    source_geometry = trial_payload['geometry'].set_index('condition_index')
    physical_fields = ['condition_id', 'metadata_bounce_count', 'x0', 'y0', 'x_end',
                       'collision_x', 'collision_y', 'collision_time_ms', 'T_ms']
    for field in physical_fields:
        if field in source_geometry.columns:
            np.testing.assert_allclose(geometry[field].to_numpy(float),
                                       source_geometry.loc[geometry.index, field].to_numpy(float),
                                       rtol=0, atol=0, equal_nan=True,
                                       err_msg=f'Physical anchor/time changed:{field}')
    rows = []
    maximum_mean_error = maximum_formula_error = maximum_endpoint_error = 0.
    valid_trials = source_mask.any(axis=1)
    for ci, cid in enumerate(np.asarray(trial_payload['condition_ids'], int)):
        expected_indices = np.flatnonzero((source_ci == ci) & valid_trials)
        member_indices = _member_lookup(average_payload['source_trial_indices_by_condition'], int(cid))
        # Ordered membership is itself part of reproducibility and traceability.
        np.testing.assert_array_equal(member_indices, expected_indices)
        if int(average_payload['n_behavior_trials'][ci]) != len(expected_indices):
            raise AssertionError('Mean label source-trial count is not the old valid-trial count')
        if not len(expected_indices):
            if common[ci].any() or np.isfinite(behavior[ci]).any() or np.isfinite(mean_endpoint[ci]):
                raise AssertionError('No old valid trials: mean behavior/endpoint must remain unknown')
            for epoch in average_payload['epoch_masks'].values():
                if np.asarray(epoch, bool)[ci].any():
                    raise AssertionError('Unknown mean candidate cannot acquire epoch support')
            rows.append({'condition_id': int(cid), 'n_source_trials': 0, 'n_bins': 0,
                         'status': 'retained_identity_no_valid_candidate'})
            continue
        masks = source_mask[expected_indices]
        if not np.all(masks == masks[0]):
            raise AssertionError('Variable trial membership across bins requires an explicit separate rule; silent nanmean rejected')
        np.testing.assert_array_equal(common[ci], masks[0])
        for epoch, source in trial_payload['epoch_masks'].items():
            source_epoch = np.asarray(source, bool)[expected_indices]
            if not np.all(source_epoch == source_epoch[0]):
                raise AssertionError(f'Variable per-trial support within epoch{epoch}')
            np.testing.assert_array_equal(np.asarray(average_payload['epoch_masks'][epoch], bool)[ci], source_epoch[0])
        endpoints = source_endpoint[expected_indices]
        if not np.isfinite(endpoints).all():
            raise AssertionError('An unknown endpoint entered the mean of old valid trials')
        endpoint_error = float(abs(float(endpoints.mean()) - mean_endpoint[ci]))
        if endpoint_error > tolerance:
            raise AssertionError('Average endpoint changed membership or weights')
        mask = common[ci]
        original_y = candidate[expected_indices][:, mask, 1]
        if not np.isfinite(original_y).all():
            raise AssertionError('Old valid support contains missing individual candidate labels')
        mean_error = float(np.max(np.abs(original_y.mean(axis=0) - behavior[ci, mask, 1])))
        if mean_error > tolerance:
            raise AssertionError('Mean candidate differs from same-trial original-label mean')
        if not np.array_equal(behavior[ci, mask, 0], obj[ci, mask, 0]):
            raise AssertionError('Average objective and behavior x must remain bit-identical')
        g = geometry.loc[ci]
        x = obj[ci, mask, 0]
        if int(g['metadata_bounce_count']) > 0:
            ax, ay = float(g['collision_x']), float(g['collision_y'])
            change = x >= ax
        else:
            ax, ay = float(g['x0']), float(g['y0'])
            change = np.ones(len(x), bool)
        x_end = float(g['x_end'])
        if not np.isfinite([ax, ay, x_end]).all() or x_end <= ax:
            raise AssertionError('No legal source physical anchor for endpoint averaging identity')
        formula = obj[ci, mask, 1].copy()
        alpha = (x[change] - ax) / (x_end - ax)
        formula[change] = ay + alpha * (mean_endpoint[ci] - ay)
        formula_error = float(np.max(np.abs(formula - behavior[ci, mask, 1])))
        if formula_error > tolerance:
            raise AssertionError('Mean endpoint construction is not equal to mean original candidate trajectory')
        maximum_mean_error = max(maximum_mean_error, mean_error)
        maximum_endpoint_error = max(maximum_endpoint_error, endpoint_error)
        maximum_formula_error = max(maximum_formula_error, formula_error)
        rows.append({'condition_id': int(cid), 'n_source_trials': len(expected_indices),
                     'n_bins': int(mask.sum()), 'status': 'same_valid_trials_constant_support_verified',
                     'endpoint_mean_max_error': endpoint_error, 'candidate_mean_max_error': mean_error,
                     'mean_endpoint_formula_max_error': formula_error})
    result = {'animal': str(average_payload.get('animal', trial_payload.get('animal', 'unknown'))),
              'same_old_valid_trial_membership': True, 'membership_is_constant_across_valid_bins': True,
              'all_old_condition_identities_retained': True, 'unknown_candidates_not_filled': True,
              'mean_of_candidates_equals_candidate_from_mean_endpoint': True,
              'maximum_candidate_mean_error': maximum_mean_error,
              'maximum_mean_endpoint_formula_error': maximum_formula_error,
              'maximum_endpoint_mean_error': maximum_endpoint_error, 'tolerance': tolerance,
              'n_source_behavior_trials': int(valid_trials.sum()),
              'n_valid_conditions': int(sum(row['n_source_trials'] > 0 for row in rows)),
              'n_condition_time_rows': int(common.sum()),
              'averaging_weights': 'each retained real trial once; pooled across its source sessions within condition',
              'interpretation': 'Condition-mean supervision changes OLS row weighting and removes within-condition behavior variance from the scored target; it does not add trial-specific neural information.',
              'conditions': rows}
    if output_path is not None:
        write_json(output_path, result)
    return result


def validate_reused_representations(source_root, new_root, splits):
    """Read-only verification; reuse source files by path with exact hashes."""
    source_root, new_root = Path(source_root), Path(new_root)
    if source_root.resolve() == new_root.resolve():
        raise ValueError('New analysis must not write into historical analysis')
    original_splits = read_json(source_root / 'configs/condition_splits_100.json')
    if splits != original_splits:
        raise AssertionError('Reused representation and new readout splits differ')
    if len(splits) != 100:
        raise AssertionError('Expected the original100 condition splits')
    lock_path = source_root / 'configs/protocol_lock.json'
    protocol_path = source_root / 'configs/analysis_protocol.json'
    protocol_hash = sha256(protocol_path)
    if protocol_hash != read_json(lock_path)['sha256']:
        raise AssertionError('Historical frozen protocol changed')
    protocol = read_json(protocol_path)
    if protocol['n_components'] != 50 or protocol['fit_neural_half'] != 'half1':
        raise AssertionError('Historical representations do not match this followup')
    frozen_configuration = read_json(source_root / 'configs/representation_config.json')
    module_hash = sha256(source_root / 'representations.py')
    records, rows = [], []
    maximum_prefix_error = 0.
    for animal in ('mahler', 'perle'):
        reference_ids = reference_times = None
        source_hashes = set()
        for iteration, split in enumerate(splits):
            folder = source_root / 'representations' / animal / f'round_{iteration:03d}'
            path = folder / 'training.json'
            metadata = read_json(path)
            if metadata['animal'] != animal or metadata['round_id'] != iteration:
                raise AssertionError('Representation animal/round identity changed')
            fingerprint = metadata['fingerprint']
            if fingerprint['configuration'] != frozen_configuration or fingerprint['module_sha256'] != module_hash:
                raise AssertionError('Historical representation implementation or frozen settings differ')
            if set(metadata['fit_conditions']) != set(split['train_condition_ids']):
                raise AssertionError('Reused model was not fit on this readout train39')
            if set(metadata['test_conditions']) != set(split['test_condition_ids']):
                raise AssertionError('Historical representation test40 differs')
            if (len(metadata['fit_conditions']) != 39 or len(metadata['test_conditions']) != 40
                    or set(metadata['fit_conditions']) & set(metadata['test_conditions'])):
                raise AssertionError('Historical condition grouping violated')
            if metadata['latent_dimensions'] != 50 or metadata['fit_half'] != 'half1':
                raise AssertionError('Reused dimension/half differs')
            if metadata['OLS_coefficient_count_xy'] != 102:
                raise AssertionError('Expected50 inputs plus intercept for each coordinate')
            if fingerprint['source_sha256'] != metadata['upstream']['source_sha256']:
                raise AssertionError('Historical input provenance hash mismatch')
            source_hashes.add(fingerprint['source_sha256'])
            records.append({'path': str(path.resolve()), 'sha256': sha256(path),
                            'purpose': 'unchanged source representation training/provenance'})
            for name, expected_hash in metadata['file_hashes'].items():
                file_path = folder / name
                actual_hash = sha256(file_path)
                if actual_hash != expected_hash:
                    raise AssertionError(f'Historical representation artifact changed: {file_path}')
                records.append({'path': str(file_path.resolve()), 'sha256': actual_hash,
                                'purpose': 'byte-identical reused representation artifact'})
            causal = read_json(folder / 'causal_audit.json')
            if causal['provided_input_filtering'] != 'pass' or causal['fit_provenance'] != 'pass':
                raise AssertionError('A source representation failed its local causal/provenance audit')
            if causal['raw_preprocessing'] != 'fail':
                raise AssertionError('Known upstream preprocessing limitation must not be relabeled')
            maximum_prefix_error = max(maximum_prefix_error, causal['max_error'])
            with np.load(folder / 'latents.npz', allow_pickle=False) as latent:
                ids, times = latent['condition_ids'], latent['times_ms']
                if reference_ids is None:
                    reference_ids, reference_times = ids.copy(), times.copy()
                else:
                    np.testing.assert_array_equal(ids, reference_ids)
                    np.testing.assert_array_equal(times, reference_times)
                if len(ids) != 79 or len(np.unique(ids)) != 79:
                    raise AssertionError('Historical all79 identity changed')
                np.testing.assert_array_equal(ids[np.asarray(split['train_indices'], int)], split['train_condition_ids'])
                np.testing.assert_array_equal(ids[np.asarray(split['test_indices'], int)], split['test_condition_ids'])
                for name in ('FA50', 'GPFA50'):
                    if latent[name].shape != (2, 79, len(times), 50):
                        raise AssertionError('Historical half/condition/time/dimension axes changed')
                    if not np.isfinite(latent[name][:, latent['common_mask']]).all():
                        raise AssertionError('Reused common support contains nonfinite latents')
            rows.append({'animal': animal, 'iteration': iteration, 'representation_dimensions': 50,
                         'n_train_conditions': 39, 'n_test_conditions': 40,
                         'source_folder': str(folder.resolve()), 'weights_and_latents_byte_identical': True,
                         'provided_input_filtering': 'pass', 'fit_provenance': 'pass',
                         'raw_preprocessing': 'fail'})
        if len(source_hashes) != 1:
            raise AssertionError('Historical animal rounds disagree on released response source')
        print(f'Unchanged representations verified: {animal}100 rounds', flush=True)
    old_final_path = source_root / 'results/final_integrity_and_causal_audit.json'
    old_final = read_json(old_final_path)
    old_readout_audits = []
    for item in old_final['round0_and_99_actual_weight_causal_audits']:
        if sha256(item['path']) != item['sha256']:
            raise AssertionError('Historical actual-weight causal audit changed')
        old_readout_audits.append(item)
    references = [
        {'path': str(protocol_path.resolve()), 'sha256': protocol_hash, 'purpose': 'source frozen protocol'},
        {'path': str(lock_path.resolve()), 'sha256': sha256(lock_path), 'purpose': 'source protocol lock'},
        {'path': str((source_root / 'configs/condition_splits_100.json').resolve()),
         'sha256': sha256(source_root / 'configs/condition_splits_100.json'), 'purpose': 'unchanged physical condition splits'},
        {'path': str(old_final_path.resolve()), 'sha256': sha256(old_final_path), 'purpose': 'historical complete integrity and actual-weight audit'},
    ]
    result = {'source_root': str(source_root.resolve()), 'new_root': str(new_root.resolve()),
              'representation_animal_rounds_verified': len(rows), 'FA_models_reused': 200,
              'GPFA_models_reused': 200, 'latent_dimensions': 50, 'fit_half': 'half1',
              'old_weights_preprocessing_latents_changed': False, 'new_representation_fits': 0,
              'reuse_mode': 'direct verified references to existing source files',
              'source_condition_splits_unchanged': True, 'provided_input_filtering': 'pass',
              'fit_provenance': 'pass', 'raw_preprocessing': 'fail',
              'maximum_historical_prefix_test_error': maximum_prefix_error,
              'new_causal_inference_tests_required': False,
              'causal_reuse_reason': 'Same parameters, preprocessing, neural identities, time axes and saved latent bytes; the existing per-round prefix tests remain applicable. New OLS parameters must still be fit only on the same train39 conditions.',
              'historical_actual_weight_causal_audits': old_readout_audits,
              'source_records': references + records, 'rounds': rows}
    write_json(new_root / 'results/reused_representation_audit.json', result)
    return result
