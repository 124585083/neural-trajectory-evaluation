"""Training-condition-only FA50 and prefix GPFA50 on released mean inputs.

This module never accepts behavioral targets.  Released upstream imputation is
irreversible and is explicitly NOT certified as raw-data causal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import FactorAnalysis
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
from trajectory_project.runtime_paths import configured_root
# Original configuration/source hashes belong to the preserved scientific run.
# The English publication configuration has a separate integration checksum.
PILOT = configured_root('source_pilot')
from trajectory_project._dependencies import verified_file
from trajectory_project._dependencies import SharedTimescaleGPFA

DATA_ROOT = configured_root("source_data")
CAUSAL_STATUS = 'published_input_causal_train_only;raw_preprocessing_fail'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                              default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf-8')
    temp.replace(path)


def configuration():
    """Existing FA baseline and exact shared-RBF training choices, no search."""
    old = json.loads((PILOT / 'gpfa_config.json').read_text(encoding='utf-8'))
    cfg = {
        'latent_dim': 50, 'seed': 42,
        'fa_options': {'n_components': 50, 'random_state': 42, 'max_iter': 300,
                       'tol': .01, 'svd_method': 'lapack', 'iterated_power': 7, 'rotation': None},
        'gpfa_options': old['model_options'],
        'gpfa_initialization': 'same-round training-only FA50 C/d/R then existing 1% loading jitter seed42',
        'initialization_difference': 'old GPFA grid initialized from FA100 with omitted variance added to R; this fixed equal-rank branch shares its separately evaluated FA50 initializer; no label-based choice',
        'fit_half': 'half1', 'neuron_training_finite_fraction': .99, 'std_floor': 1e-5,
        'measurement_min_observed_neuron_fraction': .95,
        'input_dtype': 'float64', 'latent_coordinates': 'raw posterior, not orthogonalized',
        'bin_ms': 50, 'bin_available_rule': '(bin_index + 1) * 50 ms',
        'support_rule': 'published start_end_pad0 minus any f_pad0 overlap; excludes feedback-straddling terminal bins',
        'source_sha256': {str(p): sha256(p) for p in (
            PILOT / 'gpfa_shared.py', PILOT / 'gpfa_config.json',
            PILOT / 'capacity_position_baselines.py', PILOT / 'data_pipeline.py',
            PILOT / 'third_party/MentalPong/analyses/PhysDataset.py',
            PILOT / 'third_party/MentalPong/analyses/phys_utils.py')},
    }
    protocol_path = PILOT / 'trajectory_project/trial_endpoint_fa_gpfa_v1/configs/analysis_protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        lock = json.loads((protocol_path.parent / 'protocol_lock.json').read_text(encoding='utf-8'))
        if sha256(protocol_path) != lock['sha256']:
            raise ValueError('Frozen analysis protocol changed')
        assert cfg['fa_options'] == protocol['FA']
        assert cfg['gpfa_options'] == protocol['GPFA']
        assert cfg['latent_dim'] == protocol['n_components']
        assert cfg['seed'] == protocol['model_seed']
        assert cfg['std_floor'] == protocol['preprocessing']['std_floor']
        assert cfg['neuron_training_finite_fraction'] == protocol['preprocessing']['training_finite_fraction']
        cfg['analysis_protocol_sha256'] = sha256(protocol_path)
    return cfg


def load_neural_data(animal, data_root=DATA_ROOT, cache_root=None):
    """Bypass old 47-condition neuron filter/scaling and author FA/reliability."""
    path = verified_file(data_root, f'{animal}_hand_dmfc_dataset_50ms.pkl')
    source_hash = sha256(path)
    cache = None if cache_root is None else Path(cache_root) / f'{animal}_published_neural.npz'
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata']))
            if metadata['source_sha256'] != source_hash:
                raise ValueError('Neural cache source differs')
            out = {k: z[k] for k in z.files if k != 'metadata'}
        return {**out, 'metadata': metadata, 'animal': animal}
    raw = pd.read_pickle(path)
    ids = {h: np.asarray(raw['neural_idx_global']['occ'][f'neural_responses_sh{h}'], dtype=int)
           for h in (1, 2)}
    # Match published identities, not data-dependent test reliability scores.
    common = np.asarray(sorted(set(ids[1]) & set(ids[2])), dtype=int)
    out = {}
    for half in (1, 2):
        index = {int(i): j for j, i in enumerate(ids[half])}
        values = np.asarray(raw[f'neural_responses_sh{half}']['occ'])
        out[f'half{half}'] = values[[index[int(i)] for i in common]].transpose(1, 2, 0).astype(np.float64)
    masks = raw['masks']['occ']
    official = np.isfinite(masks['start_end_pad0'])
    feedback = np.isfinite(masks['f_pad0'])
    out.update(condition_ids=np.asarray(raw['meta']['py_meta_index'], dtype=np.int64),
               neuron_ids=common, valid=official & ~feedback,
               original_full_mask=official, feedback_mask=feedback,
               times_ms=(np.arange(out['half1'].shape[1]) + 1) * 50)
    if len(out['condition_ids']) != 79 or len(np.unique(out['condition_ids'])) != 79:
        raise ValueError('Expected all 79 physical conditions')
    for row in out['valid']:
        ix = np.flatnonzero(row)
        if not len(ix) or not np.all(np.diff(ix) == 1):
            raise ValueError('Support must be one contiguous active sequence')
    metadata = {
        'animal': animal, 'source_path': str(path), 'source_sha256': source_hash,
        'source_fields': ['neural_responses_sh1.occ', 'neural_responses_sh2.occ'],
        'neural_axes': ['condition', '50ms_bin', 'neuron'], 'neuron_count': len(common),
        'source_neuron_alignment': 'intersection of published global neuron identities in fixed half pair',
        'not_used': ['released reliable_neural_idx', 'released author FA factors', 'old47 prepared neuron filter/statistics'],
        'raw_preprocessing': 'fail',
        'raw_preprocessing_reason': 'PhysDataset.consolidate_sessions overwrites base arrays after phys_utils.impute_nan; missing whole-condition response is filled using all-condition/all-time neuron mean, after whole-dataset coverage selection. Unimputed arrays and constituent stable-trial membership are not included.',
        'upstream_smoothing': 'source builder uses nonoverlapping 50ms bin averages; no bilateral smoothing in this path',
        'half_pair': 'one published half pair; membership cannot be matched to endpoint trial records',
        'data_mode': 'trial_labels_with_mean_neural',
        'raw_original_timestamp': 'nonoverlapping mean over original 1ms samples; center 24.5ms relative to bin start; available at right edge50ms',
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out, metadata=np.asarray(json.dumps(metadata)))
    return {**out, 'metadata': metadata, 'animal': animal}


def fit_preprocessing(data, train_ids, cfg):
    train = np.isin(data['condition_ids'], np.asarray(train_ids, dtype=int))
    rows = data['half1'][train][data['valid'][train]]
    fraction = np.isfinite(rows).mean(axis=0)
    mean = np.nanmean(rows, axis=0)
    std = np.nanstd(rows, axis=0)
    keep = ((fraction >= cfg['neuron_training_finite_fraction']) &
            np.isfinite(mean) & np.isfinite(std) & (std > cfg['std_floor']))
    if int(keep.sum()) <= cfg['latent_dim']:
        raise ValueError('Insufficient eligible train-side neurons for FA/GPFA50')
    return {'keep': keep, 'mean': mean[keep], 'std': np.maximum(std[keep], cfg['std_floor']),
            'neuron_ids': data['neuron_ids'][keep], 'training_finite_fraction': fraction[keep]}


def preprocess(values, prep):
    """Pointwise fixed training statistics; no temporal interpolation/smoothing."""
    raw = np.asarray(values, dtype=np.float64)[..., prep['keep']]
    return np.where(np.isfinite(raw), (raw - prep['mean']) / prep['std'], 0.)


def infer_latents(fa, gpfa, data, prep):
    result = {name: np.full((2, *data['valid'].shape, 50), np.nan, dtype=np.float64)
              for name in ('FA50', 'GPFA50')}
    for h, half in enumerate(('half1', 'half2')):
        standardized = preprocess(data[half], prep)
        for ci in range(len(data['condition_ids'])):
            ix = np.flatnonzero(data['valid'][ci])
            seq = standardized[ci, ix]
            result['FA50'][h, ci, ix] = fa.transform(seq)
            result['GPFA50'][h, ci, ix] = gpfa.infer_sequence(seq, 'causal')[1]
    return result


def audit_prefix(fa, gpfa, data, prep, train_ids, test_ids, tolerance=1e-8):
    """Actual inputs, optimized causal vs explicit prefix smoothing reference."""
    lookup = {int(c): i for i, c in enumerate(data['condition_ids'])}
    checks = []
    chosen = [(side, c) for side, ids in [('train', train_ids), ('test', test_ids)]
              for c in sorted(map(int, ids))[:3]]
    digest = gpfa.parameter_digest()
    rng = np.random.default_rng(730)
    for side, cid in chosen:
        ci = lookup[cid]
        ix = np.flatnonzero(data['valid'][ci])
        for half in ('half1', 'half2'):
            raw = np.asarray(data[half][ci, ix], dtype=np.float64)
            seq = preprocess(raw, prep)
            online = gpfa.infer_sequence(seq, 'causal')[1]
            fa_full = fa.transform(seq)
            for k in sorted(set([0, len(seq) // 2, len(seq) - 2])):
                changed_raw = raw.copy()
                changed_raw[k + 1:] = rng.normal(size=changed_raw[k + 1:].shape) * 10
                changed = preprocess(changed_raw, prep)
                changed_gpfa = gpfa.infer_sequence(changed, 'causal')[1]
                prefix_gpfa = gpfa.infer_sequence(seq[:k + 1], 'smooth')[1][-1]
                errors = {
                    'pointwise_preprocessing_future_error': float(np.max(np.abs(seq[:k + 1] - changed[:k + 1]))),
                    'fa_future_error': float(np.max(np.abs(fa_full[:k + 1] - fa.transform(changed)[:k + 1]))),
                    'gpfa_future_error': float(np.max(np.abs(online[:k + 1] - changed_gpfa[:k + 1]))),
                    'gpfa_prefix_reference_error': float(np.max(np.abs(online[k] - prefix_gpfa))),
                }
                checks.append({'split': side, 'condition_id': cid, 'half': half, 'cutoff_index': k,
                               'max_error': max(errors.values()), **errors})
            # Stateful processing cannot spill into a subsequent condition.
            other = preprocess(data[half][(ci + 1) % len(data['condition_ids']),
                                           data['valid'][(ci + 1) % len(data['condition_ids'])]], prep)
            gpfa.infer_sequence(other, 'causal')
            reset_error = float(np.max(np.abs(online - gpfa.infer_sequence(seq, 'causal')[1])))
            checks.append({'split': side, 'condition_id': cid, 'half': half,
                           'next_trial_replacement_error': reset_error, 'max_error': reset_error})
    maximum = max(r['max_error'] for r in checks)
    if gpfa.parameter_digest() != digest:
        raise AssertionError('Audit changed fitted GPFA parameters')
    return {
        'provided_input_filtering': 'pass' if maximum <= tolerance else 'fail',
        'raw_preprocessing': 'fail', 'fit_provenance': 'pass',
        'causal_status': CAUSAL_STATUS, 'tolerance': tolerance,
        'max_error': maximum, 'checks': checks,
        'fit_provenance_scope': 'new local selection, scaling, FA and GPFA use only this round39 half1; published upstream values retain known leakage',
        'endpoint_independence': 'representation API accepts no endpoint or behavioral labels; actual OLS future/endpoint audit is separate',
        'parameter_digest_unchanged': True,
    }


def fit_round(data, train_ids, test_ids, round_id, output_root=HERE, cfg=None, cpu_threads=4):
    cfg = configuration() if cfg is None else cfg
    train_ids, test_ids = sorted(map(int, train_ids)), sorted(map(int, test_ids))
    if len(train_ids) != 39 or len(test_ids) != 40 or set(train_ids) & set(test_ids):
        raise ValueError('Requires disjoint39/40 groups')
    if set(train_ids + test_ids) != set(map(int, data['condition_ids'])):
        raise ValueError('Every one of the79 conditions must retain split identity')
    out = Path(output_root) / 'representations' / data['animal'] / f'round_{int(round_id):03d}'
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = {'configuration': cfg, 'train_ids': train_ids, 'test_ids': test_ids,
                   'source_sha256': data['metadata']['source_sha256'], 'module_sha256': sha256(__file__)}
    marker = out / 'training.json'
    if marker.exists():
        meta = json.loads(marker.read_text(encoding='utf-8'))
        if meta['fingerprint'] != fingerprint:
            raise ValueError(f'Cached representation differs: {out}')
        for name, expected in meta['file_hashes'].items():
            if sha256(out / name) != expected:
                raise ValueError(f'Cached representation hash differs: {out / name}')
        return load_round(out)
    started = time.perf_counter()
    prep = fit_preprocessing(data, train_ids, cfg)
    train = np.isin(data['condition_ids'], train_ids)
    standardized = preprocess(data['half1'], prep)
    train_sequences = [standardized[i, data['valid'][i]] for i in np.flatnonzero(train)]
    train_flat = np.concatenate(train_sequences)
    fa = FactorAnalysis(**cfg['fa_options'])
    print(f"{data['animal']} round{round_id:03d}: FA50 fit {train_flat.shape}", flush=True)
    with threadpool_limits(limits=cpu_threads):
        fa_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fa.fit(train_flat)
        fa_seconds = time.perf_counter() - fa_started
        gpfa = SharedTimescaleGPFA(50, seed=cfg['seed'], **cfg['gpfa_options'])
        gpfa_started = time.perf_counter()
        gpfa.fit(train_sequences, init_parameters={'C': fa.components_.T,
                                                  'd': fa.mean_, 'R': fa.noise_variance_})
        gpfa_seconds = time.perf_counter() - gpfa_started
        latent = infer_latents(fa, gpfa, data, prep)
        audit = audit_prefix(fa, gpfa, data, prep, train_ids, test_ids)
    if audit['provided_input_filtering'] != 'pass':
        save_json(out / 'failed_causal_audit.json', audit)
        raise AssertionError('Cannot use failed provided-input causal result')
    joblib.dump(fa, out / 'fa50.joblib')
    gpfa.save(out / 'gpfa50.npz')
    np.savez_compressed(out / 'preprocessing.npz', **prep, train_ids=np.asarray(train_ids),
                        source_neuron_ids=data['neuron_ids'])
    common = data['valid'].copy()
    for half in ('half1', 'half2'):
        common &= np.isfinite(data[half][..., prep['keep']]).mean(axis=-1) >= cfg['measurement_min_observed_neuron_fraction']
    np.savez_compressed(out / 'latents.npz', **latent, condition_ids=data['condition_ids'],
                        valid=data['valid'], common_mask=common, times_ms=data['times_ms'],
                        half_names=np.asarray(['half1', 'half2']))
    save_json(out / 'causal_audit.json', audit)
    pd.DataFrame(gpfa.history).to_csv(out / 'gpfa_history.csv', index=False)
    last_change = float(fa.loglike_[-1] - fa.loglike_[-2]) if len(fa.loglike_) > 1 else None
    meta = {
        'animal': data['animal'], 'round_id': int(round_id), 'fingerprint': fingerprint,
        'causal_status': CAUSAL_STATUS, 'data_mode': 'trial_labels_with_mean_neural',
        'fit_conditions': train_ids, 'test_conditions': test_ids, 'fit_half': 'half1',
        'fit_bins': len(train_flat), 'n_neurons': len(prep['mean']), 'latent_dimensions': 50,
        'OLS_coefficient_count_per_coordinate': 51, 'OLS_coefficient_count_xy': 102,
        'FA_parameter_count': fa.components_.size + 2 * len(fa.mean_),
        'GPFA_parameter_count': gpfa.parameter_count,
        'FA_n_iter': int(fa.n_iter_), 'FA_last_likelihood_change': last_change,
        'FA_strict_convergence': bool(fa.n_iter_ < fa.max_iter and last_change is not None and 0 <= last_change < fa.tol),
        'FA_warnings': [str(w.message) for w in caught],
        'GPFA_n_iter': len(gpfa.history), 'GPFA_stop_reason': gpfa.stop_reason,
        'GPFA_converged': gpfa.converged, 'GPFA_tau_seconds': gpfa.tau,
        'fa_fit_seconds': fa_seconds, 'gpfa_fit_seconds': gpfa_seconds,
        'total_seconds': time.perf_counter() - started,
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
        'precision': 'float64', 'cpu_threads': cpu_threads,
        'upstream': data['metadata'],
        'file_hashes': {name: sha256(out / name) for name in
                       ('fa50.joblib', 'gpfa50.npz', 'preprocessing.npz', 'latents.npz', 'causal_audit.json', 'gpfa_history.csv')},
    }
    save_json(marker, meta)
    print(f"{data['animal']} round{round_id:03d}: done FA={fa_seconds:.1f}s GPFA={gpfa_seconds:.1f}s total={meta['total_seconds']:.1f}s", flush=True)
    return load_round(out)


def load_round(folder):
    folder = Path(folder)
    meta = json.loads((folder / 'training.json').read_text(encoding='utf-8'))
    with np.load(folder / 'latents.npz', allow_pickle=False) as stored:
        out = {name: {'latent': stored[name][0], 'latent_half2': stored[name][1],
                      'condition_ids': stored['condition_ids'], 'common_mask': stored['common_mask'],
                      'times_ms': stored['times_ms'], 'causal_status': CAUSAL_STATUS,
                      'provenance': meta, 'model_directory': str(folder)} for name in ('FA50', 'GPFA50')}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--animal', choices=['mahler', 'perle'], required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--round-start', type=int, default=0)
    p.add_argument('--round-stop', type=int, default=100)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    data = load_neural_data(args.animal, cache_root=HERE / 'artifacts')
    splitobj = json.loads(args.splits.read_text(encoding='utf-8'))
    splits = splitobj['splits'] if isinstance(splitobj, dict) else splitobj
    for i in range(args.round_start, args.round_stop):
        split = splits[i]
        train_ids = split.get('train_condition_ids', split.get('train_ids'))
        test_ids = split.get('test_condition_ids', split.get('test_ids'))
        fit_round(data, train_ids, test_ids, i, cpu_threads=args.threads)


if __name__ == '__main__':
    main()
