"""Verified published data -> aligned physical-condition arrays; no model fitting."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIELDS = ('neural_responses', 'neural_responses_sh1', 'neural_responses_sh2')

def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def verified_file(data_root, basename):
    manifest_path = Path(data_root) / 'raw/download_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    archive = manifest['archive']
    if not archive.get('verified') or archive.get('md5') != '382e420c6f053d8ac2bdd4acd2fb02b2':
        raise ValueError('Full official archive MD5 has not been verified')
    # The verified source manifest is immutable and may contain another host's
    # absolute paths. Resolve its unique raw-relative member under data_root.
    if Path(basename).name != basename or '/' in basename or '\\' in basename:
        raise ValueError('Expected an unqualified verified member basename')
    entries = [e for e in manifest['extracted_files']
               if str(e['path']).replace('\\', '/').rsplit('/', 1)[-1] == basename]
    if len(entries) != 1: raise ValueError(f'Expected one verified {basename}, got {len(entries)}')
    entry = entries[0]
    parts = str(entry['path']).replace('\\', '/').split('/')
    anchors = [i for i, part in enumerate(parts) if part.lower() == 'raw']
    if len(anchors) != 1:
        raise ValueError('Verified member path must contain one unambiguous raw directory')
    relative_parts = parts[anchors[0] + 1:]
    if not relative_parts or any(p in ('', '.', '..') or ':' in p for p in relative_parts):
        raise ValueError('Unsafe verified member path below raw directory')
    raw_root = (Path(data_root) / 'raw').resolve()
    path = raw_root.joinpath(*relative_parts).resolve()
    if not path.is_relative_to(raw_root):
        raise ValueError('Verified member resolves outside configured raw directory')
    if file_sha(path) != entry['sha256']: raise ValueError(f'Member changed: {path}')
    return path

def load_published(data_root, subject):
    # pandas supports older pandas pickle layouts; source and member hash checked first.
    return pd.read_pickle(verified_file(data_root, f'{subject}_hand_dmfc_dataset_50ms.pkl'))

def describe(value, depth=0):
    if isinstance(value, pd.DataFrame):
        return {'type': 'DataFrame', 'shape': list(value.shape), 'columns': list(map(str, value.columns))}
    if isinstance(value, np.ndarray):
        out = {'type': 'array', 'shape': list(value.shape), 'dtype': str(value.dtype)}
        if np.issubdtype(value.dtype, np.number): out['finite_fraction'] = float(np.isfinite(value).mean())
        return out
    if isinstance(value, dict) and depth < 2:
        return {str(k): describe(v, depth + 1) for k, v in value.items()}
    return {'type': type(value).__name__}

def build_condition_table(conditions, xy, dy, valid, visible, hidden, meta):
    rows = []
    for c, condition in enumerate(conditions):
        inds = np.flatnonzero(valid[c]); finite_dy = np.flatnonzero(valid[c] & np.isfinite(dy[c]))
        v = dy[c, finite_dy]
        floor = max(float(np.nanmax(np.abs(v))) * 0.1, 1e-9) if len(v) else 1e-9
        nonzero = finite_dy[np.abs(v) > floor]
        events = []
        for left, right in zip(nonzero[:-1], nonzero[1:]):
            if np.sign(dy[c, left]) == np.sign(dy[c, right]): continue
            if right - left > 3: continue
            phase = 'hidden' if hidden[c, left] and hidden[c, right] else 'visible' if visible[c, left] and visible[c, right] else 'boundary_ambiguous'
            events.append({'pre_bin': int(left), 'post_bin': int(right), 'available_ms': int((right + 1) * 50), 'phase': phase,
                           'covered_before_3': int(np.sum(hidden[c, max(0, right-3):right])),
                           'covered_after_3': int(np.sum(hidden[c, right+1:right+4]))})
        hidden_events = [e for e in events if e['phase'] == 'hidden']
        visible_events = [e for e in events if e['phase'] == 'visible']
        kind = 'hidden_bounce' if hidden_events and not visible_events else 'visible_bounce' if visible_events and not hidden_events else 'mixed_bounce' if hidden_events and visible_events else 'boundary_ambiguous' if events else 'no_bounce'
        meta_bounce = None
        if 'n_bounce_correct' in meta.columns and condition in meta.index:
            value = meta.loc[condition, 'n_bounce_correct']
            meta_bounce = float(value) if np.isfinite(value) else None
        if meta_bounce is not None and meta_bounce > len(events):
            kind = 'unresolved_terminal_bounce' if not events else 'bounce_with_unresolved_event'
        rows.append({'condition_index': c, 'condition_id': int(condition), 'bounce_class': kind,
                     'metadata_bounce_count': meta_bounce, 'observed_bounce_count': len(events),
                     'hidden_bounces': len(hidden_events), 'visible_bounces': len(visible_events),
                     'valid_bins': int(valid[c].sum()), 'visible_bins': int(visible[c].sum()), 'hidden_bins': int(hidden[c].sum()),
                     'first_available_ms': int((inds[0]+1)*50) if len(inds) else None,
                     'last_available_ms': int((inds[-1]+1)*50) if len(inds) else None,
                     'bounce_events': json.dumps(events), 'first_hidden_bounce_bin': hidden_events[0]['post_bin'] if hidden_events else -1})
    return pd.DataFrame(rows)

def stratified_split(table, cfg):
    rng = np.random.default_rng(cfg['split_seed']); assignments = {}
    # Categories stratify *physical conditions*, never condition x time or repeat half.
    for kind, rows in table.groupby('bounce_class', sort=True):
        ids = rng.permutation(rows['condition_id'].to_numpy())
        n = len(ids)
        if n < 3:
            assignments.update({int(i): 'train' for i in ids}); continue
        nv = max(1, round(n * cfg['validation_fraction'])); nt = max(1, round(n * (1-cfg['train_fraction']-cfg['validation_fraction'])))
        if nv + nt >= n: nv = nt = 1
        for split, arr in [('validation', ids[:nv]), ('test', ids[nv:nv+nt]), ('train', ids[nv+nt:])]:
            assignments.update({int(i): split for i in arr})
    return assignments

def prepare_subject(cfg, subject):
    data_root = Path(cfg['data_root']); raw = load_published(data_root, subject); cat = cfg['category']
    conditions = np.asarray(raw['meta']['py_meta_index'], dtype=np.int64)
    assert len(np.unique(conditions)) == len(conditions), 'Duplicate physical condition IDs'
    global_ids = {f: np.asarray(raw['neural_idx_global'][cat][f], dtype=np.int64) for f in FIELDS}
    common = sorted(set(global_ids[FIELDS[0]]) & set(global_ids[FIELDS[1]]) & set(global_ids[FIELDS[2]]))
    if not common: raise ValueError('No matching neuron identities')
    arrays = {}
    for f in FIELDS:
        ids = global_ids[f]
        if len(ids) != len(np.unique(ids)): raise ValueError('Duplicate neuron identity')
        idx = {int(g): i for i, g in enumerate(ids)}
        arrays[f] = np.asarray(raw[f][cat])[[idx[g] for g in common]].transpose(1, 2, 0).astype(np.float32)
    assert len({a.shape for a in arrays.values()}) == 1
    ncond, ntime, nneur = arrays[FIELDS[0]].shape
    assert ncond == len(conditions)
    masks = raw['masks'][cat]
    valid = np.isfinite(masks['start_end_pad0'])
    terminal = valid & np.isfinite(masks['f_pad0'])
    # The released nanmean-binned mask includes an end-straddling bin. Its
    # neural average can include post-task activity; exclude it uniformly.
    valid &= ~terminal
    vis0 = np.isfinite(masks['start_occ_pad0']); occ0 = np.isfinite(masks['occ_end_pad0'])
    # Avoid assigning a bin straddling occlusion to either pure phase.
    visible = valid & vis0 & ~occ0; hidden = valid & occ0 & ~vis0
    behavior = raw['behavioral_responses'][cat]
    xy = np.stack([behavior['ball_pos_x_TRUE'], behavior['ball_pos_y_TRUE']], axis=-1).astype(np.float32)
    # Readout labels are separate; never an encoder input. Backward differences avoid anticipatory labels.
    dy = np.full((ncond, ntime), np.nan, dtype=np.float32)
    dy[:, 1:] = np.diff(xy[..., 1], axis=1) / (cfg['bin_width_ms']/1000)
    valid &= np.isfinite(xy).all(axis=-1)
    visible &= valid; hidden &= valid
    for row in valid:
        idx = np.flatnonzero(row)
        if len(idx) and np.any(np.diff(idx) != 1): raise ValueError('Non-contiguous active sequence: audit before training')
    physical_meta = pd.read_pickle(verified_file(data_root, 'valid_meta_sample_full.pkl'))
    if not isinstance(physical_meta, pd.DataFrame): raise TypeError('Unexpected physical metadata format')
    physical_meta = physical_meta.set_index('meta_index', drop=False)
    if not np.all(np.isin(conditions, physical_meta.index)): raise ValueError('Missing physical metadata IDs')
    table = build_condition_table(conditions, xy, dy, valid, visible, hidden, physical_meta)
    split_path = ROOT / 'notes/condition_split.json'
    if split_path.exists():
        split_record = json.loads(split_path.read_text(encoding='utf-8'))
        assignments = {int(k): v for k, v in split_record['assignment'].items()}
        if set(assignments) != set(conditions): raise ValueError('Subject physical condition sets differ')
    else:
        assignments = stratified_split(table, cfg)
        save_json(split_path, {'seed': cfg['split_seed'], 'source_subject': subject, 'assignment': assignments,
                              'grouping': 'physical condition ID across all bins, task categories and both repeat halves',
                              'strata': 'bounce category from physical trajectory; no neural/task-readout result selection'})
    table['split'] = [assignments[int(c)] for c in conditions]
    split = np.asarray(table['split'], dtype=str); train = split == 'train'
    train_values = arrays[FIELDS[1]][train][valid[train]]
    finite_fraction = np.isfinite(train_values).mean(axis=0)
    mean = np.nanmean(train_values, axis=0); std = np.nanstd(train_values, axis=0)
    keep = (finite_fraction >= cfg['neuron_training_finite_fraction']) & np.isfinite(std) & (std > cfg['std_floor'])
    if keep.sum() < 16: raise ValueError('Too few training-eligible neurons')
    arrays = {f: a[..., keep] for f, a in arrays.items()}; mean = mean[keep]; std = np.maximum(std[keep], cfg['std_floor'])
    common = np.asarray(common)[keep]
    full, h1, h2 = [arrays[f] for f in FIELDS]
    observed1 = np.isfinite(h1).mean(axis=-1); observed2 = np.isfinite(h2).mean(axis=-1)
    measurement_valid = valid & (observed1 >= cfg['measurement_min_observed_neuron_fraction']) & (observed2 >= cfg['measurement_min_observed_neuron_fraction'])
    before_bounce = np.zeros_like(valid); after_bounce = np.zeros_like(valid)
    for c, entry in table.iterrows():
        for event in json.loads(entry['bounce_events']):
            if event['phase'] != 'hidden': continue
            b=event['post_bin']; window=cfg['event_window_bins']
            before_bounce[c,max(0,b-window):b] = True
            after_bounce[c,b+1:b+1+window] = True
    before_bounce &= hidden & measurement_valid; after_bounce &= hidden & measurement_valid
    relation_mask = valid[...,None] & np.isfinite(full) & np.isfinite(h1) & np.isfinite(h2)
    relation_error = float(np.mean((full[relation_mask] - ((h1+h2)/2)[relation_mask])**2))
    outdir = data_root / 'processed'; outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / f'{subject}_prepared.npz', full=full, half1=h1, half2=h2,
                        mean=mean, std=std, neuron_ids=common, condition_ids=conditions,
                        valid=valid, visible=visible, hidden=hidden, xy=xy, dy=dy, split=split,
                        measurement_valid=measurement_valid, observed_fraction_half1=observed1, observed_fraction_half2=observed2,
                        before_hidden_bounce=before_bounce, after_hidden_bounce=after_bounce,
                        bin_available_ms=(np.arange(ntime)+1)*cfg['bin_width_ms'])
    (ROOT/'results').mkdir(exist_ok=True)
    table.to_csv(ROOT/'results'/f'{subject}_physical_conditions.csv', index=False)
    summary = {'subject': subject, 'source_sha256': file_sha(verified_file(data_root, f'{subject}_hand_dmfc_dataset_50ms.pkl')),
               'published_schema': describe(raw), 'analyzed_category': cat, 'array_axes': ['condition','50ms_bin','neuron'],
               'units': 'continuous condition-mean spike count per original 1ms bin, averaged in 50ms bins; x1000 = Hz',
               'available_time_rule': '(original_bin_index + 1)*50ms; never bin-start timestamp',
               'shape': list(h1.shape), 'aligned_neurons_before_training_filter': nneur,
               'neuron_filter': 'half1 training conditions only: finite fraction and nonzero variance',
               'phase_bins': {'all': int(valid.sum()), 'visible': int(visible.sum()), 'hidden': int(hidden.sum()), 'phase_boundary_excluded': int(np.sum(valid & vis0 & occ0)), 'terminal_bins_excluded': int(terminal.sum())},
               'split_counts': table['split'].value_counts().to_dict(), 'split_bounce_counts': table.groupby(['split','bounce_class']).size().to_dict().__str__(),
               'full_vs_equal_halves_mse': relation_error,
               'published_input_coverage': {'minimum_fraction_half1': float(observed1[valid].min()), 'minimum_fraction_half2': float(observed2[valid].min()),
                                             'measurement_bins_excluded': int(np.sum(valid & ~measurement_valid)),
                                             'note': 'finite published bins may already include irreversible upstream global-mean imputation'},
               'half_split': 'One released split pair, train_test_split(test_size=.5, random_state=0) within condition before condition averaging; no new trial halves generated',
               'upstream': {'pseudo_population': True, 'global_missing_condition_imputation': True, 'all_condition_coverage_filter': True,
                            'published_smoothing': 'Independent non-overlapping 50ms bin averages; no main-build bilateral smoothing located',
                            'irreversible_limit': 'Full pipeline cannot be called future-free from raw spikes: upstream missing-condition mean uses all conditions/times',
                            'trial_counts': 'trial_meta is available; exact per-neuron stable-trial membership/counts for each published half are not recovered'},
               'metadata_shape': list(physical_meta.shape)}
    save_json(ROOT/'notes'/f'{subject}_data_audit.json', summary)
    print(json.dumps({k: summary[k] for k in ('subject','shape','phase_bins','split_counts','full_vs_equal_halves_mse')}, indent=2), flush=True)
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='config.json'); parser.add_argument('--subject', default='perle')
    args = parser.parse_args(); prepare_subject(json.loads(Path(args.config).read_text()), args.subject)
