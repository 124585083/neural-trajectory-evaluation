"""Validate and replay saved arrays with NumPy; no external storage is consulted."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

SCHEMA = 'mental_pong_mini_replay_v1'
ATOL = 1e-8
RTOL = 1e-10
EPOCHS = ('full', 'visible', 'hidden', 'bounce', 'no_bounce', 'post_bounce', 'endpoint_influence')
COEFFICIENT_CONVENTION = 'prediction[head,row,xy] = features[row,dimension] @ coefficients[head,xy,dimension].T + intercepts[head,xy]'
AXES = {'condition_ids':['condition'], 'times_ms':['bin'],
        'common_mask':['condition','bin'], 'condition_index':['row'], 'bin_index':['row'],
        'epoch_masks':['epoch','condition','bin'], 'offset':['condition','bin'], 'alpha':['condition','bin'],
        'random_endpoints':['condition'], 'random_source_indices':['valid_condition'],
        'random_target_indices':['valid_condition'], 'mean_endpoints':['condition'],
        'mismatch_mapping':['condition'], 'targets':['head','row','coordinate'],
        'features':['row','latent_dimension'], 'coefficients':['head','coordinate','latent_dimension'],
        'intercepts':['head','coordinate'], 'observed_baseline_means':['objective_behavior']}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def score(y, p, baseline=None):
    """Pooled condition-time Pearson r and RMSE, with original undefined-r rule."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError('Scores require equal one-dimensional target and prediction arrays.')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('A selected row has a missing target or prediction.')
    if not len(y):
        return {'r': np.nan, 'RMSE': np.nan, 'skill': np.nan, 'baseline_SSE': np.nan}
    yc, pc = y - y.mean(), p - p.mean()
    denominator = np.sqrt(np.dot(yc, yc) * np.dot(pc, pc))
    r = float(np.dot(yc, pc) / denominator) if len(y) >= 3 and np.ptp(y) and np.ptp(p) and denominator else np.nan
    sse = float(np.sum((p-y)**2))
    out = {'r': r, 'RMSE': float(np.sqrt(sse/len(y)))}
    if baseline is not None:
        base_sse = float(np.sum((np.asarray(baseline)-y)**2))
        out.update(baseline_SSE=base_sse, skill=1-sse/base_sse if base_sse else np.nan)
    return out


def _compare(actual, expected, label):
    expected = np.nan if expected is None else expected
    if not np.isclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True):
        raise ValueError(f'Score mismatch for {label}: computed {actual}, saved {expected}.')
    return abs(float(actual-expected)) if np.isfinite(actual) and np.isfinite(expected) else 0.


def load_bundle(folder):
    """Check the manifest before reading any numeric payload; reject unsafe paths."""
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA or manifest.get('selection') != {'split': 0, 'q': 0, 'half': 'half1'}:
        raise ValueError('Unsupported replay schema or changed fixed selection.')
    if manifest.get('payload_status') not in ('available', 'private_validation_only'):
        raise FileNotFoundError('The scientific replay payload is withheld: see manifest.json and README.md.')
    conventions = {'head_order':['objective','behavior','random'], 'coordinate_order':['x','y'],
                   'epoch_order':list(EPOCHS), 'coefficient_convention':COEFFICIENT_CONVENTION,
                   'tolerance':{'atol':ATOL,'rtol':RTOL}}
    for name, value in conventions.items():
        if manifest.get(name) != value:
            raise ValueError(f'Unsupported or contradictory manifest convention: {name}')
    for relative, expected in manifest['files'].items():
        path = (folder/relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError(f'Missing or unsafe payload path: {relative}')
        if sha(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Payload checksum mismatch: {relative}')
    with np.load(folder/'bundle/arrays.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if set(arrays) != set(manifest['arrays']):
        raise ValueError('Unexpected or missing numeric array.')
    for key, value in arrays.items():
        spec = manifest['arrays'][key]
        if value.dtype.kind not in 'bifu' or list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']:
            raise ValueError(f'Array schema mismatch: {key}')
        if array_sha(value) != spec['sha256']:
            raise ValueError(f'Array checksum mismatch: {key}')
        kind = key.split('__')[-1]
        axes = ['pair','source_condition_target_condition_bin'] if kind.startswith('mismatch_pairs_') else AXES.get(kind)
        if len(spec['axes']) != value.ndim or spec['axes'] != axes:
            raise ValueError(f'Array axes mismatch: {key}')
    metadata = {name: json.loads((folder/f'bundle/{name}.json').read_text(encoding='utf-8'))
                for name in ('cases', 'condition_splits', 'expected_scores', 'source_manifest')}
    return manifest, arrays, metadata


def run(folder, output, refit=False):
    """Compute scores from packaged features, never from the expected-score table."""
    folder = Path(folder).resolve()
    output = Path(output).resolve()
    if output == folder or output.is_relative_to(folder):
        raise ValueError('Choose an output directory outside the immutable mini replay package.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose a new or empty output directory; existing records are never overwritten.')
    manifest, arrays, metadata = load_bundle(folder)
    split = metadata['condition_splits']
    train, test = np.asarray(split['train_indices']), np.asarray(split['test_indices'])
    if len(train) != 39 or len(test) != 40 or set(train) & set(test) or set(train) | set(test) != set(range(79)):
        raise ValueError('Original disjoint 39/40 condition split changed.')
    rows, discrepancies, refits = [], [], []
    for case in metadata['cases']:
        stem, animal = case['case_id'], case['animal']
        get = lambda key: arrays[f'{animal}__{key}']
        c, t, ids = get('condition_index'), get('bin_index'), get('condition_ids')
        common, epochs = get('common_mask'), get('epoch_masks')
        if not np.array_equal(np.column_stack(np.nonzero(common)), np.column_stack((c, t))):
            raise ValueError('Condition/time row identity disagrees with common mask.')
        if len(np.unique(ids)) != 79 or not np.array_equal(ids[train], split['train_condition_ids']) or not np.array_equal(ids[test], split['test_condition_ids']):
            raise ValueError('Physical condition identities disagree with saved split.')
        if common[ids == 59920].any() or common.any(1).sum() != 78:
            raise ValueError('The unresolved collision-anchor condition must remain invalid.')
        tr, te = np.isin(c, train), np.isin(c, test)
        if tr.sum() != case['n_train_rows'] or te.sum() != case['n_test_rows']:
            raise ValueError('Actual training/test row counts changed.')
        if len(np.unique(c[tr])) != case['valid_train_conditions'] or len(np.unique(c[te])) != case['valid_test_conditions']:
            raise ValueError('Actual contributing condition counts changed.')
        X = arrays[f'{stem}__features']
        if X.shape != (len(c), 50) or not np.isfinite(X).all():
            raise ValueError('The saved latent matrix must contain 50 finite features per matched row.')
        targets = get('targets')
        coefficient, intercept = arrays[f'{stem}__coefficients'], arrays[f'{stem}__intercepts']
        if coefficient.shape != (3, 2, 50) or intercept.shape != (3, 2):
            raise ValueError('Expected three heads, x/y coordinates and 50 latent coefficients.')
        pred = np.stack([X[te] @ coef.T + bias for coef, bias in zip(coefficient, intercept)])
        if not np.allclose(pred[0, :, 0], pred[1, :, 0], rtol=RTOL, atol=ATOL):
            raise ValueError('Identical x targets produced different objective/behavior predictions.')
        if not np.allclose(pred[0, :, 0], pred[2, :, 0], rtol=RTOL, atol=ATOL):
            raise ValueError('The random head does not preserve the shared original x readout.')
        if not np.array_equal(targets[0, :, 0], targets[1, :, 0]):
            raise ValueError('Objective and behavior x labels differ.')
        offset, alpha = get('offset')[c, t], get('alpha')[c, t]
        endpoint = get('random_endpoints')
        mapped_conditions, source_conditions = get('random_target_indices'), get('random_source_indices')
        if not np.array_equal(np.sort(source_conditions), mapped_conditions):
            raise ValueError('The random endpoint assignment is not the saved valid-condition permutation.')
        if not np.array_equal(endpoint[mapped_conditions], get('mean_endpoints')[source_conditions]):
            raise ValueError('Random endpoints disagree with their original source condition identities.')
        if not np.array_equal(targets[2, :, 1], offset+alpha*endpoint[c]):
            raise ValueError('Random target differs from its saved endpoint geometry.')
        valid_train = np.unique(c[tr])
        mu = endpoint[valid_train].mean()
        _compare(mu, case['random_train_mean_endpoint'], stem+' training mean endpoint')
        baseline = offset[te]+alpha[te]*mu
        expected = metadata['expected_scores'][stem]
        for ei, epoch in enumerate(EPOCHS):
            active = epochs[ei, c[te], t[te]]
            n, nc = int(active.sum()), len(np.unique(c[te][active]))
            if [nc, n] != expected['counts'][ei]:
                raise ValueError(f'Changed valid-row counts for {stem}/{epoch}.')
            if ei < 6:
                for h in range(2):
                    for target in range(2):
                        for xy in range(2):
                            values = score(targets[target, te][active, xy], pred[h, active, xy])
                            for mi, metric in enumerate(('r', 'RMSE')):
                                discrepancies.append(_compare(values[metric], expected['matrix'][ei][h][target][xy][mi], f'{stem}/{epoch}/{h}/{target}/{xy}/{metric}'))
                            rows.append(dict(case=stem, epoch=epoch, head=('objective','behavior')[h], target=('objective','behavior')[target], coordinate=('x','y')[xy], n_bins=n, n_conditions=nc, **values))
            values = score(targets[2, te][active, 1], pred[2, active, 1], baseline[active])
            for metric, value in values.items():
                discrepancies.append(_compare(value, expected['random'][ei][metric], f'{stem}/{epoch}/random/{metric}'))
            rows.append(dict(case=stem, epoch=epoch, head='random', target='random', coordinate='y', n_bins=n, n_conditions=nc, **values))
        # Mapping entries point from original source condition to target condition.
        # The source and target use the same original bin index and phase mask.
        mapping = get('mismatch_mapping')
        valid_test = np.flatnonzero(common.any(1) & np.isin(np.arange(79), test))
        if not np.array_equal(np.sort(mapping[valid_test]), valid_test):
            raise ValueError('Condition-mismatch mapping escaped the valid test conditions.')
        lookup = np.full(common.shape, -1, int); lookup[c[te], t[te]] = np.arange(te.sum())
        target_lookup = np.full(common.shape, -1, int); target_lookup[c, t] = np.arange(len(c))
        for ei, epoch in enumerate(EPOCHS[:6]):
            pairs = get(f'mismatch_pairs_{ei}')
            actual_pairs = [(int(ci), int(mapping[ci]), int(ti)) for ci in valid_test for ti in np.flatnonzero(common[ci]&common[mapping[ci]]&epochs[ei,ci]&epochs[ei,mapping[ci]])]
            if not np.array_equal(pairs, np.asarray(actual_pairs, dtype=np.int64).reshape(-1,3)):
                raise ValueError('Mismatch row pairs disagree with the mapping and phase masks.')
            ci, cj, ti = pairs.T
            ix, yi, yj = lookup[ci,ti], target_lookup[ci,ti], target_lookup[cj,ti]
            if np.any(ix < 0) or np.any(yi < 0) or np.any(yj < 0):
                raise ValueError('Mismatch references unavailable or training predictions.')
            if [len(np.unique(ci)), len(ci)] != expected['mismatch_counts'][ei]:
                raise ValueError('Mismatch coverage disagrees with saved counts.')
            for h in range(2):
                for target in range(2):
                    for xy in range(2):
                        matched = score(targets[target,yi,xy], pred[h,ix,xy])
                        null = score(targets[target,yj,xy], pred[h,ix,xy])
                        for mode, values in (('matched',matched),('null',null)):
                            for metric in ('r','RMSE'):
                                saved = expected['mismatch'][ei][h][target][xy][mode+'_'+metric]
                                discrepancies.append(_compare(values[metric],saved,f'{stem}/{epoch}/{mode}/{h}/{target}/{xy}/{metric}'))
                            rows.append(dict(case=stem,epoch=epoch,head=('objective','behavior')[h],target=('objective','behavior')[target],coordinate=('x','y')[xy],comparison='condition_mismatch_'+mode,n_bins=len(ci),n_conditions=len(np.unique(ci)),**values))
        if refit:
            from sklearn.linear_model import LinearRegression
            import sklearn
            import scipy
            for h, name in enumerate(('objective','behavior','random')):
                model = LinearRegression(fit_intercept=True, positive=False).fit(X[tr], targets[h,tr])
                fitted = model.predict(X[te])
                error = float(np.max(np.abs(fitted-pred[h])))
                if not np.allclose(fitted,pred[h],atol=ATOL,rtol=RTOL):
                    raise ValueError(f'Optional OLS refit predictions differ for {stem}/{name}: {error}')
                metric_error = 0.
                for ei in range(7):
                    active=epochs[ei,c[te],t[te]]
                    for xy in range(2):
                        a,b=score(targets[h,te][active,xy],fitted[active,xy]),score(targets[h,te][active,xy],pred[h,active,xy])
                        for metric in ('r','RMSE'):
                            metric_error=max(metric_error,_compare(a[metric],b[metric],stem+'/'+name+'/refit/'+metric))
                refits.append(dict(case=stem,head=name,rank=int(model.rank_),solver='sklearn LinearRegression dense scipy.linalg.lstsq',sklearn=sklearn.__version__,scipy=scipy.__version__,max_prediction_absolute_difference=error,max_metric_absolute_difference=metric_error,coefficients_identity_required=False))
    output.mkdir(parents=True,exist_ok=True)
    summary = dict(status='passed',operation='packaged_saved_split_replay',selection=manifest['selection'],cases=len(metadata['cases']),score_rows=len(rows),maximum_metric_absolute_difference=max(discrepancies),atol=ATOL,rtol=RTOL,raw_preprocessing='fail',scientific_status='CLOSED_EXPLORATORY_WITH_LIMITATIONS',external_storage_accessed=False,representation_refit=False,OLS_refit=refit,full_analysis_rebuilt=False,limitation='One saved split and allocation; does not reproduce the 100-split or 1000-randomization distributions.',environment=dict(python=platform.python_version(),platform={'win32':'Windows','linux':'Linux','darwin':'macOS'}.get(sys.platform,sys.platform),numpy=np.__version__),optional_refits=refits)
    write_json(output/'scores.json',rows)
    write_json(output/'summary.json',summary)
    return summary
