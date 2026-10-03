"""Verify Mental-Pong files or replay saved readout parameters.

The default command verifies publication files and registered artifact access.
The legacy replay reads saved parameters without writing. The packaged replay
writes checks to an explicit output directory; OLS refitting is an optional flag.
Historical completion and the current integration checks remain separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
CURRENT = 'trajectory_project/condition_endpoint_fa_gpfa_v1'
TRIAL = 'trajectory_project/trial_endpoint_fa_gpfa_v1'
CLOSEOUT = 'trajectory_project/closeout_v1'
ENV_ROOTS = {'source_pilot': 'MENTAL_PONG_SOURCE_ROOT',
             'source_data': 'MENTAL_PONG_DATA_ROOT',
             'preservation_snapshot': 'MENTAL_PONG_SNAPSHOT_ROOT'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


class ArtifactAccess:
    """Resolve explicit storage aliases without changing the original records."""
    def __init__(self, paths=None, root=ROOT, environ=None):
        self.root = Path(root).resolve()
        self.repository = self.root.parents[1]
        env = os.environ if environ is None else environ
        config_path = Path(paths) if paths else self.root / 'configs/paths.local.json'
        config = read_json(config_path) if config_path.is_file() else {}
        self.roots = {'published_module': self.root, 'repository': self.repository}
        for key, value in config.get('artifact_roots', {}).items():
            if value:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = config_path.parent / path
                self.roots[key] = path.resolve()
        for key, variable in ENV_ROOTS.items():
            if env.get(variable):
                self.roots[key] = Path(env[variable]).expanduser().resolve()
        registry_path = self.root / 'integration/artifact_registry.csv'
        self.records = read_csv(registry_path) if registry_path.is_file() else []

    def resolve(self, location, relative=None):
        if relative is None:
            if '://' not in location:
                raise ValueError('Artifact references require an explicit storage alias.')
            location, relative = location.split('://', 1)
        if location not in self.roots:
            raise FileNotFoundError(f'Unconfigured artifact storage alias: {location}')
        relative = str(relative).replace('\\', '/')
        if Path(relative).is_absolute() or ':' in relative or '..' in Path(relative).parts:
            raise ValueError(f'Artifact path escapes its storage root: {relative}')
        base = self.roots[location]
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f'Artifact path escapes its storage root: {relative}')
        return path

    def source(self, relative, verify=True):
        path = self.resolve('source_pilot', relative)
        if not path.is_file():
            raise FileNotFoundError(f'Unavailable source artifact: source_pilot://{relative}')
        records = [r for r in self.records if r['storage_location_id'] == 'source_pilot'
                   and r['relative_path'].replace('\\', '/') == relative]
        if verify:
            if len(records) != 1 or not records[0].get('sha256'):
                raise ValueError(f'Expected one registered checksum for source_pilot://{relative}')
            if digest(path) != records[0]['sha256']:
                raise ValueError(f'Source checksum mismatch: source_pilot://{relative}')
        return path


def verify(access, deep=False):
    migration_path = access.root / 'integration/migration_map.csv'
    published, artifacts, failures = [], [], []
    if not migration_path.exists():
        failures.append('integration/migration_map.csv is unavailable')
    else:
        spec = importlib.util.spec_from_file_location('mental_pong_publication_revision', ROOT / 'revision_manifest.py')
        revision = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(revision)
        try:
            legacy = read_csv(migration_path)
            publication = revision.publication_rows(access.root, legacy)
            map_path = access.root / 'integration/revision_20261003/publication_map.csv'
            if map_path.is_file():
                for row in revision.read_publication_map(access.root, legacy):
                    if row.get('publication_disposition') in ('PRIVATE_ONLY', 'RIGHTS_HOLD'):
                        path = access.repository / row['destination']
                        if path.exists() or path.is_symlink():
                            failures.append(f'Intentionally excluded file remains in publication tree: {row["destination"]}')
        except (ValueError, KeyError) as error:
            failures.append(f'Invalid publication revision manifest: {error}')
            publication = []
        for row in publication:
            destination = row.get('destination', '')
            if not destination or not row.get('destination_sha256'):
                continue
            path = (access.repository / destination).resolve()
            if not path.is_relative_to(access.repository):
                failures.append(f'Publication path escapes repository: {destination}')
                continue
            status = 'verified' if path.is_file() and digest(path) == row['destination_sha256'] else 'failed'
            published.append({'destination': destination, 'status': status})
            if status == 'failed':
                failures.append(f'Publication checksum mismatch or missing file: {destination}')
    if not access.records:
        failures.append('integration/artifact_registry.csv is unavailable or empty')
    for row in access.records:
        item = {'artifact_id': row['artifact_id'], 'storage_location_id': row['storage_location_id']}
        try:
            path = access.resolve(row['storage_location_id'], row['relative_path'])
            if not path.is_file():
                item['status'] = 'unavailable'
            elif deep or path.stat().st_size <= 16 * 1024 * 1024:
                item['status'] = 'verified' if digest(path) == row['sha256'] else 'failed'
                if item['status'] == 'failed':
                    failures.append(f'Artifact checksum mismatch: {row["artifact_id"]}')
            else:
                item['status'] = 'available_large_checksum_not_checked'
        except FileNotFoundError:
            item['status'] = 'unconfigured'
        except ValueError as exc:
            item.update(status='failed', reason=str(exc))
            failures.append(str(exc))
        artifacts.append(item)
    counts = {}
    for row in artifacts:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    return {'operation': 'read_only_verification', 'status': 'failed' if failures else 'passed_with_availability_report',
            'published_records_checked': len(published), 'artifact_status_counts': counts,
            'deep_external_checksums': deep, 'failures': failures,
            'unavailable_artifacts': [r for r in artifacts if r['status'] in ('unavailable', 'unconfigured')],
            'scientific_status': 'CLOSED_EXPLORATORY_WITH_LIMITATIONS', 'raw_preprocessing': 'fail',
            'new_model_fitting': False, 'files_written': False}


def replay(access, animal='mahler', representation='FA50', q=0, iteration=0):
    """Replay one saved random head and independently score its own labels."""
    import numpy as np
    if not 0 <= q < 1000 or not 0 <= iteration < 100:
        raise ValueError('q must be 0..999 and split must be 0..99.')
    labels_path = access.source(f'{CURRENT}/artifacts/{animal}_condition_labels.npz')
    split_path = access.source(f'{CURRENT}/configs/condition_splits_100.json')
    assignments_path = access.source(f'{CLOSEOUT}/artifacts/C/{animal}_endpoint_assignments.npz')
    stem = f'{animal}_{representation}'
    weight_path = access.source(f'{CLOSEOUT}/artifacts/C/{stem}_OLS_weights.npz')
    scores_path = access.source(f'{CLOSEOUT}/results/C/{stem}_all_scores.npz')
    latent_path = access.source(f'{TRIAL}/representations/{animal}/round_{iteration:03d}/latents.npz')
    with np.load(labels_path, allow_pickle=False) as labels:
        common = labels['common_mask']; ids = labels['condition_ids']; times = labels['times_ms']
        objective = labels['objective_xy'][..., 1]
    split = read_json(split_path)[iteration]
    train, test = np.asarray(split['train_indices']), np.asarray(split['test_indices'])
    assert len(train) == 39 and len(test) == 40 and not set(train) & set(test)
    mask = np.zeros_like(common); mask[test] = common[test]
    c, t = np.nonzero(mask)
    with np.load(latent_path, allow_pickle=False) as latent:
        np.testing.assert_array_equal(latent['condition_ids'], ids)
        np.testing.assert_array_equal(latent['times_ms'], times)
        assert np.all(latent['common_mask'][common])
        X = latent[representation][0, c, t].astype(np.float64)
    with np.load(assignments_path, allow_pickle=False) as saved:
        offsets, alpha = saved['offset'][c, t], saved['alpha'][c, t]
        target = offsets + alpha * saved['endpoints'][q, c]
        epochs = saved['epoch_names'].tolist()
        epoch_masks = saved['epoch_masks'][:, c, t]
        valid_train = train[common[train].any(1)]
        mean_endpoint = saved['endpoints'][q, valid_train].mean()
    with np.load(weight_path, allow_pickle=False) as weights:
        coef = weights['coefficients'][iteration, q]
        assert coef.shape == (50,)
        prediction = X @ coef + weights['intercepts'][iteration, q]
        np.testing.assert_allclose(weights['mean_train_endpoints'][iteration, q], mean_endpoint, atol=1e-14)
    baseline = offsets + alpha * mean_endpoint
    with np.load(scores_path, allow_pickle=False) as saved:
        expected = saved['random_scores'][q, iteration]
        metric_names = saved['metrics'].tolist()
    replayed = []
    maximum_error = 0.
    for e, epoch in enumerate(epochs):
        active = epoch_masks[e]
        y, p, b = target[active], prediction[active], baseline[active]
        r = float(np.corrcoef(y, p)[0, 1]) if len(y) >= 3 and np.ptp(y) and np.ptp(p) else float('nan')
        sse = np.sum((p-y)**2); baseline_sse = np.sum((b-y)**2)
        values = {'r': r, 'RMSE': float(np.sqrt(sse/len(y))),
                  'baseline_SSE': float(baseline_sse),
                  'skill': float(1-sse/baseline_sse) if baseline_sse else float('nan')}
        for name, value in values.items():
            original = expected[e, metric_names.index(name)]
            np.testing.assert_allclose(value, original, atol=1e-8, rtol=1e-12, equal_nan=True)
            if np.isfinite(value-original):
                maximum_error = max(maximum_error, abs(float(value-original)))
        replayed.append({'epoch': epoch, 'n_bins': len(y), **values})
    return {'operation': 'saved_random_head_replay', 'status': 'passed', 'animal': animal,
            'representation': representation, 'q': q, 'split': iteration, 'half': 'half1',
            'max_score_absolute_difference': maximum_error, 'scores': replayed,
            'new_model_fitting': False, 'files_written': False, 'raw_preprocessing': 'fail'}


def full_plan(access):
    """Describe the retained full runners without executing their write paths."""
    return {'operation': 'future_full_analysis_plan', 'executed': False,
        'required_inputs': ['Original verified Mental-Pong release and download manifest',
            'Prepared geometry, physical-event audit, and fixed behavioral trial membership',
            'All 200 stored FA50/GPFA50 representation folders and preprocessing',
            'The original condition labels, 100 splits, source manifests, and official audit',
            'An isolated writable analysis workspace with explicitly rebound storage paths'],
        'retained_entrypoints': [
            'trajectory_project/condition_endpoint_fa_gpfa_v1/run_analysis.py',
            'trajectory_project/closeout_v1/run_closeout.py'],
        'historical_commands_in_isolated_workspace': [
            'python -B -m trajectory_project.condition_endpoint_fa_gpfa_v1.run_analysis',
            'python -B -m trajectory_project.closeout_v1.run_closeout --all'],
        'limits': ['These commands write analyses and reports. Do not run them on the preserved source directories.',
            'The publication package registers large inputs externally; it does not materialize a full analysis workspace.',
            'Original freeze records refer to original implementation bytes. A new execution needs separate output and execution records.',
            'Full rebuild portability has not been exercised using only published files. Saved-weight replay has been checked with private artifacts.',
            'Listing these commands does not execute representation training or the 1,000-allocation analyses.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--verify', action='store_true', help='Read-only verification; the default action.')
    action.add_argument('--replay', action='store_true', help='Read and replay one saved random OLS head.')
    action.add_argument('--mini-replay', action='store_true', help='Replay the fixed packaged split without consulting external storage configuration.')
    action.add_argument('--full-plan', action='store_true', help='List full-analysis inputs and retained commands; do not execute them.')
    parser.add_argument('--paths', type=Path, help='Local path configuration; environment variables override its storage roots.')
    parser.add_argument('--deep', action='store_true', help='Hash all registered external files during verification.')
    parser.add_argument('--animal', choices=('mahler', 'perle'), default='mahler')
    parser.add_argument('--representation', choices=('FA50', 'GPFA50'), default='FA50')
    parser.add_argument('--q', type=int, default=0)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--output', type=Path, help='Required caller-selected output directory for --mini-replay.')
    parser.add_argument('--mini-bundle', type=Path, help='Optional privately exported mini package; the default is mini_replay beside this script.')
    parser.add_argument('--ols-refit', action='store_true', help='Explicit optional OLS refit of the mini package training rows.')
    args = parser.parse_args(argv)
    if (args.ols_refit or args.mini_bundle or args.output) and not args.mini_replay:
        parser.error('--output, --mini-bundle and --ols-refit apply only to --mini-replay.')
    try:
        if args.mini_replay:
            if args.output is None:
                parser.error('--mini-replay requires --output outside the immutable bundle.')
            from mini_replay.replay import run as mini_run
            result = mini_run(args.mini_bundle or ROOT/'mini_replay', args.output, args.ols_refit)
        elif args.replay:
            access = ArtifactAccess(args.paths)
            result = replay(access, args.animal, args.representation, args.q, args.split)
        elif args.full_plan:
            access = ArtifactAccess(args.paths)
            result = full_plan(access)
        else:
            access = ArtifactAccess(args.paths)
            result = verify(access, args.deep)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        result = {'status': 'unavailable_or_failed', 'reason': str(exc), 'files_written': False,
                  'new_model_fitting': False}
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=True))
    return 1 if result.get('status') in ('failed', 'unavailable_or_failed') else 0


if __name__ == '__main__':
    raise SystemExit(main())
