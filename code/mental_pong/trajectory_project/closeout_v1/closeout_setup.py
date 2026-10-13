"""Frozen closeout protocol. All earlier projects are read-only sources."""
from pathlib import Path
from datetime import datetime, timezone
import platform
import numpy as np
import scipy
import sklearn
import pandas as pd
from trajectory_project.step3_io import read_json, write_json, record, sha, collect_records, verify_records

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'condition_endpoint_fa_gpfa_v1'
TRIAL = ROOT.parent / 'trial_endpoint_fa_gpfa_v1'
ANIMALS = ('mahler', 'perle')
REPS = ('FA50', 'GPFA50')
EPOCHS = ('full', 'visible', 'hidden', 'bounce', 'no_bounce', 'post_bounce')


def initialize():
    for folder in ('configs', 'sources', 'results/A', 'results/B', 'results/C',
                   'results/descriptive', 'artifacts/B', 'artifacts/C', 'figures', 'logs', 'tests'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    previous = read_json(SOURCE / 'manifest.json')
    assert previous['completed'] and not previous['strict_raw_future_free']
    protocol = {
        'version': 'closeout_v1', 'source_version': SOURCE.name,
        'animals': list(ANIMALS), 'representations': list(REPS),
        'representation_policy': 'Read existing per-round train39 FA50/shared-one-learnable-RBF-timescale GPFA50 half1 latents; no refit or new inference.',
        'n_features': 50, 'coefficients_per_coordinate': 51, 'coefficients_per_xy_head': 102,
        'OLS': {'class': 'sklearn.linear_model.LinearRegression', 'fit_intercept': True,
                'positive': False, 'sample_weight': None, 'regularization': None,
                'scaler': None, 'precision': 'float64'},
        'splits': {'n_splits': 100, 'train_identities': 39, 'test_identities': 40,
                   'random_state': 0, 'reuse': str(SOURCE / 'configs/condition_splits_100.json')},
        'data': {'n_condition_identities': 79, 'valid_conditions_per_animal': 78,
                 'unknown_condition_id': 59920, 'valid_condition_time_rows': 3369,
                 'behavior_records_in_means': {'mahler': 7407, 'perle': 84873},
                 'neural_trial_members': 'unavailable', 'independent_neural_trial_N': None,
                 'row_unit': 'one condition x time, no behavioral trial weighting',
                 'time': 'original 50 ms completed-bin right edge; unchanged mixed/feedback masks'},
        'label_definition': 'Current fixed-membership condition mean terminal paddle endpoint; objective before collision and linear endpoint connection after original collision anchor; initial anchor for no-bounce. No stopping or update detection.',
        'epochs': list(EPOCHS), 'coordinates': ['x', 'y'],
        'A': {'type': 'cross-evaluation, not a random null',
              'cells': {'OO': ['D_obj', 'objective'], 'OB': ['D_obj', 'behavior'],
                        'BO': ['D_beh', 'objective'], 'BB': ['D_beh', 'behavior']},
              'primary': ['OO versus BB'], 'case_ids_posthoc': [55062, 241919]},
        'B': {'repeats': 1000, 'seeds': {'mahler': 314159, 'perle': 314160},
              'rng': 'numpy.random.default_rng PCG64', 'refit': False,
              'mapping': 'One permutation per q/split among that split valid test conditions; shared by representations, heads and targets; animals independently randomized.',
              'support': 'Exact common timestamps, both conditions valid and both in the scored epoch; matched rescored on every shuffled support.',
              'primary_effects': ['matched_r-shuffled_r', 'shuffled_RMSE-matched_RMSE'],
              'tail_interpretation': 'Empirical random-control proportions, not exact fixed-support permutation p values; report map uniqueness and support variation.'},
        'C': {'repeats': 1000, 'seeds': {'mahler': 271828, 'perle': 271829},
              'rng': 'numpy.random.default_rng PCG64', 'refit': True,
              'mapping': 'One permutation of 78 existing mean endpoints per animal/q; fixed across all 100 splits and times; shared by FA and GPFA.',
              'geometry': 'Keep each condition original anchors, branch, x, T and mask; audit all endpoint assignments before neural scoring; no clipping or rejection based on scores.',
              'fitting': 'Actual new float64 unweighted intercept OLS for random targets; batched multioutput solves permitted after numeric equivalence to separate original LinearRegression.',
              'additional_epoch': 'endpoint_influence: original post-bounce window for bounce, bins after original initial anchor for no-bounce; detailed mask audit frozen before scores.',
              'baseline': 'Each split/label uses training-condition mean endpoint with test condition own geometry. skill=1-SSE(neural,label)/SSE(geometry_mean,label); zero denominator NA.',
              'saved_example_q': [0, 1, 2],
              'tail_interpretation': 'Empirical proportions for this endpoint reassignment rule; do not claim randomized animal experiments or universal trajectory difficulty matching.'},
        'aggregation': 'Score each original heldout split first; mean and population SD over 100 overlapping splits. For each q first average its 100 splits, yielding 1000 random statistics. Constant/insufficient Pearson r is NA, never zero.',
        'descriptive_audit': 'Use exact existing behavioral mean members. Endpoint sign zero is exact floating zero; report variance identity with numeric tolerance only. No favorable condition selection.',
        'future_design_policy': 'Documentation only; no trial-group neural reconstruction, new model search or downloads.',
        'causal_scope': {'raw_preprocessing': 'fail', 'provided_input_filtering': 'pass relative to published inputs only',
                         'fit_provenance': 'pass relative to published inputs only',
                         'upstream_issue': 'Published global-condition/time imputation is not reversed by this closeout.',
                         'terminal_behavior': 'Strict pre-feedback timestamp unverified',
                         'collision': 'Estimated from released 50 ms trajectory, not precise trial event logs'},
        'completion_rule': 'CLOSED_EXPLORATORY_WITH_LIMITATIONS only after A, B1000 and C1000 complete and all source/model preservation checks pass; otherwise PARTIAL.'}
    write_json(ROOT / 'configs/closeout_protocol.json', protocol, immutable=True)
    lock = ROOT / 'configs/protocol_lock.json'
    if not lock.exists():
        write_json(lock, {'sha256': sha(ROOT / 'configs/closeout_protocol.json'),
                         'frozen_utc': datetime.now(timezone.utc).isoformat(),
                         'before_new_random_scores': True}, immutable=True)
    assert read_json(lock)['sha256'] == sha(ROOT / 'configs/closeout_protocol.json')
    split_path = ROOT / 'configs/condition_splits_100.json'
    write_json(split_path, read_json(SOURCE / 'configs/condition_splits_100.json'), immutable=True)
    assert sha(split_path) == sha(SOURCE / 'configs/condition_splits_100.json')
    sources = [record(SOURCE / p, 'unchanged current condition-mean source') for p in (
        'manifest.json', 'REPORT.md', 'README.md', 'configs/analysis_protocol.json',
        'configs/condition_splits_100.json', 'results/previous_trial_version_comparison.csv',
        'results/self_reconstruction_main_table.csv', 'results/cross_2x2_summary.csv',
        'condition_data.py', 'decoding_condition.py', 'sources/source_trial_membership.json',
        'results/condition_geometry.csv', 'results/reused_representation_audit.json')]
    sources += [record(SOURCE / 'artifacts' / f'{a}_condition_labels.npz') for a in ANIMALS]
    sources += [record(TRIAL / 'manifest.json', 'unchanged original trial-version manifest')]
    write_json(ROOT / 'sources/frozen_source_records.json', sources, immutable=True)
    write_json(ROOT / 'sources/runtime.json', {
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'libraries_upgraded': False}, immutable=True)
    return protocol


def verify_sources():
    verify_records(read_json(ROOT / 'sources/frozen_source_records.json'))
    counts = {}
    for path in (SOURCE, TRIAL):
        counts[path.name] = verify_records(collect_records(read_json(path / 'manifest.json')))
    assert sha(ROOT / 'configs/closeout_protocol.json') == read_json(ROOT / 'configs/protocol_lock.json')['sha256']
    return counts


if __name__ == '__main__':
    initialize()
    counts = verify_sources()
    write_json(ROOT / 'sources/initial_source_verification.json', {
        'checked_utc': datetime.now(timezone.utc).isoformat(), 'verified_unchanged': counts})
    print(counts, flush=True)
