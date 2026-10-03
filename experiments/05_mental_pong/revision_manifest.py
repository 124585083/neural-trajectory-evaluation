"""Apply a recorded publication revision without altering historical source hashes."""
from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath
import re

MAP_PATH = 'experiments/05_mental_pong/integration/revision_20261003/publication_map.csv'
PUBLIC_DISPOSITIONS = {'KEEP_PUBLIC', 'EDIT_PUBLIC'}
# These exclusions are bounded distribution decisions, not permission to
# retire an arbitrary result, protocol, checkpoint or source dependency.
RETIREMENTS = {
    'models/official_dynamic/best.pt': 'RIGHTS_HOLD',
    'models/static_on_dynamic/best.pt': 'RIGHTS_HOLD',
    'models/parameter_matched_dynamic/best.pt': 'RIGHTS_HOLD',
    'models/parameter_matched_dynamic/epoch_65_validation_matched.pth': 'RIGHTS_HOLD',
    'models/gpfa_reliability/gpfa.pkl': 'RIGHTS_HOLD',
    'models/gpfa_reliability/preprocessing.npz': 'RIGHTS_HOLD',
    'models/gpfa_model_comparison/gpfa.pkl': 'RIGHTS_HOLD',
    'models/gpfa_model_comparison/preprocessing.npz': 'RIGHTS_HOLD',
    'experiments/05_mental_pong/integration/editorial_changes.md': 'PRIVATE_ONLY',
    'experiments/05_mental_pong/integration/revision_20261003/starting_state.json': 'PRIVATE_ONLY',
    'experiments/05_mental_pong/trajectory_project/closeout_v1/sources/user_request_record.json': 'PRIVATE_ONLY',
    'experiments/05_mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/artifacts/mahler_condition_labels.npz': 'RIGHTS_HOLD',
    'experiments/05_mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/artifacts/perle_condition_labels.npz': 'RIGHTS_HOLD',
    'experiments/05_mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/data/mahler_condition_time_index.csv': 'RIGHTS_HOLD',
    'experiments/05_mental_pong/trajectory_project/condition_endpoint_fa_gpfa_v1/data/perle_condition_time_index.csv': 'RIGHTS_HOLD',
}
def read_publication_map(module_root, legacy_rows=()):
    """Validate explicit public membership and the bounded retirement records.

    Historical baseline hashes remain unchanged. A sanitized technical edition
    records the hash of its externally preserved original separately. The map
    itself has one explicit member row without a recursive self-hash.
    """
    revision = Path(module_root) / 'integration/revision_20261003/publication_map.csv'
    with revision.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    previous = {row['destination']: row['destination_sha256'] for row in legacy_rows
                if row.get('destination') and row.get('destination_sha256')}
    seen = set()
    explicit = any('publication_disposition' in row for row in rows)
    for row in rows:
        name = row['destination']
        path = PurePosixPath(name)
        if not name or path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
            raise ValueError(f'Unsafe revised publication path: {name}')
        if name in seen:
            raise ValueError(f'Duplicate revised publication path: {name}')
        seen.add(name)
        if name in previous and row['baseline_sha256'] != previous[name]:
            raise ValueError(f'Revision baseline differs from historical publication checksum: {name}')
        disposition = row.get('publication_disposition', 'KEEP_PUBLIC')
        checksum = row['publication_sha256']
        change_class = row.get('change_class', '')
        if 'review_coverage' in row:
            raise ValueError('Editorial coverage belongs in an external review ledger.')
        if disposition in ('PRIVATE_ONLY', 'RIGHTS_HOLD'):
            if RETIREMENTS.get(name) != disposition or change_class != 'intentional_retirement':
                raise ValueError(f'Retirement is outside the bounded distribution exclusions: {name}')
            if checksum or not row.get('reason'):
                raise ValueError(f'Retired files need an empty publication hash and a neutral reason: {name}')
        elif disposition not in PUBLIC_DISPOSITIONS:
            raise ValueError(f'Unknown publication disposition: {name}')
        elif change_class == 'manifest_self':
            if name != MAP_PATH or checksum:
                raise ValueError('Only the publication map may omit its recursive self-hash.')
        elif not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError(f'Invalid revised publication checksum: {name}')
        if change_class in ('intentional_retirement', 'sanitized_technical_edition'):
            if not re.fullmatch('[0-9a-f]{64}', row.get('preserved_original_sha256', '')):
                raise ValueError(f'Missing externally preserved original hash: {name}')
        if change_class in ('unchanged', 'unchanged_scientific_artifact'):
            baseline = row.get('baseline_sha256')
            if baseline and checksum != baseline:
                raise ValueError(f'An unchanged artifact has a different publication hash: {name}')
    missing = set(previous) - seen
    if missing:
        raise ValueError(f'Revision omits historical publication files: {sorted(missing)}')
    if explicit and not any(r['destination'] == MAP_PATH and r.get('change_class') == 'manifest_self' for r in rows):
        raise ValueError('Explicit publication membership requires the map self-exclusion row.')
    return rows


def publication_rows(module_root, legacy_rows):
    """Chain revised publication checksums to the original integration manifest."""
    revision = Path(module_root) / 'integration/revision_20261003/publication_map.csv'
    if not revision.is_file():
        return legacy_rows
    return [{'destination': row['destination'], 'destination_sha256': row['publication_sha256']}
            for row in read_publication_map(module_root, legacy_rows)
            if row.get('publication_disposition', 'KEEP_PUBLIC') in PUBLIC_DISPOSITIONS
            and row.get('change_class') != 'manifest_self']
