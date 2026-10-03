"""Apply a recorded publication revision without altering historical source hashes."""
from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath
import re


def publication_rows(module_root, legacy_rows):
    """Chain revised publication checksums to the original integration manifest."""
    revision = Path(module_root) / 'integration/revision_20261003/publication_map.csv'
    if not revision.is_file():
        return legacy_rows
    with revision.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    previous = {row['destination']: row['destination_sha256'] for row in legacy_rows
                if row.get('destination') and row.get('destination_sha256')}
    revised = {}
    for row in rows:
        name = row['destination']
        path = PurePosixPath(name)
        if not name or path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
            raise ValueError(f'Unsafe revised publication path: {name}')
        if name in revised:
            raise ValueError(f'Duplicate revised publication path: {name}')
        checksum = row['publication_sha256']
        if not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError(f'Invalid revised publication checksum: {name}')
        if name in previous and row['baseline_sha256'] != previous[name]:
            raise ValueError(f'Revision baseline differs from historical publication checksum: {name}')
        revised[name] = {'destination': name, 'destination_sha256': checksum}
    missing = set(previous) - set(revised)
    if missing:
        raise ValueError(f'Revision omits historical publication files: {sorted(missing)}')
    return list(revised.values())
