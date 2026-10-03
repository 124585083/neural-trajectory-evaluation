"""Publication revisions must retain the historical checksum chain."""
import csv
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'mental_pong_revision_manifest', Path(__file__).resolve().parents[1] / 'revision_manifest.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

OLD = [{'destination': 'README.md', 'destination_sha256': 'a' * 64}]


def save(root, rows):
    file = root / 'integration/revision_20261003/publication_map.csv'
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('w', newline='', encoding='utf-8') as stream:
        names = ['destination', 'baseline_sha256', 'publication_sha256']
        names += sorted({key for row in rows for key in row} - set(names))
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    return file


def test_revision_chain_and_new_file_are_read_only(tmp_path):
    rows = [{'destination': 'README.md', 'baseline_sha256': 'a' * 64, 'publication_sha256': 'b' * 64},
            {'destination': 'docs/new.md', 'baseline_sha256': '', 'publication_sha256': 'c' * 64}]
    file = save(tmp_path, rows)
    before = file.read_bytes()
    result = module.publication_rows(tmp_path, OLD)
    assert result == [{'destination': 'README.md', 'destination_sha256': 'b' * 64},
                      {'destination': 'docs/new.md', 'destination_sha256': 'c' * 64}]
    assert file.read_bytes() == before and OLD[0]['destination_sha256'] == 'a' * 64


def test_no_revision_uses_original_manifest(tmp_path):
    assert module.publication_rows(tmp_path, OLD) is OLD


@pytest.mark.parametrize('rows,match', [
    ([{'destination': 'README.md', 'baseline_sha256': 'c' * 64, 'publication_sha256': 'b' * 64}], 'baseline differs'),
    ([], 'omits historical'),
    ([{'destination': '../README.md', 'baseline_sha256': '', 'publication_sha256': 'b' * 64}], 'Unsafe'),
    ([{'destination': 'README.md', 'baseline_sha256': 'a' * 64, 'publication_sha256': 'b' * 64}] * 2, 'Duplicate'),
])
def test_incomplete_or_broken_revision_chain_is_rejected(tmp_path, rows, match):
    save(tmp_path, rows)
    with pytest.raises(ValueError, match=match):
        module.publication_rows(tmp_path, OLD)


def explicit_row(name, *, disposition='KEEP_PUBLIC', kind='unchanged', baseline='a' * 64, checksum='a' * 64):
    return dict(destination=name, baseline_sha256=baseline, publication_sha256=checksum,
                publication_disposition=disposition,
                change_class=kind, preserved_original_sha256='a' * 64,
                reason='Synthetic contract fixture.')


def self_row():
    return explicit_row(module.MAP_PATH, kind='manifest_self', baseline='', checksum='')


def test_intentional_private_retirement_is_bounded_and_preserves_chain(tmp_path):
    name = next(name for name, value in module.RETIREMENTS.items() if value == 'PRIVATE_ONLY')
    original = [{'destination': name, 'destination_sha256': 'a' * 64}]
    retirement = explicit_row(name, disposition='PRIVATE_ONLY', kind='intentional_retirement', checksum='')
    file = save(tmp_path, [retirement, self_row()])
    before = file.read_bytes()
    assert module.publication_rows(tmp_path, original) == []
    assert file.read_bytes() == before


def test_rights_hold_is_limited_to_named_derived_inputs(tmp_path):
    name = next(name for name, value in module.RETIREMENTS.items() if value == 'RIGHTS_HOLD')
    retirement = explicit_row(name, disposition='RIGHTS_HOLD', kind='intentional_retirement', checksum='')
    save(tmp_path, [retirement, self_row()])
    assert module.publication_rows(tmp_path, [{'destination': name, 'destination_sha256': 'a' * 64}]) == []
    retirement['destination'] = 'results/protected_score.csv'
    save(tmp_path, [retirement, self_row()])
    with pytest.raises(ValueError, match='bounded distribution'):
        module.publication_rows(tmp_path, [])


def test_sanitized_edition_requires_original_hash(tmp_path):
    edition = explicit_row('README.md', disposition='EDIT_PUBLIC', kind='sanitized_technical_edition', checksum='b' * 64)
    save(tmp_path, [edition, self_row()])
    assert module.publication_rows(tmp_path, OLD)[0]['destination_sha256'] == 'b' * 64
    edition['preserved_original_sha256'] = ''
    save(tmp_path, [edition, self_row()])
    with pytest.raises(ValueError, match='preserved original'):
        module.publication_rows(tmp_path, OLD)


def test_public_map_does_not_republish_private_editorial_coverage(tmp_path):
    item = explicit_row('README.md')
    item['review_coverage'] = 'full_text_review'
    save(tmp_path, [item, self_row()])
    with pytest.raises(ValueError, match='external review ledger'):
        module.publication_rows(tmp_path, OLD)


def test_unchanged_artifact_identity_is_required(tmp_path):
    item = explicit_row('README.md', checksum='b' * 64)
    save(tmp_path, [item, self_row()])
    with pytest.raises(ValueError, match='unchanged artifact'):
        module.publication_rows(tmp_path, OLD)


def test_self_hash_exclusion_cannot_hide_a_required_file(tmp_path):
    item = explicit_row('README.md', kind='manifest_self', checksum='')
    save(tmp_path, [item, self_row()])
    with pytest.raises(ValueError, match='Only the publication map'):
        module.publication_rows(tmp_path, OLD)
