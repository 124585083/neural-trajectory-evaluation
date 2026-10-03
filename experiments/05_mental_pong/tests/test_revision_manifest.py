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
    file.parent.mkdir(parents=True)
    with file.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['destination', 'baseline_sha256', 'publication_sha256'])
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
