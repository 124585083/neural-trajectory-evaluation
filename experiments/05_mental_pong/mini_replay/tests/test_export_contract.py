"""Inspect real export identities and provenance without accessing source roots."""
import json
import importlib.util
from pathlib import Path
import re

import numpy as np
import pytest

from conftest import replay


def test_private_export_cannot_enter_another_repository_directory(tmp_path, monkeypatch):
    module = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('mini_export_contract', module / 'scripts/export_mini_replay.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    repository = tmp_path / 'repository'
    monkeypatch.setattr(exporter, 'REPOSITORY', repository)
    output = repository / 'another_module' / 'payload'
    with pytest.raises(ValueError, match='explicit passed rights record'):
        exporter.export(tmp_path / 'preserved_source', output)
    assert not output.exists()


def test_export_has_complete_fixed_cases_and_safe_numeric_arrays(real_bundle):
    manifest,arrays,meta=replay.load_bundle(real_bundle)
    assert {(r['animal'],r['representation']) for r in meta['cases']}=={(a,r) for a in ('mahler','perle') for r in ('FA50','GPFA50')}
    assert all(r['split']==r['q']==0 and r['half']=='half1' for r in meta['cases'])
    assert all(v.dtype.kind in 'bifu' for v in arrays.values())
    assert len(meta['source_manifest']['parents'])==27
    assert all(len(p['sha256'])==64 and p['source_artifact_id'].startswith('source_pilot://') for p in meta['source_manifest']['parents'])
    assert sum(r['bytes'] for r in manifest['files'].values())<25*1024**2


def test_export_keeps_invalid_condition_and_separates_train_test(real_bundle):
    _,arrays,meta=replay.load_bundle(real_bundle)
    split=meta['condition_splits']
    assert len(split['train_indices'])==39 and len(split['test_indices'])==40
    assert not set(split['train_indices'])&set(split['test_indices'])
    for animal in ('mahler','perle'):
        ids=arrays[animal+'__condition_ids'];mask=arrays[animal+'__common_mask']
        assert len(ids)==79 and mask.any(1).sum()==78
        assert not mask[ids==59920].any()
        assert all(r['valid_train_conditions']==39 and r['valid_test_conditions']==39 for r in meta['cases'] if r['animal']==animal)


def test_export_contains_no_private_paths_or_nonstandard_json(real_bundle):
    for path in [real_bundle/'manifest.json',*(real_bundle/'bundle').glob('*.json')]:
        text=path.read_text(encoding='utf-8')
        json.loads(text,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
        assert not re.search(r'(?<![A-Za-z_])[A-Za-z]:[/\\]',text)
        assert not re.search(r'[\u3400-\u9fff]',text)
