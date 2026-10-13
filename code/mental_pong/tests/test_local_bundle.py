"""Synthetic checks of the local allowlist; no scientific arrays are generated."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('mental_pong_local_package',
    Path(__file__).resolve().parents[1] / 'scripts/prepare_local_bundle.py')
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


@pytest.mark.parametrize('name', ['../private.txt', '/absolute/file.npz',
    'trajectory_project/x/../../file.json', 'private_notes.json', 'trajectory_project/prompt.md'])
def test_allowlist_rejects_unsafe_or_unrelated_sources(name):
    with pytest.raises(ValueError):
        bundle.safe_relative(name)


def test_verification_uses_local_files_and_rejects_extras(tmp_path):
    relative = 'trajectory_project/example/config.json'
    path = tmp_path / 'source_artifacts' / relative
    path.parent.mkdir(parents=True); path.write_text('{}')
    record = {'schema': bundle.SCHEMA, 'complete': True,
              'files': {relative: {'sha256': bundle.sha(path), 'bytes': path.stat().st_size}}}
    (tmp_path / 'bundle_manifest.json').write_text(json.dumps(record))
    assert bundle.verify(tmp_path)['source_pilot_accessed'] is False
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='changed'):
        bundle.verify(tmp_path)
    path.write_text('{}')
    path.with_name('unlisted.json').write_text('{}')
    with pytest.raises(ValueError, match='Unexpected files'):
        bundle.verify(tmp_path)


def test_changed_inventory_does_not_extend_a_completed_bundle(tmp_path, monkeypatch):
    source, destination = tmp_path / 'original', tmp_path / 'completed'
    rows = []
    for name in ('existing', 'added'):
        relative = f'trajectory_project/example/{name}.json'
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
        rows.append({'relative_path': relative, 'bytes': path.stat().st_size,
                     'sha256': bundle.sha(path), 'purpose': 'synthetic'})
    existing = destination / 'source_artifacts' / rows[0]['relative_path']
    existing.parent.mkdir(parents=True); existing.write_text('{}')
    manifest = destination / 'bundle_manifest.json'
    manifest.write_text(json.dumps({'schema': bundle.SCHEMA, 'complete': True,
                                    'files': bundle.file_records(rows[:1])}))
    before = {p.relative_to(destination): p.read_bytes() for p in destination.rglob('*') if p.is_file()}
    monkeypatch.setattr(bundle, 'inventory', lambda: (rows, tmp_path / 'inventory.json'))
    monkeypatch.setattr(bundle, 'ignored_destination', lambda _: destination)
    with pytest.raises(ValueError, match='inventory differs'):
        bundle.prepare(source)
    after = {p.relative_to(destination): p.read_bytes() for p in destination.rglob('*') if p.is_file()}
    assert after == before


def directory_link(target, link):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        try:
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        except (ImportError, AttributeError, OSError):
            pytest.skip('Directory links are unavailable on this system')


@pytest.mark.parametrize('escaped_root', ['source', 'destination'])
def test_preparation_rejects_escaped_children_before_writing(tmp_path, monkeypatch, escaped_root):
    source, destination, outside = (tmp_path / name for name in ('source', 'bundle', 'outside'))
    outside.mkdir()
    relative = 'trajectory_project/example/config.json'
    external = outside / 'config.json'
    external.write_text('{}')
    rows = [{'relative_path': relative, 'bytes': external.stat().st_size,
             'sha256': bundle.sha(external), 'purpose': 'synthetic'}]
    source_file = source / relative
    if escaped_root == 'source':
        source_file.parent.parent.mkdir(parents=True)
        directory_link(outside, source_file.parent)
    else:
        source_file.parent.mkdir(parents=True)
        source_file.write_text('{}')
        output_parent = destination / 'source_artifacts/trajectory_project/example'
        output_parent.parent.mkdir(parents=True)
        directory_link(outside, output_parent)
    before = external.read_bytes()
    monkeypatch.setattr(bundle, 'inventory', lambda: (rows, tmp_path / 'inventory.json'))
    monkeypatch.setattr(bundle, 'ignored_destination', lambda _: destination)
    with pytest.raises(ValueError, match='escapes'):
        bundle.prepare(source)
    assert external.read_bytes() == before
    assert not (destination / 'bundle_manifest.json').exists()


def test_verification_rejects_external_bundle_and_mini_children(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    path = outside / 'config.json'
    path.write_text('{}')
    root = tmp_path / 'bundle'
    parent = root / 'source_artifacts/trajectory_project'
    parent.mkdir(parents=True)
    directory_link(outside, parent / 'example')
    relative = 'trajectory_project/example/config.json'
    (root / 'bundle_manifest.json').write_text(json.dumps({
        'schema': bundle.SCHEMA, 'complete': True,
        'files': {relative: {'sha256': bundle.sha(path), 'bytes': path.stat().st_size}}}))
    with pytest.raises(ValueError, match='escapes'):
        bundle.verify(root)
    mini = tmp_path / 'mini'
    mini.mkdir()
    directory_link(outside, mini / 'payload')
    (mini / 'manifest.json').write_text(json.dumps({'files': {'payload/config.json': {}}}))
    with pytest.raises(ValueError, match='escapes'):
        bundle.verify_mini(mini)
