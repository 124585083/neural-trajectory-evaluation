"""Portable adapter checks; no source analysis runner is invoked."""
from pathlib import Path
import importlib.util
import hashlib
import json
import os
import subprocess
import sys

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mental_pong_integrated_entry', MODULE_ROOT/'run.py')
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def test_alias_cannot_escape_storage_root(tmp_path):
    config=tmp_path/'paths.json'
    config.write_text(json.dumps({'artifact_roots':{'source_pilot':str(tmp_path)}}))
    access=entry.ArtifactAccess(config,root=MODULE_ROOT,environ={})
    assert access.resolve('source_pilot://folder/file.npz')==tmp_path/'folder/file.npz'
    for bad in ('../file','x/../../file','C:/private/file','/absolute/file'):
        with pytest.raises(ValueError):access.resolve('source_pilot',bad)


def test_environment_override_and_missing_alias(tmp_path):
    access=entry.ArtifactAccess(tmp_path/'missing.json',root=MODULE_ROOT,
        environ={'MENTAL_PONG_SOURCE_ROOT':str(tmp_path)})
    assert access.resolve('source_pilot','file')==tmp_path/'file'
    with pytest.raises(FileNotFoundError):access.resolve('source_data','file')


def test_full_plan_is_read_only_and_does_not_claim_execution(tmp_path):
    access=entry.ArtifactAccess(tmp_path/'missing.json',root=MODULE_ROOT,environ={})
    before=set(tmp_path.iterdir())
    result=entry.full_plan(access)
    assert result['executed'] is False
    assert set(tmp_path.iterdir())==before
    assert any('not been exercised' in x for x in result['limits'])


def test_entrypoint_works_outside_repository_without_path_mutation(tmp_path):
    previous=list(sys.path)
    proc=subprocess.run([sys.executable,'-B',str(MODULE_ROOT/'run.py'),'--full-plan'],
        cwd=tmp_path,capture_output=True,text=True,check=True)
    result=json.loads(proc.stdout)
    assert result['executed'] is False
    assert list(sys.path)==previous
    assert not list(tmp_path.iterdir())


def test_missing_data_returns_precise_unavailable_without_writing(tmp_path):
    proc=subprocess.run([sys.executable,'-B',str(MODULE_ROOT/'run.py'),'--replay','--paths',str(tmp_path/'absent.json')],
        cwd=tmp_path,capture_output=True,text=True,
        env={k:v for k,v in os.environ.items() if not k.startswith('MENTAL_PONG_')})
    assert proc.returncode==1
    result=json.loads(proc.stdout)
    assert result['status']=='unavailable_or_failed'
    assert 'source_pilot' in result['reason']
    assert result['files_written'] is False
    assert not list(tmp_path.iterdir())


def test_small_replay_uses_only_registered_inputs():
    access=entry.ArtifactAccess(root=MODULE_ROOT)
    if 'source_pilot' not in access.roots or not access.records:
        pytest.skip('External saved artifacts or their registry are unavailable.')
    try:
        result=entry.replay(access,'mahler','FA50',0,0)
    except FileNotFoundError as exc:
        pytest.skip(f'External saved replay artifact unavailable: {exc}')
    assert result['status']=='passed'
    assert result['max_score_absolute_difference']<1e-8
    assert result['new_model_fitting'] is False
    assert result['files_written'] is False
    assert result['raw_preprocessing']=='fail'


def _relocated_manifest(tmp_path, paths=None, content=b'verified published member'):
    raw=tmp_path/'raw'; member=raw/'download'/'member.pkl'
    member.parent.mkdir(parents=True); member.write_bytes(content)
    original_paths=paths or ['Z:\\original-host\\MentalPong\\raw\\download\\member.pkl']
    manifest={'archive':{'verified':True,'md5':'382e420c6f053d8ac2bdd4acd2fb02b2'},
        'extracted_files':[{'path':p,'sha256':hashlib.sha256(content).hexdigest()} for p in original_paths]}
    manifest_path=raw/'download_manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    return member,manifest_path,manifest_path.read_bytes()


def _verified_file():
    spec=importlib.util.spec_from_file_location('mental_pong_relocated_data',MODULE_ROOT/'data_pipeline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.verified_file


def test_raw_manifest_relocation_preserves_hash_and_source_manifest(tmp_path):
    member,manifest,before=_relocated_manifest(tmp_path)
    assert _verified_file()(tmp_path,'member.pkl')==member.resolve()
    assert manifest.read_bytes()==before


def test_raw_manifest_relocation_rejects_changed_member(tmp_path):
    member,manifest,before=_relocated_manifest(tmp_path)
    member.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Member changed'):
        _verified_file()(tmp_path,'member.pkl')
    assert manifest.read_bytes()==before


@pytest.mark.parametrize('paths,error',[
    (['Z:/old/raw/a/member.pkl','Z:/old/raw/b/member.pkl'],'Expected one verified'),
    (['Z:/old/raw/another/raw/member.pkl'],'unambiguous raw'),
    (['Z:/old/raw/../member.pkl'],'Unsafe verified member')])
def test_raw_manifest_relocation_rejects_ambiguous_or_unsafe_paths(tmp_path,paths,error):
    _relocated_manifest(tmp_path,paths)
    with pytest.raises(ValueError,match=error):
        _verified_file()(tmp_path,'member.pkl')
