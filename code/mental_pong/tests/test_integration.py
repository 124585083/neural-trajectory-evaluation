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


def test_project_local_bundle_is_default_only_without_explicit_paths(tmp_path):
    module = tmp_path / 'code/mental_pong'
    module.mkdir(parents=True)
    bundle = tmp_path / 'models/mental_pong/current'
    bundle.mkdir(parents=True)
    (bundle / 'bundle_manifest.json').write_text(json.dumps(
        {'schema': 'mental_pong_local_artifacts_v1', 'complete': True}))
    access = entry.ArtifactAccess(root=module, environ={})
    assert access.resolve('source_pilot', 'example') == bundle / 'source_artifacts/example'
    explicit = entry.ArtifactAccess(tmp_path / 'absent.json', root=module, environ={})
    with pytest.raises(FileNotFoundError):
        explicit.resolve('source_pilot', 'example')
    configured = entry.ArtifactAccess(root=module, environ={'MENTAL_PONG_SOURCE_ROOT': str(tmp_path / 'external')})
    assert configured.resolve('source_pilot', 'example') == tmp_path / 'external/example'


@pytest.mark.parametrize('prefix', ['experiments/05_mental_pong/trajectory_project/',
                                    'code/mental_pong/trajectory_project/'])
def test_current_evidence_layout_keeps_versions_and_figure_names(tmp_path, prefix):
    for version in ('condition_endpoint_fa_gpfa_v1', 'closeout_v1'):
        for old, new, name in (('results', 'evidence', 'A/scores.csv'),
                               ('figures', 'figures', 'A_curves.png')):
            assert entry.current_evidence_path(tmp_path, f'{prefix}{version}/{old}/{name}') == (
                tmp_path / 'results/mental_pong' / new / version / name)
    config = prefix + 'closeout_v1/configs/closeout_protocol.json'
    assert entry.current_evidence_path(tmp_path, config) == (
        tmp_path / 'code/mental_pong/trajectory_project/closeout_v1/configs/closeout_protocol.json')


def test_excluded_artifacts_are_checked_at_both_layouts(tmp_path):
    historical = 'experiments/05_mental_pong/mini_replay/bundle/held.npz'
    current = tmp_path / 'code/mental_pong/mini_replay/bundle/held.npz'
    current.parent.mkdir(parents=True)
    current.write_bytes(b'synthetic rights-held fixture')
    candidates = entry.excluded_publication_paths(tmp_path, historical)
    assert tmp_path / historical in candidates
    assert any(path.is_file() for path in candidates)


@pytest.mark.parametrize('invalid_document,content', [
    (None, None),
    ('01_task_motion/SETUP.md', None),
    ('01_task_motion/RESULTS.md', ' \n'),
    ('02_behavior_relation/SETUP.md', None),
    ('02_behavior_relation/RESULTS.md', ''),
])
def test_current_verification_accepts_final_documents_and_requires_all_four(tmp_path, invalid_document, content):
    module = tmp_path / 'code/mental_pong'
    integration = module / 'integration'
    integration.mkdir(parents=True)
    migration = integration / 'migration_map.csv'
    migration.write_text('destination,destination_sha256\n', encoding='utf-8')
    for goal in ('01_task_motion', '02_behavior_relation'):
        for name in ('SETUP.md', 'RESULTS.md'):
            document = tmp_path / 'results/mental_pong' / goal / name
            document.parent.mkdir(parents=True, exist_ok=True)
            document.write_text(f'# {goal}: {name}\n\nFinal research narrative.\n', encoding='utf-8')
    if invalid_document is not None:
        document = tmp_path / 'results/mental_pong' / invalid_document
        if content is None:
            document.unlink()
        else:
            document.write_text(content, encoding='utf-8')
    access = entry.ArtifactAccess(root=module, environ={})
    access.records = [{'artifact_id': 'synthetic', 'storage_location_id': 'published_module',
                       'relative_path': 'integration/migration_map.csv', 'sha256': entry.digest(migration)}]
    result = entry.verify(access)
    expected = [] if invalid_document is None else [f'Missing or empty goal document: {invalid_document}']
    assert result['failures'] == expected
    assert result['status'] == ('failed' if expected else 'passed_with_availability_report')
    assert result['new_model_fitting'] is False and result['files_written'] is False


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
