"""Small saved-case replay, tampering, and CPU offline cold-copy checks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import MINI, replay


def test_public_payload_status_is_honest():
    manifest=json.loads((MINI/'manifest.json').read_text())
    if manifest['payload_status'].startswith('withheld'):
        assert not (MINI/'bundle/arrays.npz').exists()
        with pytest.raises(FileNotFoundError,match='withheld'):
            replay.load_bundle(MINI)


def test_saved_cases_recompute_scores(real_bundle,tmp_path):
    result=replay.run(real_bundle,tmp_path/'output')
    assert result['cases']==4 and result['score_rows']==604
    assert result['maximum_metric_absolute_difference']<1e-8
    assert result['external_storage_accessed'] is False
    assert result['OLS_refit'] is False
    assert result['raw_preprocessing']=='fail'
    assert len(json.loads((tmp_path/'output/scores.json').read_text()))==604


def test_output_cannot_modify_bundle(real_bundle):
    with pytest.raises(ValueError,match='outside'):
        replay.run(real_bundle,real_bundle/'new_results')


def test_output_cannot_overwrite_existing_records(real_bundle,tmp_path):
    original=tmp_path/'scores.json';original.write_text('existing scientific record')
    with pytest.raises(ValueError,match='never overwritten'):
        replay.run(real_bundle,tmp_path)
    assert original.read_text()=='existing scientific record'


@pytest.mark.parametrize('key', ['mahler_FA50__features','mahler__common_mask','mahler_FA50__coefficients'])
def test_payload_tamper_is_rejected(real_bundle,tmp_path,key):
    copy=tmp_path/'copy';shutil.copytree(real_bundle,copy)
    path=copy/'bundle/arrays.npz'
    with np.load(path,allow_pickle=False) as z: values={k:z[k].copy() for k in z.files}
    values[key].flat[0]=not values[key].flat[0] if values[key].dtype.kind=='b' else values[key].flat[0]+1
    np.savez_compressed(path,**values)
    with pytest.raises(ValueError,match='checksum'):
        replay.run(copy,tmp_path/'output')


@pytest.mark.parametrize('key', ['mahler_FA50__features','mahler_FA50__coefficients'])
def test_changed_values_fail_scores_even_after_hash_refresh(real_bundle,tmp_path,key):
    copy=tmp_path/'copy';shutil.copytree(real_bundle,copy)
    path=copy/'bundle/arrays.npz'
    with np.load(path,allow_pickle=False) as z: values={k:z[k].copy() for k in z.files}
    if key.endswith('features'):
        # Change an actual test row; changing only a train row has no saved-weight effect.
        c=values['mahler__condition_index'];split=json.loads((copy/'bundle/condition_splits.json').read_text())
        row=np.flatnonzero(np.isin(c,split['test_indices']))[0]
        values[key][row,0]+=10
    else:values[key][0,1,0]+=1
    np.savez_compressed(path,**values)
    manifest=json.loads((copy/'manifest.json').read_text())
    manifest['files']['bundle/arrays.npz']={'sha256':replay.sha(path),'bytes':path.stat().st_size}
    manifest['arrays'][key]['sha256']=replay.array_sha(values[key])
    replay.write_json(copy/'manifest.json',manifest)
    with pytest.raises(ValueError,match='Score mismatch'):
        replay.run(copy,tmp_path/'output')


def test_file_hash_tamper_is_rejected(real_bundle,tmp_path):
    copy=tmp_path/'copy';shutil.copytree(real_bundle,copy)
    manifest=json.loads((copy/'manifest.json').read_text())
    manifest['files']['bundle/arrays.npz']['sha256']='0'*64
    replay.write_json(copy/'manifest.json',manifest)
    with pytest.raises(ValueError,match='checksum'):
        replay.run(copy,tmp_path/'output')


@pytest.mark.parametrize('field,value',[
    ('head_order',['behavior','objective','random']),
    ('coordinate_order',['y','x']),
    ('epoch_order',['visible','full','hidden','bounce','no_bounce','post_bounce','endpoint_influence']),
    ('coefficient_convention','prediction = coefficients @ features'),
])
def test_contradictory_convention_is_rejected(real_bundle,tmp_path,field,value):
    copy=tmp_path/'copy';shutil.copytree(real_bundle,copy)
    manifest=json.loads((copy/'manifest.json').read_text());manifest[field]=value
    replay.write_json(copy/'manifest.json',manifest)
    with pytest.raises(ValueError,match='convention'):
        replay.run(copy,tmp_path/'output')


def test_wrong_array_axis_labels_are_rejected(real_bundle,tmp_path):
    copy=tmp_path/'copy';shutil.copytree(real_bundle,copy)
    manifest=json.loads((copy/'manifest.json').read_text())
    manifest['arrays']['mahler_FA50__features']['axes']=['latent_dimension','row']
    replay.write_json(copy/'manifest.json',manifest)
    with pytest.raises(ValueError,match='axes mismatch'):
        replay.run(copy,tmp_path/'output')


def test_ols_refit_matches_saved_predictions(real_bundle,tmp_path):
    pytest.importorskip('sklearn')
    result=replay.run(real_bundle,tmp_path/'refit',refit=True)
    assert len(result['optional_refits'])==12
    assert all(r['max_prediction_absolute_difference']<1e-8 for r in result['optional_refits'])


def test_cold_copy_has_no_source_or_network_access(real_bundle,tmp_path):
    module=tmp_path/'public checkout with spaces'/'experiments'/'05_mental_pong'
    mini=module/'mini_replay';mini.mkdir(parents=True)
    shutil.copy2(MINI.parent/'run.py',module/'run.py')
    for name in ('__init__.py','replay.py'):shutil.copy2(MINI/name,mini/name)
    shutil.copytree(real_bundle/'bundle',mini/'bundle')
    shutil.copy2(real_bundle/'manifest.json',mini/'manifest.json')
    # Only this new checkout, the caller output, and Python's installed runtime
    # may be opened. The original source/data roots are not on the allowlist.
    wrapper = r'''
import json, os, pathlib, runpy, sys
checkout=pathlib.Path(sys.argv[1]).resolve()
allowed=[checkout.parent.parent.parent, pathlib.Path(sys.prefix).resolve(), pathlib.Path(sys.base_prefix).resolve()]
def audit(event,args):
    if event.startswith('socket.') and event not in ('socket.__new__','socket.gethostname'):
        raise RuntimeError('Network disabled for cold replay')
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=pathlib.Path(os.fsdecode(args[0])).resolve()
        if not any(p==root or p.is_relative_to(root) for root in allowed):
            raise RuntimeError('File outside public checkout and Python runtime denied')
sys.addaudithook(audit)
sys.path.insert(0,str(checkout))
sys.argv=[str(checkout/'run.py'),'--mini-replay','--output',str(checkout.parent.parent.parent/'new output')]
try:runpy.run_path(str(checkout/'run.py'),run_name='__main__')
except SystemExit as exc:
    if exc.code:raise
assert 'torch' not in sys.modules and 'scipy' not in sys.modules and 'sklearn' not in sys.modules
'''
    other=tmp_path/'unrelated working directory';other.mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith('MENTAL_PONG_') and k!='PYTHONPATH'}
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    proc=subprocess.run([sys.executable,'-I','-B','-c',wrapper,str(module)],cwd=other,env=env,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    result=json.loads(proc.stdout)
    assert result['status']=='passed' and result['cases']==4
