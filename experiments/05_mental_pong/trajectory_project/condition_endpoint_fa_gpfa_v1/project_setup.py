"""Immutable protocol and source preservation for the condition-mean rerun."""
from pathlib import Path
import platform
from datetime import datetime,timezone
import numpy as np
import scipy
import pandas as pd
import sklearn
from trajectory_project.step3_io import read_json,write_json,sha,record,verify_records,collect_records

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'trial_endpoint_fa_gpfa_v1'


def initialize():
    for name in ('configs','artifacts','data','results','sources','figures','readouts','logs','tests'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    prior=read_json(SOURCE/'manifest.json')
    assert prior['completed'] is True and prior['strict_raw_future_free'] is False
    write_json(ROOT/'sources/prior_manifest_record.json',record(SOURCE/'manifest.json'),immutable=True)
    verify_records([read_json(ROOT/'sources/prior_manifest_record.json')])
    old=read_json(SOURCE/'configs/analysis_protocol.json')
    config={'version':'condition_endpoint_fa_gpfa_v1','data_mode':'condition_mean_neural_and_behavior',
        'animals':['mahler','perle'],'representations':['FA50','GPFA50'],'n_components':50,
        'n_splits':100,'train_conditions':39,'test_conditions':40,'split_seed':0,
        'representation_policy':'Reference exactly the prior per-round train39 FA50/GPFA50 weights, preprocessing and half1 prefix latents; no fitting or inference in this version.',
        'neural_half':'half1','behavior_average':'Mean of the SAME previously valid real-trial candidate pool, one fixed pool per condition; no binwise membership changes; x copied exactly from objective.',
        'label_construction':'Existing initial/collision endpoint interpolation is affine in b_i, so averaging valid trial candidates equals constructing from mean endpoint. No new rules or thresholds.',
        'invalid_condition_policy':'Keep all79 split identities; keep59920 unknown; no replacement or padding.',
        'OLS':old['OLS'],'training_rows':'One valid condition x time row; no repeated trial rows, sample_weight, or condition-equal weighting. Longer conditions contribute more bins.',
        'epochs':old['epochs'],'aggregation':'Each heldout split pooled condition x time Pearson r/RMSE; then mean/population SD over100 shared splits. Condition scores and full2x2 are supplementary.',
        'differences':old['primary_differences'],
        'comparison_to_trial_version':'Both target trial variance and trial-count weighting changed; old/new metric changes cannot be attributed solely to averaging labels.',
        'causal_scope':old['causal_scope'],'independent_neural_trials':0,
        'source_records':[record(SOURCE/'configs/analysis_protocol.json'),record(SOURCE/'configs/condition_splits_100.json'),
            record(SOURCE/'results/final_integrity_and_causal_audit.json'),record(SOURCE/'representations.py'),
            record(SOURCE/'protocol_diff.md'),record(SOURCE/'results/self_reconstruction_main_table.csv')]}
    write_json(ROOT/'configs/analysis_protocol.json',config,immutable=True)
    write_json(ROOT/'configs/protocol_lock.json',{'sha256':sha(ROOT/'configs/analysis_protocol.json'),'locked_before_new_scores':True},immutable=True)
    splits=read_json(SOURCE/'configs/condition_splits_100.json')
    write_json(ROOT/'configs/condition_splits_100.json',splits,immutable=True)
    assert sha(ROOT/'configs/condition_splits_100.json')==sha(SOURCE/'configs/condition_splits_100.json')
    write_json(ROOT/'sources/runtime.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'pandas':pd.__version__,'sklearn':sklearn.__version__,'libraries_upgraded':False},immutable=True)
    return config,splits


def verify_prior():
    verify_records([read_json(ROOT/'sources/prior_manifest_record.json')])
    return verify_records(collect_records(read_json(SOURCE/'manifest.json')))


def seal():
    cfg=read_json(ROOT/'configs/analysis_protocol.json')
    assert sha(ROOT/'configs/analysis_protocol.json')==read_json(ROOT/'configs/protocol_lock.json')['sha256']
    old_count=verify_prior()
    outputs=[record(p,'condition-mean analysis output or implementation') for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.name!='manifest.json' and not any(x in p.parts for x in ('tmp','__pycache__','.pytest_cache'))]
    sources=cfg['source_records']+[read_json(ROOT/'sources/prior_manifest_record.json')]
    for name in ('label_sources.json','reused_representations_manifest.json'):
        path=ROOT/'sources'/name
        if path.exists():sources+=collect_records(read_json(path))
    reuse=ROOT/'results/reused_representation_audit.json'
    if reuse.exists():sources+=collect_records(read_json(reuse))
    m={'completed':True,'sealed_utc':datetime.now(timezone.utc).isoformat(),'data_mode':cfg['data_mode'],
        'strict_raw_future_free':False,'representation_models_retrained':0,'prior_records_verified_unchanged':old_count,
        'protocol_sha256':sha(ROOT/'configs/analysis_protocol.json'),'sources':sources,'outputs':outputs}
    write_json(ROOT/'manifest.json',m)
    print('Sealed',verify_records(collect_records(m)),'records; unchanged prior',old_count,flush=True)
