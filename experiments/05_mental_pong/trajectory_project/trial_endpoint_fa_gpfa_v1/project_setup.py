"""Freeze the analysis definition before reading any new reconstruction scores."""
from pathlib import Path
import json
import platform
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PILOT = PROJECT.parent

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from trajectory_project.step3_io import read_json, write_json, record, sha, collect_records, verify_records

from trajectory_project.runtime_paths import configured_root
REQUEST = configured_root("source_pilot") / "trajectory_project/trial_endpoint_fa_gpfa_v1/sources/user_request.txt"


def initialize():
    for directory in ('configs','sources','data','representations','readouts','results','reports','figures','tests','logs'):
        (ROOT/directory).mkdir(exist_ok=True)
    prior=[]
    for name in ('all79_official_readout_v1','behavior_feasibility','condition_mean_behavior_preview_v1','condition_mean_behavior_stageB_v1'):
        path=PROJECT/name/'manifest.json'
        manifest=read_json(path)
        records=collect_records(manifest)
        count=verify_records(records)
        prior.append(dict(name=name,manifest=record(path,'immutable historical analysis'),records=records,count=count))
    write_json(ROOT/'sources/historical_preservation.json',prior,immutable=True)
    request_copy=ROOT/'sources/user_request.txt'
    if not request_copy.exists():request_copy.write_bytes(REQUEST.read_bytes())
    assert sha(request_copy)==sha(REQUEST)
    config={
        'version':'trial_endpoint_fa_gpfa_v1','data_mode':'trial_labels_with_mean_neural',
        'primary_comparison':'Each objective/candidate trajectory scored against its own trial-labelled reference; OO vs BB.',
        'animals':['mahler','perle'],'representations':['FA50','GPFA50'],
        'n_components':50,'model_seed':42,'split_seed':0,'n_splits':100,'train_conditions':39,'test_conditions':40,
        'fit_neural_half':'half1','fit_scope':'All representation parameters, feature eligibility and fixed normalization refitted using only this split training39 half1 condition responses.',
        'FA':{'n_components':50,'random_state':42,'max_iter':300,'tol':0.01,'svd_method':'lapack','iterated_power':7,'rotation':None},
        'GPFA':read_json(PILOT/'gpfa_config.json')['model_options'],
        'GPFA_initialization':'This round train-only FA50 C/d/R, with existing GPFA seeded 1% loading jitter; no FA100 discarded-loading initialization. Fixed before scores, not selected by labels.',
        'GPFA_family':'One shared learnable RBF timescale, exact Gaussian prefix posterior; same existing SharedTimescaleGPFA implementation.',
        'preprocessing':{'training_finite_fraction':0.99,'std_floor':1e-5,'bin_width_ms':50,'time_alignment':'completed-bin right edge; deterministic objective physical-branch evaluation at that timestamp; no lag search'},
        'causal_scope':{'provided_input_filtering':'must pass prefix/future perturbation tests','fit_provenance':'must pass train39 vs test40 isolation','raw_preprocessing':'fail: irreversible published global-condition/time missing-condition imputation; no claim of strictly causal raw-spike analysis','endpoint_timing':'published terminal scalar; exact pre-feedback sampling not established'},
        'OLS':{'class':'sklearn.linear_model.LinearRegression','fit_intercept':True,'positive':False,'regularization':None,'scaler':None,'sample_weights':None,'precision':'float64','coefficients_per_coordinate':51,'coefficients_per_xy_head':102,'solve':'Multioutput [x_obj,y_obj,x_beh,y_beh], algebraically independent OLS coordinates with one shared matrix decomposition; verify against separate fits.'},
        'labels':{'endpoint':'Each real trial own finite valid terminal paddle_y, never averaged before supervision','no_bounce':'Initial physical anchor to own terminal paddle, using x progression','bounce':'Objective before actual collision; collision anchor to own terminal paddle afterwards','same_x':True,'stable_paddle_detection':False,'dynamic_updates':False,'visible_override':False,'clip_target':False,'endpoint_correctness_filter':False,'unknown_terminal_scalar':'Keep identity; invalid candidate rather than interpret default zero as final paddle','unresolved_collision':'Keep condition and trial identities; no invented collision anchor'},
        'epochs':['full','visible','hidden','bounce','no_bounce','post_bounce'],
        'aggregation':'Raw held-out trial x time rows per split, then mean and population SD over overlapping splits; per-trial, session and condition descriptive summaries. No independent-experiment significance from split repetitions.',
        'primary_differences':{'Delta_r':'r_beh-r_obj','Delta_RMSE':'RMSE_obj-RMSE_beh'},
        'storage':'Lossless trial labels and sample index; held-out predictions factorized as same-condition same-input predictions plus real test-trial index. Explicit per-trial expansion/export supported. Scores always evaluate every real trial label.',
        'source_records':[record(REQUEST,'user-authorized complete request'),record(PILOT/'gpfa_shared.py','unchanged shared RBF model'),record(PILOT/'gpfa_config.json','unchanged GPFA hyperparameter source'),record(PILOT/'capacity_position_baselines.py','unchanged FA50 training setting source'),record(PILOT/'official_reproduction.py','verified instantaneous OLS reproduction'),record(PROJECT/'all79_official_readout_v1/sources/official_protocol_audit.json','previous verified official source audit')]
    }
    write_json(ROOT/'configs/analysis_protocol.json',config,immutable=True)
    write_json(ROOT/'configs/protocol_lock.json',{'sha256':sha(ROOT/'configs/analysis_protocol.json'),'locked_before_new_scores':True},immutable=True)
    write_json(ROOT/'sources/runtime.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'libraries_upgraded':False},immutable=True)
    return config


def fixed_splits(condition_ids):
    ids=np.asarray(condition_ids,dtype=np.int64)
    assert len(ids)==len(np.unique(ids))==79
    splitter=GroupShuffleSplit(n_splits=100,train_size=39,test_size=40,random_state=0)
    result=[]
    for i,(tr,te) in enumerate(splitter.split(ids[:,None],groups=ids)):
        assert len(tr)==39 and len(te)==40 and not set(tr)&set(te)
        result.append({'iteration':i,'train_indices':tr.tolist(),'test_indices':te.tolist(),
            'train_condition_ids':ids[tr].tolist(),'test_condition_ids':ids[te].tolist(),'random_state':0})
    write_json(ROOT/'configs/condition_splits_100.json',result,immutable=True)
    return result


def verify_historical():
    prior=read_json(ROOT/'sources/historical_preservation.json')
    for item in prior:
        verify_records([item['manifest'],*item['records']])
    return {item['name']:item['count'] for item in prior}


def seal(completed=True):
    config=read_json(ROOT/'configs/analysis_protocol.json')
    assert sha(ROOT/'configs/analysis_protocol.json')==read_json(ROOT/'configs/protocol_lock.json')['sha256']
    verify_records(config['source_records'])
    historical=verify_historical()
    outputs=[record(p,'trial endpoint analysis output or reproducible implementation') for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.name!='manifest.json' and not any(part in p.parts for part in ('tmp','__pycache__','.pytest_cache'))]
    sources=config['source_records']
    for p in (ROOT/'sources').glob('*audit*.json'):
        sources += collect_records(read_json(p))
    manifest={'completed':completed,'sealed_utc':datetime.now(timezone.utc).isoformat(),'data_mode':config['data_mode'],
        'strict_raw_future_free':False,'historical_files_unchanged':historical,
        'protocol_sha256':sha(ROOT/'configs/analysis_protocol.json'),'sources':sources,'outputs':outputs}
    write_json(ROOT/'manifest.json',manifest)
    print('Sealed and verified',verify_records(collect_records(manifest)),'records',flush=True)


if __name__=='__main__':
    initialize()
    print('Protocol locked:',sha(ROOT/'configs/analysis_protocol.json'),flush=True)
