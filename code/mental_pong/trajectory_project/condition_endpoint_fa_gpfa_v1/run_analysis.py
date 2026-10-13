"""Reuse all100 representation fits; refit condition×time OLS only."""
from pathlib import Path
import argparse
import sys
import time
HERE=Path(__file__).resolve().parent
if __name__ == "__main__": sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
from trajectory_project.step3_io import read_json,write_json,sha,record,verify_records,collect_records
from trajectory_project.condition_endpoint_fa_gpfa_v1.project_setup import initialize,seal,verify_prior,SOURCE
from trajectory_project.condition_endpoint_fa_gpfa_v1.condition_data import build_condition_data,load_condition_data


def decode_one(payload,split,threads):
    from trajectory_project.trial_endpoint_fa_gpfa_v1.representations import load_round
    from trajectory_project.condition_endpoint_fa_gpfa_v1.decoding_condition import run_condition_round
    a=payload['animal'];i=split['iteration'];folder=SOURCE/'representations'/a/f'round_{i:03d}'
    marker=HERE/'readouts/completed'/f'{a}_round_{i:03d}.json'
    meta=read_json(folder/'training.json')
    verify_records([{'path':str(folder/k),'sha256':v} for k,v in meta['file_hashes'].items()])
    fingerprint={'protocol':sha(HERE/'configs/analysis_protocol.json'),'decoder':sha(HERE/'decoding_condition.py'),
        'labels':sha(payload['labels_path']),'representation':sha(folder/'training.json'),'split':split}
    if marker.exists():
        info=read_json(marker);assert info['fingerprint']==fingerprint
        verify_records(info['outputs']);return info
    reps=load_round(folder)
    for rep in reps.values():
        np.testing.assert_array_equal(rep['condition_ids'],payload['condition_ids'])
        np.testing.assert_array_equal(rep['times_ms'],payload['times_ms'])
        assert not np.any(payload['common_mask']&~rep['common_mask'])
    started=time.perf_counter()
    result=run_condition_round(payload,reps,split,HERE/'readouts',threads=threads)
    info={'animal':a,'iteration':i,'fingerprint':fingerprint,'seconds':time.perf_counter()-started,
        'outputs':[record(p) for p in result['files']],'prediction_paths':result['prediction_paths']}
    write_json(marker,info)
    return info


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot-only',action='store_true');ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--report-only',action='store_true');ap.add_argument('--verify',action='store_true');ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    if args.verify:
        count=verify_records(collect_records(read_json(HERE/'manifest.json')))
        print('Verified',count,'new records; unchanged prior',verify_prior(),flush=True);return
    cfg,splits=initialize()
    if not all((HERE/'artifacts'/f'{a}_condition_labels.npz').exists() for a in cfg['animals']):
        print('Verifying prior manifest before creating condition means',flush=True);verify_prior()
        data=build_condition_data(HERE)
    else:data=load_condition_data(HERE)
    for a,p in data.items():print(a,'condition rows',int(p['common_mask'].sum()),'valid conditions',int(p['common_mask'].any(1).sum()),flush=True)
    from trajectory_project.condition_endpoint_fa_gpfa_v1.reuse_audit import validate_reused_representations
    validate_reused_representations(SOURCE,HERE,splits)
    if args.prepare_only:return
    if not args.report_only:
        for a in data:decode_one(data[a],splits[0],args.threads)
        write_json(HERE/'results/end_to_end_pilot.json',{'completed':True,'round':0,'formal_round_reused':True,'models_retrained':0})
        if args.pilot_only:return
        # No training of a representation is available from this entry point.
        start=time.perf_counter()
        for i in range(1,100):
            for a in data:decode_one(data[a],splits[i],args.threads)
            if (i+1)%10==0:print(f'OLS completed {i+1}/100 shared splits, both animals',flush=True)
        write_json(HERE/'logs/execution.json',{'completed_animal_rounds':200,'representation_models_retrained':0,
            'OLS_heads':800,'seconds_excluding_pilot':time.perf_counter()-start,'threads':args.threads})
    from trajectory_project.condition_endpoint_fa_gpfa_v1.reporting_condition import make_report
    make_report(data,HERE)
    seal()


if __name__=='__main__':main()
