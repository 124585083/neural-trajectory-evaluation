"""Independent trial-label analysis; resume verified caches without changing science."""
from pathlib import Path
import argparse
import gc
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE=Path(__file__).resolve().parent
PILOT=HERE.parent.parent
if __name__ == "__main__": sys.path.insert(0, str(PILOT))
import numpy as np
import pandas as pd
from trajectory_project.step3_io import read_json,write_json,sha,record,verify_records,collect_records
from trajectory_project.trial_endpoint_fa_gpfa_v1.project_setup import initialize,fixed_splits,seal,verify_historical


def representation_worker(animal,iterations,threads):
    from trajectory_project.trial_endpoint_fa_gpfa_v1.representations import load_neural_data,fit_round
    data=load_neural_data(animal,cache_root=HERE/'artifacts')
    splits=read_json(HERE/'configs/condition_splits_100.json')
    for i in iterations:
        s=splits[i]
        fit_round(data,s['train_condition_ids'],s['test_condition_ids'],i,output_root=HERE,cpu_threads=threads)
    return {'animal':animal,'iterations':iterations}


def load_inputs():
    from trajectory_project.trial_endpoint_fa_gpfa_v1.trial_data import load_trial_data,build_trial_data
    if not all((HERE/'artifacts'/f'{a}_trial_labels.npz').exists() for a in ('mahler','perle')):
        data=build_trial_data(HERE)
    else:data=load_trial_data(HERE)
    ids=data['mahler']['condition_ids']
    np.testing.assert_array_equal(ids,data['perle']['condition_ids'])
    return data,fixed_splits(ids)


def representation_ready(animal,i):
    return (HERE/'representations'/animal/f'round_{i:03d}'/'training.json').exists()


def decode_one(payload,split,threads):
    from trajectory_project.trial_endpoint_fa_gpfa_v1.representations import load_round
    from trajectory_project.trial_endpoint_fa_gpfa_v1.decoding_trial import run_trial_round
    animal=payload['animal'];i=int(split['iteration'])
    marker=HERE/'readouts'/'completed'/f'{animal}_round_{i:03d}.json'
    folder=HERE/'representations'/animal/f'round_{i:03d}'
    meta=read_json(folder/'training.json')
    verify_records([{'path':str(folder/name),'sha256':digest} for name,digest in meta['file_hashes'].items()])
    fingerprint={'protocol':sha(HERE/'configs/analysis_protocol.json'),
        'decoder':sha(HERE/'decoding_trial.py'),'labels':sha(payload['labels_path']),
        'representation':sha(folder/'training.json'),'split':split}
    if marker.exists():
        prior=read_json(marker)
        assert prior['fingerprint']==fingerprint,'Readout cache input changed'
        verify_records(prior['outputs'])
        return prior
    representations=load_round(folder)
    for rep in representations.values():
        np.testing.assert_array_equal(rep['condition_ids'],payload['condition_ids'])
        np.testing.assert_array_equal(rep['times_ms'],payload['times_ms'])
    start=time.perf_counter()
    print(f'OLS START {animal} round {i:03d}',flush=True)
    result=run_trial_round(payload,representations,split,HERE/'readouts',threads=threads)
    outputs=[record(p,'actual trial-expanded OLS result') for p in result['files']]
    result_record={'fingerprint':fingerprint,'animal':animal,'iteration':i,
        'seconds':time.perf_counter()-start,'outputs':outputs,'prediction_paths':result['prediction_paths'],
        'common_row_count':result['common_row_count']}
    write_json(marker,result_record)
    print(f'OLS DONE {animal} round {i:03d}: {result_record["seconds"]:.1f}s',flush=True)
    gc.collect()
    return result_record


def write_shared_indices(data):
    from trajectory_project.trial_endpoint_fa_gpfa_v1.trial_data import write_sample_index
    from trajectory_project.trial_endpoint_fa_gpfa_v1.representations import load_round
    records=[]
    for animal,payload in data.items():
        first=load_round(HERE/'representations'/animal/'round_000')
        common=payload['common_mask'].copy()
        for rep in first.values():common &= rep['common_mask'][payload['condition_index']]
        p=HERE/'data'/f'{animal}_sample_index.csv.gz'
        if not p.exists():
            info=write_sample_index(payload,p,common_mask=common)
            write_json(HERE/'data'/f'{animal}_sample_index.json',info)
        else:
            info=read_json(HERE/'data'/f'{animal}_sample_index.json')
            verify_records([info])
        records.append(info)
    write_json(HERE/'results/shared_sample_index_audit.json',records)


def run_pilot(data,splits,threads):
    for animal in ('mahler','perle'):
        representation_worker(animal,[0],threads)
        decode_one(data[animal],splits[0],threads)
    write_shared_indices(data)
    write_json(HERE/'results/end_to_end_pilot.json',{'completed':True,'round':0,
        'animals':['mahler','perle'],'representations':['FA50','GPFA50'],
        'formal_round_reused':True,'hyperparameter_selection':False})


_worker_data=None


def decoding_worker(animal,iteration,threads):
    global _worker_data
    if _worker_data is None:
        from trajectory_project.trial_endpoint_fa_gpfa_v1.trial_data import load_trial_data
        _worker_data=load_trial_data(HERE)
    split=read_json(HERE/'configs/condition_splits_100.json')[iteration]
    return decode_one(_worker_data[animal],split,threads)


def run_remaining(data,splits,workers,decode_workers,threads):
    pending={(a,i) for a in ('mahler','perle') for i in range(100)
        if not (HERE/'readouts'/'completed'/f'{a}_round_{i:03d}.json').exists()}
    jobs=[]
    # Static cost ordering, fixed before further scores; no fit is omitted.
    for animal,begin,end in [('perle',0,25),('perle',25,50),('mahler',0,50),
                              ('perle',50,75),('perle',75,100),('mahler',50,100)]:
        todo=[i for i in range(begin,end) if not representation_ready(animal,i)]
        if todo:jobs.append((animal,todo))
    with ProcessPoolExecutor(max_workers=workers) as pool, ProcessPoolExecutor(max_workers=decode_workers) as decode_pool:
        futures=[pool.submit(representation_worker,a,ix,threads) for a,ix in jobs]
        active={}
        while pending or active:
            for f in futures:
                if f.done():f.result()
            for f in list(active):
                if f.done():
                    f.result();active.pop(f)
                    print(f'Completed animal-rounds {200-len(pending)-len(active)}/200',flush=True)
            ready=sorted((a,i) for a,i in pending if representation_ready(a,i))
            while ready and len(active)<decode_workers:
                animal,i=min(ready,key=lambda v:(v[1],v[0]))
                pending.remove((animal,i))
                ready.remove((animal,i))
                active[decode_pool.submit(decoding_worker,animal,i,threads)]=(animal,i)
            if pending and not active and not ready and all(f.done() for f in futures):
                raise RuntimeError(f'No representation for remaining jobs: {sorted(pending)[:3]}')
            time.sleep(.5)
        for f in futures:f.result()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--decode-workers',type=int,default=3)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.verify:
        m=read_json(HERE/'manifest.json')
        print('Verified',verify_records(collect_records(m)),'records',verify_historical());return
    initialize()
    data,splits=load_inputs()
    if not args.report_only:
        run_pilot(data,splits,args.threads)
        if args.pilot_only:return
        run_remaining(data,splits,args.workers,args.decode_workers,args.threads)
    from trajectory_project.trial_endpoint_fa_gpfa_v1.reporting_trial import make_report
    make_report(data,HERE)
    seal()


if __name__=='__main__':main()
