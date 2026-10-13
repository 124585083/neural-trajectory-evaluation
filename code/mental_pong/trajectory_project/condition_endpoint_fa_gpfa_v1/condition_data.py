"""Condition means of the exact prior valid behavioral-trial pool."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from trajectory_project.step3_io import record,write_json,read_json
from trajectory_project.trial_endpoint_fa_gpfa_v1.trial_data import load_trial_data

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'trial_endpoint_fa_gpfa_v1'
MODE='condition_mean_neural_and_behavior'
EPOCHS=('full','visible','hidden','bounce','no_bounce','post_bounce')


def mean_payload(p):
    ids=p['condition_ids']; times=p['times_ms']; ci=p['condition_index']
    objective=p['objective_xy'].copy()
    behavior=np.full_like(objective,np.nan,dtype=np.float64)
    mask=np.zeros(objective.shape[:2],bool)
    counts=np.zeros(len(ids),int); endpoints=np.full(len(ids),np.nan)
    memberships={}; coverage=[]; max_mean_error=0.
    table=p['trial_table']
    for c,cid in enumerate(ids):
        pool=np.flatnonzero((ci==c)&p['common_mask'].any(1))
        memberships[int(cid)]=pool.tolist();counts[c]=len(pool)
        if len(pool):
            support=p['common_mask'][pool]
            if not np.all(support==support[0]):
                raise ValueError(f'Trial support varies within {p["animal"]}/{cid}; do not silently change the averaging pool over time')
            mask[c]=support[0]
            endpoints[c]=np.mean(p['endpoint'][pool],dtype=np.float64)
            values=p['behavior_xy'][pool][:,mask[c],1].astype(np.float64)
            assert np.isfinite(values).all()
            behavior[c,mask[c],1]=values.mean(0)
            # Keep the shared x targets bit-identical, not merely close.
            behavior[c,mask[c],0]=objective[c,mask[c],0]
            max_mean_error=max(max_mean_error,float(np.max(np.abs(behavior[c,mask[c],1]-values.mean(0)))))
        source_rows=table.iloc[np.flatnonzero(ci==c)]
        coverage.append({'animal':p['animal'],'condition_id':int(cid),'condition_index':c,
            'n_all_source_behavior_records':len(source_rows),'n_valid_terminal_endpoints':int(source_rows.endpoint_valid.sum()),
            'n_behavior_trials_in_mean':len(pool),'mean_endpoint':endpoints[c],
            'n_condition_time_rows':int(mask[c].sum()),'candidate_status':'valid_condition_mean' if len(pool) else 'unknown_geometry_or_no_valid_trial',
            'n_neural_trials':0,'data_mode':MODE})
    epochs={e:p['epoch_condition_masks'][e]&mask for e in EPOCHS}
    return {'animal':p['animal'],'condition_ids':ids.copy(),'times_ms':times.copy(),
        'objective_xy':objective,'behavior_xy':behavior,'common_mask':mask,'epoch_masks':epochs,
        'n_behavior_trials':counts,'mean_endpoint':endpoints,'source_trial_indices_by_condition':memberships,
        'geometry':p['geometry'].copy(),'data_mode':MODE,'n_neural_trials':0,
        'coverage':pd.DataFrame(coverage),'mean_label_max_error':max_mean_error}


def build_condition_data(root=ROOT):
    root=Path(root)
    for name in ('artifacts','data','results','sources'): (root/name).mkdir(parents=True,exist_ok=True)
    original=load_trial_data(SOURCE)
    data={};audit=[];cover=[];members={}
    from trajectory_project.condition_endpoint_fa_gpfa_v1.reuse_audit import validate_condition_averaging
    for animal,p in original.items():
        a=mean_payload(p)
        verification=validate_condition_averaging(p,a)
        path=root/'artifacts'/f'{animal}_condition_labels.npz'
        np.savez_compressed(path,condition_ids=a['condition_ids'],times_ms=a['times_ms'],
            objective_xy=a['objective_xy'],behavior_xy=a['behavior_xy'],common_mask=a['common_mask'],
            n_behavior_trials=a['n_behavior_trials'],mean_endpoint=a['mean_endpoint'],
            **{'epoch_'+e:m for e,m in a['epoch_masks'].items()})
        a['labels_path']=str(path.resolve());data[animal]=a;cover.append(a['coverage'])
        members[animal]={'source_records':record(p['trial_records_path']),
            'source_labels':record(p['labels_path']),
            'trial_indices_by_condition':a['source_trial_indices_by_condition']}
        c,b=np.nonzero(a['common_mask'])
        index=pd.DataFrame({'animal':animal,'condition_id':a['condition_ids'][c],
            'condition_index':c,'time_index':b,'time_ms':a['times_ms'][b],
            'unique_neural_input_id':c*len(a['times_ms'])+b,
            'n_behavior_trials_in_mean':a['n_behavior_trials'][c],
            'x_obj':a['objective_xy'][c,b,0],'y_obj':a['objective_xy'][c,b,1],
            'x_beh':a['behavior_xy'][c,b,0],'y_beh':a['behavior_xy'][c,b,1],
            **{e:m[c,b] for e,m in a['epoch_masks'].items()}})
        index.to_csv(root/'data'/f'{animal}_condition_time_index.csv',index=False,float_format='%.17g')
        audit.append({'animal':animal,'verification':verification,'condition_count':len(a['condition_ids']),
            'valid_conditions':int(a['common_mask'].any(1).sum()),'condition_time_rows':int(a['common_mask'].sum()),
            'n_behavior_trials_in_means':int(a['n_behavior_trials'].sum()),
            'trials_do_not_weight_OLS':True,'neural_input_averaged_again':False})
    write_json(root/'sources/source_trial_membership.json',members)
    pd.concat(cover,ignore_index=True).to_csv(root/'results/condition_label_coverage.csv',index=False,float_format='%.17g')
    pd.concat([a['geometry'] for a in data.values()],ignore_index=True).to_csv(root/'results/condition_geometry.csv',index=False,float_format='%.17g')
    write_json(root/'results/condition_averaging_audit.json',audit)
    write_json(root/'sources/label_sources.json',[record(SOURCE/'artifacts'/f'{a}_trial_labels.npz') for a in data]+[
        record(SOURCE/'trial_data.py'),record(SOURCE/'configs/label_geometry_protocol.json'),record(SOURCE/'results/actual_label_validation.json')])
    return data


def load_condition_data(root=ROOT):
    root=Path(root);result={}
    geometry=pd.read_csv(root/'results/condition_geometry.csv')
    members=read_json(root/'sources/source_trial_membership.json')
    for animal in ('mahler','perle'):
        path=root/'artifacts'/f'{animal}_condition_labels.npz'
        with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        a['epoch_masks']={e:a.pop('epoch_'+e) for e in EPOCHS}
        a.update(animal=animal,labels_path=str(path.resolve()),data_mode=MODE,n_neural_trials=0,
            geometry=geometry[geometry.animal.eq(animal)].copy(),
            source_trial_indices_by_condition={int(k):v for k,v in members[animal]['trial_indices_by_condition'].items()})
        result[animal]=a
    return result
