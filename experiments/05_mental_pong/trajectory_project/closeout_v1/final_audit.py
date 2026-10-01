"""Cross-module delivery checks and final read-only source preservation."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from trajectory_project.step3_io import read_json, write_json, record, sha, collect_records, verify_records
from trajectory_project.closeout_v1.closeout_setup import ROOT, SOURCE, ANIMALS, REPS, EPOCHS, verify_sources
from trajectory_project.closeout_v1.replay_a import compare_numeric, score


def audit():
    for lock in ('b_mapping_lock.json','c_endpoint_lock.json'):
        verify_records(collect_records(read_json(ROOT/'configs'/lock)))
    a=read_json(ROOT/'results/A/replay_audit.json')
    b=read_json(ROOT/'results/B/B_validation.json')
    c=read_json(ROOT/'results/C/completed.json')
    assert a['completed'] and a['n_prediction_files']==400
    assert b['status']=='B_COMPLETE' and b['B']==1000 and b['n_splits_per_q']==100
    assert not b['heads_refitted'] and not b['representations_refitted']
    assert c['completed'] and c['repeats_per_animal']==1000 and c['splits_per_repeat']==100
    assert c['actual_batched_sklearn_fits']==400 and c['independent_random_xy_heads']==400000
    assert a['raw_preprocessing']==b['raw_preprocessing']==c['raw_preprocessing']=='fail'
    b_observed=pd.read_csv(ROOT/'results/B/B_observed_full_support.csv').rename(columns={'split':'iteration'})
    across=pd.read_csv(ROOT/'results/A/round_cross_2x2.csv')
    b_error=compare_numeric(b_observed,across,
        ['animal','representation','iteration','epoch','coordinate','head','target'],
        ['r','RMSE','bias','amplitude_ratio'])
    from trajectory_project.condition_endpoint_fa_gpfa_v1.condition_data import load_condition_data
    from trajectory_project.closeout_v1.null_condition import generate_mappings, SEEDS as BSEEDS
    from trajectory_project.closeout_v1.null_endpoint import make_assignments, SEEDS as CSEEDS
    data=load_condition_data(SOURCE)
    splits=read_json(ROOT/'configs/condition_splits_100.json')
    extra=[];c_error=0.
    for animal in ANIMALS:
        p=data[animal];valid=p['common_mask'].any(1)
        valid_test=[np.sort(np.array(s['test_indices'])[valid[np.array(s['test_indices'])]]) for s in splits]
        with np.load(ROOT/'artifacts/B'/f'{animal}_mappings.npz') as z:
            # Each B artifact records axis names, source identities and the full mapping.
            key='mapping' if 'mapping' in z.files else 'mappings'
            np.testing.assert_array_equal(z[key],generate_mappings(valid_test,79,1000,BSEEDS[animal]))
        with np.load(ROOT/'artifacts/C'/f'{animal}_endpoint_assignments.npz') as z:
            ix,mapping,endpoints=make_assignments(p,CSEEDS[animal])
            np.testing.assert_array_equal(z['source_condition_indices'],mapping)
            np.testing.assert_array_equal(z['endpoints'],endpoints)
            common=z['common_mask'].copy();offset=z['offset'].copy();alpha=z['alpha'].copy()
            cmasks=z['epoch_masks'].copy();cepoch=z['epoch_names'].tolist()
        np.testing.assert_array_equal(common,p['common_mask'])
        assert len(np.unique(mapping,axis=0))==1000
        for rep in REPS:
            stem=f'{animal}_{rep}'
            done=read_json(ROOT/'results/C'/f'{stem}_completed.json')
            assert done['completed'] and done['OLS_calls']==100 and done['independent_random_y_refits']==100000
            assert not done['new_representation_fit'] and not done['new_inference']
            verify_records(done['outputs']+done['source_training_records'])
            with np.load(ROOT/'results/C'/f'{stem}_all_scores.npz') as z:
                observed=z['observed_scores'].copy();random=z['random_scores'].copy()
                metrics=z['metric_names'].tolist() if 'metric_names' in z.files else z['metrics'].tolist()
            assert random.shape==(1000,100,7,len(metrics))
            rows=[]
            for h,head in enumerate(('D_obj','D_beh')):
                for r in range(100):
                    for e,epoch in enumerate(EPOCHS):
                        rows.append(dict(animal=animal,representation=rep,iteration=r,epoch=epoch,coordinate='y',
                            head=head,target='objective' if h==0 else 'behavior',
                            **{m:observed[h,r,e,metrics.index(m)] for m in ('r','RMSE','bias','amplitude_ratio')}))
            own=pd.DataFrame(rows)
            old=across[(across.animal==animal)&(across.representation==rep)&(across.coordinate=='y')&
                       (across.cell.isin(['OO','BB']))]
            c_error=max(c_error,compare_numeric(own,old,
                ['animal','representation','iteration','epoch','coordinate','head','target'],
                ['r','RMSE','bias','amplitude_ratio']))
            # Independently score saved random weights; regenerate baselines from training endpoints.
            with np.load(ROOT/'artifacts/C'/f'{stem}_OLS_weights.npz') as w:
                for q,r in ((0,0),(999,99)):
                    split=splits[r];train=np.asarray(split['train_indices']);test=np.asarray(split['test_indices'])
                    tmask=np.zeros_like(common);tmask[test]=common[test];ci,ti=np.nonzero(tmask)
                    with np.load(ROOT.parent/'trial_endpoint_fa_gpfa_v1/representations'/animal/f'round_{r:03d}/latents.npz') as z:
                        latent=z[rep][0]
                    pred=latent[ci,ti]@w['coefficients'][r,q]+w['intercepts'][r,q]
                    target=offset[ci,ti]+alpha[ci,ti]*endpoints[q,ci]
                    mu=endpoints[q,train[valid[train]]].mean()
                    np.testing.assert_allclose(mu,w['mean_train_endpoints'][r,q],atol=1e-14)
                    base=offset[ci,ti]+alpha[ci,ti]*mu
                    for e,epoch in enumerate(cepoch):
                        mask=cmasks[e,ci,ti]
                        independent=score(target[mask],pred[mask])
                        for metric in ('r','RMSE','bias','amplitude_ratio'):
                            np.testing.assert_allclose(independent[metric],random[q,r,e,metrics.index(metric)],atol=2e-10,rtol=2e-10,equal_nan=True)
                        denominator=np.sum((base[mask]-target[mask])**2)
                        skill=1-np.sum((pred[mask]-target[mask])**2)/denominator if denominator>0 else np.nan
                        np.testing.assert_allclose(skill,random[q,r,e,metrics.index('skill')],atol=2e-10,rtol=2e-10,equal_nan=True)
                    extra.append(dict(animal=animal,representation=rep,q=q,split=r,n_test_rows=len(ci),
                                      independent_r_RMSE_bias_amplitude_skill_replay='pass'))
            del random,observed
    old_counts=verify_sources()
    output={'completed':True,'checked_utc':datetime.now(timezone.utc).isoformat(),
        'A_B_observed_score_max_difference':b_error,'A_C_observed_score_max_difference':c_error,
        'B_mapping_seed_regeneration':'pass','C_mapping_seed_regeneration':'pass',
        'C_independent_weight_and_train_mean_baseline_replay':extra,
        'A_no_refit':True,'B_no_refit':True,'C_actual_batched_sklearn_fits':400,
        'C_random_xy_heads':400000,'n_randomizations_each_null_each_animal':1000,
        'old_manifest_records_verified_unchanged':old_counts,
        'new_neural_inputs_or_representation_training':False,
        'causal_boundary':'No new neural future access; original raw preprocessing dependency remains.',
        'raw_preprocessing':'fail','provided_input_filtering':'pass only relative to released inputs',
        'fit_provenance':'pass only relative to released inputs'}
    write_json(ROOT/'results/final_acceptance_audit.json',output)
    return output


def seal():
    acceptance=read_json(ROOT/'results/final_acceptance_audit.json')
    assert acceptance['completed'] and acceptance['raw_preprocessing']=='fail'
    tests=read_json(ROOT/'results/test_execution.json')
    assert tests['exit_code']==0
    figure_qa=read_json(ROOT/'results/figure_qa.json')
    assert figure_qa['status']=='pass'
    verify_records(collect_records(figure_qa))
    for name in ('FINAL_REPORT.md','FINAL_SUMMARY.md','future_design.md','experiment_ledger.md','README.md'):
        assert (ROOT/name).is_file()
    records=[record(p,'closeout generated artifact or implementation') for p in sorted(ROOT.rglob('*'))
             if p.is_file() and p.name!='manifest.json' and not any(s in p.parts for s in ('__pycache__','.pytest_cache','tmp'))]
    sources=read_json(ROOT/'sources/frozen_source_records.json')+collect_records(read_json(ROOT/'sources/A_sources.json'))
    m=dict(status='CLOSED_EXPLORATORY_WITH_LIMITATIONS',completed=True,
           sealed_utc=datetime.now(timezone.utc).isoformat(),raw_preprocessing='fail',
           provided_input_filtering='pass relative to released inputs only',
           fit_provenance='pass relative to released inputs only',
           A_completed=True,B_repeats=1000,C_repeats=1000,original_splits_per_repeat=100,
           protocol_sha256=sha(ROOT/'configs/closeout_protocol.json'),sources=sources,outputs=records,
           old_manifest_records_verified_unchanged=acceptance['old_manifest_records_verified_unchanged'],
           future_work_executed=False)
    write_json(ROOT/'manifest.json',m)
    print('Sealed',verify_records(collect_records(m)),'records',flush=True)
    return m


if __name__=='__main__':
    audit()
