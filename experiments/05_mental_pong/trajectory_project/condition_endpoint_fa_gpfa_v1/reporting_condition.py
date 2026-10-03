"""Own-target reconstruction of condition-average trajectories."""
from pathlib import Path
import numpy as np
import pandas as pd
from trajectory_project.step3_io import write_json,read_json,record
from trajectory_project.trial_endpoint_fa_gpfa_v1.reporting_trial import summarize,md
from trajectory_project.condition_endpoint_fa_gpfa_v1.project_setup import SOURCE

METRICS=['r_obj','r_beh','RMSE_obj','RMSE_beh','Delta_r','Delta_RMSE']
EPOCHS=['full','visible','hidden','bounce','no_bounce','post_bounce']
KEYS=['animal','representation','data_mode','causal_status','epoch','coordinate']


def collect(data,root):
    frames={}
    for kind in ('self','cross_2x2','condition','fit_audit'):
        paths=sorted((root/'readouts/round_results').glob('*_'+kind+'.csv'))
        assert paths,kind
        frames[kind]=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    raw=frames['self']
    assert raw.groupby(['animal','representation']).iteration.nunique().eq(100).all()
    raw.to_csv(root/'results/round_self_metrics.csv',index=False,float_format='%.17g')
    frames['cross_2x2'].to_csv(root/'results/round_cross_2x2.csv',index=False,float_format='%.17g')
    frames['condition'].to_csv(root/'results/round_condition_metrics.csv.gz',index=False,float_format='%.17g')
    frames['fit_audit'].to_csv(root/'results/ols_fit_audit.csv',index=False)
    counts=['n_conditions','n_bins','n_condition_time_rows','n_unique_neural_input_ids','n_behavior_trials_in_means']
    main=summarize(raw,KEYS,METRICS+counts)
    main=main.rename(columns={c:c+'_per_round_mean' for c in counts})
    coverage=[]
    for animal,p in data.items():
        for epoch,mask in p['epoch_masks'].items():
            coverage.append({'animal':animal,'epoch':epoch,'n_conditions':int(mask.any(1).sum()),
                'n_bins':int(mask.sum()),'n_condition_time_rows':int(mask.sum()),'n_unique_neural_input_ids':int(mask.sum()),
                'n_behavior_trials_in_means':int(p['n_behavior_trials'][mask.any(1)].sum()),'n_neural_trials':0})
    coverage=pd.DataFrame(coverage);coverage.to_csv(root/'results/coverage.csv',index=False)
    main=main.merge(coverage,on=['animal','epoch'],validate='many_to_one')
    main['epoch']=pd.Categorical(main.epoch,EPOCHS,ordered=True)
    main=main.sort_values(['animal','representation','coordinate','epoch'])
    main.to_csv(root/'results/self_reconstruction_main_table.csv',index=False,float_format='%.17g')
    shape_fields=['r','RMSE','bias','target_sd','prediction_sd','amplitude_ratio','target_min','target_max']
    cross=summarize(frames['cross_2x2'],KEYS+['head','target'],shape_fields)
    cross.to_csv(root/'results/cross_2x2_summary.csv',index=False,float_format='%.17g')
    cond=frames['condition'];base=KEYS+['condition_id']
    condition_summary=summarize(cond,base+['head','target'],shape_fields)
    condition_summary.to_csv(root/'results/condition_cross_summary.csv',index=False,float_format='%.17g')
    selected=['r','RMSE','bias','amplitude_ratio']
    oo=cond[cond['head'].eq('D_obj')&cond.target.eq('objective')]
    bb=cond[cond['head'].eq('D_beh')&cond.target.eq('behavior')]
    pairs=oo[base+['iteration']+selected].merge(bb[base+['iteration']+selected],on=base+['iteration'],suffixes=('_obj','_beh'),validate='one_to_one')
    pairs['Delta_r']=pairs.r_beh-pairs.r_obj;pairs['Delta_RMSE']=pairs.RMSE_obj-pairs.RMSE_beh
    conditions=summarize(pairs,base,METRICS+['bias_obj','bias_beh','amplitude_ratio_obj','amplitude_ratio_beh'])
    conditions=conditions.rename(columns={'n_rounds':'n_test_splits'})
    conditions.to_csv(root/'results/condition_self_comparison.csv',index=False,float_format='%.17g')
    comparison=main[['animal','representation','epoch','coordinate',*METRICS]].merge(
        pd.read_csv(SOURCE/'results/self_reconstruction_main_table.csv')[['animal','representation','epoch','coordinate',*METRICS]],
        on=['animal','representation','epoch','coordinate'],suffixes=('_condition_mean','_trial_version'),validate='one_to_one')
    for m in METRICS:comparison[m+'_new_minus_old']=comparison[m+'_condition_mean']-comparison[m+'_trial_version']
    comparison.to_csv(root/'results/previous_trial_version_comparison.csv',index=False,float_format='%.17g')
    return frames,main,cross,conditions,coverage,comparison


def time_diagnostics(data,root):
    rows=[]
    for animal,p in data.items():
        target=np.stack([p['objective_xy'][...,1],p['behavior_xy'][...,1]])
        for rep in ('FA50','GPFA50'):
            arrays=[]
            for f in sorted((root/'readouts/predictions').glob(f'{animal}_{rep}_r*_test_predictions.npz')):
                with np.load(f,allow_pickle=False) as z:arrays.append(z['predictions'][...,1])
            pred=np.stack(arrays);n=np.isfinite(pred).sum(0)
            mean=np.divide(np.nansum(pred,0),n,out=np.full(target.shape,np.nan),where=n>0)
            mse=np.divide(np.nansum((pred-target)**2,0),n,out=np.full(target.shape,np.nan),where=n>0)
            for c,b in zip(*np.nonzero(p['common_mask'])):
                rows.append({'animal':animal,'representation':rep,'condition_id':int(p['condition_ids'][c]),
                    'time_ms':float(p['times_ms'][b]),'n_test_splits':int(n[0,c,b]),
                    'RMSE_obj':float(np.sqrt(mse[0,c,b])),'RMSE_beh':float(np.sqrt(mse[1,c,b])),
                    'Delta_RMSE':float(np.sqrt(mse[0,c,b])-np.sqrt(mse[1,c,b])),
                    'bias_obj':float(mean[0,c,b]-target[0,c,b]),'bias_beh':float(mean[1,c,b]-target[1,c,b]),
                    'target_gap':float(target[1,c,b]-target[0,c,b]),'decoded_gap_mean':float(mean[1,c,b]-mean[0,c,b])})
    pd.DataFrame(rows).to_csv(root/'results/binwise_reconstruction.csv',index=False,float_format='%.17g')


def polish_report(root, output_dir=None):
    """Render the maintained narrative without recomputing scientific outputs."""
    from trajectory_project.publication_reports import render_condition
    return render_condition(root, output_dir if output_dir is not None else root)


def make_report(data, root):
    """Historical full-output path; use polish_report for saved-table rendering only."""
    root = Path(root)
    frames, main, cross, conditions, coverage, comparison = collect(data, root)
    from trajectory_project.condition_endpoint_fa_gpfa_v1.final_validation import validate_delivery
    validate_delivery(data, root)
    time_diagnostics(data, root)
    from trajectory_project.condition_endpoint_fa_gpfa_v1.plotting_condition import make_plots
    make_plots(data, frames['self'], root)
    counts = []
    for (animal, rep, epoch), group in conditions[conditions.coordinate.eq('y')].groupby(['animal', 'representation', 'epoch']):
        counts.append({'animal': animal, 'representation': rep, 'epoch': epoch,
            'n_valid_conditions': int(group.Delta_RMSE.notna().sum()),
            'candidate_lower_RMSE': int(group.Delta_RMSE.gt(0).sum()),
            'candidate_higher_r': int(group.Delta_r.gt(0).sum())})
    pd.DataFrame(counts).to_csv(root/'results/condition_difference_counts.csv', index=False)
    polish_report(root)
