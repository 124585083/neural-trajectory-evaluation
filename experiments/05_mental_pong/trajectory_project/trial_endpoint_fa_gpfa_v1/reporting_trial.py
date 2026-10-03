"""Report own-target trial reconstruction before any representational claims."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from trajectory_project.step3_io import write_json,record

METRICS=['r_obj','r_beh','RMSE_obj','RMSE_beh','Delta_r','Delta_RMSE']
EPOCHS=['full','visible','hidden','bounce','no_bounce','post_bounce']


def md(df,columns=None,digits=4):
    if columns is not None:df=df[columns]
    def fmt(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}'
        return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(map(fmt,row))+' |' for row in df.itertuples(index=False,name=None)])


def summarize(frame,keys,metrics):
    rows=[]
    for key,g in frame.groupby(keys,dropna=False,sort=True):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(keys,key));row['n_rounds']=len(g)
        for c in metrics:
            a=g[c].to_numpy(float);a=a[np.isfinite(a)]
            row[c]=float(a.mean()) if len(a) else np.nan
            row[c+'_sd']=float(a.std(ddof=0)) if len(a) else np.nan
            row[c+'_n_finite']=len(a)
        rows.append(row)
    return pd.DataFrame(rows)


def collect_tables(data,root):
    folder=root/'readouts/round_results'
    frames={}
    for kind in ('self','cross_2x2','condition','session','fit_audit'):
        paths=sorted(folder.glob('*_'+kind+'.csv'))
        if not paths:raise RuntimeError('Missing '+kind+' round results')
        frames[kind]=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    raw=frames['self']
    assert raw.groupby(['animal','representation']).iteration.nunique().eq(100).all()
    raw.to_csv(root/'results/round_self_metrics.csv',index=False,float_format='%.17g')
    frames['cross_2x2'].to_csv(root/'results/round_cross_2x2.csv',index=False,float_format='%.17g')
    frames['session'].to_csv(root/'results/round_session_metrics.csv.gz',index=False,float_format='%.17g')
    frames['condition'].to_csv(root/'results/round_condition_metrics.csv.gz',index=False,float_format='%.17g')
    frames['fit_audit'].to_csv(root/'results/ols_fit_audit.csv',index=False)
    keys=['animal','representation','data_mode','causal_status','epoch','coordinate']
    main=summarize(raw,keys,METRICS+['n_conditions','n_real_trials','n_neural_trials','n_bins','n_unique_neural_input_ids'])
    for name in ('n_conditions','n_real_trials','n_bins','n_unique_neural_input_ids'):
        main=main.rename(columns={name:name+'_per_round_mean'})
    cover=[]
    for animal,payload in data.items():
        for epoch in EPOCHS:
            mask=payload['epoch_masks'][epoch]
            seen=np.zeros((79,len(payload['times_ms'])),bool)
            np.logical_or.at(seen,payload['condition_index'],mask)
            cover.append({'animal':animal,'epoch':epoch,'n_conditions':int(seen.any(1).sum()),
                'n_real_trials':int(mask.any(1).sum()),'n_bins':int(mask.sum()),
                'n_unique_neural_input_ids':int(seen.sum()),'n_neural_trials':0})
    coverage=pd.DataFrame(cover)
    main=main.drop(columns=['n_neural_trials']).merge(coverage,on=['animal','epoch'],validate='many_to_one')
    main['epoch']=pd.Categorical(main.epoch,EPOCHS,ordered=True)
    main=main.sort_values(['animal','representation','coordinate','epoch'])
    main.to_csv(root/'results/self_reconstruction_main_table.csv',index=False,float_format='%.17g')
    coverage.to_csv(root/'results/coverage.csv',index=False)
    cross=summarize(frames['cross_2x2'],keys+['head','target'],['r','RMSE','bias','target_sd','prediction_sd','amplitude_ratio','target_min','target_max'])
    cross.to_csv(root/'results/cross_2x2_summary.csv',index=False,float_format='%.17g')
    conditions=summarize(frames['condition'],keys+['condition_id','head','target'],['r','RMSE','bias','target_sd','prediction_sd','amplitude_ratio'])
    conditions.to_csv(root/'results/condition_metrics_summary.csv',index=False,float_format='%.17g')
    sessions=summarize(frames['session'],keys+['session_id','head','target'],['r','RMSE','bias','target_sd','prediction_sd'])
    sessions.to_csv(root/'results/session_metrics_summary.csv',index=False,float_format='%.17g')
    base=['animal','representation','epoch','coordinate','condition_id']
    own=conditions[conditions['head'].eq('D_obj')&conditions.target.eq('objective')]
    beh=conditions[conditions['head'].eq('D_beh')&conditions.target.eq('behavior')]
    pairs=own[base+['r','RMSE','bias','amplitude_ratio']].merge(beh[base+['r','RMSE','bias','amplitude_ratio']],on=base,suffixes=('_obj','_beh'),validate='one_to_one')
    # Preserve within-round pairing even when only one r is undefined.
    source=frames['condition']; round_keys=base+['iteration']
    oo=source[source['head'].eq('D_obj')&source.target.eq('objective')]
    bb=source[source['head'].eq('D_beh')&source.target.eq('behavior')]
    paired_rounds=oo[round_keys+['r','RMSE']].merge(bb[round_keys+['r','RMSE']],on=round_keys,suffixes=('_obj','_beh'),validate='one_to_one')
    paired_rounds['Delta_r']=paired_rounds.r_beh-paired_rounds.r_obj
    paired_rounds['Delta_RMSE']=paired_rounds.RMSE_obj-paired_rounds.RMSE_beh
    deltas=paired_rounds.groupby(base,dropna=False)[['Delta_r','Delta_RMSE']].mean().reset_index()
    pairs=pairs.merge(deltas,on=base,validate='one_to_one')
    pairs.to_csv(root/'results/condition_self_comparison.csv',index=False,float_format='%.17g')
    return frames,main,cross,coverage,pairs,sessions


def actual_output_audit(root,data,frames):
    rows=[]
    for animal in ('mahler','perle'):
        for i in range(100):
            folder=root/'representations'/animal/f'round_{i:03d}'
            meta=json.loads((folder/'training.json').read_text())
            audit=json.loads((folder/'causal_audit.json').read_text())
            with np.load(folder/'latents.npz',allow_pickle=False) as a:
                common=a['common_mask']
                payload=data[animal]
                assert not np.any(payload['common_mask'] & ~common[payload['condition_index']]), 'Global label index contains unsupported neural rows; report per-round coverage instead.'
            rows.append({'animal':animal,'iteration':i,'n_neurons':meta['n_neurons'],
                'FA_n_iter':meta['FA_n_iter'],'FA_converged':meta['FA_strict_convergence'],
                'GPFA_n_iter':meta['GPFA_n_iter'],'GPFA_converged':meta['GPFA_converged'],
                'tau_seconds':meta['GPFA_tau_seconds'],'raw_preprocessing':audit['raw_preprocessing'],
                'provided_input_filtering':audit['provided_input_filtering'],'fit_provenance':audit['fit_provenance'],
                'prefix_max_error':audit['max_error'],'fa_fit_seconds':meta['fa_fit_seconds'],
                'gpfa_fit_seconds':meta['gpfa_fit_seconds']})
    audits=pd.DataFrame(rows)
    assert audits.provided_input_filtering.eq('pass').all() and audits.fit_provenance.eq('pass').all()
    audits.to_csv(root/'results/representation_training_and_causal_audit.csv',index=False)
    fit=frames['fit_audit']
    assert len(fit)==400
    assert fit.n_train_conditions_assigned.eq(39).all() and fit.n_test_conditions_assigned.eq(40).all()
    assert fit.target_averaging_before_fit.eq(False).all()
    assert fit.expanded_design_rank.eq(fit.unique_design_rank).all()
    x_error=float(max(fit.x_coefficient_max_abs_difference.max(),fit.x_intercept_abs_difference.max()))
    assert x_error<1e-10
    write_json(root/'results/completion_audit.json',{'completed_animal_rounds':200,'representation_pairs':200,'xy_OLS_heads':800,
        'all100_splits_per_animal':True,'trial_expanded_actual_OLS_fit':True,'averaged_labels_used_to_fit':False,
        'copied_rows_do_not_increase_design_rank':True,'x_weight_max_difference':x_error,
        'provided_input_future_test_max_error':float(audits.prefix_max_error.max()),
        'raw_preprocessing':'fail','strict_raw_future_free_analysis_completed':False,
        'independent_neural_trials':0,'data_mode':'trial_labels_with_mean_neural'})
    return audits


def binwise_diagnostics(data,root):
    """Exact trial and held-out-split error moments, not scores of an averaged curve."""
    outputs=[]
    for animal,payload in data.items():
        ci=payload['condition_index'];mask=payload['common_mask'];T=mask.shape[1]
        n=np.zeros((79,T)); sy=np.zeros((2,79,T)); sy2=np.zeros_like(sy)
        for c in range(79):
            ix=np.flatnonzero(ci==c);m=mask[ix]
            n[c]=m.sum(0)
            obj=np.broadcast_to(payload['objective_xy'][c,:,1],m.shape)
            beh=payload['behavior_xy'][ix,:,1]
            for h,y in enumerate((obj,beh)):
                sy[h,c]=np.where(m,y,0).sum(0);sy2[h,c]=np.where(m,y*y,0).sum(0)
        mean_y=np.divide(sy,n[None],out=np.full_like(sy,np.nan),where=n[None]>0)
        second_y=np.divide(sy2,n[None],out=np.full_like(sy2,np.nan),where=n[None]>0)
        for rep in ('FA50','GPFA50'):
            values=[]
            for p in sorted((root/'readouts/predictions').glob(f'{animal}_{rep}_r*_test_predictions.npz')):
                with np.load(p,allow_pickle=False) as a:values.append(a['predictions'][...,1])
            pred=np.stack(values)
            count=np.isfinite(pred).sum(0)
            mp=np.divide(np.nansum(pred,0),count,out=np.full_like(mean_y,np.nan),where=count>0)
            p2=np.divide(np.nansum(pred*pred,0),count,out=np.full_like(mean_y,np.nan),where=count>0)
            rmse=np.sqrt(np.maximum(p2-2*mp*mean_y+second_y,0))
            for c,cid in enumerate(payload['condition_ids']):
                for b in np.flatnonzero((n[c]>0)&(count[0,c]>0)):
                    outputs.append({'animal':animal,'representation':rep,'condition_id':int(cid),'time_ms':float(payload['times_ms'][b]),
                        'original_bin_index':int(b),'n_real_trials':int(n[c,b]),'n_test_splits':int(count[0,c,b]),
                        'RMSE_obj':rmse[0,c,b],'RMSE_beh':rmse[1,c,b],'Delta_RMSE':rmse[0,c,b]-rmse[1,c,b],
                        'bias_obj':mp[0,c,b]-mean_y[0,c,b],'bias_beh':mp[1,c,b]-mean_y[1,c,b],
                        'trial_label_sd':np.sqrt(max(second_y[1,c,b]-mean_y[1,c,b]**2,0))})
    frame=pd.DataFrame(outputs);frame.to_csv(root/'results/binwise_trial_reconstruction.csv',index=False,float_format='%.17g')
    return frame


def fixed_case_metrics(data,root):
    examples=json.loads((root/'configs/figure_trial_examples.json').read_text())
    selected=pd.DataFrame([r for r in examples if r['trial_index'] is not None])
    selected=selected.sort_values('condition_id').groupby(['animal','bounce_class'],sort=True).head(1)
    records=[]
    for (animal,rep) in [(a,r) for a in data for r in ('FA50','GPFA50')]:
        payload=data[animal]
        arrays=[]
        for p in sorted((root/'readouts/predictions').glob(f'{animal}_{rep}_r*_test_predictions.npz')):
            with np.load(p,allow_pickle=False) as z:arrays.append(z['predictions'][...,1])
        predictions=np.stack(arrays)
        for item in selected[selected.animal.eq(animal)].itertuples(index=False):
            ti,ci=int(item.trial_index),int(item.condition_index)
            for epoch in ('full','hidden','post_bounce'):
                mask=payload['epoch_masks'][epoch][ti]
                bins=np.flatnonzero(mask)
                if not len(bins):continue
                row={'animal':animal,'representation':rep,'condition_id':item.condition_id,
                    'trial_id':item.trial_id,'bounce_class':item.bounce_class,'epoch':epoch,
                    'first_bin_ms':payload['times_ms'][bins[0]],'last_bin_ms':payload['times_ms'][bins[-1]],'n_bins':len(bins)}
                for hi,head in enumerate(('obj','beh')):
                    target=payload['objective_xy'][ci,mask,1] if hi==0 else payload['behavior_xy'][ti,mask,1]
                    p=predictions[:,hi,ci][:,mask]
                    p=p[np.isfinite(p).all(1)]
                    row['n_test_splits']=len(p)
                    target_sd=np.std(target)
                    row[f'bias_{head}']=float(np.mean(p-target))
                    row[f'RMSE_{head}']=float(np.sqrt(np.mean((p-target)**2,axis=1)).mean())
                    row[f'amplitude_ratio_{head}']=float(np.std(p,axis=1).mean()/target_sd) if target_sd>0 else np.nan
                    row[f'target_net_change_{head}']=float(target[-1]-target[0])
                    row[f'prediction_net_change_{head}']=float(np.mean(p[:,-1]-p[:,0]))
                row['Delta_RMSE']=row['RMSE_obj']-row['RMSE_beh']
                records.append(row)
    result=pd.DataFrame(records)
    result.to_csv(root/'results/fixed_trial_case_metrics.csv',index=False,float_format='%.17g')
    return result


def polish_report(root):
    """Add plain-language findings derived only from the completed score tables."""
    import re
    root=Path(root)
    main=pd.read_csv(root/'results/self_reconstruction_main_table.csv')
    y=main[main.coordinate.eq('y')]
    full=y[y.epoch.eq('full')]
    assert len(full)==4 and full.n_rounds.eq(100).all()
    objective_better=bool(full.Delta_r.lt(0).all() and full.Delta_RMSE.lt(0).all())
    phases=y.pivot(index=['animal','representation'],columns='epoch',values='Delta_RMSE')
    larger_hidden_gap=bool((phases.hidden<phases.visible).all())
    if objective_better:
        finding='**FA50 and GPFA50 agree in both animals: objective own-target reconstruction has higher correlation and lower RMSE than individual-trial endpoint candidates. The hypothesis of easier candidate reconstruction is unsupported in this route-B diagnostic using repeated condition means.**'
    else:
        finding='Own-target results do not uniformly favor one label. Interpret each animal, representation, and metric separately.'
    if larger_hidden_gap:
        finding+=' Candidate RMSE disadvantages are larger in hidden than visible samples in all four groups. Full-epoch correlation alone does not describe this temporal difference.'
    cross=pd.read_csv(root/'results/cross_2x2_summary.csv')
    own=cross[cross.coordinate.eq('y')&cross.epoch.eq('full')&(((cross['head']=='D_obj')&(cross.target=='objective'))|((cross['head']=='D_beh')&(cross.target=='behavior')))]
    counts=pd.read_csv(root/'results/condition_self_comparison.csv')
    counts=counts[counts.coordinate.eq('y')&counts.epoch.eq('full')]
    benefit=counts.groupby(['animal','representation']).Delta_RMSE.apply(lambda a:int((a>0).sum()))
    shapes=(f'Full-epoch mean bias ranges from {own.bias.min():.3f} to {own.bias.max():.3f}, but prediction/target SD ratios are only {own.amplitude_ratio.min():.3f}–{own.amplitude_ratio.max():.3f}. Both readouts compress position variation. A common vertical offset does not explain the full error, and near-zero pooled bias can hide cancellation across conditions.'
            f'Candidate disadvantage is not universal across conditions: {int(benefit.min())}–{int(benefit.max())} of78 valid conditions per group have lower candidate RMSE. All condition and time-bin results remain available.')
    path=root/'REPORT.md';text=path.read_text(encoding='utf-8')
    for tag in ('MAIN_FINDING','SHAPE_FINDING'):
        text=re.sub(r'<!-- BEGIN '+tag+r' -->.*?<!-- END '+tag+r' -->\n\n','',text,flags=re.S)
    anchor='## 1. Reconstruction of each training target\n\n'
    assert anchor in text
    text=text.replace(anchor,anchor+'<!-- BEGIN MAIN_FINDING -->\n'+finding+'\n<!-- END MAIN_FINDING -->\n\n',1)
    anchor='Candidate endpoint ranges, variances, and within-condition trial variance at each time'
    assert anchor in text
    text=text.replace(anchor,'<!-- BEGIN SHAPE_FINDING -->\n'+shapes+'\n<!-- END SHAPE_FINDING -->\n\n'+anchor,1)
    path.write_text(text,encoding='utf-8')
    write_json(root/'results/interpretation_summary.json',{'all_four_full_y_objective_better_on_both_metrics':objective_better,
        'all_four_hidden_RMSE_gap_larger_than_visible':larger_hidden_gap,'full_own_bias_range':[float(own.bias.min()),float(own.bias.max())],
        'full_own_amplitude_ratio_range':[float(own.amplitude_ratio.min()),float(own.amplitude_ratio.max())],
        'condition_counts_with_candidate_lower_RMSE':{f'{a}/{r}':int(n) for (a,r),n in benefit.items()},
        'sources':[record(root/'results'/name) for name in ('self_reconstruction_main_table.csv','cross_2x2_summary.csv','condition_self_comparison.csv')]})


def make_report(data,root):
    root=Path(root)
    frames,main,cross,coverage,conditions,sessions=collect_tables(data,root)
    audits=actual_output_audit(root,data,frames)
    bins=binwise_diagnostics(data,root)
    cases=fixed_case_metrics(data,root)
    from trajectory_project.trial_endpoint_fa_gpfa_v1.plotting_trial import make_plots
    make_plots(data,frames['self'],root)
    from trajectory_project.trial_endpoint_fa_gpfa_v1.final_validation import validate_delivery
    validate_delivery(data,root)
    from trajectory_project.trial_endpoint_fa_gpfa_v1.hierarchy_summary import write_unit_heterogeneity
    hierarchy=write_unit_heterogeneity(data,root,frames)['summary']
    audit=json.loads((root/'results/data_pairing_audit.json').read_text())
    y=main[main.coordinate.eq('y')]
    core=y[y.epoch.isin(['full','visible','hidden'])]
    def v(animal,rep,epoch,column):
        return float(y[(y.animal==animal)&(y.representation==rep)&(y.epoch==epoch)][column].iloc[0])
    def compare(animal,rep):
        dr=v(animal,rep,'full','Delta_r');de=v(animal,rep,'full','Delta_RMSE')
        return f'{animal.capitalize()} / {rep}：Δr={dr:+.4f}，ΔRMSE={de:+.4f}。'
    concordance=[]
    for animal in ('mahler','perle'):
        signs={m:np.sign(v(animal,'FA50','full',m))==np.sign(v(animal,'GPFA50','full',m)) for m in ('Delta_r','Delta_RMSE')}
        concordance.append(f'{animal.capitalize()}: FA and GPFA full-epoch RMSE effects '+('agree' if signs['Delta_RMSE'] else 'differ')+'; correlation effects '+('agree' if signs['Delta_r'] else 'differ')+'.')
    def sh(animal,rep,head,col,epoch='full'):
        target='objective' if head=='D_obj' else 'behavior'
        frame=cross[(cross.animal==animal)&(cross.representation==rep)&(cross.epoch==epoch)&cross.coordinate.eq('y')&cross['head'].eq(head)&cross.target.eq(target)]
        return float(frame[col].iloc[0])
    cover=[]
    for a,p in audit['animals'].items():
        cover.append({'animal':a,'behavior_records':p['all79_occ_records'],'valid_endpoints':p['n_valid_terminal_endpoints'],
            'valid_candidate_trials':p['n_trials_with_candidate_labels'],'session':p['n_sessions'],'regression_rows':p['n_label_rows'],'independent_neural_trials':0})
    condition_counts=[]
    for (a,r,e),g in conditions[conditions.coordinate.eq('y')].groupby(['animal','representation','epoch']):
        condition_counts.append({'animal':a,'representation':r,'epoch':e,'valid_conditions':int(g.Delta_RMSE.notna().sum()),
            'conditions_with_lower_candidate_RMSE':int((g.Delta_RMSE>0).sum()),'conditions_with_higher_candidate_r':int((g.Delta_r>0).sum())})
    condition_counts=pd.DataFrame(condition_counts)
    condition_counts.to_csv(root/'results/condition_difference_counts.csv',index=False)
    tablecols=['animal','representation','epoch',*METRICS]
    stability=core[core.epoch.eq('full')][['animal','representation']].copy()
    for metric in METRICS:
        stability[metric+' (mean ± SD)']=[f'{m:.4f} ± {s:.4f}' for m,s in zip(core[core.epoch.eq('full')][metric],core[core.epoch.eq('full')][metric+'_sd'])]
    shapecols=['animal','representation','epoch','head','target_sd','prediction_sd','bias','amplitude_ratio']
    shape=cross[(cross.coordinate=='y')&(cross.epoch=='full')&(((cross['head']=='D_obj')&(cross.target=='objective'))|((cross['head']=='D_beh')&(cross.target=='behavior')))]
    text=f'''# Trial endpoint candidates with FA50/GPFA50 and OLS position reconstruction

## 1. Reconstruction of each training target

Compare each recorded behavioral trial's objective target with objective-head predictions and its own endpoint candidate with candidate-head predictions. Score each held-out trial and time before summarizing 100 complete condition splits. Trial targets are not averaged before the primary score.

{compare('mahler','FA50')} {compare('mahler','GPFA50')}

{compare('perle','FA50')} {compare('perle','GPFA50')}

{' '.join(concordance)}

Delta_r=r_beh-r_obj; Delta_RMSE=RMSE_obj-RMSE_beh. Positive differences favor the candidate on the corresponding own-target metric. Target ranges, variances, and trial variation differ; lower error alone does not establish neural preference.

**The local data lack paired single-trial neural and behavioral records. data_mode=trial_labels_with_mean_neural throughout: recorded behavioral trial targets repeat the same condition-mean neural input. This is descriptive. raw_preprocessing=fail; strict future-free raw-response processing was not established. Training-source isolation and prefix checks below apply only to published inputs and cannot remove upstream dependencies.**

## 2. Data, endpoints, and available analysis route

{md(pd.DataFrame(cover),digits=0)}

All 79 condition identities enter each nominal 39/40 split.78 conditions have usable candidates;59920 has an unresolved terminal collision anchor and retains its identity with zero valid contribution. Representations still fit the specified39 training conditions, while OLS valid counts can be 38/39 for training and 39/40 for testing; see the [fit audit](results/ols_fit_audit.csv). Unknown records also retain identities:576 Mahler and3641 Perle terminal scalars disagree with display positions and may be defaults. Successes and failures remain included without accuracy or error-size selection.

Released paddle_y is assigned from joystick_output as a terminal-position scalar, rather than treating a joystick control curve as position. Comparison with final display samples differs by about one0.17-unit discrete control step; no correction was imposed. Strictly pre-feedback sampling remains unverified. No duplicate session+t_sync record keys were found. Physical initial parameters agree within conditions, allowing objective targets to be broadcast by real-trial identity. This adds no neural trials. Raw trial responses and stable/half membership are unavailable. [Pairing audit](results/data_pairing_audit.json), [labels and records](artifacts/), and [sample indices](data/) preserve provenance.

## 3. Candidate rules and time support

Without a collision, connect task initial(x0,y0) to the trial endpoint. With a collision, preserve the objective path before the estimated anchor and connect that anchor to the endpoint afterward. x_beh=x_obj throughout; horizontal progress reaches b_i at common x_end=10 and common T. There is no stop detection, stability threshold, repeated proxy update, visible-phase reset, clipping, or new reflection. The target is a retrospective endpoint-constrained candidate, not a measured internal path.

Released bins average integer-ms samples and have centers50k+24.5ms; complete neural bins are available at50k+50ms. Objective branches are evaluated at bin right edges. The25.5ms shift follows timing definitions and was not selected. Collision estimates use released50ms physical branches with uncertainty intervals. Two conditions with only one post-collision point use a known single-reflection constraint and are labeled low-support estimates. T comes from the x branch reaching the paddle plane. Differences from design timing and metadata y(T) remain in the [geometry table](results/condition_geometry.csv), without adjustments toward paper or behavioral results.

Score completed bins before estimated arrival and within trusted released neural support; exclude bins crossing feedback/end boundaries. Exact feedback times remain unknown. Mixed occlusion-boundary bins enter full only, so pure visible and hidden bin counts may sum to less than full. Phase masks define scoring, not candidate activation. [Label checks](results/actual_label_validation.json) verify endpoints, continuity, common x, and missing labels; the [geometry protocol](configs/label_geometry_protocol.json) records estimation rules.

## 4. FA/GPFA and the paper OLS framework

Both animals use the same100 physical-condition39/40 splits with seed0. Each split refits finite/variance neuron selection, fixed normalization, FA50, and shared-time-scale GPFA50 on training39 half1 only. The same-round FA50 initializes GPFA50. FA preserves300 iterations, tolerance.01, LAPACK, and seed42; GPFA preserves400 iterations and tolerance1e-6. No parameter or seed search was performed. Local fits do not directly read test conditions, although published imputation creates upstream indirect dependencies. Earlier GPFA64 and paper-reference full-condition FA50 remain separate historical fits.

All four OLS groups use50 inputs,51 coefficients per coordinate, and 102 per xy head. LinearRegression(fit_intercept=True,positive=False) uses float64 with no regularization, added scaling, or sample weights. Training trial-by-time rows are explicitly expanded. Four outputs share a matrix decomposition, numerically equivalent to separate position regressions. Conditions with more trials contribute more rows. One fixed mapping covers visible and hidden times; phases change scoring only. See [protocol differences](protocol_diff.md) and [configuration](configs/analysis_protocol.json).

Future perturbation, explicit-prefix endpoint comparisons, and cross-condition state resets were checked in every representation split; maximum error is {audits.prefix_max_error.max():.3g}. Local fit_provenance and provided_input_filtering pass. Released arrays already contain imputation using all conditions and times, leaving irreversible raw_preprocessing=fail. [Training and causal audits](results/representation_training_and_causal_audit.csv) also retain convergence status and fixed-iteration-limit outcomes without downstream-score-based refitting.

## 5. Own-target main results

The table below reports y. Position/RMSE units are centered MWorks display coordinates; table times use ms and plots use seconds. Scores average 100 held-out splits. The [full table](results/self_reconstruction_main_table.csv) retains data_mode, causal_status, and counts. n_conditions, n_real_trials, and n_bins describe unique full-data support; per-split means are separate. n_neural_trials remains 0 because repeated regression rows are not independent neural trials.

{md(core,tablecols)}

Full-epoch variation across splits:

{md(stability)}

Collision, no-collision, and post-collision scores use the same fixed heads:

{md(y[y.epoch.isin(['bounce','no_bounce','post_bounce'])],tablecols)}

x consistency reference: the two heads have identical x predictions, so only one is shown here; phase-specific x remains in the full table.

{md(main[main.coordinate.eq('x')&main.epoch.eq('full')],['animal','representation','r_obj','r_obj_sd','RMSE_obj','RMSE_obj_sd'])}

Variation across 100 splits describes readout/representation split stability on shared data. It is not 100 independent experiments. Lossless [trial scores](readouts/trial_scores/) preserve original scores, with [condition](results/condition_metrics_summary.csv) and [session](results/session_metrics_summary.csv) summaries.

A separate [descriptive heterogeneity table](results/descriptive_unit_heterogeneity.csv) first averages each trial, condition, or session across its held-out splits, then reports count, SD, median, Q25/Q75, and finite pair counts. Paired differences are formed within split before aggregation. These hierarchical descriptions are not confidence intervals over independent neural observations and do not replace primary trial-by-time scoring.

![Paired scores over 100 splits](figures/paired_self_reconstruction_differences.png)

## 6. Condition, time, offset, and amplitude

{md(condition_counts[condition_counts.epoch.isin(['full','hidden','post_bounce'])],digits=0)}

All condition and time-bin directions are retained without score-based selection. [Condition comparisons](results/condition_self_comparison.csv) and [binwise errors](results/binwise_trial_reconstruction.csv) locate differences. The latter retains actual time, behavioral record count, and test frequency, with errors against original trial targets rather than averaged curves.

Full-epoch offset/amplitude diagnostics follow. bias=prediction-target. Prediction/target SD ratios above one indicate greater predicted variation; ratios below one indicate compression. These ratios are not measures of information content.

{md(shape,shapecols)}

Candidate endpoint ranges, variances, and within-condition trial variance at each time are in the [label variability table](results/trial_label_variability.csv). They help explain possible r/RMSE disagreement. Targets were not smoothed or rescaled to ease reconstruction.

## 7. Trial-specific four-curve examples and all conditions

The [Mahler](figures/mahler_all79_trial_atlas.pdf) and [Perle](figures/perle_all79_trial_atlas.pdf) atlases cover all79 conditions. Trial ID determines one recorded trial per condition, shared between FA and GPFA with common axes. The target is that trial's own endpoint candidate, not a condition mean. Unknown anchors remain missing.

Each panel shows the objective path, objective-head reconstruction, trial candidate, and candidate-head reconstruction. Predictions use only splits holding out that condition; shading is split stability. Different trials within one condition have different candidates but identical predictions from the same mean latent, a route-B limitation. Collision, occlusion, and estimated T are marked; exact feedback timing is unknown. The [atlas manifest](figures/figure_manifest.json) indexes fixed trials and pages.

Examples below use the smallest condition ID per collision type and the atlas fixed trial. Diagnostics are computed per held-out split before aggregation. Bias describes offsets; net change compares first and last values. RMSE and complete time curves jointly describe own-target agreement, without treating a translation as a trend improvement. Exact phase ranges and amplitude ratios remain in the [fixed-trial table](results/fixed_trial_case_metrics.csv). Examples were not score-selected and do not replace all-trial results.

{md(cases[cases.epoch.eq('full')],['animal','representation','condition_id','trial_id','RMSE_obj','RMSE_beh','bias_obj','bias_beh','target_net_change_obj','prediction_net_change_obj','target_net_change_beh','prediction_net_change_beh'])}

## 8. Cross-scoring and evidence limits

E_OO/E_BB compare own-target reconstruction. E_OB/E_BB fix the candidate target; E_OO/E_BO fix the objective target. The [complete2x2 means/SDs](results/cross_2x2_summary.csv) retain all phases and x/y without mislabeling cross-errors as own-errors. Within a representation, both heads have identical x targets, weights, and predictions to numerical precision. FA and GPFA need not predict identical x.

For identical neural inputs within a condition, squared error decomposes as:

`sum_i ||y_i-f(z)||² = n||mean(y_i)-f(z)||² + sum_i||y_i-mean(y_i)||²`。

Real-data and unit checks confirm this identity. Trial supervision is retained and used in actual fitting; weighted means only test algebraic equivalence. Repeating inputs increases neither design rank nor trial-specific neural information. Exchanging trial labels within a condition cannot create a trial-specific neural test or independent significance.

This diagnostic asks how easily both paths can be reconstructed, when differences occur, and whether FA/GPFA agree under published condition-mean inputs and training-side representation isolation. It does not validate single-trial behavioral representations, future-free raw responses, or internal simulation. Unknown endpoint timing, condition-mean pseudopopulations, and global filling limit temporal, trial-specific, and causal interpretation respectively.

## 9. Files and reproduction

The [sample index](data/) identifies recorded trials and condition/bin pairs. [Models](readouts/models/) and [held-out predictions](readouts/predictions/) use lossless factorization: condition predictions plus actual test_trial_indices and exact common masks. Each trial/time prediction is recoverable without averaging labels or reducing scored rows. Exporters expand selected splits/trials. Representation weights, preprocessing, and provenance are in [representations](representations/); requests and runtime records are in [sources](sources/).

The [README](README.md), [completion checks](results/completion_audit.json), [file/weight audit](results/final_integrity_and_causal_audit.json), and [manifest](manifest.json) trace reproducibility. Historical code, models, and results remain preserved.
'''
    (root/'REPORT.md').write_text(text,encoding='utf-8')
    (root/'README.md').write_text('''# Trial endpoint candidates with FA50/GPFA50 and OLS

This is a trial_labels_with_mean_neural descriptive diagnostic without paired trial neural data. raw_preprocessing=fail. Each split fits50-dimensional FA/GPFA using39 training conditions; prefix-causal checks apply only to published inputs.

Run in a separately materialized, isolated analysis workspace:

```powershell
& '.venv/Scripts/python.exe' -B 'trajectory_project/trial_endpoint_fa_gpfa_v1/run_analysis.py' --workers 3 --decode-workers 3 --threads 2
```

Run an initial end-to-end check(round0 can be reused), or generate reports from completed models:

```powershell
& '.venv/Scripts/python.exe' -B 'trajectory_project/trial_endpoint_fa_gpfa_v1/run_analysis.py' --pilot-only
& '.venv/Scripts/python.exe' -B 'trajectory_project/trial_endpoint_fa_gpfa_v1/run_analysis.py' --report-only
& '.venv/Scripts/python.exe' -B 'trajectory_project/trial_endpoint_fa_gpfa_v1/run_analysis.py' --verify
& '.venv/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider 'trajectory_project/trial_endpoint_fa_gpfa_v1/tests'
```

Trial labels are in artifacts/*_trial_labels.npz and records in results/*_trial_records.csv.gz. Each trial retains its own candidate; objective targets are broadcast by verified condition identity. Predictions in readouts/predictions/*_test_predictions.npz combine predictions[head,condition,time,xy], test_trial_indices, and test_common_mask_packed to recover every held-out trial prediction. Training predictions remain NaN. readouts/trial_scores/*_self_scores.npz retains trial, phase, head, xy, and own-target metrics.

data/*_sample_index.csv.gz identifies trial, condition, and time. For one split, inputs are FA50[0,condition_index,time_index,:] and GPFA50[0,condition_index,time_index,:] in representations/<animal>/round_XXX/latents.npz;0 denotes half1. Targets are objective_xy[objective_condition_index,time_index,:] and behavior_xy[behavior_trial_index,time_index,:]. These are lossless indices, not extra neural trials. Rows sharing unique_neural_input_id add no independent neural information. Shared support was checked in all 100 splits.

Expand saved held-out trial predictions; omit --trial-id to export all real test trials in that split. This does not refit models:

```powershell
& '.venv/Scripts/python.exe' -B 'trajectory_project/trial_endpoint_fa_gpfa_v1/export_trial.py' --animal perle --representation GPFA50 --round 0 --trial-id 'perle:source_row_1' --output-gz 'trajectory_project/trial_endpoint_fa_gpfa_v1/results/export_examples/readme_example.csv.gz'
```

Add --scores to export phase-specific own-target metrics. A trial outside that split's test set does not receive an invented prediction. Check IDs against the record table and fixed splits.

[Report](REPORT.md) · [Protocol differences](protocol_diff.md) · [Main results](results/self_reconstruction_main_table.csv) · [2x2](results/cross_2x2_summary.csv) · [Causal audit](results/representation_training_and_causal_audit.csv) · [Pairing audit](results/data_pairing_audit.json) · [Fixed splits](configs/condition_splits_100.json) · [Sample index](data/) · [All-condition atlas](figures/)
''',encoding='utf-8')
    polish_report(root)
    print(core[tablecols].to_string(index=False),flush=True)
