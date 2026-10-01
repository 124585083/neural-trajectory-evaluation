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


def polish_report(root):
    """Readable statements calculated from completed, unchanged score tables."""
    import re
    root=Path(root)
    new=pd.read_csv(root/'results/self_reconstruction_main_table.csv')
    old=pd.read_csv(SOURCE/'results/self_reconstruction_main_table.csv')
    full=new[new.coordinate.eq('y')&new.epoch.eq('full')]
    previous=old[old.coordinate.eq('y')&old.epoch.eq('full')]
    assert len(full)==4 and full.n_rounds.eq(100).all()
    statements=[]
    if full.Delta_r.lt(0).all() and full.Delta_RMSE.lt(0).all():
        statements.append('**The objective path retains better full-epoch own-target reconstruction in all four groups after condition averaging. The RMSE gap between the two targets is smaller.**')
    gap=-full.Delta_RMSE;oldgap=-previous.Delta_RMSE
    statements.append(f'The candidate full-epoch RMSE exceeds the objective RMSE by {gap.min():.4f}–{gap.max():.4f}; the trial-label version had gaps of {oldgap.min():.4f}–{oldgap.max():.4f}. This is a descriptive comparison of two executed pipelines. Their difference cannot be attributed entirely to behavioral averaging.')
    no_bounce=new[new.coordinate.eq('y')&new.epoch.eq('no_bounce')]
    post=new[new.coordinate.eq('y')&new.epoch.eq('post_bounce')]
    if no_bounce.Delta_RMSE.gt(0).all() and no_bounce.Delta_r.lt(0).all():
        statements.append(f'No-collision segments are an exception: candidate RMSE is lower by {no_bounce.Delta_RMSE.min():.4f}–{no_bounce.Delta_RMSE.max():.4f}, while correlation is lower. The two metrics do not jointly favor the candidate.')
    if post.Delta_RMSE.lt(0).all():
        statements.append(f'After collision, candidate RMSE is higher by {-post.Delta_RMSE.max():.4f}–{-post.Delta_RMSE.min():.4f}. These temporal and condition-type differences should remain visible beside full-epoch averages.')
    path=root/'REPORT.md';text=path.read_text(encoding='utf-8')
    tag='COMPLETED_SCORE_INTERPRETATION'
    text=re.sub(r'<!-- BEGIN '+tag+r' -->.*?<!-- END '+tag+r' -->\n\n','',text,flags=re.S)
    anchor='## 1. Own-target reconstruction results\n\n';assert anchor in text
    text=text.replace(anchor,anchor+'<!-- BEGIN '+tag+' -->\n'+'\n\n'.join(statements)+'\n<!-- END '+tag+' -->\n\n',1)
    path.write_text(text,encoding='utf-8')
    write_json(root/'results/interpretation_summary.json',{'statements':statements,
        'metric_comparison_is_descriptive_not_significance':True,
        'sources':[record(root/'results/self_reconstruction_main_table.csv'),record(SOURCE/'results/self_reconstruction_main_table.csv')]})


def make_report(data,root):
    root=Path(root)
    frames,main,cross,conditions,coverage,comparison=collect(data,root)
    from trajectory_project.condition_endpoint_fa_gpfa_v1.final_validation import validate_delivery
    validate_delivery(data,root)
    time_diagnostics(data,root)
    from trajectory_project.condition_endpoint_fa_gpfa_v1.plotting_condition import make_plots
    make_plots(data,frames['self'],root)
    y=main[main.coordinate.eq('y')];full=y[y.epoch.eq('full')]
    descriptions=[]
    for row in full.itertuples():
        if row.Delta_r>0 and row.Delta_RMSE>0:verdict='The condition-mean candidate has better own-target reconstruction'
        elif row.Delta_r<0 and row.Delta_RMSE<0:verdict='The objective path has better own-target reconstruction'
        else:verdict='Correlation and RMSE do not jointly favor either target'
        descriptions.append(f'{row.animal.capitalize()} / {row.representation}：{verdict}，Δr={row.Delta_r:+.4f}，ΔRMSE={row.Delta_RMSE:+.4f}。')
    summary='\n\n'.join(descriptions)
    stats=full[['animal','representation']].copy()
    for name in METRICS:stats[name+' (mean ± SD)']=[f'{m:.4f} ± {s:.4f}' for m,s in zip(full[name],full[name+'_sd'])]
    cols=['animal','representation','epoch',*METRICS]
    cover=pd.read_csv(root/'results/condition_label_coverage.csv').groupby('animal').agg(
        source_behavior_records=('n_all_source_behavior_records','sum'),valid_trials_in_mean=('n_behavior_trials_in_mean','sum'),
        valid_conditions=('n_condition_time_rows',lambda v:int((v>0).sum())),condition_time_rows=('n_condition_time_rows','sum')).reset_index()
    own=cross[cross.coordinate.eq('y')&cross.epoch.eq('full')&(((cross['head']=='D_obj')&(cross.target=='objective'))|((cross['head']=='D_beh')&(cross.target=='behavior')))]
    counts=[]
    for (a,r,e),g in conditions[conditions.coordinate.eq('y')].groupby(['animal','representation','epoch']):
        counts.append({'animal':a,'representation':r,'epoch':e,'n_valid_conditions':int(g.Delta_RMSE.notna().sum()),
            'candidate_lower_RMSE':int(g.Delta_RMSE.gt(0).sum()),'candidate_higher_r':int(g.Delta_r.gt(0).sum())})
    counts=pd.DataFrame(counts);counts.to_csv(root/'results/condition_difference_counts.csv',index=False)
    fullold=comparison[comparison.epoch.eq('full')&comparison.coordinate.eq('y')]
    text=f'''# Condition-mean candidates with FA50/GPFA50 and OLS position reconstruction

## 1. Own-target reconstruction results

{summary}

Delta_r=r_beh-r_obj; Delta_RMSE=RMSE_obj-RMSE_beh. A positive difference favors the candidate on the corresponding own-target metric. Score the original held-out condition-by-time predictions in each split before summarizing 100 splits. Scores of averaged prediction curves are separate descriptions.

{md(stats)}

**This version uses condition-mean neural inputs and condition-mean behavioral targets. It adds no single-trial neural information. raw_preprocessing remains fail. Fit-source isolation and prefix checks apply to the published inputs and do not establish future-free raw responses.**

## 2. Changes and preserved methods

For each condition, average the final paddle endpoints of the same valid trial members used in the preceding version. Endpoint interpolation is affine, so this equals averaging their candidate paths. Membership is fixed over all valid times. No filtering uses behavioral accuracy, endpoint deviation, or decoding performance. Copy objective x directly so both readout targets have bit-identical x values.

Replace trial-by-time expansion with one row per condition and time. Trial counts describe mean membership and provide no weights. Longer valid sequences contribute more rows, as in time-point expansion; there is no extra equal-condition weight. One intercept OLS mapping covers all valid visible and hidden samples: 50 inputs, 51 coefficients per coordinate, 102 per xy head, no regularization, new scaling, or added features.

Read the preceding version's FA50 and GPFA50 fits from each 39-condition training split. GPFA has one shared learnable RBF time scale. Reuse half1 latents without refitting, averaging halves, or selecting another model. Preserve the 100 nominal 39/40 splits, completed-bin right edges at 50 ms, neuron order, input scale, physical paths, collision anchors, and time support. See the [configuration](configs/analysis_protocol.json), [representation reuse audit](results/reused_representation_audit.json), and [protocol differences](protocol_diff.md).

## 3. Coverage and label checks

{md(cover,digits=0)}

All 79 physical conditions retain their split identities. Condition 59920 has an unresolved collision anchor and remains unknown, with no replacement or objective-value fill. Each animal therefore has 78 valid conditions and 3369 condition-by-time rows. Actual valid counts can be 38/39 for training and 39/40 for testing; nominal allocation stays 39/40. Behavioral means use 7407 Mahler and 84873 Perle records. These are not independent neural trials.

The [coverage table](results/condition_label_coverage.csv) preserves all identities, source record counts, valid endpoints, mean members, and missingness. The [averaging audit](results/condition_averaging_audit.json) checks fixed members and support, the mean-endpoint formula, missing values, and identical x. The [membership index](sources/source_trial_membership.json) traces records; [condition-by-time indices](data/) list both targets and their time coordinates.

Behavioral members are traceable, but the original unit-level members of published half1 neural means are unavailable. Alignment is by physical condition; exact neural-behavioral trial membership is unverified. The endpoint is a released terminal paddle scalar with unverified strictly pre-feedback timing. Candidates are retrospective labels. Collision anchors are estimated from released paths, not precise trial event logs. Feedback/end bins remain excluded; mixed occlusion-boundary bins enter full only, not both pure phase masks.

## 4. Phase-specific own-target reconstruction

Position and RMSE use MWorks centered display-coordinate units. Main-table condition/bin counts describe unique support; per-split counts are also retained. Behavioral trial counts describe mean membership. The [complete table](results/self_reconstruction_main_table.csv) includes data_mode, causal_status, counts, x, correlation, RMSE, and split SD.

{md(y,cols)}

The 100 splits reuse the same conditions. SD describes split stability, not 100 independent experiments. Condition count limits interpretation. [Split scores](results/round_self_metrics.csv) and [condition comparisons](results/condition_self_comparison.csv) preserve NA and test frequencies. Correlation is NA for constant targets; RMSE can remain defined.

![Paired own-target differences](figures/paired_self_reconstruction_differences.png)

## 5. Conditions, time, offsets, and amplitude

{md(counts[counts.epoch.isin(['full','hidden','post_bounce'])],digits=0)}

Aggregate averages do not replace individual-condition results. [Binwise errors](results/binwise_reconstruction.csv) retain actual times, test counts, both own-target RMSE/bias values, target separation, and prediction separation. These errors use individual held-out predictions rather than replacing the main score with an averaged-curve error.

Full-epoch amplitude and position-offset diagnostics:

{md(own,['animal','representation','head','target_sd','prediction_sd','bias','amplitude_ratio'])}

Bias is prediction minus its target. A standard-deviation ratio below one indicates compressed prediction variation. Opposing condition biases can cancel in the full epoch; inspect individual curves before interpreting every error as one common vertical offset.

Each animal's 20-page atlas includes all 79 conditions in ID order. Each condition shows FA/GPFA objective paths, objective-head reconstructions, condition-mean candidates, and candidate-head reconstructions. Axes are shared; occlusion, estimated collision intervals, and T are marked. Condition 59920 has no invented candidate. Shading denotes held-out split stability. [Mahler atlas](figures/mahler_all79_condition_atlas.pdf), [Perle atlas](figures/perle_all79_condition_atlas.pdf), and [figure provenance](figures/figure_manifest.json).

## 6. Comparison with the trial-label version

{md(fullold,['animal','representation','r_obj_trial_version','r_obj_condition_mean','r_beh_trial_version','r_beh_condition_mean','RMSE_obj_trial_version','RMSE_obj_condition_mean','RMSE_beh_trial_version','RMSE_beh_condition_mean'])}

See the [full phase comparison](results/previous_trial_version_comparison.csv). This version removes both within-condition target trial variation and the larger fitting/scoring contribution of conditions with more trials. The preceding expanded OLS is algebraically equivalent to fitting condition means weighted by valid trial count. This version has no such weights. The comparison is therefore not a one-factor averaging ablation. Neural representations are identical in both versions.

## 7. Cross-scoring and interpretation

The [complete 2x2 table](results/cross_2x2_summary.csv) reports every readout against both targets. E_OO versus E_BB is the own-target comparison. E_OB versus E_BB fixes the candidate target; E_OO versus E_BO fixes the objective target. x targets, weights, and predictions agree for a given representation. x provides an implementation check rather than separate behavioral evidence.

Target variance and reconstruction difficulty may differ. Lower own-target error does not directly establish a neural preference. Condition means cannot resolve trial differences within a condition or establish an internal path or unique algorithm. Averaging targets and refitting OLS do not remove upstream filling across conditions and times: raw_preprocessing=fail; provided_input_filtering and fit_provenance pass only within the published-input scope.

## 8. Reproduction and outputs

Use the [README routes](README.md), [weight-replay audit](results/final_integrity_and_score_audit.json), [protocol](configs/analysis_protocol.json), [100 splits](configs/condition_splits_100.json), and [manifest](manifest.json). Per-split weights are in [models](readouts/models/); pure-test predictions are in [predictions](readouts/predictions/). Historical representations are referenced through verified sources. Source files remain unchanged.
'''
    (root/'REPORT.md').write_text(text,encoding='utf-8')
    (root/'protocol_diff.md').write_text('''# Differences from the trial-label version

| Item | Preceding version | This version |
|---|---|---|
| Behavioral supervision | Each recorded trial's own endpoint candidate | Mean of the same valid members within each condition; equivalent to a mean-endpoint candidate |
| OLS rows | Trial by time, repeating condition-mean latents | One condition-by-time row |
| Condition contribution | Trial count times valid bins | Valid bins; no extra equal-condition weights |
| Main score | Held-out trial-by-time targets | Held-out condition-by-time mean targets |
| Representation, dimension, half, preprocessing | Per-split train39 FA50/GPFA50, half1 | Exactly the same saved weights and latents; no representation refit |
| Splits, timing, geometry | 100 nominal39/40 splits, seed0, audited physical labels | Preserved |
| Readout capacity | 50 inputs plus intercept;102 coefficients per xy head | Preserved |
| Atlas | Fixed trial examples within each condition | Condition-mean targets and held-out reconstruction |
| Causal boundary | Published-input prefix pass; raw_preprocessing fail | Preserved |

The paper expands conditions by time and uses instantaneous unregularized OLS. This version returns to that row unit while retaining the current half1 FA/GPFA representation, per-split training provenance, completed-bin timing, and constructed candidate targets. The protocols are not identical. See the [preceding protocol and source audit](../trial_endpoint_fa_gpfa_v1/protocol_diff.md).

Target trial variation and condition weighting both change. Differences are not an isolated effect of averaging labels. New summaries use saved predictions without selecting models, conditions, time windows, or seeds.
''',encoding='utf-8')
    (root/'README.md').write_text('''# Condition-mean endpoint reconstruction

Run from an isolated analysis workspace configured through the module path adapter:

```powershell
& '.venv/Scripts/python.exe' -B 'trajectory_project/condition_endpoint_fa_gpfa_v1/run_analysis.py'
```

This version references the preceding 100 FA50/GPFA50 fits and does not refit representations. The historical runner validates existing OLS caches. Saved results can be summarized or checked separately:

```powershell
& '.venv/Scripts/python.exe' -B 'trajectory_project/condition_endpoint_fa_gpfa_v1/run_analysis.py' --report-only
& '.venv/Scripts/python.exe' -B 'trajectory_project/condition_endpoint_fa_gpfa_v1/run_analysis.py' --verify
& '.venv/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider 'trajectory_project/condition_endpoint_fa_gpfa_v1/tests'
```

Labels are in `artifacts/<animal>_condition_labels.npz`; data indices have one row per condition and time. Neural inputs come from the preceding version's `representations/<animal>/round_XXX/latents.npz`, using `FA50[0,c,t,:]` or `GPFA50[0,c,t,:]`; index0 is half1. This version does not re-average neural inputs. Behavioral mean members are indexed in sources/source_trial_membership.json.

Each `readouts/predictions/<animal>_<representation>_rXXX_test_predictions.npz` contains `predictions[head,condition,time,xy]` only on valid test rows. Head0 is objective; head1 is candidate. condition_ids, times_ms, test_condition_indices, and test_input_support specify identity and support. Training predictions stay NaN.

[Report](REPORT.md) · [Own-target table](results/self_reconstruction_main_table.csv) · [2x2](results/cross_2x2_summary.csv) · [Conditions](results/condition_self_comparison.csv) · [Preceding version comparison](results/previous_trial_version_comparison.csv) · [Atlas](figures/) · [Protocol differences](protocol_diff.md) · [Coverage](results/condition_label_coverage.csv)
''',encoding='utf-8')
    polish_report(root)
    print(full[cols].to_string(index=False),flush=True)
