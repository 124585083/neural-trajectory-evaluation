"""Independent scoring of the current version's already held-out predictions."""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from trajectory_project.step3_io import read_json, write_json, record
from trajectory_project.condition_endpoint_fa_gpfa_v1.condition_data import load_condition_data
from trajectory_project.closeout_v1.closeout_setup import ROOT, SOURCE, ANIMALS, REPS, EPOCHS

METRICS = ('r', 'RMSE', 'bias', 'target_sd', 'prediction_sd', 'amplitude_ratio', 'target_min', 'target_max')


def score(y, p):
    """Use centered Pearson sums and unweighted original-row errors."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not len(y):
        return {m: np.nan for m in METRICS}
    assert y.shape == p.shape and np.isfinite(y).all() and np.isfinite(p).all()
    yc, pc = y-y.mean(), p-p.mean()
    sy, sp = y.std(), p.std()
    norm = np.sqrt((yc@yc)*(pc@pc))
    r = float(np.clip((yc@pc)/norm, -1, 1)) if len(y)>=3 and np.ptp(y)>0 and np.ptp(p)>0 and norm>0 else np.nan
    return dict(r=r, RMSE=float(np.sqrt(np.mean((p-y)**2))), bias=float(np.mean(p-y)),
                target_sd=float(sy), prediction_sd=float(sp), amplitude_ratio=float(sp/sy) if sy>0 else np.nan,
                target_min=float(y.min()), target_max=float(y.max()))


def summarize(frame, keys, columns):
    rows = []
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        item = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        item['n_rounds'] = len(group)
        for col in columns:
            x = group[col].to_numpy(float)
            x = x[np.isfinite(x)]
            item.update({col: float(x.mean()) if len(x) else np.nan,
                         col+'_sd': float(x.std()) if len(x) else np.nan,
                         col+'_n_finite': len(x)})
        rows.append(item)
    return pd.DataFrame(rows)


def compare_numeric(new, old, keys, columns):
    joint = new.merge(old, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    assert len(joint) == len(new) == len(old)
    maximum = 0.
    for col in columns:
        a, b = joint[col+'_new'].to_numpy(float), joint[col+'_old'].to_numpy(float)
        np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11, equal_nan=True)
        if np.isfinite(a-b).any():
            maximum=max(maximum,float(np.nanmax(np.abs(a-b))))
    return maximum


def run():
    data = load_condition_data(SOURCE)
    splits = read_json(ROOT / 'configs/condition_splits_100.json')
    output = ROOT/'results/A'
    output.mkdir(parents=True, exist_ok=True)
    rows, selfrows, sources, xerror = [], [], [], 0.
    for animal, payload in data.items():
        common = payload['common_mask']
        assert common.sum()==3369 and common.any(1).sum()==78
        for split in splits:
            iteration = split['iteration']
            train, test = np.array(split['train_indices']), np.array(split['test_indices'])
            assert len(train)==39 and len(test)==40 and not set(train)&set(test)
            for rep in REPS:
                path = SOURCE/'readouts/predictions'/f'{animal}_{rep}_r{iteration:03d}_test_predictions.npz'
                sources.append(record(path,'immutable pure-test prediction for A and B'))
                with np.load(path,allow_pickle=False) as z:
                    predictions=z['predictions'].copy()
                    support=z['test_input_support'].copy()
                    np.testing.assert_array_equal(z['condition_ids'],payload['condition_ids'])
                    np.testing.assert_array_equal(z['times_ms'],payload['times_ms'])
                    np.testing.assert_array_equal(z['common_mask'],common)
                    np.testing.assert_array_equal(z['test_condition_indices'],test)
                expected=np.zeros_like(common);expected[test]=common[test]
                np.testing.assert_array_equal(support,expected)
                assert np.isnan(predictions[:,train]).all()
                assert np.isfinite(predictions[:,support]).all()
                xerror=max(xerror,float(np.max(np.abs(predictions[0,support,0]-predictions[1,support,0]))))
                for epoch in EPOCHS:
                    mask=support & payload['epoch_masks'][epoch]
                    n=int(mask.sum()); nc=int(mask.any(1).sum())
                    for axis,coordinate in enumerate(('x','y')):
                        scores={}
                        for h,head in enumerate(('D_obj','D_beh')):
                            for k,target in enumerate(('objective','behavior')):
                                y=payload['objective_xy' if k==0 else 'behavior_xy'][mask,axis]
                                value=score(y,predictions[h,mask,axis])
                                cell=('O' if h==0 else 'B')+('O' if k==0 else 'B')
                                scores[cell]=value
                                rows.append(dict(animal=animal,representation=rep,iteration=iteration,
                                    epoch=epoch,coordinate=coordinate,head=head,target=target,cell=cell,
                                    n_conditions=nc,n_bins=n,raw_preprocessing='fail',**value))
                        oo,bb=scores['OO'],scores['BB']
                        selfrows.append(dict(animal=animal,representation=rep,iteration=iteration,
                            epoch=epoch,coordinate=coordinate,n_conditions=nc,n_bins=n,
                            r_obj=oo['r'],r_beh=bb['r'],RMSE_obj=oo['RMSE'],RMSE_beh=bb['RMSE'],
                            Delta_r=bb['r']-oo['r'],Delta_RMSE=oo['RMSE']-bb['RMSE']))
        print(f'A independently rescored {animal}: 100 splits x 2 representations',flush=True)
    cross=pd.DataFrame(rows);own=pd.DataFrame(selfrows)
    crosskeys=['animal','representation','iteration','epoch','coordinate','head','target']
    maxcross=compare_numeric(cross,pd.read_csv(SOURCE/'results/round_cross_2x2.csv'),crosskeys,METRICS)
    ownkeys=['animal','representation','iteration','epoch','coordinate']
    owncols=['r_obj','r_beh','RMSE_obj','RMSE_beh','Delta_r','Delta_RMSE']
    maxown=compare_numeric(own,pd.read_csv(SOURCE/'results/round_self_metrics.csv'),ownkeys,owncols)
    cross_summary=summarize(cross,[k for k in crosskeys if k!='iteration']+['cell'],METRICS)
    maxsummary=compare_numeric(cross_summary,pd.read_csv(SOURCE/'results/cross_2x2_summary.csv'),
                              [k for k in crosskeys if k!='iteration'],METRICS)
    summary=summarize(own,[k for k in ownkeys if k!='iteration'],owncols+['n_conditions','n_bins'])
    # Population-wide unique support and per-split coverage are explicitly distinct.
    summary=summary.rename(columns={'n_conditions':'n_conditions_split_mean','n_bins':'n_bins_split_mean'})
    summary['n_conditions']=[int((data[r.animal]['common_mask'] & data[r.animal]['epoch_masks'][r.epoch]).any(1).sum()) for r in summary.itertuples()]
    summary['n_bins']=[int((data[r.animal]['common_mask'] & data[r.animal]['epoch_masks'][r.epoch]).sum()) for r in summary.itertuples()]
    maxmain=compare_numeric(summary,pd.read_csv(SOURCE/'results/self_reconstruction_main_table.csv'),
                            [k for k in ownkeys if k!='iteration'],owncols)
    cross.to_csv(output/'round_cross_2x2.csv',index=False,float_format='%.17g')
    own.to_csv(output/'round_self_metrics.csv',index=False,float_format='%.17g')
    cross_summary.to_csv(output/'cross_2x2_summary.csv',index=False,float_format='%.17g')
    summary.to_csv(output/'self_reconstruction_main_table.csv',index=False,float_format='%.17g')
    # Form every contrast within split, then aggregate; do not confuse own-target and fixed-target changes.
    contrast=[]
    for key,g in cross.groupby(ownkeys,sort=False):
        cells=g.set_index('cell')
        for name,left,right in (('own_trajectory','OO','BB'),('same_behavior_target','OB','BB'),
                                ('same_objective_target','OO','BO'),('fixed_objective_head_reference','OO','OB'),
                                ('fixed_behavior_head_reference','BO','BB')):
            contrast.append(dict(zip(ownkeys,key))|dict(comparison=name,left_cell=left,right_cell=right,
                Delta_r=cells.loc[right,'r']-cells.loc[left,'r'],
                Delta_RMSE=cells.loc[left,'RMSE']-cells.loc[right,'RMSE']))
    contrasts=pd.DataFrame(contrast)
    contrasts.to_csv(output/'round_contrasts.csv',index=False,float_format='%.17g')
    summarize(contrasts,[k for k in ownkeys if k!='iteration']+['comparison','left_cell','right_cell'],
              ['Delta_r','Delta_RMSE']).to_csv(output/'contrast_summary.csv',index=False,float_format='%.17g')
    copied=[]
    for name in ('condition_self_comparison.csv','condition_cross_summary.csv','binwise_reconstruction.csv',
                 'previous_trial_version_comparison.csv','condition_label_coverage.csv'):
        source=SOURCE/'results'/name;dest=output/name
        shutil.copy2(source,dest);copied.append(record(source,'byte-identical retained detailed current results'))
    sources+=copied
    for animal in ANIMALS:
        sources.append(record(SOURCE/'figures'/f'{animal}_all79_condition_atlas.pdf','retained all-79-condition atlas; no new curve averaging'))
    write_json(ROOT/'sources/A_sources.json',sources)
    audit=dict(completed=True,n_animals=2,n_representations=2,n_splits=100,n_prediction_files=400,
        all_predictions_test_only=True,no_refitting=True,x_prediction_max_difference=xerror,
        maximum_replay_errors=dict(round_cross=maxcross,round_self=maxown,cross_summary=maxsummary,main_summary=maxmain),
        original_support_unchanged=True,OO_BB_equals_self=True,raw_preprocessing='fail',
        unique_support='78 valid conditions and 3369 condition-time rows per animal; 79 split identities',
        interpretation='A is cross-evaluation, not a randomized null; positive Delta means the right cell improves the named metric.')
    assert xerror<1e-12
    write_json(output/'replay_audit.json',audit)
    print(audit,flush=True)
    return audit


if __name__=='__main__':
    run()
