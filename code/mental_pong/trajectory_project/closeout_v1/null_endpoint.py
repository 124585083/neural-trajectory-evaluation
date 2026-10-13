"""C: actual multioutput OLS refits of fixed, permuted endpoint candidates.

All source directories are read-only. Columns in the batched regression are
separately parameterized OLS targets. They share the design matrix and are
not statistically independent biological observations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from trajectory_project.step3_io import read_json, write_json, record, sha
from trajectory_project.condition_endpoint_fa_gpfa_v1.condition_data import load_condition_data
from trajectory_project.trial_endpoint_fa_gpfa_v1.trial_data import endpoint_candidates

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'condition_endpoint_fa_gpfa_v1'
TRIAL = ROOT.parent / 'trial_endpoint_fa_gpfa_v1'
EPOCHS = ('full', 'visible', 'hidden', 'bounce', 'no_bounce', 'post_bounce', 'endpoint_influence')
METRICS = ('r', 'RMSE', 'bias', 'target_sd', 'prediction_sd', 'amplitude_ratio',
           'target_min', 'target_max', 'SSE', 'baseline_SSE', 'skill', 'objective_distance_RMSE',
           'baseline_r', 'baseline_RMSE')
SEEDS = {'mahler': 271828, 'perle': 271829}
REPEATS = 1000
EXAMPLES = [0, 1, 2]
TOL = 1e-8


def geometry_basis(p):
    """The exact old endpoint_candidates expressed as offset + alpha * b."""
    valid = np.asarray(p['common_mask'], bool)
    offset = np.full(valid.shape, np.nan)
    alpha = np.full(valid.shape, np.nan)
    influence = np.zeros_like(valid)
    for g in p['geometry'].itertuples(index=False):
        c = int(g.condition_index)
        if not valid[c].any():
            continue
        assert bool(g.geometry_valid)
        kwargs = dict(x_start=g.x0, y_start=g.y0, x_end=g.x_end)
        if int(g.metadata_bounce_count):
            kwargs.update(collision_x=g.collision_x, collision_y=g.collision_y)
        candidates = endpoint_candidates(p['objective_xy'][c], [0., 1.], **kwargs)
        offset[c] = candidates[0, :, 1]
        alpha[c] = candidates[1, :, 1] - candidates[0, :, 1]
        influence[c] = (p['epoch_masks']['post_bounce'][c].copy() if int(g.metadata_bounce_count)
                        else valid[c] & (p['objective_xy'][c, :, 0] > g.x0))
    offset[~valid] = np.nan
    alpha[~valid] = np.nan
    epochs = dict(p['epoch_masks'], endpoint_influence=influence)
    return offset, alpha, epochs


def make_assignments(p, seed, repeats=REPEATS):
    valid_conditions = np.flatnonzero(p['common_mask'].any(1))
    assert np.isfinite(p['mean_endpoint'][valid_conditions]).all()
    rng = np.random.default_rng(seed)
    maps = np.stack([rng.permutation(valid_conditions) for _ in range(repeats)])
    endpoints = np.full((repeats, len(p['condition_ids'])), np.nan)
    endpoints[:, valid_conditions] = p['mean_endpoint'][maps]
    return valid_conditions, maps, endpoints


def prepare(root=ROOT):
    """Freeze mappings and audit geometry before any random neural score."""
    root = Path(root)
    for d in ('configs', 'results/C', 'artifacts/C', 'logs/C'):
        (root / d).mkdir(parents=True, exist_ok=True)
    source_protocol = read_json(root / 'configs/closeout_protocol.json')
    assert source_protocol['C']['repeats'] == REPEATS and source_protocol['C']['seeds'] == SEEDS
    config = dict(version='C_endpoint_refit_v1', B=REPEATS, seeds=SEEDS,
        rng='numpy.random.default_rng PCG64', examples=EXAMPLES, epochs=EPOCHS, metrics=METRICS,
        OLS=dict(class_name='sklearn.linear_model.LinearRegression', fit_intercept=True,
                 positive=False, dtype='float64', sample_weight=None, regularization=None),
        batch='1000 separate y outputs plus one identical shared x output; actual sklearn fit per split/representation',
        eligibility='original 78 valid conditions only; all 79 retain split identities',
        geometry='exact prior endpoint_candidates; original offset + alpha*endpoint; no clipping or rejections',
        feasibility='all 78 x 78 assignment pairs audited before neural outcomes; alpha convexity tolerance 1e-10 only',
        baseline='unweighted mean endpoint of valid training conditions, with each test condition own geometry',
        objective_baseline='same candidate geometry with training-condition mean objective_end_y; original objective labels remain scoring target',
        endpoint_influence='no bounce: common bins with x>x0 (alpha>0); bounce: unchanged original post_bounce mask',
        correlation='Pearson; fewer than 3 rows or constant target/prediction -> NA',
        aggregation='score each split first; q statistic = nanmean of its 100 split scores; std ddof=0',
        tail='empirical proportion of q-mean random scores at least as good as observed, no exact p claim',
        numerical_equivalence_tolerance=TOL, raw_preprocessing='fail',
        provided_input_filtering='pass only relative to released inputs', fit_provenance='pass only relative to released inputs',
        source_protocol=record(root / 'configs/closeout_protocol.json'),
        source_splits=record(SOURCE / 'configs/condition_splits_100.json'))
    write_json(root / 'configs/c_endpoint_protocol.json', config, immutable=True)
    data = load_condition_data(SOURCE)
    audits = []
    for animal, p in data.items():
        offset, alpha, epochs = geometry_basis(p)
        ix, maps, endpoints = make_assignments(p, SEEDS[animal])
        mask = p['common_mask']
        reconstructed = offset + alpha * p['mean_endpoint'][:, None]
        label_error = float(np.max(np.abs(reconstructed[mask] - p['behavior_xy'][..., 1][mask])))
        assert label_error < TOL
        assert len(ix) == 78 and int(mask.sum()) == 3369
        assert np.nanmin(alpha) >= -1e-10 and np.nanmax(alpha) <= 1 + 1e-10
        assert np.array_equal(epochs['endpoint_influence'] & epochs['bounce'], epochs['post_bounce'])
        assert np.array_equal(p['behavior_xy'][..., 0][mask], p['objective_xy'][..., 0][mask])
        all_pair_min, all_pair_max = np.inf, -np.inf
        for c in ix:
            labels = offset[c, mask[c], None] + alpha[c, mask[c], None] * p['mean_endpoint'][ix][None, :]
            assert np.isfinite(labels).all()
            all_pair_min = min(all_pair_min, float(labels.min()))
            all_pair_max = max(all_pair_max, float(labels.max()))
        # Every endpoint belongs to the same already accepted endpoint coordinate
        # domain. Convex segments cannot add a wall crossing between their ends.
        original_range = [float(np.nanmin(p['behavior_xy'][..., 1])), float(np.nanmax(p['behavior_xy'][..., 1]))]
        geometric_endpoints=p['geometry'].sort_values('condition_index')['objective_end_y'].to_numpy(float)
        physical_candidate=offset+alpha*geometric_endpoints[:,None]
        physical_candidate_error=float(np.max(np.abs(physical_candidate[mask]-p['objective_xy'][...,1][mask])))
        physical_candidate_rmse=float(np.sqrt(np.mean((physical_candidate[mask]-p['objective_xy'][...,1][mask])**2)))
        path = root / 'artifacts/C' / f'{animal}_endpoint_assignments.npz'
        if path.exists():
            with np.load(path) as z:
                assert np.array_equal(z['source_condition_indices'], maps)
                assert np.array_equal(z['endpoints'], endpoints, equal_nan=True)
        else:
            np.savez_compressed(path, endpoints=endpoints, source_condition_indices=maps,
                target_condition_indices=ix, condition_ids=p['condition_ids'],
                offset=offset, alpha=alpha, common_mask=mask,
                epoch_names=np.asarray(EPOCHS), epoch_masks=np.stack([epochs[e] for e in EPOCHS]),
                source_mean_endpoints=p['mean_endpoint'], random_seed=SEEDS[animal])
        changed = np.mean(maps != ix[None, :], axis=1)
        changed_values = np.mean(endpoints[:, ix] != p['mean_endpoint'][ix][None, :], axis=1)
        audits.append(dict(animal=animal, status='pass', n_valid_conditions=len(ix), n_valid_rows=int(mask.sum()),
            all_assignment_pairs_audited=len(ix)**2, own_label_replay_max_abs_error=label_error,
            alpha_min=float(np.nanmin(alpha)), alpha_max=float(np.nanmax(alpha)),
            random_path_all_assignments_range=[all_pair_min, all_pair_max], original_path_range=original_range,
            endpoint_range=[float(np.nanmin(endpoints)), float(np.nanmax(endpoints))],
            endpoint_SD=float(np.std(p['mean_endpoint'][ix])), original_support_preserved=True,
            objective_endpoint_SD=float(np.std(geometric_endpoints[ix])),
            objective_endpoint_range=[float(geometric_endpoints[ix].min()),float(geometric_endpoints[ix].max())],
            own_objective_endpoint_through_candidate_F_vs_original_objective_max_abs_error=physical_candidate_error,
            own_objective_endpoint_through_candidate_F_vs_original_objective_RMSE=physical_candidate_rmse,
            original_objective_not_replaced_by_F=True,
            geometric_conclusion='finite convex segments between original anchors and previously accepted endpoints; no compatibility restriction, clipping, rejection, or extra reflection',
            changed_identity_fraction_mean=float(changed.mean()), changed_value_fraction_mean=float(changed_values.mean()),
            unique_endpoint_mappings=int(len(np.unique(maps, axis=0))),
            epoch_n_bins={e:int(epochs[e].sum()) for e in EPOCHS},
            assignments=record(path), labels=record(p['labels_path']), raw_preprocessing='fail'))
        pd.DataFrame({'q':np.arange(REPEATS), 'changed_identity_fraction':changed,
                      'changed_value_fraction':changed_values}).to_csv(root/'results/C'/f'{animal}_mapping_summary.csv', index=False)
    write_json(root/'results/C/geometry_feasibility.json', dict(before_random_neural_scoring=True, status='pass', animals=audits), immutable=True)
    write_json(root/'configs/c_endpoint_lock.json', {'protocol':record(root/'configs/c_endpoint_protocol.json'),
        'geometry_audit':record(root/'results/C/geometry_feasibility.json'), 'implementation':record(Path(__file__))}, immutable=True)
    return data


def fit_batch(X, targets):
    with threadpool_limits(limits=2):
        return LinearRegression(fit_intercept=True, positive=False).fit(
            np.asarray(X, dtype=np.float64), np.asarray(targets, dtype=np.float64))


def score_batch(target, prediction, baseline, objective):
    """Rows x separately parameterized outputs -> outputs x named metrics."""
    target, prediction, baseline = map(lambda a: np.asarray(a, np.float64), (target, prediction, baseline))
    if target.ndim == 1:
        target, prediction, baseline = target[:,None], prediction[:,None], baseline[:,None]
    n, k = target.shape
    out = np.full((k,len(METRICS)), np.nan)
    if not n:
        return out
    assert target.shape == prediction.shape == baseline.shape
    assert np.isfinite(target).all() and np.isfinite(prediction).all() and np.isfinite(baseline).all()
    yc = target - target.mean(0)
    pc = prediction - prediction.mean(0)
    bc = baseline - baseline.mean(0)
    yss, pss, bss = (np.sum(a*a, axis=0) for a in (yc,pc,bc))
    error = prediction-target
    sse = np.sum(error*error, axis=0)
    base_sse = np.sum((baseline-target)**2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.sum(yc*pc,axis=0)/np.sqrt(yss*pss)
        br = np.sum(yc*bc,axis=0)/np.sqrt(yss*bss)
        ratio = np.sqrt(pss/yss)
        skill = 1-sse/base_sse
    invalid_y = (np.ptp(target,axis=0)==0)|(yss==0)
    r[invalid_y|(np.ptp(prediction,axis=0)==0)|(pss==0)|(n<3)] = np.nan
    br[invalid_y|(np.ptp(baseline,axis=0)==0)|(bss==0)|(n<3)] = np.nan
    ratio[invalid_y]=np.nan
    skill[base_sse==0]=np.nan
    objective=np.asarray(objective).reshape(n,1)
    return np.column_stack([r,np.sqrt(sse/n),error.mean(0),np.sqrt(yss/n),np.sqrt(pss/n),ratio,
        target.min(0),target.max(0),sse,base_sse,skill,np.sqrt(np.mean((target-objective)**2,axis=0)),br,np.sqrt(base_sse/n)])


def _load_source(p, iteration, representation):
    folder = TRIAL/'representations'/p['animal']/f'round_{iteration:03d}'
    with np.load(folder/'latents.npz') as z:
        latent=z[representation][0].astype(np.float64)
        assert np.array_equal(z['condition_ids'],p['condition_ids'])
        assert np.array_equal(z['times_ms'],p['times_ms'])
        assert np.all(z['common_mask'][p['common_mask']])
    return latent, folder


def _indices(mask, split):
    train=np.zeros(mask.shape[0],bool);train[split['train_indices']]=True
    test=np.zeros_like(train);test[split['test_indices']]=True
    assert train.sum()==39 and test.sum()==40 and not (train&test).any()
    return np.nonzero(mask&train[:,None]),np.nonzero(mask&test[:,None])


def validate_real_OLS(p, assignments, offset, alpha, split, representation):
    """Numerically compare batched fits with separately fitted sklearn OLS."""
    z,_=_load_source(p,0,representation)
    train,test=_indices(p['common_mask'],split)
    tc,tb=train;vc,vb=test
    random_y=offset[tc,tb,None]+alpha[tc,tb,None]*assignments[EXAMPLES][:,tc].T
    targets=np.column_stack([p['objective_xy'][tc,tb,0],p['behavior_xy'][tc,tb,1],random_y])
    fitted=fit_batch(z[train],targets)
    errors=[]
    for j in range(targets.shape[1]):
        single=fit_batch(z[train],targets[:,j])
        errors.append(float(np.max(np.abs(single.predict(z[test])-fitted.predict(z[test])[:,j]))))
        assert np.max(np.abs(single.coef_-fitted.coef_[j])) < TOL
        assert abs(single.intercept_-fitted.intercept_[j]) < TOL
    with np.load(SOURCE/'readouts/models'/f'{p["animal"]}_{representation}_r000_ols.npz') as old:
        saved_prediction=z[test]@old['coefficients'][1].T+old['intercepts'][1]
    current=fitted.predict(z[test])[:,:2]
    old_error=float(np.max(np.abs(saved_prediction-current)))
    assert max(errors+[old_error])<TOL
    identical=fit_batch(z[train],np.column_stack([targets[:,1],targets[:,1]]))
    identical_error=float(np.max(np.abs(identical.coef_[0]-identical.coef_[1])))
    assert identical_error<TOL
    return dict(animal=p['animal'],representation=representation,status='pass',
                independent_vs_multioutput_max_prediction_error=max(errors), old_real_OLS_replay_error=old_error,
                checked_random_q=EXAMPLES, identical_label_coefficients_max_abs_error=identical_error)


def execute(animal, representation, root=ROOT):
    root=Path(root)
    started=time.perf_counter()
    p=load_condition_data(SOURCE)[animal]
    config=read_json(root/'configs/c_endpoint_lock.json')
    assert config['implementation']['sha256']==sha(Path(__file__))
    assert config['protocol']['sha256']==sha(root/'configs/c_endpoint_protocol.json')
    assert config['geometry_audit']['sha256']==sha(root/'results/C/geometry_feasibility.json')
    splits=read_json(root/'configs/condition_splits_100.json')
    with np.load(root/'artifacts/C'/f'{animal}_endpoint_assignments.npz') as a:
        endpoint=a['endpoints'].copy();offset=a['offset'].copy();alpha=a['alpha'].copy()
        masks=a['epoch_masks'].copy()
    equivalence=validate_real_OLS(p,endpoint,offset,alpha,splits[0],representation)
    write_json(root/'results/C'/f'{animal}_{representation}_OLS_equivalence.json',equivalence)
    scores=np.full((REPEATS,100,len(EPOCHS),len(METRICS)),np.nan)
    observed=np.full((2,100,len(EPOCHS),len(METRICS)),np.nan)
    coefficients=np.empty((100,REPEATS,50));intercepts=np.empty((100,REPEATS))
    x_coefficients=np.empty((100,50));x_intercepts=np.empty(100)
    examples=np.full((len(EXAMPLES),100,*p['common_mask'].shape),np.nan)
    support=np.zeros((100,*p['common_mask'].shape),bool)
    counts=np.zeros((100,len(EPOCHS),2),int)
    mean_train_endpoints=np.empty((100,REPEATS))
    observed_mean_train_endpoints=np.empty((2,100))
    design_rank=[];source_records=[];x_errors=[]
    geometry=p['geometry'].sort_values('condition_index')
    objective_endpoint=geometry['objective_end_y'].to_numpy(float)
    labelpath=root/'artifacts/C'/f'{animal}_endpoint_assignments.npz'
    for iteration, split in enumerate(splits):
        z,folder=_load_source(p,iteration,representation)
        source_records.append(record(folder/'training.json'))
        train,test=_indices(p['common_mask'],split)
        tc,tb=train;vc,vb=test
        train_y=offset[tc,tb,None]+alpha[tc,tb,None]*endpoint[:,tc].T
        model=fit_batch(z[train],np.column_stack([p['objective_xy'][tc,tb,0],train_y]))
        with threadpool_limits(limits=2):
            prediction=model.predict(z[test])
        coefficients[iteration]=model.coef_[1:];intercepts[iteration]=model.intercept_[1:]
        x_coefficients[iteration]=model.coef_[0];x_intercepts[iteration]=model.intercept_[0]
        design_rank.append(int(model.rank_))
        targets=offset[vc,vb,None]+alpha[vc,vb,None]*endpoint[:,vc].T
        validtrain=np.unique(tc)
        means=endpoint[:,validtrain].mean(1)
        mean_train_endpoints[iteration]=means
        baseline=offset[vc,vb,None]+alpha[vc,vb,None]*means[None,:]
        with np.load(SOURCE/'readouts/predictions'/f'{animal}_{representation}_r{iteration:03d}_test_predictions.npz') as old:
            old_prediction=old['predictions'].copy()
            assert np.array_equal(old['test_input_support'][test],np.ones(len(vc),bool))
            assert not np.isfinite(old_prediction[:,tc,tb,:]).any()
        xerror=float(np.max(np.abs(prediction[:,0]-old_prediction[1,vc,vb,0])))
        assert xerror<TOL
        x_errors.append(xerror)
        support[iteration][test]=True
        examples[:,iteration,vc,vb]=prediction[:,1+np.asarray(EXAMPLES)].T
        actual_endpoints=(objective_endpoint,p['mean_endpoint'])
        for h in range(2):
            observed_mean_train_endpoints[h,iteration]=actual_endpoints[h][validtrain].mean()
        for e,epoch in enumerate(EPOCHS):
            chosen=masks[e,vc,vb]
            counts[iteration,e]=[len(np.unique(vc[chosen])),int(chosen.sum())]
            obj=p['objective_xy'][vc[chosen],vb[chosen],1]
            scores[:,iteration,e]=score_batch(targets[chosen],prediction[chosen,1:],baseline[chosen],obj)
            for h,label in enumerate((p['objective_xy'],p['behavior_xy'])):
                mu=observed_mean_train_endpoints[h,iteration]
                base=offset[vc[chosen],vb[chosen]]+alpha[vc[chosen],vb[chosen]]*mu
                observed[h,iteration,e]=score_batch(label[vc[chosen],vb[chosen],1],
                    old_prediction[h,vc[chosen],vb[chosen],1],base,obj)[0]
        if iteration%10==0 or iteration==99:
            print(f'C {animal} {representation} completed {iteration+1}/100 x {REPEATS} random targets; {time.perf_counter()-started:.1f}s',flush=True)
    stem=f'{animal}_{representation}'
    np.savez_compressed(root/'artifacts/C'/f'{stem}_OLS_weights.npz',coefficients=coefficients,intercepts=intercepts,
        x_coefficients=x_coefficients,x_intercepts=x_intercepts,coefficient_axes=['split','q','latent_dimension'],
        condition_ids=p['condition_ids'],source_assignments_sha256=sha(labelpath),design_rank=design_rank,
        mean_train_endpoints=mean_train_endpoints,observed_mean_train_endpoints=observed_mean_train_endpoints)
    np.savez_compressed(root/'artifacts/C'/f'{stem}_fixed_examples.npz',predictions=examples,example_q=EXAMPLES,
        test_support=support,axes=['example_q','split','condition','time'],condition_ids=p['condition_ids'],times_ms=p['times_ms'])
    np.savez_compressed(root/'results/C'/f'{stem}_all_scores.npz',random_scores=scores,observed_scores=observed,
        random_axes=['q','split','epoch','metric'],observed_axes=['head','split','epoch','metric'],
        head_names=['D_obj_to_objective','D_beh_to_behavior'],epochs=EPOCHS,metrics=METRICS,
        counts=counts,count_axes=['split','epoch','quantity'],count_names=['n_conditions','n_bins'])
    rows=[];qs=[]
    qmeans=np.nanmean(scores,axis=1)
    for e,epoch in enumerate(EPOCHS):
        for q in range(REPEATS):
            qs.append(dict(animal=animal,representation=representation,q=q,epoch=epoch,
                           **{m:qmeans[q,e,j] for j,m in enumerate(METRICS)}))
        for j,metric in enumerate(METRICS):
            v=qmeans[:,e,j];finite=np.isfinite(v)
            row=dict(animal=animal,representation=representation,epoch=epoch,metric=metric,
                null_mean=np.nanmean(v),null_sd=np.nanstd(v),null_q025=np.nanquantile(v,.025) if finite.any() else np.nan,
                null_q975=np.nanquantile(v,.975) if finite.any() else np.nan,n_random_statistics=int(finite.sum()),
                n_splits=100,n_conditions_mean=counts[:,e,0].mean(),n_bins_mean=counts[:,e,1].mean(),
                raw_preprocessing='fail')
            for h,name in enumerate(('objective','behavior')):
                real=np.nanmean(observed[h,:,e,j]); row[f'{name}_mean']=real
                row[f'{name}_split_sd']=np.nanstd(observed[h,:,e,j])
                row[f'{name}_minus_null']=real-np.nanmean(v)
                if metric in ('r','skill'):
                    tail=np.mean(v[finite]>=real) if finite.any() and np.isfinite(real) else np.nan
                elif metric=='RMSE':
                    tail=np.mean(v[finite]<=real) if finite.any() and np.isfinite(real) else np.nan
                else:tail=np.nan
                row[f'{name}_empirical_random_at_least_as_good_fraction']=tail
            rows.append(row)
    pd.DataFrame(rows).to_csv(root/'results/C'/f'{stem}_summary.csv',index=False,float_format='%.17g')
    pd.DataFrame(qs).to_csv(root/'results/C'/f'{stem}_per_q_mean100.csv',index=False,float_format='%.17g')
    paths=[root/'artifacts/C'/f'{stem}_OLS_weights.npz',root/'artifacts/C'/f'{stem}_fixed_examples.npz',
           root/'results/C'/f'{stem}_all_scores.npz',root/'results/C'/f'{stem}_summary.csv',
           root/'results/C'/f'{stem}_per_q_mean100.csv']
    completion=dict(animal=animal,representation=representation,completed=True,random_repeats=REPEATS,
        condition_splits=100,OLS_calls=100,independent_random_y_refits=REPEATS*100,random_xy_heads=REPEATS*100,
        batch_target_count=REPEATS+1,shared_x_used_for_every_q=True,OLS_equivalence=equivalence,
        max_identical_x_prediction_error=max(x_errors),actual_test_support_count=support.sum(axis=(1,2)).tolist(),
        endpoint_assignments=record(labelpath),source_training_records=source_records,
        outputs=[record(f) for f in paths],elapsed_seconds=time.perf_counter()-started,
        raw_preprocessing='fail',new_representation_fit=False,new_inference=False)
    write_json(root/'results/C'/f'{stem}_completed.json',completion)
    return completion


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true',help='Run all four animal/representation branches, then summarize')
    parser.add_argument('--replay',action='store_true',help='Replay saved random weights for specified q/split')
    parser.add_argument('--q',type=int,default=0);parser.add_argument('--split',type=int,default=0)
    parser.add_argument('--animal',choices=tuple(SEEDS));parser.add_argument('--representation',choices=('FA50','GPFA50'))
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.replay:
        assert args.animal and args.representation
        print(json.dumps(replay(args.animal,args.representation,args.q,args.split),indent=2),flush=True)
    elif args.run:
        for animal in SEEDS:
            for representation in ('FA50','GPFA50'):
                marker=ROOT/'results/C'/f'{animal}_{representation}_completed.json'
                if marker.exists():
                    from trajectory_project.step3_io import verify_records
                    completed=read_json(marker)
                    verify_records(completed['outputs']+completed['source_training_records'])
                else:execute(animal,representation)
        finalize()
    elif args.animal and args.representation:execute(args.animal,args.representation)


def replay(animal, representation, q=0, iteration=0, root=ROOT):
    root=Path(root)
    assert 0<=q<REPEATS and 0<=iteration<100
    p=load_condition_data(SOURCE)[animal]
    split=read_json(root/'configs/condition_splits_100.json')[iteration]
    _,test=_indices(p['common_mask'],split);vc,vb=test
    z,_=_load_source(p,iteration,representation)
    stem=f'{animal}_{representation}'
    with np.load(root/'artifacts/C'/f'{animal}_endpoint_assignments.npz') as a:
        offset=a['offset'][test];alpha=a['alpha'][test]
        target=offset+alpha*a['endpoints'][q,vc]
        masks=a['epoch_masks'][:,vc,vb]
    with np.load(root/'artifacts/C'/f'{stem}_OLS_weights.npz') as w:
        predicted=z[test]@w['coefficients'][iteration,q]+w['intercepts'][iteration,q]
        base=offset+alpha*w['mean_train_endpoints'][iteration,q]
    replayed=np.stack([score_batch(target[m],predicted[m],base[m],p['objective_xy'][vc[m],vb[m],1])[0] for m in masks])
    with np.load(root/'results/C'/f'{stem}_all_scores.npz') as r:
        saved=r['random_scores'][q,iteration]
    assert np.array_equal(np.isnan(replayed),np.isnan(saved))
    error=float(np.nanmax(np.abs(replayed-saved)))
    assert np.allclose(replayed,saved,atol=TOL,rtol=1e-12,equal_nan=True)
    return dict(animal=animal,representation=representation,q=q,split=iteration,status='pass',
                saved_score_replay_max_abs_error=error,heldout_rows=len(vc),raw_preprocessing='fail')


def finalize(root=ROOT):
    root=Path(root)
    from trajectory_project.step3_io import verify_records
    completed=[];summaries=[];checks=[]
    for animal in SEEDS:
        for representation in ('FA50','GPFA50'):
            stem=f'{animal}_{representation}'
            marker=root/'results/C'/f'{stem}_completed.json'
            m=read_json(marker)
            assert m['completed'] and m['random_repeats']==1000 and m['condition_splits']==100
            verify_records(m['outputs']+m['source_training_records'])
            completed.append(record(marker))
            summaries.append(pd.read_csv(root/'results/C'/f'{stem}_summary.csv'))
            for q,iteration in ((0,0),(1,49),(2,99),(999,99)):
                checks.append(replay(animal,representation,q,iteration,root))
    combined=root/'results/C/random_endpoint_summary.csv'
    pd.concat(summaries,ignore_index=True).to_csv(combined,index=False,float_format='%.17g')
    write_json(root/'results/C/completed.json',dict(completed=True,branches=completed,
        repeats_per_animal=1000,splits_per_repeat=100,independent_random_xy_heads=400000,
        actual_batched_sklearn_fits=400,replay_checks=checks,summary=record(combined),
        raw_preprocessing='fail',provided_input_filtering='pass relative to released inputs only',
        fit_provenance='pass relative to released inputs only'))


if __name__=='__main__': main()
