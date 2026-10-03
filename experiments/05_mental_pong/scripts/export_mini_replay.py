"""Export the fixed split-0/q-0 replay from verified original saved artifacts.

This command reads the local source only. It never fits a model, changes a
condition split, regenerates a random assignment, or publishes a remote asset.
The default export is private until the artifact-specific rights audit passes.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

MODULE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mini_replay_core', MODULE/'mini_replay/replay.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
CURRENT = 'trajectory_project/condition_endpoint_fa_gpfa_v1'
TRIAL = 'trajectory_project/trial_endpoint_fa_gpfa_v1'
CLOSEOUT = 'trajectory_project/closeout_v1'


def export(source, output, rights_record=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError('The export must be outside the preserved scientific source.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new, empty export directory; existing exports are immutable.')
    if output.is_relative_to(MODULE) and rights_record is None:
        raise ValueError('Publication-tree payload export requires an explicit passed rights record.')
    rights = {'status':'unresolved','public_payload_permitted':False}
    if rights_record:
        rights = json.loads(Path(rights_record).read_text(encoding='utf-8'))
        if rights.get('status') != 'passed' or rights.get('public_payload_permitted') is not True:
            raise ValueError('The supplied rights audit does not authorize a public payload.')
    with (MODULE/'integration/artifact_registry.csv').open(encoding='utf-8-sig',newline='') as stream:
        registry = {r['relative_path'].replace('\\','/'):r['sha256'] for r in csv.DictReader(stream) if r['storage_location_id']=='source_pilot'}
    parents, arrays, array_schema, cases, expected = {}, {}, {}, [], {}

    def checked(relative):
        path=source/relative
        actual=core.sha(path)
        if relative not in registry or registry[relative] != actual:
            raise ValueError(f'Unregistered or changed original artifact: {relative}')
        parents[relative]={'source_artifact_id':'source_pilot://'+relative,'sha256':actual,'bytes':path.stat().st_size}
        return path

    def load(relative):
        with np.load(checked(relative),allow_pickle=False) as z:
            return {k:z[k].copy() for k in z.files}

    def add(name,value,axes,origin,transform='exact saved array or indexing; no rounding or normalization'):
        value=np.asarray(value)
        if value.dtype.kind not in 'bifu':raise ValueError('Only numeric arrays may enter the replay.')
        arrays[name]=value
        array_schema[name]={'shape':list(value.shape),'dtype':value.dtype.str,'axes':axes,'sha256':core.array_sha(value),'source':origin,'transform':transform}

    splits_path=f'{CURRENT}/configs/condition_splits_100.json'
    split=json.loads(checked(splits_path).read_text(encoding='utf-8'))[0]
    train,test=np.asarray(split['train_indices']),np.asarray(split['test_indices'])
    for animal in ('mahler','perle'):
        labels_path=f'{CURRENT}/artifacts/{animal}_condition_labels.npz'
        latent_path=f'{TRIAL}/representations/{animal}/round_000/latents.npz'
        assign_path=f'{CLOSEOUT}/artifacts/C/{animal}_endpoint_assignments.npz'
        mapping_path=f'{CLOSEOUT}/artifacts/B/{animal}_mappings.npz'
        coverage_path=f'{CLOSEOUT}/artifacts/B/{animal}_coverage.npz'
        labels,latent,assignment=load(labels_path),load(latent_path),load(assign_path)
        mapping=load(mapping_path)['mapping'][0,0]
        coverage=load(coverage_path)
        common=labels['common_mask']; c,t=np.nonzero(common)
        tr,te=np.isin(c,train),np.isin(c,test)
        np.testing.assert_array_equal(labels['condition_ids'],latent['condition_ids'])
        np.testing.assert_array_equal(labels['times_ms'],latent['times_ms'])
        if not np.all(latent['common_mask'][common]):raise ValueError('Current rows exceed saved representation validity.')
        if assignment['epoch_names'].tolist()!=list(core.EPOCHS):raise ValueError('Epoch order changed.')
        names={'condition_ids':(['condition'],labels['condition_ids'],labels_path),
               'times_ms':(['bin'],labels['times_ms'],labels_path),
               'common_mask':(['condition','bin'],common,labels_path),
               'condition_index':(['row'],c,labels_path),
               'bin_index':(['row'],t,labels_path),
               'epoch_masks':(['epoch','condition','bin'],assignment['epoch_masks'],assign_path),
               'offset':(['condition','bin'],assignment['offset'],assign_path),
               'alpha':(['condition','bin'],assignment['alpha'],assign_path),
               'random_endpoints':(['condition'],assignment['endpoints'][0],assign_path),
               'random_source_indices':(['valid_condition'],assignment['source_condition_indices'][0],assign_path),
               'random_target_indices':(['valid_condition'],assignment['target_condition_indices'],assign_path),
               'mean_endpoints':(['condition'],labels['mean_endpoint'],labels_path),
               'mismatch_mapping':(['condition'],mapping,mapping_path)}
        for key,(axes,value,origin) in names.items():add(animal+'__'+key,value,axes,origin)
        target=np.stack([labels['objective_xy'][c,t],labels['behavior_xy'][c,t],labels['objective_xy'][c,t].copy()])
        target[2,:,1]=assignment['offset'][c,t]+assignment['alpha'][c,t]*assignment['endpoints'][0,c]
        add(animal+'__targets',target,['head','row','coordinate'],[labels_path,assign_path],'objective and behavior exact saved values; random y=offset+alpha*saved endpoint for q=0')
        validtest=np.flatnonzero(common.any(1)&np.isin(np.arange(79),test))
        for ei in range(6):
            mask=assignment['epoch_masks'][ei]&common
            pairs=np.asarray([(int(ci),int(mapping[ci]),int(ti)) for ci in validtest for ti in np.flatnonzero(mask[ci]&mask[mapping[ci]])],dtype=np.int64).reshape(-1,3)
            add(animal+f'__mismatch_pairs_{ei}',pairs,['pair','source_condition_target_condition_bin'],[assign_path,mapping_path],'same-bin intersection of both original common and epoch masks; fixed saved mapping')
        for rep in ('FA50','GPFA50'):
            stem=animal+'_'+rep
            weights_path=f'{CURRENT}/readouts/models/{stem}_r000_ols.npz'
            random_path=f'{CLOSEOUT}/artifacts/C/{stem}_OLS_weights.npz'
            c_scores_path=f'{CLOSEOUT}/results/C/{stem}_all_scores.npz'
            b_scores_path=f'{CLOSEOUT}/artifacts/B/{stem}_scores.npz'
            weights,rweights=load(weights_path),load(random_path)
            cs,bs=load(c_scores_path),load(b_scores_path)
            np.testing.assert_array_equal(weights['training_condition_bin_indices'],np.column_stack((c[tr],t[tr])))
            np.testing.assert_array_equal(weights['train_condition_indices'],train)
            np.testing.assert_array_equal(weights['test_condition_indices'],test)
            coef=np.concatenate([weights['coefficients'],np.stack([rweights['x_coefficients'][0],rweights['coefficients'][0,0]])[None]])
            bias=np.concatenate([weights['intercepts'],np.asarray([rweights['x_intercepts'][0],rweights['intercepts'][0,0]])[None]])
            add(stem+'__features',latent[rep][0,c,t],['row','latent_dimension'],latent_path,'half1 and exact original common-row indexing')
            add(stem+'__coefficients',coef,['head','coordinate','latent_dimension'],[weights_path,random_path])
            add(stem+'__intercepts',bias,['head','coordinate'],[weights_path,random_path])
            add(stem+'__observed_baseline_means',rweights['observed_mean_train_endpoints'][:,0],['objective_behavior'],random_path)
            cm=cs['metrics'].tolist(); bm=bs['metric_names'].tolist()
            expected[stem]={
              'matrix':bs['observed_full_support'][0,...,:2],
              'random':[{name:row[cm.index(name)] for name in ('r','RMSE','baseline_SSE','skill')} for row in cs['random_scores'][0,0]],
              'counts':cs['counts'][0],
              'mismatch_counts':np.column_stack([coverage['n_conditions'][0,0],coverage['n_bins'][0,0]]),
              'mismatch':[[[[{name:bs['scores'][0,0,e,h,k,xy,bm.index(name)] for name in ('matched_r','null_r','matched_RMSE','null_RMSE')} for xy in range(2)] for k in range(2)] for h in range(2)] for e in range(6)],
              'source_scores':{'matrix':b_scores_path+'#observed_full_support[0]','random':c_scores_path+'#random_scores[0,0]','mismatch':b_scores_path+'#scores[0,0]'}}
            cases.append({'case_id':stem,'animal':animal,'representation':rep,'half':'half1','split':0,'q':0,'n_train_rows':int(tr.sum()),'n_test_rows':int(te.sum()),'valid_train_conditions':len(np.unique(c[tr])),'valid_test_conditions':len(np.unique(c[te])),'representation_source_sha256':parents[latent_path]['sha256'],'random_train_mean_endpoint':float(rweights['mean_train_endpoints'][0,0]),'recorded_rank':int(weights['design_rank']),'original_versions':{'numpy':'1.26.4','scipy':'1.14.1','sklearn':'1.5.2'}})
    bundle=output/'bundle';bundle.mkdir(parents=True)
    np.savez_compressed(bundle/'arrays.npz',**arrays)
    core.write_json(bundle/'condition_splits.json',split)
    core.write_json(bundle/'cases.json',cases)
    core.write_json(bundle/'expected_scores.json',expected)
    core.write_json(bundle/'source_manifest.json',{'parents':list(parents.values()),'exporter_sha256':core.sha(__file__),'selection':{'split':0,'q':0,'half':'half1'},'new_fitting':False,'rights':rights,'source_values_rounded':False,'private_metadata_omitted':['machine paths','embedded provenance strings','raw behavioral trial records','executable model pickles']})
    (bundle/'LICENSES.md').write_text('# Replay data and software rights\n\n'+rights.get('public_statement','The replay payload is private pending an artifact-specific redistribution audit. The project MIT license does not establish permission for upstream data.')+'\n\nProject-owned replay code is covered by the repository MIT license. See the repository THIRD_PARTY_NOTICES.md for upstream credits.\n',encoding='utf-8')
    files={str(p.relative_to(output)).replace('\\','/'):{'sha256':core.sha(p),'bytes':p.stat().st_size} for p in sorted(bundle.iterdir())}
    manifest={'schema':core.SCHEMA,'selection':{'split':0,'q':0,'half':'half1'},'payload_status':'available' if rights_record else 'private_validation_only','public_payload_permitted':bool(rights_record),'files':files,'arrays':array_schema,'coefficient_convention':'prediction[head,row,xy] = features[row,dimension] @ coefficients[head,xy,dimension].T + intercepts[head,xy]','head_order':['objective','behavior','random'],'coordinate_order':['x','y'],'epoch_order':list(core.EPOCHS),'tolerance':{'atol':core.ATOL,'rtol':core.RTOL},'raw_preprocessing':'fail','scientific_status':'CLOSED_EXPLORATORY_WITH_LIMITATIONS'}
    core.write_json(output/'manifest.json',manifest)
    # Read back every array and verify exact exported values and parent bytes.
    with np.load(bundle/'arrays.npz',allow_pickle=False) as saved:
        for name,value in arrays.items():np.testing.assert_array_equal(saved[name],value)
    for relative,row in parents.items():
        if core.sha(source/relative)!=row['sha256']:raise ValueError('Source changed during export: '+relative)
    return {'status':'exported','cases':len(cases),'payload_bytes':sum(x['bytes'] for x in files.values()),'parent_artifacts_verified':len(parents),'public_payload_permitted':bool(rights_record),'scientific_sources_changed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pilot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rights-record',type=Path,help='Explicit passed artifact-specific redistribution audit.')
    args=parser.parse_args()
    print(json.dumps(export(args.source_pilot,args.output,args.rights_record),indent=2))


if __name__=='__main__':
    main()
