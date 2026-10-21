#!/usr/bin/env python3
"""Frozen B3 traffic / coverage permutation negative controls.

Run only on TASK internal_test, where same task has multiple dates. Reuse
Step6-B2-R's exactly documented same-task, different-date, state-wise
permutation. Models, alpha and threshold are never retrained or retuned.

WARNING: This is a diagnostic *feature swap*, NOT a realistic new traffic
scenario or a causal intervention. It may break route/traffic consistency.
"""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import joblib

ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6b3','road_vrp7_step6b1_v2',
               'road_vrp7_step6b2','road_vrp7_step6b2_r'):
    p=str(ROOT/folder)
    if p not in sys.path:sys.path.insert(0,p)
from model import FEATURES,STRUCTURE,TRAFFIC,RELIABILITY,from_full # needed for joblib load
from features import FEATURE_MODES
from train_b2 import read_extra,get_features
from dataset import load,masks,select_rows,PRIMARY,ORACLE,sha256
from train import choose,summarize
from frozen_permutation import shuffle_same_task_date

SOURCE='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1'
BOUNDARY='runs/chicago_july_v1/step6b2/features_boundary_v1'
METHODS=('M1','M2','M3','M4')
SHUFFLES=('traffic_and_mask','traffic_only','mask_only')


def readjson(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def run(source,features,root,output,seeds,methods,perms,shuffles=SHUFFLES):
    source,features,root,output=map(Path,(source,features,root,output))
    if output.exists():raise FileExistsError(f'Use a fresh standalone directory: {output}')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Empty/repeated model seeds')
    if not perms or len(set(perms))!=len(perms):raise ValueError('Empty/repeated permutation seeds')
    if any(m not in METHODS for m in methods):raise ValueError('Invalid method')
    if any(x not in SHUFFLES for x in shuffles):raise ValueError('Invalid shuffle type')
    d=load(source);extra,feature_manifest,xhash=read_extra(d,features)
    X,names=get_features(d,extra,'F4')
    if tuple(names)!=tuple(FEATURES) or tuple(FEATURE_MODES['F4'])!=tuple(FEATURES):
        raise ValueError('F4 order mismatch')
    fold=masks(d,'task')['internal_test']
    rows=select_rows(d,fold);y=d.label(PRIMARY)[fold];oracle=d.label(ORACLE)[fold]
    if not rows or not any(r['date']!=rows[0]['date'] for r in rows):
        raise ValueError('Task internal test requires different dates')
    for task in {g['task_id'] for g in rows}:
        if len({g['date'] for g in rows if g['task_id']==task})<2:
            raise ValueError('Not enough different-date donors for '+task)
    static_inds=[names.index(f) for f in STRUCTURE]
    columns_traffic=[names.index(f) for f in TRAFFIC]
    columns_rel=[names.index(f) for f in RELIABILITY]
    # Cache common permuted input only once for ALL models: strict paired design.
    swapped_by_seed={}
    for perm in perms:
        swapped,info=shuffle_same_task_date(X,names,d.groups,fold,int(perm))
        if info['donor_fraction']<.95:raise ValueError('Insufficient donors: '+str(info))
        if not np.array_equal(swapped[:,static_inds],X[:,static_inds]):
            raise AssertionError('Static features accidentally shuffled')
        swapped_by_seed[perm]=(swapped,info)
    result=[]
    for seed in seeds:
        study=root/f'b3_anchor_film_seed{seed}_v1'
        if readjson(study/'summary.json')['status']!='COMPLETE':
            raise ValueError('Incomplete B3 study: '+str(study))
        meta=readjson(study/'manifest.json')
        if meta['seed']!=seed or meta['stage']!='step6b3-anchored-film-v1':
            raise ValueError('B3 seed/stage mismatch')
        if meta['feature_npz_sha256']!=xhash or meta['source_sha256']!=feature_manifest['source_sha256']:
            raise ValueError('Data SHA mismatch')
        for method in methods:
            artifact_path=study/'task'/f'{method}_model.joblib'
            artifact=joblib.load(artifact_path) # TRUSTED self-trained local model files only.
            if (artifact['b3_method'],artifact['mode'],artifact['method'],artifact['protocol'])!=(method,'F4','P2','task'):
                raise ValueError('Artifact identity mismatch')
            if artifact['feature_names']!=list(FEATURES) or artifact['boundary_X_sha256']!=xhash:
                raise ValueError('Artifact feature mismatch')
            model=artifact['model'];threshold=artifact['threshold']
            if not np.isclose(model.alpha,artifact['alpha']):raise ValueError('Alpha inconsistent')
            original_scores,original_gates,_=model.predict(X[fold])
            # Exact saved offline candidate-score identity detects source/schema/version mistakes.
            with (study/'task'/f'{method}_test_candidates.csv').open(newline='',encoding='utf-8') as f:
                import csv
                saved=list(csv.DictReader(f))
            if [r['sample_id'] for r in saved]!=[r['sample_id'] for r in rows]:
                raise ValueError('Test candidate alignment/order changed')
            saved_scores=np.asarray([float(r['model_score_sec']) for r in saved])
            if not np.allclose(original_scores,saved_scores,atol=1e-3,rtol=1e-6):
                raise ValueError('Frozen model score not reproducing saved test predictions')
            baseline=summarize(choose(y,oracle,original_scores,original_gates,rows,threshold))
            base=baseline['mean_signed_gain_sec']
            _,_,_,q_identity=from_full(X[fold],FEATURES)
            result.append({'seed':seed,'method':method,'shuffle':'identity','permutation_seed':None,
                'mean_signed_gain_sec':base,'delta_vs_identity_sec_per_state':0.,
                'selected_alpha':artifact['alpha'],'threshold':threshold,
                'executed_states':baseline['executed_states'],
                'nonpositive_executions':baseline['nonpositive_executions'],
                'gate_zero_rate':float(np.mean(q_identity<=1e-12)),
                'states':baseline['states'],'donor_fraction':1.})
            for shuffle in shuffles:
                for perm in perms:
                    candidate,info=swapped_by_seed[perm]
                    xt=candidate.copy()
                    # Joint history and mask is the primary placebo. The two
                    # isolated permutations are intentionally stress controls.
                    if shuffle=='traffic_only':xt[:,columns_rel]=X[:,columns_rel]
                    elif shuffle=='mask_only':xt[:,columns_traffic]=X[:,columns_traffic]
                    if not np.array_equal(xt[:,static_inds],X[:,static_inds]):
                        raise AssertionError('Static input modified')
                    if np.array_equal(xt[fold],X[fold]):
                        raise ValueError('No effective permutation')
                    rank,gate,_=model.predict(xt[fold])
                    rr=summarize(choose(y,oracle,rank,gate,rows,threshold))
                    _,_,_,q=from_full(xt[fold],FEATURES)
                    result.append({'seed':seed,'method':method,'shuffle':shuffle,
                        'permutation_seed':perm,
                        'mean_signed_gain_sec':rr['mean_signed_gain_sec'],
                        'delta_vs_identity_sec_per_state':rr['mean_signed_gain_sec']-base,
                        'selected_alpha':artifact['alpha'],'threshold':threshold,
                        'executed_states':rr['executed_states'],
                        'nonpositive_executions':rr['nonpositive_executions'],
                        'gate_zero_rate':float(np.mean(q<=1e-12)),
                        'states':rr['states'],'donor_fraction':info['donor_fraction']})
    aggregates=[]
    for method in methods:
        for typ in shuffles:
            deltas=[z['delta_vs_identity_sec_per_state'] for z in result
                    if z['method']==method and z['shuffle']==typ]
            dd=np.asarray(deltas)
            aggregates.append({'method':method,'shuffle':typ,'observations':len(deltas),
                'mean_delta_vs_true_sec_per_state':float(dd.mean()),
                'median_delta_vs_true_sec_per_state':float(np.median(dd)),
                'fraction_traffic_placebo_worse_than_true':float(np.mean(dd< -1e-7)),
                'independent_sample_warning':'Model/permutation repetitions reuse heldout states; do not use as independent sample size.'})
    output.mkdir(parents=True,exist_ok=False)
    obj={'stage':'step6b3r-frozen-placebo-v1','source':str(source),'features':str(features),
         'input_feature_sha256':xhash,'model_seeds':list(seeds),'permutation_seeds':list(perms),
         'protocol':'task/internal_test','results':result,'aggregate':aggregates,
         'notes':['No refit, no threshold/alpha tuning, no post-hoc selection.',
             'Same-task/different-date state-wise donor used for comparability with B2-R frozen_permutation.',
             'Different-date donor Section positions may have different semantics: feature-swap sensitivity, NOT a causal treatment.',
             'Traffic-only and mask-only break history-mask consistency and are intentionally stress controls.',
             'Date split cannot support within-task across-date test permutation; no date/placebo result is claimed.',
             'July repeated development data, not blind; repeated permutations and model seeds are not independent heldout states.']}
    (output/'placebo.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(aggregates,ensure_ascii=False,indent=2))
    return obj


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',default=SOURCE)
    p.add_argument('--features-dir',default=BOUNDARY)
    p.add_argument('--study-root',default='runs/chicago_july_v1/step6b3')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--model-seeds',default='42,43,44,45,46,47')
    p.add_argument('--permutation-seeds',default=','.join(map(str,range(100,120))))
    p.add_argument('--methods',default=','.join(METHODS))
    p.add_argument('--shuffles',default=','.join(SHUFFLES))
    a=p.parse_args()
    run(a.source_dir,a.features_dir,a.study_root,a.output_dir,
        tuple(int(x) for x in a.model_seeds.split(',')),a.methods.split(','),
        tuple(int(x) for x in a.permutation_seeds.split(',')),a.shuffles.split(','))
if __name__=='__main__':main()
