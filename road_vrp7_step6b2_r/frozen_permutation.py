#!/usr/bin/env python3
"""Frozen P2/P4 F3/F4 task model, same-task different-date traffic placebo.

Neither models nor thresholds are refit. Labels are used ONLY to score the
already chosen action. Tests are July development internal test, not blind.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import joblib
R=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6b2','road_vrp7_step6b1_v2'):
    p=str(R/folder)
    if p not in sys.path:sys.path.insert(0,p)
from features import FEATURE_MODES, TRAFFIC, RELIABILITY
from train_b2 import read_extra,get_features
from collections import defaultdict
from dataset import load,masks,select_rows,PRIMARY,ORACLE,sha256
from train import choose,summarize



def shuffle_same_task_date(X,names,groups,mask,seed):
    """Strict original-rows donor swap. Works for F3 (no reliability) and F4."""
    ids=np.flatnonzero(mask)
    selected=[groups[i] for i in ids]
    by_state=defaultdict(list)
    for i,g in enumerate(selected):by_state[g['state_id']].append(i)
    dates={s:selected[ix[0]]['date'] for s,ix in by_state.items()}
    tasks={s:selected[ix[0]]['task_id'] for s,ix in by_state.items()}
    if len(set(dates.values()))<2:raise ValueError('Cannot shuffle single-date test split')
    cols=[names.index(k) for k in TRAFFIC+RELIABILITY if k in names]
    if not cols:raise ValueError('No traffic history to permute')
    rng=np.random.default_rng(seed);out=X.copy();mapped=0
    for state,idx in sorted(by_state.items()):
        donor_keys=[v for v,ix in by_state.items()
            if tasks[v]==tasks[state] and dates[v]!=dates[state] and len(ix)==len(idx)]
        if not donor_keys:continue
        donor=by_state[donor_keys[int(rng.integers(len(donor_keys)))]]
        out[np.ix_(ids[idx],cols)]=X[np.ix_(ids[donor],cols)]
        mapped+=1
    return out,{'donor_fraction':mapped/max(1,len(by_state)),
                'donor_mapped_states':mapped,'states':len(by_state)}


def probe(source,features,study,output,models=('P2','P4'),modes=('F3','F4'),
          seeds=tuple(range(100,120)),shuffles=('traffic_only','traffic_and_mask','mask_only')):
    source=Path(source);features=Path(features);study=Path(study);output=Path(output)
    if output.exists():raise FileExistsError(output)
    d=load(source);extra,mf,xhash=read_extra(d,features)
    mask=masks(d,'task')['internal_test']
    grp=select_rows(d,mask);y=d.label(PRIMARY)[mask];ora=d.label(ORACLE)[mask]
    study_meta=json.loads((study/'manifest.json').read_text())
    if study_meta.get('stage') not in ('step6b2-ablation-v1','step6b2r-ablation-v1'):
        raise ValueError('Incompatible model study')
    if study_meta['boundary_X_sha256']!=xhash or study_meta['frozen_source_sha256']!=mf['source_sha256']:
        raise ValueError('Source mismatch')
    if study_meta['seed'] is None:raise ValueError('Unknown model seed')
    result=[]
    for method in models:
        for mode in modes:
            if mode not in ('F3','F4'):raise ValueError('Only F3/F4 have traffic permutation')
            path=study/'task'/f'{method}_{mode}_model.joblib'
            # Trusted local study artifacts only. Joblib files are executable pickle content.
            artifact=joblib.load(path)
            if artifact['method']!=method or artifact['mode']!=mode or artifact['protocol']!='task':
                raise ValueError('Artifact mismatch')
            if artifact['boundary_X_sha256']!=xhash:raise ValueError('Artifact training data mismatch')
            X,names=get_features(d,extra,mode)
            if list(names)!=artifact['feature_names']:raise ValueError('Model input mismatch')
            model=artifact['model'];thr=artifact['threshold']
            z,g,_=model.predict(X[mask]);control=summarize(choose(y,ora,z,g,grp,thr))
            base=control['mean_signed_gain_sec']
            result.append({'model':method+'_'+mode,'permutation':'identity','seed':None,
                           'mean_signed_gain_sec':base,'delta_vs_identity_sec':0.,
                           'executed_states':control['executed_states'],
                           'nonpositive_executions':control['nonpositive_executions'],
                           'states':len(set(r['state_id'] for r in grp))})
            for which in shuffles:
                if which=='mask_only' and not any(t in names for t in RELIABILITY):
                    continue  # F3 has no reliability features; a mask-only placebo is undefined.
                if which not in ('traffic_only','traffic_and_mask','mask_only'):raise ValueError(which)
                for seed in seeds:
                    swapped,info=shuffle_same_task_date(X,names,d.groups,mask,int(seed))
                    if info['donor_fraction']<.95:raise ValueError('Not enough same-task cross-date donors')
                    out=swapped.copy()
                    restore=(RELIABILITY if which=='traffic_only' else
                             TRAFFIC if which=='mask_only' else ())
                    for feat in restore:
                        
                        if feat in names:
                            j=names.index(feat);out[:,j]=X[:,j]
                    if np.array_equal(out[mask],X[mask]):raise ValueError('Permutation had no effect')
                    r,gate,_=model.predict(out[mask])
                    stats=summarize(choose(y,ora,r,gate,grp,thr))
                    result.append({'model':method+'_'+mode,'permutation':which,'seed':int(seed),
                       'mean_signed_gain_sec':stats['mean_signed_gain_sec'],
                       'delta_vs_identity_sec':stats['mean_signed_gain_sec']-base,
                       'executed_states':stats['executed_states'],
                       'nonpositive_executions':stats['nonpositive_executions'],
                       'states':len(set(x['state_id'] for x in grp)),
                       'donor_fraction':info['donor_fraction']})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'source_study':str(study),'source_hashes':mf['source_sha256'],
      'feature_sha256':xhash,'scope':'task internal_test / same-task different-date within-test permuting',
      'note':('Fixed model and validation threshold; labels used for scoring only. '
              'Not valid for date protocol: internal_test contains one date.'),
      'results':result},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for method in models:
        for mode in modes:
            name=f'{method}_{mode}';rows=[z for z in result if z['model']==name]
            print(name,'identity',round(rows[0]['mean_signed_gain_sec'],3),
                  'shuffles',[(t,round(float(np.mean([r['delta_vs_identity_sec'] for r in rows if r['permutation']==t])),3)) for t in shuffles if any(r['permutation']==t for r in rows)])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--features-dir',default='runs/chicago_july_v1/step6b2/features_boundary_v1')
    p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    p.add_argument('--models',default='P2,P4');p.add_argument('--modes',default='F3,F4')
    p.add_argument('--seeds',default=','.join(map(str,range(100,120))))
    a=p.parse_args()
    seeds=[int(z) for z in a.seeds.split(',')]
    if len(seeds)!=len(set(seeds)) or not seeds:raise ValueError('Duplicate/empty seeds')
    probe(a.source_dir,a.features_dir,a.study,a.output,a.models.split(','),a.modes.split(','),seeds)
if __name__=='__main__':main()
