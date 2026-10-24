#!/usr/bin/env python3
"""TASK-HELD-OUT diagnostic: one-step ExtraTrees vs multi-restart mean/risk.

All four task groups are originally TRAIN. No heldout R2 validation/internal-test
states/labels used for selecting hyperparameters or fitting models. This is NOT
an online model or a claim about 30-second policy improvement.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
ROOT=Path(__file__).resolve().parents[1]
for f in ('road_vrp7_step6c1_r3_vq','road_vrp7_step6c1_r2'):
    path=str(ROOT/f)
    if path not in sys.path:sys.path.insert(0,path)
from vq_core import STAGE,LABELS,sha256,atomic_json,paired_features,row_labels,select_by_score,decision_metrics
from r2_policy import group_split

BASE='runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1'

def fit_tree(X,y,seed=42):
    if len(y)<12:raise ValueError('Insufficient candidate samples')
    m=ExtraTreesRegressor(n_estimators=160,max_depth=12,min_samples_leaf=3,
                          max_features=.8,n_jobs=1,random_state=seed)
    m.fit(np.asarray(X,np.float32),np.asarray(y,float))
    return m

def load(a):
    c=Path(a.collection_dir);config=json.loads((c/'manifest.json').read_text());report=json.loads((c/'report.json').read_text())
    if config.get('stage')!=STAGE or report.get('stage')!=LABELS or report.get('status')!='COMPLETE':
        raise ValueError('Need a COMPLETE R3-VQ collection')
    source=Path(a.dataset_dir);mf=json.loads((source/'manifest.json').read_text())
    if mf.get('stage')!='step6c1r2-real-C3-labelled-v1' or sha256(source/'paired.npz')!=mf['pair_sha256'] or sha256(source/'groups.jsonl')!=mf['groups_sha256']:
        raise ValueError('Wrong original labels')
    if mf['pair_sha256']!=config['label_sha256'] or mf['groups_sha256']!=config['group_sha256']:
        raise ValueError('Labels/source do not match collection')
    with np.load(source/'paired.npz',allow_pickle=False) as z:
        X,B,Y,BY,valid,same=(z[k].copy() for k in ('X','B','Y','BY','valid','same'))
    groups=[json.loads(v) for v in (source/'groups.jsonl').read_text().splitlines() if v.strip()]
    _,roles=group_split(groups,'task')
    lookup={g['decision_id']:i for i,g in enumerate(groups)}
    selection=config['selected_decision_ids']
    if len(selection)!=len(set(selection)) or any(di not in lookup for di in selection):raise ValueError('Invalid collection decisions')
    if any(roles[str(groups[lookup[di]]['task_id'])]!='train' for di in selection):
        raise ValueError('VQ training collection contains non-TRAIN states!')
    if len(set(groups[lookup[di]]['task_id'] for di in selection))<3:raise ValueError('Need >=3 train tasks for CV')
    with (c/'rollouts.jsonl').open() as f:
        rows=[json.loads(s) for s in f if s.strip()]
    if len(rows)!=report['n_restart_action_rows']:raise ValueError('Incomplete label rows')
    measured=row_labels(rows,'road')
    H=config['horizons'];reps=int(config['replicates'])
    if reps<2:raise ValueError('VQ uncertainty model requires >=2 repetitions')
    if any(v['n_replicates']!=reps for v in measured.values()):raise ValueError('Incomplete restart repeats')
    for di in selection:
        idx=lookup[di];opts=set(np.flatnonzero(valid[idx] & (~same[idx])))|{-1}
        for h in H:
            found={a for d,hh,a in measured if d==di and hh==h}
            if found!=opts:raise ValueError(f'Missing/extra candidate labels: {di}/H{h}')
    return config,report,X,B,Y,BY,valid,same,groups,lookup,selection,measured

def run(a):
    (config,report,X,B,Y,BY,valid,same,groups,lookup,selection,measured)=load(a)
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('New independent result directory required')
    tasks=sorted({groups[lookup[di]]['task_id'] for di in selection})
    # Flatten old train ONE-STEP labels only; task leave-out means held out task is not fitted.
    roles=group_split(groups,'task')[1]
    if any(roles[t]!='train' for t in tasks):raise AssertionError('Nontrain fold')
    raw=[]
    for i,g in enumerate(groups):
        if roles[g['task_id']]!='train':continue
        j=np.flatnonzero(valid[i] & (~same[i]))
        if len(j):
            feats=paired_features(X[i,:,:26],B[i,:26]); y=Y[i,j]-BY[i]
            for jj,z in zip(j,y):raw.append((g['task_id'],feats[jj].copy(),float(z)))
    if not raw:raise ValueError('Empty baseline training labels')
    fold_rows=[];results={}
    for h in config['horizons']:
        report_h={}
        for holdout in tasks:
            train_ids=[di for di in selection if groups[lookup[di]]['task_id']!=holdout]
            val_ids=[di for di in selection if groups[lookup[di]]['task_id']==holdout]
            if not train_ids or not val_ids:raise ValueError('Empty cross-validation fold')
            ft=[];yt=[];rt=[]
            for di in train_ids:
                i=lookup[di]; matrix=paired_features(X[i,:,:26],B[i,:26]);
                for j in np.flatnonzero(valid[i] & (~same[i])):
                    label=measured[(di,h,int(j))]
                    ft.append(matrix[j]);yt.append(label['mean_gain']);rt.append(label['sd_gain'])
            one_x=[z for task,z,y in raw if task!=holdout];one_y=[y for task,z,y in raw if task!=holdout]
            m_one=fit_tree(one_x,one_y,seed=a.seed)
            m_mean=fit_tree(ft,yt,seed=a.seed+101)
            m_sd=fit_tree(ft,rt,seed=a.seed+202)
            policies=['C3','ONE_STEP_ET','VQ_MEAN','VQ_LCB05','VQ_LCB10']
            pred={k:[] for k in policies};truth=[];masks=[];task_keys=[]
            for di in val_ids:
                i=lookup[di];features=paired_features(X[i,:,:26],B[i,:26]);mask=valid[i]&(~same[i])
                scores0=m_one.predict(features);scores=m_mean.predict(features);sds=np.maximum(0,m_sd.predict(features))
                choices={'C3':-1,'ONE_STEP_ET':select_by_score(scores0,mask),
                         'VQ_MEAN':select_by_score(scores,mask),
                         'VQ_LCB05':select_by_score(scores,mask,lam=.5,std=sds),
                         'VQ_LCB10':select_by_score(scores,mask,lam=1.,std=sds)}
                target=np.array([measured[(di,h,j)]['mean_gain'] if (di,h,j) in measured else 0. for j in range(8)],float)
                truth.append(target);masks.append(mask);task_keys.append(holdout)
                for method,j in choices.items():
                    pred[method].append(j)
                    fold_rows.append({'decision_id':di,'task_id':holdout,'horizon':h,'policy':method,
                        'selected_index':j,'realized_mean_road_gain':0. if j==-1 else float(target[j]),
                        'realized_gain_std':0. if j==-1 else float(measured[(di,h,j)]['sd_gain']),
                        'best_possible_gain':max(0.,float(max(target[mask],default=0.))),
                        'baseline_pred_gain':float(scores0[j]) if j!=-1 else 0.,
                        'vq_pred_gain':float(scores[j]) if j!=-1 else 0.,
                        'vq_pred_std':float(sds[j]) if j!=-1 else 0.})
            report_h[holdout]={'n_train_selected_states':len(train_ids),'n_valid_selected_states':len(val_ids),
                'n_train_vq_candidate_samples':len(yt),'n_train_one_step_samples':len(one_y),
                'metrics':{method:decision_metrics(pred[method],truth,masks,task_keys)
                           for method in policies}}
        # Report each task once; per-task means rather than action/seed count.
        aggregate={}
        for pol in ['C3','ONE_STEP_ET','VQ_MEAN','VQ_LCB05','VQ_LCB10']:
            vals=[report_h[t]['metrics'][pol] for t in tasks]
            aggregate[pol]={k:float(np.mean([v[k] for v in vals])) for k in
                ('mean_gain','mean_regret','harm_over_40_per_state','override_rate','top1_hit')}
        results[str(h)]={'per_heldout_task':report_h,'macro_task_metrics':aggregate}
    out.mkdir(parents=True)
    with (out/'fold_decisions.jsonl').open('w') as f:
        for z in fold_rows:f.write(json.dumps(z,ensure_ascii=False,allow_nan=False)+'\n')
    atomic_json(out/'report.json',{'stage':'R3-VQ-LOTO-v1','status':'COMPLETE',
        'n_selected_states':len(selection),'n_train_tasks':len(tasks),'horizons':config['horizons'],
        'risk_penalties':[0,.5,1],
        'no_internal_test_or_validation_data_used':True,
        'protocol':'task-held-out on originally TRAIN tasks, exploratory only; risk/mean labels from paired C3 restart',
        'results':results,'limits':['Road.replay labels are retrospective; never online inputs',
             'Train selected states are small/correlated; not blind independent test',
             'Multi-step labels from fresh C3 restart, not exact continuation',
             'E0 has larger one-step dataset by design; report training counts']})
    atomic_json(out/'provenance.json',{'source_collection_manifest_sha256':sha256(Path(a.collection_dir)/'manifest.json'),
       'source_collection_rollouts_sha256':sha256(Path(a.collection_dir)/'rollouts.jsonl'),
       'source_r2_labels_sha256':config['label_sha256'],
       'code_sha256':{n:sha256(Path(__file__).with_name(n)) for n in ('train_vq.py','vq_core.py')},
       'seed':a.seed})
    print('R3-VQ TASK-HELDOUT COMPLETE',out)
    for h,z in results.items():
        print('H',h, {m:round(t['mean_gain'],3) for m,t in z['macro_task_metrics'].items()})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-dir',required=True);p.add_argument('--dataset-dir',default=BASE)
    p.add_argument('--output-dir',required=True);p.add_argument('--seed',type=int,default=42)
    run(p.parse_args())
