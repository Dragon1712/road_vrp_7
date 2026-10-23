#!/usr/bin/env python3
"""Fast same-state selector evaluation on the 1024 REAL labeled C3 snapshots.

ZERO new Oracle queries. Only Y, BY labels already obtained ex post are used.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1]
# Old M0 joblib artifact records models.Predictor from step6b1_v2.
# The pickle loader imports that module by its ORIGINAL top-level name.
# Match preflight_sv.py's legacy module search path before joblib.load().
for name in ('road_vrp7_step6c1_r3_sv','road_vrp7_step6c1_r3_tricks',
             'road_vrp7_step6c1_r2','road_vrp7_step6b1_v2'):
    if str(ROOT/name) not in sys.path:sys.path.insert(0,str(ROOT/name))
from sv_core import ARMS,sha256,atomic_json,selection,m0_selection,classify_row,summary,pick_states
from r2_policy import group_split

DEFAULT_LABEL='runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1'
DEFAULT_T0='runs/chicago_july_v1/step6c1_r2/r2_task_seed42_v1/R2_F2.joblib'
DEFAULT_T2='runs/chicago_july_v1/step6c1_r3_tricks/r3_seed42_v1/T2_CB_HUBER.joblib'
DEFAULT_M0='runs/chicago_july_v1/step6b2/b2_full_seed42_v1/task/P2_F2_model.joblib'


def load(a):
    directory=Path(a.dataset_dir);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['stage']!='step6c1r2-real-C3-labelled-v1':raise ValueError('Not real C3 labels')
    if sha256(directory/'paired.npz')!=manifest['pair_sha256'] or sha256(directory/'groups.jsonl')!=manifest['groups_sha256']:
        raise ValueError('Paired C3 label checksum mismatch')
    with np.load(directory/'paired.npz',allow_pickle=False) as f:
        X,B,Y,BY,valid,same=(f[k].copy() for k in ('X','B','Y','BY','valid','same'))
    groups=[json.loads(l) for l in (directory/'groups.jsonl').read_text().splitlines() if l.strip()]
    if X.shape!=(len(groups),8,44) or B.shape!=(len(groups),44) or Y.shape!=(len(groups),8) or BY.shape!=(len(groups),) or valid.shape!=Y.shape or same.shape!=Y.shape:
        raise ValueError('Bad R2 data shape')
    folds,roles=group_split(groups,'task')
    artifacts={name:joblib.load(p) for name,p in [('R2_F2',a.r2_model),('T2_HUBER',a.huber_model),('M0',a.m0_model)]}
    if artifacts['R2_F2'].get('stage')!='step6c1r2-c3-advantage-train-v1' or artifacts['R2_F2'].get('mode')!='F2':raise ValueError('R2 model mismatch')
    if artifacts['T2_HUBER'].get('stage')!='step6c1r3t-tricks-training-v1' or artifacts['T2_HUBER'].get('r3_arm')!='T2_CB_HUBER' or artifacts['T2_HUBER'].get('mode')!='F2':raise ValueError('T2 model mismatch')
    if artifacts['T2_HUBER']['r2_dataset_sha256']!=manifest['pair_sha256'] or artifacts['R2_F2']['r2_dataset_sha256']!=manifest['pair_sha256']:
        raise ValueError('Model trained on different labels')
    if artifacts['M0'].get('mode')!='F2':raise ValueError('M0 must be F2')
    return X,B,Y,BY,valid,same,groups,roles,artifacts,manifest


def run(a):
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('Independent NEW output directory required: '+str(out))
    X,B,Y,BY,valid,same,groups,roles,art,manifest=load(a)
    sig={'stage':'R3-SV-one-step-v1','labels_sha256':manifest['pair_sha256'],
         'groups_sha256':manifest['groups_sha256'],
         'model_sha256':{n:sha256(p) for n,p in [('R2_F2',a.r2_model),('T2_HUBER',a.huber_model),('M0',a.m0_model)]},
         'code_sha256':{n:sha256(Path(__file__).with_name(n)) for n in ('sv_core.py','same_state.py')},
         'note':'M0_PROXY is same-state fixed 8 candidates with C3 defer, NOT original M0 trajectory'}
    # compute before creating output dir to avoid leaving false COMPLETE on failure
    rows=[]
    for i,g in enumerate(groups):
        idx={'C3':(-1,'frozen_actual_C3',0.)}
        for label,k in [('R2_F2','R2_F2'),('T2_HUBER','T2_HUBER')]:
            idx[label]=selection(art[k]['model'],X[i,:,:26],B[i,:26],valid[i],same[i])
        idx['M0_PROXY']=m0_selection(art['M0'],X[i,:,:26],valid[i],same[i])
        meta={k:g[k] for k in ('decision_id','task_id','scenario_id','seed','date','section_call')}
        meta['role']=roles[str(g['task_id'])]
        rows.extend(classify_row(Y[i]-BY[i],valid[i],same[i],idx,meta))
    out.mkdir(parents=True)
    atomic_json(out/'manifest.json',sig)
    with (out/'decisions.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    ids=pick_states(rows,max_states=a.max_pilot_states,seed=a.pilot_seed,role=a.pilot_role)
    atomic_json(out/'pilot_states.json',{'role':a.pilot_role,'seed':a.pilot_seed,'max_states':a.max_pilot_states,
      'selected_decision_ids':ids,'sampling':'disagreement_enriched_not_population_representative'})
    atomic_json(out/'report.json',{'status':'COMPLETE','n_states':len(groups),'n_rows':len(rows),
                                  'by_role':{role:summary([r for r in rows if r['role']==role]) for role in ('train','validation','internal_test')},
                                  'sampling_plan':str(out/'pilot_states.json')})
    print('R3-SV ONE STEP DONE states=',len(groups),'pilot=',len(ids),'out=',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',default=DEFAULT_LABEL)
    p.add_argument('--r2-model',default=DEFAULT_T0)
    p.add_argument('--huber-model',default=DEFAULT_T2)
    p.add_argument('--m0-model',default=DEFAULT_M0)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--max-pilot-states',type=int,default=12)
    p.add_argument('--pilot-role',choices=('train','validation','internal_test'),default='internal_test')
    p.add_argument('--pilot-seed',type=int,default=20261023)
    run(p.parse_args())
