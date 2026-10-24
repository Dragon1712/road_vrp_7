#!/usr/bin/env python3
"""Fit deployable F2 E0_MATCHED and OOF-residual on ORIGINAL TRAIN tasks only.

Uses same VQ3 ExtraTrees training recipe; no threshold tuning and no access to
original validation/internal_test labels. Does not edit the frozen solver.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np
ROOT=Path(__file__).resolve().parents[1]
for f in ('road_vrp7_step6c1_r3_vq6','road_vrp7_step6c1_r3_vq3',
          'road_vrp7_step6c1_r3_vq2','road_vrp7_step6c1_r3_vq',
          'road_vrp7_step6c1_r2','road_vrp7_step6b2'):
    p=str(ROOT/f)
    if p not in sys.path:sys.path.insert(0,p)
import train_vq
from r2_policy import group_split
from vq_core import paired_features
from vq2_core import one_step_rows,rows_for_selected,train_arrays,fit_extra,oof_residual_training,sha256,write_json
from vq6_policy import VQ6Policy,KINDS
from features import FEATURE_MODES

STAGE='R3-VQ6-ONLINE-FROZEN-TRAIN-v1'


def fit_from_records(full,matched,tasks,seed=42,trees=160):
    tasks=set(tasks)
    if len(tasks)!=4 or set(z['task'] for z in matched)!=tasks:
        raise ValueError('Four original TRAIN tasks are required')
    key={(z['decision'],z['action']) for z in matched}
    if len(key)!=len(matched):raise ValueError('Duplicate multi-step action')
    full_matched={(z['decision'],z['action']) for z in full if z['decision'] in {p['decision'] for p in matched}}
    if full_matched!=key:raise ValueError('Matched one-step and VQ candidate IDs differ')
    xm,ym,wm=train_arrays(matched,'one')
    xf,yf,wf=train_arrays(full,'y')
    matched_model=fit_extra(xm,ym,seed=seed+101,trees=trees)
    full_model=fit_extra(xf,yf,seed=seed,trees=trees)
    residual_rows,info=oof_residual_training(full,matched,tasks,seed=seed+404,trees=trees)
    xr,yr,wr=train_arrays(residual_rows,'residual')
    res_model=fit_extra(xr,yr,seed=seed+505,trees=trees)
    models={
        'E0_MATCHED':VQ6Policy(kind='E0_MATCHED',mode='F2',seed=seed,threshold=0.,model=matched_model),
        'RESID_OOF50':VQ6Policy(kind='RESID_OOF50',mode='F2',seed=seed,threshold=0.,model=full_model,residual_model=res_model,alpha=.5),
        'DEFER_RESID':VQ6Policy(kind='DEFER_RESID',mode='F2',seed=seed,threshold=0.,model=full_model,residual_model=res_model,alpha=.5),
    }
    return models,{'n_matched_actions':len(ym),'n_full_onestep_actions':len(yf),
                   'n_residual_actions':len(yr),'n_train_tasks':len(tasks),
                   'oof_folds':info,'model_recipe':'VQ3 ExtraTrees maxdepth12 leaf3 160 trees, no weighting'}


def run(a):
    if a.trees!=160 or a.seed<0 or a.horizon not in (20,50):
        raise ValueError('Frozen initial pilot uses TREES=160 and H in {20,50}')
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('Independent NEW training study directory required')
    config,report,X,B,Y,BY,valid,same,groups,lookup,selected,measured=train_vq.load(a)
    if len(selected)!=192 or sorted(config['horizons'])!=[20,50] or config['replicates']!=3:
        raise ValueError('Only complete VQ4 192-state H20,H50/R3 labels are allowed')
    roles=group_split(groups,'task')[1]
    selected_tasks=sorted({str(groups[lookup[di]]['task_id']) for di in selected})
    if len(selected_tasks)!=4 or any(roles[t]!='train' for t in selected_tasks):
        raise ValueError('Training leaked to non-Train tasks')
    one=one_step_rows(X,B,Y,BY,valid,same,groups,roles,set(selected_tasks),paired_features)
    many=rows_for_selected(selected,groups,lookup,X,B,Y,BY,valid,same,measured,a.horizon,paired_features)
    models,stats=fit_from_records(one,many,selected_tasks,a.seed,a.trees)
    source=Path(a.model_source)
    cfg=json.loads((source/'run_config.json').read_text())
    mf=json.loads((Path(a.dataset_dir)/'manifest.json').read_text())
    if cfg['source_cost_hash']!=mf['road_sha256'] or cfg['source_edge_hash']!=mf['edge_sha256']:
        raise ValueError('Training labels / online source roads differ')
    hashes={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('vq6_policy.py'))}
    base={'stage':STAGE,'mode':'F2','method':'P2','protocol':'task','online_compatible':True,
          'feature_names':list(FEATURE_MODES['F2']),
          'training_label':'static_repair_td_gain_sec',  # compatibility token; see actual target below
          'target_description':'E0 static-repair TD gain; residual uses C3-restart H-step ex-post Road.replay',
          'source_sha256':{'config':sha256(source/'run_config.json'),
                           'groups':sha256(source/'analysis/groups.csv'),
                           'X':sha256(source/'analysis/features_X.npz')},
          'r2_source':str(a.dataset_dir),'r2_dataset_sha256':mf['pair_sha256'],
          'vq_collection_dir':str(a.collection_dir),'vq_collection_manifest_sha256':sha256(Path(a.collection_dir)/'manifest.json'),
          'vq_collection_rollouts_sha256':sha256(Path(a.collection_dir)/'rollouts.jsonl'),
          'training_code_sha256':hashes,'training_horizon':a.horizon,
          'decision_threshold_sec':0.,'training_seed':a.seed,
          'notes':'Predictions are static-history F2 only; road replay only labels; no online oracle access.'}
    out.mkdir(parents=True)
    report_models={}
    for kind,model in models.items():
        path=out/f'{kind}.joblib'
        obj={**base,'vq6_kind':kind,'model':model}
        joblib.dump(obj,path,compress=3)
        report_models[kind]={'path':str(path),'sha256':sha256(path)}
    write_json(out/'report.json',{'status':'COMPLETE','stage':STAGE,
       'models':report_models,'training':stats,'train_task_ids':selected_tasks,'n_states':len(selected),
       'horizon':a.horizon,'seed':a.seed,'trees':a.trees,
       'caveats':['Four original TRAIN tasks only','No new model-family hyperparameter search',
                  'The residual is a restart surrogate, not original continuous-ALNS Q']})
    write_json(out/'provenance.json',base)
    print('VQ6 TRAIN COMPLETE:',out,stats,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-dir',required=True)
    p.add_argument('--dataset-dir',default='runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1')
    p.add_argument('--model-source',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--horizon',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trees',type=int,default=160)
    run(p.parse_args())
