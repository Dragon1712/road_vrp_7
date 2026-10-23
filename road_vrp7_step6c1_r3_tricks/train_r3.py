#!/usr/bin/env python3
"""First-batch R3 tricks: honest task-disjoint labels, val-only model/gate selection.

T0 frozen R2-F2 (external)
T1 CatBoost RMSE
T2 CatBoost Huber
T3 lightweight pairwise neural scorer (+ regression anchor)
T4 regret-weighted pairwise
T5 T2 + selective margin
T6 T4 + selective margin
T7 fixed 50/50 T2+T4 blend + selective margin (exploratory)
"""
from __future__ import annotations
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import joblib,numpy as np
ROOT=Path(__file__).resolve().parents[1]
for name in ('road_vrp7_step6c1_r3_tricks','road_vrp7_step6c1_r2',
             'road_vrp7_step6b1_v2','road_vrp7_step6b2'):
    p=str(ROOT/name)
    if p not in sys.path:sys.path.insert(0,p)
from r2_policy import group_split
from r3_models import R3Advantage,ARMS,make_blend
from decision_metrics import evaluate,select_gate

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_data(directory,model_source):
    d=Path(directory);m=json.loads((d/'manifest.json').read_text())
    if m['stage']!='step6c1r2-real-C3-labelled-v1' or m['pair_sha256']!=sha(d/'paired.npz') or m['groups_sha256']!=sha(d/'groups.jsonl'):
        raise ValueError('Changed/wrong C3 relative label dataset')
    with np.load(d/'paired.npz',allow_pickle=False) as z:
        X,B,Y,BY,valid,same=[z[k].copy() for k in ('X','B','Y','BY','valid','same')]
    groups=[json.loads(r) for r in (d/'groups.jsonl').read_text().splitlines() if r.strip()]
    if X.shape!=(len(groups),8,44) or B.shape!=(len(groups),44) or Y.shape!=(len(groups),8) or BY.shape!=(len(groups),) or valid.shape!=Y.shape or same.shape!=Y.shape:
        raise ValueError('R2 paired data shape mismatch')
    if not np.isfinite(X).all() or not np.isfinite(B).all() or not np.isfinite(Y).all() or not np.isfinite(BY).all():
        raise ValueError('Nonfinite R2 pair tensor')
    cfg=json.loads((Path(model_source)/'run_config.json').read_text())
    if cfg['source_cost_hash']!=m['road_sha256'] or cfg['source_edge_hash']!=m['edge_sha256']:
        raise ValueError('Model source and label roads mismatch')
    return X,B,Y,BY,valid,same,groups,m


def train(a):
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('Use new R3 output directory: '+str(out))
    X,B,Y,BY,valid,same,groups,labelmeta=load_data(a.dataset_dir,a.model_source)
    folds,roles=group_split(groups,a.protocol)
    if a.protocol!='task':raise ValueError('First batch is strictly task-disjoint; date can be a separate later experiment')
    indices={k:np.flatnonzero(v) for k,v in folds.items()}
    out.mkdir(parents=True)
    manifest={'stage':'step6c1r3t-tricks-training-v1','seed':a.seed,
      'protocol':a.protocol,'label_sha256':labelmeta['pair_sha256'],
      'groups_sha256':labelmeta['groups_sha256'],'dataset_dir':str(a.dataset_dir),
      'model_source':str(a.model_source),'folds':roles,
      'baseline_artifact':str(a.baseline_artifact),'epoch_count':a.epochs,
      'selection':'validation only; internal_test diagnostic, never model selection',
      'r3_code_sha256':{'r3_models.py':sha(Path(__file__).with_name('r3_models.py')),
         'decision_metrics.py':sha(Path(__file__).with_name('decision_metrics.py')),
         'train_r3.py':sha(Path(__file__))}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    models={};report={}
    if a.baseline_artifact:
        orig=joblib.load(a.baseline_artifact)
        if orig['mode']!='F2' or orig['stage']!='step6c1r2-c3-advantage-train-v1' or orig['r2_dataset_sha256']!=labelmeta['pair_sha256'] or orig['protocol']!=a.protocol:
            raise ValueError('Frozen R2 baseline identity/data/protocol mismatch')
        b=orig['model'];report['T0_R2_F2']={'frozen_artifact_sha256':sha(a.baseline_artifact),
               'scores':{k:evaluate(b,X,B,Y,BY,valid,same,v) for k,v in indices.items()}}
    for name,kind in [('T1_CB_RMSE','catboost_rmse'),('T2_CB_HUBER','catboost_huber'),
                      ('T3_PAIRWISE','pairwise'),('T4_WEIGHTED_PAIR','weighted_pairwise')]:
        st=time.perf_counter()
        m=R3Advantage(mode='F2',seed=a.seed,algorithm=kind)
        m.fit(X,B,Y,BY,valid,same,folds['train'],epochs=a.epochs)
        models[name]=m
        print('R3 trained',name,'sec',round(time.perf_counter()-st,2),flush=True)
    # Each base model independently calibrated on VALIDATION. No comparison with internal_test.
    for name in ('T1_CB_RMSE','T2_CB_HUBER','T3_PAIRWISE','T4_WEIGHTED_PAIR'):
        models[name],grid,best=select_gate(models[name],X,B,Y,BY,valid,same,indices['validation'])
        report[name]={'gate_grid':grid,'selected_gate':{'threshold':best['threshold'],'margin':best['margin']}}
    # Margin variants inherit their parent's already-validated threshold.
    for new,old in [('T5_HUBER_MARGIN','T2_CB_HUBER'),('T6_WEIGHTED_MARGIN','T4_WEIGHTED_PAIR')]:
        m=copy.deepcopy(models[old]);m.margin=0.
        # Keep parent's threshold fixed to avoid a 2-D parameter search on 256 states.
        _,grid,best=select_gate(m,X,B,Y,BY,valid,same,indices['validation'],
                         thresholds=(m.threshold,),margins=(0.,5.,10.,20.,40.))
        models[new]=m;report[new]={'parent':old,'gate_grid':grid,
             'selected_gate':{'threshold':best['threshold'],'margin':best['margin']}}
    # A fixed blend; the validation set selects ONLY its final gate.
    blend=make_blend(models['T2_CB_HUBER'],models['T4_WEIGHTED_PAIR'],a.seed)
    blend,grid,best=select_gate(blend,X,B,Y,BY,valid,same,indices['validation'],
                      thresholds=(0.,20.,40.,80.,float('inf')),
                      margins=(0.,10.,20.))
    models['T7_BLEND_MARGIN']=blend
    report['T7_BLEND_MARGIN']={'gate_grid':grid,
       'selected_gate':{'threshold':best['threshold'],'margin':best['margin']}}
    src=Path(a.model_source)
    source_sha={'config':sha(src/'run_config.json'),
        'groups':sha(src/'analysis/groups.csv'),
        'X':sha(src/'analysis/features_X.npz')}
    from features import FEATURE_MODES
    for name,m in models.items():
        path=out/(name+'.joblib')
        art={'stage':'step6c1r3t-tricks-training-v1',
          'model':m,'mode':'F2','method':'P2','protocol':a.protocol,
          'feature_names':list(FEATURE_MODES['F2']),
          'threshold':0.,'online_compatible':True,
          'training_label':'static_repair_td_gain_sec',
          'source_sha256':source_sha,'r2_dataset_sha256':labelmeta['pair_sha256'],
          'r2_source':str(a.dataset_dir),'r3_arm':name,
          'r3_training_code_sha256':manifest['r3_code_sha256']}
        joblib.dump(art,path,compress=3)
        report[name]['artifact_sha256']=sha(path)
        report[name]['scores']={k:evaluate(m,X,B,Y,BY,valid,same,v) for k,v in indices.items()}
        val=report[name]['scores']['validation'];test=report[name]['scores']['internal_test']
        print(name,'gate',m.threshold,m.margin,'VALID regret',round(val['mean_regret_sec'],3),
             'INTERNAL regret',round(test['mean_regret_sec'],3),flush=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps({'status':'COMPLETE','states':len(groups),
      'seed':a.seed,'arms':list(models),'protocol':a.protocol},indent=2)+'\n')
    print('R3 TRAIN COMPLETE',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',default='runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1')
    p.add_argument('--model-source',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--baseline-artifact',default='runs/chicago_july_v1/step6c1_r2/r2_task_seed42_v1/R2_F2.joblib')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--protocol',choices=('task','date'),default='task')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=100)
    a=p.parse_args();train(a)
