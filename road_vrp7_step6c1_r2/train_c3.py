#!/usr/bin/env python3
"""Train baseline-relative Section predictors; val-only threshold selection."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
import joblib,numpy as np
ROOT=Path(__file__).resolve().parents[1]
for f in ('road_vrp7_step6c1_r2','road_vrp7_step6b1_v2','road_vrp7_step6b2'):
    sys.path.insert(0,str(ROOT/f))
from r2_policy import R2Advantage,evaluate,group_split
from dataset import sha256
from features import FEATURE_MODES


def train(a):
    out=Path(a.output_dir);dataset=Path(a.dataset_dir)
    if out.exists():raise FileExistsError('Use independent output directory')
    meta=json.loads((dataset/'manifest.json').read_text())
    if meta['stage']!='step6c1r2-real-C3-labelled-v1' or meta['pair_sha256']!=sha256(dataset/'paired.npz') or meta['groups_sha256']!=sha256(dataset/'groups.jsonl'):
        raise ValueError('Invalid or modified C3 labels')
    with np.load(dataset/'paired.npz',allow_pickle=False) as z:
        X,B,Y,BY,valid,same=(z[k].copy() for k in ('X','B','Y','BY','valid','same'))
    groups=[json.loads(v) for v in (dataset/'groups.jsonl').read_text().splitlines() if v.strip()]
    if len(groups)!=len(X) or X.shape!=(len(groups),8,44) or B.shape!=(len(groups),44) or valid.shape!=(len(groups),8):
        raise ValueError('R2 labeled shape mismatch')
    source_config=json.loads((Path(a.model_source)/'run_config.json').read_text())
    if source_config['source_cost_hash']!=meta['road_sha256'] or source_config['source_edge_hash']!=meta['edge_sha256']:
        raise ValueError('C3 labels and frozen online model source reference different roads')
    folds,roles=group_split(groups,a.protocol)
    out.mkdir(parents=True)
    mf={'stage':'step6c1r2-c3-advantage-train-v1','seed':a.seed,'protocol':a.protocol,'modes':a.modes.split(','),
        'training_source':str(dataset),'label_sha256':meta['pair_sha256'],'groups_sha256':meta['groups_sha256'],
        'split_roles':roles,'model_source':a.model_source,
        'training_objective':'Static Repair one-step TD advantage vs ACTUAL frozen C3 action',
        'validation_objective':'real C3-relative signed gain, conservative harm penalty',
        'test_role':'internal_test is development only, never used in model selection',
        'code_sha256':{'r2_policy.py':sha256(Path(__file__).with_name('r2_policy.py')),
             'train_c3.py':sha256(Path(__file__))}}
    (out/'manifest.json').write_text(json.dumps(mf,indent=2,ensure_ascii=False)+'\n')
    reports={};thresholds=(0.,5.,10.,20.,40.,80.,float('inf'))
    for mode in a.modes.split(','):
        if mode not in ('F2','F4'):raise ValueError(mode)
        model=R2Advantage(mode=mode,seed=a.seed)
        model.fit(X,B,Y,BY,valid,same,folds['train'])
        grid=[]
        for thresh in thresholds:
            model.threshold=thresh
            res=evaluate(model,X,B,Y,BY,valid,same,np.flatnonzero(folds['validation']))
            # Conservative: account for outsized losses, ties favor fewer overrides.
            utility=res['mean_delta_vs_actual_C3_sec_per_state']-.25*res['mean_harm_tail_gt_40sec']
            grid.append({'threshold':None if not np.isfinite(thresh) else thresh,
                         'utility':utility,'result':res})
        best=max(grid,key=lambda x:(round(x['utility'],8),-x['result']['overrides'],
                                   x['threshold'] if x['threshold'] is not None else float('inf')))
        model.threshold=float('inf') if best['threshold'] is None else float(best['threshold'])
        stats={p:evaluate(model,X,B,Y,BY,valid,same,np.flatnonzero(m)) for p,m in folds.items()}
        source=Path(a.model_source)
        # B2-R online artifact checks. These source files already existed before R2.
        train_src={'config':sha256(source/'run_config.json'),
                   'groups':sha256(source/'analysis/groups.csv'),
                   'X':sha256(source/'analysis/features_X.npz')}
        artifact={'stage':'step6c1r2-c3-advantage-train-v1','model':model,'mode':mode,
           'method':'P2','protocol':a.protocol,'feature_names':list(FEATURE_MODES[mode]),
           'threshold':0.,'online_compatible':True,'training_label':'static_repair_td_gain_sec',
           'source_sha256':train_src,'r2_dataset_sha256':meta['pair_sha256'],
           'r2_source':str(dataset),'r2_training_code_sha256':mf['code_sha256'],
           'r2_policy':'actual_C3_shadow_advantage'}
        fn=out/f'R2_{mode}.joblib';joblib.dump(artifact,fn,compress=3)
        reports[mode]={'validation_gate_candidates':grid,'selected_threshold':best['threshold'],
          'selected_validation_utility':best['utility'],'scores':stats,
          'artifact_sha256':sha256(fn)}
        print('R2TRAIN',a.protocol,mode,'threshold',model.threshold,
              'validation',stats['validation']['mean_delta_vs_actual_C3_sec_per_state'],
              'internal_test',stats['internal_test']['mean_delta_vs_actual_C3_sec_per_state'],flush=True)
    (out/'report.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps({'status':'COMPLETE','n_states':len(groups),'protocol':a.protocol,
            'modes':list(reports)},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--model-source',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--protocol',choices=('task','date'),default='task')
    p.add_argument('--modes',default='F2,F4');p.add_argument('--seed',type=int,default=42)
    train(p.parse_args())
