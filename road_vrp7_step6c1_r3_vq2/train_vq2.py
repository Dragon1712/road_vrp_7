#!/usr/bin/env python3
"""R3-VQ2: fair one-step vs multi-step labels + OOF residual shrinkage.

Uses ORIGINAL TRAIN task labels only. Evaluates within the four ORIGINAL train
 tasks via outer leave-one-task-out; NEVER reads original validation/test labels.
The objective is a C3-restart surrogate ROAD value, not true on-policy Q.
"""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r3_vq2','road_vrp7_step6c1_r3_vq','road_vrp7_step6c1_r2'):
    p=str(ROOT/folder)
    if p not in sys.path:sys.path.insert(0,p)
import train_vq  # Frozen VQ loader: SHA, label completeness, strict training-only state verification.
from vq_core import paired_features
from r2_policy import group_split
from vq2_core import (STAGE,METHODS,sha256,write_json,fit_extra,select,one_step_rows,
                      rows_for_selected,train_arrays,oof_residual_training,
                      metrics_for_decisions,diagnostic_for_candidates,task_macro)


def build_fold(*,X,B,Y,BY,valid,same,groups,lookup,selection,measured,h,holdout,
               roles,all_one,all_multi,seed,trees):
    selected_tasks=sorted({z['task'] for z in all_multi})
    train_tasks=set(selected_tasks)-{holdout}
    if len(train_tasks)<3:raise ValueError('Need >=4 selected train tasks')
    test_ids=[di for di in selection if groups[lookup[di]]['task_id']==holdout]
    train_ids=[di for di in selection if groups[lookup[di]]['task_id'] in train_tasks]
    if len(test_ids)<1 or len(train_ids)<1:raise ValueError('Empty train/test tasks')
    if set(train_ids)&set(test_ids):raise AssertionError('Outer leakage')
    full_records=[z for z in all_one if z['task'] in train_tasks]
    matched_records=[z for z in all_multi if z['task'] in train_tasks]
    if {z['decision'] for z in matched_records}!=set(train_ids):raise AssertionError('Matched states mismatch')
    # Critical fairness: each E0_MATCHED row has precisely the same state/action ID
    # as the corresponding VQ_MEAN row, and both have identical sample weights.
    matchkey={(z['decision'],z['action']) for z in matched_records}
    if len(matchkey)!=len(matched_records):raise ValueError('Duplicate matched candidates')
    onestep_key={(z['decision'],z['action']) for z in full_records if z['decision'] in set(train_ids)}
    if onestep_key!=matchkey:raise AssertionError('One-step and multi-step matched candidate IDs differ')
    if any(z['task']==holdout for z in full_records+matched_records):raise AssertionError('Heldout contamination')
    xf,yf,wf=train_arrays(full_records,'y');xm,ym,wm=train_arrays(matched_records,'one')
    xq,yq,wq=train_arrays(matched_records,'mean');xu,yu,wu=train_arrays(matched_records,'sd')
    if not (len(xm)==len(xq)==len(xu)) or not np.array_equal(xm,xq) or not np.array_equal(wm,wq):
        raise AssertionError('Unmatched candidate rows or state weights')
    mfull=fit_extra(xf,yf,seed=seed,trees=trees)
    mmatched=fit_extra(xm,ym,seed=seed+101,trees=trees)
    mq=fit_extra(xq,yq,seed=seed+101,trees=trees)
    msd=fit_extra(xu,yu,seed=seed+202,trees=trees)
    residual_train,oof_info=oof_residual_training(full_records,matched_records,train_tasks,seed=seed+404,trees=trees)
    xr,yr,wr=train_arrays(residual_train,'residual')
    mres=fit_extra(xr,yr,seed=seed+505,trees=trees)
    decisions=[];candidates=[]
    for di in test_ids:
        i=lookup[di];task=groups[i]['task_id'];mask=valid[i]&(~same[i])
        if task!=holdout or roles[task]!='train':raise AssertionError('Not original training task')
        feat=paired_features(X[i,:,:26],B[i,:26]);scores_full=mfull.predict(feat)
        scores_mat=mmatched.predict(feat);scores_q=mq.predict(feat)
        scores_sd=np.maximum(0,msd.predict(feat));scores_res=mres.predict(feat)
        scores={'C3':np.zeros(8),
             'E0_FULL':scores_full,'E0_MATCHED':scores_mat,'VQ_MEAN':scores_q,
             'BLEND25':.75*scores_full+.25*scores_q,
             'BLEND50':.5*scores_full+.5*scores_q,
             'RESID_OOF25':scores_full+.25*scores_res,
             'RESID_OOF50':scores_full+.5*scores_res,
             'RESID_OOF100':scores_full+scores_res,
             'VQ_LCB10':scores_q-.10*scores_sd,
             'VQ_LCB25':scores_q-.25*scores_sd}
        true=np.zeros(8,float)
        for j in np.flatnonzero(mask):
            lab=measured[(di,int(h),int(j))];true[j]=float(lab['mean_gain'])
            candidates.append({'decision_id':di,'task_id':task,'horizon':int(h),'action_index':int(j),
                'one_step_gain':float(Y[i,j]-BY[i]),'realized_mean_road_gain':true[j],
                'realized_restart_sd':float(lab['sd_gain']),
                'pred_e0_full':float(scores_full[j]),'pred_e0_matched':float(scores_mat[j]),
                'pred_vq_mean':float(scores_q[j]),'predicted_restart_sd':float(scores_sd[j]),
                'predicted_residual':float(scores_res[j]),
                'pred_resid_oof50':float(scores['RESID_OOF50'][j])})
        oracle=max(0.,float(np.max(true[mask])) if np.any(mask) else 0.)
        for name in METHODS:
            j=-1 if name=='C3' else select(scores[name],mask)
            gain=0. if j==-1 else float(true[j]);pred=0. if j==-1 else float(scores[name][j])
            decisions.append({'decision_id':di,'task_id':task,'horizon':int(h),'policy':name,
                'selected_index':j,'selected_predicted_gain_sec':pred,
                'realized_mean_road_gain':gain,'oracle_gain':oracle,'regret':float(oracle-gain),
                'realized_restart_sd':0. if j==-1 else float(measured[(di,int(h),j)]['sd_gain']),
                'predicted_restart_sd':0. if j==-1 else float(scores_sd[j])})
    details={'n_train_tasks':len(train_tasks),'train_tasks':sorted(train_tasks),'heldout_task':holdout,
             'n_training_states':len(train_ids),'n_heldout_states':len(test_ids),
             'n_E0_FULL_one_step_actions':len(yf),'n_E0_MATCHED_one_step_actions':len(ym),
             'n_VQ_MEAN_actions':len(yq),'n_RESID_OOF_actions':len(yr),
             'same_candidate_ids_and_weights':True,'residual_inner_task_oof':oof_info,
             'seed':int(seed),'n_trees':int(trees)}
    return decisions,candidates,details


def run(a):
    if a.trees<10 or a.seed<0:raise ValueError('Invalid estimator arguments')
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('Use NEW output directory: '+str(out))
    (config,collection_report,X,B,Y,BY,valid,same,groups,lookup,selection,measured)=train_vq.load(a)
    _,roles=group_split(groups,'task')
    selected_tasks=sorted({str(groups[lookup[di]]['task_id']) for di in selection})
    if len(selected_tasks)<4:raise ValueError('Need >=4 original TRAIN tasks for OOF residual and held-out task')
    if any(roles[t]!='train' for t in selected_tasks):raise ValueError('Forbidden validation/test state in VQ collection')
    # Original Train is the *only* set whose TD one-step targets we expose.
    all_one=one_step_rows(X,B,Y,BY,valid,same,groups,roles,set(selected_tasks),paired_features)
    all_decisions=[];all_candidates=[];folds={};macro={}
    for h in config['horizons']:
        by_task={}
        all_multi=rows_for_selected(selection,groups,lookup,X,B,Y,BY,valid,same,measured,h,paired_features)
        for k,holdout in enumerate(selected_tasks):
            decisions,candidates,details=build_fold(X=X,B=B,Y=Y,BY=BY,valid=valid,same=same,
                groups=groups,lookup=lookup,selection=selection,measured=measured,h=h,holdout=holdout,
                roles=roles,all_one=all_one,all_multi=all_multi,seed=a.seed,trees=a.trees)
            per={m:metrics_for_decisions([r for r in decisions if r['policy']==m]) for m in METHODS}
            by_task[holdout]={'counts':details,'policies':per,
                'prediction_diagnostics':diagnostic_for_candidates(candidates)}
            all_decisions.extend(decisions);all_candidates.extend(candidates)
        folds[str(h)]=by_task
        macro[str(h)]=task_macro({t:z['policies'] for t,z in by_task.items()})
    if len({(z['decision_id'],z['horizon'],z['policy']) for z in all_decisions})!=len(all_decisions):
        raise AssertionError('Duplicated state/policy decisions')
    expected=len(selection)*len(config['horizons'])*len(METHODS)
    if len(all_decisions)!=expected:raise AssertionError(f'Incomplete decisions {len(all_decisions)} != {expected}')
    out.mkdir(parents=True,exist_ok=False)
    with (out/'fold_decisions.jsonl').open('w',encoding='utf-8') as f:
        for z in all_decisions:f.write(json.dumps(z,allow_nan=False,ensure_ascii=False)+'\n')
    with (out/'candidate_predictions.jsonl').open('w',encoding='utf-8') as f:
        for z in all_candidates:f.write(json.dumps(z,allow_nan=False,ensure_ascii=False)+'\n')
    report={'stage':STAGE,'status':'COMPLETE','seed':a.seed,'trees':a.trees,
       'n_original_train_tasks':len(selected_tasks),'n_selected_states':len(selection),
       'horizons':config['horizons'],'policies':list(METHODS),
       'source_protocol':'original TRAIN tasks only; outer leave-one-task-out, OOF residual excludes own task',
       'folds':folds,'macro_task_metrics':macro,
       'interpretation_limits':[
           'Only 4 training tasks, 12 selected states per held-out task. No statistical generalization guarantee.',
           'The multi-step value is C3-restart surrogate, NOT exact continuous ALNS value.',
           'ONE_STEP_FULL has intentionally more training labels; MATCHED vs VQ is the fair target comparison.',
           'OOF residual uses 2 training tasks per inner fit and 3 for outer inference; possible distribution shift.',
           'Predicted restart SD measures search randomness, NOT epistemic uncertainty or a valid LCB.',
           'All road outcomes are retrospective labels; no 30s online performance claim.',
           'Validation/internal_test labels are never used for training, hyperparameter choice or cross-validation.']}
    write_json(out/'report.json',report)
    from report_vq2 import markdown_report
    (out/'report.md').write_text(markdown_report(report),encoding='utf-8')
    root_pkg=Path(__file__).parent
    import train_vq as orig
    manifest={'stage':STAGE,'config':{'seed':a.seed,'trees':a.trees,'horizons':config['horizons']},
       'collection_manifest_sha256':sha256(Path(a.collection_dir)/'manifest.json'),
       'collection_rollouts_sha256':sha256(Path(a.collection_dir)/'rollouts.jsonl'),
       'dataset_paired_sha256':sha256(Path(a.dataset_dir)/'paired.npz'),
       'dataset_groups_sha256':sha256(Path(a.dataset_dir)/'groups.jsonl'),
       'code_sha256':{p.name:sha256(p) for p in [Path(__file__),root_pkg/'vq2_core.py',Path(orig.__file__)]}}
    write_json(out/'provenance.json',manifest)
    print('R3-VQ2 COMPLETE:',out,' states=',len(selection),'horizons=',config['horizons'])
    for h,v in macro.items():print('H='+h, {m:round(x['mean_gain_sec'],3) for m,x in v.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',default='runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1')
    p.add_argument('--collection-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trees',type=int,default=160)
    run(p.parse_args())
