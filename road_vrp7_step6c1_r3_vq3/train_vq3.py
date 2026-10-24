#!/usr/bin/env python3
"""VQ3: decision/regret-oriented and task-calibrated harm-gate diagnostics.

Original R2 TRAIN tasks only, outer leave-one-task-out and inner leave-one-
training-task-out. NEVER use validation/internal_test labels or heldout labels in
fitting, Platt calibration, policy threshold, or coverage matching. No online
ALNS changes, and H-step values are restart surrogates, not continuous Q.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r3_vq3','road_vrp7_step6c1_r3_vq2',
               'road_vrp7_step6c1_r3_vq','road_vrp7_step6c1_r2'):
    path=str(ROOT/folder)
    if path not in sys.path:sys.path.insert(0,path)
import train_vq
from vq_core import paired_features
from r2_policy import group_split
from vq2_core import (sha256,write_json,fit_extra,select,one_step_rows,
                      rows_for_selected,train_arrays,oof_residual_training,
                      metrics_for_decisions)
from vq3_core import (STAGE,POLICIES,HARM_CUTOFF,train_pairrank,
                      crossfit_calibrated_harm,prob_diagnostics,pairwise_accuracy,
                      risk_gate,coverage_matched_tau)


def match_train_one_step(full,multi,train_tasks):
    records=[z for z in full if z['task'] in train_tasks]
    matched=[z for z in multi if z['task'] in train_tasks]
    fst={(z['decision'],z['action']) for z in records if z['decision'] in {a['decision'] for a in matched}}
    mst={(z['decision'],z['action']) for z in matched}
    if fst!=mst or len(mst)!=len(matched):raise AssertionError('Matched baseline IDs differ from VQ')
    if any(z['task'] not in train_tasks for z in records+matched):raise AssertionError('Train task boundary broken')
    return records,matched


def train_score_coverage_control(matched,train_tasks,oof_prob,seed,trees):
    """Outer train ONLY: crossfit E0_MATCHED and estimate risk-gate coverage."""
    best_scores=[];good=0;folds=[]
    for task in sorted(train_tasks):
        train=[z for z in matched if z['task']!=task]
        val=[z for z in matched if z['task']==task]
        X,Y,W=train_arrays(train,'one')
        model=fit_extra(X,Y,seed=seed+101,trees=trees)
        by=defaultdict(list)
        for z in val:by[z['decision']].append(z)
        for _,candidates in sorted(by.items()):
            p=model.predict(np.stack([c['x'] for c in candidates]))
            j=int(np.argmax(p));score=float(p[j]);item=candidates[j]
            best_scores.append(score)
            if score>0 and oof_prob[(item['decision'],item['action'])]<=.35:good+=1
        folds.append({'oof_task':task,'fit_tasks':sorted(set(train_tasks)-{task}),
                      'n_states':len(by)})
    frac=good/len(best_scores)
    tau=coverage_matched_tau(best_scores,frac)
    return tau,{'oof_risk35_coverage':frac,'score_control_oof_coverage':float(np.mean(np.array(best_scores)>tau)),
                'score_control_threshold':tau,'folds':folds}


def build_fold(*,X,B,Y,BY,valid,same,groups,lookup,selection,measured,h,
               holdout,roles,all_one,all_multi,seed,trees):
    tasks=sorted({z['task'] for z in all_multi});train_tasks=set(tasks)-{holdout}
    if len(train_tasks)!=3:raise ValueError('Exactly 4 original Train tasks required')
    train_ids=[di for di in selection if groups[lookup[di]]['task_id'] in train_tasks]
    test_ids=[di for di in selection if groups[lookup[di]]['task_id']==holdout]
    if not test_ids or set(train_ids)&set(test_ids):raise AssertionError('Invalid outer fold')
    if any(roles[groups[lookup[di]]['task_id']]!='train' for di in train_ids+test_ids):raise AssertionError('Forbidden split')
    full,matched=match_train_one_step(all_one,all_multi,train_tasks)
    xf,yf,wf=train_arrays(full,'y');xm,ym,wm=train_arrays(matched,'one');xq,yq,wq=train_arrays(matched,'mean')
    if not(np.array_equal(xm,xq) and np.array_equal(wm,wq)):raise AssertionError('Sample or weight mismatch')
    mfull=fit_extra(xf,yf,seed=seed,trees=trees)
    mmatched=fit_extra(xm,ym,seed=seed+101,trees=trees)
    mvq=fit_extra(xq,yq,seed=seed+101,trees=trees)
    residual,oo_info=oof_residual_training(full,matched,train_tasks,seed=seed+404,trees=trees)
    xr,yr,wr=train_arrays(residual,'residual')
    mres=fit_extra(xr,yr,seed=seed+505,trees=trees)
    rank= train_pairrank(matched,weighted=False,seed=seed+709)
    rankw=train_pairrank(matched,weighted=True,seed=seed+710)
    risk,oofrisk,risk_train_diag=crossfit_calibrated_harm(matched,train_tasks,seed=seed+801,trees=trees)
    tau,coverage_diag=train_score_coverage_control(matched,train_tasks,oofrisk,seed,trees)
    decisions=[];predictions=[];ranks=defaultdict(list)
    for di in test_ids:
        idx=lookup[di];task=groups[idx]['task_id'];mask=np.asarray(valid[idx]&(~same[idx]),bool)
        ft=paired_features(X[idx,:,:26],B[idx,:26]);pf=mfull.predict(ft);pm=mmatched.predict(ft)
        pq=mvq.predict(ft);pr=mres.predict(ft);pair=rank.predict(ft);pairw=rankw.predict(ft)
        harmrisk=risk.predict(ft)
        scores={'C3':np.zeros(8), 'E0_FULL':pf,'E0_MATCHED':pm,'VQ_MEAN':pq,
                'RESID_OOF25':pf+.25*pr,'RESID_OOF50':pf+.5*pr,
                'PAIR_RANK':pair,'PAIR_REGRET':pairw,
                'E0_RISK20':pm,'E0_RISK35':pm,
                'RESID25_RISK35':pf+.25*pr,
                'PAIR_REGRET_RISK35':pairw,
                'E0_SCORE_COVERAGE35':pm}
        realized=np.zeros(8,float)
        for j in np.flatnonzero(mask):
            z=measured[(di,int(h),int(j))];realized[j]=float(z['mean_gain'])
            predictions.append({'decision_id':di,'task_id':task,'horizon':int(h),'action_index':int(j),
                'true_restart_mean_gain_sec':realized[j], 'true_restart_sd_sec':float(z['sd_gain']),
                'true_mean_harm_gt40':int(realized[j]<HARM_CUTOFF),
                'pred_risk_harm':float(harmrisk[j]),
                'pred_step_full':float(pf[j]),'pred_step_matched':float(pm[j]),
                'pred_vq_mean':float(pq[j]),'pred_residual_oof':float(pr[j]),
                'pred_pair_rank':float(pair[j]),'pred_pair_regret':float(pairw[j])})
        oracle=max(0.,float(np.max(realized[mask])) if mask.any() else 0.)
        for name in POLICIES:
            if name=='C3':choice=-1
            elif name=='E0_RISK20':choice=risk_gate(pm,mask,harmrisk,.20)
            elif name=='E0_RISK35':choice=risk_gate(pm,mask,harmrisk,.35)
            elif name=='RESID25_RISK35':choice=risk_gate(scores[name],mask,harmrisk,.35)
            elif name=='PAIR_REGRET_RISK35':choice=risk_gate(pairw,mask,harmrisk,.35)
            elif name=='E0_SCORE_COVERAGE35':choice=select(pm,mask,threshold=tau)
            else:choice=select(scores[name],mask)
            gain=0. if choice==-1 else float(realized[choice])
            decisions.append({'decision_id':di,'task_id':task,'horizon':int(h),'policy':name,
                'selected_index':int(choice),'realized_mean_road_gain':gain,
                'regret':oracle-gain,'oracle_gain':oracle,
                'selected_pred_score':float(scores[name][choice]) if choice!=-1 else 0.,
                'selected_pred_mean_harm_probability':float(harmrisk[choice]) if choice!=-1 else None,
                'selected_is_severe_harm':bool(gain<HARM_CUTOFF)})
            ranks[name].append((scores[name],realized,mask))
    diag=prob_diagnostics([r['true_mean_harm_gt40'] for r in predictions],
                          [r['pred_risk_harm'] for r in predictions],
                          # State-normalized weights on heldout predictions
                          weights=[1./sum(z['decision_id']==r['decision_id'] for z in predictions) for r in predictions])
    rmet={name:pairwise_accuracy(rs) for name,rs in ranks.items() if name in
          ('E0_FULL','E0_MATCHED','VQ_MEAN','RESID_OOF25','RESID_OOF50','PAIR_RANK','PAIR_REGRET')}
    details={'heldout_task':holdout,'n_train_states':len(train_ids),'n_test_states':len(test_ids),
         'n_full_one_step_actions':len(yf),'n_matched_one_step_actions':len(ym),
         'n_multi_step_actions':len(yq),'exactly_matched_candidate_ids_and_weights':True,
         'train_tasks':sorted(train_tasks),'risk_model_oof':risk_train_diag,
         'coverage_control':coverage_diag,'oof_residual_folds':oo_info,
         'rank_training_pairs':rank.training_pairs,'rank_regret_training_pairs':rankw.training_pairs,
         'seed':int(seed),'trees':int(trees)}
    return decisions,predictions,details,diag,rmet


def policy_macro(per_task):
    keys=('mean_gain_sec','mean_regret_sec','top1_hit','override_rate',
          'harm_gt40_per_state','harm_gt40_per_override','mean_excess_harm_gt40')
    return {name:{k:float(np.mean([per_task[t][name][k] for t in per_task])) for k in keys}
            for name in POLICIES}


def run(a):
    if a.trees<10 or a.seed<0:raise ValueError('Invalid seed/trees')
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('New independent output directory required')
    config,_,X,B,Y,BY,valid,same,groups,lookup,selection,measured=train_vq.load(a)
    _,roles=group_split(groups,'task')
    tasks=sorted({str(groups[lookup[di]]['task_id']) for di in selection})
    if len(tasks)!=4 or any(roles[t]!='train' for t in tasks):raise ValueError('Require exactly 4 original TRAIN tasks')
    all_one=one_step_rows(X,B,Y,BY,valid,same,groups,roles,set(tasks),paired_features)
    folds={};macro={};outdec=[];outcand=[]
    for h in config['horizons']:
        all_multi=rows_for_selected(selection,groups,lookup,X,B,Y,BY,valid,same,measured,h,paired_features)
        per_task={}
        for task in tasks:
            decisions,pred,details,diag,ranks=build_fold(X=X,B=B,Y=Y,BY=BY,valid=valid,same=same,
                groups=groups,lookup=lookup,selection=selection,measured=measured,h=h,holdout=task,
                roles=roles,all_one=all_one,all_multi=all_multi,seed=a.seed,trees=a.trees)
            metrics={name:metrics_for_decisions([d for d in decisions if d['policy']==name]) for name in POLICIES}
            per_task[task]={'counts':details,'metrics':metrics,'candidate_risk_diagnostics':diag,
                            'pairwise_accuracy_including_C3':ranks}
            outdec.extend(decisions);outcand.extend(pred)
        folds[str(h)]=per_task
        macro[str(h)]=policy_macro({t:z['metrics'] for t,z in per_task.items()})
    if len(outdec)!=len(selection)*len(config['horizons'])*len(POLICIES):raise AssertionError('Missing state/policy evaluations')
    if len({(r['decision_id'],r['horizon'],r['policy']) for r in outdec})!=len(outdec):raise AssertionError('Repeated decision')
    out.mkdir(parents=True,exist_ok=False)
    for name,rows in [('fold_decisions.jsonl',outdec),('candidate_predictions.jsonl',outcand)]:
        with (out/name).open('w',encoding='utf8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    report={'stage':STAGE,'status':'COMPLETE','seed':int(a.seed),'trees':int(a.trees),
       'n_original_train_tasks':4,'n_selected_states':len(selection),'horizons':config['horizons'],
       'policies':list(POLICIES),
       'macro_task_metrics':macro,'folds':folds,
       'risk_target':'1[mean of 3 paired C3-restart road gains < -40 seconds]',
       'ranking_target':'Within-state mean Road.replay restart gains including actual C3=0',
       'limits':['Only original TRAIN tasks; outer task leave-one-out. No R2 validation or internal_test labels.',
                 'LOTO evaluation contains only 4 correlated tasks x 12 states, NOT independent blind validation.',
                 'Search restart surrogate is NOT continuous 30s ALNS action value.',
                 'Three random restarts create noisy mean-harm labels; probability has no formal safety guarantee.',
                 'Platt calibration trained on only 3 outer train tasks; even task-OOF Platt can be unstable.',
                 'Online latency not measured; no 30-second improvement claim.',
                 'All alpha and risk cutoffs were fixed BEFORE LOTO evaluation, never tuned on heldout outcomes.']}
    write_json(out/'report.json',report)
    from report_vq3 import markdown
    (out/'report.md').write_text(markdown(report),encoding='utf8')
    base=Path(a.collection_dir);source=Path(a.dataset_dir);pkg=Path(__file__).parent
    import vq_core,vq2_core,r2_policy
    write_json(out/'provenance.json',{'stage':STAGE,'seed':a.seed,'trees':a.trees,
             'collection_manifest_sha256':sha256(base/'manifest.json'),
             'collection_rollouts_sha256':sha256(base/'rollouts.jsonl'),
             'dataset_paired_sha256':sha256(source/'paired.npz'),
             'dataset_groups_sha256':sha256(source/'groups.jsonl'),
             'code_sha256':{p.name:sha256(p) for p in [Path(__file__),pkg/'vq3_core.py',pkg/'report_vq3.py',
                  Path(train_vq.__file__),Path(vq_core.__file__),Path(vq2_core.__file__),Path(r2_policy.__file__)]}})
    print('R3-VQ3 COMPLETE',out,flush=True)
    for h,met in macro.items():print('H='+h+' gain/sec/state:',{k:round(v['mean_gain_sec'],3) for k,v in met.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',default='runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1')
    p.add_argument('--collection-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trees',type=int,default=160)
    run(p.parse_args())
