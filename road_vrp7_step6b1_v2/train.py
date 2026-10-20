#!/usr/bin/env python3
"""Step6-B1 v2 grouped offline comparison; signed static-repair TD objective reward.

No online ALNS claim and no Oracle choice is used at test-time.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import average_precision_score,roc_auc_score,mean_absolute_error,brier_score_loss
from dataset import (load,DEFAULT_SOURCE,PRIMARY,ORACLE,masks,select_rows,
                     grouped_indices,sha256,route_overlap)
from models import Predictor,METHODS,cls_target
from audit_static_repair import analyze as audit_repair

CLASS_THRESHOLDS=(0.,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99)
GAIN_THRESHOLDS=(-100.,-40.,-20.,-10.,0.,5.,10.,20.,40.,60.,100.)


def csv_write(path,rows):
    if not rows:raise ValueError('Empty output '+str(path))
    path=Path(path)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)


def choose(y,oracle,scores,gate,groups,threshold):
    """One candidate per state or no-op. Signed negative reward is NOT hindsight clipped.

    Selected action is determined exclusively by scores and gate (X derived).
    Ground-truth arrays are read ONLY after the action has been selected.
    """
    if len(groups)!=len(y) or len(oracle)!=len(y) or len(scores)!=len(y) or len(gate)!=len(y):
        raise ValueError('Prediction/reward alignment mismatch')
    if not (np.isfinite(scores).all() and np.isfinite(gate).all()):raise ValueError('Nonfinite model scores')
    result=[]
    for sid,idx in grouped_indices(groups).items():
        chosen=int(idx[np.argmax(scores[idx])])
        execute=(threshold is not None and float(gate[chosen])>=threshold)
        row=groups[chosen]
        raw=float(y[chosen]);upper=max(0.,float(np.max(y[idx])))
        upper_td=max(0.,float(np.max(oracle[idx])))
        result.append({'state_id':sid,'task_id':row['task_id'],'date':row['date'],
            'scenario_id':row['scenario_id'],'selected_candidate_id':row['sample_id'],
            'execute':int(execute),'score':float(scores[chosen]),'gate_score':float(gate[chosen]),
            'chosen_static_repair_td_gain_sec':raw,
            'realized_signed_gain_sec':raw if execute else 0.,
            'realized_oracle_td_repair_gain_sec':float(oracle[chosen]) if execute else 0.,
            'oracle_static_repair_noop_bound_sec':upper,
            'oracle_td_repair_noop_bound_sec':upper_td,
            'ranking_top1_matches_static_repair_best':int(chosen==int(idx[np.argmax(y[idx])]))})
    return result


def summarize(rows):
    if not rows:raise ValueError('No states')
    reward=np.array([r['realized_signed_gain_sec'] for r in rows],float)
    alt=np.array([r['realized_oracle_td_repair_gain_sec'] for r in rows],float)
    upper=np.array([r['oracle_static_repair_noop_bound_sec'] for r in rows],float)
    act=np.array([r['execute'] for r in rows],bool)
    upgrade=(alt-reward)
    return {'states':len(rows),'executed_states':int(act.sum()),
        'execution_rate':float(np.mean(act)),
        'positive_executions':int(np.sum(act&(reward>1e-7))),
        'nonpositive_executions':int(np.sum(act&(reward<=1e-7))),
        'mean_signed_gain_sec':float(np.mean(reward)),
        'total_signed_gain_sec':float(np.sum(reward)),
        'mean_static_repair_oracle_noop_bound_sec':float(np.mean(upper)),
        'mean_policy_regret_to_static_repair_oracle_sec':float(np.mean(upper-reward)),
        'mean_oracle_td_repair_same_chosen_section_executed_sec':float(np.mean(alt)),
        'mean_oracle_repair_upgrade_all_states_sec':float(np.mean(upgrade)),
        'mean_oracle_repair_upgrade_executed_only_sec':float(np.mean(upgrade[act])) if np.any(act) else None,
        'candidate_ranking_top1_match_rate':float(np.mean([r['ranking_top1_matches_static_repair_best'] for r in rows]))}


def threshold_options(kind,validation_gate):
    base=CLASS_THRESHOLDS if kind in ('cls','pair','pair_inter') else GAIN_THRESHOLDS
    # Extra cut points are learned from VAL only, never test.
    q=np.quantile(validation_gate,[0.,.1,.25,.5,.75,.9,1.])
    options=sorted(set([float(x) for x in (*base,*q)]))
    return options+[None]


def tune_threshold(y,oracle,score,gate,groups,kind):
    choices=[]
    for t in threshold_options(kind,gate):
        records=choose(y,oracle,score,gate,groups,t)
        met=summarize(records)
        # Max validation realized signed reward; conservative tie => fewer actions.
        key=(round(met['mean_signed_gain_sec'],9),-met['executed_states'],float(t) if t is not None else float('inf'))
        choices.append((key,t,met))
    _,t,met=max(choices,key=lambda x:x[0]);return t,met


def midranks(values):
    values=np.asarray(values)
    sort=np.argsort(values,kind='stable');res=np.zeros(len(values),float)
    i=0
    while i<len(values):
        j=i+1
        while j<len(values) and values[sort[i]]==values[sort[j]]:j+=1
        res[sort[i:j]]=(i+j-1)*.5
        i=j
    return res


def state_rank_stats(score,y,groups):
    corr=[];exact=0;states=grouped_indices(groups)
    for idx in states.values():
        a=midranks(score[idx]);b=midranks(y[idx])
        exact+=int(int(idx[np.argmax(score[idx])])==int(idx[np.argmax(y[idx])]))
        if len(idx)>=3 and np.std(a)>1e-12 and np.std(b)>1e-12:
            corr.append(float(np.corrcoef(a,b)[0,1]))
    return {'mean_state_spearman':float(np.mean(corr)) if corr else None,
            'states_spearman_defined':len(corr),
            'state_top1_hit_rate':exact/len(states)}


def test_metrics(y,ranking,gate_probability,groups,kind):
    ys=cls_target(y)
    result=state_rank_stats(ranking,y,groups)
    if kind in ('pair','pair_inter'):
        # Pairwise scores are meaningful within state, not across arbitrary states.
        result['global_ranking_PR_AUC']=None
        result['global_ranking_ROC_AUC']=None
    else:
        result['global_ranking_PR_AUC']=(float(average_precision_score(ys,ranking)) if len(set(ys))>1 else None)
        result['global_ranking_ROC_AUC']=(float(roc_auc_score(ys,ranking)) if len(set(ys))>1 else None)
    if gate_probability is not None:
        if np.min(gate_probability)<-1e-10 or np.max(gate_probability)>1+1e-10:
            raise ValueError('Invalid gate probabilities')
        result['gate_positive_PR_AUC']=float(average_precision_score(ys,gate_probability)) if len(set(ys))>1 else None
        result['gate_positive_ROC_AUC']=float(roc_auc_score(ys,gate_probability)) if len(set(ys))>1 else None
        result['gate_Brier_score']=float(brier_score_loss(ys,gate_probability))
    else:
        result['gate_positive_PR_AUC']=None
        result['gate_positive_ROC_AUC']=None
        result['gate_Brier_score']=None
    if kind in ('reg','two','two_cal'):
        result['signed_prediction_MAE_sec']=float(mean_absolute_error(y,ranking))
    return result


def evaluate_protocol(d,protocol,methods,output,seed,trees):
    split=masks(d,protocol)
    y=d.label(PRIMARY);ora=d.label(ORACLE)
    local=output/protocol;local.mkdir(parents=True,exist_ok=False)
    s={'protocol':protocol,'primary_label':PRIMARY,'methods':{},'controls':{},'splits':{}}
    for k,mask in split.items():
        gs=select_rows(d,mask)
        s['splits'][k]={'states':len(grouped_indices(gs)),'candidates':len(gs),
            'positive_candidates':int(np.sum(y[mask]>1e-7)),
            'tasks':sorted(set(r['task_id'] for r in gs)),
            'dates':sorted(set(r['date'] for r in gs))}
    test=split['internal_test'];test_groups=select_rows(d,test)
    for name,score,gate,threshold in [
        ('always_noop',np.zeros(int(test.sum())),np.zeros(int(test.sum())),None),
        ('static_score_unconditional',d.X[test,d.names.index('static_section_cost_sec')],
         np.ones(int(test.sum())),-float('inf'))]:
        records=choose(y[test],ora[test],score,gate,test_groups,threshold)
        s['controls'][name]=summarize(records)
        csv_write(local/f'{name}_test_states.csv',records)
    for name in methods:
        feature_mode,kind=METHODS[name]
        X,feat_names=d.features(feature_mode)
        train=split['train'];validation=split['val']
        model=Predictor(kind,seed,trees,feat_names,protocol)
        model.fit(X[train],y[train],select_rows(d,train))
        score_v,gate_v,p_v=model.predict(X[validation])
        th,val_report=tune_threshold(y[validation],ora[validation],score_v,gate_v,
                       select_rows(d,validation),kind)
        score_t,gate_t,p_t=model.predict(X[test])
        records=choose(y[test],ora[test],score_t,gate_t,test_groups,th)
        csv_write(local/f'{name}_test_states.csv',records)
        csv_write(local/f'{name}_validation_states.csv',choose(y[validation],ora[validation],score_v,gate_v,
                       select_rows(d,validation),th))
        candidates=[]
        for g,rank,gate,pr,real,oracle_gain in zip(test_groups,score_t,gate_t,
                              (p_t if p_t is not None else [None]*len(score_t)),y[test],ora[test]):
            candidates.append({'state_id':g['state_id'],'sample_id':g['sample_id'],
                'task_id':g['task_id'],'date':g['date'],'rank_score':float(rank),
                'gate_score':float(gate),'predicted_positive_probability':float(pr) if pr is not None else '',
                'static_repair_td_gain_sec':float(real),'oracle_td_repair_gain_sec':float(oracle_gain)})
        csv_write(local/f'{name}_test_candidates.csv',candidates)
        artifact={'model':model,'feature_names':feat_names,'feature_mode':feature_mode,
            'method':name,'protocol':protocol,'source_schema_sha256':sha256(d.source_files['schema']),
            'primary_target':PRIMARY,'seed':seed}
        joblib.dump(artifact,local/f'{name}_model.joblib',compress=3)
        s['methods'][name]={'kind':kind,'feature_mode':feature_mode,
             'feature_count':len(feat_names),'validation_threshold':th,
             'validation_policy':val_report,'internal_test_policy':summarize(records),
             'test_metrics':test_metrics(y[test],score_t,p_t,test_groups,kind),
             'group_training_calibration':model.calibration_diagnostics,
             'training_pair_count':getattr(model,'training_pairs',None)}
        print(f'{protocol:4s} {name:34s} gain={s["methods"][name]["internal_test_policy"]["mean_signed_gain_sec"]:8.3f} sec/state; threshold={th}',flush=True)
    (local/'report.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return s


def run(source,output,protocols,seed=42,trees=120,methods=None):
    output=Path(output)
    if output.exists():raise FileExistsError('Independent study directory required: '+str(output))
    data=load(source)
    methods=list(METHODS) if methods is None else list(methods)
    if len(methods)!=len(set(methods)) or any(k not in METHODS for k in methods) or not methods:
        raise ValueError('Unknown/duplicate/empty methods')
    if len(protocols)!=len(set(protocols)) or not protocols:raise ValueError('Invalid protocols')
    for p in protocols:masks(data,p)
    if trees<5:raise ValueError('Too few trees')
    audit=audit_repair(data)
    output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('models.py'),Path(__file__).with_name('dataset.py'),
           Path(__file__).with_name('audit_static_repair.py')]
    manifest={'stage':'step6b1-v2','source':str(data.source),'seed':seed,'trees':trees,
      'methods':methods,'protocols':list(protocols),'target':PRIMARY,
      'source_sha256':{k:sha256(v) for k,v in data.source_files.items()},
      'code_sha256':{p.name:sha256(p) for p in files},
      'caveats':['Development data: no blind month, offline single-step TD objective labels.',
        'No full 30s ALNS integration or road replay; no model inference latency included.',
        'Dates can share identical static route structure. Repeated seeds reuse held-out states.',
        'Class probabilities calibrated only on grouped training OOF, not the validation or test split.']}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (output/'static_repair_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    reports=[evaluate_protocol(data,p,methods,output,seed,trees) for p in protocols]
    result={'status':'COMPLETE','reports':reports}
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--protocol',choices=('task','date','both'),default='both')
    p.add_argument('--methods',default=','.join(METHODS))
    p.add_argument('--trees',type=int,default=120);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    run(a.source_dir,a.output_dir,('task','date') if a.protocol=='both' else (a.protocol,),a.seed,a.trees,
        [n.strip() for n in a.methods.split(',') if n.strip()])

if __name__=='__main__':main()
