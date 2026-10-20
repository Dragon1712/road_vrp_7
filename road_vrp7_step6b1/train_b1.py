#!/usr/bin/env python3
"""Step6-B1: grouped, leak-audited Section utility and ranking baselines.

Primary target: static_repair_td_gain_sec, i.e. STATIC repair evaluated
retrospectively under Oracle TD costs. td_gain_sec is an auxiliary Oracle
repair upper bound, NEVER a model input or primary policy reward.
"""
from __future__ import annotations
import argparse,csv,json,math,os,sys
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score,mean_absolute_error
from dataset import DEFAULT_SOURCE,PRIMARY,ORACLE,load,group_masks,rows_at,states_index,sha256
from audit_static_repair import analyze as audit_repair

METHODS={
 'P0_class_core':('core','cls'),
 'P1_class_static_time':('static','cls'),
 'P2_reg_static_time':('static','reg'),
 'P3_pairwise_static_time':('static','pair'),
 'P4_two_part_static_time':('static','two'),
 'P5_reg_history':('traffic','reg'),
 'P6_pairwise_history':('traffic','pair'),
 'P7_two_part_history':('traffic','two'),
 'P8_class_history':('traffic','cls'),
}
CLASS_THRESHOLDS=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,None)
GAIN_THRESHOLDS=(-40.,-20.,-10.,0.,5.,10.,20.,40.,60.,100.,None)


def classification_target(y):return (np.asarray(y)>1e-7).astype(int)


def forest_classifier(X,y,seed,trees):
    yc=classification_target(y)
    if len(np.unique(yc))<2:
        return None,float(yc.mean())
    m=RandomForestClassifier(n_estimators=trees,min_samples_leaf=5,
                             max_features='sqrt',class_weight='balanced_subsample',
                             random_state=seed,n_jobs=1)
    m.fit(X,yc)
    return m,None


def probabilities(m,constant,X):
    if m is None:return np.full(len(X),constant,dtype=float)
    return m.predict_proba(X)[:,list(m.classes_).index(1)]


def forest_regressor(X,y,seed,trees,leaf=5):
    m=RandomForestRegressor(n_estimators=trees,min_samples_leaf=leaf,
                            max_features='sqrt',random_state=seed,n_jobs=1)
    m.fit(X,y)
    return m


class Predictor:
    """All scores are computed from X only; no test labels are consulted."""
    def __init__(self,kind,seed,trees):self.kind,self.seed,self.trees=kind,seed,trees
    def fit(self,X,y,groups):
        kind=self.kind
        if kind in ('cls','pair','two'):
            self.cls,self.const=forest_classifier(X,y,self.seed,self.trees)
        if kind=='cls':return self
        if kind=='reg':
            self.reg=forest_regressor(X,y,self.seed,self.trees)
        elif kind=='two':
            pos=(y>1e-7)
            if not np.any(pos) or np.all(pos):raise ValueError('Need both positive and nonpositive training actions')
            self.regpos=forest_regressor(X[pos],y[pos],self.seed+11,self.trees,leaf=3)
            self.regneg=forest_regressor(X[~pos],y[~pos],self.seed+19,self.trees,leaf=5)
        elif kind=='pair':
            self.scaler=StandardScaler().fit(X)
            scaled=self.scaler.transform(X)
            states=states_index(groups)
            diff=[];lab=[];weights=[]
            for idx in states.values():
                for k in range(len(idx)):
                    for j in range(k+1,len(idx)):
                        i,z=int(idx[k]),int(idx[j]);g=float(y[i]-y[z])
                        if abs(g)<=1e-6:continue
                        d=scaled[i]-scaled[z]
                        diff.extend((d,-d))
                        lab.extend((int(g>0),int(g<0)))
                        w=math.sqrt(min(100.,abs(g)))
                        weights.extend((w,w))
            if len(set(lab))<2:raise ValueError('Not enough unequal within-state pairs')
            model=LogisticRegression(max_iter=1500,fit_intercept=False,C=1.,solver='lbfgs')
            model.fit(np.asarray(diff),np.asarray(lab),sample_weight=np.asarray(weights))
            self.ranker=model
            self.pairs=len(lab)//2
        else:raise ValueError(kind)
        return self
    def predict(self,X):
        """Return ranking score, gate score, and probability-like classifier score."""
        kind=self.kind
        if kind=='cls':
            p=probabilities(self.cls,self.const,X)
            return p,p,p
        if kind=='reg':
            z=self.reg.predict(X)
            return z,z,z
        if kind=='two':
            p=probabilities(self.cls,self.const,X)
            z=p*self.regpos.predict(X)+(1.-p)*self.regneg.predict(X)
            return z,z,p
        p=probabilities(self.cls,self.const,X)
        z=self.ranker.decision_function(self.scaler.transform(X))
        return z,p,p


def policy(y_primary,y_oracle,scores,gate,groups,threshold):
    """Evaluate ONE selected section per state or no-op, without hindsight clipping.

    Threshold is chosen on validation ONLY. Runtime model inputs contain no y.
    """
    result=[]
    for sid,ix in states_index(groups).items():
        chosen=int(ix[np.argmax(scores[ix])])
        execute=(threshold is not None and float(gate[chosen])>=threshold)
        g=float(y_primary[chosen]) if execute else 0.
        alt=float(y_oracle[chosen]) if execute else 0.
        s=groups[chosen]
        result.append({'state_id':sid,'task_id':s['task_id'],'date':s['date'],
          'scenario_id':s['scenario_id'],'sample_id':s['sample_id'] if execute else '',
          'selected_candidate_id':s['sample_id'],
          'selected_index_in_rows':chosen,'execute':int(execute),
          'score':float(scores[chosen]),'gate_score':float(gate[chosen]),
          'static_repair_td_gain_sec':g,
          'oracle_repair_td_gain_same_section_sec':alt,
          'oracle_best_static_repair_gain_with_noop_sec':max(0.,float(np.max(y_primary[ix]))),
          'oracle_best_td_repair_gain_with_noop_sec':max(0.,float(np.max(y_oracle[ix]))),
          'selected_static_repair_gain_sec':float(y_primary[chosen]),
          'selected_oracle_repair_gain_sec':float(y_oracle[chosen]),
          'top1_oracle_static_repair_match':int(int(ix[np.argmax(y_primary[ix])])==chosen)})
    return result


def policy_summary(records):
    if not records:raise ValueError('No states for evaluation')
    gains=np.asarray([r['static_repair_td_gain_sec'] for r in records],float)
    oracle=np.asarray([r['oracle_best_static_repair_gain_with_noop_sec'] for r in records],float)
    ran=np.asarray([r['oracle_repair_td_gain_same_section_sec'] for r in records],float)
    on=np.asarray([r['execute'] for r in records],bool)
    return {'states':len(records),'executed_states':int(on.sum()),
            'execution_rate':float(on.mean()),'positive_executions':int(np.sum((gains>1e-7)&on)),
            'nonpositive_executions':int(np.sum((gains<=1e-7)&on)),
            'mean_signed_gain_sec':float(gains.mean()),
            'total_signed_gain_sec':float(gains.sum()),
            'mean_oracle_static_repair_noop_bound_sec':float(oracle.mean()),
            'mean_policy_regret_sec':float((oracle-gains).mean()),
            'mean_oracle_td_repair_same_section_sec':float(ran.mean()),
            'mean_oracle_repair_upgrade_on_executed_sec':float((ran-gains).mean()),
            'section_top1_agreement_rate':float(np.mean([r['top1_oracle_static_repair_match'] for r in records]))}


def threshold_grid(kind):return CLASS_THRESHOLDS if kind in ('cls','pair') else GAIN_THRESHOLDS


def select_threshold(y,oray,score,gate,groups,kind):
    """Fixed grid, conservative tie-break; test set cannot affect selection."""
    choices=[]
    for order,thr in enumerate(threshold_grid(kind)):
        ps=policy(y,oray,score,gate,groups,thr)
        summary=policy_summary(ps)
        choices.append((summary['mean_signed_gain_sec'], -summary['executed_states'],order,thr,summary))
    winner=max(choices,key=lambda a:(a[0],a[1],a[2]))
    return winner[3],winner[4]


def ranks(a):
    a=np.asarray(a);sort=np.argsort(a,kind='stable');r=np.empty(len(a),float);j=0
    while j<len(a):
        k=j+1
        while k<len(a) and a[sort[j]]==a[sort[k]]:k+=1
        r[sort[j:k]]=(j+k-1)*.5
        j=k
    return r


def spearman_per_state(score,y,groups):
    values=[]
    for idx in states_index(groups).values():
        if len(idx)<3:continue
        a,b=ranks(score[idx]),ranks(y[idx])
        if np.std(a)<1e-12 or np.std(b)<1e-12:continue
        values.append(float(np.corrcoef(a,b)[0,1]))
    return float(np.mean(values)) if values else None


def ranking_metrics(y,score,groups):
    test_cls=classification_target(y)
    if len(np.unique(test_cls))<2: ap=auc=None
    else:
        ap=float(average_precision_score(test_cls,score))
        auc=float(roc_auc_score(test_cls,score))
    return {'PR_AUC':ap,'ROC_AUC':auc,
            'mean_within_state_spearman':spearman_per_state(score,y,groups)}


def write_csv(path,rows):
    if not rows:raise ValueError('Refusing to write empty CSV')
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def method_report(d,protocol,seed,trees,methods,output):
    masks=group_masks(d,protocol)
    Xall={mode:d.features(mode) for mode in ('core','static','traffic')}
    y=d.label(PRIMARY);oracle=d.label(ORACLE)
    report={'protocol':protocol,'target':PRIMARY,'source':str(d.source),
       'positive_rate_all':float(np.mean(y>1e-7)),
       'splits':{k:{'states':len(states_index(rows_at(d,m))),
                      'candidates':int(m.sum()),'positive_candidates':int(np.sum(y[m]>1e-7)),
                      'task_ids':sorted(set(g['task_id'] for g in rows_at(d,m))),
                      'dates':sorted(set(g['date'] for g in rows_at(d,m)))} for k,m in masks.items()},
       'methods':{},'control':{}}
    outdir=output/protocol;outdir.mkdir(parents=True,exist_ok=False)
    test_mask=masks['internal_test'];test_groups=rows_at(d,test_mask)
    test_y=y[test_mask];test_oracle=oracle[test_mask]
    score_static=d.X[:,d.names.index('static_section_cost_sec')]
    controls={
        'always_noop':(np.zeros(int(test_mask.sum())),np.zeros(int(test_mask.sum())),None),
        'static_score_unconditional':(score_static[test_mask],np.ones(int(test_mask.sum())),-float('inf')),
    }
    for name,(score,gate,thr) in controls.items():
        ps=policy(test_y,test_oracle,score,gate,test_groups,thr)
        report['control'][name]=policy_summary(ps)
        write_csv(outdir/f'{name}_states.csv',ps)
    for method in methods:
        mode,kind=METHODS[method]
        X,names=Xall[mode]
        m=Predictor(kind,seed,trees).fit(X[masks['train']],y[masks['train']],rows_at(d,masks['train']))
        val_score,val_gate,val_pos=m.predict(X[masks['val']])
        thr,valsummary=select_threshold(y[masks['val']],oracle[masks['val']],
                                        val_score,val_gate,rows_at(d,masks['val']),kind)
        tscore,tgate,tprob=m.predict(X[test_mask])
        prec=policy(test_y,test_oracle,tscore,tgate,test_groups,thr)
        write_csv(outdir/f'{method}_test_states.csv',prec)
        write_csv(outdir/f'{method}_validation_states.csv',policy(y[masks['val']],oracle[masks['val']],
                   val_score,val_gate,rows_at(d,masks['val']),thr))
        cand=[]
        for row,score,gate,p,actual,ora in zip(test_groups,tscore,tgate,tprob,test_y,test_oracle):
            cand.append({'sample_id':row['sample_id'],'state_id':row['state_id'],
                 'task_id':row['task_id'],'date':row['date'],
                 'score':float(score),'gate_score':float(gate),'estimated_positive_probability':float(p)
                 if kind in ('cls','pair','two') else '',
                 'static_repair_td_gain_sec':float(actual),
                 'oracle_td_repair_gain_sec':float(ora)})
        write_csv(outdir/f'{method}_test_candidates.csv',cand)
        metrics=ranking_metrics(test_y,tscore,test_groups)
        if kind in ('cls','pair','two'):
            metrics['positive_PR_AUC']=(
                float(average_precision_score(classification_target(test_y),tprob))
                if len(np.unique(classification_target(test_y)))>1 else None)
        if kind in ('reg','two'):
            metrics['signed_regression_MAE_sec']=float(mean_absolute_error(test_y,tscore))
        report['methods'][method]={'features':names,'mode':mode,'kind':kind,
             'threshold_chosen_on_val':thr, 'validation_policy':valsummary,
             'internal_test_policy':policy_summary(prec),'test_metrics':metrics,
             'candidate_predictions_file':f'{protocol}/{method}_test_candidates.csv',
             'state_predictions_file':f'{protocol}/{method}_test_states.csv',
             'training_pairs':int(m.pairs) if kind=='pair' else None}
        print(f"{protocol:4s} {method:27s} val_thr={str(thr):>7s} test_gain={report['methods'][method]['internal_test_policy']['mean_signed_gain_sec']:8.3f} sec/state",flush=True)
    (outdir/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return report


def run(source,output,protocols,seed=42,trees=160,methods=None):
    if Path(output).exists():raise FileExistsError('Result directory already exists; choose a NEW STUDY_NAME')
    d=load(source)
    methods=list(methods or METHODS.keys())
    if len(methods)!=len(set(methods)) or any(m not in METHODS for m in methods):
        raise ValueError('Unknown or repeated methods')
    for p in protocols: group_masks(d,p)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest={'stage':'step6b1-supervised-v1','source':str(d.source),
        'source_sha256':{k:sha256(v) for k,v in d.source_files.items()},
        'code_sha256':{p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('dataset.py'),
                                   Path(__file__).with_name('audit_static_repair.py'))},
        'protocols':list(protocols),'seed':seed,'trees':trees,'methods':methods,
        'primary_target':PRIMARY,'auxiliary_oracle_upper_bound':ORACLE,
        'warnings':['July development: internal task/date tests only, no unseen-month generalization.',
        'The evaluator uses retrospective Oracle TD objective to score STATIC repaired actions.',
        'Scores are offline single-step counterfactuals, not 30-second ALNS or road-level replay.',
        'All repair label generation happened offline; full online feature/repair cost is not included.',
        'Static current-state generation can yield identical route structures in multiple days.',
        'Within-state label differences are not independent examples for significance testing.']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    audit=audit_repair(d)
    (output/'static_repair_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    reports=[method_report(d,p,seed,trees,methods,output) for p in protocols]
    (output/'summary.json').write_text(json.dumps({'status':'COMPLETE','source':str(source),
          'reports':reports},indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print('PASS: Step6-B1 output',output)
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--protocol',choices=('task','date','both'),default='both')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--trees',type=int,default=160)
    parser.add_argument('--methods',default=','.join(METHODS))
    args=parser.parse_args()
    if args.trees<5:parser.error('--trees must be >=5')
    methods=[x.strip() for x in args.methods.split(',') if x.strip()]
    protocols=('task','date') if args.protocol=='both' else (args.protocol,)
    run(args.source_dir,args.output_dir,protocols,args.seed,args.trees,methods)
if __name__=='__main__':main()
