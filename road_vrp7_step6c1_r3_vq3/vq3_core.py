"""R3-VQ3: task-isolated ranking, harm classification and diagnostics.

Harm means the *3-restart mean* road gain is < -40 sec, NOT a probability of
failure for an individual restart. Calibrated numbers are exploratory, not
conformal guarantees. No post-departure road information used as an input.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

STAGE='R3-VQ3-DECISION-RISK-LOTO-v1'
POLICIES=('C3','E0_FULL','E0_MATCHED','VQ_MEAN','RESID_OOF25','RESID_OOF50',
          'PAIR_RANK','PAIR_REGRET','E0_RISK20','E0_RISK35','RESID25_RISK35',
          'PAIR_REGRET_RISK35','E0_SCORE_COVERAGE35')
HARM_CUTOFF=-40.0


def candidate_baseline_feature(x):
    x=np.asarray(x,dtype=float)
    if x.shape!=(78,):raise ValueError('Expected F2 [candidate, C3, candidate-C3] (78D)')
    b=x[26:52]
    return np.concatenate([b,b,np.zeros(26)])


def _normalized_states(records):
    by=defaultdict(list)
    for row in records:
        k=str(row['decision'])
        by[k].append(row)
    return by


@dataclass
class PairRanker:
    scaler:object
    model:object
    training_pairs:int
    training_states:int
    weight_mode:str

    def predict(self,features):
        x=np.asarray(features,float)
        if x.ndim!=2 or x.shape[1]!=78:raise ValueError('Wrong ranker feature dimension')
        # C3 feature is [b,b,0], NOT an all-zero input vector.
        baseline=np.stack([candidate_baseline_feature(v) for v in x])
        v=self.scaler.transform(x-baseline)
        return self.model.decision_function(v)


def train_pairrank(records,*,weighted,seed=42,C=0.2):
    rng=np.random.default_rng(seed)
    samples=[];labels=[];weights=[];nstate=0
    for _,rr in sorted(_normalized_states(records).items()):
        if not rr:continue
        X=np.stack([np.asarray(r['x'],float) for r in rr]);Y=np.array([float(r['mean']) for r in rr])
        base=candidate_baseline_feature(X[0]);XX=np.vstack([base,X]);YY=np.r_[0.,Y]
        diffs=[];gaps=[]
        for i in range(len(YY)):
            for j in range(i+1,len(YY)):
                gap=YY[i]-YY[j]
                if abs(gap)<=1e-8:continue
                diffs.append(XX[i]-XX[j]);gaps.append(gap)
        if not diffs:continue
        nstate+=1
        ws=np.clip(np.sqrt(np.abs(gaps)/40.0),0.25,3.) if weighted else np.ones(len(gaps))
        ws=ws/ws.sum() # Every original STATE carries equal total weight.
        for delta,gap,w in zip(diffs,gaps,ws):
            samples.extend((delta,-delta));labels.extend((int(gap>0),int(gap<0)))
            weights.extend((float(w),float(w)))
    if nstate<3 or not samples:raise ValueError('Insufficient distinct pairwise training states')
    data=np.stack(samples)
    scaler=StandardScaler(with_mean=False).fit(data)
    clf=LogisticRegression(fit_intercept=False,C=C,solver='lbfgs',max_iter=2000,
                           random_state=seed)
    clf.fit(scaler.transform(data),np.asarray(labels),sample_weight=np.asarray(weights))
    return PairRanker(scaler,clf,len(samples)//2,nstate,'regret_sqrt' if weighted else 'uniform')


class HarmPredictor:
    def __init__(self,trees=160,seed=42):
        self.trees=int(trees);self.seed=int(seed)
        self.clf=None;self.constant=None;self.calibrator=None;self.oof=None

    def fit(self,records):
        X=np.stack([r['x'] for r in records]);y=np.array([float(r['mean'])<HARM_CUTOFF for r in records],int)
        w=np.array([r['weight'] for r in records],float)
        if len(y)<12:raise ValueError('Not enough harm rows')
        if len(set(y))==1:self.constant=float(y[0])
        else:
            self.clf=ExtraTreesClassifier(n_estimators=self.trees,max_depth=8,
                 min_samples_leaf=5,max_features=.8,n_jobs=1,random_state=self.seed)
            self.clf.fit(X,y,sample_weight=w)
        return self

    def raw(self,X):
        x=np.asarray(X,float)
        if x.ndim!=2 or x.shape[1]!=78:raise ValueError('Wrong risk features')
        if self.clf is None:return np.full(len(x),self.constant,dtype=float)
        return self.clf.predict_proba(x)[:,int(np.where(self.clf.classes_==1)[0][0])]

    def predict(self,X):
        p=np.clip(self.raw(X),1e-5,1-1e-5)
        if self.calibrator is None:return p
        return self.calibrator.predict_proba(np.log(p/(1-p))[:,None])[:,1]


def crossfit_calibrated_harm(records,train_tasks,*,seed=42,trees=160):
    """Task OOF raw risk scores, then train-task-only Platt scaling.

    Inner risk model for each task sees OTHER 2 outer-training tasks.
    Calibration uses outcomes of the 3 outer-training tasks, never outer holdout.
    """
    tasks=sorted(set(train_tasks));assert len(tasks)==3
    if {str(r['task']) for r in records}!=set(tasks):raise ValueError('Unexpected tasks in risk records')
    p=np.full(len(records),np.nan)
    labels=np.array([float(r['mean'])<HARM_CUTOFF for r in records],int)
    weights=np.array([float(r['weight']) for r in records],float)
    folds=[]
    for k,task in enumerate(tasks):
        a=[r for r in records if r['task']!=task];idx=[i for i,r in enumerate(records) if r['task']==task]
        model=HarmPredictor(trees,seed+1000+k).fit(a)
        p[idx]=model.raw(np.stack([records[i]['x'] for i in idx]))
        folds.append({'oof_task':task,'fit_tasks':[t for t in tasks if t!=task],
                      'n_eval_candidates':len(idx),'n_train_candidates':len(a)})
    if not np.isfinite(p).all():raise AssertionError('Missing task OOF harm scores')
    calibrated=HarmPredictor(trees,seed+5000).fit(records)
    if len(set(labels))>=2:
        p2=np.clip(p,1e-5,1-1e-5);logits=np.log(p2/(1-p2))
        calibrator=LogisticRegression(C=0.1,solver='lbfgs',max_iter=1500,random_state=seed)
        calibrator.fit(logits[:,None],labels,sample_weight=weights)
        calibrated.calibrator=calibrator
        pcal=calibrator.predict_proba(logits[:,None])[:,1]
    else:pcal=p.copy()
    diagnostic=prob_diagnostics(labels,pcal,weights)
    diagnostic.update({'calibration_folds':folds,'n_training_candidates':len(records),
                       'n_harm_candidates':int(labels.sum()),
                       'note':'Platt scaling on training-task OOF raw scores; three training tasks only'})
    oof_map={(r['decision'],int(r['action'])):float(pi) for r,pi in zip(records,pcal)}
    if len(oof_map)!=len(records):raise ValueError('Repeated OOF candidate IDs')
    return calibrated,oof_map,diagnostic


def prob_diagnostics(y,p,weights=None,n_bins=5):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if len(y)!=len(p) or len(y)==0 or not np.isfinite(p).all():raise ValueError('Bad probability diagnostics')
    w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    if np.any(w<=0) or np.sum(w)<=0:raise ValueError('Bad sample weights')
    p=np.clip(p,0.,1.)
    brier=float(np.average((p-y)**2,weights=w))
    ece=0.;bins=[]
    for lo,hi in zip(np.linspace(0,1,n_bins+1)[:-1],np.linspace(0,1,n_bins+1)[1:]):
        m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if np.any(m):
            wm=w[m].sum()/w.sum();mp=float(np.average(p[m],weights=w[m]));my=float(np.average(y[m],weights=w[m]));ece+=wm*abs(mp-my)
            bins.append({'lower':float(lo),'upper':float(hi),'n':int(m.sum()),'pred_mean':mp,'harm_rate':my})
    return {'n':len(y),'harm_prevalence':float(np.average(y,weights=w)),
            'brier':brier,'ece5':float(ece),
            'roc_auc':float(roc_auc_score(y,p,sample_weight=w)) if len(set(y))==2 else None,
            'average_precision':float(average_precision_score(y,p,sample_weight=w)) if len(set(y))==2 else None,
            'bins':bins}


def pairwise_accuracy(states):
    """State-normalized pair accuracy including real C3 0 as a ninth option."""
    per=[]
    for scores,truth,mask in states:
        s=np.asarray(scores,float)[np.asarray(mask,bool)];y=np.asarray(truth,float)[np.asarray(mask,bool)]
        s=np.r_[0.,s];y=np.r_[0.,y]
        good=total=0
        for i in range(len(y)):
            for j in range(i+1,len(y)):
                if abs(y[i]-y[j])<1e-8:continue
                good+=0.5 if s[i]==s[j] else int((s[i]-s[j])*(y[i]-y[j])>0)
                total+=1
        if total:per.append(good/total)
    return float(np.mean(per)) if per else None


def risk_gate(scores,mask,risk,limit,score_threshold=0.):
    s=np.asarray(scores,float);v=np.asarray(mask,bool);p=np.asarray(risk,float)
    if s.shape!=v.shape or s.shape!=p.shape:raise ValueError('Score/risk shape mismatch')
    if not np.isfinite(s).all() or not np.isfinite(p).all():raise ValueError('NaN in score or risk')
    if not v.any():return -1
    j=int(np.argmax(np.where(v,s,-np.inf)))
    return j if s[j]>score_threshold and p[j]<=limit else -1


def coverage_matched_tau(oof_best,eligible_coverage):
    """Score-only threshold matched to risk gate training OOF state coverage.

    This is a diagnostic, NOT calibrated under distribution shift.
    No outer-heldout states/outcomes are ever used here.
    """
    scores=np.asarray(oof_best,float)
    if len(scores)==0 or not np.isfinite(scores).all():raise ValueError('Bad OOF coverage scores')
    target=float(eligible_coverage)
    if not 0<=target<=1:raise ValueError('Coverage outside 0..1')
    if target<=0:return max(0.,float(scores.max()+1.))
    thresholds=[0.,float(scores.max()+1.)]+[float(x) for x in np.unique(scores[scores>0])]
    return float(min(thresholds,key=lambda t:(abs(float(np.mean(scores>t))-target),t)))
