"""Small, cost-sensitive Section advantage estimator relative to actual C3.

Inspired by Learning to Delegate / Learning to Defer / safe baseline fallback.
No claim of original SPIBB or SPO+ guarantees.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


def paired_matrix(actions, baseline):
    a=np.asarray(actions,np.float32);b=np.asarray(baseline,np.float32)
    if a.ndim!=2 or b.shape!=(a.shape[1],):raise ValueError('R2 requires Kxd actions and d baseline features')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite predictors')
    bb=np.broadcast_to(b,a.shape)
    return np.concatenate([a,bb,a-bb],axis=-1).astype(np.float32)


@dataclass
class R2Advantage:
    mode:str='F2'
    seed:int=42
    threshold:float=20.
    min_effective_actions:int=1
    model:object=None

    def fit(self,X,B,Y,BY,valid,same,train_mask):
        dim={'F2':26,'F4':44}.get(self.mode)
        if dim is None:raise ValueError('Only F2 and F4 are supported')
        feats=[];labels=[]
        for i in np.flatnonzero(train_mask):
            available=np.flatnonzero(valid[i]&(~same[i]))
            if len(available)==0:continue
            paired=paired_matrix(X[i,:,:dim],B[i,:dim]);feats.append(paired[available]);
            labels.extend((Y[i,available]-BY[i]).tolist())
        if not feats:raise ValueError('No valid non-C3 candidate advantage labels in training split')
        x=np.concatenate(feats)
        if len(x)<12:raise ValueError('Insufficient paired samples to train R2')
        self.model=ExtraTreesRegressor(n_estimators=160,max_depth=14,min_samples_leaf=5,
                 max_features=.8,bootstrap=False,n_jobs=1,random_state=self.seed)
        self.model.fit(x,np.asarray(labels,np.float64))
        return self

    def predict_delta(self,actions,baseline):
        if self.model is None:raise ValueError('Train model before inference')
        dim={'F2':26,'F4':44}[self.mode]
        xx=np.asarray(actions,np.float32);bb=np.asarray(baseline,np.float32)
        if xx.shape[1]!=dim or bb.shape!=(dim,):raise ValueError('Feature mode mismatch')
        return self.model.predict(paired_matrix(xx,bb)).astype(np.float64)

    def select(self,actions,baseline,same=None,valid=None):
        preds=self.predict_delta(actions,baseline)
        k=len(preds)
        mask=np.ones(k,bool) if valid is None else np.asarray(valid,bool).copy()
        if same is not None:mask &= ~np.asarray(same,bool)
        if not mask.any():return None,float('-inf'),'no_different_candidate'
        j=int(np.argmax(np.where(mask,preds,-np.inf)))
        if float(preds[j])<=float(self.threshold):return None,float(preds[j]),'insufficient_predicted_advantage'
        return j,float(preds[j]),'override_C3'


def evaluate(model,X,B,Y,BY,valid,same,states):
    gains=[];predicted=[];picked=[];nvalid=[]
    for i in states:
        j,score,reason=model.select(X[i,:,:model_dim(model)],B[i,:model_dim(model)],same[i],valid[i])
        gains.append(0. if j is None else float(Y[i,j]-BY[i]));picked.append(j is not None)
        predicted.append(float(score) if np.isfinite(score) else None)
        nvalid.append(int((valid[i]&(~same[i])).sum()))
    g=np.asarray(gains)
    return {'states':len(states),'mean_delta_vs_actual_C3_sec_per_state':float(g.mean()),
      'median_delta_vs_actual_C3_sec_per_state':float(np.median(g)),
      'override_rate':float(np.mean(picked)), 'overrides':int(sum(picked)),
      'effective_action_rate':float(np.mean(np.asarray(nvalid)>0)),
      'positive_override_count':int(((g>1e-8)&np.asarray(picked)).sum()),
      'negative_override_count':int(((g< -1e-8)&np.asarray(picked)).sum()),
      'max_harm_sec':float(max(0.,-g.min())),'mean_harm_tail_gt_40sec':float(np.maximum(-g-40.,0.).mean()),
      'sum_signed_gain_sec':float(g.sum())}


def model_dim(model):return 26 if model.mode=='F2' else 44


def group_split(groups,by='task'):
    """Strictly group-disjoint folds. No per-seed resampling of held-out groups."""
    if by not in ('task','date'):raise ValueError('Use task or date split')
    col='task_id' if by=='task' else 'date'
    keys=sorted({str(z[col]) for z in groups})
    if len(keys)<3:raise ValueError(f'{by} requires >=3 distinct groups; found {len(keys)}')
    # Sorted deterministic partition; never use labels or train random seed for splitting.
    n=len(keys);a=max(1,int(.6*n));b=max(a+1,int(.8*n));b=min(b,n-1)
    roles={key:('train' if j<a else 'validation' if j<b else 'internal_test') for j,key in enumerate(keys)}
    masks={role:np.array([roles[str(z[col])]==role for z in groups],bool) for role in ('train','validation','internal_test')}
    if any(not value.any() for value in masks.values()):raise ValueError('Empty split')
    return masks,roles
