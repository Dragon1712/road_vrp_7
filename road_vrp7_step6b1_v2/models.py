#!/usr/bin/env python3
"""Reproducible Section models. Predictions depend ONLY on numeric feature matrix."""
from __future__ import annotations
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


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
 'P9_pairwise_interaction_static':('static','pair_inter'),
 'P10_pairwise_interaction_history':('traffic','pair_inter'),
 'P11_two_part_calibrated_static':('static','two_cal'),
 'P12_two_part_calibrated_history':('traffic','two_cal'),
}


def cls_target(y):return (np.asarray(y,dtype=float)>1e-7).astype(int)


def fit_classifier(X,y,seed,trees):
    binary=cls_target(y)
    if len(np.unique(binary))<2:return None,float(binary[0])
    clf=RandomForestClassifier(n_estimators=trees,min_samples_leaf=5,max_features='sqrt',
          class_weight='balanced_subsample',random_state=seed,n_jobs=1)
    clf.fit(X,binary)
    return clf,None


def prob(clf,constant,X):
    if clf is None:return np.full(len(X),constant,dtype=float)
    return clf.predict_proba(X)[:,list(clf.classes_).index(1)]


def regressor(X,y,seed,trees,min_leaf=5):
    model=RandomForestRegressor(n_estimators=trees,min_samples_leaf=min_leaf,
          max_features='sqrt',random_state=seed,n_jobs=1)
    model.fit(X,y)
    return model


def interaction_matrix(X,names):
    """Add explicit state × candidate terms; within-state contrasts retain time effects."""
    context=('original_static_cost_sec','departure_hour_local','departure_weekday_local')
    named={n:i for i,n in enumerate(names)}
    if any(k not in named for k in context):raise ValueError('Missing state-context feature for interaction')
    cand=[i for i,n in enumerate(names) if n not in context]
    # Raw products are standardized later, but pre-normalize large context for stability.
    divisors={'original_static_cost_sec':3600.,'departure_hour_local':24.,'departure_weekday_local':6.}
    z=[X]
    for k in context:
        c=X[:,named[k]]/divisors[k]
        z.append(c[:,None]*X[:,cand])
    M=np.column_stack(z)
    if not np.isfinite(M).all():raise ValueError('Non-finite interactions')
    return M


class Predictor:
    def __init__(self,kind,seed,trees,feature_names,protocol):
        self.kind,self.seed,self.trees=kind,seed,trees
        self.feature_names=list(feature_names);self.protocol=protocol
        self.calibration=None;self.calibration_diagnostics={}

    def _fit_platt_oof(self,X,y,groups):
        """Platt calibration on OOF predictions from grouped *TRAINING* folds only.

        No validation/test labels are read. Uses held-out tasks or dates inside train.
        """
        field={'task':'task_id','date':'date'}[self.protocol]
        keys=np.array([row[field] for row in groups]);folds=len(np.unique(keys))
        if folds<2:raise ValueError('Need >=2 training groups for grouped probability calibration')
        kf=GroupKFold(n_splits=min(3,folds))
        oof=np.full(len(y),np.nan,dtype=float)
        for train_idx,val_idx in kf.split(X,cls_target(y),groups=keys):
            clf,cons=fit_classifier(X[train_idx],y[train_idx],self.seed,self.trees)
            oof[val_idx]=prob(clf,cons,X[val_idx])
        if not np.isfinite(oof).all():raise RuntimeError('Incomplete out-of-group predicted probabilities')
        binary=cls_target(y)
        if len(np.unique(binary))<2:raise ValueError('Calibrator requires both outcomes')
        logits=np.log(np.clip(oof,1e-5,1-1e-5)/np.clip(1-oof,1e-5,1-1e-5))
        calibrator=LogisticRegression(solver='lbfgs',max_iter=1000,C=1.0)
        calibrator.fit(logits[:,None],binary)
        self.calibration=calibrator
        self.calibration_diagnostics={'method':'group_oof_platt','n_groups':folds,
             'oof_positive_rate':float(binary.mean()),'oof_raw_brier':float(np.mean((oof-binary)**2)),
             'oof_platt_brier':float(np.mean((calibrator.predict_proba(logits[:,None])[:,1]-binary)**2))}

    def _calibrate(self,p):
        if self.calibration is None:return p
        logits=np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1-1e-5))
        return self.calibration.predict_proba(logits[:,None])[:,1]

    def fit(self,X,y,groups):
        kind=self.kind
        X=np.asarray(X,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
        if len(X)!=len(y) or len(groups)!=len(y):raise ValueError('Training rows misaligned')
        if not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Training NaN')
        if kind in ('cls','pair','pair_inter','two','two_cal'):
            self.cls,self.const=fit_classifier(X,y,self.seed,self.trees)
        if kind=='cls':return self
        if kind=='reg':
            self.reg=regressor(X,y,self.seed,self.trees)
        elif kind in ('two','two_cal'):
            positive=y>1e-7
            if not np.any(positive) or np.all(positive):raise ValueError('Two part needs both classes')
            if kind=='two_cal':self._fit_platt_oof(X,y,groups)
            self.regpos=regressor(X[positive],y[positive],self.seed+11,self.trees,3)
            self.regnon=regressor(X[~positive],y[~positive],self.seed+19,self.trees,5)
        elif kind in ('pair','pair_inter'):
            rankX=interaction_matrix(X,self.feature_names) if kind=='pair_inter' else X
            self.ranker_scaler=StandardScaler().fit(rankX)
            rankX=self.ranker_scaler.transform(rankX)
            from dataset import grouped_indices
            diffs=[];labels=[];weights=[];n_states=0
            for idx in grouped_indices(groups).values():
                pairs=[]
                for u in range(len(idx)):
                    for v in range(u+1,len(idx)):
                        i,j=int(idx[u]),int(idx[v]);gap=float(y[i]-y[j])
                        if abs(gap)<1e-6:continue
                        pairs.append((i,j,gap))
                if not pairs:continue
                n_states+=1
                # Equalize each source state's total sample weight.
                raw=np.array([math.sqrt(min(100.,abs(d))) for _,_,d in pairs])
                if kind=='pair_inter':
                    # Normalize per state only in the NEW interaction treatment.
                    # P3/P6 retain original v1 pair-weight semantics.
                    raw/=max(raw.sum(),1e-12)
                for (i,j,gap),weight in zip(pairs,raw):
                    d=rankX[i]-rankX[j]
                    diffs.extend((d,-d));labels.extend((int(gap>0),int(gap<0)))
                    weights.extend((float(weight),float(weight)))
            if not diffs:raise ValueError('Pairwise training has no non-tied actions')
            self.ranker=LogisticRegression(fit_intercept=False,C=1.,max_iter=1500,solver='lbfgs')
            self.ranker.fit(np.asarray(diffs),np.asarray(labels),sample_weight=np.asarray(weights))
            self.training_pairs=len(labels)//2;self.training_pair_states=n_states
        else:raise ValueError(kind)
        return self

    def predict(self,X):
        X=np.asarray(X,dtype=np.float64)
        if X.ndim!=2 or X.shape[1]!=len(self.feature_names):raise ValueError('Prediction schema differs from training')
        if self.kind=='cls':
            p=prob(self.cls,self.const,X);return p,p,p
        if self.kind=='reg':
            z=self.reg.predict(X);return z,z,None
        if self.kind in ('two','two_cal'):
            p=self._calibrate(prob(self.cls,self.const,X))
            z=p*self.regpos.predict(X)+(1.-p)*self.regnon.predict(X)
            return z,z,p
        p=prob(self.cls,self.const,X)
        rankX=interaction_matrix(X,self.feature_names) if self.kind=='pair_inter' else X
        rank=self.ranker.decision_function(self.ranker_scaler.transform(rankX))
        return rank,p,p
