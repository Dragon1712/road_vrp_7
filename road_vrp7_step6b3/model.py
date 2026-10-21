"""Independent tabular FiLM-inspired traffic residual models.

Inspiration: Perez et al. (AAAI 2018), official source in
literature/code/film-master/vr/models/{film_gen,filmed_net}.py.
The original model/code is NOT copied here (original license CC BY-NC 4.0).

SPIN (NeurIPS 2022) motivates explicit observation masks and a safe
zero-traffic fallback; this is not a reproduction of SPIN GNN layers.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass

from typing import Optional
try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

try:
    torch.set_num_threads(1)
except (AttributeError, RuntimeError):
    pass


# Names are stable and must be equal to official B2 FEATURE_MODES['F4'].
STRUCTURE = ('original_static_cost_sec','departure_hour_local','departure_weekday_local',
      'route_index','position','size','block_demand','route_load','vehicle_capacity',
      'static_section_cost_sec','route_length','relative_position',
      'remove_static_saved_sec','insert_proxy_static_added_sec','proxy_static_net_gain_sec',
      'target_route_index','target_position','target_route_length','target_route_load',
      'target_route_slack','same_route','source_arrival_static_sec',
      'target_arrival_static_sec','arrival_offset_static_sec',
      'target_orientation_reversed','orientation_internal_delta_static_sec')
TRAFFIC = ('remove_history_saved_sec','insert_history_added_sec',
     'proxy_history_net_gain_sec','removal_boundary_rel_cost','insertion_boundary_rel_cost',
     'removal_history_latest_rel','insertion_history_latest_rel',
     'removal_history_trend','insertion_history_trend',
     'orientation_internal_delta_history_sec')
RELIABILITY = ('removal_observed_share','insertion_observed_share',
     'removal_observed_fraction','insertion_observed_fraction',
     'removal_all_missing','insertion_all_missing',
     'boundary_observed_min','boundary_observed_gap')
FEATURES = STRUCTURE + TRAFFIC + RELIABILITY


def from_full(X, names=FEATURES):
    """Derive ratio-normalized traffic *increments*, no target/Oracle access.

    Returns normalized structure and traffic arrays plus SPIN-inspired q.
    q == 0 if neither boundary has historical observation, q in [0,1].
    """
    X=np.asarray(X,np.float64)
    if X.ndim!=2 or X.shape[1]!=len(names):raise ValueError('B3 requires full F4 feature matrix')
    if len(names)!=len(set(names)) or set(names)!=set(FEATURES):raise ValueError('Unknown or repeated F4 feature names')
    if not np.isfinite(X).all():raise ValueError('Nonfinite model feature')
    n={k:X[:,i] for i,k in enumerate(names)}
    xs=np.stack([n[k] for k in STRUCTURE],axis=1)
    rem=np.abs(n['remove_static_saved_sec'])
    ins=np.abs(n['insert_proxy_static_added_sec'])
    norm=np.maximum(30.,rem+ins)
    # Signed net-proxy traffic delta and separated removal/insertion increments.
    delta=np.stack([
       (n['proxy_history_net_gain_sec']-n['proxy_static_net_gain_sec'])/norm,
       (n['remove_history_saved_sec']-n['remove_static_saved_sec'])/norm,
       (n['insert_history_added_sec']-n['insert_proxy_static_added_sec'])/norm,
       n['removal_boundary_rel_cost']-1.,
       n['insertion_boundary_rel_cost']-1.,
       n['removal_history_latest_rel']-1.,
       n['insertion_history_latest_rel']-1.,
       n['removal_history_trend']/norm,
       n['insertion_history_trend']/norm,
       (n['orientation_internal_delta_history_sec']-
         n['orientation_internal_delta_static_sec'])/norm,
    ],axis=1)
    # Robust outlier clipping: shared, fixed numerical constants; no test fit.
    delta=np.clip(delta,-5.,5.)
    masks=np.stack([n[k] for k in RELIABILITY],axis=1)
    c_rem=np.clip(n['removal_observed_share'],0.,1.)
    c_ins=np.clip(n['insertion_observed_share'],0.,1.)
    q=np.sqrt(np.minimum(c_rem,c_ins))
    # Critical fallback invariant: when observation coverage is zero, q=0.
    q=np.where((c_rem<=0)|(c_ins<=0),0.,q)
    return xs,delta,masks,q


def group_oof_anchor(X,y,groups,protocol,seed=42,trees=120):
    """OOF static anchor to avoid supervising traffic residual on in-sample RF errors.

    Only train rows are passed; fold entities are task or date. Full frozen
    baseline from B2 is used at inference, separately verified by caller.
    """
    X=np.asarray(X,np.float64);y=np.asarray(y,np.float64)
    field={'task':'task_id','date':'date'}[protocol]
    keys=np.array([g[field] for g in groups])
    n=len(np.unique(keys))
    if n<2:raise ValueError('OOF residual requires >=2 training groups')
    cv=GroupKFold(n_splits=min(3,n))
    oof=np.full(len(y),np.nan)
    for fit,hold in cv.split(X,y,keys):
        m=RandomForestRegressor(n_estimators=trees,random_state=seed,n_jobs=1,
                  max_features='sqrt',min_samples_leaf=5)
        m.fit(X[fit],y[fit]);oof[hold]=m.predict(X[hold])
    if not np.isfinite(oof).all():raise ValueError('OOF incomplete')
    return oof


if nn is not None:
    class TinyFiLMResidual(nn.Module):
        """Structure features modulated by traffic context: h*(1+gamma)+beta.

        This is an independent 1-D tabular adaptation of the FiLM *operation*,
        not the officially published visual/CLEVR FiLMedNet architecture.
        """
        def __init__(self,n_struct=26,n_traffic=18,hidden=24):
            super().__init__()
            self.s_net=nn.Sequential(nn.Linear(n_struct,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
            self.condition=nn.Sequential(nn.Linear(n_traffic,hidden),nn.ReLU(),nn.Linear(hidden,2*hidden))
            self.head=nn.Sequential(nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Linear(hidden//2,1))
            # FiLM starts as structural identity and residual starts at zero.
            nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        def forward(self,s,t):
            h=self.s_net(s)
            gamma,beta=self.condition(t).chunk(2,dim=-1)
            return self.head(torch.relu(h*(1.+gamma)+beta)).squeeze(-1)


class ResidualFusionPredictor:
    """Pickleable B2-R-compatible model: predict(X)->(score,gate,None).

    X input is EXACT official B2 F4 (44 dims), from offline data or B2-R online.
    No Oracle/label columns are ever present in X.
    """
    def __init__(self,anchor,method,feature_names=FEATURES,seed=42):
        if method not in ('M0','M1','M2','M3','M4'):raise ValueError('Unknown B3 method')
        self.anchor=anchor;self.method=method;self.feature_names=tuple(feature_names)
        self.seed=seed;self.alpha=0.
        self.residual=None;self.struct_scaler=None;self.traffic_scaler=None;self.network=None

    def _anchor(self,X):
        names=list(self.feature_names)
        xx=np.stack([X[:,names.index(n)] for n in STRUCTURE],axis=1)
        z,gate,_=self.anchor.predict(xx)
        return np.asarray(z,np.float64)

    def prepare(self, trainX, train_residual, epochs=55, hidden=24, batch=128,
                learning_rate=0.002, dropout_p=0.25, consistency_weight=0.05):
        if self.method=='M0':return self
        xs,delta,mask,q=from_full(trainX,self.feature_names)
        if self.method in ('M1','M2'):
            # Pure traffic increments (including reliability input). M2 multiplies by q.
            feat=np.column_stack([delta,mask])
            self.residual=make_pipeline(StandardScaler(),Ridge(alpha=30.0))
            self.residual.fit(feat,train_residual)
            return self
        if torch is None:raise ImportError('torch is required for M3/M4 FiLM')
        torch.manual_seed(self.seed)
        self.struct_scaler=StandardScaler().fit(xs)
        self.traffic_scaler=StandardScaler().fit(np.column_stack([delta,mask]))
        s=self.struct_scaler.transform(xs).astype(np.float32)
        t=self.traffic_scaler.transform(np.column_stack([delta,mask])).astype(np.float32)
        # Fixed target scaling: seconds -> roughly unit scale, fitted only on train.
        self.target_scale=max(10.,float(np.std(train_residual)))
        self.network=TinyFiLMResidual(len(STRUCTURE),len(TRAFFIC)+len(RELIABILITY),hidden)
        optim=torch.optim.AdamW(self.network.parameters(),lr=learning_rate,weight_decay=0.015)
        n=len(s);rng=np.random.default_rng(self.seed)
        qs=torch.from_numpy(q.astype(np.float32))
        a=torch.from_numpy(s);b=torch.from_numpy(t)
        yy=torch.from_numpy(np.asarray(train_residual,np.float32)/self.target_scale)
        self.network.train()
        for epoch in range(int(epochs)):
            for inds in np.array_split(rng.permutation(n),max(1,int(np.ceil(n/batch)))):
                ss,tt,qq,target=a[inds],b[inds],qs[inds],yy[inds]
                value=self.network(ss,tt)*qq
                loss=torch.nn.functional.smooth_l1_loss(value,target,beta=.7)
                if self.method=='M4':
                    # Modality dropout: masks and values disappear TOGETHER.
                    keep=(torch.rand(len(tt),1)>=dropout_p).to(tt.dtype)
                    aug_t=tt*keep
                    aug_q=qq*keep.squeeze(-1)
                    with_drop=self.network(ss,aug_t)*aug_q
                    loss=loss+consistency_weight*torch.mean((value-with_drop)**2)
                optim.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),2.)
                optim.step()
        self.network.eval()
        return self

    def correction(self,X):
        X=np.asarray(X,np.float64)
        if self.method=='M0':return np.zeros(len(X))
        xs,delta,mask,q=from_full(X,self.feature_names)
        if self.method in ('M1','M2'):
            pred=self.residual.predict(np.column_stack([delta,mask]))
            return pred if self.method=='M1' else pred*q
        ss=self.struct_scaler.transform(xs).astype(np.float32)
        tt=self.traffic_scaler.transform(np.column_stack([delta,mask])).astype(np.float32)
        with torch.inference_mode():pred=self.network(torch.from_numpy(ss),torch.from_numpy(tt)).numpy()
        return pred*self.target_scale*q

    def predict(self,X):
        X=np.asarray(X,np.float64)
        base=self._anchor(X)
        score=base+self.alpha*self.correction(X)
        # P2 original: gate == estimated signed utility. No label in prediction.
        return score,score,None
