"""R3 first-batch decision tricks: CatBoost RMSE/Huber, pairwise RankNet,
regret-weighted RankNet, and selective margin gate.

All candidate labels are signed one-step TD advantage relative to the *actual*
frozen C3 decision. Offline labels only; online predictors receive no TD oracle.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import numpy as np
from sklearn.preprocessing import StandardScaler
from r2_policy import R2Advantage, paired_matrix

ARMS=('T1_CB_RMSE','T2_CB_HUBER','T3_PAIRWISE','T4_WEIGHTED_PAIR',
      'T5_HUBER_MARGIN','T6_WEIGHTED_MARGIN','T7_BLEND_MARGIN')
TRAINED=('T1_CB_RMSE','T2_CB_HUBER','T3_PAIRWISE','T4_WEIGHTED_PAIR')


def prepare_group(X,B,Y,BY,valid,same,idx,dim=26):
    """A state has 8 candidates, C3 baseline added as action 9 with gain 0."""
    import numpy as np
    rows=[]; labels=[]; mask=[]
    for i in idx:
        feat=paired_matrix(X[i,:,:dim],B[i,:dim])
        bb=paired_matrix(B[i,:dim][None,:],B[i,:dim])
        rows.append(np.concatenate([feat,bb]))
        labels.append(np.r_[Y[i]-BY[i],0.].astype(np.float32))
        mask.append(np.r_[valid[i]&(~same[i]),True])
    if not rows:
        return (np.zeros((0,9,dim*3),np.float32),np.zeros((0,9),np.float32),np.zeros((0,9),bool))
    return np.asarray(rows,np.float32),np.asarray(labels,np.float32),np.asarray(mask,bool)


def _training_points(X,B,Y,BY,valid,same,train_mask):
    x,y,m=prepare_group(X,B,Y,BY,valid,same,np.flatnonzero(train_mask))
    # CatBoost regressors train on valid DIFFERENT actions, not zero-labelled C3 baseline.
    ok=m[:,:8]
    f=x[:,:8][ok];lab=y[:,:8][ok]
    if len(f)<12:raise ValueError('Insufficient valid non-C3 labels')
    return f,lab


class PairNet:
    """Small network kept as a picklable class with deterministic CPU inference."""
    def __init__(self,dim=78,seed=42):
        import torch
        torch.manual_seed(seed)
        self.net=torch.nn.Sequential(torch.nn.Linear(dim,64),torch.nn.ReLU(),
                  torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1)).cpu()

    def __call__(self,x):return self.net(x).squeeze(-1)


def fit_ranker(X,B,Y,BY,valid,same,train_mask,seed=42,weighted=False,epochs=100):
    import torch
    import torch.nn.functional as F
    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rows,y,mask=prepare_group(X,B,Y,BY,valid,same,np.flatnonzero(train_mask))
        n=len(rows)
        if n<8:raise ValueError('RankNet needs >=8 training states')
        scaler=StandardScaler().fit(rows[mask])
        xx=np.clip(scaler.transform(rows.reshape(-1,78)).reshape(rows.shape),-10,10)
        x=torch.from_numpy(xx.astype(np.float32));target=torch.from_numpy(y)
        mask_t=torch.from_numpy(mask)
        torch.manual_seed(seed)
        net=PairNet(78,seed=seed)
        optim=torch.optim.AdamW(net.net.parameters(),lr=0.0015,weight_decay=.002)
        g=torch.Generator().manual_seed(seed)
        ii,jj=torch.triu_indices(9,9,offset=1)
        history=[]
        for epoch in range(epochs):
            net.net.train();total=0.;seen=0
            for ids in torch.randperm(n,generator=g).split(48):
                a=x[ids];label=target[ids];validity=mask_t[ids]
                # Anchor each state's actual C3 action to precisely zero.
                raw=net(a);pred=raw-raw[:,8:9]
                dif=label[:,ii]-label[:,jj]
                ok=validity[:,ii]&validity[:,jj]&(torch.abs(dif)>1e-6)
                if not ok.any():continue
                sign=torch.sign(dif)
                logloss=F.softplus(-sign*(pred[:,ii]-pred[:,jj])/40.)
                if weighted:
                    # Weighted per state, not treating up to 36 pairs as independent cases.
                    w=torch.clamp(torch.abs(dif)/40.,.5,3.)
                else:w=torch.ones_like(dif)
                numerator=(logloss*w*ok).sum(-1)
                denominator=(w*ok).sum(-1).clamp(min=1)
                rank=(numerator/denominator).mean()
                # Small calibrated regression auxiliary, so zero is meaningful at deployment.
                sample_ok=validity[:,:8]
                target_normal=torch.clamp(label[:,:8]/40.,-10,10)
                err=F.smooth_l1_loss(pred[:,:8]/40.,target_normal,reduction='none')
                reg=(err*sample_ok).sum()/sample_ok.sum().clamp(min=1)
                loss=rank+.4*reg
                optim.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(net.net.parameters(),2.)
                optim.step();total+=float(loss.detach())*len(ids);seen+=len(ids)
            history.append(total/max(1,seen))
        net.net.eval()
        for p in net.net.parameters():p.requires_grad_(False)
        return net,scaler,history
    finally:
        torch.set_num_threads(old_threads)


@dataclass
class R3Advantage(R2Advantage):
    algorithm:str='catboost_rmse'
    margin:float=0.
    scaler:object=None
    components:list=field(default_factory=list)
    rank_history:list=field(default_factory=list)

    def fit(self,X,B,Y,BY,valid,same,train_mask,epochs=100):
        if self.mode!='F2':raise ValueError('First-batch R3 allows F2 only')
        if self.algorithm in ('catboost_rmse','catboost_huber'):
            from catboost import CatBoostRegressor
            f,y=_training_points(X,B,Y,BY,valid,same,train_mask)
            kind='RMSE' if self.algorithm=='catboost_rmse' else 'Huber:delta=40'
            self.model=CatBoostRegressor(iterations=360,depth=5,learning_rate=.04,
                 l2_leaf_reg=6,loss_function=kind,random_seed=self.seed,
                 thread_count=2,verbose=False,allow_writing_files=False)
            self.model.fit(f,y)
        elif self.algorithm in ('pairwise','weighted_pairwise'):
            self.model,self.scaler,self.rank_history=fit_ranker(
                X,B,Y,BY,valid,same,train_mask,self.seed,
                weighted=(self.algorithm=='weighted_pairwise'),epochs=epochs)
        else:raise ValueError('Cannot directly fit '+self.algorithm)
        return self

    def predict_delta(self,actions,baseline):
        if self.mode!='F2':raise ValueError('R3 supports F2 only')
        x=np.asarray(actions,np.float32);b=np.asarray(baseline,np.float32)
        if x.ndim!=2 or x.shape[1]!=26 or b.shape!=(26,):raise ValueError('F2 input mismatch')
        if self.algorithm=='blend':
            if len(self.components)!=2:raise ValueError('Need exactly two blend components')
            return .5*self.components[0].predict_delta(x,b)+.5*self.components[1].predict_delta(x,b)
        m=paired_matrix(x,b)
        if self.algorithm in ('catboost_rmse','catboost_huber'):
            return np.asarray(self.model.predict(m),np.float64)
        if self.algorithm in ('pairwise','weighted_pairwise'):
            import torch
            xx=np.vstack([m,paired_matrix(b[None,:],b)])
            transformed=np.clip(self.scaler.transform(xx),-10.,10.)
            with torch.no_grad():
                result=self.model(torch.from_numpy(transformed.astype(np.float32))).cpu().numpy()
            return (result[:-1]-result[-1]).astype(np.float64)
        raise ValueError('Unknown predictor '+self.algorithm)

    def select(self,actions,baseline,same=None,valid=None):
        pred=self.predict_delta(actions,baseline)
        k=len(pred);mask=np.ones(k,bool) if valid is None else np.asarray(valid,bool).copy()
        if same is not None:mask &= ~np.asarray(same,bool)
        if not mask.any():return None,float('-inf'),'no_different_candidate'
        masked=np.where(mask,pred,-np.inf)
        j=int(np.argmax(masked));score=float(pred[j])
        if not np.isfinite(score) or score<=float(self.threshold):
            return None,score,'insufficient_predicted_advantage'
        # Margin compares against runner-up candidate AND C3's zero anchor.
        runner_up=max(0.,float(np.max(np.delete(masked,j))) if k>1 else 0.)
        if score-runner_up<=float(self.margin):return None,score,'insufficient_rank_margin'
        return j,score,'override_C3'


def make_blend(huber,weighted,seed=42):
    if huber.mode!='F2' or weighted.mode!='F2':raise ValueError('F2 only')
    return R3Advantage(mode='F2',seed=seed,algorithm='blend',components=[copy.deepcopy(huber),copy.deepcopy(weighted)])
