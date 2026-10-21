"""Step6-C1: true raw-edge masked directed encoder + set-wise delegate selection.

Independent SPIN-inspired small implementation. NOT the official SPIN network.
No SPCLT contrastive objective is claimed here. Offline no-op's reward is zero;
in live ALNS a predicted no-op means fallback to C3, not literal zero reward.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from graph_features import GRAPH_DIM,N,T,META,unpack

try:torch.set_num_threads(1)
except RuntimeError:pass

F4_DIM=44
STRUCT_DIM=26
H=24

class DirectedMaskedEncoder(nn.Module):
    """6-slot masked temporal attention + 2-step directed edge attention.

    Direction: A[i,j]=1 iff physical road edge i ends at edge j's start.
    Zero masks never substitute fabricated traffic values.
    """
    def __init__(self,hidden=H):
        super().__init__()
        self.token=nn.Linear(3,hidden)
        self.tkey=nn.Linear(hidden,1)
        self.inp=nn.Linear(hidden+META+1,hidden)
        self.q=nn.Linear(hidden,hidden)
        self.k=nn.Linear(hidden,hidden)
        self.v=nn.Linear(hidden,hidden)
        self.up=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.norm=nn.LayerNorm(hidden)
        self.node_out=nn.Linear(hidden,1)
        self.pool=nn.Linear(hidden,1)
        self.hidden=hidden

    def forward(self,seq,meta,adj,valid):
        # [B,N,T,2]; mask at seq[...,1]; missing values are always zero.
        mask=(seq[...,1]>0.5).float()*valid.unsqueeze(-1)
        delta=seq[...,0]*mask
        time=torch.linspace(-1,1,T,device=seq.device).reshape(1,1,T,1)
        tok=self.token(torch.cat([delta.unsqueeze(-1),mask.unsqueeze(-1),
                                  time.expand_as(delta.unsqueeze(-1))],dim=-1))
        logits=self.tkey(torch.tanh(tok)).squeeze(-1)
        logits=logits.masked_fill(mask==0,-1e4)
        weight=torch.softmax(logits,dim=-1)*mask
        weight=weight/(weight.sum(-1,keepdim=True)+1e-8)
        h=(weight.unsqueeze(-1)*tok).sum(-2)
        coverage=mask.mean(-1,keepdim=True)
        h=torch.relu(self.inp(torch.cat([h,meta,coverage],dim=-1)))
        h=h*valid.unsqueeze(-1)
        # Directed graph attention only over actually connected sampled edges.
        connected=(adj>0.5)&(valid.unsqueeze(-1)>0.5)&(valid.unsqueeze(-2)>0.5)
        for _ in range(2):
            Q=self.q(h);K=self.k(h);V=self.v(h)
            a=(Q@K.transpose(-1,-2))/(self.hidden**0.5)
            a=a.masked_fill(~connected,-1e4)
            w=torch.softmax(a,-1)*connected.float()
            w=w/(w.sum(-1,keepdim=True)+1e-8)
            message=w@V
            h=self.norm(h+self.up(torch.cat([h,message],dim=-1)))
            h=h*valid.unsqueeze(-1)
        # Observed node weighting and explicit zero-data fallback.
        node_weight=coverage.squeeze(-1)*valid
        scores=self.pool(h).squeeze(-1).masked_fill(node_weight==0,-1e4)
        w=torch.softmax(scores,dim=-1)*node_weight
        w=w/(w.sum(-1,keepdim=True)+1e-8)
        pooled=(w.unsqueeze(-1)*h).sum(-2)
        total_coverage=node_weight.sum(-1)/(valid.sum(-1)+1e-8)
        return pooled*total_coverage.unsqueeze(-1),h


class DelegateNet(nn.Module):
    def __init__(self,variant,hidden=H):
        super().__init__()
        if variant not in ('D0','D1','D2T','D2'):raise ValueError(variant)
        self.variant=variant
        self.struct=nn.Sequential(nn.Linear(STRUCT_DIM,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.traffic=nn.Sequential(nn.Linear(F4_DIM-STRUCT_DIM,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.graph=DirectedMaskedEncoder(hidden) if variant in ('D2','D2T') else None
        self.graph_to_context=nn.Linear(hidden,hidden)
        self.attn=nn.MultiheadAttention(hidden,4,batch_first=True)
        self.norm=nn.LayerNorm(hidden)
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,stat,traffic,graph_tuple=None):
        """Inputs [B,K,26], [B,K,18]; output correction [B,K] sec."""
        b,k,_=stat.shape
        h=self.struct(stat)
        if self.variant!='D0':
            # Traffic features cannot modify scores when both observations absent.
            # F4 has c_rem at 10th and c_ins at 11th from traffic start.
            c=torch.clamp(traffic[:,:,10],min=0.,max=1.)
            d=torch.clamp(traffic[:,:,11],min=0.,max=1.)
            q=torch.sqrt(torch.minimum(c,d))
            h=h+self.traffic(traffic)*q.unsqueeze(-1)
        if self.variant in ('D2','D2T'):
            if graph_tuple is None:raise ValueError('Missing raw graph tensor for D2')
            seq,meta,adj,valid=graph_tuple
            graph_repr,_=self.graph(seq.reshape(-1,N,T,2),meta.reshape(-1,N,META),
                                    adj.reshape(-1,N,N),valid.reshape(-1,N))
            # Graph encoder has zero embedding in entirely missing history.
            h=h+self.graph_to_context(graph_repr).reshape(b,k,-1)*(
                    seq[...,1].sum((-1,-2))>0).float().unsqueeze(-1)
        context,_=self.attn(h,h,h,need_weights=False)
        z=self.norm(h+context)
        return 80.*torch.tanh(self.head(z).squeeze(-1)/80.)


class C1Model:
    """Serializable inference interface: predict([F4 | graph_flat]) -> (score,score,None).

    Model is trained for GROUPS of candidates: every predict call must contain
    candidates from the same source state (as does the existing B2-R adapter).
    """
    def __init__(self,anchor,variant,seed=42):
        if variant not in ('D0','D1','D2T','D2'):raise ValueError(variant)
        self.anchor=anchor;self.variant=variant;self.seed=seed
        self.static_scaler=None;self.traffic_scaler=None;self.net=None
        self.feature_names=None

    def prepare(self,X):
        X=np.asarray(X,dtype=np.float64)
        if X.ndim!=2 or X.shape[1]!=F4_DIM or not np.isfinite(X).all():raise ValueError('F4 required')
        self.static_scaler=StandardScaler().fit(X[:,:STRUCT_DIM])
        self.traffic_scaler=StandardScaler().fit(X[:,STRUCT_DIM:])
        torch.manual_seed(self.seed)
        self.net=DelegateNet(self.variant)
        return self

    def tensorize(self,X,G=None):
        X=np.asarray(X,dtype=np.float64)
        if X.ndim!=2 or X.shape[1]!=F4_DIM:raise ValueError('44 F4 features expected')
        if not np.isfinite(X).all():raise ValueError('Nonfinite F4 values')
        s=torch.from_numpy(self.static_scaler.transform(X[:,:STRUCT_DIM]).astype('float32'))
        t=torch.from_numpy(self.traffic_scaler.transform(X[:,STRUCT_DIM:]).astype('float32'))
        # Coverage for gate must use physical original scale (not standardized).
        # transform has modified columns 10,11, so overwrite with original reliability.
        t[:,10]=torch.from_numpy(np.clip(X[:,STRUCT_DIM+10],0,1).astype('float32'))
        t[:,11]=torch.from_numpy(np.clip(X[:,STRUCT_DIM+11],0,1).astype('float32'))
        g=None
        if self.variant in ('D2','D2T'):
            if G is None:raise ValueError('D2 requires extracted graph')
            g=tuple(torch.from_numpy(x.copy()) for x in unpack(G))
            if self.variant=='D2T':
                seq,meta,adj,valid=g
                eye=torch.eye(N,dtype=adj.dtype).unsqueeze(0)
                g=(seq,meta,eye*valid.unsqueeze(-1),valid)
        return s,t,g

    def get_anchor(self,X):
        score,_,_=self.anchor.predict(np.asarray(X)[:,:STRUCT_DIM])
        return np.asarray(score,dtype=np.float32)

    def forward_set(self,X,G=None):
        # [K,44] offline or online, one state; no hidden cross-state mixing.
        s,t,g=self.tensorize(X,G)
        gs=None if g is None else tuple(x.unsqueeze(0) for x in g)
        return self.net(s.unsqueeze(0),t.unsqueeze(0),gs).squeeze(0)

    def predict(self,full):
        z=np.asarray(full,np.float32)
        if z.ndim!=2:raise ValueError('2D batch required')
        if self.variant in ('D2','D2T') and z.shape[1]!=F4_DIM+GRAPH_DIM:raise ValueError('D2 graph payload required')
        if self.variant not in ('D2','D2T') and z.shape[1] not in (F4_DIM,F4_DIM+GRAPH_DIM):raise ValueError('Bad F4 batch')
        x=z[:,:F4_DIM]
        graph=z[:,F4_DIM:] if self.variant in ('D2','D2T') else None
        with torch.inference_mode():
            delta=self.forward_set(x,graph).cpu().numpy()
        scores=self.get_anchor(x)+delta
        return scores,scores,None


def state_loss(pred,y,reg_weight=.25,rank_weight=.25,decision_weight=1.):
    """Signed, unclipped reward evaluation; clipped only within gradients."""
    yy=y.clamp(-300.,300.)
    # Explicit no-op score/reward of zero. Live online no-op uses C3 fallback.
    logits=torch.cat([pred,torch.zeros(1,device=pred.device)])/35.
    rew=torch.cat([yy,torch.zeros(1,device=pred.device)])/100.
    p=torch.softmax(logits,dim=0)
    decision=-(p*rew).sum()
    reg=F.smooth_l1_loss(pred/60.,yy/60.)
    gap=(yy[:,None]-yy[None,:]);pred_gap=(pred[:,None]-pred[None,:])
    active=torch.abs(gap)>2.
    loss_rank=(F.softplus(-torch.sign(gap)*pred_gap/35.)*
               torch.clamp(torch.abs(gap)/50.,max=4.))[active].mean() if active.any() else pred.new_zeros(())
    return reg_weight*reg + rank_weight*loss_rank + decision_weight*decision
