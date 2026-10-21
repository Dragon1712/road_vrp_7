"""Step6-C1-R: reproducible baseline-aware, reject-option Section selection.

Inspired by learning-to-defer (cost-sensitive choice) and SPIBB (baseline
fallback), *not* faithful copies or theoretical guarantees of their algorithms.
The training proxy is frozen M0 within the 8 labeled candidates; online rejected
proposals actually fall back to frozen C3. These are explicitly different.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from c1_model import C1Model, state_loss as original_state_loss
from graph_features import GRAPH_DIM

F2_DIM = 26
F4_DIM = 44


def m0_proxy_loss(pred, reward, anchor, anchor_threshold, decision_weight=1.0):
    """Cost-sensitive defer surrogate: add M0 action as *fixed* fallback logit.

    Only called for training states. Oracle is in reward but never in predictor.
    M0 fallback is the same-state top static RF candidate, or zero when gated.
    No claims about the actual C3 fallback reward are made here.
    """
    if pred.ndim != 1 or reward.shape != pred.shape or anchor.shape != pred.shape:
        raise ValueError('Expected one state of K candidate predictions, rewards, M0 anchor scores')
    b = int(torch.argmax(anchor).item())
    base_value = reward[b] if float(anchor[b]) >= float(anchor_threshold) else reward.new_zeros(())
    logits = torch.cat([pred, anchor[b:b+1].detach()]) / 35.
    target = torch.cat([reward, base_value.reshape(1)]).clamp(-300., 300.) / 100.
    decision = -(torch.softmax(logits, dim=0) * target).sum()
    # Keep the original regression + within-state weighted ranking losses;
    # preserve the original state_loss scale / hyperparameters exactly.
    prior = original_state_loss(pred,reward,reg_weight=.25,rank_weight=.25,decision_weight=0.)
    return prior + float(decision_weight)*decision


def selected_action(scores, anchor, margin=0., max_correction=1e9, min_coverage=0., coverage=None):
    """Choose an action only when policy's predicted advantage is credible.

    This is an empirical selective-decision rule, NOT an uncertainty confidence
    interval. Returns (selected index, accepted boolean, reason).
    The default action in the *online* adapter is frozen C3, not M0.
    """
    scores=np.asarray(scores,dtype=float);anchor=np.asarray(anchor,dtype=float)
    if scores.ndim!=1 or scores.shape!=anchor.shape or not np.isfinite(scores).all() or not np.isfinite(anchor).all():
        raise ValueError('Nonfinite/mismatched score arrays')
    if not len(scores):raise ValueError('Empty Section set')
    if margin<0 or max_correction<=0 or min_coverage<0 or min_coverage>1:raise ValueError('Bad gate config')
    a=int(np.argmax(scores));b=int(np.argmax(anchor))
    if scores[a]<0:return a,False,'score_below_zero'
    if a==b:return a,True,'same_as_static'
    if scores[a]-scores[b]<margin:return a,False,'insufficient_rank_margin'
    if scores[a]-anchor[b]<margin:return a,False,'insufficient_static_margin'
    if abs(scores[a]-anchor[a])>max_correction:return a,False,'large_residual'
    if coverage is not None and float(coverage[a])<min_coverage:return a,False,'sparse_traffic'
    return a,True,'passed'


class ReliableC1(C1Model):
    """Pickleable C1 subclass with policy-time reject option.

    Offline can use [K,44] or [K,44+984]; D0 *online* uses [K,26] to
    avoid history and F4 feature extraction entirely.
    """
    def __init__(self,anchor,variant,seed=42):
        super().__init__(anchor,variant,seed)
        self.safe_gate=False
        self.margin=0.
        self.max_correction=1.e9
        self.min_coverage=0.
        self.train_anchor_mode='full'
        self.loss_mode='zero'

    def score_components(self,full):
        arr=np.asarray(full,dtype=np.float32)
        if arr.ndim!=2 or arr.shape[1] not in (F2_DIM,F4_DIM,F4_DIM+GRAPH_DIM):
            raise ValueError('Unexpected Section features')
        if arr.shape[1]==F2_DIM:
            if self.variant!='D0':raise ValueError('F2 only allowed for D0')
            # Traffic is never accessed for D0, so this static-only sidecar is safe.
            arr=np.column_stack((arr,np.zeros((len(arr),F4_DIM-F2_DIM),dtype=np.float32)))
        if self.variant in ('D2','D2T') and arr.shape[1]!=F4_DIM+GRAPH_DIM:
            raise ValueError('Graph variant requires raw edge sidecar')
        if self.variant in ('D0','D1') and arr.shape[1]==F4_DIM+GRAPH_DIM:
            raise ValueError('Unnecessary graph sidecar for tabular variant')
        x=arr[:,:44]
        g=arr[:,44:] if self.variant in ('D2','D2T') else None
        with torch.inference_mode():delta=self.forward_set(x,g).detach().cpu().numpy().astype(np.float64)
        static=self.get_anchor(x).astype(np.float64)
        return static+delta,static

    def predict(self,full):
        score,base=self.score_components(full)
        if not self.safe_gate:
            return score,score,None
        full=np.asarray(full)
        coverage=None
        if self.variant!='D0':
            if full.shape[1]<38:raise ValueError('Missing original reliability features')
            coverage=np.minimum(np.clip(full[:,36],0,1),np.clip(full[:,37],0,1))
        chosen,accepted,_=selected_action(score,base,self.margin,self.max_correction,
                                              self.min_coverage,coverage)
        gate=np.full_like(score,-1.e9)
        if accepted:gate[chosen]=score[chosen]
        # B2-R chooses argmax(score), then checks gate[chosen] >= threshold=0.
        # Rejection therefore invokes its original C3 fallback path.
        return score,gate,None


def evaluate_m0_proxy(model,X,G,groups,y,anchor_threshold=0.):
    """Retrospective M0-proxy fallback metric; NEVER call this 'C3 performance'.

    score/gates inferred once per 8 candidates; baseline M0 is separately
    calculated from frozen RF, independent of any trained correction.
    """
    import numpy as np
    total=[];baseline=[];delegated=[];reasons=[]
    ids={}
    for i,g in enumerate(groups):ids.setdefault(g['state_id'],[]).append(i)
    for inds in ids.values():
        if len(inds)!=8:raise ValueError('Exactly 8 candidates required')
        xx=X[inds];gg=G[inds]
        full=np.concatenate((xx,gg),axis=1) if model.variant in ('D2','D2T') else xx
        s,b=model.score_components(full)
        a=int(np.argmax(s));j=int(np.argmax(b))
        base_value=float(y[inds[j]]) if b[j]>=anchor_threshold else 0.
        cover=np.minimum(np.clip(xx[:,36],0,1),np.clip(xx[:,37],0,1)) if model.variant!='D0' else None
        if model.safe_gate:
            a,accepted,reason=selected_action(s,b,model.margin,model.max_correction,model.min_coverage,cover)
        else:
            accepted=s[a]>=0
            reason='no_gate' if accepted else 'score_below_zero'
        r=float(y[inds[a]]) if accepted else base_value
        total.append(r);baseline.append(base_value);delegated.append(int(accepted));reasons.append(reason)
    return {'states':len(total),'mean_signed_gain_sec_per_state':float(np.mean(total)),
       'mean_m0_proxy_gain_sec_per_state':float(np.mean(baseline)),
       'mean_improvement_vs_m0_proxy_sec_per_state':float(np.mean(np.asarray(total)-baseline)),
       'delegation_rate':float(np.mean(delegated)),
       'reasons':{k:reasons.count(k) for k in sorted(set(reasons))},
       'warning':'Rejected action substitutes M0 candidate/zero OFFLINE; online B2-R rejects to ORIGINAL C3.'}
