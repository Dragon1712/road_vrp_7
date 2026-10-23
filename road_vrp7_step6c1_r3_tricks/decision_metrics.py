"""State-level R2/R3 decision auditing. Invalid and same-C3 actions never selected."""
from __future__ import annotations
import numpy as np


def evaluate(model,X,B,Y,BY,valid,same,indices,harm_limit=40.):
    n=len(indices)
    if n<1:raise ValueError('Empty evaluation fold')
    regrets=[];gains=[];oracle=[];picked=[];harm=[];hit=[];pa=[];pr=[]
    for i in indices:
        idx=np.flatnonzero(valid[i]&(~same[i]));d=model.mode
        dim=26 if d=='F2' else 44
        # Use the actual selector interface, including baseline defer and margin.
        j,score,reason=model.select(X[i,:,:dim],B[i,:dim],same[i],valid[i])
        gain=0. if j is None else float(Y[i,j]-BY[i]);best=max([0.]+[float(Y[i,k]-BY[i]) for k in idx])
        gains.append(gain);oracle.append(best);regrets.append(max(0.,best-gain))
        picked.append(j is not None);harm.append(j is not None and gain < -harm_limit)
        hit.append(best-gain<1e-5)
        # Pairwise accuracy among all valid different candidates and C3.
        if len(idx):
            pred=model.predict_delta(X[i,:,:dim],B[i,:dim]);truth=np.r_[Y[i,idx]-BY[i],0.]
            scores=np.r_[pred[idx],0.];correct=[]
            for a in range(len(truth)):
                for b in range(a+1,len(truth)):
                    if abs(float(truth[a]-truth[b]))>1e-6:
                        correct.append(float((truth[a]-truth[b])*(scores[a]-scores[b])>0))
            if correct:pa.append(float(np.mean(correct)))
        pr.append(float(score) if np.isfinite(score) else None)
    g=np.array(gains);o=np.array(oracle);r=np.array(regrets);h=np.array(harm);p=np.array(picked)
    return {'states':n,'mean_signed_advantage_sec':float(g.mean()),
      'mean_oracle_upper_bound_sec':float(o.mean()),'mean_regret_sec':float(r.mean()),
      'median_regret_sec':float(np.median(r)),'top1_with_C3_accuracy':float(np.mean(hit)),
      'pairwise_accuracy':float(np.mean(pa)) if pa else None,
      'override_rate':float(p.mean()),'overrides':int(p.sum()),
      'positive_overrides':int(((g>1e-6)&p).sum()),
      'negative_overrides':int(((g< -1e-6)&p).sum()),
      'harm_over_40_sec':int(h.sum()),
      'harm_per_state':float(h.mean()),
      'harm_per_override':float(h.sum()/max(1,p.sum())),
      'mean_excess_harm_gt_40':float(np.maximum(-g-harm_limit,0.).mean()),
      'max_harm_sec':float(max(0.,-min(g))),
      'gain_p10':float(np.quantile(g,.1)),
      'gain_p50':float(np.median(g))}


def utility(result):
    # Only for validation-based model selection. Penalties discourage unsafe overrides.
    return (result['mean_signed_advantage_sec']-
      1.5*result['mean_excess_harm_gt_40']-
      25.*result['harm_per_state'])


def select_gate(model,X,B,Y,BY,valid,same,val_ids,thresholds=(0.,10.,20.,40.,80.,float('inf')),
                margins=(0.,),harm_limit=40.):
    """Pick gate ONLY from validation fold; ties prefer fewer overrides."""
    import copy
    grid=[]
    for threshold in thresholds:
        for margin in margins:
            c=copy.copy(model);c.threshold=float(threshold);c.margin=float(margin)
            r=evaluate(c,X,B,Y,BY,valid,same,val_ids,harm_limit)
            score=utility(r)
            grid.append({'threshold':None if not np.isfinite(threshold) else float(threshold),
                'margin':float(margin),'utility':float(score),'validation':r})
    best=max(grid,key=lambda q:(round(q['utility'],8),-q['validation']['overrides'],
            q['margin'],q['threshold'] if q['threshold'] is not None else float('inf')))
    model.threshold=float('inf') if best['threshold'] is None else best['threshold']
    model.margin=best['margin']
    return model,grid,best
