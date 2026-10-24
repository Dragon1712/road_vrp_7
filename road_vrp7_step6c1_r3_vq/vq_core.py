"""Pure R3-VQ utilities. All TD/Road costs are retrospective labels ONLY."""
from __future__ import annotations
import hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

STAGE='R3-VQ-train-only-restart-v1'
LABELS='R3-VQ-labels-v1'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def hashkey(*things):
    return hashlib.sha256(json.dumps(things,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(exist_ok=True,parents=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(p)

def uniform_train_selection(groups,train_tasks,n,seed):
    """Outcome-blind, round robin among TRAIN tasks; never touch labels."""
    if n<1:raise ValueError('Number of states must be positive')
    bins=defaultdict(list)
    for i,g in enumerate(groups):
        if str(g['task_id']) in train_tasks:
            bins[str(g['task_id'])].append((hashkey(seed,g['decision_id']),i))
    if not bins or set(bins)!=set(train_tasks):raise ValueError('Missing train tasks')
    for k in bins:bins[k].sort()
    if n>sum(len(v) for v in bins.values()):raise ValueError('Requested more than train pool')
    ans=[]
    while len(ans)<n:
        progressed=False
        for k in sorted(bins):
            if bins[k] and len(ans)<n:
                ans.append(bins[k].pop(0)[1]);progressed=True
        if not progressed:raise RuntimeError('Sampler unexpectedly exhausted')
    if any(str(groups[i]['task_id']) not in train_tasks for i in ans):raise AssertionError('train leakage')
    return ans

def paired_features(actions,baseline):
    a=np.asarray(actions,dtype=np.float32);b=np.asarray(baseline,dtype=np.float32)
    if a.ndim!=2 or b.shape!=(a.shape[1],):raise ValueError('Bad paired feature shape')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite features')
    b0=np.broadcast_to(b,a.shape)
    return np.concatenate((a,b0,a-b0),axis=1).astype(np.float32)

def restart_seed(decision,rep):
    """Same seed for each action and H for given decision/replicate (CRN)."""
    return 1+int(hashkey('vq-restart-v1',decision,int(rep))[:12],16)%(2**31-2)

def row_labels(rows,metric='road'):
    """Mean and sample std of C3-relative gains, with complete replicate requirement."""
    field={'road':'final_road','td_od':'final_td_od'}[metric]
    by=defaultdict(dict);per_action=defaultdict(list)
    for r in rows:
        k=(r['decision_id'],int(r['horizon']),int(r['replicate']))
        a=int(r['action_index'])
        if a in by[k]:raise ValueError('Duplicate action in restart unit')
        by[k][a]=r
    for (di,h,rep),d in by.items():
        if -1 not in d:raise ValueError('Missing C3 arm')
        for a,r in d.items():
            if field not in r or field not in d[-1]:raise ValueError('Missing posthoc cost: '+field)
            delta=float(d[-1][field])-float(r[field])
            if a==-1 and abs(delta)>1e-8:raise AssertionError('C3 must be zero')
            per_action[(di,h,a)].append((rep,delta))
    result={}
    for k,entries in per_action.items():
        reps=[j for j,_ in entries]
        if len(set(reps))!=len(reps):raise ValueError('Duplicate replicate')
        vals=np.asarray([v for _,v in sorted(entries)],dtype=np.float64)
        result[k]={'mean_gain':float(vals.mean()),'sd_gain':float(vals.std(ddof=1)) if len(vals)>1 else None,
                   'n_replicates':len(vals),'per_rep':vals.tolist()}
    return result

def select_by_score(pred,valid,lam=0.,std=None,threshold=0.):
    y=np.asarray(pred,dtype=float);m=np.asarray(valid,dtype=bool)
    if len(y)!=len(m):raise ValueError('Scores/mask differ')
    if not np.isfinite(y).all():raise ValueError('Nonfinite score')
    if std is None:std=np.zeros_like(y)
    std=np.asarray(std,dtype=float)
    if std.shape!=y.shape or not np.isfinite(std).all():raise ValueError('Invalid uncertainty')
    z=y-lam*np.maximum(0.,std)
    z[~m]=-np.inf
    if not m.any():return -1
    j=int(np.argmax(z))
    return j if z[j]>threshold else -1

def decision_metrics(preds, truths, valid, state_keys, regret_limit=40.):
    """Pairwise state metrics; each state counts once (not every candidate)."""
    if not(len(preds)==len(truths)==len(valid)==len(state_keys)):raise ValueError('Shape disagreement')
    gains=[];regrets=[];harm=[];overrides=[];hits=[];per_task=defaultdict(list)
    for predicted,truth,mask,key in zip(preds,truths,valid,state_keys):
        j=int(predicted);t=np.asarray(truth,float);m=np.asarray(mask,bool)
        if j!=-1 and (j>=len(t) or not m[j]):raise ValueError('Invalid decision')
        oracle=max(0.,max(t[m],default=0.));real=0. if j==-1 else float(t[j]);r=oracle-real
        gains.append(real);regrets.append(r);harm.append(int(real< -regret_limit));overrides.append(int(j!=-1));hits.append(int(r<1e-5))
        per_task[key].append(real)
    return {'n_states':len(preds),'mean_gain':float(np.mean(gains)),'mean_regret':float(np.mean(regrets)),
            'harm_over_40_per_state':float(np.mean(harm)),'override_rate':float(np.mean(overrides)),
            'top1_hit':float(np.mean(hits)),
            'per_task_gain':{k:float(np.mean(v)) for k,v in sorted(per_task.items())}}
