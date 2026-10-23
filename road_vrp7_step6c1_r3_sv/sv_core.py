"""R3-SV: pure action comparisons, audits, deterministic sampling and statistics.

The recorded label Y is the ONE-STEP TD-OD gain of static repair; do not
interpret it as 30-second performance or exact Road.replay cost.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ARMS=('C3','M0_PROXY','R2_F2','T2_HUBER')


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(p)


def selection(model, actions, baseline, valid, same):
    # Reuses the EXACT R2/R3 select signature. Invalid and baseline-identical
    # Section alternatives never win. C3 is an explicit defer action, index -1.
    j,score,reason=model.select(actions,baseline,same=same,valid=valid)
    if j is not None and (not bool(valid[j]) or bool(same[j])):
        raise ValueError('Selector picked invalid or C3-identical action')
    return -1 if j is None else int(j),reason,None if not np.isfinite(score) else float(score)


def m0_selection(artifact,actions,valid,same):
    """Reconstructed M0 recommendation on the SAME 8 labeled actions.

    Unlike frozen M0's own online trajectory, this is a C3-normalized proxy:
    when M0's gate rejects, defer to actual C3 instead of allocating no-op.
    It is NOT claimed to reproduce M0 online action selection exactly.
    """
    ids=np.flatnonzero(valid&(~same))
    if not len(ids):return -1,'no_valid_nonC3_candidate',None
    raw=artifact['model'].predict(actions[ids])
    if not isinstance(raw,(tuple,list)) or len(raw)<2:raise ValueError('Unexpected M0 predict API')
    score=np.asarray(raw[0],float).reshape(-1)
    gate=np.asarray(raw[1],float).reshape(-1)
    if len(score)!=len(ids) or len(gate)!=len(ids) or not np.all(np.isfinite(score)) or not np.all(np.isfinite(gate)):
        raise ValueError('M0 score shape or finiteness mismatch')
    local=int(np.argmax(score));thr=artifact['threshold']
    if thr is None or gate[local]<float(thr):return -1,'gate_defer_to_C3_PROXY',float(score[local])
    return int(ids[local]),'M0_PROXY_pick',float(score[local])


def classify_row(deltas,valid,same,indices,meta):
    """Strict same-state oracle/regret and gain comparison; gain of C3 is 0."""
    deltas=np.asarray(deltas,float);valid=np.asarray(valid,bool);same=np.asarray(same,bool)
    possible=np.flatnonzero(valid&(~same))
    best=max([0.]+[float(deltas[j]) for j in possible]);out=[]
    if set(indices)!=set(ARMS):raise ValueError('Four arms required')
    for arm in ARMS:
        j=indices[arm][0]
        if j!=-1 and j not in possible:raise ValueError('Selected invalid action')
        gain=0. if j==-1 else float(deltas[j])
        out.append({**meta,'arm':arm,'selected_index':int(j),
           'reason':indices[arm][1], 'predicted_score':indices[arm][2],
           'gain_sec':gain,'oracle_sec':best,'regret_sec':max(0.,best-gain),
           'override':j!=-1,'harm_gt40':bool(gain< -40.),'top1_hit':bool(best-gain<1e-5),
           'n_actions':int(len(possible))})
    return out


def summary(rows):
    out={}
    for arm in ARMS:
        z=[r for r in rows if r['arm']==arm]
        if not z:continue
        def avg(key):return float(np.mean([r[key] for r in z]))
        out[arm]={'n_states':len(z),'avg_gain_sec':avg('gain_sec'),
            'avg_regret_sec':avg('regret_sec'),'top1_hit':avg('top1_hit'),
            'override_rate':avg('override'),'harm_gt40_per_state':avg('harm_gt40')}
    return out


def pick_states(rows,max_states=12,seed=20261023,role='internal_test'):
    """Disagreement-enriched diagnostic sample; NOT population-representative."""
    if max_states<1:raise ValueError('max_states must be positive')
    idx={}
    for r in rows:
        if r['arm']=='R2_F2':idx.setdefault(r['decision_id'],{})['F2']=r
        if r['arm']=='T2_HUBER':idx.setdefault(r['decision_id'],{})['HUBER']=r
    eligible=[]
    for decision,v in idx.items():
        if 'F2' not in v or 'HUBER' not in v:continue
        a,b=v['F2'],v['HUBER']
        if a['role']!=role or b['role']!=role:continue
        eligible.append((decision,int(a['selected_index']!=b['selected_index']),a['task_id'],a['scenario_id']))
    buckets={0:[],1:[]}
    for decision,disagree,t,s in eligible:
        buckets[disagree].append((digest([seed,t,s,decision]),decision))
    for x in buckets.values():x.sort()
    selected=[];quota=max_states//2
    for k in (1,0):selected.extend([x[1] for x in buckets[k][:quota]])
    remaining=sorted([(h,k) for x in buckets.values() for h,k in x if k not in selected])
    selected.extend(k for _,k in remaining[:max_states-len(selected)])
    return selected[:max_states]


def paired_summary(roll_rows,metric):
    """Only complete paired groups, same state/horizon/restart replicate."""
    groups={}
    for r in roll_rows:
        k=(r['decision_id'],int(r['horizon']),int(r['replicate']))
        d=groups.setdefault(k,{})
        if r['arm'] in d:raise ValueError('Duplicate arm in paired group')
        d[r['arm']]=r
    results={}
    for h in sorted({int(r['horizon']) for r in roll_rows}):
        vs={arm:[] for arm in ARMS if arm!='C3'}
        for k,d in groups.items():
            if k[1]!=h:continue
            if set(d)!=set(ARMS):raise ValueError('Incomplete paired group')
            for arm in vs:
                vs[arm].append(float(d['C3'][metric])-float(d[arm][metric]))
        results[str(h)]={arm:{'n_pairs':len(a),'mean_gain_vs_C3_sec':float(np.mean(a)) if a else None,
                        'wins':int(sum(v>1e-8 for v in a)),
                        'losses':int(sum(v< -1e-8 for v in a)),
                        'ties':int(sum(abs(v)<=1e-8 for v in a))}
                         for arm,a in vs.items()}
    return results
