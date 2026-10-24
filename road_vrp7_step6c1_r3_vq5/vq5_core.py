"""R3-VQ5 fixed-panel, original-train-only learning-curve utilities.

No model-fitting / label computation here. Never mix posterior road costs with online features.
"""
from __future__ import annotations
import hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

STAGE='R3-VQ5-FIXED-PANEL-LOTO-v1'
REPORT_STAGE='R3-VQ5-FIXED-PANEL-CURVE-v1'
DEFAULT_SIZES=(48,96,144,192)
WATCH=('C3','E0_FULL','E0_MATCHED','VQ_MEAN','RESID_OOF25','RESID_OOF50',
       'PAIR_RANK','PAIR_REGRET','E0_RISK35','E0_SCORE_COVERAGE35')
METRICS=('mean_gain_sec','mean_regret_sec','override_rate','harm_gt40_per_state',
         'harm_gt40_per_override','top1_hit')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(p)

def check_balanced_prefix(ids, groups, lookup, roles, sizes=DEFAULT_SIZES, tasks=None):
    """Reject nonbalanced / non-nested / non-TRAIN panel. Order is fixed by source manifest."""
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Empty/duplicated decision identifiers')
    sizes=tuple(int(n) for n in sizes)
    if not sizes or len(set(sizes))!=len(sizes) or sizes!=tuple(sorted(sizes)):
        raise ValueError('Sizes must be unique sorted ascending')
    if sizes[-1]!=len(ids):raise ValueError('Largest training size must equal fixed-panel size')
    ts=sorted({str(groups[lookup[di]]['task_id']) for di in ids})
    if tasks is not None and ts!=sorted(tasks):raise ValueError('Task identity changed')
    if len(ts)!=4 or any(roles[t]!='train' for t in ts):raise ValueError('Exactly 4 ORIGINAL Train tasks needed')
    for n in sizes:
        if n<8 or n%4:raise ValueError('Each sample size must be divisible by 4')
        got=defaultdict(int)
        for di in ids[:n]:
            if di not in lookup:raise ValueError('Unknown decision identifier')
            got[str(groups[lookup[di]]['task_id'])]+=1
        if dict(got)!={t:n//4 for t in ts}:
            raise ValueError(f'Prefix is not exactly task-balanced at n={n}: {dict(got)}')
    return ts

def hybrid_outer_selection(panel, prefix, groups, lookup, heldout):
    """Train on OTHER tasks in prefix; evaluate ALL states of heldout task in full panel."""
    prefix=set(prefix)
    if not prefix.issubset(panel):raise ValueError('Training prefix escapes fixed panel')
    result=[di for di in panel if str(groups[lookup[di]]['task_id'])==heldout or di in prefix]
    train=[di for di in result if str(groups[lookup[di]]['task_id'])!=heldout]
    evalu=[di for di in result if str(groups[lookup[di]]['task_id'])==heldout]
    if not train or not evalu or set(train)&set(evalu):raise ValueError('Invalid train/eval partition')
    if any(str(groups[lookup[di]]['task_id'])==heldout for di in train):raise AssertionError('Train leakage')
    return result,train,evalu

def aggregate_metrics(folds,methods):
    names=('mean_gain_sec','mean_regret_sec','top1_hit','override_rate',
           'harm_gt40_per_state','harm_gt40_per_override','mean_excess_harm_gt40')
    return {name:{k:float(np.mean([folds[t]['metrics'][name][k] for t in sorted(folds)])) for k in names}
            for name in methods}

def anchor_filter(rows,anchor_ids):
    keep=set(anchor_ids)
    return [x for x in rows if x['decision_id'] in keep]

def same_decision_rows(new,old,tol=1e-7):
    """Compare same-policy decisions against original VQ4 n48, irrespective of row ordering."""
    def index(rows):
        d={}
        for r in rows:
            key=(r['decision_id'],int(r['horizon']),r['policy'])
            if key in d:raise ValueError('Duplicate anchor evaluation')
            d[key]=r
        return d
    n=index(new);o=index(old)
    if n.keys()!=o.keys():raise ValueError('Anchor decision identity mismatch')
    for key in n:
        a,b=n[key],o[key]
        if int(a['selected_index'])!=int(b['selected_index']):
            raise ValueError('Frozen VQ3 action mismatch: '+str(key))
        for field in ('realized_mean_road_gain','regret'):
            if abs(float(a[field])-float(b[field]))>tol:
                raise ValueError('Frozen VQ3 outcome mismatch: '+str(key))
    return len(n)
