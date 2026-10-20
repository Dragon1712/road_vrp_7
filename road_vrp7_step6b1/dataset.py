#!/usr/bin/env python3
"""Frozen Step6-A v2 NPZ reader. No labels are placed in model feature matrices."""
from __future__ import annotations
import csv, hashlib, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

DEFAULT_SOURCE=Path('runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
PRIMARY='static_repair_td_gain_sec'
ORACLE='td_gain_sec'
FEATURE_BLACKLIST=('td_gain', 'td_advantage', 'static_repair_', 'original_td_',
                   'repaired_routes', 'source_solver', 'seed', 'task_id', 'scenario_id')


def sha256(p:Path)->str:
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

@dataclass
class Data:
    X:np.ndarray
    Y:np.ndarray
    groups:list[dict]
    names:list[str]
    labels:list[str]
    source:Path
    source_files:dict
    @property
    def n(self):return len(self.X)
    def label(self,k):return self.Y[:,self.labels.index(k)]
    def features(self,mode:str)->tuple[np.ndarray,list[str]]:
        static=[i for i,n in enumerate(self.names) if not n.startswith('history_')]
        history=[i for i,n in enumerate(self.names) if n.startswith('history_')]
        if mode=='core':
            idx=[i for i in static if self.names[i] not in ('departure_hour_local','departure_weekday_local')]
        elif mode=='static':idx=static
        elif mode=='traffic':idx=static+history
        else:raise ValueError(mode)
        return self.X[:,idx],[self.names[i] for i in idx]


def load(source:Path)->Data:
    source=Path(source)
    meta=source/'analysis'
    files={k:meta/name for k,name in {'X':'features_X.npz','y':'labels_y.npz',
        'groups':'groups.csv','schema':'dataset_schema.json'}.items()}
    for p in files.values():
        if not p.is_file():raise FileNotFoundError(p)
    if (source/'run_config.json').exists():
        cfg=json.loads((source/'run_config.json').read_text())
        if cfg['stage'] not in ('step6a-offline-v2','step6a-offline-v1'):
            raise ValueError('Unexpected dataset stage')
    if (source/'benchmark_summary.json').exists():
        summary=json.loads((source/'benchmark_summary.json').read_text())
        if summary['status']!='COMPLETE':raise ValueError('Incomplete source dataset')
    with np.load(files['X'],allow_pickle=False) as z:X=np.asarray(z['X'],np.float64)
    with np.load(files['y'],allow_pickle=False) as z:Y=np.asarray(z['y'],np.float64)
    schema=json.loads(files['schema'].read_text(encoding='utf-8'))
    names=list(schema['feature_names']);labels=list(schema['label_names'])
    with files['groups'].open(newline='',encoding='utf-8') as f: groups=list(csv.DictReader(f))
    if X.ndim!=2 or Y.ndim!=2 or X.shape!=(len(groups),len(names)) or Y.shape!=(len(groups),len(labels)):
        raise ValueError('Misaligned NPZ / schema / groups')
    if tuple(X.shape)!=tuple(schema['shape_X']) or tuple(Y.shape)!=tuple(schema['shape_y']):
        raise ValueError('Schema shape mismatch')
    if not np.isfinite(X).all() or not np.isfinite(Y).all():raise ValueError('Non-finite data')
    if len(names)!=len(set(names)) or len(labels)!=len(set(labels)):
        raise ValueError('Duplicate schema names')
    for field in (PRIMARY,ORACLE,'static_repair_static_gain_sec','td_advantage_over_static_repair_sec'):
        if field not in labels:raise ValueError('Missing label '+field)
    if any(any(k in n for k in FEATURE_BLACKLIST) for n in names):
        raise ValueError('Future-target or identifier leakage into X')
    if not any(n.startswith('history_') for n in names):raise ValueError('Missing history ablation')
    if len(set(g['sample_id'] for g in groups))!=len(groups):raise ValueError('Repeated samples')
    per_state={}
    for g in groups:
        for k in ('state_id','task_id','scenario_id','date','task_role','date_role','sample_id'):
            if not g.get(k):raise ValueError('Missing group field '+k)
        key=g['state_id']
        ident=(g['task_id'],g['scenario_id'],g['date'],g['task_role'],g['date_role'])
        if key in per_state and per_state[key]!=ident:raise ValueError('Inconsistent group attributes within state')
        per_state[key]=ident
    td=Y[:,labels.index(ORACLE)]; st=Y[:,labels.index(PRIMARY)]
    adv=Y[:,labels.index('td_advantage_over_static_repair_sec')]
    if np.any(td+2e-3<st) or not np.allclose(td-st,adv,rtol=0,atol=2e-3):
        raise ValueError('Oracle TD repair must not be worse than static repair in same candidate pool')
    return Data(X,Y,groups,names,labels,source,files)


def group_masks(data:Data,protocol:str):
    col={'task':'task_role','date':'date_role'}[protocol]
    groupcol={'task':'task_id','date':'date'}[protocol]
    role=np.asarray([g[col] for g in data.groups])
    if set(role)!=set(('train','val','internal_test')):raise ValueError('Invalid split roles')
    by_entity={};by_state={}
    for g in data.groups:
        k=g[groupcol];r=g[col]
        if k in by_entity and by_entity[k]!=r:raise ValueError('Leaked '+groupcol)
        by_entity[k]=r
        s=g['state_id']
        if s in by_state and by_state[s]!=r:raise ValueError('Leaked state')
        by_state[s]=r
    return {r:role==r for r in ('train','val','internal_test')}


def rows_at(data,mask):return [g for g,m in zip(data.groups,mask) if m]


def states_index(groups):
    out={}
    for i,g in enumerate(groups):out.setdefault(g['state_id'],[]).append(i)
    return {s:np.asarray(idx,dtype=np.int64) for s,idx in out.items()}
