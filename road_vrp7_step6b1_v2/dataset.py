#!/usr/bin/env python3
"""Frozen Step6-A v2 dataset: strict validation, state/group splits, no target in X."""
from __future__ import annotations
import csv, hashlib, json
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import numpy as np

DEFAULT_SOURCE=Path('runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
PRIMARY='static_repair_td_gain_sec'
ORACLE='td_gain_sec'
LABELS_REQUIRED=(PRIMARY,ORACLE,'static_repair_static_gain_sec','td_advantage_over_static_repair_sec')
BANNED=('td_gain','td_advantage','static_repair_','original_td_',
        'repaired_routes','source_solver','seed','task_id','scenario_id')


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):h.update(buf)
    return h.hexdigest()

@dataclass
class Data:
    X:np.ndarray
    Y:np.ndarray
    groups:list[dict]
    names:list[str]
    labels:list[str]
    source:Path
    source_files:dict[str,Path]
    @property
    def n(self):return len(self.X)
    def label(self,k):return self.Y[:,self.labels.index(k)]
    def features(self,mode):
        stat=[i for i,n in enumerate(self.names) if not n.startswith('history_')]
        hist=[i for i,n in enumerate(self.names) if n.startswith('history_')]
        if mode=='core':idx=[i for i in stat if self.names[i] not in ('departure_hour_local','departure_weekday_local')]
        elif mode=='static':idx=stat
        elif mode=='traffic':idx=stat+hist
        else:raise ValueError('Invalid feature mode '+mode)
        return self.X[:,idx],[self.names[i] for i in idx]


def load(source=DEFAULT_SOURCE):
    source=Path(source)
    paths={'X':source/'analysis/features_X.npz','y':source/'analysis/labels_y.npz',
           'groups':source/'analysis/groups.csv','schema':source/'analysis/dataset_schema.json',
           'config':source/'run_config.json','summary':source/'benchmark_summary.json'}
    for p in paths.values():
        if not p.is_file():raise FileNotFoundError(p)
    config=json.loads(paths['config'].read_text(encoding='utf-8'))
    summary=json.loads(paths['summary'].read_text(encoding='utf-8'))
    if config.get('stage')!='step6a-offline-v2' or summary.get('status')!='COMPLETE':
        raise ValueError('Must use frozen completed Step6-A v2 data')
    if config.get('source_solvers')!=['C3_topk']:
        raise ValueError('Unexpected source solvers')
    with np.load(paths['X'],allow_pickle=False) as f:X=np.asarray(f['X'],dtype=np.float64)
    with np.load(paths['y'],allow_pickle=False) as f:Y=np.asarray(f['y'],dtype=np.float64)
    schema=json.loads(paths['schema'].read_text(encoding='utf-8'))
    names=list(schema['feature_names']);labels=list(schema['label_names'])
    with paths['groups'].open(newline='',encoding='utf-8') as f:groups=list(csv.DictReader(f))
    if X.ndim!=2 or Y.ndim!=2 or X.shape!=(len(groups),len(names)) or Y.shape!=(len(groups),len(labels)):
        raise ValueError('NPZ/groups/schema alignment mismatch')
    if list(X.shape)!=schema['shape_X'] or list(Y.shape)!=schema['shape_y']:
        raise ValueError('Declared NPZ shapes mismatch')
    if len(groups)<1 or len(set(names))!=len(names) or len(set(labels))!=len(labels):
        raise ValueError('Empty or repeated schema field')
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):raise ValueError('Non-finite NPZ values')
    if any(any(b in n for b in BANNED) for n in names):raise ValueError('Model feature leaks source ID or future Oracle target')
    if not all(t in labels for t in LABELS_REQUIRED):raise ValueError('Incomplete counterfactual labels')
    if not any(n.startswith('history_') for n in names):raise ValueError('No historical traffic features')
    if not {'departure_hour_local','departure_weekday_local','original_static_cost_sec'}.issubset(names):
        raise ValueError('Missing state context features')
    nvalid=summary.get('expected_max_actions')
    if nvalid is not None and X.shape[0]>nvalid:raise ValueError('More actions than source count')
    keys=[x['sample_id'] for x in groups]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate candidate sample_id')
    mapping={}
    for g in groups:
        for col in ('state_id','task_id','scenario_id','date','sample_id','task_role','date_role'):
            if not g.get(col):raise ValueError('Missing group identifier '+col)
        sig=(g['task_id'],g['scenario_id'],g['date'],g['task_role'],g['date_role'])
        if g['state_id'] in mapping and mapping[g['state_id']]!=sig:
            raise ValueError('Inconsistent state/group metadata')
        mapping[g['state_id']]=sig
    if summary.get('actual_states')!=len(mapping):raise ValueError('Expected states != exported states')
    td=Y[:,labels.index(ORACLE)];st=Y[:,labels.index(PRIMARY)]
    adv=Y[:,labels.index('td_advantage_over_static_repair_sec')]
    if not np.allclose(td-st,adv,rtol=0.,atol=2e-3) or np.any(adv< -2e-3):
        raise ValueError('TD vs static repair gain identity violated')
    return Data(X,Y,groups,names,labels,source,paths)


def masks(data,protocol):
    role={'task':'task_role','date':'date_role'}[protocol]
    group={'task':'task_id','date':'date'}[protocol]
    roles=np.array([g[role] for g in data.groups])
    if set(roles)!=set(('train','val','internal_test')):raise ValueError('Incomplete split')
    assigned={};states={}
    for g in data.groups:
        k=g[group];r=g[role]
        if k in assigned and assigned[k]!=r:raise ValueError('Split contamination: '+group)
        assigned[k]=r
        if g['state_id'] in states and states[g['state_id']]!=r:raise ValueError('State split contamination')
        states[g['state_id']]=r
    return {k:roles==k for k in ('train','val','internal_test')}


def select_rows(data,mask):return [g for g,keep in zip(data.groups,mask) if keep]


def grouped_indices(groups):
    by=defaultdict(list)
    for i,g in enumerate(groups):by[g['state_id']].append(i)
    return {k:np.asarray(v,dtype=np.int64) for k,v in by.items()}


def route_overlap(data):
    """Optional audit of repeated static routes, not a label/feature input."""
    file=data.source/'state_records.jsonl'
    if not file.is_file():return {'status':'NOT_AVAILABLE'}
    routes=defaultdict(list)
    seen=set()
    with file.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            rec=json.loads(line)
            sid=rec['state_id']
            if sid in seen:raise ValueError('Duplicate state in raw state_records')
            seen.add(sid)
            sig=hashlib.sha256(json.dumps(rec['routes'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            routes[(rec['task_id'],sig)].append((sid,rec['date']))
    if set(seen)!=set(g['state_id'] for g in data.groups):raise ValueError('Raw/export state mismatch')
    overlap=[r for r in routes.values() if len(set(x[1] for x in r))>1]
    return {'status':'CHECKED','distinct_task_route_structures':len(routes),
            'repeated_across_dates_structures':len(overlap),
            'states_on_cross_date_repeated_structures':sum(len(x) for x in overlap),
            'note':'Cross-date repeated route structures are expected for static source; date holdout does not imply unseen route topology.'}
