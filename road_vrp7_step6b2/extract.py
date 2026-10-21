#!/usr/bin/env python3
"""Produce B2 boundary features from frozen Step6-A v2 states (restartable).

Consumes NO Oracle result fields when constructing features. Existing B1 groups/NPZ
and TD targets remain untouched. Output is exclusively new feature data.
"""
from __future__ import annotations
import argparse,json,hashlib,os,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from features import ROOT,B2_SUFFIX,CORE,extract_boundary,core_online
from dataset import load,sha256
from section_dataset import HistoricalCorridorFeatures,read_raw_cost
from road_vrp import Road,OD,Objective,read_json


def code_signature():return {p:sha256(Path(__file__).with_name(p)) for p in ('extract.py','features.py')}

def make_manifest(source,cost,task,od_cache,max_states,config):
    d=load(source)
    required={k:sha256(v) for k,v in d.source_files.items()}
    state_path=d.source/'state_records.jsonl'
    if not state_path.exists():raise FileNotFoundError(state_path)
    return {'stage':'step6b2-features-v1','source':str(source),'source_sha256':required,
            'state_records_sha256':sha256(state_path),'code_sha256':code_signature(),
            'cost_dir':str(cost),'task_dir':str(task),'od_cache_dir':str(od_cache),
            'max_states':max_states,'feature_names':list(B2_SUFFIX),
            'source_edge_sha256':config['source_edge_hash'],
            'source_cost_sha256':config['source_cost_hash'],
            'historical_lookback_slots':config['lookback_slots'],
            'source_top_k':config['top_k'],
            'description':'Static first-proxy relocation only, never TD/Oracle insertion; historical raw before departure.'}

def raw_states(p):
    seen=set()
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            x=json.loads(line)
            if x['state_id'] in seen:raise ValueError('Duplicate raw state')
            seen.add(x['state_id']);yield x

def run(source,out,cost,task,cache,max_states=None,resume=False):
    source,out,cost,task,cache=map(Path,(source,out,cost,task,cache))
    d=load(source);cfg=read_json(source/'run_config.json')
    if int(cfg['lookback_slots'])!=6:raise ValueError('F1 frozen 6-slot schema expected')
    pinned=cfg.get('code_sha256',{})
    for relative in ('road_vrp7_step5b_v3/section_v2.py','road_vrp7_stage3/road_vrp.py'):
        if relative not in pinned or sha256(ROOT/relative)!=pinned[relative]:
            raise ValueError('Frozen Step6-A version mismatch: '+relative)
    manifest=make_manifest(source,cost,task,cache,max_states,cfg)
    mf=out/'run_config.json';checkpoint=out/'features_checkpoint.jsonl'
    if mf.exists():
        if not resume:raise FileExistsError('Existing study: use --resume with identical config or NEW directory')
        if read_json(mf)!=manifest:raise ValueError('Different source/code/cost settings in resumed study')
    else:
        if out.exists() and any(out.iterdir()):raise FileExistsError('Nonempty unmanifested feature directory')
        out.mkdir(parents=True,exist_ok=True)
        mf.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    completed={}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row['state_id'] in completed:raise ValueError('Duplicate checkpoint state')
            completed[row['state_id']]=row
    wanted=set(g['sample_id'] for g in d.groups)
    keys_by_state=defaultdict(set)
    for g in d.groups:keys_by_state[g['state_id']].add(g['sample_id'])
    raw=list(raw_states(source/'state_records.jsonl'))
    if set(x['state_id'] for x in raw)!=set(keys_by_state):raise ValueError('Source state mismatch')
    if max_states is not None:raw=raw[:max_states]
    if not raw:raise ValueError('No states')
    selected={x['state_id'] for x in raw}
    if not set(completed)<=selected:raise ValueError('Existing checkpoint outside selected states')
    cost_validation=read_json(cost.parent/'stage2_validation.json')
    if cost_validation.get('validation')!='PASS':raise ValueError('Stage2 validation not PASS')
    road=Road(cost)
    if road.cost_file_hash!=cfg['source_cost_hash'] or road.edge_file_hash!=cfg['source_edge_hash']:
        raise ValueError('Frozen road tensor SHA mismatch')
    tm=read_json(task/'manifest.json')
    if tm['cost_edge_index_sha256']!=road.edge_file_hash or tm['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Task and road manifest mismatch')
    hist=HistoricalCorridorFeatures(road,read_raw_cost(cost),6)
    task_cache={};context=None;static=None
    for ix,st in enumerate(raw,1):
        sid=st['state_id']
        if sid in completed:continue
        task_id=st['task_id']
        if task_id not in task_cache:
            task_cache[task_id]=read_json(task/f'{task_id}.json')
        objtask=task_cache[task_id]
        sc=next((s for s in objtask['scenarios'] if s['scenario_id']==st['scenario_id']),None)
        if sc is None:raise ValueError('Missing source scenario')
        if (task_id,st['scenario_id'])!=context:
            od=OD(road,objtask,sc,step_min=int(cfg['od_step_min']),hours=float(cfg['od_hours']),cache_dir=cache)
            static=Objective(objtask,od,'static')
            hist.enter(task_id,od,sc['departure_local']);context=(task_id,st['scenario_id'])
        rows=[]
        for cand in st['candidates']:
            if not cand['label_valid']:continue
            info={k:int(cand[k]) for k in ('route_index','position','size')}
            info={'route':info['route_index'],'pos':info['position'],'size':info['size']}
            if list(st['routes'][info['route']][info['pos']:info['pos']+info['size']])!=cand['block_ids']:
                raise ValueError('Section block mismatch')
            result=extract_boundary(st['routes'],info,static,hist)
            if result is None:raise ValueError('No static relocation proxy for valid labeled section')
            # Training core MUST mean the same as online core. Inspect values on source.
            expected=core_online(st['routes'],info,static,st['departure_hour_local'],st['departure_weekday_local'])
            for k,v in expected.items():
                observed=float(st[k]) if k in st else float(cand[k])
                if abs(v-observed)>max(.02,abs(observed)*1e-5):
                    raise ValueError(f'Core train/online mismatch {sid}: {k}: {v} != {observed}')
            rows.append({'sample_id':cand['candidate_id'],'extra':[float(result[k]) for k in B2_SUFFIX]})
        if set(z['sample_id'] for z in rows)!=keys_by_state[sid]:raise ValueError('Missing/extra section labels in state')
        row={'state_id':sid,'rows':rows}
        with checkpoint.open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
        completed[sid]=row
        if ix<=3 or ix%25==0:print('B2 extracted',ix,'/',len(raw),sid,flush=True)
    lookup={}
    for state in completed.values():
        for record in state['rows']:
            if record['sample_id'] in lookup:raise ValueError('Duplicate feature ID')
            lookup[record['sample_id']]=record['extra']
    desired={g['sample_id'] for g in d.groups if g['state_id'] in selected}
    if set(lookup)!=desired:raise ValueError('Incomplete extracted features')
    # Leave npz output FULL-length only; partial pilot is an explicit audit-only artifact.
    if max_states is None:
        X=np.asarray([lookup[g['sample_id']] for g in d.groups],dtype=np.float64)
        if X.shape!=(d.n,len(B2_SUFFIX)) or not np.isfinite(X).all():raise ValueError('Feature array malformed')
        np.savez_compressed(out/'boundary_X.npz',X=X)
        manifest['feature_npz_sha256']=sha256(out/'boundary_X.npz')
    summary={'status':'COMPLETE' if max_states is None else 'PILOT_ONLY',
       'states':len(completed),'candidate_features':len(lookup),'expected_all_candidates':d.n,
       'feature_names':list(B2_SUFFIX),'source_sha256':manifest['source_sha256'],
       'feature_npz_sha256':manifest.get('feature_npz_sha256'),
       'warning':'Planning-time static proxies and strictly predeparture raw traffic, not future TD repair.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('feature_names','source_sha256')},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--max-states',type=int)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.source_dir,a.output_dir,a.cost_dir,a.task_dir,a.od_cache_dir,a.max_states,a.resume)
if __name__=='__main__':main()
