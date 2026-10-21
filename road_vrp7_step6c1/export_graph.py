#!/usr/bin/env python3
"""Export strictly predeparture raw OSM-edge local graph for every frozen A-v2 action.

Resumable JSONL, exact sample_id alignment to B2 F4, SHA-verified sources.
"""
from __future__ import annotations
import argparse,json,hashlib,sys,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6b1_v2','road_vrp7_step6b2','road_vrp7_step6a','road_vrp7_stage3'):
    p=str(ROOT/folder)
    if p not in sys.path:sys.path.insert(0,p)
from dataset import load,sha256
from train_b2 import read_extra,get_features
from features import FEATURE_MODES,extract_boundary,core_online
from section_dataset import HistoricalCorridorFeatures,read_raw_cost
from road_vrp import Road,OD,Objective,read_json
from graph_features import candidate_graph,GRAPH_DIM

DEFAULT_SOURCE='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1'
DEFAULT_FEAT='runs/chicago_july_v1/step6b2/features_boundary_v1'
DEFAULT_COST='datasets/processed/chicago/july_v1/dynamic_costs'
DEFAULT_TASK='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10'
DEFAULT_CACHE='runs/chicago_july_v1/stage3/od_cache'


def run(args):
    out=Path(args.output_dir)
    source=Path(args.source_dir); feat=Path(args.features_dir)
    d=load(source);extra,mf,xhash=read_extra(d,feat)
    X,names=get_features(d,extra,'F4')
    if tuple(names)!=tuple(FEATURE_MODES['F4']):raise ValueError('F4 schema mismatch')
    config=read_json(source/'run_config.json')
    if config.get('lookback_slots')!=6:raise ValueError('Only six true lookback slots allowed')
    state_path=source/'state_records.jsonl'
    manifest={'stage':'step6c1-raw-graph-v1','source':str(source),'features':str(feat),
              'source_sha256':mf['source_sha256'],'b2_boundary_sha256':xhash,
              'raw_states_sha256':sha256(state_path),
              'cost_hash':config['source_cost_hash'],'edge_hash':config['source_edge_hash'],
              'graph_dim':GRAPH_DIM,'graph_n_edges':24,'lookback_slots':6,
              'max_states':args.max_states,
              'graph_code_sha256':sha256(Path(__file__).with_name('graph_features.py')),
              'export_code_sha256':sha256(Path(__file__)),
              'notes':['Only raw observed edge values and mask at strictly predeparture timestamps.',
                       'Nodes are sampled physical directed OSM road edges from B2 static relocation boundary paths.',
                       'Directed adjacency uses OSM destination equals OSM source, no inferred future traffic.']}
    mfpath=out/'manifest.json';checkpoint=out/'checkpoint.jsonl'
    if out.exists() and any(out.iterdir()):
        if not args.resume or not mfpath.exists() or read_json(mfpath)!=manifest:
            raise FileExistsError('Nonempty output; use new directory, or --resume identical config')
    else:
        out.mkdir(parents=True,exist_ok=False)
        mfpath.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    completed={}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if not line.strip():continue
            item=json.loads(line)
            if item['state_id'] in completed:raise ValueError('duplicate checkpoint state')
            completed[item['state_id']]=item['items']
    if not state_path.is_file():raise FileNotFoundError(state_path)
    statemap={}
    for row in state_path.read_text(encoding='utf-8').splitlines():
        if not row.strip():continue
        x=json.loads(row)
        if x['state_id'] in statemap:raise ValueError('Duplicate state')
        statemap[x['state_id']]=x
    groupby={}
    for g in d.groups:groupby.setdefault(g['state_id'],set()).add(g['sample_id'])
    if set(statemap)!=set(groupby):raise ValueError('Raw state/sample group mismatch')
    if args.max_states is not None:
        if args.max_states<1:raise ValueError('max_states must be positive')
        statemap=dict(list(statemap.items())[:args.max_states])
    partial=args.max_states is not None
    road=Road(args.cost_dir)
    if road.cost_file_hash!=config['source_cost_hash'] or road.edge_file_hash!=config['source_edge_hash']:
        raise ValueError('Frozen source road SHA mismatch')
    tm=read_json(Path(args.task_dir)/'manifest.json')
    if tm['cost_edge_index_sha256']!=road.edge_file_hash or tm['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Task/road mismatch')
    raw=read_raw_cost(args.cost_dir)
    hist=HistoricalCorridorFeatures(road,raw,6)
    tasks={};context=None;obj=None
    samplemap={}
    sample_idx={g['sample_id']:i for i,g in enumerate(d.groups)}
    for z,(sid,state) in enumerate(statemap.items(),1):
        if sid in completed:
            samplemap.update(completed[sid]);continue
        task_id=state['task_id']
        if task_id not in tasks:tasks[task_id]=read_json(Path(args.task_dir)/f'{task_id}.json')
        task=tasks[task_id]
        sc=next((x for x in task['scenarios'] if x['scenario_id']==state['scenario_id']),None)
        if sc is None:raise ValueError('Unknown source scenario')
        if (task_id,state['scenario_id'])!=context:
            od=OD(road,task,sc,step_min=int(config['od_step_min']),hours=float(config['od_hours']),
                  cache_dir=args.od_cache_dir)
            obj=Objective(task,od,'static')
            hist.enter(task_id,od,sc['departure_local']);context=(task_id,state['scenario_id'])
        values={}
        for c in state['candidates']:
            if not c['label_valid']:continue
            action={'route':int(c['route_index']),'pos':int(c['position']),'size':int(c['size'])}
            if list(state['routes'][action['route']][action['pos']:action['pos']+action['size']])!=c['block_ids']:
                raise ValueError('Misaligned section block')
            sid2=c['candidate_id']
            if sid2 not in sample_idx or sid2 in values:raise ValueError('Bad candidate sample id')
            if args.validate_f4:
                core=core_online(state['routes'],action,obj,state['departure_hour_local'],state['departure_weekday_local'])
                ext=extract_boundary(state['routes'],action,obj,hist)
                if ext is None:raise ValueError('Frozen B2 valid action missing proxy')
                merged={**core,**ext};expected=np.asarray([merged[k] for k in names])
                reference=X[sample_idx[sid2]]
                if not np.allclose(expected,reference,rtol=1e-5,atol=.025):
                    idx=int(np.argmax(abs(expected-reference)))
                    raise ValueError(f'F4 extraction mismatch: {sid2} {names[idx]} {expected[idx]} vs {reference[idx]}')
            values[sid2]=candidate_graph(state['routes'],action,obj,hist).flat().tolist()
        if set(values)!=groupby[sid]:raise ValueError('Labeled candidate export mismatch')
        with checkpoint.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'state_id':sid,'items':values},separators=(',',':'))+'\n')
            f.flush();os.fsync(f.fileno())
        completed[sid]=values;samplemap.update(values)
        if z<4 or z%25==0:print(f'C1 raw-edge graph {z}/{len(statemap)}',flush=True)
    if partial:
        summary={'status':'PILOT_ONLY','states':len(statemap),'candidates':len(samplemap),
                 'warning':'Partial export for smoke; no graph_X.npz; use NEW full directory for training.'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('PILOT_ONLY',json.dumps(summary));return
    if set(samplemap)!=set(sample_idx):raise ValueError('Incomplete source actions')
    output=np.asarray([samplemap[g['sample_id']] for g in d.groups],dtype=np.float32)
    if output.shape!=(d.n,GRAPH_DIM) or not np.isfinite(output).all():raise ValueError('Invalid graph tensor')
    np.savez_compressed(out/'graph_X.npz',X=output)
    summary={'status':'COMPLETE','candidates':int(d.n),'states':len(statemap),
             'shape':list(output.shape),'graph_sha256':sha256(out/'graph_X.npz'),
             'observed_slot_fraction':float(np.mean(output[:,0:24*6*2].reshape(-1,24,6,2)[:,:,:,1])),
             'manifest_sha256':sha256(mfpath)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print('COMPLETE',json.dumps(summary,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',default=DEFAULT_SOURCE)
    p.add_argument('--features-dir',default=DEFAULT_FEAT)
    p.add_argument('--cost-dir',default=DEFAULT_COST)
    p.add_argument('--task-dir',default=DEFAULT_TASK)
    p.add_argument('--od-cache-dir',default=DEFAULT_CACHE)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--validate-f4',action='store_true',help='Audit every new raw graph against frozen B2 F4')
    p.add_argument('--max-states',type=int,help='Smoke-only partial extraction; cannot be used for training')
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
if __name__=='__main__':main()
