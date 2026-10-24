#!/usr/bin/env python3
"""R4B: compare original ALNS 7.0.0 transitions to exact snapshot continuation.

No new Section intervention, traffic feature, learned model, or oracle label.
The original reference delegates to official ALNS.iterate; the resumed arm
runs the source-mirrored fixed-K transition loop from R4B core.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r4b','road_vrp7_step6c1_r4a','road_vrp7_step5b_v3',
               'road_vrp7_step6b2_r','road_vrp7_stage3'):
    p=str(ROOT/folder)
    if p not in sys.path:sys.path.insert(0,p)
from r4b_core import (STAGE,require,load_json,common_data_audit,take_snapshots,
    OfficialReferenceObserver,restored_iterator,check_start,verify_events)
from r4a_core import canonical_hash,sha256,write_json


@contextmanager
def reference_recording(windows):
    from alns import ALNS
    original=ALNS.iterate
    instances=[]
    wanted={i for low,k in windows for i in range(low,low+k)}
    def intercept(engine,initial,op_select,accept,stop,**kwargs):
        obs=OfficialReferenceObserver(op_select,accept,stop,wanted)
        instances.append(obs)
        result=original(engine,initial,obs,accept,stop,**kwargs)
        require(obs.calls==obs.updates,'Reference observer missed ALNS outcome updates')
        return result
    ALNS.iterate=intercept
    try:yield instances
    finally:ALNS.iterate=original


@contextmanager
def exact_resume(snapshot,k,holder):
    from alns import ALNS
    original=ALNS.iterate
    def intercept(engine,initial,op_select,accept,stop,**kwargs):
        result,trace=restored_iterator(engine,initial,op_select,accept,stop,snapshot,k,**kwargs)
        holder.append(trace)
        return result
    ALNS.iterate=intercept
    try:yield
    finally:ALNS.iterate=original


def preflight(a):
    import importlib.metadata
    require(importlib.metadata.version('alns')=='7.0.0','Require official alns==7.0.0')
    import run_experiment as v3
    from road_vrp import Road,OD,Objective,legal,read_json
    from benchmark import load_tasks
    capture=Path(a.capture_dir)
    mf,snapshots=common_data_audit(capture)
    require(sha256(ROOT/'road_vrp7_step5b_v3/run_experiment.py')==mf['solver_sha256'],'Frozen v3.solve SHA changed')
    require(sha256(ROOT/'road_vrp7_step5b_v3/section_v2.py')==mf['section_sha256'],'Frozen Section SHA changed')
    require(sha256(ROOT/'road_vrp7_step6c1_r4a/r4a_core.py')==mf['core_sha256'],'R4A snapshot schema source changed')
    src=Path(a.model_source)/'run_config.json'
    require(sha256(src)==mf['source_sha256'],'Model source SHA different from R4A')
    config=read_json(src);road=Road(a.cost_dir)
    require(road.cost_file_hash==mf['road_sha256'] and road.edge_file_hash==mf['edge_sha256'],'Road source differs')
    paths,tasks,tm=load_tasks(a.task_dir)
    require(tm['cost_edge_npz_sha256']==road.cost_file_hash and tm['cost_edge_index_sha256']==road.edge_file_hash,'Task manifest differs')
    taskmap={x['task_id']:x for x in tasks}
    for tid in mf['task_ids']:require(tid in taskmap,'Missing task '+tid)
    for tid in mf['task_ids']:
        scenarios={z['scenario_id']:z for z in taskmap[tid]['scenarios']}
        for sid in mf['scenario_ids']:
            require(sid in scenarios and scenarios[sid]['eligible'] and scenarios[sid]['split']=='validation','Invalid scenario')
    require(mf['max_iterations']==20000 and mf['anneal_iterations']==5000 and mf['search_wall_sec']==30,'Frozen R4A search config mismatch')
    return v3,Road,OD,Objective,legal,taskmap,config,road,mf,snapshots


def main(a):
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('Use new independent study directory: '+str(out))
    horizons=sorted({int(x) for x in a.horizons.split(',') if x.strip()})
    require(horizons and min(horizons)>0 and max(horizons)<=50,'Supported strict horizons K=1..50')
    require(a.max_cases>=0 and a.max_snapshots_per_run>=1,'Bad case/state caps')
    v3,Road,OD,Objective,legal,taskmap,cfg,road,mf,snapshots=preflight(a)
    by=defaultdict(list)
    for s in snapshots:by[(s['task_id'],s['scenario_id'],int(s['seed']))].append(s)
    ordered=sorted(by)
    if a.max_cases:ordered=ordered[:a.max_cases]
    require(bool(ordered),'No snapshots selected')
    out.mkdir(parents=True)
    write_json(out/'manifest.json',{'stage':STAGE,'status':'RUNNING',
        'source_capture_dir':str(a.capture_dir),'source_capture_snapshot_sha256':sha256(Path(a.capture_dir)/'snapshots.jsonl'),
        'source_capture_manifest_sha256':sha256(Path(a.capture_dir)/'manifest.json'),
        'code_sha256':{p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('r4b_core.py')]},
        'source_solver_sha256':mf['solver_sha256'],'source_section_sha256':mf['section_sha256'],
        'max_cases':a.max_cases,'max_snapshots_per_run':a.max_snapshots_per_run,
        'horizons':horizons,'no_original_train_labels_or_new_models':True,
        'note':'Exact iteration K, NOT exact remaining-wall-clock continuation.'})
    rows=[];failures=[]
    try:
        for case_num,(tid,sid,seed) in enumerate(ordered,1):
            task=taskmap[tid];sc=next(s for s in task['scenarios'] if s['scenario_id']==sid)
            od=OD(road,task,sc,step_min=int(cfg['od_step_min']),hours=float(cfg['od_hours']),cache_dir=a.od_cache_dir)
            obj=Objective(task,od,'static')
            selected=take_snapshots(snapshots,task_id=tid,scenario_id=sid,seed=seed,max_states=a.max_snapshots_per_run)
            windows=[(int(s['stop_count']),max(horizons)) for s in selected]
            with reference_recording(windows) as ref_instances:
                answer=v3.solve('C3_topk',task,obj,seed,mf['search_wall_sec'],mf['max_iterations'],
                                top_k=mf['top_k'],anneal_iterations=mf['anneal_iterations'])
            require(len(ref_instances)==1,'Expected exactly one reference ALNS.iterate')
            require(answer['iterations']==mf['max_iterations'],'Original trajectory wall-ended, cannot verify all K')
            events=ref_instances[0].events
            require(all(i in events for low,k in windows for i in range(low,low+k)),'Uninterrupted reference missing trace boundary')
            replay=road.replay(task,answer['routes'],sc['departure_local'])
            for snapshot in selected:
                start=int(snapshot['stop_count'])
                check_start(snapshot,events[start])
                for k in horizons:
                    trace=[]
                    with exact_resume(snapshot,k,trace):
                        # Original operator closures, objective, selector and SA are
                        # constructed by frozen v3.solve and restored at iterate entry.
                        v3.solve('C3_topk',task,obj,seed,mf['search_wall_sec'],mf['max_iterations'],
                                 top_k=mf['top_k'],anneal_iterations=mf['anneal_iterations'])
                    require(len(trace)==1,'Expected one resumed ALNS.iterate')
                    ref=[events[i] for i in range(start,start+k)]
                    verify_events(ref,trace[0])
                    rows.append({'task_id':tid,'scenario_id':sid,'seed':seed,
                        'snapshot_stop_count':start,'horizon':k,'verified_steps':k,
                        'exact':True,'starting_RNG_sha256':snapshot['rng_state_sha256'],
                        'last_outcome':ref[-1]['outcome'],
                        'end_candidate_sha256':ref[-1]['candidate_route_sha256']})
            print('R4B VERIFIED',case_num,'/',len(ordered),tid,sid,seed,
                  'original_road',round(replay['ex_post_road_travel_sec'],3),
                  'snapshots',len(selected),'K',horizons,flush=True)
            with (out/'paired_rows.jsonl').open('a',encoding='utf8') as f:
                for z in rows:
                    if z.get('_flushed'):continue
                    f.write(json.dumps(z,ensure_ascii=False,allow_nan=False)+'\n');z['_flushed']=True
    except Exception as e:
        failures.append({'type':type(e).__name__,'reason':str(e),'verified_so_far':len(rows)})
        write_json(out/'report.json',{'stage':STAGE,'status':'FAILED','failures':failures,
                                     'cases_requested':len(ordered),'verified_rows':len(rows)})
        raise
    finally:
        pass
    manifest=load_json(out/'manifest.json');manifest['status']='COMPLETE';write_json(out/'manifest.json',manifest)
    clean=[{k:v for k,v in z.items() if k!='_flushed'} for z in rows]
    report={'stage':STAGE,'status':'COMPLETE','cases':len(ordered),'snapshots':len(clean)//len(horizons),
        'horizons':horizons,'n_comparisons':len(clean),'exact_matches':len(clean),
        'per_horizon':{str(k):{'n':sum(z['horizon']==k for z in clean),'exact':sum(z['horizon']==k and z['exact'] for z in clean)} for k in horizons},
        'source_snapshot_sha256':sha256(Path(a.capture_dir)/'snapshots.jsonl'),
        'limitations':['Only 4 corners of original July validation pilot, not blind test.',
                       'Source-verified ALNS v7.0.0 iteration K; not exact 30s wall-clock remaining time.',
                       'Replay uses frozen original C3, no intervention or learned Section substitution.',
                       'Exact transitions require no deadline-dependent branch within evaluated K steps.',
                       'R4-B establishes continuation machinery; counterfactual Section labels are future R4-C work.']}
    write_json(out/'report.json',report)
    (out/'report.md').write_text('# R4-B 官方ALNS精确续接逐步审计\n\n'+
        f"结果：**{report['exact_matches']}/{report['n_comparisons']}** 个快照×Horizon组合逐步完全一致。\n\n"+
        '| K | 比较数 | 精确一致 |\n|---:|---:|---:|\n'+
        ''.join(f"| {k} | {report['per_horizon'][str(k)]['n']} | {report['per_horizon'][str(k)]['exact']} |\n" for k in horizons)+
        '\n只验证固定迭代续接，不验证剩余墙钟时间或首步反事实Section替换。\n',encoding='utf8')
    print('R4B COMPLETE',str(out),'exact',len(clean),'/',len(clean),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir',default='runs/chicago_july_v1/step6c1_r4a/r4a_pilot_20k_fix1_capture')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--model-source',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--horizons',default='1,5,20,50')
    p.add_argument('--max-cases',type=int,default=0,help='0=all cases; 1=single-case Smoke')
    p.add_argument('--max-snapshots-per-run',type=int,default=2)
    main(p.parse_args())
