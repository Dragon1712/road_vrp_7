#!/usr/bin/env python3
"""R4-G TRAIN-only pre-selection capture using original frozen ALNS and R4-A observer.

No oracle labels, validation, or test scenarios are admitted.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for name in ('road_vrp7_step6c1_r4a','road_vrp7_step5b_v3','road_vrp7_step6b2_r','road_vrp7_stage3'):
    p=str(ROOT/name)
    if p not in sys.path:sys.path.insert(0,p)
from r4a_core import write_json,sha256,canonical_hash,instrument_official_alns
STAGE='R4G-TRAIN-SECTION-SNAPSHOT-v1'


def main(a):
    import importlib.metadata
    if importlib.metadata.version('alns')!='7.0.0':raise RuntimeError('Must install alns==7.0.0')
    import run_experiment as v3
    from road_vrp import Road,OD,Objective,legal,read_json
    from benchmark import load_tasks

    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('Use a fresh independent output directory: '+str(out))
    if a.wall_sec!=30 or a.max_iterations<1 or a.anneal_iterations<1:
        raise ValueError('30s wall and positive iteration/anneal schedule required')
    save_at=sorted({int(v) for v in a.snapshots.split(',') if v.strip()})
    if a.arm=='capture' and (not save_at or min(save_at)<1):raise ValueError('Invalid snapshot iterations')
    if a.max_snapshots<1:raise ValueError('Invalid max snapshots')
    road=Road(a.cost_dir)
    taskfiles,tasks,tm=load_tasks(a.task_dir)
    if tm['cost_edge_index_sha256']!=road.edge_file_hash or tm['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Road/task edge SHA mismatch')
    src=read_json(Path(a.model_source)/'run_config.json')
    if src['source_cost_hash']!=road.cost_file_hash or src['source_edge_hash']!=road.edge_file_hash:
        raise ValueError('Road/model-source SHA mismatch')
    pin=src['code_sha256']
    for relative in ('road_vrp7_step5b_v3/section_v2.py','road_vrp7_stage3/road_vrp.py'):
        if relative not in pin or sha256(ROOT/relative)!=pin[relative]:
            raise ValueError('Frozen solver-source SHA changed: '+relative)
    selected_tasks=set(v.strip() for v in a.task_ids.split(',') if v.strip())
    selected_scenarios=set(v.strip() for v in a.scenario_ids.split(',') if v.strip())
    seeds=[int(x) for x in a.seeds.split(',')]
    if not selected_tasks or not selected_scenarios or not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('Task/scenario/seed list invalid')
    if not selected_tasks <= {'task_0000','task_0001','task_0002','task_0003'}:
        raise ValueError('R4G only accepts original TRAIN tasks 0000..0003')
    tasks=[t for t in tasks if t['task_id'] in selected_tasks]
    if {t['task_id'] for t in tasks}!=selected_tasks:raise ValueError('Unknown task ID')
    scenarios=[s for s in tasks[0]['scenarios'] if s['scenario_id'] in selected_scenarios]
    if {s['scenario_id'] for s in scenarios}!=selected_scenarios:raise ValueError('Unknown scenario ID')
    if any(s['split']!='train' or not s['eligible'] for s in scenarios):
        raise ValueError('R4G refuses non-TRAIN or ineligible scenarios')
    for task in tasks:
        all_s={s['scenario_id']:s for s in task['scenarios']}
        if any(all_s.get(s['scenario_id'])!=s for s in scenarios):raise ValueError('Scenario mismatch across tasks')
        if any(all_s[s['scenario_id']]['split']!='train' or not all_s[s['scenario_id']]['eligible'] for s in scenarios):
            raise ValueError('Non-TRAIN scenario in another task')
    mf={'stage':STAGE,'arm':a.arm,'source_repository_commit':'5014866fc4d2428b9b603959ea3dffa0c8761c9d',
        'task_ids':sorted(selected_tasks),'scenario_ids':sorted(selected_scenarios),'seeds':seeds,
        'search_wall_sec':a.wall_sec,'max_iterations':a.max_iterations,'anneal_iterations':a.anneal_iterations,
        'snapshot_iterations':save_at,'section_only':bool(a.section_only and a.arm=='capture'),
        'max_snapshots':a.max_snapshots,'top_k':a.top_k,
        'source_sha256':sha256(Path(a.model_source)/'run_config.json'),
        'road_sha256':road.cost_file_hash,'edge_sha256':road.edge_file_hash,
        'solver_sha256':sha256(ROOT/'road_vrp7_step5b_v3/run_experiment.py'),
        'section_sha256':sha256(ROOT/'road_vrp7_step5b_v3/section_v2.py'),
        'collector_sha256':sha256(Path(__file__)),'core_sha256':sha256(ROOT/'road_vrp7_step6c1_r4a/r4a_core.py'),
        'alns_version':'7.0.0','split':'train',
        'limitations':['Unmodified original C3, captured BEFORE operator selection, no perturbations.',
                       'A snapshot by itself does NOT yet enable exact continuation; restoration still requires R4B.',
                       'Capture occurs within 30s search wall budget; JSON flushed afterwards.',
                       'Strict task manifest TRAIN only; no labels created by capture.']}
    out.mkdir(parents=True,exist_ok=False);write_json(out/'manifest.json',mf)
    rows=[];snapshots=[]
    for t in tasks:
        for sc in scenarios:
            od=OD(road,t,sc,step_min=int(src['od_step_min']),hours=float(src['od_hours']),cache_dir=a.od_cache_dir)
            for seed in seeds:
                obj=Objective(t,od,'static')
                if a.arm=='capture':
                    with instrument_official_alns(save_at,max_snapshots=a.max_snapshots,section_only=a.section_only) as observed:
                        ans=v3.solve('C3_topk',t,obj,seed,a.wall_sec,a.max_iterations,
                                     top_k=a.top_k,anneal_iterations=a.anneal_iterations)
                    if len(observed)!=1:raise RuntimeError('Expected EXACTLY one official ALNS.iterate call')
                    obs=observed[0];audit=obs.summary()
                    if audit['selector_calls']!=ans['iterations']:
                        raise AssertionError('Snapshot selector calls != frozen solver iterations')
                    for s in obs.snapshots:
                        snapshots.append({'task_id':t['task_id'],'scenario_id':sc['scenario_id'],'seed':seed,**s})
                else:
                    # Plain control must run the UNMODIFIED frozen C3 solver.
                    # R4A v1 erroneously accessed ans before assigning it here.
                    ans=v3.solve('C3_topk',t,obj,seed,a.wall_sec,a.max_iterations,
                                 top_k=a.top_k,anneal_iterations=a.anneal_iterations)
                    audit={'selector_calls':ans['iterations'],
                           'selector_updates':ans['iterations'],
                           'snapshot_count':0}
                legal(t,ans['routes'])
                replay=road.replay(t,ans['routes'],sc['departure_local'])
                rows.append({'task_id':t['task_id'],'scenario_id':sc['scenario_id'],'seed':seed,
                    'arm':a.arm,'search_wall_sec':float(ans['search_wall_sec']),
                    'iterations':int(ans['iterations']),'termination_reason':str(ans['termination_reason']),
                    'planned_cost_sec':float(ans['planned_cost']),
                    'road_cost_sec':float(replay['ex_post_road_travel_sec']),
                    'best_route_sha256':canonical_hash(ans['routes']),
                    'captured':audit['snapshot_count'],'observer':audit})
                print('R4G',a.arm,t['task_id'],sc['scenario_id'],seed,
                      'iter',ans['iterations'],'road',replay['ex_post_road_travel_sec'],
                      'snaps',audit['snapshot_count'],flush=True)
    expected=len(tasks)*len(scenarios)*len(seeds)
    if len(rows)!=expected:raise AssertionError('Incomplete rows')
    with (out/'metrics.jsonl').open('w',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'snapshots.jsonl').open('w',encoding='utf8') as f:
        for s in snapshots:f.write(json.dumps(s,ensure_ascii=False,allow_nan=False)+'\n')
    write_json(out/'report.json',{'status':'COMPLETE','stage':STAGE,'arm':a.arm,'records':len(rows),
          'expected_records':expected,'snapshots':len(snapshots),'ended_normally':True})
    write_json(out/'provenance.json',{'metrics_sha256':sha256(out/'metrics.jsonl'),
                                      'snapshots_sha256':sha256(out/'snapshots.jsonl'),
                                      'manifest_sha256':sha256(out/'manifest.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['plain','capture'],required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--model-source',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--task-ids',default='task_0000,task_0001,task_0002,task_0003')
    p.add_argument('--scenario-ids',default='2025-07-02_0800,2025-07-09_1700')
    p.add_argument('--seeds',default='60,61')
    p.add_argument('--wall-sec',type=int,default=30)
    p.add_argument('--max-iterations',type=int,default=2000)
    p.add_argument('--anneal-iterations',type=int,default=5000)
    p.add_argument('--top-k',type=int,default=12)
    p.add_argument('--snapshots',default='1000,4000,8000,12000,16000,19000')
    p.add_argument('--max-snapshots',type=int,default=16)
    p.add_argument('--section-only',action='store_true',help='Capture first REAL Section selection on/after each milestone')
    main(p.parse_args())
