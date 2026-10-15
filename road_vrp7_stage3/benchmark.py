#!/usr/bin/env python3
"""Matched July stage3 baseline, strict resume, exact-road evaluation.

Offline after official alns==7.0.0 installed. Oracle is hindsight, never deployed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
import pandas as pd
from road_vrp import Road,OD,Objective,sha256,legal,write_json,read_json
from solvers import METHODS,solve

COLS=['task_id','scenario_id','date','split','eligible','departure_local','solver','planner','seed',
      'ex_post_road_travel_sec','ex_post_total_vehicle_elapsed_sec','observed_edge_share',
      'surrogate_travel_sec','initial_surrogate_travel_sec','construct_wall_sec','search_wall_sec',
      'od_precompute_wall_sec','total_solver_wall_sec','iterations','score_calls','vehicle_loads',
      'num_vehicles_used','od_cache_hit']
KEY=('task_id','scenario_id','solver','planner','seed')


def load_tasks(path):
    files=sorted(Path(path).glob('task_*.json'))
    if not files:raise ValueError('No generated task_*.json')
    records=[read_json(p) for p in files]
    if len({t['task_id'] for t in records})!=len(records):raise ValueError('Duplicate task id')
    manifest=read_json(Path(path)/'manifest.json')
    if manifest['task_ids']!=[t['task_id'] for t in records]:raise ValueError('Task manifest mismatch')
    return files,records,manifest


def frozen_scenarios(task,split,include_ineligible,max_scenarios,scenario_ids=None):
    rows=[r for r in task['scenarios'] if r['split']==split and (include_ineligible or r['eligible'])]
    if scenario_ids is not None:rows=[r for r in rows if r['scenario_id'] in scenario_ids]
    if max_scenarios:rows=rows[:max_scenarios]
    return rows


def assert_existing_config(out,config,resume):
    cfg=out/'run_config.json'
    if cfg.exists():
        old=read_json(cfg)
        if old!=config:raise ValueError('Output directory has DIFFERENT run config; choose a new output-dir')
        if not resume:raise FileExistsError('Existing benchmark output; set --resume or use a new directory')
    else:
        if (out/'metrics.csv').exists() or (out/'solutions.jsonl').exists():raise ValueError('Partial output without run_config.json; inspect manually')
        write_json(cfg,config)


def inspect_resume(out):
    m=out/'metrics.csv';s=out/'solutions.jsonl'
    if not m.exists() and not s.exists():return set()
    if not m.exists() or not s.exists():raise ValueError('Incomplete checkpoint: metrics and solutions files must both exist')
    df=pd.read_csv(m)
    if not set(KEY).issubset(df.columns):raise ValueError('metrics.csv schema invalid')
    if df.duplicated(list(KEY)).any():raise ValueError('Duplicate metric records')
    met={(str(r.task_id),str(r.scenario_id),str(r.solver),str(r.planner),int(r.seed)) for r in df.itertuples()}
    rec=[]
    with s.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            rec.append(tuple([str(row[k]) for k in KEY[:-1]]+[int(row['seed'])]))
    if len(rec)!=len(set(rec)) or set(rec)!=met:raise ValueError('Resume checkpoint metrics/solutions key mismatch')
    return met


def run(args):
    out=Path(args.output_dir)
    if args.split not in ('train','validation','test'):raise ValueError('Unknown split')
    only_scenarios=set(args.scenario_ids.split(',')) if args.scenario_ids else None
    methods=[a.strip() for a in args.methods.split(',') if a.strip()]
    planners=[a.strip() for a in args.planners.split(',') if a.strip()]
    seeds=[int(s) for s in args.seeds.split(',')]
    if not methods or len(set(methods))!=len(methods) or not set(methods).issubset(METHODS):raise ValueError('Invalid method names')
    if not planners or len(set(planners))!=len(planners) or not set(planners).issubset({'static','oracle'}):raise ValueError('Invalid planner names')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Duplicate/empty seeds')
    if 'official_alns' in methods:
        import importlib.metadata
        try:version=importlib.metadata.version('alns')
        except importlib.metadata.PackageNotFoundError:raise RuntimeError('Missing official ALNS: pip install alns==7.0.0')
        if version!='7.0.0':raise RuntimeError('ALNS version must equal 7.0.0; got '+version)
    road=Road(args.cost_dir)
    stage2=read_json(Path(args.cost_dir).parent/'stage2_validation.json')
    if stage2.get('validation')!='PASS':raise RuntimeError('Stage2 verification did not PASS')
    tasks_files,task_records,task_manifest=load_tasks(args.task_dir)
    if task_manifest['cost_edge_index_sha256']!=road.edge_file_hash or task_manifest['cost_edge_npz_sha256']!=road.cost_file_hash:raise ValueError('Tasks generated from different July road costs')
    if args.max_tasks:task_records=task_records[:args.max_tasks];tasks_files=tasks_files[:args.max_tasks]
    all_ids=set()
    for task in task_records:
        legal(task,task['seed_routes'])
        if task['task_id'] in all_ids:raise ValueError('Duplicate tasks')
        all_ids.add(task['task_id'])
    expected=[(task['task_id'],sc['scenario_id'],method,plan,int(seed))
              for task in task_records for sc in frozen_scenarios(task,args.split,args.include_ineligible,args.max_scenarios,only_scenarios)
              for method in methods for plan in planners for seed in seeds]
    if not expected:raise ValueError('Selected split has no scenarios')
    if len(expected)!=len(set(expected)):raise ValueError('Expected repeated task/scenario seed keys')
    config={'stage':'3.0','source':'road_vrp_7 July only','cost_file_sha256':road.cost_file_hash,
            'edge_file_sha256':road.edge_file_hash,
            'tasks_manifest_sha256':sha256(Path(args.task_dir)/'manifest.json'),
            'task_sha256':{p.name:sha256(p) for p in tasks_files},
            'split':args.split,'only_eligible':not args.include_ineligible,
            'methods':methods,'planners':planners,'seeds':seeds,'wall_sec':args.wall_sec,
            'iterations':args.max_iterations,'od_step_min':args.od_step_min,'od_hours':args.od_hours,
            'max_tasks':args.max_tasks,'max_scenarios':args.max_scenarios,
            'requested_scenario_ids':sorted(only_scenarios) if only_scenarios else None,
            'selected_scenario_ids':sorted({x[1] for x in expected}),
            'expected_count':len(expected),'historical_label_usage':'oracle planning uses complete historical future; hindsight only'}
    out.mkdir(parents=True,exist_ok=True)
    assert_existing_config(out,config,args.resume)
    done=inspect_resume(out)
    if not done:
        if (out/'metrics.csv').exists():raise ValueError('Empty/inconsistent existing CSV; manual correction needed')
        with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:csv.DictWriter(f,fieldnames=COLS).writeheader()
        (out/'solutions.jsonl').touch()
    cache=Path(args.od_cache_dir)
    tic=time.perf_counter()
    completed=0
    for task in task_records:
        for sc in frozen_scenarios(task,args.split,args.include_ineligible,args.max_scenarios,only_scenarios):
            jobs=[(m,p,s) for m in methods for p in planners for s in seeds if (task['task_id'],sc['scenario_id'],m,p,s) not in done]
            if not jobs:continue
            od_t=time.perf_counter()
            od=OD(road,task,sc,step_min=args.od_step_min,hours=args.od_hours,cache_dir=cache)
            precomp=time.perf_counter()-od_t
            print(f'OD {task["task_id"]} {sc["scenario_id"]} jobs={len(jobs)} cache={od.cache_hit}',flush=True)
            for method,planner,seed in jobs:
                obj=Objective(task,od,planner)
                result=solve(method,task,obj,seed=seed,wall=args.wall_sec,max_iter=args.max_iterations)
                exact=road.replay(task,result['routes'],sc['departure_local'])
                row={'task_id':task['task_id'],'scenario_id':sc['scenario_id'],'date':sc['date'],
                     'split':sc['split'],'eligible':bool(sc['eligible']),
                     'departure_local':sc['departure_local'],'solver':method,'planner':planner,
                     'seed':seed,'ex_post_road_travel_sec':exact['ex_post_road_travel_sec'],
                     'ex_post_total_vehicle_elapsed_sec':exact['ex_post_total_vehicle_elapsed_sec'],
                     'observed_edge_share':exact['observed_edge_share'],
                     'surrogate_travel_sec':result['planned_cost'],'initial_surrogate_travel_sec':result['initial_cost'],
                     'construct_wall_sec':result['construct_wall_sec'],'search_wall_sec':result['search_wall_sec'],
                     'od_precompute_wall_sec':precomp if completed==0 else 0.,
                     'total_solver_wall_sec':result['construct_wall_sec']+result['search_wall_sec'],
                     'iterations':result['iterations'],'score_calls':obj.score_calls,
                     'vehicle_loads':','.join(map(str,exact['vehicle_loads'])),
                     'num_vehicles_used':exact['num_vehicles_used'],'od_cache_hit':od.cache_hit}
                sol={'task_id':task['task_id'],'scenario_id':sc['scenario_id'],
                     'solver':method,'planner':planner,'seed':seed,
                     'routes':result['routes'],'recorded_cost_sec':exact['ex_post_road_travel_sec'],
                     'departure_local':sc['departure_local'],'diagnostics':result['diagnostics']}
                # Append solutions before metrics. In case of crash, reject incomplete pair on resume.
                with (out/'solutions.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(sol,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                with (out/'metrics.csv').open('a',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=COLS);w.writerow(row);f.flush();os.fsync(f.fileno())
                completed+=1
                print(f'  {method:13} {planner:6} seed={seed} ex-post={exact["ex_post_road_travel_sec"]:.1f}s planned={result["planned_cost"]:.1f}s search={result["search_wall_sec"]:.2f}s',flush=True)
    final=inspect_resume(out)
    if set(final)!=set(expected):raise AssertionError(f'Incomplete experiment grid: expected {len(expected)}, got {len(final)}')
    report={'status':'COMPLETE','expected_count':len(expected),'actual_count':len(final),
            'cost_sha256':road.cost_file_hash,'split':args.split,'include_ineligible':args.include_ineligible,
            'total_run_elapsed_sec':time.perf_counter()-tic,'note':'Oracle hindsight only; all final roads replayed on same July FIFO hybrid costs'}
    write_json(out/'benchmark_summary.json',report)
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--split',choices=['train','validation','test'],default='validation')
    p.add_argument('--include-ineligible',action='store_true',help='Stress test only; include QC-rejected scenarios')
    p.add_argument('--methods',default=','.join(METHODS))
    p.add_argument('--planners',default='static,oracle')
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--max-tasks',type=int,default=None)
    p.add_argument('--max-scenarios',type=int,default=None)
    p.add_argument('--scenario-ids',default='',help='Optional comma-separated exact frozen scenario IDs')
    p.add_argument('--wall-sec',type=float,default=5.)
    p.add_argument('--max-iterations',type=int,default=5000)
    p.add_argument('--od-step-min',type=int,default=20)
    p.add_argument('--od-hours',type=float,default=5.)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.wall_sec<0 or a.max_iterations<1 or a.od_step_min<1 or a.od_hours<1:raise ValueError('Invalid budgets')
    run(a)
if __name__=='__main__':main()
