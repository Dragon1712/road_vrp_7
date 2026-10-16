#!/usr/bin/env python3
"""Step5-A traffic-aware ALNS ablations on frozen July VALIDATION only.

Does not modify Stage3 or Stage4. Official ALNS 7.0.0 & Stage3 objective
and historical replay are required on the user's offline server.
"""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_stage3'))
from road_vrp import Road,OD,Objective,legal,sha256,read_json,write_json
from solvers import (greedy,copied,choose_destroy,insert_removed,VRPState,BothBudget)
from benchmark import load_tasks,frozen_scenarios
from traffic_operators import TrafficDestroy
from collections import defaultdict

# Fixed, PREDECLARED variants, no tuning on test set.
VARIANTS={
    'baseline':('random','worst','related'),
    'at_space':('random','worst','related','at_space'),
    'at_timeonly':('random','worst','related','at_timeonly'),
    'at_rr':('random','worst','related','at_rr'),
    'at_misaligned':('random','worst','related','at_misaligned'),
    'te_static_local':('random','worst','related','te_static_local'),
    'te_local':('random','worst','related','te_local'),
    'te_wr':('random','worst','related','te_wr'),
    'te_positive':('random','worst','related','te_positive'),
    'both':('random','worst','related','at_rr','te_wr'),
    'placebo_both':('random','worst','related','related_clone','worst_clone'),
}

KEY=('task_id','scenario_id','solver','planner','seed')
COLUMNS=['task_id','scenario_id','date','split','eligible','departure_local',
         'solver','planner','seed','ex_post_road_travel_sec',
         'ex_post_total_vehicle_elapsed_sec','observed_edge_share',
         'surrogate_travel_sec','initial_surrogate_travel_sec',
         'construct_wall_sec','search_wall_sec','iterations','score_calls',
         'failed_repairs','num_vehicles_used','vehicle_loads','od_cache_hit',
         'te_nonzero_externality_calls','te_static_fallback_calls',
         'selected_mean_externality_sec','selected_mean_local_gain_sec',
         'extra_traffic_route_cost_calls']


def assert_alns():
    try:version=importlib.metadata.version('alns')
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError('Requires OFFICIAL alns==7.0.0: python -m pip install alns==7.0.0')
    if version!='7.0.0':raise RuntimeError('Expected official alns==7.0.0, got '+version)
    return version


class OperatorTelemetry:
    """Pairs Destroy/Repair with the subsequent official acceptance decision.

    Times are INCLUDED in the 5-second search budget, across ALL arms.
    Attribution belongs to Destroy+Repair PAIRS, not to Destroy alone.
    """
    def __init__(self,max_events=120):
        self.groups=defaultdict(lambda:dict(calls=0,destroy_sec=0.,repair_sec=0.,
                    candidate_gain_sum=0.,positive_candidates=0,
                    accepted=0,accepted_worse=0,new_best=0))
        self.pairs=defaultdict(lambda:dict(calls=0,gain_sum=0.,positive_candidates=0,
                                            accepted=0,new_best=0))
        self.pending=None
        self.events=[];self.max_events=max_events
    def destroy(self,name,seconds):
        if self.pending is not None:raise AssertionError('Destroy called before previous candidate processed')
        self.pending={'destroy':name,'destroy_sec':float(seconds),'repair':None,'repair_sec':0.}
        g=self.groups[name];g['calls']+=1;g['destroy_sec']+=float(seconds)
    def repair(self,name,seconds):
        if self.pending is None:raise AssertionError('Repair without Destroy')
        self.pending['repair']=name;self.pending['repair_sec']=float(seconds)
        self.groups[self.pending['destroy']]['repair_sec']+=float(seconds)
    def accept(self,name,accepted,current_gain,best_gain):
        if not self.pending or not self.pending['repair']:
            raise AssertionError('Acceptance unmatched with a Destroy/Repair pair')
        d=self.pending; op=d['destroy'];pair=f'{op}|{d["repair"]}'
        g=self.groups[op];p=self.pairs[pair]
        for a in (g,p):
            if a is p:a['calls']+=1
            key='candidate_gain_sum' if a is g else 'gain_sum'
            a[key]+=float(current_gain)
            a['positive_candidates']+=int(current_gain>1e-7)
            a['accepted']+=int(accepted)
            a['new_best']+=int(bool(accepted) and best_gain>1e-7)
        g['accepted_worse']+=int(bool(accepted) and current_gain < -1e-7)
        if len(self.events)<self.max_events:
            self.events.append({'destroy':op,'repair':d['repair'],
                'destroy_sec':d['destroy_sec'],'repair_sec':d['repair_sec'],
                'candidate_gain_sec':float(current_gain),
                'new_best_gain_sec':float(max(0.,best_gain)) if accepted else 0.,
                'accepted':bool(accepted)})
        self.pending=None
    def summary(self):
        if self.pending is not None:raise AssertionError('Unmatched final ALNS proposal')
        return {'per_destroy':dict(self.groups),'per_pair':dict(self.pairs),
                'sample_events':self.events,'event_limit':self.max_events}


def solve_alns(variant,task,obj,seed,wall_sec,max_iterations):
    from alns import ALNS
    from alns.accept import SimulatedAnnealing
    from alns.select import RouletteWheel
    if variant not in VARIANTS:raise ValueError(variant)
    if wall_sec<=0 or max_iterations<1:raise ValueError('Invalid budgets')
    st=time.perf_counter()
    initial,info=greedy(task,obj,seed)
    start=VRPState(tuple(tuple(r) for r in initial),obj)
    initial_cost=float(start.objective())
    build_elapsed=time.perf_counter()-st
    traffic=TrafficDestroy(obj)
    stop=BothBudget(max_iterations,wall_sec)
    engine=ALNS(np.random.default_rng(seed))
    chosen=VARIANTS[variant]
    operator_calls={name:0 for name in chosen}
    repaired={'greedy':0,'regret2':0};fails=[0]
    telemetry=OperatorTelemetry()

    def make_destroy(name):
        def destroy(s,rng):
            op_start=time.perf_counter()
            operator_calls[name]+=1
            q=int(rng.integers(max(2,len(obj.dem)//10),max(3,len(obj.dem)//4+1)))
            if name in ('random','worst','related','related_clone','worst_clone'):
                label={'related_clone':'related','worst_clone':'worst'}.get(name,name)
                partial,removed=choose_destroy(label,s.routes,obj,rng,q)
            else:
                partial,removed=traffic(name,s.routes,rng,q)
            telemetry.destroy(name,time.perf_counter()-op_start)
            return VRPState(tuple(tuple(r) for r in partial),obj,tuple(removed),s.routes)
        return destroy

    def make_repair(name):
        def repair(s,rng):
            op_start=time.perf_counter()
            repaired[name]+=1
            candidate=insert_removed(s.routes,s.removed,obj,name,deadline=stop.deadline)
            if candidate is None or not obj.feasible(candidate):
                candidate=copied(s.previous);fails[0]+=1
            telemetry.repair(name,time.perf_counter()-op_start)
            return VRPState(tuple(tuple(r) for r in candidate),obj)
        return repair

    for op in chosen:engine.add_destroy_operator(make_destroy(op),name=op)
    for op in ('greedy','regret2'):engine.add_repair_operator(make_repair(op),name=op)
    selector=RouletteWheel(scores=[8.,4.,1.,0.],decay=.8,
                           num_destroy=len(chosen),num_repair=2)
    # Keep Stage3 baseline EXACT, including its per-iteration SA calibration.
    acceptance=SimulatedAnnealing.autofit(init_obj=max(initial_cost,1e-6),
                                         worse=.05,accept_prob=.75,num_iters=max_iterations)
    # Keep upstream SimulatedAnnealing as acceptance authority; this proxy
    # RECORDS outcomes without overriding the algorithm's decisions.
    best_reference=[initial_cost]
    class LoggedAcceptance:
        def __call__(self,rng,best,current,candidate):
            old=float(current.objective())
            cand=float(candidate.objective())
            historical=best_reference[0]
            accepted=bool(acceptance(rng,best,current,candidate))
            telemetry.accept('',accepted,old-cand,historical-cand)
            if accepted and cand<historical:
                best_reference[0]=cand
            return accepted
    search_start=time.perf_counter()
    result=engine.iterate(start,selector,LoggedAcceptance(),stop)
    search_elapsed=time.perf_counter()-search_start
    best=copied(result.best_state.routes)
    legal(task,best)
    value=float(obj.cost(best))
    if value>initial_cost+1e-5:raise AssertionError('Best solution worsened from greedy initial')
    diag={'initial_fallback':bool(info.get('fallback_seed_routes',False)),
          'operator_usage':{'destroy':operator_calls,'repair':repaired},
          'failed_repairs':fails[0], 'termination_reason':stop.reason,
          'alns_version':assert_alns(),
          'operator_contribution':telemetry.summary(),**traffic.diagnostics()}
    return {'routes':best,'planned_cost':value,'initial_cost':initial_cost,
            'search_wall_sec':search_elapsed,'construct_wall_sec':build_elapsed,
            'iterations':stop.count,'diagnostics':diag}


def _checkpoint(path):
    m=path/'metrics.csv';s=path/'solutions.jsonl'
    if not m.exists() and not s.exists():return set()
    if not m.exists() or not s.exists():raise ValueError('Only one checkpoint file present; repair before resume')
    import pandas as pd
    df=pd.read_csv(m,dtype={'task_id':str,'scenario_id':str,'solver':str,'planner':str})
    if not set(KEY).issubset(df.columns) or df.duplicated(list(KEY)).any():
        raise ValueError('Stage5 checkpoint metrics invalid or duplicated')
    k1={(str(r.task_id),str(r.scenario_id),str(r.solver),str(r.planner),int(r.seed)) for r in df.itertuples()}
    keys=[]
    for row in s.read_text(encoding='utf-8').splitlines():
        if row.strip():
            x=json.loads(row);keys.append((str(x['task_id']),str(x['scenario_id']),str(x['solver']),str(x['planner']),int(x['seed'])))
    if len(keys)!=len(set(keys)) or set(keys)!=k1:
        raise ValueError('Metrics / solutions mismatch, possibly interrupted write; inspect manually')
    return k1


def run(args):
    if args.split!='validation':
        raise ValueError('Step5-A is validation-only; July held-out test data is locked')
    if args.wall_sec<=0 or args.max_iterations<=0 or args.od_hours<1 or args.od_step_min<1:
        raise ValueError('Invalid wall/iteration/OD horizon settings')
    versions={'alns':assert_alns()}
    variants=[x.strip() for x in args.variants.split(',') if x.strip()]
    plans=[x.strip() for x in args.planners.split(',') if x.strip()]
    seeds=[int(x) for x in args.seeds.split(',')]
    if not variants or len(variants)!=len(set(variants)) or not set(variants)<=set(VARIANTS):raise ValueError('Unknown or repeated variants')
    if not plans or len(plans)!=len(set(plans)) or not set(plans)<={'static','oracle'}:raise ValueError('Unknown or repeated planners')
    if not seeds or len(seeds)!=len(set(seeds)):raise ValueError('Empty or repeated seeds')
    road=Road(args.cost_dir)
    stage2=read_json(Path(args.cost_dir).parent/'stage2_validation.json')
    if stage2.get('validation')!='PASS':raise RuntimeError('Stage2 road graph not validated')
    taskfiles,tasks,manifest=load_tasks(args.task_dir)
    if (manifest['cost_edge_index_sha256']!=road.edge_file_hash or
        manifest['cost_edge_npz_sha256']!=road.cost_file_hash):
        raise ValueError('Stage5 tasks and road costs do not match')
    if args.max_tasks:taskfiles,tasks=taskfiles[:args.max_tasks],tasks[:args.max_tasks]
    scenarios={};ids=None
    for task in tasks:
        legal(task,task['seed_routes'])
        selected=frozen_scenarios(task,'validation',False,args.max_scenarios)
        labels=[s['scenario_id'] for s in selected]
        if ids is None:ids=labels
        if labels!=ids:raise ValueError('Task-specific validation scenario ordering differs')
        scenarios[task['task_id']]=selected
    if not ids:raise ValueError('No eligible validation scenarios')
    expected={(t['task_id'],sc,m,p,seed) for t in tasks for sc in ids
              for m in variants for p in plans for seed in seeds}
    if len(expected)!=len(tasks)*len(ids)*len(variants)*len(plans)*len(seeds):raise ValueError('Duplicate experiment keys')
    config={'stage':'5A-traffic-operators-v2-audit','split':'validation','baseline_contract':'Stage3-official-alns-7.0.0',
            'alns_version':versions['alns'],'cost_file_sha256':road.cost_file_hash,
            'edge_file_sha256':road.edge_file_hash,
            'task_manifest_sha256':sha256(Path(args.task_dir)/'manifest.json'),
            'task_sha256':{p.name:sha256(p) for p in taskfiles},
            'source_sha256':{p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('traffic_operators.py'))},
            'stage3_sol_sha256':sha256(ROOT/'road_vrp7_stage3'/'solvers.py'),
            'stage3_cost_sha256':sha256(ROOT/'road_vrp7_stage3'/'road_vrp.py'),
            'variants':variants,'planners':plans,'seeds':seeds,
            'scenario_ids':ids,'wall_sec':args.wall_sec,'max_iterations':args.max_iterations,
            'od_step_min':args.od_step_min,'od_hours':args.od_hours,
            'max_tasks':args.max_tasks,'max_scenarios':args.max_scenarios,
            'expected_count':len(expected),
            'caveat':'Oracle uses actual future as historical hindsight; never deploy. Measured road share sparse.',
            'instrumentation':'all arms use acceptance+operator timers within 5s search budget'}
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    path=out/'run_config.json'
    if path.exists():
        if read_json(path)!=config:raise ValueError('Existing run config/code mismatch; choose new output directory')
        if not args.resume:raise FileExistsError('Existing output; pass --resume to continue')
    else:
        if (out/'metrics.csv').exists() or (out/'solutions.jsonl').exists():raise ValueError('Existing output without config')
        write_json(path,config)
    done=_checkpoint(out)
    if not (out/'metrics.csv').exists():
        with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=COLUMNS).writeheader()
        (out/'solutions.jsonl').touch()
    start=time.perf_counter()
    for task in tasks:
        for sc in scenarios[task['task_id']]:
            jobs=[(m,p,seed) for m in variants for p in plans for seed in seeds
                  if (task['task_id'],sc['scenario_id'],m,p,seed) not in done]
            if not jobs:continue
            od=OD(road,task,sc,step_min=args.od_step_min,hours=args.od_hours,
                  cache_dir=args.od_cache_dir)
            print(f"OD task={task['task_id']} sc={sc['scenario_id']} cache={od.cache_hit} jobs={len(jobs)}",flush=True)
            for variant,planner,seed in jobs:
                obj=Objective(task,od,planner)
                res=solve_alns(variant,task,obj,seed,args.wall_sec,args.max_iterations)
                replay=road.replay(task,res['routes'],sc['departure_local'])
                diag=res['diagnostics']
                row={'task_id':task['task_id'],'scenario_id':sc['scenario_id'],
                     'date':sc['date'],'split':sc['split'],'eligible':bool(sc['eligible']),
                     'departure_local':sc['departure_local'],'solver':variant,
                     'planner':planner,'seed':seed,
                     'ex_post_road_travel_sec':replay['ex_post_road_travel_sec'],
                     'ex_post_total_vehicle_elapsed_sec':replay['ex_post_total_vehicle_elapsed_sec'],
                     'observed_edge_share':replay['observed_edge_share'],
                     'surrogate_travel_sec':res['planned_cost'],'initial_surrogate_travel_sec':res['initial_cost'],
                     'construct_wall_sec':res['construct_wall_sec'],'search_wall_sec':res['search_wall_sec'],
                     'iterations':res['iterations'],'score_calls':obj.score_calls,
                     'failed_repairs':diag['failed_repairs'],
                     'num_vehicles_used':replay['num_vehicles_used'],
                     'vehicle_loads':','.join(map(str,replay['vehicle_loads'])),
                     'od_cache_hit':od.cache_hit,
                     'te_nonzero_externality_calls':diag['te_nonzero_externality_calls'],
                     'te_static_fallback_calls':diag['te_static_fallback_calls'],
                     'selected_mean_externality_sec':diag['selected_mean_externality_sec'],
                     'selected_mean_local_gain_sec':diag['selected_mean_local_gain_sec'],
                     'extra_traffic_route_cost_calls':diag['extra_traffic_route_cost_calls']}
                solution={k:row[k] for k in KEY}
                solution.update(routes=res['routes'],diagnostics=diag)
                # Guarded resume: if interrupted between files, stop instead of silently skipping.
                with (out/'solutions.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(solution,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                with (out/'metrics.csv').open('a',newline='',encoding='utf-8') as f:
                    csv.DictWriter(f,fieldnames=COLUMNS).writerow(row);f.flush();os.fsync(f.fileno())
                print(f"  {variant:14} {planner:6} seed={seed} road={row['ex_post_road_travel_sec']:.1f}s search={row['search_wall_sec']:.2f}s iter={row['iterations']}",flush=True)
    actual=_checkpoint(out)
    if actual!=expected:raise ValueError(f'Incomplete grid {len(actual)} / {len(expected)}')
    summary={'status':'COMPLETE','actual':len(actual),'expected':len(expected),
             'elapsed_sec':time.perf_counter()-start,'validation_only':True,
             'cost_file_sha256':road.cost_file_hash}
    write_json(out/'benchmark_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--split',choices=['validation'],default='validation')
    p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--planners',default='static,oracle')
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--wall-sec',type=float,default=5.)
    p.add_argument('--max-iterations',type=int,default=5000,
                   help='Same as frozen Stage3 official ALNS for matched comparison')
    p.add_argument('--od-step-min',type=int,default=20)
    p.add_argument('--od-hours',type=float,default=5.)
    p.add_argument('--max-tasks',type=int,default=None)
    p.add_argument('--max-scenarios',type=int,default=None)
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
