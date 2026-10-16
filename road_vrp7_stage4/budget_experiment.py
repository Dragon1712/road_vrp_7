#!/usr/bin/env python3
"""Stage4 validation-only compute-budget ablations, July 2025 TD-CVRP.

Uses exactly the Stage3 July Road/OD/Objective/task contracts. Never edits
Stage3, never trains on or optimizes to the held-out test scenarios.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'road_vrp7_stage3'))
from road_vrp import Road, OD, Objective, legal, sha256, write_json, read_json  # noqa: E402
from solvers import (greedy, perturb, vnd, copied, choose_destroy, insert_removed,
                     VRPState, BothBudget)  # noqa: E402
from benchmark import load_tasks, frozen_scenarios  # noqa: E402

VARIANTS = ('sa_timecool', 'vnd_multistart', 'alns_timecool')
KEY = ('task_id', 'scenario_id', 'solver', 'planner', 'seed')
COLS = ['task_id','scenario_id','date','split','solver','planner','seed',
        'ex_post_road_travel_sec','ex_post_total_vehicle_elapsed_sec',
        'observed_edge_share','initial_surrogate_travel_sec','surrogate_travel_sec',
        'construct_wall_sec','search_wall_sec','iterations','score_calls',
        'accepted_moves','improvements','restarts','vehicle_loads',
        'num_vehicles_used','od_cache_hit']


def time_temperature(start_obj:float, elapsed:float, wall:float,
                     worse:float=.05, accept_prob:float=.75, final_ratio:float=.005):
    """Exponential schedule in WALL TIME, not number of iterations.

    At t=0 a 5% worse proposal would be accepted with probability 0.75.
    End temperature is a configurable fraction of initial temperature.
    """
    if not (start_obj>0 and wall>0 and worse>0 and 0<accept_prob<1
            and 0<final_ratio<=1):
        raise ValueError('Invalid time-cooling parameters')
    t0 = max(1e-8, (start_obj*worse) / (-math.log(accept_prob)))
    progress = min(1., max(0., elapsed / wall))
    return t0 * math.exp(math.log(final_ratio) * progress)


class WallTimeAnneal:
    """alns.accept acceptance protocol, time-based SA, independent of max_iters."""
    def __init__(self, *, initial_obj:float, wall_sec:float, start_sec:float,
                 worse:float=.05, accept_prob:float=.75, final_ratio:float=.005):
        self.initial_obj = float(initial_obj)
        self.wall_sec = float(wall_sec)
        self.start_sec = float(start_sec)
        self.worse = worse
        self.accept_prob = accept_prob
        self.final_ratio = final_ratio
        self.calls = 0
        self.accepted_worse = 0
        self.accepted = 0

    def __call__(self, rng, best, current, candidate):
        self.calls += 1
        old = float(current.objective())
        new = float(candidate.objective())
        if new <= old:
            self.accepted += 1
            return True
        temp = time_temperature(self.initial_obj, time.perf_counter()-self.start_sec,
                                self.wall_sec, self.worse,self.accept_prob,self.final_ratio)
        good = rng.random() < math.exp(max(-700.,(old-new)/temp))
        if good:
            self.accepted += 1
            self.accepted_worse += 1
        return bool(good)


def solve_variant(name, task, obj, seed, wall, max_iterations, final_ratio):
    if name not in VARIANTS:
        raise ValueError(f'Unknown variant {name}')
    if wall<=0 or max_iterations<1:
        raise ValueError('wall and max iterations must be >0')
    t0=time.perf_counter()
    initial,init_info=greedy(task,obj,seed)
    initial_cost=float(obj.cost(initial))
    construct=time.perf_counter()-t0
    st=time.perf_counter()
    deadline=st+wall
    rng=np.random.default_rng(seed)
    best=copied(initial)
    best_cost=initial_cost
    iters=accepted=improved=restarts=0
    diagnostics={'greedy_seed_fallback':bool(init_info.get('fallback_seed_routes',False))}

    if name=='sa_timecool':
        curr=copied(initial);curr_cost=initial_cost
        while iters < max_iterations and time.perf_counter()<deadline:
            candidate=perturb(curr,rng)
            iters+=1
            if not obj.feasible(candidate):continue
            val=float(obj.cost(candidate))
            temp=time_temperature(initial_cost,time.perf_counter()-st,wall,
                                  final_ratio=final_ratio)
            if val <= curr_cost or rng.random() < math.exp(max(-700.,(curr_cost-val)/temp)):
                accepted+=1; curr=candidate;curr_cost=val
            if val < best_cost-1e-8:
                best=copied(candidate);best_cost=val;improved+=1
    elif name=='vnd_multistart':
        # Avoid one deterministic VND local optimum taking only 0.15 s;
        # start fresh from perturbed incumbent until the SAME search deadline.
        while iters < max_iterations and time.perf_counter()<deadline:
            if restarts == 0:
                origin=copied(best)
            else:
                origin=None
                for _ in range(80):
                    attempt=copied(best)
                    for _ in range(int(rng.integers(2,7))):
                        attempt=perturb(attempt,rng)
                    if obj.feasible(attempt):
                        origin=attempt;break
                if origin is None:break
            left=max_iterations-iters
            candidate,used,diag=vnd(task,obj,origin,deadline,left)
            iters+=int(used)
            val=float(obj.cost(candidate))
            if val<best_cost-1e-8:
                best=copied(candidate);best_cost=val;improved+=1
            restarts+=1
            # VND can return without consuming one iteration at strict local optimum.
            if used==0 and time.perf_counter()<deadline:
                iters+=1
            if restarts>=max_iterations:break
    elif name=='alns_timecool':
        version=importlib.metadata.version('alns')
        if version!='7.0.0':
            raise RuntimeError('Requires pinned official alns==7.0.0, got '+version)
        from alns import ALNS
        from alns.select import RouletteWheel
        stop=BothBudget(max_iterations,wall)
        engine=ALNS(np.random.default_rng(seed))
        ops={'destroy':{},'repair':{}}
        fails=[0]

        def d_factory(label):
            def op(state,rng):
                ops['destroy'][label]=ops['destroy'].get(label,0)+1
                q=int(rng.integers(max(2,len(obj.dem)//10),max(3,len(obj.dem)//4+1)))
                partial,removed=choose_destroy(label,state.routes,obj,rng,q)
                return VRPState(tuple(tuple(r) for r in partial),obj,tuple(removed),state.routes)
            return op

        def r_factory(label):
            def op(state,rng):
                ops['repair'][label]=ops['repair'].get(label,0)+1
                candidate=insert_removed(state.routes,state.removed,obj,label,
                                         deadline=stop.deadline)
                if candidate is None or not obj.feasible(candidate):
                    candidate=copied(state.previous)
                    fails[0]+=1
                return VRPState(tuple(tuple(r) for r in candidate),obj)
            return op

        for label in ('random','worst','related'):
            engine.add_destroy_operator(d_factory(label),name=label)
        for label in ('greedy','regret2'):
            engine.add_repair_operator(r_factory(label),name=label)
        selector=RouletteWheel(scores=[8.,4.,1.,0.],decay=.8,
                               num_destroy=3,num_repair=2)
        initial_state=VRPState(tuple(tuple(r) for r in initial),obj)
        acceptance=WallTimeAnneal(initial_obj=initial_cost,wall_sec=wall,
                                  start_sec=st,final_ratio=final_ratio)
        result=engine.iterate(initial_state,selector,acceptance,stop)
        best=copied(result.best_state.routes)
        best_cost=float(obj.cost(best))
        iters=stop.count
        accepted=acceptance.accepted
        diagnostics.update({'accepted_worse':acceptance.accepted_worse,
                            'total_acceptance_calls':acceptance.calls,
                            'failed_repairs':fails[0],
                            'operator_usage':ops,'termination_reason':stop.reason,
                            'alns_version':version})
    search=time.perf_counter()-st
    legal(task,best)
    score=float(obj.cost(best))
    if score>initial_cost+1e-5:
        raise AssertionError('Final best solution worse than same greedy initial state')
    return {'routes':best,'planned_cost':score,'initial_cost':initial_cost,
            'search_wall_sec':search,'construct_wall_sec':construct,
            'iterations':iters,'accepted_moves':accepted,
            'improvements':improved,'restarts':restarts,'diagnostics':diagnostics}


def read_checkpoint(output):
    metrics=output/'metrics.csv';solutions=output/'solutions.jsonl'
    if not metrics.exists() and not solutions.exists():return set()
    if not metrics.exists() or not solutions.exists():
        raise ValueError('Checkpoint requires BOTH metrics and solutions files')
    import pandas as pd
    df=pd.read_csv(metrics)
    if not set(KEY).issubset(df.columns) or df.duplicated(list(KEY)).any():
        raise ValueError('Malformed or duplicate metric records')
    mk={tuple([str(getattr(r,k)) for k in KEY[:-1]]+[int(r.seed)]) for r in df.itertuples()}
    sk=[]
    for line in solutions.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        o=json.loads(line)
        sk.append(tuple([str(o[k]) for k in KEY[:-1]]+[int(o['seed'])]))
    if len(sk)!=len(set(sk)) or set(sk)!=mk:
        raise ValueError('Incomplete checkpoint pair (possible interrupted write); inspect before retry')
    return mk


def frozen_config(args,road,taskfiles,chosen,methods,planners,seeds):
    return {'stage':'4-budget-timecool-v1','source':'road_vrp_7 July validation ONLY',
            'cost_file_sha256':road.cost_file_hash,'edge_file_sha256':road.edge_file_hash,
            'tasks':{p.name:sha256(p) for p in taskfiles},
            'task_manifest_sha256':sha256(Path(args.task_dir)/'manifest.json'),
            'stage3_solvers_sha256':sha256(ROOT/'road_vrp7_stage3'/'solvers.py'),
            'stage3_road_sha256':sha256(ROOT/'road_vrp7_stage3'/'road_vrp.py'),
            'stage4_script_sha256':sha256(Path(__file__)),
            'split':'validation','wall_sec':args.wall_sec,
            'max_iterations':args.max_iterations,'final_temperature_ratio':args.final_ratio,
            'od_step_min':args.od_step_min,'od_hours':args.od_hours,
            'methods':methods,'planners':planners,'seeds':seeds,
            'scenario_ids':sorted(chosen),'max_tasks':args.max_tasks,
            'max_scenarios':args.max_scenarios,
            'expected_count':len(taskfiles)*len(chosen)*len(methods)*len(planners)*len(seeds),
            'metrics':'ex-post FIFO July hybrid road travel; ALNS oracle = hindsight only'}


def run(args):
    if args.wall_sec<=0 or args.max_iterations<1 or not(0<args.final_ratio<=1):
        raise ValueError('Invalid budget or final ratio')
    if args.od_hours<1 or args.od_step_min<=0:raise ValueError('Invalid OD params')
    if args.split!='validation':
        raise ValueError('Stage4 ablation is validation-only; DO NOT optimize on held-out test')
    variants=[s.strip() for s in args.variants.split(',') if s.strip()]
    plans=[s.strip() for s in args.planners.split(',') if s.strip()]
    seeds=[int(s) for s in args.seeds.split(',')]
    if not variants or len(set(variants))!=len(variants) or not set(variants)<=set(VARIANTS):
        raise ValueError('Bad solver variants')
    if not plans or len(set(plans))!=len(plans) or not set(plans)<={'static','oracle'}:
        raise ValueError('Bad planners')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Bad seed list')
    if 'alns_timecool' in variants:
        try:ver=importlib.metadata.version('alns')
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError('Official alns==7.0.0 is required for alns_timecool')
        if ver!='7.0.0':raise RuntimeError('Pinned alns==7.0.0 required')
    road=Road(args.cost_dir)
    if read_json(Path(args.cost_dir).parent/'stage2_validation.json')['validation']!='PASS':
        raise RuntimeError('Stage2 must be validated')
    taskfiles,tasks,manifest=load_tasks(args.task_dir)
    if manifest['cost_edge_index_sha256']!=road.edge_file_hash or manifest['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Stage3 tasks were built from different road costs')
    if args.max_tasks:
        taskfiles=taskfiles[:args.max_tasks];tasks=tasks[:args.max_tasks]
    chosen_ids=None;scenarios_by_task={}
    for task in tasks:
        legal(task,task['seed_routes'])
        scenarios=frozen_scenarios(task,'validation',False,args.max_scenarios)
        ids=[s['scenario_id'] for s in scenarios]
        if chosen_ids is None:chosen_ids=ids
        if ids!=chosen_ids:raise ValueError('Scenario ordering mismatch among tasks')
        scenarios_by_task[task['task_id']]=scenarios
    if not chosen_ids:raise ValueError('No eligible validation scenarios')
    expected={(task['task_id'],sc,met,plan,seed) for task in tasks for sc in chosen_ids
              for met in variants for plan in plans for seed in seeds}
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=frozen_config(args,road,taskfiles,chosen_ids,variants,plans,seeds)
    configfile=out/'run_config.json'
    if configfile.exists():
        if read_json(configfile)!=cfg:raise ValueError('Existing output has different CODE or CONFIG! Use another directory')
        if not args.resume:raise FileExistsError('Existing output: pass --resume if untouched')
    else:
        if (out/'metrics.csv').exists() or (out/'solutions.jsonl').exists():
            raise ValueError('No run_config with existing results')
        write_json(configfile,cfg)
    done=read_checkpoint(out)
    if not (out/'metrics.csv').exists():
        with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=COLS).writeheader()
        (out/'solutions.jsonl').touch()
    start=time.perf_counter()
    for task in tasks:
        for sc in scenarios_by_task[task['task_id']]:
            jobs=[(met,plan,seed) for met in variants for plan in plans for seed in seeds
                  if (task['task_id'],sc['scenario_id'],met,plan,seed) not in done]
            if not jobs:continue
            od=OD(road,task,sc,step_min=args.od_step_min,hours=args.od_hours,
                  cache_dir=args.od_cache_dir)
            print(f'OD {task["task_id"]} {sc["scenario_id"]} cache_hit={od.cache_hit} jobs={len(jobs)}',flush=True)
            for met,plan,seed in jobs:
                obj=Objective(task,od,plan)
                result=solve_variant(met,task,obj,seed,args.wall_sec,args.max_iterations,args.final_ratio)
                ex=road.replay(task,result['routes'],sc['departure_local'])
                row={'task_id':task['task_id'],'scenario_id':sc['scenario_id'],'date':sc['date'],
                     'split':'validation','solver':met,'planner':plan,'seed':seed,
                     'ex_post_road_travel_sec':ex['ex_post_road_travel_sec'],
                     'ex_post_total_vehicle_elapsed_sec':ex['ex_post_total_vehicle_elapsed_sec'],
                     'observed_edge_share':ex['observed_edge_share'],
                     'initial_surrogate_travel_sec':result['initial_cost'],
                     'surrogate_travel_sec':result['planned_cost'],
                     'construct_wall_sec':result['construct_wall_sec'],
                     'search_wall_sec':result['search_wall_sec'],
                     'iterations':result['iterations'],'score_calls':obj.score_calls,
                     'accepted_moves':result['accepted_moves'],'improvements':result['improvements'],
                     'restarts':result['restarts'],
                     'vehicle_loads':','.join(map(str,ex['vehicle_loads'])),
                     'num_vehicles_used':ex['num_vehicles_used'],'od_cache_hit':od.cache_hit}
                sol={k:row[k] for k in KEY}
                sol.update(routes=result['routes'],diagnostics=result['diagnostics'])
                with (out/'solutions.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(sol,ensure_ascii=False,allow_nan=False)+'\n')
                    f.flush();os.fsync(f.fileno())
                with (out/'metrics.csv').open('a',newline='',encoding='utf-8') as f:
                    csv.DictWriter(f,fieldnames=COLS).writerow(row)
                    f.flush();os.fsync(f.fileno())
                print(f'  {met:17} {plan:6} seed={seed} ex-post={ex["ex_post_road_travel_sec"]:.2f}s '
                      f'search={result["search_wall_sec"]:.3f}s iters={result["iterations"]}',flush=True)
    completed=read_checkpoint(out)
    if completed!=expected:
        raise AssertionError(f'Incomplete Stage4 grid {len(completed)}/{len(expected)}')
    summary={'status':'COMPLETE','expected':len(expected),'actual':len(completed),
             'elapsed_sec':time.perf_counter()-start,'stage2_sha256':road.cost_file_hash,
             'split':'validation','read_only_stage3_baseline':True}
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
    p.add_argument('--max-iterations',type=int,default=1000000)
    p.add_argument('--final-ratio',type=float,default=.005)
    p.add_argument('--od-step-min',type=int,default=20)
    p.add_argument('--od-hours',type=float,default=5.)
    p.add_argument('--max-tasks',type=int,default=None)
    p.add_argument('--max-scenarios',type=int,default=None)
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
