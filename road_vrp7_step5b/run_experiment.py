#!/usr/bin/env python3
"""Independent Stage5-B (July DEVELOPMENT) Section ablations on official ALNS 7.0.0.

Results are for hypothesis development. Because July 26-31 was inspected during
Stage3, it cannot be portrayed as a pristine held-out test for final method.
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
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_stage3'))
from road_vrp import Road,OD,Objective,legal,read_json,write_json,sha256  # noqa
from benchmark import load_tasks  # noqa
from solvers import choose_destroy,insert_removed,greedy,VRPState,BothBudget,copied  # noqa
from section_operators import SectionSelector,SectionState,PathSupportProxy,repair_section

VARIANTS=('baseline','section_random','section_static','section_static_probe','section_traffic','section_shuffled','at_rr_section')
KEY=('task_id','scenario_id','solver','planner','seed')
METRICS=['task_id','scenario_id','date','split','eligible','solver','planner','seed',
         'ex_post_road_travel_sec','ex_post_total_vehicle_elapsed_sec','observed_edge_share',
         'surrogate_travel_sec','initial_surrogate_travel_sec','construct_wall_sec',
         'search_wall_sec','iterations','score_calls','num_vehicles_used','vehicle_loads',
         'support_precompute_sec','od_precompute_sec','od_cache_hit','section_moves',
         'section_traffic_change_rate','section_mean_support_proxy',
         'section_mean_traffic_score','section_mean_static_score','section_swaps','failed_repairs']


def _installed_version():
    try:x=importlib.metadata.version('alns')
    except importlib.metadata.PackageNotFoundError as e:
        raise RuntimeError('Install official ALNS: python -m pip install alns==7.0.0') from e
    if x!='7.0.0':raise RuntimeError('Expected pinned official alns==7.0.0, got '+x)
    return x


class Audit:
    def __init__(self):
        self.destroy=defaultdict(lambda:{'calls':0,'sec':0.})
        self.repair=defaultdict(lambda:{'calls':0,'sec':0.})
        self.pairs=defaultdict(lambda:{'calls':0,'accepted':0,'new_best':0,
                                        'positive':0,'sum_candidate_delta':0.})
        self.last_destroy='';self.last_repair='';self.swaps=0;self.failed=0
    def asdict(self):
        return {'destroy':dict(self.destroy),'repair':dict(self.repair),
                'pairs':{a+'|'+b:v for (a,b),v in self.pairs.items()},
                'swaps':self.swaps,'failed_repairs':self.failed}


def solve(variant,task,obj,seed,wall,max_iter,*,support=None,traffic_weight=.45,support_weight=.15):
    if variant not in VARIANTS:raise ValueError(variant)
    from alns import ALNS
    from alns.accept import SimulatedAnnealing
    from alns.select import RouletteWheel
    start=time.perf_counter()
    initial,info=greedy(task,obj,seed)
    first_cost=float(obj.cost(initial))
    construction=time.perf_counter()-start
    s0=time.perf_counter()
    engine=ALNS(np.random.default_rng(seed));stop=BothBudget(max_iter,wall)
    telemetry=Audit()
    sec=SectionSelector(obj,support=support,traffic_weight=traffic_weight,support_weight=support_weight)
    traffic_destroy=None
    if variant=='at_rr_section':
        if obj.planner=='oracle':
            sys.path.insert(0,str(ROOT/'road_vrp7_step5a_v2'))
            from traffic_operators import TrafficDestroy
            traffic_destroy=TrafficDestroy(obj)
        else:
            # static must NOT access traffic signatures; exclude AT to preserve B3/B4 fairness
            traffic_destroy=None

    names=['random','worst','related']
    if variant!='baseline':names.append(variant if variant!='at_rr_section' else 'section_traffic')
    if traffic_destroy is not None:names.append('at_rr')

    def make_destroy(name):
        def destroy(state,rng):
            st=time.perf_counter()
            if name.startswith('section_'):
                result=sec.destroy(name,state.routes,rng)
            else:
                q=int(rng.integers(max(2,len(obj.dem)//10),max(3,len(obj.dem)//4+1)))
                if name=='at_rr':partial,removed=traffic_destroy(name,state.routes,rng,q)
                else:partial,removed=choose_destroy(name,state.routes,obj,rng,q)
                result=VRPState(tuple(map(tuple,partial)),obj,tuple(removed),state.routes)
            v=telemetry.destroy[name];v['calls']+=1;v['sec']+=time.perf_counter()-st
            telemetry.last_destroy=name
            return result
        return destroy

    def make_repair(name):
        def repair(state,rng):
            st=time.perf_counter()
            if isinstance(state,SectionState):
                routes,diagnostics=repair_section(state,obj,name,deadline=stop.deadline)
                if diagnostics['strategy']=='fallback':telemetry.failed+=1
                if diagnostics['strategy']=='swap':telemetry.swaps+=1
            else:
                routes=insert_removed(state.routes,state.removed,obj,name,deadline=stop.deadline)
                if routes is None or not obj.feasible(routes):
                    routes=copied(state.previous);telemetry.failed+=1
            result=VRPState(tuple(map(tuple,routes)),obj)
            v=telemetry.repair[name];v['calls']+=1;v['sec']+=time.perf_counter()-st
            telemetry.last_repair=name
            return result
        return repair

    for name in names:engine.add_destroy_operator(make_destroy(name),name)
    for name in ('greedy','regret2'):engine.add_repair_operator(make_repair(name),name)
    selector=RouletteWheel(scores=[8.,4.,1.,0.],decay=.8,num_destroy=len(names),num_repair=2)
    starting=VRPState(tuple(map(tuple,initial)),obj)
    anneal=SimulatedAnnealing.autofit(init_obj=max(float(starting.objective()),1e-6),
                                     worse=.05,accept_prob=.75,num_iters=max_iter)

    class LoggedAcceptance:
        def __call__(self,rng,best,current,candidate):
            old=float(current.objective());new=float(candidate.objective());bestv=float(best.objective())
            accepted=bool(anneal(rng,best,current,candidate))
            p=telemetry.pairs[(telemetry.last_destroy,telemetry.last_repair)]
            p['calls']+=1;p['accepted']+=int(accepted);p['positive']+=int(new<old-1e-6)
            p['new_best']+=int(new<bestv-1e-6)
            p['sum_candidate_delta']+=float(old-new)
            return accepted

    result=engine.iterate(starting,selector,LoggedAcceptance(),stop)
    runtime=time.perf_counter()-s0
    best=copied(result.best_state.routes)
    legal(task,best)
    value=float(obj.cost(best))
    if value>first_cost+1e-5:raise AssertionError('ALNS returned best worse than shared initial greedy')
    return {'routes':best,'planned_cost':value,'initial_cost':first_cost,
            'construct_wall_sec':construction,'search_wall_sec':runtime,'iterations':stop.count,
            'section_stats':sec.diagnostics(),'audit':telemetry.asdict(),
            'traffic_destroy_stats':traffic_destroy.diagnostics() if traffic_destroy is not None else {}}


def read_checkpoint(out):
    m=out/'metrics.csv';s=out/'solutions.jsonl'
    if not m.exists() and not s.exists():return set()
    if not m.exists() or not s.exists():raise ValueError('Missing metrics or solution file')
    import pandas as pd
    data=pd.read_csv(m)
    if not set(KEY).issubset(data.columns) or data.duplicated(list(KEY)).any():raise ValueError('Invalid metric keys')
    a={tuple([str(getattr(r,k)) for k in KEY[:-1]]+[int(r.seed)]) for r in data.itertuples()}
    b=[]
    for line in s.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line);b.append(tuple([str(row[k]) for k in KEY[:-1]]+[int(row['seed'])]))
    if len(set(b))!=len(b) or set(b)!=a:raise ValueError('Checkpoint partial writes; inspect before resume')
    return a


def _scenarios(tasks, splits, include_ineligible,limit,exact_ids=None):
    selected=[]
    for sc in tasks[0]['scenarios']:
        if sc['split'] not in splits or (not include_ineligible and not sc['eligible']):continue
        if exact_ids is not None and sc['scenario_id'] not in exact_ids:continue
        selected.append(sc)
    if limit:selected=selected[:limit]
    if not selected:raise ValueError('No eligible scenarios')
    for t in tasks[1:]:
        lookup={s['scenario_id']:s for s in t['scenarios']}
        if any(sc['scenario_id'] not in lookup or lookup[sc['scenario_id']]!=sc for sc in selected):
            raise ValueError('Different scenarios across tasks')
    return selected


def run(args):
    _installed_version()
    if args.wall_sec<=0 or args.max_iterations<1 or args.od_step_min<1 or args.od_hours<1:
        raise ValueError('Invalid budgets')
    variants=tuple(x.strip() for x in args.variants.split(',') if x.strip())
    planners=tuple(x.strip() for x in args.planners.split(',') if x.strip())
    seeds=tuple(int(x) for x in args.seeds.split(','))
    splits=tuple(x.strip() for x in args.splits.split(',') if x.strip())
    if not variants or len(set(variants))!=len(variants) or not set(variants).issubset(VARIANTS):raise ValueError('Invalid variants')
    if not planners or len(set(planners))!=len(planners) or not set(planners)<={'static','oracle'}:raise ValueError('Invalid planners')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Invalid seeds')
    if not splits or not set(splits)<= {'train','validation','test'}:raise ValueError('Invalid splits')
    road=Road(args.cost_dir)
    if read_json(Path(args.cost_dir).parent/'stage2_validation.json').get('validation')!='PASS':raise RuntimeError('Stage2 not validated')
    task_files,tasks,manifest=load_tasks(args.task_dir)
    if manifest['cost_edge_index_sha256']!=road.edge_file_hash or manifest['cost_edge_npz_sha256']!=road.cost_file_hash:raise ValueError('Task cost hash mismatch')
    if args.max_tasks:task_files=task_files[:args.max_tasks];tasks=tasks[:args.max_tasks]
    for task in tasks:legal(task,task['seed_routes'])
    exact_ids=set(args.scenario_ids.split(',')) if args.scenario_ids else None
    chosen=_scenarios(tasks,splits,args.include_ineligible,args.max_scenarios,exact_ids)
    if exact_ids is not None and {x['scenario_id'] for x in chosen}!=exact_ids:raise ValueError('Requested scenario-ids missing or QC-ineligible')
    expected={(t['task_id'],s['scenario_id'],v,p,int(seed)) for t in tasks for s in chosen
              for v in variants for p in planners for seed in seeds}
    out=Path(args.output_dir)
    config={'stage':'step5b-section-v1','model_role':'July exploratory development, including July26-31 if selected',
            'cost_sha256':road.cost_file_hash,'edge_sha256':road.edge_file_hash,
            'tasks_manifest_sha256':sha256(Path(args.task_dir)/'manifest.json'),
            'task_sha256':{p.name:sha256(p) for p in task_files},
            'source_sha256':{p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('section_operators.py')]},
            'stage3_code_sha256':{x:sha256(ROOT/'road_vrp7_stage3'/x) for x in ('road_vrp.py','solvers.py')},
            'stage5a_code_sha256':sha256(ROOT/'road_vrp7_step5a_v2'/'traffic_operators.py') if 'at_rr_section' in variants else None,
            'alns_version':_installed_version(),'splits':splits,'include_ineligible':bool(args.include_ineligible),
            'scenarios':[x['scenario_id'] for x in chosen],'requested_scenario_ids':sorted(exact_ids) if exact_ids else None,'variants':variants,'planners':planners,'seeds':seeds,
            'wall_sec':args.wall_sec,'max_iterations':args.max_iterations,
            'od_step_min':args.od_step_min,'od_hours':args.od_hours,'traffic_weight':args.traffic_weight,
            'support_weight':args.support_weight,'support_method':'static OSM shortest-path monitored-edge snapshot proxy',
            'max_tasks':args.max_tasks,'max_scenarios':args.max_scenarios,'expected_count':len(expected)}
    out.mkdir(parents=True,exist_ok=True)
    configfile=out/'run_config.json'
    if configfile.exists():
        if read_json(configfile)!=json.loads(json.dumps(config)):
            raise ValueError('Config or SOURCE checksum changed. Use a new output directory')
        if not args.resume:raise FileExistsError('Output exists; use --resume')
    else:
        if (out/'metrics.csv').exists() or (out/'solutions.jsonl').exists():raise ValueError('Partial result without config')
        write_json(configfile,config)
    done=read_checkpoint(out)
    if not (out/'metrics.csv').exists():
        with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:csv.DictWriter(f,fieldnames=METRICS).writeheader()
        (out/'solutions.jsonl').touch()
    cache_root=Path(args.od_cache_dir)
    wall_start=time.perf_counter()
    for task in tasks:
        for sc in chosen:
            jobs=[(variant,planner,seed) for variant in variants for planner in planners for seed in seeds
                  if (task['task_id'],sc['scenario_id'],variant,planner,seed) not in done]
            if not jobs:continue
            st=time.perf_counter()
            od=OD(road,task,sc,step_min=args.od_step_min,hours=args.od_hours,cache_dir=cache_root)
            od_sec=time.perf_counter()-st
            support=None;support_sec=0.
            if any(p=='oracle' and v in ('section_static_probe','section_traffic','section_shuffled','at_rr_section') for v,p,_ in jobs) and args.support_weight>0:
                st=time.perf_counter()
                support=PathSupportProxy(road,od,cache_dir=args.support_cache_dir)
                support_sec=time.perf_counter()-st
            print(f'OD task={task["task_id"]} scenario={sc["scenario_id"]} cache={od.cache_hit} jobs={len(jobs)} support_sec={support_sec:.2f}',flush=True)
            for variant,planner,seed in jobs:
                obj=Objective(task,od,planner)
                res=solve(variant,task,obj,seed,args.wall_sec,args.max_iterations,
                          support=support if planner=='oracle' else None,
                          traffic_weight=args.traffic_weight,support_weight=args.support_weight)
                actual=road.replay(task,res['routes'],sc['departure_local'])
                sec=res['section_stats']
                row={'task_id':task['task_id'],'scenario_id':sc['scenario_id'],'date':sc['date'],
                     'split':sc['split'],'eligible':sc['eligible'],'solver':variant,'planner':planner,'seed':seed,
                     'ex_post_road_travel_sec':actual['ex_post_road_travel_sec'],
                     'ex_post_total_vehicle_elapsed_sec':actual['ex_post_total_vehicle_elapsed_sec'],
                     'observed_edge_share':actual['observed_edge_share'],
                     'surrogate_travel_sec':res['planned_cost'],
                     'initial_surrogate_travel_sec':res['initial_cost'],
                     'construct_wall_sec':res['construct_wall_sec'],
                     'search_wall_sec':res['search_wall_sec'],'iterations':res['iterations'],
                     'score_calls':obj.score_calls,'num_vehicles_used':actual['num_vehicles_used'],
                     'vehicle_loads':','.join(map(str,actual['vehicle_loads'])),
                     'support_precompute_sec':support_sec,'od_precompute_sec':od_sec,
                     'od_cache_hit':od.cache_hit,'section_moves':sec['calls'],
                     'section_traffic_change_rate':sec['traffic_change_rate'],
                     'section_mean_support_proxy':sec['mean_support_proxy'],
                     'section_mean_traffic_score':sec['mean_traffic_score'],
                     'section_mean_static_score':sec['mean_static_score'],
                     'section_swaps':res['audit']['swaps'],
                     'failed_repairs':res['audit']['failed_repairs']}
                solution={k:row[k] for k in KEY}
                solution.update(routes=res['routes'],section_stats=sec,operator_audit=res['audit'],
                                traffic_destroy_stats=res['traffic_destroy_stats'],
                                ex_post_road_travel_sec=actual['ex_post_road_travel_sec'])
                with (out/'solutions.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(solution,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                with (out/'metrics.csv').open('a',encoding='utf-8',newline='') as f:
                    csv.DictWriter(f,fieldnames=METRICS).writerow(row);f.flush();os.fsync(f.fileno())
                print(f'  {variant:<18} {planner:<6} seed={seed} road={actual["ex_post_road_travel_sec"]:.2f}s iters={res["iterations"]} section={sec["calls"]}',flush=True)
    done=read_checkpoint(out)
    if done!=expected:raise AssertionError(f'Incomplete grid expected={len(expected)}, got={len(done)}')
    write_json(out/'benchmark_summary.json',{'status':'COMPLETE','expected_count':len(expected),'actual_count':len(done),
                                               'run_wall_sec':time.perf_counter()-wall_start,
                                               'blind_test_warning':'July26-31 already explored in Stage3; use separate unseen month for final test'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--support-cache-dir',default='runs/chicago_july_v1/step5b/od_support_cache')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--splits',default='validation',help='May use train,validation,test as JULY DEVELOPMENT; not fresh holdout')
    p.add_argument('--include-ineligible',action='store_true')
    p.add_argument('--scenario-ids',default='',help='Optional exact frozen scenario IDs, comma-separated')
    p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--planners',default='oracle')
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--wall-sec',type=float,default=5.)
    p.add_argument('--max-iterations',type=int,default=5000)
    p.add_argument('--od-step-min',type=int,default=20)
    p.add_argument('--od-hours',type=float,default=5.)
    p.add_argument('--traffic-weight',type=float,default=.45)
    p.add_argument('--support-weight',type=float,default=.15)
    p.add_argument('--max-tasks',type=int)
    p.add_argument('--max-scenarios',type=int)
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
