#!/usr/bin/env python3
"""Time-dependent Chicago TD-CVRP30 Stage5-B v2 C0-C6, official ALNS 7.0.0.

All output directories are NEW, immutable and checksum pinned. July previously
inspected scenarios are development data, never an independent final test.
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
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_stage3'))
sys.path.insert(0,str(ROOT/'road_vrp7_step5b'))
from road_vrp import Road,OD,Objective,legal,read_json,write_json,sha256
from benchmark import load_tasks
from solvers import greedy,VRPState,BothBudget,choose_destroy,insert_removed,copied
from section_operators import SectionSelector as OldSelector,SectionState as OldPartial,repair_section as old_repair,PathSupportProxy
from section_v2 import Block,Partial,Selector as NewSelector,Stats,frozen,mutable,repair as new_repair
from time_selector import TimeUCB
from selectors_v3 import ControlledUCB
from snapshots_v3 import BestTrace,parse_times
from selectors_step5c import make_step5c_selector
import hashlib

VARIANTS=('C0_baseline','C1_original_section','C2_noop_guard','C3_topk',
          'C4_time_ucb','C5_gated','C6_shuffled','U0_gain','U1_walltime','U2_calibrated',
          'C7_alpha_ucb','C8_thompson','C9_lrw')
KEY=('task_id','scenario_id','solver','planner','seed')
COLS=('task_id','scenario_id','date','split','solver','planner','seed',
      'ex_post_road_travel_sec','ex_post_total_vehicle_elapsed_sec',
      'observed_edge_share','surrogate_travel_sec','initial_surrogate_travel_sec',
      'search_wall_sec','construct_wall_sec','iterations','score_calls',
      'od_precompute_sec','od_cache_hit','support_precompute_sec',
      'section_calls','section_changed','section_unchanged','section_improving',
      'section_new_best','section_accepted','section_candidate_exact',
      'section_noop_rejected','section_fallback','section_swap_attempts','section_swap_selected','section_destroy_sec',
      'section_repair_sec','section_traffic_change_rate',
      'section_actual_choice_change_rate','section_mean_support_proxy',
      'failed_repairs','solve_and_replay_wall_sec', 'anneal_iterations','cost_profile_sha256','snapshot_count')


def check_alns():
    try:ver=importlib.metadata.version('alns')
    except importlib.metadata.PackageNotFoundError as e:raise RuntimeError('Install alns==7.0.0') from e
    if ver!='7.0.0':raise RuntimeError(f'Expected alns==7.0.0; got {ver}')


class Audit:
    def __init__(self):
        self.destroy=defaultdict(lambda:{'calls':0,'seconds':0.})
        self.repair=defaultdict(lambda:{'calls':0,'seconds':0.})
        self.pairs=defaultdict(lambda:{'calls':0,'changed':0,'improving':0,'worsening':0,
                                        'accepted':0,'new_best':0,'saved_sec':0.})
        self.last_destroy='';self.last_repair='';self.failed=0
    def summary(self):
        return {'destroy':dict(self.destroy),'repair':dict(self.repair),
                'pairs':{f'{a}|{b}':v for (a,b),v in self.pairs.items()},
                'failed_repairs':self.failed}


def solve(name,task,obj,seed,wall,max_iter,*,support=None,traffic_weight=.45,top_k=12,
          ucb_c=.35,ucb_epsilon=.08,anneal_iterations=None,cost_profile=None,lrw_profile=None):
    """Shared official ALNS engine; only designated Section/selector varies."""
    check_alns()
    from alns import ALNS
    from alns.accept import SimulatedAnnealing
    from alns.select import RouletteWheel
    if name not in VARIANTS:raise ValueError(name)
    tic=time.perf_counter();initial,_=greedy(task,obj,seed)
    start_cost=float(obj.cost(initial));construction=time.perf_counter()-tic
    rng=np.random.default_rng(seed)
    engine=ALNS(rng);stop=BothBudget(max_iter,wall)
    audit=Audit();stats=Stats()
    old=OldSelector(obj,section_probability=0.)
    new=NewSelector(obj,support=support,traffic_weight=traffic_weight)
    ds=['random','worst','related']+(['section'] if name!='C0_baseline' else [])

    def make_destroy(kind):
        def destroy(state,random):
            t0=time.perf_counter()
            if kind!='section':
                q=int(random.integers(max(2,len(obj.dem)//10),max(3,len(obj.dem)//4+1)))
                routes,ids=choose_destroy(kind,state.routes,obj,random,q)
                part=VRPState(frozen(routes),obj,tuple(ids),state.routes)
            elif name in ('C3_topk','C4_time_ucb','C5_gated','C6_shuffled','U0_gain','U1_walltime','U2_calibrated','C7_alpha_ucb','C8_thompson','C9_lrw'):
                mode={'C3_topk':'static','C4_time_ucb':'static',
                      'C5_gated':'gated','C6_shuffled':'shuffled',
                      'U0_gain':'static','U1_walltime':'static','U2_calibrated':'static',
                      'C7_alpha_ucb':'static','C8_thompson':'static','C9_lrw':'static'}[name]
                part=new.destroy(mode,state.routes,random)
            else:
                # Frozen V1 section selector; force one contiguous Section for
                # comparable repairs C1-C4; V1 30% double-block is NOT copied.
                part=old.destroy('section_static',state.routes,random)
            a=audit.destroy[kind];a['calls']+=1;a['seconds']+=time.perf_counter()-t0
            audit.last_destroy=kind
            return part
        return destroy

    def make_repair(method):
        def repair(state,random):
            t0=time.perf_counter()
            if audit.last_destroy!='section':
                candidate=insert_removed(state.routes,state.removed,obj,method,deadline=stop.deadline)
                if candidate is None or not obj.feasible(candidate):
                    audit.failed+=1;candidate=copied(state.previous)
            elif name=='C1_original_section':
                candidate,diag=old_repair(state,obj,method,deadline=stop.deadline,enable_swap=True)
                stats.add('repair_calls')
                if diag.get('strategy') in ('fallback','none'):
                    stats.add('repair_fallback');audit.failed+=1
            else:
                if isinstance(state,OldPartial):
                    p=Partial(frozen(state.routes),obj,tuple(Block(tuple(b.customers),b.original_route,b.original_position)
                                                            for b in state.blocks),frozen(state.previous))
                elif isinstance(state,Partial):p=state
                else:raise TypeError('Unknown partial Section state')
                assert len(p.blocks)<=1, 'All C2-C6 expected single-block Section'
                # C2 uses exact exhaustive search (remove the no-op only).
                # C3-6 use cheap static ordering followed by exact time-aware
                # evaluation of the top-K feasible WHOLE-block candidates.
                k=None if name=='C2_noop_guard' else top_k
                candidate,reason=new_repair(p,obj,stop.deadline,stats,top_k=k,nontrivial=True,
                                            enable_swap=name in ('C3_topk','C4_time_ucb','C5_gated','C6_shuffled',
                                                                 'U0_gain','U1_walltime','U2_calibrated','C7_alpha_ucb','C8_thompson','C9_lrw'))
                if reason!='moved':audit.failed+=1
            if not obj.feasible(candidate):raise AssertionError('Invalid repaired CVRP solution')
            a=audit.repair[method];a['calls']+=1;a['seconds']+=time.perf_counter()-t0
            audit.last_repair=method
            return VRPState(frozen(candidate),obj)
        return repair

    for kind in ds:engine.add_destroy_operator(make_destroy(kind),name=kind)
    for method in ('greedy','regret2'):engine.add_repair_operator(make_repair(method),name=method)
    awareness=name in ('C4_time_ucb','C5_gated','C6_shuffled','U0_gain','U1_walltime','U2_calibrated','C7_alpha_ucb','C8_thompson','C9_lrw')
    if name in ('C7_alpha_ucb','C8_thompson','C9_lrw'):
        selector=make_step5c_selector(name,seed,lrw_profile,len(ds),2)
    elif name in ('U0_gain','U1_walltime','U2_calibrated'):
        selector=ControlledUCB(len(ds),2,mode=name,calibration=cost_profile,c=ucb_c,epsilon=ucb_epsilon)
    elif awareness:
        selector=TimeUCB(len(ds),2,c=ucb_c,epsilon=ucb_epsilon)
    else:
        selector=RouletteWheel(scores=[8.,4.,1.,0.],decay=.8,num_destroy=len(ds),num_repair=2)
    initial_state=VRPState(frozen(initial),obj)
    anneal_iterations=max_iter if anneal_iterations is None else int(anneal_iterations)
    if anneal_iterations<1:raise ValueError('anneal_iterations must be positive')
    anneal=SimulatedAnnealing.autofit(init_obj=max(start_cost,1e-6),worse=.05,
                                      accept_prob=.75,num_iters=anneal_iterations)
    trace=BestTrace(initial_state.routes,start_cost)

    class Acceptance:
        def __call__(self,rand,best,current,candidate):
            before=float(current.objective());now=float(candidate.objective());minimum=float(best.objective())
            changed=candidate.routes!=current.routes
            accepted=bool(anneal(rand,best,current,candidate))
            new_best=now<minimum-1e-7
            if new_best:trace.note_best(now,candidate.routes)
            pair=audit.pairs[(audit.last_destroy,audit.last_repair)]
            pair['calls']+=1;pair['changed']+=int(changed)
            pair['improving']+=int(now<before-1e-7)
            pair['worsening']+=int(now>before+1e-7)
            pair['new_best']+=int(new_best)
            pair['accepted']+=int(accepted)
            pair['saved_sec']+=before-now
            if audit.last_destroy=='section':
                stats.add('changed' if changed else 'unchanged')
                stats.add('improving' if now<before-1e-7 else ('worsening' if now>before+1e-7 else 'tied'))
                stats.add('new_best',int(new_best));stats.add('accepted',int(accepted))
                stats.add('accepted_changed',int(accepted and changed))
            if awareness:selector.record_candidate(before,now)
            return accepted

    trace.begin()
    t0=time.perf_counter();result=engine.iterate(initial_state,selector,Acceptance(),stop)
    wall_used=time.perf_counter()-t0
    best=copied(result.best_state.routes);legal(task,best)
    planned=float(obj.cost(best))
    if planned>start_cost+1e-5:raise AssertionError('Returned solution worse than initial')
    section=stats.summary()
    if name in ('C1_original_section','C2_noop_guard'):
        o=old.diagnostics()
        section['destroy_calls']=o['calls']
        section['destroy_sec']=audit.destroy['section']['seconds'] if name!='C0_baseline' else 0.
    elif name!='C0_baseline':
        n=new.stats.summary()
        for k in ('destroy_calls','destroy_sec','traffic_change_rate','actual_choice_change_rate','avg_selected_support'):
            section[k]=n[k]
    section['repair_calls']=audit.destroy.get('section',{}).get('calls',0)
    return {'routes':best,'initial_cost':start_cost,'planned_cost':planned,
            'search_wall_sec':wall_used,'construct_wall_sec':construction,
            'iterations':stop.count,'termination_reason':stop.reason,
            'trace':trace,'anneal_iterations':anneal_iterations,
            'stats':section,'audit':audit.summary(),
            'selector':selector.diagnostics() if awareness else {'type':'roulette_wheel'}}


def _keys(row):return tuple(str(row[k]) if k!='seed' else int(row[k]) for k in KEY)


def checkpoint(out):
    p=out/'metrics.csv';s=out/'solutions.jsonl';sn=out/'snapshot_records.jsonl'
    if not p.exists() and not s.exists() and not sn.exists():return set()
    if not p.exists() or not s.exists() or not sn.exists():raise ValueError('Incomplete three-file checkpoint')
    import pandas as pd
    df=pd.read_csv(p)
    if not set(KEY).issubset(df.columns):raise ValueError('Checkpoint key columns missing')
    a=[_keys(dict(x)) for x in df.to_dict(orient='records')]
    b=[_keys(json.loads(l)) for l in s.read_text(encoding='utf-8').splitlines() if l.strip()]
    c=[_keys(json.loads(l)) for l in sn.read_text(encoding='utf-8').splitlines() if l.strip()]
    if len(set(a))!=len(a) or len(set(b))!=len(b) or len(set(c))!=len(c) or set(a)!=set(b) or set(a)!=set(c):
        raise ValueError('Interrupted or inconsistent checkpoint; inspect CSV/JSONL before resume')
    return set(a)


def run(args):
    check_alns()
    variants=tuple(v.strip() for v in args.variants.split(',') if v.strip())
    planners=tuple(v.strip() for v in args.planners.split(',') if v.strip())
    seeds=tuple(int(x) for x in args.seeds.split(','))
    splits=tuple(v.strip() for v in args.splits.split(',') if v.strip())
    if not variants or len(set(variants))!=len(variants) or not set(variants)<=set(VARIANTS):raise ValueError('Bad variants')
    if not planners or len(set(planners))!=len(planners) or not set(planners)<={'static','oracle'}:raise ValueError('Bad planners')
    if not splits or not set(splits)<={'train','validation','test'}:raise ValueError('Bad splits')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Bad seeds')
    if args.wall_sec<=0 or args.max_iterations<1 or args.anneal_iterations<1 or args.top_k<1 or args.od_step_min<1 or args.od_hours<=0:
        raise ValueError('Bad budgets')
    snapshot_times=parse_times(args.snapshot_times)
    profile_path=Path(args.calibration_file) if args.calibration_file else None
    needs_profile=bool(set(variants)&{'U0_gain','U2_calibrated'})
    if needs_profile and (profile_path is None or not profile_path.is_file()):
        raise ValueError('U0/U2 requires --calibration-file. Run calibrate_cost.py on July development first.')
    cost_profile=json.loads(profile_path.read_text(encoding='utf-8')) if needs_profile else None
    profile_hash=sha256(profile_path) if needs_profile else None
    needs_lrw='C9_lrw' in variants
    lrw_path=Path(args.lrw_profile) if args.lrw_profile else None
    if needs_lrw and (lrw_path is None or not lrw_path.is_file()):
        raise ValueError('C9_lrw requires frozen --lrw-profile. Run learn_lrw.py on prior development logs.')
    lrw_profile=json.loads(lrw_path.read_text(encoding='utf-8')) if needs_lrw else None
    lrw_hash=sha256(lrw_path) if needs_lrw else None
    road=Road(args.cost_dir)
    if read_json(Path(args.cost_dir).parent/'stage2_validation.json').get('validation')!='PASS':
        raise ValueError('Stage2 validation not passed')
    files,tasks,manifest=load_tasks(args.task_dir)
    if manifest['cost_edge_index_sha256']!=road.edge_file_hash or manifest['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Road/task cost sha mismatch')
    if args.max_tasks:files=files[:args.max_tasks];tasks=tasks[:args.max_tasks]
    for task in tasks:legal(task,task['seed_routes'])
    requested=set(args.scenario_ids.split(',')) if args.scenario_ids else None
    scenarios=[s for s in tasks[0]['scenarios'] if s['split'] in splits and
               (args.include_ineligible or s['eligible']) and (requested is None or s['scenario_id'] in requested)]
    if args.max_scenarios:scenarios=scenarios[:args.max_scenarios]
    if not scenarios:raise ValueError('No scenarios')
    if requested is not None and {s['scenario_id'] for s in scenarios}!=requested:raise ValueError('Requested scenarios not found/eligible')
    for task in tasks[1:]:
        other={s['scenario_id']:s for s in task['scenarios']}
        if any(other.get(s['scenario_id'])!=s for s in scenarios):raise ValueError('Inconsistent task scenarios')
    expected={(str(t['task_id']),str(s['scenario_id']),v,p,int(seed)) for t in tasks for s in scenarios
              for v in variants for p in planners for seed in seeds}
    source=[Path(__file__),Path(__file__).with_name('section_v2.py'),Path(__file__).with_name('time_selector.py'),
            Path(__file__).with_name('selectors_v3.py'),Path(__file__).with_name('snapshots_v3.py'),
            Path(__file__).with_name('calibrate_cost.py'),Path(__file__).with_name('analyze_step5c.py'),
            Path(__file__).with_name('selectors_step5c.py'),Path(__file__).with_name('learn_lrw.py'),
            ROOT/'road_vrp7_step5b'/'section_operators.py',
            ROOT/'road_vrp7_stage3'/'solvers.py',ROOT/'road_vrp7_stage3'/'road_vrp.py']
    cfg={'stage':'step5c-v1','data_role':'July development only, prior test split labels NOT blind test',
         'cost_hash':road.cost_file_hash,'edge_hash':road.edge_file_hash,
         'task_manifest_hash':sha256(Path(args.task_dir)/'manifest.json'),
         'tasks':{f.name:sha256(f) for f in files},'source':{str(f.relative_to(ROOT) if f.is_relative_to(ROOT) else f):sha256(f) for f in source},
         'variants':variants,'planners':planners,'seeds':seeds,'splits':splits,
         'scenarios':[s['scenario_id'] for s in scenarios],'include_ineligible':args.include_ineligible,
         'max_tasks':args.max_tasks,'max_scenarios':args.max_scenarios,'scenario_ids':sorted(requested) if requested else None,
         'wall_sec':args.wall_sec,'max_iterations':args.max_iterations,
         'anneal_iterations':args.anneal_iterations,'snapshot_times':snapshot_times,
         'calibration_file':str(profile_path) if needs_profile else None,
         'calibration_sha256':profile_hash,'lrw_profile_file':str(lrw_path) if needs_lrw else None,
         'lrw_profile_sha256':lrw_hash,'od_step_min':args.od_step_min,
         'od_hours':args.od_hours,'traffic_weight':args.traffic_weight,'top_k':args.top_k,
         'ucb_c':args.ucb_c,'ucb_epsilon':args.ucb_epsilon,'expected_count':len(expected),
         'support_note':'Static-shortest-path observed-mask snapshot proxy; NOT dynamic-path observation'}
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    cf=out/'run_config.json'
    if cf.exists():
        if read_json(cf)!=json.loads(json.dumps(cfg)):raise ValueError('Changed manifest, code, config or budget; use NEW output directory')
        if not args.resume:raise FileExistsError('Output dir exists; use --resume')
    else:
        if any((out/x).exists() for x in ('metrics.csv','solutions.jsonl','snapshot_records.jsonl')):
            raise ValueError('Partial output without config')
        write_json(cf,cfg)
    done=checkpoint(out)
    if not done.issubset(expected):raise ValueError('Existing checkpoint contains unexpected keys')
    if not (out/'metrics.csv').exists():
        with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=COLS).writeheader()
        (out/'solutions.jsonl').touch()
        (out/'snapshot_records.jsonl').touch()
    for task in tasks:
        for sc in scenarios:
            todo=[(v,p,s) for v in variants for p in planners for s in seeds
                  if (task['task_id'],sc['scenario_id'],v,p,s) not in done]
            if not todo:continue
            t=time.perf_counter()
            od=OD(road,task,sc,step_min=args.od_step_min,hours=args.od_hours,cache_dir=args.od_cache_dir)
            od_sec=time.perf_counter()-t
            support=None;support_sec=0.
            if any(v in ('C5_gated','C6_shuffled') and p=='oracle' for v,p,_ in todo):
                t=time.perf_counter();support=PathSupportProxy(road,od,cache_dir=args.support_cache_dir)
                support_sec=time.perf_counter()-t
            for v,p,s in todo:
                t=time.perf_counter()
                obj=Objective(task,od,p)
                result=solve(v,task,obj,s,args.wall_sec,args.max_iterations,
                             support=support if p=='oracle' else None,
                             traffic_weight=args.traffic_weight,top_k=args.top_k,
                             ucb_c=args.ucb_c,ucb_epsilon=args.ucb_epsilon,
                             anneal_iterations=args.anneal_iterations,cost_profile=cost_profile,lrw_profile=lrw_profile)
                actual=road.replay(task,result['routes'],sc['departure_local'])
                m=result['stats'];spent=time.perf_counter()-t
                row={'task_id':task['task_id'],'scenario_id':sc['scenario_id'],
                     'date':sc['date'],'split':sc['split'],'solver':v,'planner':p,'seed':s,
                     'ex_post_road_travel_sec':actual['ex_post_road_travel_sec'],
                     'ex_post_total_vehicle_elapsed_sec':actual['ex_post_total_vehicle_elapsed_sec'],
                     'observed_edge_share':actual['observed_edge_share'],
                     'surrogate_travel_sec':result['planned_cost'],
                     'initial_surrogate_travel_sec':result['initial_cost'],
                     'search_wall_sec':result['search_wall_sec'],
                     'construct_wall_sec':result['construct_wall_sec'],
                     'iterations':result['iterations'],'score_calls':obj.score_calls,
                     'od_precompute_sec':od_sec,'od_cache_hit':od.cache_hit,
                     'support_precompute_sec':support_sec,
                     'section_calls':m['destroy_calls'],'section_changed':m['changed'],
                     'section_unchanged':m['unchanged'],'section_improving':m['improving'],
                     'section_new_best':m['new_best'],'section_accepted':m['accepted'],
                     'section_candidate_exact':m['repair_candidates_exact'],
                     'section_noop_rejected':m['repair_noop_rejected'],
                     'section_fallback':m['repair_fallback'],
                     'section_swap_attempts':m['swap_attempts'],'section_swap_selected':m['swap_selected'],
                     'section_destroy_sec':m['destroy_sec'],'section_repair_sec':m['repair_sec'],
                     'section_traffic_change_rate':m.get('traffic_change_rate',0.),
                     'section_actual_choice_change_rate':m.get('actual_choice_change_rate',0.),
                     'section_mean_support_proxy':m.get('avg_selected_support',0.),
                     'failed_repairs':result['audit']['failed_repairs'],'solve_and_replay_wall_sec':spent,
                     'anneal_iterations':args.anneal_iterations,'cost_profile_sha256':profile_hash or '',
                     'snapshot_count':len(snapshot_times)}
                sol={k:row[k] for k in KEY};sol.update(routes=result['routes'],
                      planned_cost=result['planned_cost'],section_stats=m,operator_audit=result['audit'],
                      selector_stats=result['selector'],termination_reason=result['termination_reason'],
                      trace_new_best_events=len(result['trace'].events)-1)
                # Same-run snapshots: replay outside the search budget.
                snaprows=[]
                for snapshot in result['trace'].snapshots(snapshot_times,result['search_wall_sec']):
                    item={k:snapshot[k] for k in ('target_sec','status','best_elapsed_sec','planned_cost')}
                    item['ex_post_road_travel_sec']=None
                    item['ex_post_total_vehicle_elapsed_sec']=None
                    item['observed_edge_share']=None
                    if snapshot['status']=='reached':
                        road_replay=road.replay(task,snapshot['routes'],sc['departure_local'])
                        item['ex_post_road_travel_sec']=road_replay['ex_post_road_travel_sec']
                        item['ex_post_total_vehicle_elapsed_sec']=road_replay['ex_post_total_vehicle_elapsed_sec']
                        item['observed_edge_share']=road_replay['observed_edge_share']
                    item['routes']=snapshot['routes']
                    snaprows.append(item)
                # Include offline snapshot replay in total solve-and-replay wall time.
                row['solve_and_replay_wall_sec']=time.perf_counter()-t
                snapshot_record={k:row[k] for k in KEY}
                snapshot_record['snapshots']=snaprows
                # Do not auto-repair half-appended result files: fail closed on --resume.
                with (out/'snapshot_records.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(snapshot_record,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                with (out/'solutions.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(sol,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                with (out/'metrics.csv').open('a',newline='',encoding='utf-8') as f:
                    csv.DictWriter(f,fieldnames=COLS).writerow(row);f.flush();os.fsync(f.fileno())
                print(f'{v:<22} {p:<6} seed={s} td_road={row["ex_post_road_travel_sec"]:.2f} '
                      f'iter={row["iterations"]} changed={row["section_changed"]}/{row["section_calls"]}',flush=True)
    done=checkpoint(out)
    if done!=expected:raise AssertionError(f'Expected {len(expected)} rows, found {len(done)}')
    write_json(out/'benchmark_summary.json',{'status':'COMPLETE','actual_count':len(done),
                                             'expected_count':len(expected),'code_role':'DEVELOPMENT'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--support-cache-dir',default='runs/chicago_july_v1/step5b_v2/support_cache')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--splits',default='validation')
    p.add_argument('--scenario-ids',default='')
    p.add_argument('--include-ineligible',action='store_true')
    p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--planners',default='oracle')
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--wall-sec',type=float,default=5.)
    p.add_argument('--max-iterations',type=int,default=20000)
    p.add_argument('--anneal-iterations',type=int,default=5000)
    p.add_argument('--snapshot-times',default='5,15,30,60')
    p.add_argument('--calibration-file',default='')
    p.add_argument('--lrw-profile',default='')
    p.add_argument('--od-step-min',type=int,default=20)
    p.add_argument('--od-hours',type=float,default=5.)
    p.add_argument('--max-tasks',type=int)
    p.add_argument('--max-scenarios',type=int)
    p.add_argument('--top-k',type=int,default=12)
    p.add_argument('--traffic-weight',type=float,default=.45)
    p.add_argument('--ucb-c',type=float,default=.35)
    p.add_argument('--ucb-epsilon',type=float,default=.08)
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
if __name__=='__main__':main()
