#!/usr/bin/env python3
"""Step6-A v2: capture ACCEPTED CURRENT states from frozen Step5-B v3 ALNS.

The ALNS solver is instrumented IN MEMORY with SHA-pinned, two-anchor changes:
  (1) optional state_sink in solve signature;
  (2) on accepted changed candidate, save a cheap current-state snapshot.
All objective, operator, acceptance, selection, stopping logic remains unchanged.
Snapshots themselves are not hindsight labels. Plan='static' by default so
source solutions do not use the future Oracle traffic tensor.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'road_vrp7_step5b_v3'/'run_experiment.py'
sys.path.insert(0,str(ROOT/'road_vrp7_step5b_v3'))
sys.path.insert(0,str(ROOT/'road_vrp7_stage3'))
from road_vrp import Road, OD, Objective, legal, read_json, write_json, sha256
from benchmark import load_tasks


def sha_bytes(b): return hashlib.sha256(b).hexdigest()


def instrument():
    raw = SOURCE.read_text(encoding='utf-8')
    sig = 'ucb_c=.35,ucb_epsilon=.08,anneal_iterations=None,cost_profile=None):'
    hook = "            if awareness:selector.record_candidate(before,now)\n            return accepted"
    for text in (sig,hook):
        if raw.count(text) != 1:
            raise RuntimeError('Frozen Step5-B v3 source incompatible with Step6-A v2 instrumentation')
    patch = raw.replace(sig, sig[:-2] + ',state_sink=None):')
    patch = patch.replace(hook,
        '            if accepted and changed and state_sink is not None:\n'
        '                state_sink.observe(candidate.routes,now,minimum,\n'
        '                                   time.perf_counter()-trace.start)\n' + hook)
    m = types.ModuleType('_step6a_v2_instrumented_v3')
    m.__file__ = str(SOURCE)
    m.__package__ = ''
    sys.modules[m.__name__] = m
    exec(compile(patch,str(SOURCE),'exec'),m.__dict__)
    return m, sha_bytes(patch.encode('utf-8'))


class AcceptedCollector:
    """One accepted, changed, preferentially non-best state per elapsed slot.

    Sampling is light and bounded; no TD counterfactuals are evaluated in search.
    """
    def __init__(self, wall:float, interval:float):
        if wall<=0 or interval<=0:raise ValueError('Budgets must be positive')
        self.wall=float(wall);self.interval=float(interval);self.buckets={}
    def observe(self,routes,cost,best_cost,elapsed):
        if elapsed<0 or elapsed>=self.wall: return
        slot=int(elapsed//self.interval)
        nonbest=bool(float(cost)>float(best_cost)+1e-7)
        previous=self.buckets.get(slot)
        if previous is not None and (previous['nonbest'] or not nonbest): return
        self.buckets[slot]={'bucket':slot,'target_sec':min(self.wall,(slot+1)*self.interval),
          'elapsed_sec':float(elapsed),'routes':[list(x) for x in routes],
          'planned_cost_sec':float(cost),'best_before_sec':float(best_cost),
          'gap_before_sec':float(cost)-float(best_cost),'nonbest':nonbest}
    def get(self): return [self.buckets[k] for k in sorted(self.buckets)]


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def cfg_of(a,road,manifest,taskfiles,scenarios,patch_sha,source_config):
    return {'stage':'step6a-current-source-v1',
        'data_role':'July development; frozen static-planner current states, no Oracle routing features',
        'planner':a.planner,'solvers':[a.solver], 'source_solves': 'Step5-B v3 in-memory accepted-state callback',
        'instrumented_sha256':patch_sha,'source_core_sha256':sha256(SOURCE),
        'source':source_config['source'],
        'cost_hash':road.cost_file_hash,'edge_hash':road.edge_file_hash,
        'task_manifest_sha256':sha256(Path(a.task_dir)/'manifest.json'),
        'tasks':{p.name:sha256(p) for p in taskfiles},
        'scenarios':[s['scenario_id'] for s in scenarios],
        'seeds':[int(x) for x in a.seeds.split(',')],
        'wall_sec':a.wall_sec, 'sample_interval_sec':a.interval_sec,
        'anneal_iterations':a.anneal_iterations, 'max_iterations':a.max_iterations,
        'top_k':a.top_k,'od_step_min':a.od_step_min,'od_hours':a.od_hours,
        'source_step5b_v3_run_config_hash':sha256(Path(a.frozen_source_dir)/'run_config.json'),
        'capture_code_hash':sha256(Path(__file__))}


def run(a):
    if a.solver != 'C3_topk':raise ValueError('Only frozen C3_topk is supported in first v2')
    if a.planner != 'static':raise ValueError('Only static planner allowed in prospective-input v2')
    if a.wall_sec<=0 or a.interval_sec<=0 or a.max_iterations<1 or a.anneal_iterations<1:raise ValueError('Invalid budgets')
    if a.wall_sec/a.interval_sec>100:raise ValueError('Too many slots')
    v3,patched_hash=instrument()
    source_cfg=read_json(Path(a.frozen_source_dir)/'run_config.json')
    if source_cfg.get('stage')!='step5b-v3':raise ValueError('Expected frozen Step5B v3 source config')
    if source_cfg['source']['road_vrp7_step5b_v3/run_experiment.py'] != sha256(SOURCE):
        raise ValueError('Step5B v3 source changed since frozen 1200-run study')
    v3.check_alns()
    road=Road(a.cost_dir)
    if source_cfg['cost_hash']!=road.cost_file_hash or source_cfg['edge_hash']!=road.edge_file_hash:
        raise ValueError('Road files changed since v3')
    files,tasks,manifest=load_tasks(a.task_dir)
    if manifest['cost_edge_index_sha256']!=road.edge_file_hash or manifest['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Road/task manifest differs')
    if a.max_tasks:files=files[:a.max_tasks];tasks=tasks[:a.max_tasks]
    for t in tasks:legal(t,t['seed_routes'])
    scenes=[s for s in tasks[0]['scenarios'] if s['split']==a.split and s['eligible']]
    if a.max_scenarios:scenes=scenes[:a.max_scenarios]
    if not scenes:raise ValueError('No eligible scenarios')
    for t in tasks[1:]:
        byid={x['scenario_id']:x for x in t['scenarios']}
        if any(byid.get(s['scenario_id'])!=s for s in scenes):
            raise ValueError('Scenarios differ between tasks')
    seeds=[int(x) for x in a.seeds.split(',')]
    if len(set(seeds)) != len(seeds):raise ValueError('Duplicate seeds')
    cfg=cfg_of(a,road,manifest,files,scenes,patched_hash,source_cfg)
    out=Path(a.output_dir)
    print(f'Step6-A v2 CURRENT states: {len(tasks)*len(scenes)*len(seeds)} runs, wall={a.wall_sec}s, slots={a.interval_sec}s',flush=True)
    if a.dry_run:return
    cf=out/'run_config.json'
    if cf.exists():
        if read_json(cf)!=cfg:raise ValueError('Changed configuration/source: must choose NEW directory')
        if not a.resume:raise FileExistsError('Existing output requires --resume')
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty result directory without matching manifest')
    else:atomic_json(cf,cfg)
    expected=[(t['task_id'],s['scenario_id'],seed) for t in tasks for s in scenes for seed in seeds]
    complete=0;all_samples=[]
    for task in tasks:
        for scene in scenes:
            missing=[seed for seed in seeds if not (out/'runs'/f"{task['task_id']}__{scene['scenario_id']}__{seed}.json").exists()]
            if missing:
                od=OD(road,task,scene,step_min=a.od_step_min,hours=a.od_hours,cache_dir=a.od_cache_dir)
            for seed in seeds:
                fname=out/'runs'/f"{task['task_id']}__{scene['scenario_id']}__{seed}.json"
                if fname.exists():
                    obj=read_json(fname)
                    if tuple(obj['key'])!=(task['task_id'],scene['scenario_id'],seed):
                        raise ValueError('Invalid run checkpoint')
                else:
                    objective=Objective(task,od,a.planner)
                    sink=AcceptedCollector(a.wall_sec,a.interval_sec)
                    res=v3.solve(a.solver,task,objective,seed,a.wall_sec,a.max_iterations,
                        top_k=a.top_k,anneal_iterations=a.anneal_iterations,state_sink=sink)
                    obj={'key':[task['task_id'],scene['scenario_id'],seed],
                         'source_solver':a.solver,'planner':a.planner,
                         'sampled_states':sink.get(),
                         'search_wall_sec':res['search_wall_sec'],
                         'iterations':res['iterations'],'termination_reason':res['termination_reason'],
                         'cost_before_sec':res['initial_cost'],'best_cost_sec':res['planned_cost']}
                    for s in obj['sampled_states']:legal(task,s['routes'])
                    atomic_json(fname,obj)
                    print(f"saved {fname.name}: {len(obj['sampled_states'])} current-state samples",flush=True)
                all_samples.append(obj)
                complete+=1
    if complete!=len(expected):raise AssertionError('Missed runs')
    # Create atomic archival JSONL after ALL per-run checkpoints exist.
    lines=[json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':')) for obj in all_samples]
    path=out/'current_states.jsonl';tmp=path.with_name(path.name+'.tmp')
    tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8');os.replace(tmp,path)
    total=sum(len(r['sampled_states']) for r in all_samples)
    nonbest=sum(s['nonbest'] for r in all_samples for s in r['sampled_states'])
    atomic_json(out/'benchmark_summary.json',{'status':'COMPLETE',
        'expected_runs':len(expected),'actual_runs':complete,'raw_current_states':total,
        'raw_nonbest_states':nonbest,'source_planner':a.planner,
        'note':'Accept changed CURRENT solutions; sampling does not run any repair oracle'})
    print(f'PASS: {total} captured states of which {nonbest} not a historical best (source planning={a.planner})')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-source-dir',default='runs/chicago_july_v1/step5b_v3/full30s_1200_v1/30s_anneal5000_cap20000')
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--split',default='validation')
    p.add_argument('--planner',default='static')
    p.add_argument('--solver',default='C3_topk')
    p.add_argument('--max-tasks',type=int,default=2)
    p.add_argument('--max-scenarios',type=int,default=2)
    p.add_argument('--seeds',default='42,43')
    p.add_argument('--wall-sec',type=float,default=30.)
    p.add_argument('--interval-sec',type=float,default=5.)
    p.add_argument('--anneal-iterations',type=int,default=5000)
    p.add_argument('--max-iterations',type=int,default=20000)
    p.add_argument('--top-k',type=int,default=12)
    p.add_argument('--od-step-min',type=int,default=20)
    p.add_argument('--od-hours',type=float,default=5.)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    run(p.parse_args())
if __name__=='__main__':main()
