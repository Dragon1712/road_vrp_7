#!/usr/bin/env python3
"""Collect the ACTUAL original-C3 Section draw during its own 30s ALNS search.

Records are sampled *after* FROZEN_C3_DESTROY has drawn the real action. No
second RNG draw, no teacher oracle, and no feature extraction inside search.
Capturing itself has a small measured wall-clock overhead, so this collection
is NOT a paired solver-performance benchmark.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r2','road_vrp7_step6b1_v2','road_vrp7_step6b2_r','road_vrp7_step6a','road_vrp7_step5b_v3','road_vrp7_stage3'):
    sys.path.insert(0,str(ROOT/folder))
from r2_core import state_key,block_action
from dataset import sha256
from road_vrp import Road,OD,Objective,legal,read_json
from benchmark import load_tasks
import section_v2
import run_experiment as v3

FROZEN_DESTROY=section_v2.Selector.destroy

@contextlib.contextmanager
def capture_hook(sink,task,sc,seed,stride,max_per_run):
    original=section_v2.Selector.destroy
    if original is not FROZEN_DESTROY or v3.NewSelector is not section_v2.Selector:
        raise RuntimeError('Frozen C3 selector was modified by another hook')
    count=[0];seen=set();perf=[0.]
    def hooked(self, mode, routes, rng):
        result=FROZEN_DESTROY(self,mode,routes,rng)   # exactly ONE true C3 draw
        count[0]+=1
        if mode!='static' or not result.blocks or count[0]%stride or len(sink)>=max_per_run:
            return result
        tick=time.perf_counter()
        skey=state_key(task['task_id'],sc['scenario_id'],routes)
        if skey not in seen:
            seen.add(skey)
            sink.append({'decision_id':hashlib.sha256(json.dumps([skey,seed,count[0]]).encode()).hexdigest()[:24],
                'state_id':skey,'task_id':task['task_id'],'scenario_id':sc['scenario_id'],
                'seed':int(seed),'section_call':count[0],
                'routes':[list(map(int,r)) for r in routes],
                'c3_action':block_action(result), 'source_mode':'ACTUAL_FROZEN_C3_RNG',
                'departure_local':sc['departure_local'],'date':sc['date']})
        perf[0]+=time.perf_counter()-tick
        return result
    section_v2.Selector.destroy=hooked
    try:yield (count,perf)
    finally:section_v2.Selector.destroy=original


def run(a):
    v3.check_alns()
    if a.wall_sec!=30 or a.max_iterations<=0 or a.stride<1 or a.max_per_run<1:
        raise ValueError('Only frozen 30s/positive budgets supported')
    out=Path(a.output_dir)
    if out.exists() and not a.resume:raise FileExistsError('Independent new output directory required; --resume to continue')
    src=read_json(Path(a.source_dir)/'run_config.json')
    for path in ['road_vrp7_step5b_v3/section_v2.py','road_vrp7_stage3/road_vrp.py']:
        if src.get('code_sha256',{}).get(path)!=sha256(ROOT/path):
            raise ValueError('Frozen source code hash mismatch: '+path)
    road=Road(a.cost_dir)
    if src['source_cost_hash']!=road.cost_file_hash or src['source_edge_hash']!=road.edge_file_hash:
        raise ValueError('Frozen road tensor hash mismatch')
    _,tasks,task_manifest=load_tasks(a.task_dir)
    if task_manifest['cost_edge_index_sha256']!=road.edge_file_hash or task_manifest['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Road/task manifest hash mismatch')
    if a.task_ids:
        want=set(a.task_ids.split(','));tasks=[t for t in tasks if t['task_id'] in want]
        if {t['task_id'] for t in tasks}!=want:raise ValueError('Unknown task ids')
    scenes=[s for s in tasks[0]['scenarios'] if s['split']==a.split and s['eligible']]
    if a.scenario_ids:
        want=set(a.scenario_ids.split(','));scenes=[s for s in scenes if s['scenario_id'] in want]
        if {s['scenario_id'] for s in scenes}!=want:raise ValueError('Unknown scenario ids')
    if not scenes:raise ValueError('No eligible scenarios in specified split')
    tasks=tasks[:a.max_tasks] if a.max_tasks else tasks
    scenes=scenes[:a.max_scenarios] if a.max_scenarios else scenes
    seeds=[int(x) for x in a.seeds.split(',')]
    if len(set(seeds))!=len(seeds):raise ValueError('Repeated seeds')
    runs=[(t,s,k) for t in tasks for s in scenes for k in seeds]
    if a.max_runs:runs=runs[:a.max_runs]
    manifest={'stage':'step6c1r2-real-C3-capture-v1','source_run_config_sha256':sha256(Path(a.source_dir)/'run_config.json'),
      'section_sha256':sha256(ROOT/'road_vrp7_step5b_v3/section_v2.py'),
      'solver_sha256':sha256(ROOT/'road_vrp7_step5b_v3/run_experiment.py'),
      'road_sha256':road.cost_file_hash,'edge_sha256':road.edge_file_hash,
      'collector_code_sha256':sha256(Path(__file__)),'split':a.split,'stride':a.stride,
      'max_per_run':a.max_per_run,'max_iterations':a.max_iterations,'anneal_iterations':a.anneal_iterations,
      'wall_sec':30,'top_k':a.top_k,'seeds':seeds,'runs':len(runs),
      'run_keys':[[t['task_id'],sc['scenario_id'],k] for t,sc,k in runs],
      'notes':['Frozen selector original.destroy performs the only C3 RNG draw.',
        'Capture wall-time overhead is included in solver search, so captured traces are NOT a cost benchmark.',
        'C3 may select a Section outside the downstream eight sampled candidates.',
        'Labels must be added RETROSPECTIVELY, never inside online ALNS.']}
    manifest_file=out/'manifest.json'
    if out.exists():
        if not manifest_file.is_file() or read_json(manifest_file)!=manifest:
            raise ValueError('Resume refused: C3 capture manifest/source/budget changed')
    else:
        out.mkdir(parents=True)
        manifest_file.write_text(json.dumps(manifest,indent=2)+'\n')
    run_dir=out/'runs';run_dir.mkdir(exist_ok=True)
    captured=[];diag=[]
    for task,sc,seed in runs:
        chk=run_dir/(f"{task['task_id']}__{sc['scenario_id']}__{seed}.json")
        if chk.is_file():
            record=read_json(chk)
            if record['key']!=[task['task_id'],sc['scenario_id'],seed]:
                raise ValueError('Resume checkpoint identity mismatch '+str(chk))
            samples=record['captured'];diagnostic=record['diagnostics']
        else:
            legal(task,task['seed_routes'])
            od=OD(road,task,sc,step_min=int(src['od_step_min']),hours=float(src['od_hours']),cache_dir=a.od_cache_dir)
            obj=Objective(task,od,'static');samples=[]
            with capture_hook(samples,task,sc,seed,a.stride,a.max_per_run) as counters:
                result=v3.solve('C3_topk',task,obj,seed,30,a.max_iterations,top_k=a.top_k,
                                anneal_iterations=a.anneal_iterations)
            calls=counters[0][0]
            diagnostic={'task_id':task['task_id'],'scenario_id':sc['scenario_id'],'seed':seed,
                         'samples':len(samples),'c3_calls':calls,'capture_added_wall_sec':counters[1][0],
                         'search_wall_sec':result['search_wall_sec'],'iterations':result['iterations']}
            record={'key':[task['task_id'],sc['scenario_id'],seed],
                    'captured':samples,'diagnostics':diagnostic}
            tmp=chk.with_suffix('.tmp')
            tmp.write_text(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
            tmp.replace(chk)
        captured.extend(samples);diag.append(diagnostic)
        print('C3CAPTURE',task['task_id'],sc['scenario_id'],seed,
              'states',len(samples),'calls',diagnostic['c3_calls'],flush=True)
    tmp=(out/'states.jsonl.tmp')
    with tmp.open('w') as dest:
        for item in captured:
            dest.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
    tmp.replace(out/'states.jsonl')
    (out/'runs.json').write_text(json.dumps(diag,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps({'status':'COMPLETE','runs':len(runs),'captured':len(captured),
                     'state_sha256':sha256(out/'states.jsonl')},indent=2)+'\n')
    if not captured:raise RuntimeError('No C3 decision-state captured')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--split',choices=('train','validation','test'),default='train')
    p.add_argument('--task-ids',default='');p.add_argument('--scenario-ids',default='')
    p.add_argument('--max-tasks',type=int);p.add_argument('--max-scenarios',type=int)
    p.add_argument('--seeds',default='42,43');p.add_argument('--max-runs',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stride',type=int,default=8);p.add_argument('--max-per-run',type=int,default=16)
    p.add_argument('--wall-sec',type=int,default=30);p.add_argument('--max-iterations',type=int,default=20000)
    p.add_argument('--anneal-iterations',type=int,default=5000);p.add_argument('--top-k',type=int,default=12)
    run(p.parse_args())
