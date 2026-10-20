#!/usr/bin/env python3
"""Label captured CURRENT states using unchanged Step6-A v1 Section counterfactual machinery.

The frozen Step6-A v1 collector is imported but its source-state loader is
replaced with an explicit, audited current-state loader. Original v1 is NEVER
modified. Step6-A v2 labels are still retrospective Oracle training labels.
"""
from __future__ import annotations
import argparse, importlib.util, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'road_vrp7_step6a'/'section_dataset.py'
sys.path.insert(0,str(ROOT/'road_vrp7_stage3'))
sys.path.insert(0,str(ROOT/'road_vrp7_step5b_v3'))


def module_load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def prepare():
    v1=module_load(V1,'step6a_v1_frozen')
    frozen_cfg=json.loads((ROOT/'runs/chicago_july_v1/step6a/section_utility_1000_v1/run_config.json').read_text(encoding='utf-8'))
    expected=frozen_cfg['code_sha256']['road_vrp7_step6a/section_dataset.py']
    if v1.sha256(V1)!=expected:raise ValueError('Step6A v1 collector differs from its frozen July manifest')
    original_hashes=v1.code_hashes
    def hashes():
        return {**original_hashes(),
            'road_vrp7_step6a_v2/label_current.py':v1.sha256(Path(__file__)),
            'road_vrp7_step6a_v2/capture_current.py':v1.sha256(Path(__file__).with_name('capture_current.py'))}
    v1.code_hashes=hashes
    v1.VERSION='step6a-offline-v2'
    v1.load_snapshot_states=load_current_states
    return v1


def load_current_states(source_dir, source_solvers, max_tasks=None,
                        max_scenarios=None, target_times=None, allowed_seeds=None):
    path=Path(source_dir)/'current_states.jsonl'
    config=json.loads((Path(source_dir)/'run_config.json').read_text(encoding='utf-8'))
    summary=json.loads((Path(source_dir)/'benchmark_summary.json').read_text(encoding='utf-8'))
    if config.get('stage')!='step6a-current-source-v1' or summary.get('status')!='COMPLETE':
        raise ValueError('Not a complete Step6A v2 current-state source')
    if config.get('planner')!='static':
        raise ValueError('No Oracle-generated source states allowed')
    if summary['source_planner']!='static':raise ValueError('Source planner mismatch')
    allowed=set(source_solvers)
    if allowed!={'C3_topk'}:raise ValueError('Use --source-solvers C3_topk')
    tasks=set(sorted(Path(p).stem for p in config['tasks'])[:max_tasks]) if max_tasks else None
    scenes=set(config['scenarios'][:max_scenarios]) if max_scenarios else None
    target=set(target_times) if target_times is not None else None
    seeds=set(allowed_seeds) if allowed_seeds is not None else None
    states={};found=0
    with path.open(encoding='utf-8') as f:
        for line_index,line in enumerate(f,1):
            if not line.strip():continue
            r=json.loads(line);task_id,scid,seed=r['key'];found+=1
            if r.get('planner')!='static' or r.get('source_solver')!='C3_topk':
                raise ValueError('Unexpected source run planner/solver')
            if (tasks is not None and task_id not in tasks) or (scenes is not None and scid not in scenes):continue
            if seeds is not None and int(seed) not in seeds:continue
            for snap in r['sampled_states']:
                bucket=float(snap['target_sec'])
                if target is not None and bucket not in target:continue
                routes=tuple(tuple(int(c) for c in route) for route in snap['routes'])
                # Dedup by real solution, not by time/seed: prevent inflation.
                ident=hashlib.sha256(json.dumps([task_id,scid,routes],sort_keys=True).encode()).hexdigest()[:24]
                st={'state_id':ident,'task_id':task_id,'scenario_id':scid,
                    'routes':[list(x) for x in routes], 'snapshot_sec':bucket,
                    'source_solver':'C3_topk','source_planner':'static','seed':int(seed),
                    'source_line':line_index,
                    'capture_elapsed_sec':float(snap['elapsed_sec']),
                    'captured_nonbest':bool(snap['nonbest']),
                    'capture_gap_before_sec':float(snap['gap_before_sec'])}
                if ident not in states:states[ident]=st
    if found!=summary['expected_runs']:raise ValueError('Incomplete captured runs')
    if not states:raise ValueError('No accepted current states for selection')
    return list(states.values()),config,path


def main():
    v1=prepare()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--source-solvers',default='C3_topk')
    p.add_argument('--target-times',default='5,10,15,20,25,30')
    p.add_argument('--max-states',type=int,default=1000)
    p.add_argument('--max-tasks',type=int,default=10)
    p.add_argument('--max-scenarios',type=int,default=8)
    p.add_argument('--actions-per-state',type=int,default=8)
    p.add_argument('--states-seed',type=int,default=20261019)
    p.add_argument('--candidate-deadline-sec',type=float,default=3.)
    p.add_argument('--top-k',type=int,default=12)
    p.add_argument('--lookback-slots',type=int,default=6)
    p.add_argument('--od-step-min',type=int,default=20)
    p.add_argument('--od-hours',type=float,default=5.)
    p.add_argument('--enable-swap',action='store_true',default=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    v1.run(args)
if __name__=='__main__':main()
