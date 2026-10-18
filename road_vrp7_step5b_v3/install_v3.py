#!/usr/bin/env python3
"""Install self-contained Step5-B v3 from the frozen v2 source WITHOUT altering v2.

Run from repository root.  Refuses code drift and existing destinations.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import py_compile
import shutil

ROOT=Path.cwd().resolve()
V2=ROOT/'road_vrp7_step5b_v2'
V3=ROOT/'road_vrp7_step5b_v3'
EXPECTED_RUNNER='1e69a4f9e206990775016c2ad01b5845422402f570b7bed737574456e65f8d7c'


def change(source,old,new):
    count=source.count(old)
    if count!=1:raise ValueError(f'Expected exactly 1 anchor; found {count}: {old[:170]!r}')
    return source.replace(old,new,1)


def build(source):
    s=source
    s=change(s,'from time_selector import TimeUCB\n',
             'from time_selector import TimeUCB\nfrom selectors_v3 import ControlledUCB\nfrom snapshots_v3 import BestTrace,parse_times\nimport hashlib\n')
    s=change(s,"'C4_time_ucb','C5_gated','C6_shuffled')\nKEY=", 
             "'C4_time_ucb','C5_gated','C6_shuffled','U0_gain','U1_walltime','U2_calibrated')\nKEY=")
    s=change(s,"'failed_repairs','solve_and_replay_wall_sec')", 
             "'failed_repairs','solve_and_replay_wall_sec', 'anneal_iterations','cost_profile_sha256','snapshot_count')")
    s=change(s,'          ucb_c=.35,ucb_epsilon=.08):',
             '          ucb_c=.35,ucb_epsilon=.08,anneal_iterations=None,cost_profile=None):')
    s=change(s,"elif name in ('C3_topk','C4_time_ucb','C5_gated','C6_shuffled'):",
             "elif name in ('C3_topk','C4_time_ucb','C5_gated','C6_shuffled','U0_gain','U1_walltime','U2_calibrated'):")
    s=change(s,"'C5_gated':'gated','C6_shuffled':'shuffled'}[name]",
             "'C5_gated':'gated','C6_shuffled':'shuffled',\n                      'U0_gain':'static','U1_walltime':'static','U2_calibrated':'static'}[name]")
    s=change(s,"enable_swap=name in ('C3_topk','C4_time_ucb','C5_gated','C6_shuffled'))",
             "enable_swap=name in ('C3_topk','C4_time_ucb','C5_gated','C6_shuffled',\n                                                                 'U0_gain','U1_walltime','U2_calibrated'))")
    s=change(s,"awareness=name in ('C4_time_ucb','C5_gated','C6_shuffled')\n    selector=TimeUCB(len(ds),2,c=ucb_c,epsilon=ucb_epsilon) if awareness else RouletteWheel(\n        scores=[8.,4.,1.,0.],decay=.8,num_destroy=len(ds),num_repair=2)",
             "awareness=name in ('C4_time_ucb','C5_gated','C6_shuffled','U0_gain','U1_walltime','U2_calibrated')\n    if name in ('U0_gain','U1_walltime','U2_calibrated'):\n        selector=ControlledUCB(len(ds),2,mode=name,calibration=cost_profile,c=ucb_c,epsilon=ucb_epsilon)\n    elif awareness:\n        selector=TimeUCB(len(ds),2,c=ucb_c,epsilon=ucb_epsilon)\n    else:\n        selector=RouletteWheel(scores=[8.,4.,1.,0.],decay=.8,num_destroy=len(ds),num_repair=2)")
    s=change(s,"initial_state=VRPState(frozen(initial),obj)\n    anneal=SimulatedAnnealing.autofit(init_obj=max(start_cost,1e-6),worse=.05,\n                                      accept_prob=.75,num_iters=max_iter)",
             "initial_state=VRPState(frozen(initial),obj)\n    anneal_iterations=max_iter if anneal_iterations is None else int(anneal_iterations)\n    if anneal_iterations<1:raise ValueError('anneal_iterations must be positive')\n    anneal=SimulatedAnnealing.autofit(init_obj=max(start_cost,1e-6),worse=.05,\n                                      accept_prob=.75,num_iters=anneal_iterations)\n    trace=BestTrace(initial_state.routes,start_cost)")
    s=change(s,"new_best=now<minimum-1e-7\n            pair=", 
             "new_best=now<minimum-1e-7\n            if new_best:trace.note_best(now,candidate.routes)\n            pair=")
    s=change(s,"t0=time.perf_counter();result=engine.iterate(initial_state,selector,Acceptance(),stop)",
             "trace.begin()\n    t0=time.perf_counter();result=engine.iterate(initial_state,selector,Acceptance(),stop)")
    s=change(s,"'iterations':stop.count,'termination_reason':stop.reason,\n            'stats':section,", 
             "'iterations':stop.count,'termination_reason':stop.reason,\n            'trace':trace,'anneal_iterations':anneal_iterations,\n            'stats':section,")
    s=change(s,"p=out/'metrics.csv';s=out/'solutions.jsonl'\n    if not p.exists() and not s.exists():return set()\n    if not p.exists() or not s.exists():raise ValueError('Incomplete checkpoint pair')",
             "p=out/'metrics.csv';s=out/'solutions.jsonl';sn=out/'snapshot_records.jsonl'\n    if not p.exists() and not s.exists() and not sn.exists():return set()\n    if not p.exists() or not s.exists() or not sn.exists():raise ValueError('Incomplete three-file checkpoint')")
    s=change(s,"b=[_keys(json.loads(l)) for l in s.read_text(encoding='utf-8').splitlines() if l.strip()]\n    if len(set(a))!=len(a) or len(set(b))!=len(b) or set(a)!=set(b):",
             "b=[_keys(json.loads(l)) for l in s.read_text(encoding='utf-8').splitlines() if l.strip()]\n    c=[_keys(json.loads(l)) for l in sn.read_text(encoding='utf-8').splitlines() if l.strip()]\n    if len(set(a))!=len(a) or len(set(b))!=len(b) or len(set(c))!=len(c) or set(a)!=set(b) or set(a)!=set(c):")
    s=change(s,"    if args.wall_sec<=0 or args.max_iterations<1 or args.top_k<1 or args.od_step_min<1 or args.od_hours<=0:",
             "    if args.wall_sec<=0 or args.max_iterations<1 or args.anneal_iterations<1 or args.top_k<1 or args.od_step_min<1 or args.od_hours<=0:")
    s=change(s,"    road=Road(args.cost_dir)",
             "    snapshot_times=parse_times(args.snapshot_times)\n    profile_path=Path(args.calibration_file) if args.calibration_file else None\n    needs_profile=bool(set(variants)&{'U0_gain','U2_calibrated'})\n    if needs_profile and (profile_path is None or not profile_path.is_file()):\n        raise ValueError('U0/U2 requires --calibration-file. Run calibrate_cost.py on July development first.')\n    cost_profile=json.loads(profile_path.read_text(encoding='utf-8')) if needs_profile else None\n    profile_hash=sha256(profile_path) if needs_profile else None\n    road=Road(args.cost_dir)")
    s=change(s,"source=[Path(__file__),Path(__file__).with_name('section_v2.py'),Path(__file__).with_name('time_selector.py'),\n            Path(__file__).with_name('analyze_step5b_v2.py'),ROOT/'road_vrp7_step5b'/'section_operators.py',",
             "source=[Path(__file__),Path(__file__).with_name('section_v2.py'),Path(__file__).with_name('time_selector.py'),\n            Path(__file__).with_name('selectors_v3.py'),Path(__file__).with_name('snapshots_v3.py'),\n            Path(__file__).with_name('calibrate_cost.py'),Path(__file__).with_name('analyze_v3.py'),\n            ROOT/'road_vrp7_step5b'/'section_operators.py',")
    s=change(s,"cfg={'stage':'step5b-v2'", "cfg={'stage':'step5b-v3'")
    s=change(s,"'wall_sec':args.wall_sec,'max_iterations':args.max_iterations,'od_step_min':args.od_step_min,",
             "'wall_sec':args.wall_sec,'max_iterations':args.max_iterations,\n         'anneal_iterations':args.anneal_iterations,'snapshot_times':snapshot_times,\n         'calibration_file':str(profile_path) if needs_profile else None,\n         'calibration_sha256':profile_hash,'od_step_min':args.od_step_min,")
    s=change(s,"if (out/'metrics.csv').exists() or (out/'solutions.jsonl').exists():raise ValueError('Partial output without config')",
             "if any((out/x).exists() for x in ('metrics.csv','solutions.jsonl','snapshot_records.jsonl')):\n            raise ValueError('Partial output without config')")
    s=change(s,"(out/'solutions.jsonl').touch()", "(out/'solutions.jsonl').touch()\n        (out/'snapshot_records.jsonl').touch()")
    s=change(s,"ucb_c=args.ucb_c,ucb_epsilon=args.ucb_epsilon)\n                actual=road.replay",
             "ucb_c=args.ucb_c,ucb_epsilon=args.ucb_epsilon,\n                             anneal_iterations=args.anneal_iterations,cost_profile=cost_profile)\n                actual=road.replay")
    s=change(s,"'failed_repairs':result['audit']['failed_repairs'],'solve_and_replay_wall_sec':spent}",
             "'failed_repairs':result['audit']['failed_repairs'],'solve_and_replay_wall_sec':spent,\n                     'anneal_iterations':args.anneal_iterations,'cost_profile_sha256':profile_hash or '',\n                     'snapshot_count':len(snapshot_times)}")
    s=change(s,"selector_stats=result['selector'],termination_reason=result['termination_reason'])",
             "selector_stats=result['selector'],termination_reason=result['termination_reason'],\n                      trace_new_best_events=len(result['trace'].events)-1)")
    s=change(s,"                # Do not auto-repair half-appended result files: fail closed on --resume.",
             "                # Same-run snapshots: replay outside the search budget.\n                snaprows=[]\n                for snapshot in result['trace'].snapshots(snapshot_times,result['search_wall_sec']):\n                    item={k:snapshot[k] for k in ('target_sec','status','best_elapsed_sec','planned_cost')}\n                    item['ex_post_road_travel_sec']=None\n                    item['ex_post_total_vehicle_elapsed_sec']=None\n                    item['observed_edge_share']=None\n                    if snapshot['status']=='reached':\n                        road_replay=road.replay(task,snapshot['routes'],sc['departure_local'])\n                        item['ex_post_road_travel_sec']=road_replay['ex_post_road_travel_sec']\n                        item['ex_post_total_vehicle_elapsed_sec']=road_replay['ex_post_total_vehicle_elapsed_sec']\n                        item['observed_edge_share']=road_replay['observed_edge_share']\n                    item['routes']=snapshot['routes']\n                    snaprows.append(item)\n                # Include offline snapshot replay in total solve-and-replay wall time.\n                row['solve_and_replay_wall_sec']=time.perf_counter()-t\n                snapshot_record={k:row[k] for k in KEY}\n                snapshot_record['snapshots']=snaprows\n                # Do not auto-repair half-appended result files: fail closed on --resume.\n                with (out/'snapshot_records.jsonl').open('a',encoding='utf-8') as f:\n                    f.write(json.dumps(snapshot_record,ensure_ascii=False,allow_nan=False)+'\\n');f.flush();os.fsync(f.fileno())")
    s=change(s,"    p.add_argument('--max-iterations',type=int,default=5000)",
             "    p.add_argument('--max-iterations',type=int,default=20000)\n    p.add_argument('--anneal-iterations',type=int,default=5000)\n    p.add_argument('--snapshot-times',default='5,15,30,60')\n    p.add_argument('--calibration-file',default='')")
    return s


def install():
    if not (ROOT/'road_vrp7_stage3'/'solvers.py').exists():
        raise FileNotFoundError('Run from original road_vrp_7 repository root')
    src=V2/'run_experiment.py'
    if not src.exists():raise FileNotFoundError(src)
    sha=hashlib.sha256(src.read_bytes()).hexdigest()
    if sha!=EXPECTED_RUNNER:
        raise ValueError(f'Unexpected Step5B v2 runner SHA256 {sha}; required {EXPECTED_RUNNER}. Refusing patch.')
    target=V3/'run_experiment.py'
    if target.exists():
        rebuilt=build(src.read_text(encoding='utf-8'))
        if target.read_text(encoding='utf-8')!=rebuilt:
            raise FileExistsError('V3 runner differs from frozen generated code; refusing overwrite')
        print('Existing V3 runner verified')
    else:
        code=build(src.read_text(encoding='utf-8'))
        compile(code,str(target),'exec')
        target.write_text(code,encoding='utf-8')
        print('Generated',target)
    for fn in ('section_v2.py','time_selector.py'):
        srcfile=V2/fn;dest=V3/fn
        if dest.exists():
            if dest.read_bytes()!=srcfile.read_bytes():raise FileExistsError(f'{dest} has changed')
        else:shutil.copy2(srcfile,dest)
    py_compile.compile(str(target),doraise=True)
    print('Installed independent Step5B-v3 runner; Step5B-v2 NOT modified.')

if __name__=='__main__':install()
