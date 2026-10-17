#!/usr/bin/env python3
"""Versioned, isolated Step5-B-v2 wall-budget study and strict cross-budget audit.

No edits to original ALNS solvers. Reuses run_experiment.py and its content-
addressed run_config safeguards. Never writes to legacy pilot/validation folders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

BASE = Path('runs/chicago_july_v1/step5b_v2/budget_studies')
COMP = [('C3_topk', 'C0_baseline'), ('C4_time_ucb', 'C3_topk'),
        ('C5_gated', 'C3_topk'), ('C5_gated', 'C4_time_ucb'),
        ('C5_gated', 'C0_baseline')]
KEY = ('task_id', 'scenario_id', 'planner', 'seed')


def digest(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix('.tmp')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')
    tmp.replace(path)


def read_json(path: Path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def budgets_parse(raw: str) -> list[float]:
    try:
        values = [float(v.strip()) for v in raw.split(',')]
    except ValueError as exc:
        raise ValueError('BUDGETS must be comma-separated positive numbers') from exc
    if not values or len(values) != len(set(values)) or any(
            not 0 < v <= 3600 or v != v or v in (float('inf'), float('-inf')) for v in values):
        raise ValueError('Budgets must be unique, finite and in (0,3600] seconds')
    return sorted(values)


def text_label(sec: float) -> str:
    return (f'{sec:g}'.replace('.', 'p') + 's')


def safe_name(name: str) -> str:
    if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_.-]{0,63}', name) or name in ('.', '..'):
        raise ValueError('study-name must be a simple directory name starting with a letter')
    return name


def csv_rows(file: Path):
    with file.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        # still create a predictable empty artifact
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def expected_from_config(cfg):
    task_ids = [p.removesuffix('.json') for p in cfg['tasks']]
    return {(t, scenario, variant, planner, int(seed))
            for t in task_ids for scenario in cfg['scenarios']
            for variant in cfg['variants'] for planner in cfg['planners']
            for seed in cfg['seeds']}


def verify_run(path: Path, budget: float, reference_config: dict | None = None):
    cfg = read_json(path / 'run_config.json')
    bench = read_json(path / 'benchmark_summary.json')
    if bench.get('status') != 'COMPLETE':
        raise ValueError(f'{path}: benchmark incomplete')
    if abs(float(cfg['wall_sec']) - budget) > 1e-9:
        raise ValueError(f'{path}: wrong wall budget: {cfg["wall_sec"]}')
    expected = expected_from_config(cfg)
    if bench['actual_count'] != len(expected) or cfg['expected_count'] != len(expected):
        raise ValueError(f'{path}: unexpected record count')
    rows = csv_rows(path / 'metrics.csv')
    found = [(r['task_id'], r['scenario_id'], r['solver'], r['planner'], int(r['seed'])) for r in rows]
    if len(rows) != len(expected) or len(set(found)) != len(found) or set(found) != expected:
        raise ValueError(f'{path}: incomplete or duplicated metrics grid')
    for r in rows:
        for col in ('ex_post_road_travel_sec', 'surrogate_travel_sec', 'search_wall_sec', 'iterations'):
            value = float(r[col])
            if value != value or value == float('inf') or value == float('-inf'):
                raise ValueError(f'{path}: nonfinite {col}')
        if float(r['ex_post_road_travel_sec']) <= 0:
            raise ValueError(f'{path}: invalid road cost')
    if reference_config is not None:
        # The only intentional difference between the original experiments is
        # wall_sec. All SHA256, scenarios, seeds and solver configurations match.
        for key in cfg:
            if key == 'wall_sec':
                continue
            if cfg[key] != reference_config.get(key):
                raise ValueError(f'Budget comparisons have unmatched config: {key}')
    return cfg, rows


def aggregate(study: Path, manifest: dict) -> None:
    from collections import defaultdict
    import numpy as np
    configurations = []
    all_rows = []
    for sec in manifest['budgets']:
        folder = study / f'{text_label(sec)}_iter{manifest["max_iterations"]}'
        cfg, rows = verify_run(folder, sec, configurations[0] if configurations else None)
        configurations.append(cfg)
        for r in rows:
            all_rows.append({'budget_sec':sec, **r})

    all_data = defaultdict(list)
    paired = []
    progress = []
    rng = np.random.default_rng(20261017)
    for x in all_rows:
        all_data[(x['budget_sec'], x['solver'], x['planner'])].append(x)
    summary = []
    for (sec, method, planner), arr in sorted(all_data.items()):
        arr_cost = np.array([float(a['ex_post_road_travel_sec']) for a in arr])
        arr_sur = np.array([float(a['surrogate_travel_sec']) for a in arr])
        arr_iter = np.array([float(a['iterations']) for a in arr])
        arr_wall = np.array([float(a['search_wall_sec']) for a in arr])
        summary.append({'budget_sec':sec, 'solver':method, 'planner':planner,
                        'records':len(arr), 'mean_road_sec':float(arr_cost.mean()),
                        'mean_planned_sec':float(arr_sur.mean()),
                        'mean_iterations':float(arr_iter.mean()),
                        'mean_search_sec':float(arr_wall.mean()),
                        'hit_iteration_limit_count':int(np.count_nonzero(arr_iter >= manifest['max_iterations']))})

    for sec in manifest['budgets']:
        for planner in manifest['params']['planners'].split(','):
            for newer, older in COMP:
                a = all_data.get((sec, newer, planner))
                b = all_data.get((sec, older, planner))
                if a is None or b is None:
                    continue
                index = {tuple(y[k] for k in KEY): y for y in b}
                if len(index) != len(b):
                    raise ValueError('Duplicate paired key')
                diffs, gains, tasks = [], [], defaultdict(list)
                for r in a:
                    k = tuple(r[c] for c in KEY)
                    if k not in index:
                        raise ValueError(f'Unmatched paired comparison: {newer} vs {older}')
                    base = float(index[k]['ex_post_road_travel_sec'])
                    v = base - float(r['ex_post_road_travel_sec'])
                    p = 100*v/base
                    diffs.append(v)
                    gains.append(p)
                    tasks[r['task_id']].append(p)
                if len(a) != len(b):
                    raise ValueError('Missing paired record')
                means = np.array([np.mean(vals) for vals in tasks.values()])
                samples = rng.choice(means, size=(3000,len(means)), replace=True).mean(axis=1)
                paired.append({'budget_sec':sec,'solver':newer,'baseline':older,'planner':planner,
                               'pairs':len(gains),'tasks':len(tasks),
                               'mean_gain_pct':float(np.mean(gains)),
                               'mean_saved_sec':float(np.mean(diffs)),
                               'paired_wins':int(np.count_nonzero(np.array(diffs)>1e-7)),
                               'ci_low_pct':float(np.quantile(samples,0.025)),
                               'ci_high_pct':float(np.quantile(samples,0.975))})

    for planner in manifest['params']['planners'].split(','):
        for solver in manifest['params']['variants'].split(','):
            for shorter, longer in zip(manifest['budgets'], manifest['budgets'][1:]):
                old = all_data[(shorter, solver, planner)]
                new = all_data[(longer, solver, planner)]
                base = {tuple(r[k] for k in KEY):r for r in old}
                if len(base)!=len(old) or len(new)!=len(old):
                    raise ValueError('Unmatched progress keys')
                save_road, save_plan = [], []
                for r in new:
                    k=tuple(r[t] for t in KEY)
                    if k not in base:raise ValueError('Missing same-seed budget comparison')
                    save_road.append(float(base[k]['ex_post_road_travel_sec'])-float(r['ex_post_road_travel_sec']))
                    save_plan.append(float(base[k]['surrogate_travel_sec'])-float(r['surrogate_travel_sec']))
                progress.append({'solver':solver,'planner':planner,'from_budget_sec':shorter,
                                 'to_budget_sec':longer,'matched_runs':len(new),
                                 'mean_saved_road_sec':float(np.mean(save_road)),
                                 'mean_saved_plan_sec':float(np.mean(save_plan)),
                                 'road_wins':int(np.count_nonzero(np.array(save_road)>1e-7)),
                                 'plan_wins':int(np.count_nonzero(np.array(save_plan)>1e-7))})

    out=study/'analysis'
    out.mkdir(exist_ok=True)
    write_csv(out/'budget_summary.csv',summary)
    write_csv(out/'budget_pairwise.csv',paired)
    write_csv(out/'budget_progression.csv',progress)
    atomic_json(out/'summary.json',{
        'complete':True, 'budgets_sec':manifest['budgets'],
        'runs':len(all_rows),'per_budget_runs':len(all_rows)//len(manifest['budgets']),
        'max_iterations':manifest['max_iterations'],
        'method_note':'fixed max_iterations and SimulatedAnnealing.autofit num_iters across budget arms',
        'interpretation':'longer-time best states can be better, equal or worse across separately rerun trajectories; road ex-post replay is a different cost metric',
        'comparison_note':'task-cluster bootstrap is exploratory; roads/speeds are Oracle hindsight'})
    print(f'Budget analysis complete: {out}')
    for row in summary:
        print(f"{row['budget_sec']:>5g}s {row['planner']:<6} {row['solver']:<16} road={row['mean_road_sec']:>9.2f}s iters={row['mean_iterations']:>8.0f}")
    print('Read budget_pairwise.csv and budget_progression.csv for causal checks.')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['run','analyze','all','dry-run'],default='run')
    p.add_argument('--project-root',default='.',help='Git repository root (default: cwd)')
    p.add_argument('--study-name',default=os.getenv('STUDY_NAME','section_budget_pilot_v1'))
    p.add_argument('--budgets',default=os.getenv('BUDGETS','5,15,30'))
    p.add_argument('--variants',default=os.getenv('METHODS','C0_baseline,C3_topk,C4_time_ucb,C5_gated'))
    p.add_argument('--planners',default=os.getenv('PLANNERS','oracle'))
    p.add_argument('--seeds',default=os.getenv('SEEDS','42,43'))
    p.add_argument('--splits',default=os.getenv('SPLITS','validation'))
    p.add_argument('--max-tasks',type=int,default=int(os.getenv('MAX_TASKS','3')))
    p.add_argument('--max-scenarios',type=int,default=int(os.getenv('MAX_SCENARIOS','2')))
    p.add_argument('--max-iterations',type=int,default=int(os.getenv('MAX_ITER','5000')))
    p.add_argument('--resume',action='store_true',default=os.getenv('RESUME','0')=='1')
    return p.parse_args()


def execute(args) -> None:
    root=Path(args.project_root).resolve()
    runner=root/'road_vrp7_step5b_v2/run_experiment.py'
    analyzer=root/'road_vrp7_step5b_v2/analyze_step5b_v2.py'
    if not runner.is_file() or not analyzer.is_file():
        raise FileNotFoundError('run_experiment.py/analyze_step5b_v2.py missing from project root')
    name=safe_name(args.study_name)
    study=root/BASE/name
    budget_list=budgets_parse(args.budgets)
    if args.max_iterations < 1 or args.max_tasks<1 or args.max_scenarios<1:
        raise ValueError('max_iterations, max_tasks and max_scenarios must be positive')
    for label,val in [('variants',args.variants),('planners',args.planners),('seeds',args.seeds),('splits',args.splits)]:
        if not val or any(not x.strip() for x in val.split(',')):
            raise ValueError(f'Invalid {label}')
    manifest={'study':'Step5B-v2 isolated wall-clock ablation','name':name,
              'budgets':budget_list, 'max_iterations':args.max_iterations,
              'params':{'variants':args.variants,'planners':args.planners,'seeds':args.seeds,
                        'splits':args.splits,'max_tasks':args.max_tasks,'max_scenarios':args.max_scenarios},
              'source_sha256':{'run_experiment.py':digest(runner),
                               'analyze_step5b_v2.py':digest(analyzer)}}
    manifest_path=study/'study_manifest.json'
    if args.action=='dry-run':
        print(json.dumps({'output_root':str(study),'manifest':manifest,
                          'run_dirs':[str(study/f'{text_label(sec)}_iter{args.max_iterations}') for sec in budget_list]},indent=2,ensure_ascii=False))
        return
    if args.action in ('run','all'):
        if manifest_path.exists():
            if read_json(manifest_path)!=manifest:
                raise ValueError('Study definition or source has changed; choose a NEW --study-name')
            if not args.resume:
                raise FileExistsError('Study already exists. Set RESUME=1 only for unchanged settings')
        else:
            if study.exists() and list(study.iterdir()):
                raise FileExistsError('Study directory contains data without study_manifest.json')
            study.mkdir(parents=True,exist_ok=True)
            atomic_json(manifest_path,manifest)
        for budget in budget_list:
            folder=study/f'{text_label(budget)}_iter{args.max_iterations}'
            cmd=[sys.executable,str(runner),'--output-dir',str(folder),
                 '--splits',args.splits,'--planners',args.planners,
                 '--variants',args.variants,'--seeds',args.seeds,
                 '--max-tasks',str(args.max_tasks),'--max-scenarios',str(args.max_scenarios),
                 '--wall-sec',f'{budget:g}','--max-iterations',str(args.max_iterations)]
            if args.resume and (folder/'run_config.json').exists():
                cmd.append('--resume')
            print('\nStarting budget',budget,'s in',folder,flush=True)
            subprocess.run(cmd,cwd=root,check=True)
    if args.action in ('analyze','all'):
        if not manifest_path.exists():raise FileNotFoundError('No study manifest; run budget first')
        saved=read_json(manifest_path)
        if args.action=='all' and saved!=manifest:
            raise ValueError('Study manifest changed')
        # 'analyze' always uses the SAVED manifest, not new environment defaults.
        if args.action=='analyze':manifest=saved
        for budget in manifest['budgets']:
            folder=study/f'{text_label(budget)}_iter{manifest["max_iterations"]}'
            verify_run(folder,budget)
            subprocess.run([sys.executable,str(analyzer),
                            '--experiment-dir',str(folder),
                            '--output-dir',str(folder/'analysis')],cwd=root,check=True)
        aggregate(study,manifest)


if __name__=='__main__':
    try:execute(arguments())
    except (ValueError,FileNotFoundError,FileExistsError,subprocess.CalledProcessError) as exc:
        print(f'BUDGET STUDY ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2) from exc
