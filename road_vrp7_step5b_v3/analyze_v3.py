#!/usr/bin/env python3
"""Strict v3 analysis: within-run snapshots, paired solver gains and UCB logs."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import math
import numpy as np

KEY=('task_id','scenario_id','solver','planner','seed')


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def write_csv(path,records):
    if not records:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def index(record):
    return tuple(str(record[k]) for k in KEY[:-1])+(int(record['seed']),)


def avg(a):return float(np.mean(a)) if a else None


def boot_task_ci(task_lists):
    vals=np.array([np.mean(x) for x in task_lists.values()])
    if len(vals)<2:return None,None
    rng=np.random.default_rng(20261018)
    sample=rng.choice(vals,size=(4000,len(vals)),replace=True).mean(axis=1)
    return float(np.quantile(sample,.025)),float(np.quantile(sample,.975))


def analyze(experiment_dir,output_dir):
    base=Path(experiment_dir);out=Path(output_dir)
    cfg=json.loads((base/'run_config.json').read_text())
    bench=json.loads((base/'benchmark_summary.json').read_text())
    if cfg.get('stage')!='step5b-v3' or bench.get('status')!='COMPLETE':
        raise ValueError('Not a completed step5b-v3 run')
    metrics=read_csv(base/'metrics.csv')
    solutions=[json.loads(l) for l in (base/'solutions.jsonl').read_text().splitlines() if l.strip()]
    snaps=[json.loads(l) for l in (base/'snapshot_records.jsonl').read_text().splitlines() if l.strip()]
    expected=set((str(t.removesuffix('.json')),sc,sol,p,int(seed))
                 for t in cfg['tasks'] for sc in cfg['scenarios']
                 for sol in cfg['variants'] for p in cfg['planners'] for seed in cfg['seeds'])
    series=[]
    for source,data in [('metrics',metrics),('solutions',solutions),('snapshots',snaps)]:
        keys=[index(r) for r in data]
        if len(keys)!=len(set(keys)) or set(keys)!=expected or len(keys)!=int(cfg['expected_count']):
            raise ValueError(f'{source}: expected grid incomplete or duplicated')
        series.append({index(r):r for r in data})
    metric,sols,snapshot=series
    times=[float(x) for x in cfg['snapshot_times']]
    rows=[]
    for key, rec in snapshot.items():
        m=metric[key];sol=sols[key]
        sr=rec['snapshots']
        if len(sr)!=len(times) or [float(x['target_sec']) for x in sr]!=times:
            raise ValueError('Snapshot grid mismatch '+str(key))
        last=float(m['initial_surrogate_travel_sec'])
        for x in sr:
            t=float(x['target_sec']);status=x['status']
            if status not in ('reached','censored'):raise ValueError('Unknown snapshot status')
            if status=='reached':
                if t>float(m['search_wall_sec'])+1.e-7:raise ValueError('Impossible future snapshot')
                if x['planned_cost'] is None or x['ex_post_road_travel_sec'] is None:
                    raise ValueError('Missing reached snapshot costs')
                if float(x['planned_cost'])>last+1.e-4:
                    raise ValueError('Same-trajectory best planned objective increased')
                last=float(x['planned_cost'])
                if any(not math.isfinite(float(x[k])) for k in ('planned_cost','ex_post_road_travel_sec')):
                    raise ValueError('Nonfinite snapshot cost')
            elif x.get('planned_cost') is not None or x.get('ex_post_road_travel_sec') is not None:
                raise ValueError('Censored snapshot must not fabricate cost')
            rows.append({**{k:v for k,v in zip(KEY,key)},'target_sec':t,'status':status,
                         'planned_cost':x.get('planned_cost'),
                         'road_cost':x.get('ex_post_road_travel_sec'),
                         'full_run_wall_sec':float(m['search_wall_sec']),
                         'full_run_iterations':int(m['iterations'])})
    out.mkdir(parents=True,exist_ok=True)
    summary=[];sums=defaultdict(list)
    for r in metrics:sums[(r['solver'],r['planner'])].append(r)
    for (solver,planner),rr in sorted(sums.items()):
        summary.append({'solver':solver,'planner':planner,'records':len(rr),
                        'mean_road_sec':avg([float(x['ex_post_road_travel_sec']) for x in rr]),
                        'mean_plan_sec':avg([float(x['surrogate_travel_sec']) for x in rr]),
                        'mean_iterations':avg([float(x['iterations']) for x in rr]),
                        'mean_wall_sec':avg([float(x['search_wall_sec']) for x in rr]),
                        'iteration_cap_count':sum(int(r['iterations'])>=int(cfg['max_iterations']) for r in rr)})
    ss=defaultdict(list)
    for r in rows:ss[(r['solver'],r['planner'],r['target_sec'])].append(r)
    summ_snap=[]
    for (solver,planner,t),rr in sorted(ss.items()):
        reached=[x for x in rr if x['status']=='reached']
        summ_snap.append({'solver':solver,'planner':planner,'target_sec':t,
                          'reached_count':len(reached),'censored_count':len(rr)-len(reached),
                          'mean_road_sec':avg([float(x['road_cost']) for x in reached]),
                          'mean_planned_sec':avg([float(x['planned_cost']) for x in reached])})
    paired=[]
    for planner in cfg['planners']:
        for t in times:
            for new,base_solver in [('C3_topk','C0_baseline'),('U0_gain','C3_topk'),
                                    ('U1_walltime','C3_topk'),('U2_calibrated','C3_topk'),
                                    ('U1_walltime','U0_gain'),('U2_calibrated','U0_gain'),
                                    ('U2_calibrated','U1_walltime')]:
                a={(r['task_id'],r['scenario_id'],r['seed']):r for r in rows if r['solver']==new and r['planner']==planner and r['target_sec']==t}
                b={(r['task_id'],r['scenario_id'],r['seed']):r for r in rows if r['solver']==base_solver and r['planner']==planner and r['target_sec']==t}
                if not a or not b:continue
                if set(a)!=set(b):raise ValueError('Unmatched paired keys')
                tasks=defaultdict(list);savings=[]
                for k in a:
                    x,y=a[k],b[k]
                    if x['status']!='reached' or y['status']!='reached':continue
                    y0=float(y['road_cost']);diff=y0-float(x['road_cost'])
                    tasks[x['task_id']].append(100.*diff/y0);savings.append(diff)
                lo,hi=boot_task_ci(tasks) if tasks else (None,None)
                paired.append({'planner':planner,'target_sec':t,'new':new,'reference':base_solver,
                               'matched_reached_count':len(savings),'censored_or_unmatched_count':len(a)-len(savings),
                               'mean_saved_road_sec':avg(savings),
                               'mean_gain_pct':avg([p for vals in tasks.values() for p in vals]),
                               'task_CI_lo':lo,'task_CI_hi':hi,'tasks':len(tasks),
                               'wins':sum(s>1.e-7 for s in savings)})
    progress=[]
    for solver in cfg['variants']:
        for planner in cfg['planners']:
            for before,after in zip(times,times[1:]):
                grouped=defaultdict(dict)
                for r in rows:
                    if r['solver']==solver and r['planner']==planner and r['target_sec'] in (before,after):
                        grouped[(r['task_id'],r['scenario_id'],r['seed'])][r['target_sec']]=r
                vals=[];plan=[]
                for grp in grouped.values():
                    x,y=grp.get(before),grp.get(after)
                    if x is None or y is None:raise ValueError('Missing snapshot progression keys')
                    if x['status']!='reached' or y['status']!='reached':continue
                    vals.append(float(x['road_cost'])-float(y['road_cost']))
                    v=float(x['planned_cost'])-float(y['planned_cost'])
                    if v < -1.e-4:raise ValueError('Same-trajectory planned cost increased')
                    plan.append(v)
                progress.append({'solver':solver,'planner':planner,'from_sec':before,'to_sec':after,
                                 'paired_reached':len(vals),'mean_road_saved_sec':avg(vals),
                                 'mean_planned_saved_sec':avg(plan),
                                 'road_wins':sum(x>1.e-7 for x in vals),
                                 'planned_wins':sum(x>1.e-7 for x in plan)})
    feedback=[]
    for k,sol in sols.items():
        selector=sol.get('selector_stats',{})
        if 'pair_counts' not in selector:continue
        counts=selector['pair_counts'];reward=selector['mean_reward']
        elapsed=selector.get('measured_pair_elapsed_sec')
        for i,d in enumerate(('random','worst','related','section')):
            for j,r in enumerate(('greedy','regret2')):
                feedback.append({**{field:val for field,val in zip(KEY,k)},
                                 'destroy':d,'repair':r,'count':counts[i][j],
                                 'mean_reward':reward[i][j],
                                 'sum_wall_sec':elapsed[i][j] if elapsed else None})
    write_csv(out/'solver_summary.csv',summary)
    write_csv(out/'snapshot_summary.csv',summ_snap)
    write_csv(out/'snapshot_pairwise.csv',paired)
    write_csv(out/'snapshot_progression.csv',progress)
    write_csv(out/'ucb_operator_feedback.csv',feedback)
    (out/'analysis_summary.json').write_text(json.dumps({
        'complete':True,'expected_runs':len(expected),'snapshots':len(rows),
        'all_snapshot_times':times,'anneal_iterations':cfg['anneal_iterations'],
        'max_iterations':cfg['max_iterations'],'data_role':'July development',
        'notes':['Snapshot replay performed after search, not charged to ALNS search wall time',
                 'Snapshots from the same run; censored future budgets excluded',
                 'Planned objective is nonincreasing within one trajectory; ex-post road need not be',
                 'UCB calibration derived from July developer data; U2 uses additive duration proxy',
                 'Bootstrap by task is exploratory for small task counts']},indent=2)+'\n')
    print('PASS',len(expected),'runs and',len(rows),'snapshots')
    print('\nSOLVERS\n',*summary,sep='\n')
    return {'solver':summary,'snapshot':summ_snap,'paired':paired,'progress':progress}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True)
    p.add_argument('--output-dir',default=None)
    a=p.parse_args();analyze(a.experiment_dir,a.output_dir or str(Path(a.experiment_dir)/'analysis'))
