#!/usr/bin/env python3
"""Strict paired control for instrumented unchanged C3.

A wall-clock-stopped pair cannot be proven trajectory-identical from terminal
cost alone. The deterministic fixed-iteration smoke *must* pass exact-route
and RNG-state tests before the 20k pilot.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean,median
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4a'))
from r4a_core import STAGE,sha256,write_json

FIELDS=('task_ids','scenario_ids','seeds','search_wall_sec','max_iterations',
        'anneal_iterations','top_k','source_sha256','road_sha256','edge_sha256',
        'solver_sha256','section_sha256','alns_version','split')

def read_lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]

def summary_pair(p,c,*,require_exact=False):
    m1=json.loads((Path(p)/'manifest.json').read_text());m2=json.loads((Path(c)/'manifest.json').read_text())
    if m1['stage']!=STAGE or m2['stage']!=STAGE or m1['arm']!='plain' or m2['arm']!='capture':
        raise ValueError('Wrong R4A arm identity')
    for k in FIELDS:
        if m1[k]!=m2[k]:raise ValueError('Mismatch '+k)
    r1=json.loads((Path(p)/'report.json').read_text());r2=json.loads((Path(c)/'report.json').read_text())
    if r1['status']!='COMPLETE' or r2['status']!='COMPLETE':raise ValueError('Incomplete study')
    a=read_lines(Path(p)/'metrics.jsonl');b=read_lines(Path(c)/'metrics.jsonl')
    key=lambda z:(z['task_id'],z['scenario_id'],int(z['seed']))
    A={key(z):z for z in a};B={key(z):z for z in b}
    if len(A)!=len(a) or len(B)!=len(b) or not A or A.keys()!=B.keys():raise ValueError('Unpaired cases')
    snaps=read_lines(Path(c)/'snapshots.jsonl')
    for s in snaps:
        if key(s) not in B or not s['rng_bit_generator_state'] or not s['current_routes'] or not s['best_routes']:
            raise ValueError('Incomplete snapshot or unknown case')
        if s['completed_updates_before_boundary']!=s['next_iteration_one_based']-1:
            raise ValueError('Operator outcomes not aligned with capture boundary')
        if s['stop_count']!=s['next_iteration_one_based']:
            raise ValueError('BothBudget iteration count mismatch')
    detail=[]
    for k in sorted(A):
        x=A[k];y=B[k]
        iterations_equal=x['iterations']==y['iterations']
        route_equal=x['best_route_sha256']==y['best_route_sha256']
        cost_equal=abs(x['road_cost_sec']-y['road_cost_sec'])<1e-8
        iter_limited=x['termination_reason']=='iteration_budget' and y['termination_reason']=='iteration_budget'
        detail.append({'task_id':k[0],'scenario_id':k[1],'seed':k[2],
            'plain_road_sec':x['road_cost_sec'],'capture_road_sec':y['road_cost_sec'],
            'road_delta_capture_minus_plain':y['road_cost_sec']-x['road_cost_sec'],
            'plain_iterations':x['iterations'],'capture_iterations':y['iterations'],
            'plain_wall_sec':x['search_wall_sec'],'capture_wall_sec':y['search_wall_sec'],
            'iterations_equal':iterations_equal,'route_equal':route_equal,
            'road_cost_equal':cost_equal,'both_iteration_limited':iter_limited,
            'strong_determinism_pass':bool(iter_limited and iterations_equal and route_equal and cost_equal),
            'capture_snapshot_count':y['captured']})
    strong=[r for r in detail if r['strong_determinism_pass']]
    if require_exact and len(strong)!=len(detail):
        raise AssertionError(f'Exact fixed-iteration audit FAILED: {len(strong)}/{len(detail)} passed. Inspect timing or modifications before proceeding')
    return {'stage':'R4A-AUDIT-v1','status':'COMPLETE','pairs':len(detail),
        'iteration_limited_both':sum(r['both_iteration_limited'] for r in detail),
        'strong_determinism_passes':len(strong),
        'terminal_route_equals':sum(r['route_equal'] for r in detail),
        'terminal_road_equals':sum(r['road_cost_equal'] for r in detail),
        'plain_minus_capture_road_mean_sec':mean(-r['road_delta_capture_minus_plain'] for r in detail),
        'iteration_difference_mean':mean(r['capture_iterations']-r['plain_iterations'] for r in detail),
        'capture_minus_plain_wall_mean_sec':mean(r['capture_wall_sec']-r['plain_wall_sec'] for r in detail),
        'total_snapshots':len(snaps),'require_exact':bool(require_exact),'details':detail,
        'sha256':{'plain':sha256(Path(p)/'metrics.jsonl'), 'capture':sha256(Path(c)/'metrics.jsonl'),
                  'snapshots':sha256(Path(c)/'snapshots.jsonl')},
        'limits':['Iteration-capped deterministic equality can establish noninterference for completed cap runs only.',
                  'Wall-capped runs may diverge from snapshot CPU overhead; do not call that an algorithmic improvement.',
                  'This is a read-only snapshot, NOT validated exact continuation.',
                  'Only previously inspected July VALIDATION tasks, not independent blind test.']}


def run(args):
    out=Path(args.output_dir)
    if out.exists():raise FileExistsError('New output directory required')
    report=summary_pair(args.plain,args.capture,require_exact=args.require_exact)
    out.mkdir(parents=True)
    write_json(out/'report.json',report)
    md=['# R4A 官方C3无干预捕获审计','',
        f"配对数：{report['pairs']}；严格确定性通过：{report['strong_determinism_passes']}；快照数：{report['total_snapshots']}",
        f"道路成本一致：{report['terminal_road_equals']}/{report['pairs']}；路线完全一致：{report['terminal_route_equals']}/{report['pairs']}",
        f"Capture-Plain平均搜索wall差：{report['capture_minus_plain_wall_mean_sec']:+.3f}秒；平均迭代差：{report['iteration_difference_mean']:+.2f}",
        '', '| Task | Scenario | Seed | Plain迭代 | Capture迭代 | 路线相同 | 道路成本差(Capture−Plain) |',
        '|---|---|---:|---:|---:|---|---:|']
    for r in report['details']:
        md.append(f"| {r['task_id']} | {r['scenario_id']} | {r['seed']} | {r['plain_iterations']} | {r['capture_iterations']} | {r['route_equal']} | {r['road_delta_capture_minus_plain']:+.3f} |")
    md+=['','**解释边界：** 固定迭代且双方跑满上限时，路线+成本完全一致才认为无干预审计通过。墙钟提前终止的配对不能给出精确无干预证明。快照本身还不是可恢复的ALNS checkpoint。']
    (out/'report.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    print('R4A AUDIT',report['strong_determinism_passes'],'/',report['pairs'],
          ' snapshots',report['total_snapshots'])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--plain',required=True);p.add_argument('--capture',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--require-exact',action='store_true')
    run(p.parse_args())
