#!/usr/bin/env python3
"""Read-only original 30s source-prefix K eligibility audit (no solver calls)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for mod in ('road_vrp7_step6c1_r4g_prefix','road_vrp7_step6c1_r4g','road_vrp7_step6c1_r4a','road_vrp7_step6c1_r4b'):
    if str(ROOT/mod) not in sys.path:sys.path.insert(0,str(ROOT/mod))
from prefix_core import parse_horizons,plan_cases,require,sha256
from r4g_core import load_capture,paired_audit


def run(a):
    capture=load_capture(a.capture_dir,'capture')
    plain=load_capture(a.plain_dir,'plain')
    score=paired_audit(plain,capture)
    require(score['route_road_equal']==score['pairs'],'Cannot use final-route mismatched captures')
    ks=parse_horizons(a.horizons)
    cases,skips=plan_cases(plain[1],capture[1],capture[2],ks,
                          max_cases=a.max_cases,max_snapshots=a.max_snapshots_per_case)
    by_k={str(k):{'eligible':sum(k in w['horizons'] for ww in cases.values() for w in ww),
                 'excluded':sum(z['horizon']==k for z in skips)} for k in ks}
    result={'stage':'R4G-TRAIN-PREFIX-PLAN-v1','data_role':'TRAIN_ONLY',
           'cases':len(cases),'source_30s_snapshots':score['n_snapshots'],
           'source_physical_states':score['n_distinct_physical_states'],
           'source_pairs':score['pairs'],'source_strict_pass':score['strict_pass'],
           'source_route_road_equal':score['route_road_equal'],
           'windows_to_verify':sum(sum(k in w['horizons'] for k in ks) for ws in cases.values() for w in ws),
           'by_horizon':by_k,'original_snapshot_sha256':sha256(Path(a.capture_dir)/'snapshots.jsonl'),
           'note':'ELIGIBILITY ONLY. Not a proof that any resumed trajectory matches.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--capture-dir',required=True)
    p.add_argument('--plain-dir',required=True)
    p.add_argument('--horizons',default='1,50,200,500')
    p.add_argument('--max-cases',type=int,default=0)
    p.add_argument('--max-snapshots-per-case',type=int,default=0)
    run(p.parse_args())
