#!/usr/bin/env python3
"""No-source-mutation R4B data health audit."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for dir in ('road_vrp7_step6c1_r4b','road_vrp7_step6c1_r4a'):
    sys.path.insert(0,str(ROOT/dir))
from r4b_core import common_data_audit
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture-dir',required=True);a=p.parse_args()
    mf,rows=common_data_audit(a.capture_dir)
    print(json.dumps({'status':'OK','snapshots':len(rows),'task_ids':mf['task_ids'],
        'scenario_ids':mf['scenario_ids'],'max_iterations':mf['max_iterations'],
        'alns_version':mf['alns_version'],'source_sha_verified':True},indent=2))
