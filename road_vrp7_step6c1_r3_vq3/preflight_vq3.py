#!/usr/bin/env python3
"""Read-only integrity and Train-only label preflight for VQ3."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
for f in ('road_vrp7_step6c1_r3_vq3','road_vrp7_step6c1_r3_vq','road_vrp7_step6c1_r2'):
    p=str(ROOT/f)
    if p not in sys.path:sys.path.insert(0,p)
import train_vq
from r2_policy import group_split

def main(a):
    c,rep,X,B,Y,BY,valid,same,groups,lookup,selected,measured=train_vq.load(a)
    masks,roles=group_split(groups,'task')
    task_counts=Counter(groups[lookup[di]]['task_id'] for di in selected)
    if len(task_counts)!=4 or any(roles[t]!='train' for t in task_counts):
        raise ValueError('Not four ORIGINAL TRAIN tasks')
    expected=sum(int((valid[lookup[di]]&(~same[lookup[di]])).sum())+1 for di in selected)
    for h in c['horizons']:
        if sum(1 for k in measured if k[1]==h)!=expected:raise ValueError('H-specific candidate count mismatch')
    print('R3-VQ3 preflight OK. n_states=',len(selected),'task_counts=',dict(sorted(task_counts.items())),
          'horizons=',c['horizons'],'measured_labels=',len(measured),'replicates=',c['replicates'])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-dir',required=True);p.add_argument('--collection-dir',required=True)
    main(p.parse_args())
