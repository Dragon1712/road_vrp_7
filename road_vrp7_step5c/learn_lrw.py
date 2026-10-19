#!/usr/bin/env python3
"""Learn simple, smoothed offline pair probabilities from prior July developer ALNS logs.

This is GRLOS-LRW-INSPIRED rather than original-paper GRLOS replication.
Do not evaluate on the same training trajectories and claim independent generalization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from selectors_step5c import PAIRS


def train(src):
    p=Path(src)
    if not p.is_file():raise FileNotFoundError(p)
    stats=defaultdict(lambda:{'calls':0.,'improving':0.,'new_best':0.})
    n=0; tasks=set();scenarios=set()
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        r=json.loads(line)
        if r.get('solver')!='C3_topk' or r.get('planner')!='oracle':continue
        tasks.add(r['task_id']);scenarios.add(r['scenario_id']);n+=1
        pairs=r.get('operator_audit',{}).get('pairs',{})
        if not pairs:raise ValueError('Missing operator_audit.pairs')
        for k,v in pairs.items():
            if k not in PAIRS:continue
            s=stats[k]
            for field in s:s[field]+=float(v.get(field,0))
    if n<1:raise ValueError('No C3_topk Oracle records for offline LRW training')
    # Beta-smoothed probability of new-best and improving candidates.
    # Prior keeps rarely sampled arms alive. Deliberately simple baseline.
    raw={}
    for k in PAIRS:
        s=stats[k];den=s['calls']+20.
        raw[k]=(s['new_best']+1.)/den + .25*(s['improving']+2.)/den
    total=sum(raw.values())
    probs={k:0.85*raw[k]/total+0.15/len(PAIRS) for k in PAIRS}
    return {'schema':'lrw_pair_weights_v1','method':'smoothed new-best rate + 0.25 improving rate; 15% uniform exploration floor',
            'source_path':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'source_role':'previous July development logs, NOT blind test',
            'trained_records':n,'training_tasks':sorted(tasks),'training_scenarios':sorted(scenarios),
            'weights':probs,'counts':dict(stats),
            'warning':'Not GRLOS authors original LRW. Same-task evaluation is development not held-out task generalization.'}


def main():
    a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--output',required=True)
    args=a.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(f'Frozen LRW exists, use a new output: {out}')
    obj=train(args.source);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('PASS trained',obj['trained_records'],'records; saved',out)
if __name__=='__main__':main()
