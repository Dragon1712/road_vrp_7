#!/usr/bin/env python3
"""Build frozen U0/U2 cost denominators from PRE-EXISTING development results.

Pair proxy = mean destroy wall secs/call + mean repair wall secs/call;
this is additive, not a measured real end-to-end pair runtime. No test-data
statistics or ex-post road costs are used in calibration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import math
import statistics
NAMES_D = ('random','worst','related','section')
NAMES_R = ('greedy','regret2')


def build(source:Path, output:Path, *, variant='C3_topk', planner='oracle'):
    if not source.is_file():raise FileNotFoundError(source)
    dsum={x:[0.,0] for x in NAMES_D}
    rsum={x:[0.,0] for x in NAMES_R}
    n=0
    for line in source.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        record=json.loads(line)
        if record['solver'] != variant or record['planner'] != planner:continue
        # Only JULY DEVELOPMENT scenarios. Reject previously unrecognized months.
        if not record['scenario_id'].startswith('2025-07-'):
            raise ValueError('Calibration must use July development only')
        audit=record['operator_audit']
        for name in NAMES_D:
            x=audit['destroy'].get(name)
            if x is not None:
                dsum[name][0]+=float(x['seconds']);dsum[name][1]+=int(x['calls'])
        for name in NAMES_R:
            x=audit['repair'].get(name)
            if x is not None:
                rsum[name][0]+=float(x['seconds']);rsum[name][1]+=int(x['calls'])
        n+=1
    if n<1:raise ValueError('No development calibration records found')
    if any(k[1]<1 or k[0]<0 or not math.isfinite(k[0]) for k in [*dsum.values(),*rsum.values()]):
        raise ValueError('Missing or invalid operator duration counters')
    dc={k:v[0]/v[1] for k,v in dsum.items()}
    rc={k:v[0]/v[1] for k,v in rsum.items()}
    pair={f'{d}|{r}':dc[d]+rc[r] for d in NAMES_D for r in NAMES_R}
    if any(z <= 0 or not math.isfinite(z) for z in pair.values()):
        raise ValueError('Unusable estimated pair costs')
    result={'schema':'section_pair_development_cost_v1','source_path':str(source),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_solver':variant,'source_planner':planner,
            'source_role':'July development (NOT a blind test)',
            'records_used':n,'pair_seconds':pair,
            'global_seconds':statistics.median(pair.values()),
            'method':'mean destroy sec/call + mean repair sec/call, frozen additive approximation',
            'warning':'Excludes selector and acceptance overhead; do not use final test measurements'}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        if json.loads(output.read_text(encoding='utf-8')) != result:
            raise FileExistsError('Existing profile differs; use another output name')
    else:output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='runs/chicago_july_v1/step5b_v2/validation/solutions.jsonl')
    parser.add_argument('--output',default='road_vrp7_step5b_v3/calibration/section_pair_costs_dev.json')
    parser.add_argument('--variant',default='C3_topk');parser.add_argument('--planner',default='oracle')
    args=parser.parse_args()
    print(json.dumps(build(Path(args.source),Path(args.output),variant=args.variant,planner=args.planner),indent=2))
