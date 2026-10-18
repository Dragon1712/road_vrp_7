#!/usr/bin/env python3
"""Compare genuinely repeated same-configuration experiment directories."""
import argparse
import csv
import json
from pathlib import Path
import statistics

KEY=('task_id','scenario_id','solver','planner','seed')

def metric_map(d):
    with (Path(d)/'metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    keys=[tuple(x[k] for k in KEY) for x in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate metric keys')
    return dict(zip(keys,rows))

def compare(a,b):
    x=Path(a);y=Path(b)
    ca=json.loads((x/'run_config.json').read_text())
    cb=json.loads((y/'run_config.json').read_text())
    if ca!=cb:raise ValueError('Repeat experiment configurations do not match')
    l=metric_map(x);r=metric_map(y)
    if set(l)!=set(r):raise ValueError('Repeat keys differ')
    out={}
    for solver in ca['variants']:
        vals=[];dif_iters=[]
        for k in l:
            if k[2]!=solver:continue
            vals.append(float(l[k]['ex_post_road_travel_sec'])-float(r[k]['ex_post_road_travel_sec']))
            dif_iters.append(int(l[k]['iterations'])-int(r[k]['iterations']))
        if vals:out[solver]={'n':len(vals),'mean_signed_road_delta_sec':statistics.mean(vals),
                             'mean_absolute_road_delta_sec':statistics.mean(map(abs,vals)),
                             'changed_road_records':sum(abs(a)>1.e-7 for a in vals),
                             'changed_iterations':sum(a!=0 for a in dif_iters)}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');a=p.parse_args()
    print(json.dumps(compare(a.first,a.second),indent=2,ensure_ascii=False))
