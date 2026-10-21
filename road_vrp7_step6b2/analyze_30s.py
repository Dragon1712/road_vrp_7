#!/usr/bin/env python3
"""Paired real-road outcomes; never treat re-used task/date as independent seeds."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np

def read(p):
    m=json.loads((p/'manifest.json').read_text())
    s=json.loads((p/'summary.json').read_text())
    if s['status']!='COMPLETE':raise ValueError('Incomplete run '+str(p))
    with (p/'metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    by={}
    for x in rows:
        k=(x['task_id'],x['scenario_id'],int(x['seed']))
        if k in by:raise ValueError('Duplicate matched key '+str(k))
        by[k]=x
    if len(rows)!=s['expected_records']:raise ValueError('Run record count mismatch')
    return m,by

def compare(base,treatment,out):
    a,br=read(Path(base));b,tr=read(Path(treatment))
    fields=('source_sha256','road_sha256','edge_sha256','solver_sha256',
            'section_sha256','wall_sec','max_iterations','anneal_iterations',
            'top_k','seeds','task_ids','scenario_ids','split')
    for key in fields:
        if a[key]!=b[key]:raise ValueError('Incompatible runs: '+key)
    if br.keys()!=tr.keys():raise ValueError('Unmatched task/scenario/seed')
    vals=[];rows=[];by_task=defaultdict(list);by_scenario=defaultdict(list)
    for key in sorted(br):
        x,y=br[key],tr[key]
        delta=float(x['ex_post_road_travel_sec'])-float(y['ex_post_road_travel_sec'])
        # positive delta: treatment improves road travel cost
        vals.append(delta);by_task[key[0]].append(delta);by_scenario[key[1]].append(delta)
        rows.append({'task_id':key[0],'scenario_id':key[1],'seed':key[2],
            'baseline_road_travel_sec':float(x['ex_post_road_travel_sec']),
            'treatment_road_travel_sec':float(y['ex_post_road_travel_sec']),
            'improvement_sec':delta,
            'baseline_iterations':int(x['iterations']),'treatment_iterations':int(y['iterations']),
            'treatment_feature_ms':1000.*float(y['section_feature_wall_sec']),
            'treatment_inference_ms':1000.*float(y['section_inference_wall_sec'])})
    v=np.asarray(vals)
    report={'stage':'step6b2-30s-paired-v1','base':str(base),'treatment':str(treatment),
        'records':len(v),'mean_road_travel_improvement_sec':float(v.mean()),
        'median_road_travel_improvement_sec':float(np.median(v)),
        'wins':int(np.sum(v>1e-7)),'losses':int(np.sum(v< -1e-7)),
        'ties':int(np.sum(abs(v)<=1e-7)),
        'task_mean_improvement_sec':{k:float(np.mean(z)) for k,z in by_task.items()},
        'scenario_mean_improvement_sec':{k:float(np.mean(z)) for k,z in by_scenario.items()},
        'note':'Paired July development study. Seeds on repeated tasks/scenarios are NOT independent test tasks. Do not claim blind generalization.',
        'rows':rows}
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2,ensure_ascii=False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--treatment',required=True);p.add_argument('--output',required=True)
    x=p.parse_args();compare(x.baseline,x.treatment,x.output)
if __name__=='__main__':main()
