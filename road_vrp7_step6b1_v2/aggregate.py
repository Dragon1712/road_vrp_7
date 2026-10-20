#!/usr/bin/env python3
"""Strict within-state descriptive contrasts across completed studies/seeds.

Never converts multiple training seeds into independent statistical observations.
"""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def read_states(p):
    with Path(p).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty predictions '+str(p))
    ids=[r['state_id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate state_id in '+str(p))
    return {r['state_id']:r for r in rows}


def analyze(paths,base,treatment):
    output=[];reference_source=None;reference_code=None;seen_seeds=set()
    for path in paths:
        path=Path(path)
        manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        summ=json.loads((path/'summary.json').read_text(encoding='utf-8'))
        if summ.get('status')!='COMPLETE' or manifest.get('stage')!='step6b1-v2':
            raise ValueError('Incomplete or unexpected experiment: '+str(path))
        if base not in manifest['methods'] or treatment not in manifest['methods']:
            raise ValueError('Requested method not included in study')
        s=manifest['source_sha256'];c=manifest['code_sha256']
        if reference_source is not None and s!=reference_source:raise ValueError('Mixed input datasets: '+str(path))
        if reference_code is not None and c!=reference_code:raise ValueError('Mixed implementation revisions: '+str(path))
        reference_source,reference_code=s,c
        seed=manifest['seed']
        if seed in seen_seeds:raise ValueError('Repeated training seed '+str(seed))
        seen_seeds.add(seed)
        for protocol in manifest['protocols']:
            a=read_states(path/protocol/f'{base}_test_states.csv')
            b=read_states(path/protocol/f'{treatment}_test_states.csv')
            if a.keys()!=b.keys():raise ValueError('Paired methods have different test states')
            diffs=[];per_task=defaultdict(list);samples=[]
            for state in sorted(a):
                x,y=a[state],b[state]
                for field in ('task_id','date','scenario_id','oracle_static_repair_noop_bound_sec'):
                    if x[field]!=y[field]:raise ValueError('Unpaired '+field+' for '+state)
                gain=float(y['realized_signed_gain_sec'])-float(x['realized_signed_gain_sec'])
                diffs.append(gain);per_task[x['task_id']].append(gain)
                samples.append({'state_id':state,'task_id':x['task_id'],'date':x['date'],
                                'base_gain_sec':float(x['realized_signed_gain_sec']),
                                'treatment_gain_sec':float(y['realized_signed_gain_sec']),
                                'difference_sec':gain})
            d=np.array(diffs,dtype=float)
            output.append({'path':str(path),'seed':seed,'protocol':protocol,'states':len(d),
                  'base':base,'treatment':treatment,
                  'mean_paired_gain_delta_sec':float(np.mean(d)),
                  'positive_delta_states':int(np.sum(d>1e-7)),
                  'negative_delta_states':int(np.sum(d< -1e-7)),
                  'tie_states':int(np.sum(np.abs(d)<=1e-7)),
                  'mean_delta_by_heldout_task':{t:float(np.mean(v)) for t,v in per_task.items()},
                  'no_independence_claim':'Same held-out states across seeds, no naive t-test or CI.',
                  'rows':samples})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--studies',nargs='+',type=Path,required=True)
    p.add_argument('--base',default='P1_class_static_time')
    p.add_argument('--treatment',default='P8_class_history')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=analyze(a.studies,a.base,a.treatment)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in result],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
