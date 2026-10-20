#!/usr/bin/env python3
"""Paired descriptive comparison across completed Step6-B1 studies.

Never treats multiple seeds of the same held-out states as independent examples.
"""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def load_states(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        arr=list(csv.DictReader(f))
    if not arr:raise ValueError('Empty state file '+str(path))
    return {x['state_id']:x for x in arr}

def analyze(studies,base,new):
    result=[]
    for study in studies:
        study=Path(study)
        meta=json.loads((study/'manifest.json').read_text())
        if not (study/'summary.json').exists():raise ValueError('Incomplete study: '+str(study))
        for protocol in meta['protocols']:
            d=study/protocol
            b=load_states(d/f'{base}_test_states.csv')
            n=load_states(d/f'{new}_test_states.csv')
            if b.keys()!=n.keys():raise ValueError('Unpaired comparison')
            ids=sorted(b)
            diff=np.array([float(n[s]['static_repair_td_gain_sec'])-float(b[s]['static_repair_td_gain_sec']) for s in ids])
            by_task=defaultdict(list)
            for s,z in zip(ids,diff):by_task[b[s]['task_id']].append(float(z))
            result.append({'study':str(study),'seed':meta['seed'],'protocol':protocol,
                'base':base,'treatment':new,'states':len(ids),
                'delta_mean_gain_sec_per_state':float(np.mean(diff)),
                'improved_states':int(np.sum(diff>1e-7)),
                'worse_states':int(np.sum(diff< -1e-7)),
                'ties':int(np.sum(np.abs(diff)<=1e-7)),
                'heldout_task_means':{k:float(np.mean(v)) for k,v in by_task.items()},
                'note':'Descriptive paired comparison only. Repeated seeds reuse the same held-out states; no confidence interval.'})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--studies',nargs='+',required=True)
    p.add_argument('--base',default='P1_class_static_time')
    p.add_argument('--treatment',default='P8_class_history')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    res=analyze(a.studies,a.base,a.treatment)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(res,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
