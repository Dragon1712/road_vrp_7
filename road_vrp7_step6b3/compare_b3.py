#!/usr/bin/env python3
"""Paired offline Section policy comparison for one completed B3 training study.

Counts decision changes separately from signed per-state reward differences.
This is DESCRIPTIVE and should not be called an independent statistical test.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np


def rows(path):
    with Path(path).open(newline='',encoding='utf-8') as f: v=list(csv.DictReader(f))
    if not v:raise ValueError('Empty states CSV: '+str(path))
    out={}
    for r in v:
        sid=r['state_id']
        if sid in out:raise ValueError('Duplicate state '+sid)
        out[sid]=r
    return out


def compare(study,base,treat,output):
    root=Path(study);summary=json.loads((root/'summary.json').read_text())
    if summary.get('status')!='COMPLETE':raise ValueError('Incomplete study')
    if not (root/'manifest.json').is_file():raise FileNotFoundError(root/'manifest.json')
    results={}
    for p in summary['protocols']:
        A=rows(root/p/f'{base}_test_states.csv')
        B=rows(root/p/f'{treat}_test_states.csv')
        if A.keys()!=B.keys():raise ValueError('Unmatched test states')
        delta=[];changed=[];action_changes=0;wins=0;losses=0;ties=0
        for sid in sorted(A):
            a,b=A[sid],B[sid]
            if (a['task_id'],a['date'],a.get('scenario_id'))!=(b['task_id'],b['date'],b.get('scenario_id')):
                raise ValueError('State context differs')
            gain=float(b['realized_signed_gain_sec'])-float(a['realized_signed_gain_sec'])
            diff=(a['selected_candidate_id']!=b['selected_candidate_id'] or a['execute']!=b['execute'])
            if diff:action_changes+=1
            delta.append(gain);changed.append(diff)
            wins+=gain>1e-7;losses+=gain<-1e-7;ties+=abs(gain)<=1e-7
        d=np.asarray(delta)
        results[p]={'states':len(d),'mean_improvement_sec_per_state':float(d.mean()),
                    'median_improvement_sec_per_state':float(np.median(d)),
                    'wins':int(wins),'losses':int(losses),'ties':int(ties),
                    'decision_changed_states':action_changes,
                    'decision_changed_rate':action_changes/len(d),
                    'mean_gain_when_decision_changes':float(d[np.array(changed)].mean()) if action_changes else None,
                    'note':'Paired July development offline; does not establish final 30-second road gain.'}
    if Path(output).exists():raise FileExistsError(output)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps({'baseline':base,'treatment':treat,'study':str(root),
               'protocols':results},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--study',required=True)
    p.add_argument('--base',default='M0');p.add_argument('--treatment',default='M3')
    p.add_argument('--output',required=True)
    a=p.parse_args();compare(a.study,a.base,a.treatment,a.output)
