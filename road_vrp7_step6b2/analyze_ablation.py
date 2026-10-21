#!/usr/bin/env python3
"""Matched F0-F5 ablation using signed gains; NO blind test claims."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
PAIRS=(('F2','F0'),('F3','F2'),('F4','F3'),('F1','F0'),('F3','F1'),('F5','F4'))

def get_rows(path):
    with open(path,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    d={x['state_id']:x for x in rows}
    if len(d)!=len(rows) or not rows:raise ValueError('Duplicate or empty states '+str(path))
    return d

def compare(root,output):
    root=Path(root);out=Path(output)
    if out.exists():raise FileExistsError(out)
    manifest=json.loads((root/'manifest.json').read_text())
    summ=json.loads((root/'summary.json').read_text())
    if summ['status']!='COMPLETE' or manifest['stage']!='step6b2-ablation-v1':
        raise ValueError('Not a frozen complete B2 experiment')
    report=[]
    for protocol in manifest['protocols']:
        for method in manifest['methods']:
            for treatment,base in PAIRS:
                a=root/protocol/f'{method}_{base}_test_states.csv'
                b=root/protocol/f'{method}_{treatment}_test_states.csv'
                if not a.is_file() or not b.is_file():continue
                ra,rb=get_rows(a),get_rows(b)
                if ra.keys()!=rb.keys():raise ValueError('Unpaired states')
                dif=[];changes=[];bytask=defaultdict(list)
                for state in sorted(ra):
                    x,y=ra[state],rb[state]
                    for k in ('task_id','date','scenario_id','oracle_static_repair_noop_bound_sec'):
                        if x[k]!=y[k]:raise ValueError('Unpaired metadata '+k)
                    dx=float(y['realized_signed_gain_sec'])-float(x['realized_signed_gain_sec'])
                    dif.append(dx);bytask[x['task_id']].append(dx)
                    key_x=x['selected_candidate_id'] if int(x['execute']) else 'NOOP'
                    key_y=y['selected_candidate_id'] if int(y['execute']) else 'NOOP'
                    changes.append(key_x!=key_y)
                diff=np.asarray(dif)
                report.append({'protocol':protocol,'method':method,'treatment':treatment,
                   'base':base,'states':len(diff),
                   'mean_signed_gain_delta_sec_per_state':float(diff.mean()),
                   'median_delta_sec':float(np.median(diff)),
                   'different_decisions':int(sum(changes)),
                   'decision_change_rate':float(np.mean(changes)),
                   'positive_delta_states':int(np.sum(diff>1e-7)),
                   'negative_delta_states':int(np.sum(diff< -1e-7)),
                   'ties':int(np.sum(np.abs(diff)<=1e-7)),
                   'task_mean_deltas_sec':{k:float(np.mean(v)) for k,v in bytask.items()}})
    if not report:raise ValueError('No paired comparisons')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'source_study':str(root),'comparisons':report,
       'warning':'Development data only; same states reused. Test outcomes must not guide model changes then be treated as blind.'},
       indent=2,ensure_ascii=False)+'\n')
    for r in report:print(r['protocol'],r['method'],r['treatment']+'-'+r['base'],
                         round(r['mean_signed_gain_delta_sec_per_state'],3),
                         'changed',r['different_decisions'],'/',r['states'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();compare(a.study,a.output)
if __name__=='__main__':main()
