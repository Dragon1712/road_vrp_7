#!/usr/bin/env python3
"""Summarize model training seeds; each seed shares the SAME four heldout tasks."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vq3_core import STAGE,POLICIES
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'road_vrp7_step6c1_r3_vq2'))
from vq2_core import write_json,sha256

def aggregate(paths,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Aggregation directory already exists')
    reports=[];provs=[]
    for path in paths:
        p=Path(path);z=json.loads((p/'report.json').read_text());v=json.loads((p/'provenance.json').read_text())
        if z['status']!='COMPLETE' or z['stage']!=STAGE or tuple(z['policies'])!=POLICIES:raise ValueError('Invalid VQ3 study: '+str(p))
        reports.append(z);provs.append(v)
    if len(set(x['seed'] for x in reports))!=len(reports):raise ValueError('Duplicate seed')
    for k in ['collection_manifest_sha256','collection_rollouts_sha256','dataset_paired_sha256','dataset_groups_sha256','code_sha256']:
        if any(p[k]!=provs[0][k] for p in provs):raise ValueError('Changed experiment data/code: '+k)
    for k in ['horizons','trees','n_selected_states']:
        if any(p[k]!=reports[0][k] for p in reports):raise ValueError('Inconsistent setting: '+k)
    keys=['mean_gain_sec','mean_regret_sec','override_rate','top1_hit','harm_gt40_per_state','harm_gt40_per_override']
    result={}
    for h in map(str,reports[0]['horizons']):
        result[h]={}
        for name in POLICIES:
            result[h][name]={}
            for key in keys:
                vals=[p['macro_task_metrics'][h][name][key] for p in reports]
                result[h][name][key]={'mean':float(np.mean(vals)),
                    'min':float(np.min(vals)),'max':float(np.max(vals)),
                    'by_seed':{str(r['seed']):float(v) for r,v in zip(reports,vals)}}
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'report.json',{'stage':'R3-VQ3-MULTISEED-v1','status':'COMPLETE','seeds':sorted(p['seed'] for p in reports),
       'n_states':reports[0]['n_selected_states'],'horizons':reports[0]['horizons'],
       'warning':'Same four original TRAIN tasks and 48 labeled states across six model seeds; not independent test data.',
       'results':result})
    write_json(out/'provenance.json',{'reports':{str(p):sha256(Path(p)/'report.json') for p in paths},
         'source_provenance':provs[0], 'aggregate_code_sha256':sha256(Path(__file__))})
    t=['# R3-VQ3 六种子汇总','', '只有4个原Train任务和48个状态；6种子不是独立测试。','']
    for h,by in result.items():
        t+=['## H='+h,'','| 方法 | 收益均值 ↑ | 收益种子范围 | Regret ↓ | 覆盖率 | 严重损失/状态 |','|---|---:|---|---:|---:|---:|']
        for name,r in by.items():
            gain=r['mean_gain_sec'];t.append(f"| {name} | {gain['mean']:+.3f} | [{gain['min']:+.2f}, {gain['max']:+.2f}] | {r['mean_regret_sec']['mean']:.3f} | {r['override_rate']['mean']:.1%} | {r['harm_gt40_per_state']['mean']:.1%} |")
        t+=['']
    (out/'report.md').write_text('\n'.join(t),encoding='utf8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--study-dirs',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();paths=[Path(x) for x in a.study_dirs.split(',') if x.strip()]
    if len(paths)<2:raise ValueError('Need at least two seeds')
    aggregate(paths,a.output_dir)
