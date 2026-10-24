#!/usr/bin/env python3
"""Aggregate repeated model-training seeds; seeds reuse EXACTLY the same heldout tasks/states."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from vq2_core import STAGE,METHODS,write_json,sha256

def aggregate(dirs,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Use NEW output directory')
    records=[];provs=[]
    for p in dirs:
        p=Path(p);z=json.loads((p/'report.json').read_text());v=json.loads((p/'provenance.json').read_text())
        if z['stage']!=STAGE or z['status']!='COMPLETE':raise ValueError('Incomplete study '+str(p))
        records.append(z);provs.append(v)
    if len({r['seed'] for r in records})!=len(records):raise ValueError('Repeated training seed')
    pins=['collection_manifest_sha256','collection_rollouts_sha256','dataset_paired_sha256','dataset_groups_sha256']
    for k in pins:
        if len({v[k] for v in provs})!=1:raise ValueError('Mismatched data: '+k)
    for k in ('policies','n_selected_states','horizons','trees'):
        if any(r[k]!=records[0][k] for r in records[1:]):raise ValueError('Mismatched config: '+k)
    if any(v['code_sha256']!=provs[0]['code_sha256'] for v in provs[1:]):raise ValueError('Different code versions')
    result={}
    keys=('mean_gain_sec','mean_regret_sec','top1_hit','override_rate','harm_gt40_per_state','harm_gt40_per_override')
    for h in map(str,records[0]['horizons']):
        result[h]={}
        for pol in METHODS:
            result[h][pol]={}
            for metric in keys:
                vals=[r['macro_task_metrics'][h][pol][metric] for r in records]
                result[h][pol][metric]={'mean_over_training_seeds':float(np.mean(vals)),
                    'min_over_training_seeds':float(min(vals)),'max_over_training_seeds':float(max(vals)),
                    'per_seed':{str(z['seed']):float(v) for z,v in zip(records,vals)}}
    out.mkdir(parents=True)
    write_json(out/'report.json',{'stage':'R3-VQ2-multiseed-summary-v1','status':'COMPLETE',
        'seeds':sorted(r['seed'] for r in records),'horizons':records[0]['horizons'],
        'n_original_train_tasks':records[0]['n_original_train_tasks'],
        'n_selected_states':records[0]['n_selected_states'],
        'note':'Same heldout tasks and labels repeated across training seeds: not independent samples.',
        'results':result})
    write_json(out/'provenance.json',{'source_reports':{str(p):sha256(Path(p)/'report.json') for p in dirs},
         'code_sha256':sha256(Path(__file__))})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--study-dirs',required=True,help='Comma-delimited, >=2 independent study directories')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args();dirs=[Path(s.strip()) for s in a.study_dirs.split(',') if s.strip()]
    if len(dirs)<2:raise ValueError('At least two seed studies required')
    aggregate(dirs,a.output_dir)
