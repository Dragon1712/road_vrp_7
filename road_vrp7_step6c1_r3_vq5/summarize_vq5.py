#!/usr/bin/env python3
"""Summarize VQ5 FIXED heldout panel across training sizes and 6 model seeds.

Repeated seeds are algorithmic stability checks, never independent observations.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r3_vq5'))
from vq5_core import STAGE,REPORT_STAGE,WATCH,METRICS,read_json,write_json,sha256

COMPS=(('VQ_MEAN','E0_MATCHED'),('RESID_OOF50','E0_MATCHED'),
       ('RESID_OOF50','E0_FULL'),('PAIR_RANK','E0_MATCHED'),
       ('E0_RISK35','E0_SCORE_COVERAGE35'))

def run(studies,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Use NEW analysis dir')
    reports={};pins={};source_paths={}
    for p in studies:
        p=Path(p);r=read_json(p/'report.json');v=read_json(p/'provenance.json')
        if r.get('stage')!=STAGE or r.get('status')!='COMPLETE':raise ValueError('Bad VQ5 model report')
        seed=int(r['seed'])
        if seed in reports:raise ValueError('Repeated training seed')
        if r.get('anchor_48_verification',{}).get('status')!='MATCH':
            raise ValueError('Require frozen original VQ4 n48 reproduction for every seed')
        reports[seed]=r;pins[seed]=v;source_paths[seed]=p
    if len(reports)<2:raise ValueError('Require multiple model training seeds')
    seeds=sorted(reports);first=reports[seeds[0]]
    conf=('sizes','panel_n','horizons','policies','trees','n_original_train_tasks')
    if any(any(reports[s][k]!=first[k] for k in conf) for s in seeds):
        raise ValueError('Different fixed panel / training config')
    pk=('collection_manifest_sha256','collection_rollouts_sha256','dataset_paired_sha256',
        'dataset_groups_sha256','frozen_vq3_code_sha256','code_sha256')
    if any(any(pins[s][k]!=pins[seeds[0]][k] for k in pk) for s in seeds):
        raise ValueError('Different labels / frozen code / panel between seeds')
    sizes=first['sizes'];H=list(map(str,first['horizons']))
    metrics={};anchor_metrics={};paired=[];means={};anchor_means={}
    for n in sizes:
        ns=str(n);metrics[ns]={};means[ns]={};anchor_metrics[ns]={};anchor_means[ns]={}
        for h in H:
            metrics[ns][h]={};means[ns][h]={};anchor_metrics[ns][h]={};anchor_means[ns][h]={}
            for method in WATCH:
                raw={m:[float(reports[s]['macro_task_metrics'][ns][h][method][m]) for s in seeds]
                     for m in METRICS}
                metrics[ns][h][method]={k:{'seed_mean':float(np.mean(v)),
                     'seed_min':float(np.min(v)),'seed_max':float(np.max(v))} for k,v in raw.items()}
                means[ns][h][method]=metrics[ns][h][method]['mean_gain_sec']['seed_mean']
                anchor_metrics[ns][h][method]={}
                for k in METRICS:
                    vals=[float(reports[s]['anchor_panel_48_macro_task_metrics'][ns][h][method][k])
                          for s in seeds]
                    anchor_metrics[ns][h][method][k]={'seed_mean':float(np.mean(vals)),
                        'seed_min':float(np.min(vals)),'seed_max':float(np.max(vals))}
                anchor_means[ns][h][method]=anchor_metrics[ns][h][method]['mean_gain_sec']['seed_mean']
            for left,right in COMPS:
                per_task={}
                for task in sorted(first['folds'][ns][h]):
                    d=[]
                    for s in seeds:
                        fold=reports[s]['folds'][ns][h][task]['metrics']
                        d.append(float(fold[left]['mean_gain_sec'])-float(fold[right]['mean_gain_sec']))
                    per_task[task]=float(np.mean(d))
                by_seed={str(s):float(reports[s]['macro_task_metrics'][ns][h][left]['mean_gain_sec']-
                    reports[s]['macro_task_metrics'][ns][h][right]['mean_gain_sec']) for s in seeds}
                paired.append({'n_training_selected_states_total':n,'n_fixed_eval_states':first['panel_n'],
                    'horizon':int(h),'left':left,'right':right,
                    'mean_task_paired_delta_sec':float(np.mean(list(per_task.values()))),
                    'positive_tasks':sum(x>0 for x in per_task.values()),
                    'per_task_delta_sec':per_task,'seed_delta_sec':by_seed,
                    'positive_training_seeds':sum(x>0 for x in by_seed.values())})
    out.mkdir(parents=True)
    write_json(out/'report.json',{'stage':REPORT_STAGE,'status':'COMPLETE','sizes':sizes,'seeds':seeds,
        'panel_n':first['panel_n'],'horizons':first['horizons'],
        'metrics':metrics,'anchor_panel_48_metrics':anchor_metrics,'paired_differences':paired,
        'warning':['Fixed heldout panel across n, but still only four original TRAIN tasks.',
            'Six model seeds share exactly the same labels; no independent p-values.',
            'Data are ex-post C3 restart proxy, not full 30s ALNS.']})
    write_json(out/'provenance.json',{'source_report_sha256':{
      str(s):sha256(source_paths[s]/'report.json') for s in seeds},
      'code_sha256':sha256(Path(__file__))})
    md=['# R3-VQ5 固定192状态评价面板学习曲线','','所有样本均来自原Train四任务。每折评价面板固定（每任务48状态），仅其他三任务的训练前缀增加。','']
    for h in H:
        md+=['## H='+h,'','| 方法 | '+' | '.join('train n='+str(n) for n in sizes)+' |',
             '|---|'+'---:|'*len(sizes)]
        for name in WATCH:
            md+=['| '+name+' | '+' | '.join(f'{means[str(n)][h][name]:+.3f}' for n in sizes)+' |']
        md+=['','### 同一原始48状态锚点评价面板（仅训练量改变）','','| 方法 | '+' | '.join('train n='+str(n) for n in sizes)+' |',
             '|---|'+'---:|'*len(sizes)]
        for name in WATCH:
            md+=['| '+name+' | '+' | '.join(f'{anchor_means[str(n)][h][name]:+.3f}' for n in sizes)+' |']
        md+=['','### 同一192状态评价面板上的配对差','','| n | 比较 | 差值（秒/状态） | 正向任务 | 正向种子 |',
             '|---:|---|---:|---:|---:|']
        for z in paired:
            if z['horizon']==int(h):
                md+=[f"| {z['n_training_selected_states_total']} | {z['left']} - {z['right']} | {z['mean_task_paired_delta_sec']:+.3f} | {z['positive_tasks']}/4 | {z['positive_training_seeds']}/{len(seeds)} |"]
        md+=['']
    md+=['**注意：** 这是训练规模变化的内部重评估，而非新任务盲测，也不能从代理收益推断30秒在线改善。','']
    (out/'report.md').write_text('\n'.join(md),encoding='utf8')
    print('VQ5 FIXED-PANEL SUMMARY COMPLETE:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--study-dirs',required=True,help='Comma separated VQ5 per-seed studies')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args();ds=[Path(x.strip()) for x in a.study_dirs.split(',') if x.strip()]
    run(ds,a.output_dir)
