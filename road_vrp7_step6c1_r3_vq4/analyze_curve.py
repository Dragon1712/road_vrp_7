#!/usr/bin/env python3
"""Aggregate VQ3 frozen-policy learning curves without pretending seeds are independent."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from vq4_core import SIZES,read_json,sha256,write_json

WATCH=('C3','E0_FULL','E0_MATCHED','VQ_MEAN','RESID_OOF25','RESID_OOF50',
       'PAIR_RANK','PAIR_REGRET','E0_RISK35','E0_SCORE_COVERAGE35')
MEASURES=('mean_gain_sec','mean_regret_sec','override_rate','harm_gt40_per_state','harm_gt40_per_override','top1_hit')


def check_inputs(model_root,sizes,seeds):
    records={};provenances={}
    for size in sizes:
        for seed in seeds:
            d=model_root/f'vq4_n{size}_seed{seed}_v1'
            a=read_json(d/'report.json');p=read_json(d/'provenance.json')
            if a['stage']!='R3-VQ3-DECISION-RISK-LOTO-v1' or a['status']!='COMPLETE':
                raise ValueError('Not a complete frozen VQ3 report: '+str(d))
            if int(a['seed'])!=seed or int(a['n_selected_states'])!=size or set(map(int,a['horizons']))!={20,50}:
                raise ValueError('Study size/seed/horizon identity mismatch '+str(d))
            if not set(WATCH).issubset(a['policies']):raise ValueError('Missing frozen VQ3 controls')
            if len(a['folds']['20'])!=4 or len(a['folds']['50'])!=4:raise ValueError('Only four heldout training tasks allowed')
            records[size,seed]=a;provenances[size,seed]=p
    # Pin original one-step labels and identical original source code across all sizes.
    if len({p['dataset_paired_sha256'] for p in provenances.values()})!=1:
        raise ValueError('Mixed original paired data')
    if len({p['dataset_groups_sha256'] for p in provenances.values()})!=1:
        raise ValueError('Mixed original groups data')
    if len({json.dumps(p['code_sha256'],sort_keys=True) for p in provenances.values()})!=1:
        raise ValueError('Mixed VQ3 code across sizes / seeds')
    for size in sizes:
        if len({provenances[size,seed]['collection_rollouts_sha256'] for seed in seeds})!=1:
            raise ValueError('Mixed collection labels at n='+str(size))
        if len({provenances[size,seed]['collection_manifest_sha256'] for seed in seeds})!=1:
            raise ValueError('Mixed collection manifests at n='+str(size))
    return records,provenances


def compare(fold,first,second,metric):
    return float(fold['metrics'][first][metric])-float(fold['metrics'][second][metric])


def analyze(model_root,out,sizes=SIZES,seeds=range(42,48),baseline_48=None):
    model_root=Path(model_root);out=Path(out)
    if out.exists():raise FileExistsError('Use NEW analysis study dir: '+str(out))
    sizes=tuple(int(x) for x in sizes);seeds=tuple(int(x) for x in seeds)
    if len(set(sizes))!=len(sizes) or len(set(seeds))!=len(seeds):raise ValueError('Duplicate size/seed')
    records,provs=check_inputs(model_root,sizes,seeds)
    if baseline_48:
        b=Path(baseline_48)
        for seed in seeds:
            orig=read_json(b/f'vq3_loto_seed{seed}_v1'/'report.json');new=records[48,seed]
            if orig['stage']!=new['stage'] or orig['macro_task_metrics']!=new['macro_task_metrics']:
                raise ValueError('n48 view fails exact frozen VQ3 reproduction, seed='+str(seed))
    metrics={};pairs=[]
    for n in sizes:
        metrics[str(n)]={}
        for h in ('20','50'):
            metrics[str(n)][h]={}
            for pol in WATCH:
                z={}
                for key in MEASURES:
                    vals=[records[n,s]['macro_task_metrics'][h][pol][key] for s in seeds]
                    z[key]={'seed_mean':float(np.mean(vals)),'seed_min':float(min(vals)),
                            'seed_max':float(max(vals))}
                metrics[str(n)][h][pol]=z
            for left,right in [('VQ_MEAN','E0_MATCHED'),('RESID_OOF50','E0_MATCHED'),
                               ('PAIR_RANK','E0_MATCHED'),('E0_RISK35','E0_SCORE_COVERAGE35')]:
                taskmeans={}
                for task in sorted(records[n,seeds[0]]['folds'][h]):
                    diffs=[compare(records[n,s]['folds'][h][task],left,right,'mean_gain_sec') for s in seeds]
                    taskmeans[task]=float(np.mean(diffs))
                seed_diffs=[records[n,s]['macro_task_metrics'][h][left]['mean_gain_sec']-
                            records[n,s]['macro_task_metrics'][h][right]['mean_gain_sec'] for s in seeds]
                pairs.append({'n_states':n,'h':int(h),'left':left,'right':right,
                              'mean_gain_difference_sec':float(np.mean(seed_diffs)),
                              'positive_seeds':int(np.sum(np.array(seed_diffs)>0)),
                              'per_task_mean_difference_sec':taskmeans,
                              'positive_tasks':int(sum(v>0 for v in taskmeans.values())),
                              'n_independent_task_groups':4})
    out.mkdir(parents=True)
    write_json(out/'report.json',{'stage':'R3-VQ4-nested-learning-curves-v1','status':'COMPLETE',
        'sizes':list(sizes),'seeds':list(seeds),'metrics':metrics,'paired_task_differences':pairs,
        'limits':['Only four original TRAIN tasks, more states are correlated within each task.',
                  'Seeds re-use identical labels and tasks, not independent test replications.',
                  '48/96/144/192 are nested prefixes of one outcome-blind state selection.',
                  'True online 30s ALNS gain is NOT measured; labels are C3-restart surrogates.',
                  'Use task-wise signs, not a nominal 24-seed significance test.',
                  'Original R2 validation/internal_test states are not training labels.']})
    write_json(out/'provenance.json',{'study_reports':{f'{n}_{seed}':sha256(model_root/f'vq4_n{n}_seed{seed}_v1'/'report.json')
                                                for n in sizes for seed in seeds},
                  'analysis_code_sha256':sha256(Path(__file__))})
    md=['# Step6-C1-R3-VQ4: 嵌套样本学习曲线','',
        '仅使用4个原Train任务；六种子不是六份独立测试。所有多步价值均为C3重启代理。','']
    for h in ('20','50'):
        md.extend([f'## H={h}','','| 策略 | '+ ' | '.join(f'n={n}' for n in sizes)+' |',
                   '|---|'+'---:|'*len(sizes)])
        for pol in WATCH:
            md.append('| '+pol+' | '+' | '.join(f"{metrics[str(n)][h][pol]['mean_gain_sec']['seed_mean']:+.3f}"
                                                  for n in sizes)+' |')
        md+=['','### 关键配对差（秒/状态）','',
             '| n | 对照 | 平均差 | 正向任务数/4 | 正向训练种子数/6 |',
             '|---:|---|---:|---:|---:|']
        for p in pairs:
            if p['h']==int(h):
                md.append(f"| {p['n_states']} | {p['left']} − {p['right']} | {p['mean_gain_difference_sec']:+.3f} | {p['positive_tasks']}/4 | {p['positive_seeds']}/{len(seeds)} |")
        md+=['']
    md.append('结论必须以任务级配对方向、regret、覆盖率和有害覆盖为依据；不允许只因样本量增大就宣称独立任务泛化。')
    (out/'report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    return pairs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--sizes',default='48,96,144,192');p.add_argument('--seeds',default='42,43,44,45,46,47')
    p.add_argument('--baseline48-root')
    a=p.parse_args()
    compare_rows=analyze(a.model_root,a.output_dir,tuple(map(int,a.sizes.split(','))),
                         tuple(map(int,a.seeds.split(','))),a.baseline48_root)
    print('VQ4 LEARNING CURVES COMPLETE',a.output_dir,'comparisons',len(compare_rows))
