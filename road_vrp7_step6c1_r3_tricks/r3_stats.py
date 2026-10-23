#!/usr/bin/env python3
"""Paired fixed-budget road-cost analysis with clustered descriptive bootstrap.

Bootstrap unit=(task_id, scenario_id), not individual seed; this prevents
presenting 120 correlated solver seeds as 120 independent road situations.
"""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
import numpy as np


def read_study(folder):
    f=Path(folder);cfg=json.loads((f/'plan.json').read_text());summary=json.loads((f/'summary.json').read_text())
    if summary.get('status')!='COMPLETE':raise ValueError('Incomplete study: '+str(f))
    with (f/'metrics.csv').open(newline='') as h:raw=list(csv.DictReader(h))
    def key(row):return (row['task_id'],row['scenario_id'],int(row['seed']))
    rows={key(x):x for x in raw}
    if len(rows)!=len(raw) or len(raw)!=summary['records']:raise ValueError('Bad paired metrics: '+str(f))
    return cfg,rows


def paired_stats(base,candidate,bootstrap=3000,seed=20261022):
    if set(base)!=set(candidate):raise ValueError('Study keys differ')
    keys=sorted(base)
    d=np.array([float(base[k]['ex_post_road_travel_sec'])-float(candidate[k]['ex_post_road_travel_sec']) for k in keys])
    clusters={}
    for k,val in zip(keys,d):clusters.setdefault(k[:2],[]).append(float(val))
    rng=np.random.default_rng(seed);vals=list(clusters.values());num=len(vals)
    boot=np.empty(bootstrap,float)
    for i in range(bootstrap):
        ids=rng.integers(0,num,size=num)
        # Weighted by sampled cluster's actual number of solver seeds.
        draw=np.concatenate([vals[j] for j in ids]);boot[i]=draw.mean()
    return {'n_runs':len(keys),'n_task_scenario_clusters':num,
       'mean_improvement_sec':float(d.mean()),'median_improvement_sec':float(np.median(d)),
       'win':int((d>1e-8).sum()),'loss':int((d< -1e-8).sum()),'tie':int((abs(d)<=1e-8).sum()),
       'min_improvement_sec':float(d.min()),'max_improvement_sec':float(d.max()),
       'cluster_bootstrap_95pct_interval_sec':[float(z) for z in np.quantile(boot,[.025,.975])],
       'per_cluster_mean':{f'{a}|{b}':statistics.mean(x) for (a,b),x in clusters.items()},
       'per_task_mean':{t:float(np.mean([v for key,v in zip(keys,d) if key[0]==t])) for t in sorted({k[0] for k in keys})},
       'per_scenario_mean':{sc:float(np.mean([v for key,v in zip(keys,d) if key[1]==sc])) for sc in sorted({k[1] for k in keys})},
       'per_solver_seed_mean':{str(se):float(np.mean([v for key,v in zip(keys,d) if key[2]==se])) for se in sorted({k[2] for k in keys})}}


def report(entries,output,baseline='C3'):
    loaded={k:read_study(v) for k,v in entries.items()}
    if baseline not in loaded:raise ValueError('Baseline missing '+baseline)
    fixed=('tasks','scenarios','seeds','source_sha256','frozen_solver_sha256',
           'frozen_selector_sha256','wall_sec','max_iterations','anneal_iterations','top_k',
           'action_count','sampler_seed','score_every','max_scoring_sec')
    source=loaded[baseline][0]
    for name,(cfg,_) in loaded.items():
        for k in fixed:
            if cfg.get(k)!=source.get(k):raise ValueError(f'{name} inconsistent fixed design {k}')
    results={}
    for name in loaded:
        if name==baseline:continue
        results[name]={baseline:paired_stats(loaded[baseline][1],loaded[name][1])}
        if 'M0' in loaded and name!='M0':
            results[name]['M0']=paired_stats(loaded['M0'][1],loaded[name][1])
        if 'R2_F2' in loaded and name!='R2_F2':
            results[name]['R2_F2']=paired_stats(loaded['R2_F2'][1],loaded[name][1])
    result={'stage':'step6c1r2b-paired-analysis-v1','baseline':baseline,
        'inputs':{name:str(path) for name,path in entries.items()},'comparisons':results,
        'caveats':['Same task/scenario/seed controls, but differing model decisions induce divergent ALNS trajectories.',
          'Baseline C3 original top20pct sampler differs from learned 8-action subset.',
          'M0 cache64 vs R2 cache0 preserve original algorithms, not isolated runtime parity.',
          'All July 22-24 scenarios are development/validation; no blind generalization claim.',
          'Only 2 distinct tasks and 6 scenarios: task/scenario cluster CI is descriptive, not significance.',
          'Per-task/per-scenario means diagnose heterogeneity; shared dates/tasks are correlated.',
          'Posthoc Road.replay outcomes cannot be used to set threshold or choose controls on this evaluation.']}
    out=Path(output)
    if out.exists():raise FileExistsError('Use a NEW comparison output directory')
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# Step6-C1-R2-B 120例配对报告','','改善为正表示候选成本更低。','',
           '|方法|参考|平均改善(秒)|胜/负/平|按任务×场景簇Bootstrap 95%区间|',
           '|---|---|---:|---:|---|']
    for name,comparisons in results.items():
        for ref,stats in comparisons.items():
            lo,hi=stats['cluster_bootstrap_95pct_interval_sec']
            lines.append(f'|{name}|{ref}|{stats["mean_improvement_sec"]:+.4f}|{stats["win"]}/{stats["loss"]}/{stats["tie"]}|[{lo:+.3f}, {hi:+.3f}]|')
    lines.extend(['','**注意**：这是7月开发集扩大验证，不是独立盲测。',''])
    (out/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,help='name=existing_study_dir')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args();entries={}
    for s in a.runs:
        if '=' not in s: p.error('Expected NAME=DIR: '+s)
        k,v=s.split('=',1)
        if k in entries: p.error('Duplicate name '+k)
        entries[k]=v
    report(entries,a.output_dir)
