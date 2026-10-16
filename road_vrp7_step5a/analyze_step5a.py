#!/usr/bin/env python3
"""Audited Step5-A paired validation comparisons; no test split accepted."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_stage3'))
from road_vrp import read_json,write_json
from run_experiment import VARIANTS, KEY, _checkpoint

COMPARE=[('at_rr','at_timeonly'),('te_wr','te_local'),
         ('both','placebo_both'),('at_rr','baseline'),('te_wr','baseline'),
         ('both','baseline'),('at_timeonly','baseline'),('te_local','baseline'),
         ('placebo_both','baseline')]
PAIR=['task_id','scenario_id','planner','seed']


def paired_comparison(df,new,reference,planner):
    a=df[(df.solver==new)&(df.planner==planner)]
    b=df[(df.solver==reference)&(df.planner==planner)]
    if a.empty or b.empty:return None
    a=a.set_index(PAIR)
    b=b.set_index(PAIR)
    if a.index.has_duplicates or b.index.has_duplicates or set(a.index)!=set(b.index):
        raise ValueError(f'Unpaired comparison: {new} vs {reference}, {planner}')
    a=a.loc[b.index]
    new_cost=a.ex_post_road_travel_sec.to_numpy(dtype=float)
    old_cost=b.ex_post_road_travel_sec.to_numpy(dtype=float)
    if (new_cost<=0).any() or (old_cost<=0).any():raise ValueError('Nonpositive road cost')
    gain=100*(old_cost-new_cost)/old_cost
    tasks=np.array([k[0] for k in b.index],dtype=str)
    groups=np.unique(tasks)
    means=np.asarray([gain[tasks==task].mean() for task in groups])
    rng=np.random.default_rng(20261016)
    boot=means[rng.integers(len(means),size=(4000,len(means)))].mean(axis=1)
    return dict(new=new,reference=reference,planner=planner,pairs=len(gain),tasks=len(groups),
                mean_pair_gain_pct=float(gain.mean()),
                task_bootstrap95_low=float(np.quantile(boot,.025)),
                task_bootstrap95_high=float(np.quantile(boot,.975)),
                win_rate=float((gain>1e-9).mean()),positive_tasks=int((means>0).sum()),
                average_road_delta_sec=float((old_cost-new_cost).mean()),
                mean_new_search_sec=float(a.search_wall_sec.mean()),
                mean_ref_search_sec=float(b.search_wall_sec.mean()),
                mean_new_iterations=float(a.iterations.mean()),
                mean_ref_iterations=float(b.iterations.mean()),
                mean_new_score_calls=float(a.score_calls.mean()),
                mean_ref_score_calls=float(b.score_calls.mean()))


def analyze(stage5,output,stage3=None):
    path=Path(stage5);out=Path(output)
    cfg=read_json(path/'run_config.json')
    if cfg['split']!='validation':raise ValueError('Step5-A only analyzes frozen validation')
    if not (path/'benchmark_summary.json').exists():raise ValueError('Run is not complete')
    reported=read_json(path/'benchmark_summary.json')
    if reported.get('status')!='COMPLETE':raise ValueError('Incomplete reported run')
    exact_keys=_checkpoint(path)
    expected={(Path(p).stem,sc,method,plan,int(seed)) for p in cfg['task_sha256']
              for sc in cfg['scenario_ids'] for method in cfg['variants']
              for plan in cfg['planners'] for seed in cfg['seeds']}
    if exact_keys!=expected or len(exact_keys)!=cfg['expected_count']:
        raise ValueError('Grid missing/duplicate/unexpected keys')
    a=pd.read_csv(path/'metrics.csv',dtype={'task_id':str,'scenario_id':str,'solver':str,'planner':str})
    if (a.split!='validation').any() or not a.eligible.astype(bool).all():
        raise ValueError('Contaminated validation with test or ineligible rows')
    if a.ex_post_road_travel_sec.isna().any() or (a.ex_post_road_travel_sec<=0).any():
        raise ValueError('Invalid road costs')
    g=(a.groupby(['solver','planner'],as_index=False)
       .agg(n=('ex_post_road_travel_sec','size'),
            mean_road_sec=('ex_post_road_travel_sec','mean'),
            mean_search_sec=('search_wall_sec','mean'),
            p95_search_sec=('search_wall_sec',lambda x:float(x.quantile(.95))),
            mean_iterations=('iterations','mean'),
            mean_score_calls=('score_calls','mean'),
            mean_observed_edge_share=('observed_edge_share','mean'),
            mean_failed_repairs=('failed_repairs','mean'),
            mean_te_nonzero_calls=('te_nonzero_externality_calls','mean'),
            mean_te_static_fallback_calls=('te_static_fallback_calls','mean'),
            mean_extra_route_cost_calls=('extra_traffic_route_cost_calls','mean')))
    g['fraction_90pct_wall']=g.apply(lambda row:float((a.loc[(a.solver==row.solver)&
        (a.planner==row.planner),'search_wall_sec']>=.9*cfg['wall_sec']).mean()),axis=1)
    out.mkdir(parents=True,exist_ok=True)
    g.to_csv(out/'solver_summary.csv',index=False)
    pairs=[]
    for p in cfg['planners']:
        for new,ref in COMPARE:
            if new in cfg['variants'] and ref in cfg['variants']:
                pairs.append(paired_comparison(a,new,ref,p))
    if stage3:
        prev=Path(stage3)
        basecfg=read_json(prev/'run_config.json')
        if basecfg.get('split')!='validation':raise ValueError('Stage3 baseline not a validation split')
        for k in ('cost_file_sha256','edge_file_sha256','tasks_manifest_sha256'):
            if basecfg[k]!=cfg[k]:raise ValueError('Stage3 cost/task manifest mismatch '+k)
        if (float(basecfg['wall_sec'])!=float(cfg['wall_sec']) or
            int(basecfg['iterations'])!=int(cfg['max_iterations']) or
            int(basecfg['od_step_min'])!=int(cfg['od_step_min']) or
            float(basecfg['od_hours'])!=float(cfg['od_hours'])):
            raise ValueError('Stage3 and Step5 budgets/OD configuration differ: do not compare')
        for taskname,digest in cfg['task_sha256'].items():
            if basecfg['task_sha256'].get(taskname)!=digest:
                raise ValueError('Stage3 and Step5 task bytes differ')
        b=pd.read_csv(prev/'metrics.csv',dtype={'task_id':str,'scenario_id':str,'solver':str,'planner':str})
        b=b[b.solver=='official_alns'].copy()
        baseline=a[a.solver=='baseline']
        subset=b.merge(baseline[PAIR].drop_duplicates(),on=PAIR,how='inner',validate='many_to_one')
        if len(subset)!=len(baseline):raise ValueError('Stage3 official_ALNS missing some Stage5 baseline keys')
        required=baseline[PAIR].drop_duplicates()
        for key in ('task_id','scenario_id','planner','seed'):
            if required[key].isna().any():raise ValueError('Invalid match key')
        both=pd.concat([baseline,subset.assign(solver='stage3_official_alns')],ignore_index=True)
        for p in cfg['planners']:
            pairs.append(paired_comparison(both,'baseline','stage3_official_alns',p))
    pd.DataFrame(pairs).to_csv(out/'paired_gains.csv',index=False)
    summary={'stage':'Step5-A','validation_only':True,'experiment_complete':True,
             'count':len(a),'expected':cfg['expected_count'],
             'stage3_rechecked':bool(stage3),
             'interpretation_warnings':[
                'Oracle uses recorded future; deployable traffic prediction not demonstrated.',
                'Traffic labels are hybrid: few roads measured; unmeasured edges use static fallback.',
                'Task-bootstrap interval is exploratory because dates/seeds share road samples.',
                '5s search budget equal, but different operators consume different iterations.',
                'Stage3 score_calls counts obj.cost(), not all route_cost / OD calls; extra traffic route-cost calls logged separately.',
                'AT-RR vs at_timeonly tests congestion level/trend above the common ETA/spatial mechanism.',
                'TE-WR vs te_local tests downstream time externality vs local arc saving, not vs Stage3 total Worst.',
                'Stage3 baseline comparison requires same version, seeds, clock config and hashes.']}
    write_json(out/'analysis_summary.json',summary)
    print('SOLVER GROUPS\n',g.to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    print('\nPAIRED\n',pd.DataFrame(pairs).to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage5-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--stage3-dir',default='runs/chicago_july_v1/stage3/validation')
    a=p.parse_args()
    analyze(a.stage5_dir,a.output_dir,a.stage3_dir if a.stage3_dir!='none' else None)

if __name__=='__main__':main()
