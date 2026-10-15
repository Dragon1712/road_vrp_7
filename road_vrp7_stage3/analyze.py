#!/usr/bin/env python3
"""Strict complete-grid reporting for July stage3. No partial paired gains."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from road_vrp import write_json,read_json

KEY=['task_id','scenario_id','solver','planner','seed']


def analyze(metrics,runconfig,out,seed=20261015):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(metrics)
    cfg=read_json(runconfig)
    if df.duplicated(KEY).any():raise ValueError('Duplicate benchmark run keys')
    req={'ex_post_road_travel_sec','total_solver_wall_sec','observed_edge_share'}
    if not set(KEY+list(req)).issubset(df.columns):raise ValueError('Missing benchmark columns')
    if df[list(req)].isna().any().any() or (df.ex_post_road_travel_sec<=0).any():raise ValueError('Missing or invalid metric values')
    expected={(Path(tid).stem,sc,met,planner,seed) for tid in cfg['task_sha256']
              for sc in cfg['selected_scenario_ids'] for met in cfg['methods']
              for planner in cfg['planners'] for seed in cfg['seeds']}
    actual={(r.task_id,r.scenario_id,r.solver,r.planner,int(r.seed)) for r in df.itertuples()}
    missing=expected-actual;extra=actual-expected
    if missing or extra:raise ValueError(f'Incomplete result grid: expected {len(expected)} actual {len(actual)} missing={list(missing)[:3]} extra={list(extra)[:3]}')
    result=(df.groupby(['solver','planner'],as_index=False)
              .agg(records=('ex_post_road_travel_sec','size'),
                   mean_ex_post_sec=('ex_post_road_travel_sec','mean'),
                   median_ex_post_sec=('ex_post_road_travel_sec','median'),
                   std_ex_post_sec=('ex_post_road_travel_sec','std'),
                   mean_wall_sec=('total_solver_wall_sec','mean'),
                   mean_observed_share=('observed_edge_share','mean'),
                   mean_score_calls=('score_calls','mean')))
    result.to_csv(out/'solver_summary.csv',index=False)
    gains=[]
    rng=np.random.default_rng(seed)
    def add_pair(aa,bb,label,reference):
        # Rows are *strictly* paired by task, scenario and seed (and planner in between-solver comparisons).
        a=aa.set_index(['task_id','scenario_id','seed'])[['ex_post_road_travel_sec']]
        b=bb.set_index(['task_id','scenario_id','seed'])[['ex_post_road_travel_sec']]
        if a.index.has_duplicates or b.index.has_duplicates or set(a.index)!=set(b.index):raise ValueError('Unpaired comparison '+label)
        pair=a.join(b,lsuffix='_a',rsuffix='_b',how='inner').reset_index()
        pair['gain_pct']=100*(pair.ex_post_road_travel_sec_b-pair.ex_post_road_travel_sec_a)/pair.ex_post_road_travel_sec_b
        tasks=sorted(pair.task_id.unique())
        task_gain=pair.groupby('task_id').gain_pct.mean().reindex(tasks).to_numpy()
        boots=np.mean(rng.choice(task_gain,size=(3000,len(tasks)),replace=True),axis=1)
        gains.append({'comparison':label,'reference':reference,'num_tasks':len(tasks),'paired_runs':len(pair),
                      'mean_gain_pct':float(task_gain.mean()),'ci_2_5_pct':float(np.percentile(boots,2.5)),
                      'ci_97_5_pct':float(np.percentile(boots,97.5)),
                      'paired_win_rate':float((pair.gain_pct>0).mean()),
                      'positive_tasks':int((task_gain>0).sum())})
    for plan in cfg['planners']:
        if 'official_alns' in cfg['methods']:
            for baseline in cfg['methods']:
                if baseline=='official_alns':continue
                aa=df[(df.solver=='official_alns')&(df.planner==plan)]
                bb=df[(df.solver==baseline)&(df.planner==plan)]
                add_pair(aa,bb,'official_alns vs '+baseline+' / '+plan,'official_alns')
    if 'static' in cfg['planners'] and 'oracle' in cfg['planners']:
        for method in cfg['methods']:
            aa=df[(df.solver==method)&(df.planner=='oracle')]
            bb=df[(df.solver==method)&(df.planner=='static')]
            add_pair(aa,bb,'oracle vs static / '+method,'oracle')
    pd.DataFrame(gains).to_csv(out/'paired_gains.csv',index=False)
    daily=df.groupby(['date','solver','planner'],as_index=False).agg(mean_ex_post_sec=('ex_post_road_travel_sec','mean'),
                                                count=('ex_post_road_travel_sec','size'),
                                                observed_share=('observed_edge_share','mean'))
    daily.to_csv(out/'by_date.csv',index=False)
    report={'status':'COMPLETE','grid_records':len(df),'task_count':len(cfg['task_sha256']),
            'dates':sorted(df.date.unique().tolist()),'scenario_count':len(cfg['selected_scenario_ids']),
            'comparison_count':len(gains),'main_metric':'ex_post_road_travel_sec',
            'ci_method':'bootstrap tasks, 3000 resamples; exploratory, dates are repeated environments',
            'limitations':['Historical road labels + unmeasured OSM fallback, not full ground truth',
                           'Oracle uses hindsight future traffic, not deployable',
                           'Synthetic delivery customers/demands, not real freight orders',
                           'Greedy construction time and search time reported separately',
                           'Shared traffic days and seeds are NOT independent task replicates']}
    write_json(out/'analysis_summary.json',report)
    print(result.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\n',pd.DataFrame(gains).to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metrics',required=True)
    p.add_argument('--run-config',default='')
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    config=args.run_config or str(Path(args.metrics).parent/'run_config.json')
    analyze(args.metrics,config,args.output_dir)
if __name__=='__main__':main()
