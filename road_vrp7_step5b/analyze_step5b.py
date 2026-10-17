#!/usr/bin/env python3
"""Strict full grid + task-paired Section ablation analysis (no claimed blind July test)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_stage3'))
from road_vrp import read_json,write_json

KEY=['task_id','scenario_id','planner','seed']
CONTRASTS=[('section_random','baseline'),('section_static','baseline'),
           ('section_traffic','section_static'),('section_traffic','section_static_probe'),('section_traffic','section_shuffled'),
           ('at_rr_section','section_traffic'),('section_traffic','baseline')]


def analyze(experiment_dir,output_dir,seed=20261017):
    base=Path(experiment_dir);out=Path(output_dir)
    config=read_json(base/'run_config.json');summary=read_json(base/'benchmark_summary.json')
    data=pd.read_csv(base/'metrics.csv')
    if summary.get('status')!='COMPLETE':raise ValueError('Benchmark is not COMPLETE')
    expected={(Path(p).stem,sc,v,plan,int(s)) for p in config['task_sha256']
              for sc in config['scenarios'] for v in config['variants']
              for plan in config['planners'] for s in config['seeds']}
    actual={(r.task_id,r.scenario_id,r.solver,r.planner,int(r.seed)) for r in data.itertuples()}
    if len(data)!=config['expected_count'] or actual!=expected or len(data)!=len(actual):
        raise ValueError(f'Incomplete/duplicate experiment grid: rows={len(data)} expected={len(expected)}')
    if data.ex_post_road_travel_sec.isna().any() or (data.ex_post_road_travel_sec<=0).any():
        raise ValueError('Invalid/missing road evaluation')
    allowed={key:group for key,group in data.groupby(['solver','planner'])}
    report=(data.groupby(['solver','planner'],as_index=False)
            .agg(n=('ex_post_road_travel_sec','size'),
                 mean_road_sec=('ex_post_road_travel_sec','mean'),
                 median_road_sec=('ex_post_road_travel_sec','median'),
                 mean_search_sec=('search_wall_sec','mean'),
                 mean_iterations=('iterations','mean'),
                 mean_score_calls=('score_calls','mean'),
                 mean_section_calls=('section_moves','mean'),
                 mean_section_swap=('section_swaps','mean'),
                 mean_traffic_change=('section_traffic_change_rate','mean'),
                 mean_support_proxy=('section_mean_support_proxy','mean'),
                 mean_observed_share=('observed_edge_share','mean'),
                 mean_failed_repairs=('failed_repairs','mean')))
    report['frac_90pct_budget']=report.apply(lambda x:float(
        (allowed[(x['solver'],x['planner'])].search_wall_sec>=.9*config['wall_sec']).mean()),axis=1)
    comps=[];rng=np.random.default_rng(seed)
    for planner in config['planners']:
        for new,reference in CONTRASTS:
            if new not in config['variants'] or reference not in config['variants']:continue
            a=data[(data.solver==new)&(data.planner==planner)].set_index(KEY)
            b=data[(data.solver==reference)&(data.planner==planner)].set_index(KEY)
            if a.index.has_duplicates or b.index.has_duplicates or set(a.index)!=set(b.index):
                raise ValueError('Unpaired comparison '+new+' vs '+reference)
            pair=a[['ex_post_road_travel_sec']].join(b[['ex_post_road_travel_sec']],lsuffix='_new',rsuffix='_old')
            pair=pair.reset_index()
            gain=100*(pair.ex_post_road_travel_sec_old-pair.ex_post_road_travel_sec_new)/pair.ex_post_road_travel_sec_old
            pair['gain_pct']=gain
            by_task=pair.groupby('task_id').gain_pct.mean().to_numpy()
            bootstrap=rng.choice(by_task,size=(3000,len(by_task)),replace=True).mean(axis=1)
            comps.append({'new':new,'reference':reference,'planner':planner,'pairs':len(pair),'tasks':len(by_task),
                          'mean_gain_pct':float(gain.mean()),
                          'ci_low_pct':float(np.quantile(bootstrap,.025)),
                          'ci_high_pct':float(np.quantile(bootstrap,.975)),
                          'paired_win_rate':float((gain>0).mean()),
                          'positive_tasks':int((by_task>0).sum()),
                          'mean_saved_sec':float((pair.ex_post_road_travel_sec_old-
                                                  pair.ex_post_road_travel_sec_new).mean())})
    operator={}
    diagnostics_rows=[]
    keys=[]
    for line in (base/'solutions.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        r=json.loads(line)
        k=(r['task_id'],r['scenario_id'],r['solver'],r['planner'],int(r['seed']))
        keys.append(k)
        audit=r['operator_audit']
        for name,d in audit['destroy'].items():
            rec={'task_id':r['task_id'],'scenario_id':r['scenario_id'],'solver':r['solver'],
                 'planner':r['planner'],'seed':r['seed'],'operator':name,
                 'calls':d['calls'],'sec':d['sec'],'mean_ms':1000*d['sec']/max(d['calls'],1)}
            # Pair event attribution to current destroy + repair (NOT individual causal effects).
            combos=[p for p in audit['pairs'] if p.startswith(name+'|')]
            rec['accepted']=sum(audit['pairs'][p]['accepted'] for p in combos)
            rec['improved']=sum(audit['pairs'][p]['positive'] for p in combos)
            rec['new_best']=sum(audit['pairs'][p]['new_best'] for p in combos)
            diagnostics_rows.append(rec)
    if set(keys)!=expected or len(keys)!=len(set(keys)):raise ValueError('Incomplete solutions/telemetry grid')
    output=out
    output.mkdir(parents=True,exist_ok=True)
    report.to_csv(output/'solver_summary.csv',index=False)
    pd.DataFrame(comps).to_csv(output/'paired_gains.csv',index=False)
    events=pd.DataFrame(diagnostics_rows)
    events.to_csv(output/'operator_events.csv',index=False)
    if not events.empty:
        grouped=(events.groupby(['solver','planner','operator'],as_index=False)
                 .agg(calls=('calls','sum'),seconds=('sec','sum'),accepted=('accepted','sum'),
                      positive=('improved','sum'),new_best=('new_best','sum')))
        grouped['mean_ms_per_call']=1000*grouped.seconds/grouped.calls.clip(lower=1)
        grouped['accept_rate']=grouped.accepted/grouped.calls.clip(lower=1)
        grouped['new_best_rate']=grouped.new_best/grouped.calls.clip(lower=1)
        grouped.to_csv(output/'operator_contribution.csv',index=False)
    result={'experiment_complete':True,'n_runs':len(data),'expected':len(expected),
            'comparisons':len(comps),'developer_warning':'July26-31 was inspected in Stage3, July results are exploratory development',
            'support_warning':'observed-edge support is a static-path departure-snapshot proxy, NOT dynamic-path real observation',
            'operator_attribution':'destroy-repair pair outcome assigned to destroy for descriptive purposes, not causal',
            'algorithm_warning':'Extra Section destroy is added to the pool; B3/B2 isolate traffic score within the SAME Section repair mechanism.',
            'bootstrap':'Task-level resampling 3000; dates and seeds not independent'}
    write_json(output/'analysis_summary.json',result)
    print(report.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nPAIRED\n',pd.DataFrame(comps).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment-dir',required=True)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args();analyze(a.experiment_dir,a.output_dir)
if __name__=='__main__':main()
