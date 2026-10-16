#!/usr/bin/env python3
"""Strict July validation-only matched comparisons to FROZEN Stage3 5s baseline."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_stage3'))
from road_vrp import read_json,write_json

KEY=['task_id','scenario_id','planner','seed']
BASELINE={'sa_timecool':'sa','vnd_multistart':'vnd','alns_timecool':'official_alns'}


def analyze(stage4_dir,stage3_dir,output_dir):
    stage4_dir,stage3_dir,out=map(Path,(stage4_dir,stage3_dir,output_dir))
    cfg=read_json(stage4_dir/'run_config.json')
    baseline_cfg=read_json(stage3_dir/'run_config.json')
    if cfg['split']!='validation' or baseline_cfg['split']!='validation':
        raise ValueError('Reject test-set optimization; validation only')
    if (cfg['cost_file_sha256']!=baseline_cfg['cost_file_sha256'] or
        cfg['edge_file_sha256']!=baseline_cfg['edge_file_sha256'] or
        cfg['task_manifest_sha256']!=baseline_cfg['tasks_manifest_sha256']):
        raise ValueError('Stage3 and Stage4 data/task hashes differ')
    for k in cfg['tasks']:
        if cfg['tasks'][k]!=baseline_cfg['task_sha256'][k]:
            raise ValueError('Task content mismatch '+k)
    a=pd.read_csv(stage4_dir/'metrics.csv')
    b=pd.read_csv(stage3_dir/'metrics.csv')
    if a.duplicated(['solver']+KEY).any() or b.duplicated(['solver']+KEY).any():
        raise ValueError('Duplicate runs')
    if len(a)!=cfg['expected_count'] or len(b)!=baseline_cfg['expected_count']:
        raise ValueError('Incomplete experiment grid: total row mismatch')
    expected4={(Path(x).stem,scenario,plan,seed,method)
               for x in cfg['tasks'] for scenario in cfg['scenario_ids']
               for plan in cfg['planners'] for seed in cfg['seeds']
               for method in cfg['methods']}
    actual4={(r.task_id,r.scenario_id,r.planner,int(r.seed),r.solver)
             for r in a.itertuples()}
    if expected4!=actual4:raise ValueError('Stage4 incomplete or extra run keys')
    baseline_expected={(Path(x).stem,scenario,plan,seed,method)
               for x in baseline_cfg['task_sha256'] for scenario in baseline_cfg['selected_scenario_ids']
               for plan in baseline_cfg['planners'] for seed in baseline_cfg['seeds']
               for method in baseline_cfg['methods']}
    baseline_actual={(r.task_id,r.scenario_id,r.planner,int(r.seed),r.solver)
             for r in b.itertuples()}
    if baseline_expected!=baseline_actual:raise ValueError('Stage3 baseline incomplete')
    if set(a.solver)-set(BASELINE):raise ValueError('Unknown ablation method')
    # For pilot/smoke, preserve strict FULL baseline verification but compare only
    # the Stage4 subset, with exact expected keys checked by compare().
    keys=a[KEY].drop_duplicates()
    b_subset=b.merge(keys,on=KEY,how='inner',validate='many_to_one')
    if (a.ex_post_road_travel_sec.le(0).any() or b.ex_post_road_travel_sec.le(0).any() or
        a.ex_post_road_travel_sec.isna().any() or b.ex_post_road_travel_sec.isna().any()):
        raise ValueError('Invalid road costs')
    out.mkdir(parents=True,exist_ok=True)
    group=(a.groupby(['solver','planner'],as_index=False)
           .agg(n=('ex_post_road_travel_sec','size'),
                mean_road_sec=('ex_post_road_travel_sec','mean'),
                mean_search_sec=('search_wall_sec','mean'),
                mean_score_calls=('score_calls','mean'),
                mean_iterations=('iterations','mean'),
                median_iterations=('iterations','median'),
                mean_obs_edge_share=('observed_edge_share','mean')))
    group['fraction_spent_90pct_wall']=group.apply(lambda r:float(
        (a[(a.solver==r['solver'])&(a.planner==r['planner'])].search_wall_sec
          >=.9*cfg['wall_sec']).mean()),axis=1)
    group.to_csv(out/'stage4_solver_summary.csv',index=False)
    rows=[]
    rng=np.random.default_rng(20261016)

    def compare(left, right, label,plan):
        left=left.set_index(KEY)
        right=right.set_index(KEY)
        if left.index.has_duplicates or right.index.has_duplicates or set(left.index)!=set(right.index):
            raise ValueError('Unpaired experiment '+label)
        pair=left[['ex_post_road_travel_sec','search_wall_sec','score_calls']].join(
            right[['ex_post_road_travel_sec','search_wall_sec','score_calls']],lsuffix='_new',rsuffix='_old')
        pair=pair.reset_index()
        gain=100*(pair.ex_post_road_travel_sec_old-pair.ex_post_road_travel_sec_new)/pair.ex_post_road_travel_sec_old
        pair['gain_pct']=gain
        per_task=pair.groupby('task_id').gain_pct.mean().to_numpy()
        bootstrap=rng.choice(per_task,size=(3000,len(per_task)),replace=True).mean(axis=1)
        rows.append({'comparison':label,'planner':plan,'pairs':len(pair),
                     'task_count':len(per_task),'mean_pair_gain_pct':float(gain.mean()),
                     'task_bootstrap_ci_low':float(np.quantile(bootstrap,.025)),
                     'task_bootstrap_ci_high':float(np.quantile(bootstrap,.975)),
                     'win_rate':float((gain>0).mean()),'positive_tasks':int((per_task>0).sum()),
                     'mean_road_delta_sec':float((pair.ex_post_road_travel_sec_old-pair.ex_post_road_travel_sec_new).mean()),
                     'mean_new_search_sec':float(pair.search_wall_sec_new.mean()),
                     'mean_reference_search_sec':float(pair.search_wall_sec_old.mean()),
                     'mean_new_score_calls':float(pair.score_calls_new.mean()),
                     'mean_reference_score_calls':float(pair.score_calls_old.mean())})
        return pair

    for plan in cfg['planners']:
        for variant in cfg['methods']:
            default=BASELINE[variant]
            l=a[(a.solver==variant)&(a.planner==plan)]
            r=b_subset[(b_subset.solver==default)&(b_subset.planner==plan)]
            compare(l,r,f'{variant} vs stage3_{default}',plan)
            if variant!='alns_timecool':
                alns=b_subset[(b_subset.solver=='official_alns')&(b_subset.planner==plan)]
                compare(l,alns,f'{variant} vs stage3_official_alns',plan)
    pd.DataFrame(rows).to_csv(out/'paired_gains.csv',index=False)
    report={'validation_only':True,'stage4_n':len(a),'stage3_n':len(b),
            'verified_complete_grid':True,'comparisons':len(rows),
            'important_caveats':[
             'SA wall budget + cooling changed together relative to Stage3 SA: causal separation needs another ablation.',
             'VND multistart changes the search strategy, not only wall budget; do not call pure budget equalization.',
             'Stage3 ALNS uses 5s and timecool ALNS uses same 5s; operator configuration held fixed.',
             'Oracle uses full July future as ex-post hindsight, never causal deployment claim.',
             'Bootstrap resamples 10 tasks; days/seeds shared, uncertainty is exploratory.']}
    write_json(out/'analysis_summary.json',report)
    print(group.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nPAIRED\n',pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage4-dir',required=True)
    p.add_argument('--stage3-dir',default='runs/chicago_july_v1/stage3/validation')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    analyze(a.stage4_dir,a.stage3_dir,a.output_dir)

if __name__=='__main__':main()
