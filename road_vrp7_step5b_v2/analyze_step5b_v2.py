#!/usr/bin/env python3
"""Fail-closed paired v2 analysis; report clustered task bootstrap intervals.

NOTE: only 10 tasks. Task-cluster bootstrap CIs are descriptive/exploratory;
repeated dates/seeds are not independent training examples.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['task_id','scenario_id','planner','seed']
PAIRS=[('C1_original_section','C0_baseline'),('C2_noop_guard','C1_original_section'),
       ('C3_topk','C2_noop_guard'),('C4_time_ucb','C3_topk'),
       ('C5_gated','C4_time_ucb'),('C5_gated','C6_shuffled'),
       ('C5_gated','C0_baseline')]


def analyze(src,output):
    src=Path(src);output=Path(output)
    config=json.loads((src/'run_config.json').read_text())
    end=json.loads((src/'benchmark_summary.json').read_text())
    if end['status']!='COMPLETE':raise ValueError('Incomplete run, cannot analyze')
    df=pd.read_csv(src/'metrics.csv')
    required=['task_id','scenario_id','planner','seed','solver','ex_post_road_travel_sec',
              'iterations','search_wall_sec','section_calls','section_changed','section_unchanged',
              'section_improving','section_new_best','section_accepted','section_candidate_exact',
              'section_noop_rejected','section_fallback','section_swap_attempts','section_swap_selected',
              'section_destroy_sec','section_repair_sec',
              'section_actual_choice_change_rate','section_mean_support_proxy']
    missing=set(required)-set(df)
    if missing:raise ValueError(f'Metrics columns missing: {missing}')
    idx=KEY+['solver'];expected={(t.removesuffix('.json'),sc,p,int(seed),v)
          for t in config['tasks'] for sc in config['scenarios'] for p in config['planners']
          for seed in config['seeds'] for v in config['variants']}
    seen={tuple(x) for x in df[idx].itertuples(index=False,name=None)}
    if len(df)!=len(expected) or len(seen)!=len(df) or expected!=seen:raise ValueError('Incomplete or duplicate record grid')
    if not np.isfinite(df.ex_post_road_travel_sec).all() or (df.ex_post_road_travel_sec<=0).any():raise ValueError('Nonfinite costs')
    for x in df.itertuples():
        if x.section_changed+x.section_unchanged >x.section_calls:
            raise ValueError('Section change counters exceed calls')
        if x.section_new_best>x.section_improving or x.section_improving>x.section_changed:
            raise ValueError('Invalid Section improvement counters')
        if x.section_swap_selected>x.section_swap_attempts:
            raise ValueError('Swap selected exceeds attempts')
    solutions=[json.loads(line) for line in (src/'solutions.jsonl').read_text().splitlines() if line.strip()]
    sol_keys=[tuple(x[k] for k in idx) for x in solutions]
    if len(sol_keys)!=len(df) or len(set(sol_keys))!=len(sol_keys) or set(sol_keys)!=seen:
        raise ValueError('Unpaired/inconsistent solutions')
    report=(df.groupby(['solver','planner'],as_index=False)
            .agg(n=('ex_post_road_travel_sec','size'),mean_road_sec=('ex_post_road_travel_sec','mean'),
                 median_road_sec=('ex_post_road_travel_sec','median'),mean_search_sec=('search_wall_sec','mean'),
                 mean_iterations=('iterations','mean'),mean_section_calls=('section_calls','mean'),
                 sum_section_changed=('section_changed','sum'),sum_section_unchanged=('section_unchanged','sum'),
                 sum_section_improving=('section_improving','sum'),sum_section_new_best=('section_new_best','sum'),
                 sum_section_accepted=('section_accepted','sum'),sum_exact_evaluated=('section_candidate_exact','sum'),
                 sum_noop_rejected=('section_noop_rejected','sum'),sum_fallback=('section_fallback','sum'),
                 sum_section_destroy_sec=('section_destroy_sec','sum'),sum_section_repair_sec=('section_repair_sec','sum'),
                 mean_choice_change=('section_actual_choice_change_rate','mean'),
                 mean_support_proxy=('section_mean_support_proxy','mean')))
    report['changed_per_call']=report.sum_section_changed/report.mean_section_calls.mul(report.n).clip(lower=1)
    report['positive_per_call']=report.sum_section_improving/report.mean_section_calls.mul(report.n).clip(lower=1)
    report['best_per_call']=report.sum_section_new_best/report.mean_section_calls.mul(report.n).clip(lower=1)
    report['section_ms_per_call']=1000*(report.sum_section_destroy_sec+report.sum_section_repair_sec)/(report.mean_section_calls*report.n).clip(lower=1)
    report['effective_search_fraction']=df.groupby(['solver','planner']).apply(
        lambda g:float((g.search_wall_sec>=.9*config['wall_sec']).mean()),include_groups=False).values
    paired=[];rng=np.random.default_rng(20261017)
    for planner in config['planners']:
        for new,ref in PAIRS:
            if new not in config['variants'] or ref not in config['variants']:continue
            a=df[(df.solver==new)&(df.planner==planner)].set_index(KEY)
            b=df[(df.solver==ref)&(df.planner==planner)].set_index(KEY)
            if set(a.index)!=set(b.index) or a.index.has_duplicates or b.index.has_duplicates:
                raise ValueError(f'Pairing mismatch {new} / {ref}')
            x=a[['ex_post_road_travel_sec']].join(b[['ex_post_road_travel_sec']],lsuffix='_new',rsuffix='_base').reset_index()
            x['gain']=100*(x.ex_post_road_travel_sec_base-x.ex_post_road_travel_sec_new)/x.ex_post_road_travel_sec_base
            task=x.groupby('task_id').gain.mean().to_numpy()
            draws=rng.choice(task,(4000,len(task)),replace=True).mean(axis=1)
            paired.append({'new':new,'reference':ref,'planner':planner,'pairs':len(x),'tasks':len(task),
                           'mean_gain_pct':float(x.gain.mean()),'task_CI_low_pct':float(np.quantile(draws,.025)),
                           'task_CI_high_pct':float(np.quantile(draws,.975)),
                           'pair_win_rate':float((x.gain>0).mean()),'positive_tasks':int((task>0).sum()),
                           'mean_saved_seconds':float((x.ex_post_road_travel_sec_base-x.ex_post_road_travel_sec_new).mean())})
    output.mkdir(parents=True,exist_ok=True)
    report.to_csv(output/'solver_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_gains.csv',index=False)
    # Per-arm selection statistics, not causal claims on individual destroy operations.
    details=[]
    for s in solutions:
        for pair,entry in s['operator_audit']['pairs'].items():
            d,r=pair.split('|',1)
            details.append({**{k:s[k] for k in idx},'destroy':d,'repair':r,**entry})
    pd.DataFrame(details).to_csv(output/'operator_pair_events.csv',index=False)
    result={'complete':True,'rows':len(df),'pairs':len(paired),
            'data_role':'July development; formerly held out July26-31 is not an unseen test',
            'objective':'Historical ex-post road replay; oracle is hindsight not deployment',
            'note':'C1 uses SINGLE Section only, not exact Step5-B v1 double-block distribution.',
            'note2':'C2 uses exhaustive exact legal non-noop insertion; C3 Top-K reduces evaluation cost.',
            'note3':'C5-C6 support is static-path snapshot proxy and zero support falls back to static score.',
            'note4':'Time selector uses immediate positive relative gain per wall-second; long-term search gain is only measured ex-post.',
            'bootstrap':'4000 task-cluster bootstraps; exploratory CI with 10 tasks'}
    (output/'analysis_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(report.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nPAIRED\n',pd.DataFrame(paired).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir',required=True);p.add_argument('--output-dir',required=True)
    x=p.parse_args();analyze(x.experiment_dir,x.output_dir)
if __name__=='__main__':main()
