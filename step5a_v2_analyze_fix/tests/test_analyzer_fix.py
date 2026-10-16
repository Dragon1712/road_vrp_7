#!/usr/bin/env python3
"""Offline regression reproducing real Stage3/Step5A field spellings and Pilot subset.

Tests analyzer source itself without needing the expensive Road/ALNS code or 4.7MB
GitHub solutions file. Synthetic data is not presented as experimental results.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2),encoding='utf-8')

mroad=types.ModuleType('road_vrp');mroad.read_json=read_json;mroad.write_json=write_json
mrunner=types.ModuleType('run_experiment');mrunner.VARIANTS={'baseline':(), 'at_rr':()};mrunner.KEY=('task_id','scenario_id','solver','planner','seed')
def _checkpoint(folder):
    rows=pd.read_csv(Path(folder)/'metrics.csv')
    keys=set(tuple(r) for r in rows[list(mrunner.KEY)].itertuples(index=False,name=None))
    sol=[json.loads(s) for s in (Path(folder)/'solutions.jsonl').read_text().splitlines() if s.strip()]
    other={tuple(row[k] for k in mrunner.KEY) for row in sol}
    if len(sol)!=len(other) or other!=keys:raise ValueError('Inconsistent results')
    return keys
mrunner._checkpoint=_checkpoint
sys.modules['road_vrp']=mroad;sys.modules['run_experiment']=mrunner
spec=importlib.util.spec_from_file_location('audited_analyzer',ROOT/'analyze_step5a.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

BASE={'split':'validation','cost_file_sha256':'cost-abc','edge_file_sha256':'edges-abc',
      'tasks_manifest_sha256':'tasks-abc','task_sha256':{'task_0000.json':'taskhash'},
      'wall_sec':5.,'iterations':5000,'od_step_min':20,'od_hours':5.,
      'seeds':[42,43,44], 'selected_scenario_ids':['2025-07-22_0800','2025-07-22_1700'],
      'planners':['static','oracle'],'methods':['greedy','official_alns'],'expected_count':8}
NEW={'split':'validation','cost_file_sha256':'cost-abc','edge_file_sha256':'edges-abc',
     'task_manifest_sha256':'tasks-abc','task_sha256':{'task_0000.json':'taskhash'},
     'wall_sec':5.,'max_iterations':5000,'od_step_min':20,'od_hours':5.,
     'seeds':[42], 'scenario_ids':['2025-07-22_0800'], 'planners':['static','oracle'],
     'variants':['baseline','at_rr'],'expected_count':4}

def solution(diag,k):
    s={key:val for key,val in zip(('task_id','scenario_id','solver','planner','seed'),k)}
    return {**s,'diagnostics':diag,'routes':[[1],[2]]}

def run_test():
    assert module.validate_stage3_reference_config(BASE,NEW)
    for key in ('cost_file_sha256','edge_file_sha256'):
        q=dict(NEW,**{key:'different'})
        try:module.validate_stage3_reference_config(BASE,q)
        except ValueError:pass
        else:raise AssertionError('Mismatched digest was allowed: '+key)
    for key,value in [('task_manifest_sha256','incorrect'),('wall_sec',4.),('max_iterations',6000),
                       ('od_step_min',30),('od_hours',4.5),('seeds',[99])]:
        q=dict(NEW,**{key:value})
        try:module.validate_stage3_reference_config(BASE,q)
        except ValueError:pass
        else:raise AssertionError('Incompatible configuration was allowed: '+key)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);s3=root/'stage3';s5=root/'stage5';out=root/'analysis'
        s3.mkdir();s5.mkdir()
        write_json(s3/'run_config.json',BASE);write_json(s5/'run_config.json',NEW)
        write_json(s5/'benchmark_summary.json',{'status':'COMPLETE','actual':4,'expected':4})
        diag={'operator_contribution':{'per_destroy':{'worst':{'calls':2,'destroy_sec':0.1,'repair_sec':0.2,'accepted':1,'new_best':0}},
                                       'per_pair':{'worst|greedy':{'calls':2,'gain_sum':-1,'accepted':1,'new_best':0}},
                                       'sample_events':[{'destroy':'worst','repair':'greedy','candidate_gain_sec':-1,'accepted':True}]},
              'at_feature_mean_parts':{'distance':0.3,'eta':0.2,'demand':0.1,'level':0.01,'trend':0.002}}
        rows=[];sols=[]
        for plan in ('static','oracle'):
            for method in ('baseline','at_rr'):
                key=('task_0000','2025-07-22_0800',method,plan,42)
                rows.append(dict(zip(('task_id','scenario_id','solver','planner','seed'),key)) | {
                    'split':'validation','eligible':True,'ex_post_road_travel_sec':(100 if method=='baseline' else 95),
                    'search_wall_sec':5.0,'iterations':100,'score_calls':200,'observed_edge_share':0.1,
                    'failed_repairs':0,'te_nonzero_externality_calls':0,'te_static_fallback_calls':0,
                    'extra_traffic_route_cost_calls':0})
                sols.append(solution(diag,key))
        pd.DataFrame(rows).to_csv(s5/'metrics.csv',index=False)
        (s5/'solutions.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in sols),encoding='utf-8')
        # full baseline validation reference contains more dates/seeds; only the intersection is paired.
        old=[]
        for plan in ('static','oracle'):
            for seed in (42,43,44):
                old.append({'task_id':'task_0000','scenario_id':'2025-07-22_0800',
                            'solver':'official_alns','planner':plan,'seed':seed,
                            'ex_post_road_travel_sec':100,'search_wall_sec':5.,
                            'iterations':100,'score_calls':200})
        pd.DataFrame(old).to_csv(s3/'metrics.csv',index=False)
        summary=module.analyze(s5,out,s3)
        assert summary['stage3_rechecked'] is True
        assert summary['count']==4
        assert len(pd.read_csv(out/'paired_gains.csv'))==4
        assert len(pd.read_csv(out/'operator_pair_contribution.csv'))==4
        assert len(pd.read_csv(out/'feature_effectiveness.csv'))==4
        assert len(pd.read_csv(out/'sample_operator_events.csv'))==4
        assert (out/'analysis_summary.json').is_file()
        # Negative actual run: Stage3 has mismatched provenance -> fail closed.
        wrong=dict(BASE,**{'tasks_manifest_sha256':'bad'})
        write_json(s3/'run_config.json',wrong)
        try:module.analyze(s5,root/'negative_analysis',s3)
        except ValueError as exc:assert 'reference mismatch' in str(exc)
        else:raise AssertionError('Expected mismatch refusal')
    print('PASS: actual Stage3/Step5 config field aliases, full synthetic Pilot analysis, strict negative checks')

if __name__=='__main__':run_test()
