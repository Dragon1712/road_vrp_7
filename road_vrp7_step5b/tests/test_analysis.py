#!/usr/bin/env python3
"""Strict analysis smoke; ensures duplicate/missing experiment keys fail closed."""
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
util=types.ModuleType('road_vrp')
util.read_json=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
util.write_json=lambda path,o:Path(path).write_text(json.dumps(o),encoding='utf-8')
with patch.dict(sys.modules,{'road_vrp':util}):
    from analyze_step5b import analyze

def run():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); experiment=root/'run';experiment.mkdir()
        variants=['baseline','section_static','section_traffic','section_static_probe']
        cfg={'stage':'step5b-section-v1','task_sha256':{'task_0000.json':'abc','task_0001.json':'def'},
             'scenarios':['2025-07-22_0800'],'variants':variants,'planners':['oracle'],'seeds':[42],
             'wall_sec':5.,'expected_count':len(variants)*2}
        (experiment/'run_config.json').write_text(json.dumps(cfg))
        (experiment/'benchmark_summary.json').write_text(json.dumps({'status':'COMPLETE'}))
        rows=[];sol=[]
        for task in ('task_0000','task_0001'):
            for i,var in enumerate(variants):
                key={'task_id':task,'scenario_id':'2025-07-22_0800','solver':var,'planner':'oracle','seed':42}
                rows.append({**key,'ex_post_road_travel_sec':800.+i*6,'search_wall_sec':5.,
                             'iterations':90,'score_calls':240,'section_moves':35,
                             'section_swaps':2,'section_traffic_change_rate':.2,
                             'section_mean_support_proxy':.2,'section_mean_static_score':.4,
                             'section_mean_traffic_score':.3,'observed_edge_share':.15,'failed_repairs':0})
                sol.append({**key,'operator_audit':{'destroy':{'section_static':{'calls':10,'sec':.25}},
                                                  'pairs':{'section_static|greedy':{
                                                      'calls':10,'accepted':5,'positive':4,'new_best':2,
                                                      'sum_candidate_delta':4.}}}})
        def write(rows,sol):
            pd.DataFrame(rows).to_csv(experiment/'metrics.csv',index=False)
            (experiment/'solutions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in sol))
        write(rows,sol)
        result=analyze(experiment,root/'analysis')
        assert result['experiment_complete'] and result['n_runs']==8
        assert (root/'analysis/paired_gains.csv').is_file()
        write(rows[:-1],sol)
        try:analyze(experiment,root/'fail')
        except ValueError as e:assert 'Incomplete' in str(e)
        else:raise AssertionError('Incomplete grid accepted')
        write(rows,sol[:-1])
        try:analyze(experiment,root/'fail')
        except ValueError as e:assert 'Incomplete solutions' in str(e)
        else:raise AssertionError('Incomplete solutions accepted')
        print('PASS analysis full grid and negative missing-metric/missing-solution checks')

if __name__=='__main__':run()
