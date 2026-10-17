#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analyze_step5b_v2 import analyze

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);run=root/'run';run.mkdir()
        cfg={'tasks':{'task_0000.json':'abc','task_0001.json':'xyz'},'scenarios':['2025-07-22_0800'],
             'variants':['C0_baseline','C3_topk','C4_time_ucb','C5_gated','C6_shuffled'],
             'planners':['oracle','static'],'seeds':[42], 'wall_sec':5}
        (run/'run_config.json').write_text(json.dumps(cfg))
        (run/'benchmark_summary.json').write_text(json.dumps({'status':'COMPLETE'}))
        data=[];sol=[]
        for task in ('task_0000','task_0001'):
            for solver in cfg['variants']:
                for planner in cfg['planners']:
                    key={'task_id':task,'scenario_id':'2025-07-22_0800','planner':planner,'seed':42,'solver':solver}
                    data.append({**key,'ex_post_road_travel_sec':101. if solver=='C5_gated' else 100.,
                                 'iterations':400,'search_wall_sec':5.,'section_calls':10,'section_changed':5,
                                 'section_unchanged':5,'section_improving':2,'section_new_best':1,
                                 'section_accepted':8,'section_candidate_exact':30,'section_noop_rejected':2,
                                 'section_fallback':1,'section_swap_attempts':2,'section_swap_selected':1,'section_destroy_sec':.1,'section_repair_sec':.2,
                                 'section_actual_choice_change_rate':.2,'section_mean_support_proxy':.3})
                    sol.append({**key,'operator_audit':{'pairs':{'section|greedy':{'calls':10,'changed':5,'improving':2,'new_best':1}}}})
        def save(d,s):
            pd.DataFrame(d).to_csv(run/'metrics.csv',index=False)
            (run/'solutions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in s))
        save(data,sol)
        result=analyze(run,root/'analysis')
        assert result['complete'] and result['rows']==20
        assert (root/'analysis'/'paired_gains.csv').exists()
        save(data[:-1],sol)
        try:analyze(run,root/'negative')
        except ValueError as e:assert 'grid' in str(e)
        else:raise AssertionError('Missing result not detected')
        save(data,sol[:-1])
        try:analyze(run,root/'negative')
        except ValueError as e:assert 'solutions' in str(e)
        else:raise AssertionError('Missing solution not detected')
        bad=[dict(x) for x in data];bad[0]['section_new_best']=9
        save(bad,sol)
        try:analyze(run,root/'negative')
        except ValueError as e:assert 'Invalid Section' in str(e)
        else:raise AssertionError('Counter inconsistency not detected')
    print('PASS analyzer: complete grid, paired output, missing metrics, missing solutions, counter inconsistency')

if __name__=='__main__':main()
