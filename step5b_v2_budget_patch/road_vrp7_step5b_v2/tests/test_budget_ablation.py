#!/usr/bin/env python3
"""Offline end-to-end budget runner checks using synthetic mock results.

Does NOT pretend to test real Chicago OD data or official ALNS.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace

FILE=Path(__file__).resolve().parents[1]/'budget_ablation.py'
spec=importlib.util.spec_from_file_location('budget_runner',FILE)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

MOCK_RUN = r'''#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output-dir',required=True)
p.add_argument('--splits');p.add_argument('--planners');p.add_argument('--variants');p.add_argument('--seeds')
p.add_argument('--max-tasks',type=int);p.add_argument('--max-scenarios',type=int)
p.add_argument('--wall-sec',type=float);p.add_argument('--max-iterations',type=int)
p.add_argument('--resume',action='store_true')
a=p.parse_args()
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
config={'tasks':{'task_0000.json':'hash'},'scenarios':['2025-07-22_0800'],
        'variants':a.variants.split(','),'planners':a.planners.split(','),
        'seeds':[int(i) for i in a.seeds.split(',')],
        'wall_sec':a.wall_sec,'max_iterations':a.max_iterations,
        'expected_count':len(a.variants.split(','))*len(a.planners.split(','))*len(a.seeds.split(',')),
        'source':{'mock':'hash'}}
cf=out/'run_config.json'
if cf.exists():
    if json.loads(cf.read_text())!=config:raise RuntimeError('changed config')
    if not a.resume:raise RuntimeError('would overwrite')
else:cf.write_text(json.dumps(config))
fields=['task_id','scenario_id','solver','planner','seed','ex_post_road_travel_sec',
        'surrogate_travel_sec','search_wall_sec','iterations']
rows=[]
for s in config['variants']:
    for p in config['planners']:
        for seed in config['seeds']:
            base={'C0_baseline':1000,'C3_topk':940,'C4_time_ucb':970,'C5_gated':980}[s]
            v=base-2*a.wall_sec + seed%3
            rows.append(['task_0000','2025-07-22_0800',s,p,seed,v,v+10,a.wall_sec,
                        min(a.max_iterations,int(a.wall_sec*200))])
with (out/'metrics.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(fields);w.writerows(rows)
(out/'solutions.jsonl').write_text(''.join(json.dumps(dict(zip(fields,r)))+'\n' for r in rows))
(out/'benchmark_summary.json').write_text(json.dumps({'status':'COMPLETE','actual_count':len(rows)}))
'''
MOCK_ANALYZER = r'''#!/usr/bin/env python3
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--experiment-dir');p.add_argument('--output-dir')
a=p.parse_args();Path(a.output_dir).mkdir(parents=True,exist_ok=True)
(Path(a.output_dir)/'simulated_analysis.txt').write_text('synthetic')
'''


def main():
    assert runner.budgets_parse('30,5,15')==[5.,15.,30.]
    for v in ('0,5','5,5','a,5','-2,5','NaN,5','inf,5'):
        try:runner.budgets_parse(v)
        except ValueError:pass
        else:raise AssertionError('Budget validation failed for '+v)
    for unsafe in ('../escape','/tmp','x/y','.','..',''):
        try:runner.safe_name(unsafe)
        except ValueError:pass
        else:raise AssertionError('Unsafe study-name accepted')
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);app=root/'road_vrp7_step5b_v2';app.mkdir()
        for f,t in [('run_experiment.py',MOCK_RUN),('analyze_step5b_v2.py',MOCK_ANALYZER)]:
            path=app/f;path.write_text(t);path.chmod(0o755)
        base=root/'runs/chicago_july_v1/step5b_v2'
        (base/'validation').mkdir(parents=True)
        sentinel=base/'validation'/'DO_NOT_TOUCH'
        sentinel.write_text('frozen validation')
        params={'project_root':str(root),'study_name':'synthetic_comparison',
                'budgets':'5,15,30','variants':'C0_baseline,C3_topk,C4_time_ucb,C5_gated',
                'planners':'oracle','seeds':'42,43','splits':'validation',
                'max_tasks':1,'max_scenarios':1,'max_iterations':5000,'resume':False}
        runner.execute(SimpleNamespace(action='dry-run',**params))
        assert not (base/'budget_studies').exists(), 'dry-run unexpectedly wrote files'
        runner.execute(SimpleNamespace(action='run',**params))
        study=base/'budget_studies/synthetic_comparison'
        assert (study/'5s_iter5000/metrics.csv').exists()
        assert (study/'15s_iter5000/metrics.csv').exists()
        assert (study/'30s_iter5000/metrics.csv').exists()
        assert sentinel.read_text()=='frozen validation'
        try:runner.execute(SimpleNamespace(action='run',**params))
        except FileExistsError:pass
        else:raise AssertionError('Overwrite not blocked')
        runner.execute(SimpleNamespace(action='run',**{**params,'resume':True}))
        runner.execute(SimpleNamespace(action='analyze',**params))
        report=runner.read_json(study/'analysis/summary.json')
        assert report['complete'] and report['runs']==24
        assert len(runner.csv_rows(study/'analysis/budget_summary.csv'))==12
        assert len(runner.csv_rows(study/'analysis/budget_pairwise.csv'))==3*5
        assert len(runner.csv_rows(study/'analysis/budget_progression.csv'))==2*4
        assert sentinel.read_text()=='frozen validation'
        cfg=study/'15s_iter5000/run_config.json';data=runner.read_json(cfg)
        data['source']['mock']='different';cfg.write_text(json.dumps(data))
        try:runner.execute(SimpleNamespace(action='analyze',**params))
        except ValueError as e:assert 'unmatched config' in str(e)
        else:raise AssertionError('Mixed source/config was not rejected')
    print('PASS budget parsing, isolated paths, no overwrite, resume, cross-budget analysis and config rejection')

if __name__=='__main__':main()
