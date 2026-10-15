#!/usr/bin/env python3
"""Make an isolated task directory with July ALNS scenarios (never overwrite originals).

This does NOT produce July traffic edge costs. First run batch_quality_select.py,
then rebuild July dynamic_costs using existing preprocessing/OSM mapping workflow.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected',type=Path,required=True,help='selected_scenarios.json')
    p.add_argument('--task-dir',type=Path,default=Path('datasets/processed/chicago/tdcvrp30/tasks_10'))
    p.add_argument('--output-dir',type=Path,default=Path('datasets/processed/chicago/tdcvrp30/tasks_july_validation'))
    p.add_argument('--max-tasks',type=int,default=10)
    a=p.parse_args(argv)
    scenarios=json.loads(a.selected.read_text(encoding='utf-8'))
    if not scenarios:
        raise ValueError('No selected scenarios; run batch quality audit and review thresholds')
    slim=[]
    used=set()
    for item in scenarios:
        for key in ('scenario_id','departure_local','source_file'):
            if key not in item:raise ValueError(f'Scenario missing {key}')
        if item['scenario_id'] in used:raise ValueError('Duplicate scenario IDs')
        used.add(item['scenario_id'])
        slim.append({'scenario_id':item['scenario_id'],'departure_local':item['departure_local']})
    files=sorted(a.task_dir.glob('task_????.json'))
    if a.max_tasks:files=files[:a.max_tasks]
    if not files:raise FileNotFoundError(f'No task_????.json in {a.task_dir}')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if list(a.output_dir.glob('task_????.json')):
        raise ValueError('Task directory already contains tasks; refusing to overwrite')
    for fp in files:
        task=json.loads(fp.read_text(encoding='utf-8'))
        if 'scenarios' not in task:raise ValueError(f'Missing scenarios in {fp}')
        task['scenarios']=slim
        (a.output_dir/fp.name).write_text(json.dumps(task,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    manifest={'source_tasks':str(a.task_dir),'selected_scenarios':str(a.selected),
              'task_count':len(files),'scenario_count_per_task':len(slim),
              'warning':'Same customer sets as earlier tasks. Date-holdout is NOT order-set holdout. '
                        'All July scenarios require separately rebuilt July dynamic edge costs.'}
    (a.output_dir/'july_validation_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'Wrote {len(files)} new task files × {len(slim)} July scenarios: {a.output_dir}')

if __name__=='__main__':main()
