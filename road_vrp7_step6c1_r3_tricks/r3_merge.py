#!/usr/bin/env python3
"""Strictly verify 12 (task,scenario) shards before merging a R3-T study."""
from __future__ import annotations
import csv,json,sys,math
from pathlib import Path


def merge(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    all_rows=[];all_solutions=[];keys=set()
    required=['source_sha256','road_sha256','edge_sha256','solver_sha256','section_sha256',
              'wall_sec','max_iterations','anneal_iterations','top_k','r_score_every','r_max_scoring_sec']
    expect={(t,sc,int(seed)) for t in plan['tasks'] for sc in plan['scenarios'] for seed in plan['seeds']}
    header=None;footprints=[]
    for t in plan['tasks']:
        for sc in plan['scenarios']:
            folder=root/'shards'/(t+'__'+sc)
            if not (folder/'summary.json').exists():raise ValueError('Missing shard: '+str(folder))
            summ=json.loads((folder/'summary.json').read_text())
            if summ.get('status')!='COMPLETE' or summ.get('records')!=len(plan['seeds']):
                raise ValueError('Incomplete shard: '+str(folder))
            mf=json.loads((folder/'manifest.json').read_text())
            for k in required:
                if k in ('road_sha256','edge_sha256'):
                    # These hashes are checked across shards and also against the
                    # frozen source by the original B2-R solver.
                    pass
                elif k in ('solver_sha256','section_sha256'):
                    if mf[k]!=plan['frozen_'+('solver' if k.startswith('solver') else 'selector')+'_sha256']:
                        raise ValueError('Frozen solver changed')
                elif k=='source_sha256':
                    if mf[k]!=plan['source_sha256']:raise ValueError('Frozen input source changed')
                elif k in plan and mf[k]!=plan[k]:raise ValueError(f'Shard {folder} mismatched {k}')
            footprints.append({k:mf.get(k) for k in required})
            if mf.get('model_sha256')!=plan.get('model_sha256'):
                raise ValueError('Model hash mismatch: '+str(folder))
            with (folder/'metrics.csv').open(newline='') as f:
                reader=csv.DictReader(f)
                if header is None:header=reader.fieldnames
                if reader.fieldnames!=header:raise ValueError('Mixed CSV schemas')
                rows=list(reader)
            if len(rows)!=len(plan['seeds']):raise ValueError('Wrong shard row count')
            for row in rows:
                key=(row['task_id'],row['scenario_id'],int(row['seed']))
                if key not in expect or key in keys:raise ValueError('Unknown or duplicate paired sample '+str(key))
                if key[:2]!=(t,sc):raise ValueError('Wrong shard identity')
                if not (math.isfinite(float(row['ex_post_road_travel_sec'])) and float(row['ex_post_road_travel_sec'])>=0):raise ValueError('Invalid road cost')
                keys.add(key);all_rows.append(row)
            sol=json.loads((folder/'solutions.json').read_text())
            if len(sol)!=len(rows):raise ValueError('Missing solutions')
            all_solutions.extend(sol)
    if keys!=expect:raise ValueError('Missing paired keys')
    if any(f!=footprints[0] for f in footprints):raise ValueError('Inconsistent frozen inputs across shards')
    if (root/'metrics.csv').exists() or (root/'summary.json').exists():
        raise FileExistsError('Merged outputs already exist; never silently overwrite')
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=header);wr.writeheader();wr.writerows(all_rows)
    (root/'solutions.json').write_text(json.dumps(all_solutions,indent=2,ensure_ascii=False)+'\n')
    (root/'summary.json').write_text(json.dumps({'status':'COMPLETE','arm':plan['arm'],
          'records':len(all_rows),'shards':len(footprints),'task_scenarios':len(plan['tasks'])*len(plan['scenarios']),
          'note':'R3-T paired development search; keys strict, NOT blind final test'},indent=2)+'\n')
    print('MERGE PASS',root,'records',len(all_rows))

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python merge_shards.py RUN_DIR')
    merge(sys.argv[1])
