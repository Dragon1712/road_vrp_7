#!/usr/bin/env python3
"""Strict matched 30-second road replay comparison among B3/B2-R arms.

Compare M3 vs frozen M0 and sampled-random FIRST. Differences vs C3 also
change the Section sampler, so they are not a pure model contribution.
"""
from __future__ import annotations
import argparse,json,csv,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

FIELDS=('source_sha256','road_sha256','edge_sha256','solver_sha256',
 'section_sha256','split','wall_sec','max_iterations','anneal_iterations',
 'top_k','seeds','task_ids','scenario_ids','r_score_every','r_max_scoring_sec',
 'r_cache_size','r_min_static_fraction','r_fallback','action_count','sampler_seed')


def readrun(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    if summary['status']!='COMPLETE' or summary['records']!=summary['expected_records']:
        raise ValueError('Incomplete online study: '+str(folder))
    with (folder/'metrics.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=summary['records']:raise ValueError('Metrics length mismatch')
    data={}
    for row in rows:
        key=(row['task_id'],row['scenario_id'],int(row['seed']))
        if key in data:raise ValueError('Duplicate online key')
        data[key]=row
    return manifest,data


def getfloat(row,col):return float(row.get(col) or 0.)


def run(arms,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if not {'M0','M3'}.issubset(arms):raise ValueError('M0 and M3 required')
    loaded={name:readrun(folder) for name,folder in arms.items()}
    base_manifest,base=loaded['M0']
    for name,(manifest,rows) in loaded.items():
        for f in FIELDS:
            if manifest.get(f)!=base_manifest.get(f):
                raise ValueError(f'Experiment mismatch {name}: {f}: {manifest.get(f)} != {base_manifest.get(f)}')
        if rows.keys()!=base.keys():raise ValueError(f'Unmatched task/scenario/seed keys for {name}')
    out={'stage':'step6b3r-online-paired-v1','arms':{k:str(p) for k,p in arms.items()},
      'primary_baseline':'M0','comparisons':{},'rows':[],
      'notes':['Positive improvement = reduction in ex_post_road_travel_sec.',
       'M0 vs M3 uses same 8-section sampler and C3 fallback; C3 vs M3 has sampler confound.',
       'Model selection still based on July development data, not blind.',
       'Seeds repeated on the same task/date clusters; eight cases do not imply 8 independent routing instances.',
       'Only search budget is 30s; history setup and OD precomputation reported separately.',
       'No confidence intervals for 2-task pilot; must expand to truly independent tasks/dates.']}
    keys=sorted(base)
    for name,(mf,rows) in loaded.items():
        dd=[];bytask=defaultdict(list);bydate=defaultdict(list)
        for k in keys:
            b=base[k];r=rows[k]
            diff=getfloat(b,'ex_post_road_travel_sec')-getfloat(r,'ex_post_road_travel_sec')
            dd.append(diff);bytask[k[0]].append(diff);bydate[b['date']].append(diff)
            out['rows'].append({'arm':name,'task_id':k[0],'scenario_id':k[1], 'seed':k[2],
              'improvement_sec_vs_M0':diff,'base_road_sec':getfloat(b,'ex_post_road_travel_sec'),
              'road_sec':getfloat(r,'ex_post_road_travel_sec'),
              'iterations':int(r['iterations']),'model_actions':int(r.get('r_model_actions') or 0),
              'model_eval_calls':int(r.get('r_model_eval_calls') or 0),
              'c3_fallbacks':int(r.get('r_c3_fallbacks') or 0),
              'feature_sec':getfloat(r,'section_feature_wall_sec'),
              'inference_sec':getfloat(r,'section_inference_wall_sec'),
              'precompute_sec':getfloat(r,'precompute_wall_sec')})
        d=np.asarray(dd,float)
        vals=list(rows.values())
        def mean(col):return float(np.mean([getfloat(r,col) for r in vals]))
        out['comparisons'][name]={
          'mean_road_cost_sec':mean('ex_post_road_travel_sec'),
          'mean_road_improvement_vs_M0_sec':float(d.mean()),
          'median_road_improvement_vs_M0_sec':float(np.median(d)),
          'mean_relative_improvement_vs_M0':float(np.mean([
               dd[i]/getfloat(base[keys[i]],'ex_post_road_travel_sec') for i in range(len(keys))])),
          'wins':int(np.sum(d>1e-7)),'losses':int(np.sum(d<-1e-7)),'ties':int(np.sum(abs(d)<=1e-7)),
          'mean_iterations':mean('iterations'),
          'mean_feature_sec':mean('section_feature_wall_sec'),
          'mean_inference_sec':mean('section_inference_wall_sec'),
          'mean_scoring_sec':mean('r_scoring_seconds'),
          'mean_model_actions':mean('r_model_actions'),
          'mean_c3_fallbacks':mean('r_c3_fallbacks'),
          'mean_precompute_wall_sec':mean('precompute_wall_sec'),
          'by_task':{k:float(np.mean(z)) for k,z in sorted(bytask.items())},
          'by_date':{k:float(np.mean(z)) for k,z in sorted(bydate.items())},
          'paired_diffs_sec':list(map(float,d))}
    output.mkdir(parents=True,exist_ok=False)
    (output/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'paired.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(out['rows'][0]));w.writeheader();w.writerows(out['rows'])
    print(json.dumps(out['comparisons'],ensure_ascii=False,indent=2))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True,help='frozen P2-F2 online M0 folder')
    p.add_argument('--m3',required=True,help='B3 online M3 folder')
    p.add_argument('--arm',action='append',default=[],metavar='NAME=PATH',help='additional matched arms')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    arms={'M0':a.base,'M3':a.m3}
    for x in a.arm:
        name,sep,path=x.partition('=')
        if not sep or not name or not path or name in arms: p.error('Expected new unique NAME=PATH')
        arms[name]=path
    run(arms,a.output_dir)
if __name__=='__main__':main()
