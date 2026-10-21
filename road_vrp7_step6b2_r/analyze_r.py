#!/usr/bin/env python3
"""Strict key-matched road replay + R selector overhead analysis."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def read_run(path):
    p=Path(path)
    m=json.loads((p/'manifest.json').read_text())
    s=json.loads((p/'summary.json').read_text())
    if s.get('status')!='COMPLETE':raise ValueError('Incomplete run: '+str(p))
    with (p/'metrics.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if len(rows)!=s['expected_records']:raise ValueError('Record count mismatch')
    indexed={}
    for x in rows:
        k=(x['task_id'],x['scenario_id'],int(x['seed']))
        if k in indexed:raise ValueError('Duplicate key: '+str(k))
        indexed[k]=x
    return m,indexed

def compare(base,treat,output):
    a,aa=read_run(base);b,bb=read_run(treat)
    must_match=('source_sha256','road_sha256','edge_sha256','solver_sha256',
      'section_sha256','split','wall_sec','max_iterations','anneal_iterations',
      'top_k','seeds','task_ids','scenario_ids','r_score_every',
      'r_max_scoring_sec','r_cache_size','r_min_static_fraction','r_fallback',
      'action_count','sampler_seed')
    for k in must_match:
        if a.get(k)!=b.get(k):raise ValueError(f'Unmatched {k}: {a.get(k)} vs {b.get(k)}')
    if aa.keys()!=bb.keys():raise ValueError('Unequal task/scenario/seed keys')
    paired=[];bytask=defaultdict(list);bydate=defaultdict(list)
    def f(row,key):return float(row.get(key,0.) or 0.)
    for k in sorted(aa):
        x,y=aa[k],bb[k]
        delta=f(x,'ex_post_road_travel_sec')-f(y,'ex_post_road_travel_sec')
        bytask[k[0]].append(delta);bydate[x['date']].append(delta)
        paired.append({'key':k,'improvement_sec':delta,
           'base_road_sec':f(x,'ex_post_road_travel_sec'),
           'treatment_road_sec':f(y,'ex_post_road_travel_sec'),
           'base_iterations':f(x,'iterations'),'treatment_iterations':f(y,'iterations'),
           'model_calls':f(y,'r_model_eval_calls'),'score_calls':f(y,'r_scoring_calls'),
           'scoring_sec':f(y,'r_scoring_seconds'),
           'inference_sec':f(y,'section_inference_wall_sec'),
           'c3_fallbacks':f(y,'r_c3_fallbacks'),
           'gate_fallbacks':f(y,'r_gate_fallbacks'),
           'learned_actions':f(y,'r_model_actions')})
    dif=np.asarray([z['improvement_sec'] for z in paired])
    result={'stage':'step6b2r-paired-v1','baseline':str(base),'treatment':str(treat),
       'records':len(paired),'mean_road_improvement_sec':float(np.mean(dif)),
       'median_road_improvement_sec':float(np.median(dif)),
       'wins':int(np.sum(dif>1e-7)),'losses':int(np.sum(dif< -1e-7)),
       'ties':int(np.sum(abs(dif)<=1e-7)),
       'mean_relative_road_improvement':float(np.mean([z['improvement_sec']/z['base_road_sec'] for z in paired])),
       'mean_inference_sec':float(np.mean([z['inference_sec'] for z in paired])),
       'mean_scoring_sec':float(np.mean([z['scoring_sec'] for z in paired])),
       'mean_model_calls':float(np.mean([z['model_calls'] for z in paired])),
       'mean_c3_fallbacks':float(np.mean([z['c3_fallbacks'] for z in paired])),
       'mean_learned_actions':float(np.mean([z['learned_actions'] for z in paired])),
       'by_task':{k:float(np.mean(v)) for k,v in bytask.items()},
       'by_date':{k:float(np.mean(v)) for k,v in bydate.items()},
       'notes':['Identical task/scenario/seed required. Inference+feature costs are inside 30s search.',
          'Comparison is July development, not blind. Source states used for training may overlap test tasks.',
          'Difference includes changed Section sampling and fallback behaviour; compare sampled_random and C3.'],
       'rows':paired}
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2,ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True)
    p.add_argument('--treatment',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();compare(a.baseline,a.treatment,a.output)
if __name__=='__main__':main()
