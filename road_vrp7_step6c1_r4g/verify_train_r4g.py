#!/usr/bin/env python3
"""Exact TRAIN checkpoint continuation, delegating reference to OFFICIAL ALNS 7.0.0.

Uses frozen R4-B reference observer/restored_iterator, but independently guards
TRAIN roles; never edits existing R4-A/R4-B validation experiment or solver.
"""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for d in ('road_vrp7_step6c1_r4g','road_vrp7_step6c1_r4b','road_vrp7_step6c1_r4a',
          'road_vrp7_step5b_v3','road_vrp7_step6b2_r','road_vrp7_stage3'):
    if str(ROOT/d) not in sys.path:sys.path.insert(0,str(ROOT/d))
from r4a_core import sha256,write_json
from r4b_core import require,validate_checkpoint,check_start,verify_events
from verify_r4b import reference_recording,exact_resume
from r4g_core import read_json,load_capture,select_phase_snapshots,validate_scenario_role

STAGE='R4G-TRAIN-EXACT-CONTINUATION-v1'

def preflight(a):
    require(importlib.metadata.version('alns')=='7.0.0','Require official alns==7.0.0')
    mf,metrics,snapshots=load_capture(a.capture_dir,'capture')
    require(mf['max_iterations']==20000 and mf['anneal_iterations']==5000 and mf['search_wall_sec']==30,
            'Exact 20k/30s source configuration required')
    require(sha256(ROOT/'road_vrp7_step5b_v3/run_experiment.py')==mf['solver_sha256'], 'Frozen solver SHA mismatch')
    require(sha256(ROOT/'road_vrp7_step5b_v3/section_v2.py')==mf['section_sha256'], 'Frozen Section SHA mismatch')
    require(sha256(ROOT/'road_vrp7_step6c1_r4a/r4a_core.py')==mf['core_sha256'], 'R4A observer SHA mismatch')
    require(sha256(ROOT/'road_vrp7_step6c1_r4g/collect_train_r4g.py')==mf['collector_sha256'],
            'R4G source collector SHA mismatch')
    require(sha256(Path(a.model_source)/'run_config.json')==mf['source_sha256'], 'Training source SHA mismatch')
    from road_vrp import Road,OD,Objective,legal,read_json
    from benchmark import load_tasks
    import run_experiment as v3
    road=Road(a.cost_dir)
    require(road.cost_file_hash==mf['road_sha256'] and road.edge_file_hash==mf['edge_sha256'], 'Road/edge SHA mismatch')
    _,tasks,tm=load_tasks(a.task_dir)
    require(tm['cost_edge_npz_sha256']==road.cost_file_hash and tm['cost_edge_index_sha256']==road.edge_file_hash,
            'Task manifest road mismatch')
    taskmap={t['task_id']:t for t in tasks}
    require(set(mf['task_ids'])<=taskmap.keys(),'Missing tasks')
    for tid in mf['task_ids']:validate_scenario_role(taskmap[tid],mf['scenario_ids'])
    cfg=read_json(Path(a.model_source)/'run_config.json')
    return (mf,snapshots,taskmap,road,OD,Objective,legal,v3,cfg)

def organize(snapshots,max_cases,per_case,max_k):
    by=defaultdict(list)
    for s in snapshots:by[(s['task_id'],s['scenario_id'],int(s['seed']))].append(s)
    keys=sorted(by)
    if max_cases:keys=keys[:max_cases]
    selected={}
    for key in keys:
        arr=select_phase_snapshots(by[key],per_case)
        for s in arr:
            validate_checkpoint(s,20000)
            require(int(s['stop_count'])+max_k-1<=20000,'Selected snapshot exceeds max K')
        selected[key]=arr
    return selected

def specs(a, mf, horizons,selected):
    import json
    p=Path(a.capture_dir)
    return {'stage':STAGE,'data_role':'TRAIN_ONLY','status':'CONFIGURED',
            'capture_sha256':{f:sha256(p/f) for f in ('manifest.json','metrics.jsonl','snapshots.jsonl','provenance.json')},
            'solver_sha256':mf['solver_sha256'],'section_sha256':mf['section_sha256'],
            'observer_sha256':mf['core_sha256'],'source_sha256':mf['source_sha256'],
            'horizons':horizons,'max_cases':a.max_cases,'max_snapshots_per_run':a.max_snapshots_per_run,
            'selected':[[t,s,int(seed),[int(x['stop_count']) for x in arr]] for (t,s,seed),arr in selected.items()],
            'code_sha256':{f:sha256(Path(__file__).parent/f) for f in ('verify_train_r4g.py','r4g_core.py')},
            'limits':['Only frozen C3 TRAIN role','Fixed-iteration K, NOT 30s remaining time',
                      'Wall cap can prevent original full-iteration reference; fail closed',
                      'No intervened Section or model label generated']}

def run(a):
    mf,snapshots,taskmap,road,OD,Objective,legal,v3,cfg=preflight(a)
    horizons=sorted({int(v) for v in a.horizons.split(',') if v.strip()})
    require(horizons and min(horizons)>0 and max(horizons)<=500,'Expected fixed K 1..500')
    require(a.max_cases>=0 and a.max_snapshots_per_run>0,'Bad caps')
    selected=organize(snapshots,a.max_cases,a.max_snapshots_per_run,max(horizons))
    require(bool(selected),'No snapshots to verify')
    out=Path(a.output_dir);config=specs(a,mf,horizons,selected)
    if out.exists():
        require(a.resume,'Independent new result directory required')
        require(read_json(out/'manifest.json')==config,'Resume provenance/config mismatch')
    else:
        require(not a.resume,'Cannot resume nonexistent output')
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/'manifest.json',config)
    (out/'units').mkdir(exist_ok=True)
    for (tid,sid,seed),ss in selected.items():
        ident=json.dumps([tid,sid,seed],separators=(',',':'))
        unit=out/'units'/(hashlib.sha256(ident.encode()).hexdigest()[:20]+'.json')
        if unit.exists():
            old=read_json(unit)
            require(old.get('case')==[tid,sid,seed] and old.get('snapshot_counts')==[s['stop_count'] for s in ss],
                    'Existing proof unit identity mismatch')
            require(len(old.get('rows',[]))==len(ss)*len(horizons) and all(z.get('exact') is True for z in old['rows']),
                    'Existing proof incomplete')
            print('R4G REUSE',tid,sid,seed,flush=True)
            continue
        task=taskmap[tid];sc=next(x for x in task['scenarios'] if x['scenario_id']==sid)
        validate_scenario_role(task,[sid])
        od=OD(road,task,sc,step_min=int(cfg['od_step_min']),hours=float(cfg['od_hours']),cache_dir=a.od_cache_dir)
        obj=Objective(task,od,'static')
        windows=[(s['stop_count'],max(horizons)) for s in ss]
        with reference_recording(windows) as observed:
            ans=v3.solve('C3_topk',task,obj,seed,mf['search_wall_sec'],mf['max_iterations'],
                        top_k=mf['top_k'],anneal_iterations=mf['anneal_iterations'])
        require(ans['iterations']==mf['max_iterations'],'Reference hit wall limit; cannot prove K')
        require(len(observed)==1,'One official reference ALNS expected')
        events=observed[0].events
        require(all(i in events for low,k in windows for i in range(low,low+k)),'Missing reference trace event')
        rows=[]
        for s in ss:
            start=int(s['stop_count'])
            check_start(s,events[start])
            for k in horizons:
                holder=[]
                with exact_resume(s,k,holder):
                    v3.solve('C3_topk',task,obj,seed,mf['search_wall_sec'],mf['max_iterations'],
                             top_k=mf['top_k'],anneal_iterations=mf['anneal_iterations'])
                require(len(holder)==1,'Missing resumed trace')
                verify_events([events[i] for i in range(start,start+k)],holder[0])
                rows.append({'task_id':tid,'scenario_id':sid,'seed':seed,'snapshot_stop_count':start,
                             'horizon':k,'verified_steps':k,'exact':True,
                             'starting_RNG_sha256':s['rng_state_sha256']})
        write_json(unit,{'case':[tid,sid,seed],'snapshot_counts':[s['stop_count'] for s in ss],
                         'rows':rows,'source_snapshot_sha256':config['capture_sha256']['snapshots.jsonl']})
        print('R4G VERIFIED',tid,sid,seed,'windows',len(ss),'K',horizons,flush=True)
    allrows=[]
    for u in sorted((out/'units').glob('*.json')):
        z=read_json(u)
        require(z['source_snapshot_sha256']==config['capture_sha256']['snapshots.jsonl'],'Source changed')
        allrows+=z['rows']
    expected=sum(len(x) for x in selected.values())*len(horizons)
    require(len(allrows)==expected,'Result incomplete')
    with (out/'paired_rows.jsonl').open('w',encoding='utf8') as f:
        for r in allrows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    per={str(k):{'n':sum(z['horizon']==k for z in allrows),'exact':sum(z['horizon']==k and z['exact'] for z in allrows)} for k in horizons}
    report={'status':'COMPLETE','stage':STAGE,'data_role':'TRAIN_ONLY',
            'n_cases':len(selected),'n_snapshots':sum(len(v) for v in selected.values()),
            'horizons':horizons,'comparisons':len(allrows),'exact_matches':sum(z['exact'] for z in allrows),
            'per_horizon':per,'source_capture_sha256':config['capture_sha256']['snapshots.jsonl'],
            'limits':['TRAIN only','Fixed K, not remaining wall-clock time',
                      'No CF labels; do not train until Train-only counterfactual audit']}
    write_json(out/'report.json',report)
    (out/'report.md').write_text('# R4-G Train状态精确续接\n\n'+
        f"固定K逐步完全一致：**{report['exact_matches']}/{report['comparisons']}**，物理场景轨迹：{len(selected)}。\n\n"+
        '| K | Exact/Total |\n|---:|---:|\n'+''.join(f'| {k} | {per[str(k)]["exact"]}/{per[str(k)]["n"]} |\n' for k in horizons)+
        '\n仅已采集TRAIN状态；不代表30秒剩余墙钟等价。\n',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--capture-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--model-source',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--horizons',default='1,50,500');p.add_argument('--max-cases',type=int,default=0)
    p.add_argument('--max-snapshots-per-run',type=int,default=3)
    p.add_argument('--resume',action='store_true');run(p.parse_args())
