#!/usr/bin/env python3
"""R4-E: audit original uninterrupted official ALNS vs restored K=100/200/500.

R4B K<=50 evidence is a strict prerequisite. Exact long-range verification
is required BEFORE any extended-horizon counterfactual collection.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r4e','road_vrp7_step6c1_r4c','road_vrp7_step6c1_r4b',
               'road_vrp7_step6c1_r4a','road_vrp7_step5b_v3','road_vrp7_step6b2_r',
               'road_vrp7_step6a','road_vrp7_stage3'):
    p=str(ROOT/folder)
    if p not in sys.path:sys.path.insert(0,p)
from r4e_core import (STAGE_VERIFY,require,sha,read,write,read_lines,horizons,
                      snapshot_id,verify_r4b_evidence)
import counterfactual_r4c as r4c
import verify_r4b as r4b
from r4b_core import check_start,verify_events,take_snapshots
from r4a_core import canonical_hash


def selected_snapshots(snapshots, max_cases, max_snapshots):
    return r4c._selected_snapshots(snapshots,max_cases,max_snapshots)


def run(a):
    ks=horizons(a.horizons)
    require(a.max_cases>=0 and a.max_snapshots_per_run>=1,'Invalid size')
    evidence=verify_r4b_evidence(a.r4b_dir,a.capture_dir)
    v3,OD,Objective,legal,taskmap,cfg,road,mf,snapshots,pins,verified=r4c.preflight(a)
    chosen=selected_snapshots(snapshots,a.max_cases,a.max_snapshots_per_run)
    for _,s in chosen:require(snapshot_id(s) in evidence,'R4B did not validate requested checkpoint')
    manifest={'stage':STAGE_VERIFY,'status':'RUNNING',
         'source_capture_sha256':sha(Path(a.capture_dir)/'snapshots.jsonl'),
         'source_r4b_report_sha256':sha(Path(a.r4b_dir)/'report.json'),
         'source_r4b_rows_sha256':sha(Path(a.r4b_dir)/'paired_rows.jsonl'),
         'solver_sha256':mf['solver_sha256'],'section_sha256':mf['section_sha256'],
         'code_sha256':{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('r4e_core.py'))},
         'cases':a.max_cases,'snapshots_per_case':a.max_snapshots_per_run,'horizons':list(ks),
         'source_data_role':'INSPECTED_JULY_VALIDATION_DEVELOPMENT_ONLY'}
    out=Path(a.output_dir)
    if out.exists():
        require(a.resume,'Independent new output directory required or explicit --resume')
        old=read(out/'manifest.json');require({k:v for k,v in old.items() if k!='status'}=={k:v for k,v in manifest.items() if k!='status'},'Resume proof manifest differs')
    else:
        require(not a.resume,'Cannot resume missing study')
        out.mkdir(parents=True,exist_ok=False);write(out/'manifest.json',manifest)
    (out/'units').mkdir(exist_ok=True)
    selected_groups=defaultdict(list)
    for key,s in chosen:selected_groups[key].append(s)
    total_rows=[]
    for index,((tid,sid,seed),selected) in enumerate(sorted(selected_groups.items()),1):
        unit=out/'units'/f'{tid}__{sid}__{seed}.json'
        if unit.exists():
            saved=read(unit);require(saved.get('snapshots')==[list(snapshot_id(s)) for s in selected],'Checkpoint identity mismatch')
            require(saved.get('horizons')==list(ks) and saved.get('source_capture_sha256')==manifest['source_capture_sha256'],'Checkpoint config mismatch')
            require(all(r['exact'] is True for r in saved['rows']),'Checkpoint contains failure')
            total_rows.extend(saved['rows']);print('R4E EXACT reused',index,tid,sid,seed,flush=True);continue
        task=taskmap[tid];sc=next(x for x in task['scenarios'] if x['scenario_id']==sid)
        od=OD(road,task,sc,step_min=int(cfg['od_step_min']),hours=float(cfg['od_hours']),cache_dir=a.od_cache_dir)
        obj=Objective(task,od,'static')
        windows=[(int(s['stop_count']),max(ks)) for s in selected]
        with r4b.reference_recording(windows) as observers:
            solution=v3.solve('C3_topk',task,obj,int(seed),mf['search_wall_sec'],mf['max_iterations'],
                              top_k=mf['top_k'],anneal_iterations=mf['anneal_iterations'])
        require(len(observers)==1,'Expected one reference ALNS.iterate')
        require(int(solution['iterations'])==int(mf['max_iterations']),'Original C3 wall-capped, long exact reference unavailable')
        events=observers[0].events
        require(all(t in events for start,k in windows for t in range(start,start+k)),'Reference event window incomplete')
        unit_rows=[]
        for s in selected:
            start=int(s['stop_count']);check_start(s,events[start]);trace=[]
            with r4b.exact_resume(s,max(ks),trace):
                v3.solve('C3_topk',task,obj,int(seed),mf['search_wall_sec'],mf['max_iterations'],
                         top_k=mf['top_k'],anneal_iterations=mf['anneal_iterations'])
            require(len(trace)==1 and len(trace[0])==max(ks),'Long continuation produced wrong length')
            for k in ks:
                ref=[events[t] for t in range(start,start+k)]
                verify_events(ref,trace[0][:k])
                unit_rows.append({'task_id':tid,'scenario_id':sid,'seed':int(seed),
                    'stop_count':start,'horizon':k,'verified_steps':k,'exact':True,
                    'rng_state_sha256':s['rng_state_sha256'],
                    'last_candidate_sha256':ref[-1]['candidate_route_sha256']})
        write(unit,{'snapshots':[snapshot_id(s) for s in selected],
                    'horizons':list(ks),'source_capture_sha256':manifest['source_capture_sha256'],'rows':unit_rows})
        total_rows.extend(unit_rows)
        print('R4E EXACT VERIFIED',index,'/',len(selected_groups),tid,sid,seed,'snapshots',len(selected),'maxK',max(ks),flush=True)
    expected=len(chosen)*len(ks)
    require(len(total_rows)==expected,'Long proof count incomplete')
    with (out/'paired_rows.jsonl').open('w',encoding='utf8') as f:
        import json
        for r in total_rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    report={'stage':STAGE_VERIFY,'status':'COMPLETE','cases':len(selected_groups),
        'snapshots':len(chosen),'horizons':list(ks),'n_comparisons':expected,'exact_matches':expected,
        'per_horizon':{str(k):{'n':len(chosen),'exact':len(chosen)} for k in ks},
        'limitations':['Only inspected July Validation; no independent test.',
           'Fixed-K continuation not remaining 30s wall time.',
           'Exact outcome may require avoiding deadline-dependent branches.',
           'No alternative Sections changed in this exact reference proof.']}
    write(out/'report.json',report)
    (out/'report.md').write_text('# R4-E 100/200/500步精确续接审计\n\n'+
        f"逐迭代完全一致：**{expected}/{expected}**；快照：{len(chosen)}。\n\n"+
        '| K | 通过/总数 |\n|---:|---:|\n'+''.join(f'| {k} | {len(chosen)}/{len(chosen)} |\n' for k in ks)+
        '\n仅固定K，无30秒剩余墙钟等价性。\n',encoding='utf8')
    manifest['status']='COMPLETE';write(out/'manifest.json',manifest)
    print('R4E LONG PROOF COMPLETE',out,expected,'/',expected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir',required=True);p.add_argument('--r4b-dir',required=True)
    p.add_argument('--model-source',required=True);p.add_argument('--cost-dir',required=True)
    p.add_argument('--task-dir',required=True);p.add_argument('--od-cache-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--horizons',default='100,200,500')
    p.add_argument('--max-cases',type=int,default=0)
    p.add_argument('--max-snapshots-per-run',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
