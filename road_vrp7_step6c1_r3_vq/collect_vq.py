#!/usr/bin/env python3
"""Collect TRAIN-ONLY C3 vs eight candidate H-step restart-value labels.

Frozen C3_topk RESTART, not an exact continuation of earlier ALNS. No future
traffic is read by predictor/search. Road.replay is retrospective label only.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r3_vq','road_vrp7_step6c1_r3_sv','road_vrp7_step6c1_r2',
               'road_vrp7_step6c1_r3_tricks','road_vrp7_step6b1_v2','road_vrp7_step6b2',
               'road_vrp7_step6a','road_vrp7_step5b_v3','road_vrp7_stage3'):
    d=str(ROOT/folder)
    if d not in sys.path:sys.path.insert(0,d)
from vq_core import STAGE,LABELS,sha256,atomic_json,uniform_train_selection,restart_seed,hashkey,row_labels
from r2_policy import group_split
from rollout_sv import reconstruct,search_restart
from road_vrp import Road,OD,Objective,legal,read_json
from section_dataset import repair_for
from r2_core import matches

DEFAULT_DATA='runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1'
DEFAULT_COST='datasets/processed/chicago/july_v1/dynamic_costs'
DEFAULT_TASK='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10'
DEFAULT_CACHE='runs/chicago_july_v1/stage3/od_cache'

def load(a):
    folder=Path(a.dataset_dir);m=read_json(folder/'manifest.json')
    if m.get('stage')!='step6c1r2-real-C3-labelled-v1' or sha256(folder/'paired.npz')!=m.get('pair_sha256') or sha256(folder/'groups.jsonl')!=m.get('groups_sha256'):
        raise ValueError('Wrong/mutated R2 real C3 paired label dataset')
    with np.load(folder/'paired.npz',allow_pickle=False) as npz:
        X,B,Y,BY,valid,same=[npz[k].copy() for k in ('X','B','Y','BY','valid','same')]
    groups=[json.loads(t) for t in (folder/'groups.jsonl').read_text().splitlines() if t.strip()]
    if X.shape!=(len(groups),8,44) or B.shape!=(len(groups),44) or Y.shape!=(len(groups),8) or valid.shape!=Y.shape or same.shape!=Y.shape:
        raise ValueError('Bad dataset shape')
    masks,roles=group_split(groups,'task')
    chosen=uniform_train_selection(groups,{k for k,v in roles.items() if v=='train'},a.states,a.seed)
    if not all(masks['train'][i] for i in chosen):raise AssertionError('Nontrain state selected')
    cap=Path(m['source_dir']);cm=read_json(cap/'manifest.json')
    if cm.get('stage')!='step6c1r2-real-C3-capture-v1' or sha256(cap/'states.jsonl')!=m['capture_state_sha256']:
        raise ValueError('Captured routes are not the matching R2 source')
    if cm['road_sha256']!=m['road_sha256'] or cm['edge_sha256']!=m['edge_sha256']:
        raise ValueError('Capture and labels road mismatch')
    if sha256(ROOT/'road_vrp7_step5b_v3/section_v2.py')!=cm['section_sha256'] or sha256(ROOT/'road_vrp7_step5b_v3/run_experiment.py')!=cm['solver_sha256']:
        raise ValueError('Frozen ALNS/Section implementation changed')
    records={}
    for line in (cap/'states.jsonl').open():
        if not line.strip():continue
        r=json.loads(line)
        if r['decision_id'] in records:raise ValueError('Duplicate captured decision')
        records[r['decision_id']]=r
    if any(groups[i]['decision_id'] not in records for i in chosen):raise ValueError('Missing captured route')
    return (m,cm,records,groups,chosen,Y,BY,valid,same)

def make_unit(di,task,sc,static,td,road,routes,actions,info,h,rep,seed,top_k,wall,road_replay):
    """Search each distinct post-repair route once per unit; identical routes placebo."""
    cache={};rows=[]
    for j in sorted(info):
        z=info[j];start=z['routes'];route_sig=hashkey(start)
        if route_sig not in cache:
            result=search_restart(task,static,start,seed,h,wall,5000,top_k)
            final=result['routes']
            val={'final_td_od':float(td.cost(final)),'final_static':float(static.cost(final)),
                 'iterations':int(result['iterations']),'search_wall_sec':float(result['search_wall_sec'])}
            if road_replay:
                val['final_road']=float(road.replay(task,final,sc['departure_local'])['ex_post_road_travel_sec'])
            cache[route_sig]=val
        row={'decision_id':di,'task_id':actions['task_id'],'scenario_id':actions['scenario_id'],
             'horizon':int(h),'replicate':int(rep),'rollout_seed':int(seed),'action_index':int(j),
             'one_step_gain_td_od':float(z['one_step_gain']),'action_id':z['action_id'],
             'same_repair_route_as_other_action':sum(hashkey(v['routes'])==route_sig for v in info.values())>1,
             **cache[route_sig]}
        rows.append(row)
    return rows

def run(a):
    if a.replicates<1 or a.states<1 or a.wall_per_arm<=0:raise ValueError('states/replicates/wall must be positive')
    horizons=sorted({int(h) for h in a.horizons.split(',')})
    if not horizons or any(h<1 for h in horizons):raise ValueError('H must be positive')
    m,cm,captured,groups,chosen,Y,BY,valid,same=load(a)
    from run_experiment import check_alns
    check_alns()
    road=Road(a.cost_dir)
    if road.cost_file_hash!=m['road_sha256'] or road.edge_file_hash!=m['edge_sha256']:
        raise ValueError('ROAD data differ from R2 labels')
    task_manifest=read_json(Path(a.task_dir)/'manifest.json')
    if task_manifest['cost_edge_index_sha256']!=road.edge_file_hash or task_manifest['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Road/task manifest mismatch')
    if not a.road_replay:raise ValueError('VQ requires ROAD_REPLAY=1 to learn real-road restart value')
    sig={'stage':STAGE,'selected_decision_ids':[groups[i]['decision_id'] for i in chosen],
         'selection':'outcome-blind round-robin train tasks by decision_id hash',
         'train_tasks':sorted({groups[i]['task_id'] for i in chosen}),
         'label_sha256':m['pair_sha256'],'group_sha256':m['groups_sha256'],
         'capture_sha256':sha256(Path(m['source_dir'])/'states.jsonl'),
         'frozen_solver_sha256':sha256(ROOT/'road_vrp7_step5b_v3/run_experiment.py'),
         'frozen_section_sha256':sha256(ROOT/'road_vrp7_step5b_v3/section_v2.py'),
         'sv_restart_sha256':sha256(ROOT/'road_vrp7_step6c1_r3_sv/rollout_sv.py'),
         'repair_source_sha256':sha256(ROOT/'road_vrp7_step6a/section_dataset.py'),
         'road_backend_sha256':sha256(ROOT/'road_vrp7_stage3/road_vrp.py'),
         'source_sha256':{n:sha256(Path(__file__).with_name(n)) for n in ('vq_core.py','collect_vq.py')},
         'road_sha256':road.cost_file_hash,'edge_sha256':road.edge_file_hash,
         'task_manifest_sha256':sha256(Path(a.task_dir)/'manifest.json'),
         'horizons':horizons,'replicates':a.replicates,'seed':a.seed,'sampler_seed':int(m['sampler_seed']),
         'top_k':int(cm['top_k']),'repair_deadline':float(m['deadline_sec']),
         'enable_swap':bool(m['enable_swap']),'wall_per_arm':a.wall_per_arm,
         'od_step_min':a.od_step_min,'od_hours':a.od_hours,'road_replay':True,
         'limits':['TRAIN task labels only; no internal_test/validation labels used for VQ fitting',
                   'Restart from repaired route, not exact original online trajectory',
                   'Road.replay posthoc target only; search always uses static planner']}
    out=Path(a.output_dir)
    if out.exists():
        if not a.resume:raise FileExistsError('Output exists; use NEW dir or RESUME=1')
        if not (out/'manifest.json').is_file() or read_json(out/'manifest.json')!=sig:
            raise ValueError('Cannot resume with changed data/code/config')
    else:out.mkdir(parents=True);atomic_json(out/'manifest.json',sig)
    units=out/'units';units.mkdir(exist_ok=True)
    task_cache={};all_rows=[];written=0
    for i in chosen:
        g=groups[i];di=g['decision_id'];s=captured[di];k=(s['task_id'],s['scenario_id'])
        if k not in task_cache:
            task=read_json(Path(a.task_dir)/(k[0]+'.json'))
            sc=next((x for x in task['scenarios'] if x['scenario_id']==k[1]),None)
            if sc is None or sc['departure_local']!=s['departure_local']:raise ValueError('Scenario mismatch')
            od=OD(road,task,sc,step_min=a.od_step_min,hours=a.od_hours,cache_dir=a.od_cache_dir)
            task_cache[k]=(task,sc,Objective(task,od,'static'),Objective(task,od,'oracle'))
        task,sc,static,td=task_cache[k]
        routes,c3,candidates=reconstruct(task,sc,s,static,sig['sampler_seed'])
        orig=float(td.cost(routes));info={};base_repair,reason,_,ok,_=repair_for(static,routes,c3,sig['top_k'],sig['repair_deadline'],sig['enable_swap'])
        if not ok:raise RuntimeError(f'C3 repair failed: {di}, {reason}')
        baseline_value=orig-float(td.cost(base_repair))
        if abs(baseline_value-float(BY[i]))>.12:raise ValueError(f'C3 one-step label mismatch: {di}')
        info[-1]={'routes':base_repair,'one_step_gain':0.,'action_id':[int(c3[z]) for z in ('route','pos','size')]}
        for j,c in enumerate(candidates):
            if not valid[i,j] or same[i,j]:continue
            repaired,reason,_,ok,_=repair_for(static,routes,c,sig['top_k'],sig['repair_deadline'],sig['enable_swap'])
            if not ok:raise RuntimeError(f'Candidate repair failed: {di}/action{j}: {reason}')
            gain=orig-float(td.cost(repaired))-baseline_value
            if abs(gain-float(Y[i,j]-BY[i]))>.12:raise ValueError(f'One-step label mismatch: {di}/action{j}')
            info[j]={'routes':repaired,'one_step_gain':gain,
                     'action_id':[int(c[z]) for z in ('route','pos','size')]}
        for h in horizons:
            for rep in range(a.replicates):
                unit=units/f'{di}__H{h}__R{rep}.json'
                if unit.exists():
                    obj=read_json(unit)
                    rows=obj.get('rows',[])
                    if obj.get('decision_id')!=di or obj.get('horizon')!=h or obj.get('replicate')!=rep or set(int(z['action_index']) for z in rows)!=set(info):
                        raise ValueError('Corrupt checkpoint: '+str(unit))
                    all_rows+=rows;continue
                seed=restart_seed(di,rep)
                rows=make_unit(di,task,sc,static,td,road,routes,s,info,h,rep,seed,sig['top_k'],a.wall_per_arm,True)
                atomic_json(unit,{'decision_id':di,'horizon':h,'replicate':rep,'rows':rows})
                all_rows+=rows;written+=1
                print(f'R3-VQ unit {di} H={h} rep={rep} complete',flush=True)
                if a.max_new_units and written>=a.max_new_units:
                    print('PARTIAL. Re-run with RESUME=1 and identical args.',flush=True);return
    final=out/'rollouts.jsonl'
    temp=final.with_suffix('.jsonl.tmp')
    with temp.open('w') as f:
        for row in all_rows:f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    temp.replace(final)
    labels=row_labels(all_rows,'road')
    counts=defaultdict(int)
    for k in labels:counts[k[1]]+=1
    atomic_json(out/'report.json',{'stage':LABELS,'status':'COMPLETE','n_selected_states':len(chosen),
        'horizons':horizons,'replicates':a.replicates,'n_restart_action_rows':len(all_rows),
        'n_state_horizon_action_labels':len(labels),'per_horizon_action_labels':dict(counts),
        'training_data_only':True,'restart_not_exact_continuation':True})
    print('R3-VQ COMPLETE',out,'rows',len(all_rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',default=DEFAULT_DATA);p.add_argument('--cost-dir',default=DEFAULT_COST)
    p.add_argument('--task-dir',default=DEFAULT_TASK);p.add_argument('--od-cache-dir',default=DEFAULT_CACHE)
    p.add_argument('--output-dir',required=True);p.add_argument('--states',type=int,default=48)
    p.add_argument('--seed',type=int,default=20261025);p.add_argument('--horizons',default='20,50')
    p.add_argument('--replicates',type=int,default=3);p.add_argument('--wall-per-arm',type=float,default=30.)
    p.add_argument('--od-step-min',type=int,default=20);p.add_argument('--od-hours',type=float,default=5.)
    p.add_argument('--road-replay',action='store_true');p.add_argument('--max-new-units',type=int,default=0)
    p.add_argument('--resume',action='store_true');run(p.parse_args())
