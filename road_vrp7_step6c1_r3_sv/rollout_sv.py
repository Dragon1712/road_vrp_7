#!/usr/bin/env python3
"""Paired H-step FROZEN-C3 RESTARTS from captured routes after first Section action.

IMPORTANT: original ALNS RouletteWheel scores/SA phase/RNG state were NOT
captured in R2. Every arm is restarted with identical INITIAL RNG seed and
same frozen operators. This is a RESTART SURROGATE, NOT exact continuation of
an original 30-second ALNS trajectory. No oracle TD input into ALNS search.
"""
from __future__ import annotations
import argparse,contextlib,copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r3_sv','road_vrp7_step6c1_r2','road_vrp7_step6b1_v2',
               'road_vrp7_step6b2','road_vrp7_step6a','road_vrp7_step5b_v3','road_vrp7_stage3'):
    if str(ROOT/folder) not in sys.path:sys.path.insert(0,str(ROOT/folder))
from sv_core import sha256,digest,atomic_json,paired_summary,ARMS
from road_vrp import Road,OD,Objective,legal,read_json
from section_v2 import Selector
from section_dataset import select_actions,repair_for
from r2_core import state_key,matches
import run_experiment as v3


def proposed_actions(captured):
    """Reconstruct exactly the same 8 candidates used by offline R2 labeling."""
    indexed={row['decision_id']:row for row in captured}
    if len(indexed)!=len(captured):raise ValueError('Duplicate capture decisions')
    return indexed


def reconstruct(task,sc,record,static,sampler_seed):
    routes=record['routes']; legal(task,routes)
    if state_key(record['task_id'],record['scenario_id'],routes)!=record['state_id']:
        raise ValueError('Capture state key mismatch')
    selector=Selector(static)
    cs=selector.candidates(routes,traffic=False)
    baseline=next((c for c in cs if matches(c,record['c3_action'])),None)
    if baseline is None:raise ValueError('C3 action missing from frozen enumeration')
    picks=select_actions(cs,8,record['state_id'],sampler_seed)
    if len(picks)!=8:raise ValueError('Too few sampled actions')
    return routes,baseline,[c for _,c in picks]


@contextlib.contextmanager
def fixed_greedy_start(routes):
    """Intercept ONLY the greedy starting solution within frozen v3.solve."""
    orig=v3.greedy
    def replace(task,obj,seed):
        legal(task,routes)
        return [list(r) for r in routes],None
    v3.greedy=replace
    try:yield
    finally:v3.greedy=orig


def search_restart(task,static,starting_routes,seed,horizon,wall,anneal=5000,top_k=12):
    if horizon<=0 or wall<=0:raise ValueError('Need positive H and wall cap')
    with fixed_greedy_start(starting_routes):
        result=v3.solve('C3_topk',task,static,seed,wall,horizon,
                        top_k=top_k,anneal_iterations=anneal)
    if int(result['iterations'])!=horizon:
        raise RuntimeError(f'Short rollout stopped at {result["iterations"]}/{horizon} iterations '
                           f'({result.get("termination_reason")}); increase --wall-per-arm')
    legal(task,result['routes'])
    return result


def strict_inputs(a):
    one=Path(a.one_step_dir);p=json.loads((one/'manifest.json').read_text())
    if p['stage']!='R3-SV-one-step-v1':raise ValueError('Invalid step1 stage')
    label=Path(a.dataset_dir);lm=read_json(label/'manifest.json')
    if lm['stage']!='step6c1r2-real-C3-labelled-v1' or lm['pair_sha256']!=sha256(label/'paired.npz'):
        raise ValueError('Label checksum mismatch')
    if p['labels_sha256']!=lm['pair_sha256'] or p['groups_sha256']!=sha256(label/'groups.jsonl'):
        raise ValueError('One-step experiment does not match labels')
    cap=Path(lm['source_dir']); cm=read_json(cap/'manifest.json')
    if cm['stage']!='step6c1r2-real-C3-capture-v1' or cm['road_sha256']!=lm['road_sha256'] or cm['edge_sha256']!=lm['edge_sha256']:
        raise ValueError('Capture/labels mismatch')
    if sha256(cap/'states.jsonl')!=lm['capture_state_sha256']:
        raise ValueError('Capture state contents changed')
    if cm['section_sha256']!=sha256(ROOT/'road_vrp7_step5b_v3/section_v2.py'):
        raise ValueError('Frozen Section source changed')
    if sha256(ROOT/'road_vrp7_step5b_v3/run_experiment.py')!=cm['solver_sha256']:
        raise ValueError('Frozen solver source changed')
    selection=read_json(one/'pilot_states.json')['selected_decision_ids']
    if not selection:raise ValueError('No selected pilot states')
    decision_rows=[json.loads(s) for s in (one/'decisions.jsonl').read_text().splitlines() if s.strip()]
    keyed={}
    for x in decision_rows:
        k=(x['decision_id'],x['arm'])
        if k in keyed:raise ValueError('Duplicated decision entry')
        keyed[k]=x
    for decision in selection:
        for arm in ARMS:
            if (decision,arm) not in keyed:raise ValueError('Missing choice '+arm+' '+decision)
    captured=[json.loads(s) for s in (cap/'states.jsonl').read_text().splitlines() if s.strip()]
    ix=proposed_actions(captured)
    if not set(selection)<=set(ix):raise ValueError('Pilot decisions do not exist in captured routes')
    return lm,cap,cm,selection,keyed,ix


def unit_repair(routes, baseline, options, static, td, indices, top_k, deadline,enable_swap,
                expected_step):
    result={};cache={}
    for arm in ARMS:
        idx=int(indices[arm]);cand=baseline if idx==-1 else options[idx]
        if idx!=-1 and matches(cand,baseline):raise ValueError('Non-C3 model action equals baseline')
        ckey=(int(cand['route']),int(cand['pos']),int(cand['size']))
        if ckey not in cache:
            repaired,reason,elapsed,ok,_=repair_for(static,routes,cand,top_k,deadline,enable_swap)
            if not ok:raise RuntimeError(f'Static Repair failed at {ckey}: {reason} / {elapsed:.3f}s; '
                                        'try increasing --wall-per-arm or check server load')
            legal(static.task,repaired)
            cache[ckey]=(repaired,float(td.cost(repaired)),float(static.cost(repaired)))
        after,cost_td,cost_st=cache[ckey]
        result[arm]={'initial_routes':copy.deepcopy(after),'td_initial':cost_td,
                     'static_initial':cost_st,'action_identity':list(ckey)}
    b=result['C3']['td_initial']
    for arm in ARMS:
        observed=b-result[arm]['td_initial']
        if abs(observed-expected_step[arm])>a_tol():
            raise ValueError(f'Step label mismatch ({arm}), expected={expected_step[arm]}, '
                f'reconstructed={observed:.6f}. Check action sampler/repair/top-k/deadline. '
                'Do not run multi-step with mismatched states.')
    return result


def a_tol():return .12 # float32 gain labels may have 0.01s rounding, allow 0.12 sec


def stable_seed(decision,h,rep):
    # Important: identical CRN seed for H=5/20/50 in a given state/replicate.
    # Horizon alone must NOT change the initial random stream.
    return 1+(int(digest(['r3svrestart',decision,int(rep)])[:12],16)%(2**31-2))


def make_rows(unit,state,fix,task,sc,static,td,road,h,rep,seed,wall,anneal,top_k,road_replay):
    out=[]
    for arm in ARMS:
        initial=fix[arm]['initial_routes']
        result=search_restart(task,static,initial,seed,h,wall,anneal,top_k)
        rec={'decision_id':unit,'task_id':state['task_id'],'scenario_id':state['scenario_id'],
             'capture_seed':int(state['seed']),'horizon':int(h),'replicate':int(rep),
             'rollout_seed':int(seed),'arm':arm,
             'initial_td_od_cost':fix[arm]['td_initial'],
             'initial_static_cost':fix[arm]['static_initial'],
             'final_td_od_cost':float(td.cost(result['routes'])),
             'final_static_cost':float(static.cost(result['routes'])),
             'iterations':int(result['iterations']),'search_wall_sec':float(result['search_wall_sec']),
             'action_identity':fix[arm]['action_identity'],
             'final_routes':result['routes']}
        if road_replay:
            rec['final_ex_post_road_travel_sec']=float(road.replay(task,result['routes'],sc['departure_local'])['ex_post_road_travel_sec'])
        out.append(rec)
    return out


def run(a):
    lm,cap,cm,selected,choices,ix=strict_inputs(a)
    if a.max_states:selected=selected[:a.max_states]
    horizons=sorted(set(int(x) for x in a.horizons.split(',')))
    if not horizons or any(h<=0 for h in horizons) or a.replicates<1 or a.wall_per_arm<1:
        raise ValueError('Bad horizons/replicates/wall settings')
    v3.check_alns()
    road=Road(a.cost_dir)
    if road.cost_file_hash!=lm['road_sha256'] or road.edge_file_hash!=lm['edge_sha256']:
        raise ValueError('ROAD tensor differs from labeled data')
    tm=read_json(Path(a.task_dir)/'manifest.json')
    if tm['cost_edge_index_sha256']!=road.edge_file_hash or tm['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Task manifest/cost mismatch')
    c={'stage':'R3-SV-paired-restart-v1','one_step_sha256':sha256(Path(a.one_step_dir)/'decisions.jsonl'),
       'pilot_sha256':sha256(Path(a.one_step_dir)/'pilot_states.json'),
       'label_sha256':lm['pair_sha256'],'capture_sha256':sha256(cap/'states.jsonl'),
       'frozen_solver_sha256':sha256(ROOT/'road_vrp7_step5b_v3/run_experiment.py'),
       'frozen_section_sha256':sha256(ROOT/'road_vrp7_step5b_v3/section_v2.py'),
       'code_sha256':{n:sha256(Path(__file__).with_name(n)) for n in ('sv_core.py','rollout_sv.py')},
       'road_sha256':road.cost_file_hash,'edge_sha256':road.edge_file_hash,
       'task_manifest_sha256':sha256(Path(a.task_dir)/'manifest.json'),
       'states':selected,'horizons':horizons,'replicates':a.replicates,
       'top_k':cm['top_k'],'repair_deadline':float(a.repair_deadline or lm['deadline_sec']),
       'enable_swap':bool(lm['enable_swap']),'od_step_min':int(a.od_step_min),'od_hours':float(a.od_hours),
       'sampler_seed':int(lm['sampler_seed']),'wall_per_arm':a.wall_per_arm,
       'anneal_iterations':5000,'road_replay':bool(a.road_replay),
       'meaning':'paired FROZEN C3 RESTART from repaired route, not exact original-trajectory continuation',
       'caveat':'ALNS selector weights, annealing phase and RNG internal state are not captured in R2; '
                'road costs are retrospective only, never sent to search'}
    out=Path(a.output_dir)
    if out.exists():
        if not a.resume:raise FileExistsError('New independent result directory required')
        if not (out/'manifest.json').is_file() or read_json(out/'manifest.json')!=c:
            raise ValueError('Resume refused: input/config/code SHA changed')
    else:
        out.mkdir(parents=True);atomic_json(out/'manifest.json',c)
    cache={};written=0;all_units=[]
    unit_dir=out/'units';unit_dir.mkdir(exist_ok=True)
    for di in selected:
        rec=ix[di];tid=rec['task_id'];scid=rec['scenario_id'];k=(tid,scid)
        if k not in cache:
            task=read_json(Path(a.task_dir)/(tid+'.json'))
            sc=next((s for s in task['scenarios'] if s['scenario_id']==scid),None)
            if sc is None:raise ValueError('Missing captured task/scenario')
            od=OD(road,task,sc,step_min=a.od_step_min,hours=a.od_hours,cache_dir=a.od_cache_dir)
            cache[k]=(task,sc,Objective(task,od,'static'),Objective(task,od,'oracle'))
        task,sc,static,td=cache[k]
        routes,base,opts=reconstruct(task,sc,rec,static,c['sampler_seed'])
        act={arm:choices[(di,arm)]['selected_index'] for arm in ARMS}
        expectation={arm:choices[(di,arm)]['gain_sec'] for arm in ARMS}
        # Always reconstruct, even when units are already present, to verify the
        # paired label alignment before accepting old checkpoints.
        fix=unit_repair(routes,base,opts,static,td,act,c['top_k'],c['repair_deadline'],
                        c['enable_swap'],expectation)
        for h in horizons:
            for rep in range(a.replicates):
                path=unit_dir/f'{di}__H{h}__R{rep}.json'
                if path.exists():
                    obj=read_json(path)
                    if obj.get('decision_id')!=di or obj.get('horizon')!=h or obj.get('replicate')!=rep or set(z['arm'] for z in obj['rows'])!=set(ARMS):
                        raise ValueError('Corrupt unit checkpoint '+str(path))
                    all_units.extend(obj['rows']);continue
                seed=stable_seed(di,h,rep)
                rows=make_rows(di,rec,fix,task,sc,static,td,road,h,rep,seed,
                               a.wall_per_arm,c['anneal_iterations'],c['top_k'],a.road_replay)
                atomic_json(path,{'decision_id':di,'horizon':h,'replicate':rep,'rows':rows})
                all_units.extend(rows);written+=1
                print('R3-SV',di,'H',h,'rep',rep,'completed',flush=True)
                if a.max_new_units and written>=a.max_new_units:
                    print('PARTIAL, resume with RESUME=1; new units:',written,flush=True)
                    return
    # Final merged file and report only on COMPLETE.
    with (out/'rollouts.jsonl.tmp').open('w') as f:
        for row in all_units:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    (out/'rollouts.jsonl.tmp').replace(out/'rollouts.jsonl')
    report={'status':'COMPLETE','n_states':len(selected),'replicates':a.replicates,
            'horizons':horizons,'n_arm_runs':len(all_units),'restart_not_exact_continuation':True,
            'td_od_pairs':paired_summary(all_units,'final_td_od_cost')}
    if a.road_replay:report['road_replay_pairs']=paired_summary(all_units,'final_ex_post_road_travel_sec')
    # Paired one-step sign vs H-step restart sign for every non-C3 model.
    disagreements={}
    for h in horizons:
        z=[r for r in all_units if r['horizon']==h]
        base_cost={(r['decision_id'],r['replicate']):r['final_td_od_cost'] for r in z if r['arm']=='C3'}
        for arm in ARMS[1:]:
            cases=[]
            for r in z:
                if r['arm']!=arm:continue
                step=float(choices[(r['decision_id'],arm)]['gain_sec'])
                multi=base_cost[(r['decision_id'],r['replicate'])]-float(r['final_td_od_cost'])
                if abs(step)>1e-6 and abs(multi)>1e-6:cases.append((step>0)!=(multi>0))
            disagreements.setdefault(str(h),{})[arm]={'n_nonzero_pairs':len(cases),
                          'sign_flip_fraction':float(np.mean(cases)) if cases else None}
    report['one_step_vs_h_sign_flip']=disagreements
    atomic_json(out/'report.json',report)
    print('R3-SV ROLLOUT COMPLETE',json.dumps(report['td_od_pairs']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--one-step-dir',required=True)
    p.add_argument('--dataset-dir',default='runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1')
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--horizons',default='5,20,50')
    p.add_argument('--replicates',type=int,default=2)
    p.add_argument('--max-states',type=int,default=0)
    p.add_argument('--max-new-units',type=int,default=0)
    p.add_argument('--repair-deadline',type=float,default=0.,help='0=exact label dataset deadline')
    p.add_argument('--wall-per-arm',type=float,default=30.)
    p.add_argument('--od-step-min',type=int,default=20)
    p.add_argument('--od-hours',type=float,default=5.)
    p.add_argument('--road-replay',action='store_true')
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
