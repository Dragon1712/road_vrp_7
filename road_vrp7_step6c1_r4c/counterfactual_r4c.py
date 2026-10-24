#!/usr/bin/env python3
"""R4-C v1: controlled FIRST Section intervention, followed by real frozen C3 K-step continuation.

IMPORTANT: fixed K iteration counterfactual; NOT exact 30-second continuation.
Candidates derived only from static Section features; Road.replay is POST-HOC.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for f in ('road_vrp7_step6c1_r4c','road_vrp7_step6c1_r4b','road_vrp7_step6c1_r4a',
          'road_vrp7_step5b_v3','road_vrp7_step6b2_r','road_vrp7_step6a','road_vrp7_stage3'):
    s=str(ROOT/f)
    if s not in sys.path:sys.path.insert(0,s)
from r4c_core import (STAGE,require,sha256_file,read_json,write_json,canonical_json,digest,
    state_identifier,candidate_identity,partial_identity,make_partial,validate_sham,
    select_alternatives,branch_spec,unit_name,inspect_events,compare_sham,normalize_reward,summarize)
from r4b_core import (common_data_audit,take_snapshots,validate_checkpoint,
    restored_iterator,signature)


def preflight(a):
    """Use all R4B pinning plus require a COMPLETE 64/64 exact audit."""
    import verify_r4b
    v3,Road,OD,Objective,legal,taskmap,cfg,road,mf,snapshots=verify_r4b.preflight(a)
    r4b_dir=Path(a.r4b_dir)
    b=read_json(r4b_dir/'report.json');bm=read_json(r4b_dir/'manifest.json')
    require(b.get('stage')=='R4B-OFFICIAL-EXACT-CONTINUATION-AUDIT-v1' and b.get('status')=='COMPLETE',
            'R4B audit incomplete/wrong stage')
    require(b.get('n_comparisons')==b.get('exact_matches')==64 and b.get('snapshots')==16,
            'Require R4B 64/64 exact independent continuation validation')
    require(b.get('per_horizon') and all(b['per_horizon'][str(k)]['exact']==16 for k in (1,5,20,50)),
            'R4B did not verify all windows')
    require(b['source_snapshot_sha256']==sha256_file(Path(a.capture_dir)/'snapshots.jsonl'),
            'R4B results refer to a different R4A capture')
    require(bm.get('source_capture_snapshot_sha256')==b['source_snapshot_sha256'],
            'R4B manifest disagrees with its report')
    require(bm.get('source_solver_sha256')==mf['solver_sha256'] and
            bm.get('source_section_sha256')==mf['section_sha256'],
            'R4B frozen solver/section SHA differ')
    require(len(snapshots)==42,'Expected 42 verified R4A source snapshots')
    # R4-B verified only TWO checkpoints per run, not all 42 captured states.
    # Fail closed if an unaudited checkpoint is requested for a counterfactual.
    verified_rows=[json.loads(line) for line in (r4b_dir/'paired_rows.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    require(len(verified_rows)==64 and all(x.get('exact') is True for x in verified_rows),
            'Incomplete R4B checkpoint-by-horizon evidence')
    verified=defaultdict(set)
    for x in verified_rows:
        key=(str(x['task_id']),str(x['scenario_id']),int(x['seed']),int(x['snapshot_stop_count']))
        verified[key].add(int(x['horizon']))
    require(len(verified)==16 and all(v=={1,5,20,50} for v in verified.values()),
            'R4B did not check K=1,5,20,50 for exactly 16 snapshots')
    pins={'r4b_report_sha256':sha256_file(r4b_dir/'report.json'),
          'r4b_manifest_sha256':sha256_file(r4b_dir/'manifest.json'),
          'r4b_paired_rows_sha256':sha256_file(r4b_dir/'paired_rows.jsonl'),
          'r4a_snapshots_sha256':sha256_file(Path(a.capture_dir)/'snapshots.jsonl')}
    return (v3,OD,Objective,legal,taskmap,cfg,road,mf,snapshots,pins,verified)


def shadow_action(s,obj,action_count,sampler_seed,max_alternatives):
    """Recreate C3 FIRST operator action on CLONED RNG, never mutating checkpoint."""
    import numpy as np
    from alns.select import RouletteWheel
    from section_v2 import Selector
    from section_dataset import select_actions
    r=np.random.default_rng()
    r.bit_generator.state=copy.deepcopy(s['rng_bit_generator_state'])
    selector=RouletteWheel(scores=list(s['roulette_scores']),decay=s['roulette_decay'],num_destroy=4,num_repair=2)
    selector.destroy_weights[:]=s['roulette_destroy_weights']
    selector.repair_weights[:]=s['roulette_repair_weights']
    pair=selector(r,None,None)
    require(tuple(map(int,pair))==(3,int(s['real_selected_repair_index'])),
            'Shadow selection not the actual R4A Section choice')
    sec=Selector(obj)
    p=sec.destroy('static',s['current_routes'],r)
    after_rng=copy.deepcopy(r.bit_generator.state)
    c3=partial_identity(p)
    cs=sec.candidates(s['current_routes'],traffic=False)
    require(sum(candidate_identity(c)==c3 for c in cs)==1,'C3 block not unique in candidate enumeration')
    sid=state_identifier(s['task_id'],s['scenario_id'],s['current_routes'])
    sampled=select_actions(cs,int(action_count),sid,int(sampler_seed))
    alts=select_alternatives(cs,sampled,c3,int(max_alternatives))
    require(bool(alts),'No valid non-C3 alternatives sampled')
    return {'c3_identity':list(c3),'alternatives':alts,'candidate_count':len(cs),
            'first_rng_after_original_C3_destroy_sha256':_hash_rng(after_rng),
            'sampling_state_identifier':sid,'sampling_seed':int(sampler_seed)}


def _hash_rng(x):
    from r4a_core import canonical_hash
    return canonical_hash(x)


@contextmanager
def first_section_substitution(arm,planned,real_c3,expected_rng_sha256=None):
    """Preserve original C3 destroy RNG cost; change ONLY resulting first Partial.

    For C3, caller does not patch. For SHAM, recomputes an exactly identical
    Section Partial with unchanged RNG. Later Section calls are always frozen.
    """
    from section_v2 import Selector
    old=Selector.destroy
    audit={'substitution_calls':0,'first_actual_section':None,'first_chosen_section':None,
           'sham_partial_equal':False,'original_destroy_draws_preserved':False}
    def intercept(self,mode,routes,rng):
        if audit['substitution_calls']>0:
            return old(self,mode,routes,rng)
        require(mode=='static','R4C requires frozen STATIC C3')
        # Run the *REAL* C3 draw first, including jitter, top-fraction choice,
        # internal stats increments, and identical live RNG transitions.
        real=old(self,mode,routes,rng)
        from r4a_core import canonical_hash
        after_destroy_rng_sha256=canonical_hash(copy.deepcopy(rng.bit_generator.state))
        if expected_rng_sha256 is not None:
            require(after_destroy_rng_sha256==expected_rng_sha256,
                    'Post-original-destroy RNG SHA not paired to frozen C3 shadow')
        audit['post_original_C3_destroy_rng_sha256']=after_destroy_rng_sha256
        chosen_c3=partial_identity(real)
        require(chosen_c3==tuple(real_c3),'Runtime actual C3 draw differs from prior shadow draw')
        audit['substitution_calls']+=1
        audit['first_actual_section']=list(chosen_c3)
        audit['original_destroy_draws_preserved']=True
        if arm=='SHAM_C3':
            cand=dict(zip(('route','pos','size'),chosen_c3))
            fake=make_partial(routes,cand,self.obj)
            validate_sham(real,fake)
            audit['sham_partial_equal']=True
        else:
            require(arm==planned['name'],'Intervention plan mismatched arm')
            fake=make_partial(routes,planned,self.obj)
        audit['first_chosen_section']=list(partial_identity(fake))
        return fake
    Selector.destroy=intercept
    try:yield audit
    finally:Selector.destroy=old


def run_one(v3,task,sc,seed,obj,snapshot,k,arm,choice,c3_identity,road,legal,mf,expected_rng_sha256):
    """Build true official C3 operator closures, restoring at ALNS.iterate boundary."""
    from alns import ALNS
    from r4a_core import inspect_anneal,canonical_hash,roads
    saved=ALNS.iterate
    seen=[]
    def intercept(engine,initial,op_select,accept,stop,**kwargs):
        last={}
        orig_eval=engine._eval_cand
        def tracking(accept,best,curr,cand,**kw):
            b,c,outcome=orig_eval(accept,best,curr,cand,**kw)
            last['current']=c;last['best']=b
            return b,c,outcome
        engine._eval_cand=tracking
        try:
            if arm=='C3':
                result,events=restored_iterator(engine,initial,op_select,accept,stop,snapshot,k,**kwargs)
                audit={'substitution_calls':0,'first_actual_section':list(c3_identity),
                       'first_chosen_section':list(c3_identity),'original_destroy_draws_preserved':True}
            else:
                with first_section_substitution(arm,choice,c3_identity,expected_rng_sha256) as audit:
                    result,events=restored_iterator(engine,initial,op_select,accept,stop,snapshot,k,**kwargs)
            inspect_events(events,snapshot,k)
            require('current' in last and 'best' in last,'No ALNS transition performed')
            end=signature(rng=engine._rng,selector=op_select,anneal=inspect_anneal(accept),
                          best=last['best'],curr=last['current'],count=int(snapshot['stop_count'])+k)
            seen.append({'events':events,'end_signature':end,'audit':audit,
                         'best_routes':roads(last['best']),'current_routes':roads(last['current'])})
            return result
        finally:
            engine._eval_cand=orig_eval
    ALNS.iterate=intercept
    try:
        ans=v3.solve('C3_topk',task,obj,int(seed),mf['search_wall_sec'],mf['max_iterations'],
                     top_k=mf['top_k'],anneal_iterations=mf['anneal_iterations'])
    finally:
        ALNS.iterate=saved
    require(len(seen)==1,'Expected one ALNS iterator interception')
    result=seen[0];audit=result['audit']
    require(int(audit['substitution_calls'])==(0 if arm=='C3' else 1),'Wrong intervention count')
    require(audit['original_destroy_draws_preserved'],'C3 RNG budget was not consumed')
    from section_v2 import frozen
    require(frozen(result['best_routes'])==frozen(ans['routes']),'v3 best route and tracker disagree')
    legal(task,result['best_routes']);legal(task,result['current_routes'])
    best_road=road.replay(task,result['best_routes'],sc['departure_local'])
    curr_road=road.replay(task,result['current_routes'],sc['departure_local'])
    sp={'task_id':str(snapshot['task_id']),'scenario_id':str(snapshot['scenario_id']),
        'seed':int(seed),'stop_count':int(snapshot['stop_count'])}
    return {'snapshot_spec':sp,'horizon':int(k),'arm':arm,'first_action_id':audit['first_chosen_section'],
            'first_original_c3_action_id':audit['first_actual_section'],
            'events':result['events'],'end_signature':result['end_signature'],
            'best_route_sha256':result['end_signature']['best_route_sha256'],
            'current_route_sha256':result['end_signature']['current_route_sha256'],
            'static_best_sec':result['end_signature']['best_cost'],
            'static_current_sec':result['end_signature']['current_cost'],
            'road_best_sec':float(best_road['ex_post_road_travel_sec']),
            'road_current_sec':float(curr_road['ex_post_road_travel_sec']),
            'first_section_audit':audit,'first_frozen_operator_pair':[3,int(snapshot['real_selected_repair_index'])]}


def _selected_snapshots(snapshots,max_cases,max_snaps):
    by=defaultdict(list)
    for x in snapshots:by[(x['task_id'],x['scenario_id'],int(x['seed']))].append(x)
    keys=sorted(by)
    if max_cases:keys=keys[:max_cases]
    require(bool(keys),'No snapshots selected')
    return [(key,s) for key in keys for s in take_snapshots(snapshots,task_id=key[0],
        scenario_id=key[1],seed=key[2],max_states=max_snaps)]


def main(a):
    require(0<=a.max_cases and a.max_snapshots_per_run>=1 and a.action_count>=2 and
            1<=a.max_alternatives<=a.action_count and a.max_new_units>=0,'Bad experiment caps')
    horizons=sorted(set(int(x) for x in a.horizons.split(',')))
    require(horizons and min(horizons)>0 and max(horizons)<=50,'Only K=1..50 validated in R4-B')
    v3,OD,Objective,legal,taskmap,cfg,road,mf,snapshots,pins,verified=preflight(a)
    print('R4C PREFLIGHT OK: R4B exact 64/64, SHA and original C3 static sources pinned',flush=True)
    if a.preflight_only:return
    chosen=_selected_snapshots(snapshots,a.max_cases,a.max_snapshots_per_run)
    for (_,_,_),s in chosen:
        ident=(str(s['task_id']),str(s['scenario_id']),int(s['seed']),int(s['stop_count']))
        require(ident in verified, 'R4C refuses checkpoint not covered by exact R4B 64/64 audit: '+str(ident))
    out=Path(a.output_dir)
    config={'stage':STAGE,'source':'R4A actual Section preselection',
        'frozen_solver_sha256':mf['solver_sha256'],'frozen_section_sha256':mf['section_sha256'],
        'source_model_sha256':mf['source_sha256'],**pins,
        'horizons':horizons,'max_cases':a.max_cases,'max_snapshots_per_run':a.max_snapshots_per_run,
        'action_count':a.action_count,'max_alternatives':a.max_alternatives,
        'sampler_seed':a.sampler_seed,'snapshot_ids':[branch_spec(s,0,'S0') for _,s in chosen],
        'code_sha256':{p.name:sha256_file(p) for p in [Path(__file__),Path(__file__).with_name('r4c_core.py')]},
        'evaluation':'FIXED_ITERATION_K_NOT_WALL_CLOCK_30SEC',
        'data_role':'INSPECTED_JULY_VALIDATION_DEVELOPMENT_ONLY',
        'warnings':['Intervention is an alternative FIRST Section partial, then original C3 policy runs K-1 steps.',
                    'Every arm consumes identical original Section RNG draw before substitution.',
                    'Road.replay is posthoc label, never online feature.',
                    'No inference about identical RNG draw streams after divergent interventions.',
                    'K-step counterfactual is NOT an exact 30-second remaining wall-clock effect.']}
    if out.exists():
        require(a.resume,'Existing output requires explicit --resume')
        require(read_json(out/'manifest.json')==config,'RESUME config/source SHA mismatch')
    else:
        require(not a.resume,'Cannot resume a new study')
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/'manifest.json',config)
    (out/'units').mkdir(exist_ok=True)
    completed=0;new_units=0;all_results=[];all_rewards=[];all_counts=[]
    prefix_by_arm={}
    for (tid,sid,seed),snap in chosen:
        task=taskmap[tid];sc=next(z for z in task['scenarios'] if z['scenario_id']==sid)
        od=OD(road,task,sc,step_min=int(cfg['od_step_min']),hours=float(cfg['od_hours']),cache_dir=a.od_cache_dir)
        obj=Objective(task,od,'static')
        plan=shadow_action(snap,obj,a.action_count,a.sampler_seed,a.max_alternatives)
        alts=plan['alternatives'];c3=plan['c3_identity']
        all_counts.append({'task_id':tid,'scenario_id':sid,'seed':int(seed),'stop_count':int(snap['stop_count']),
                           'alternatives':len(alts),'candidate_count':plan['candidate_count'],
                           'c3_identity':c3})
        arms=[('C3',None),('SHAM_C3',None)]+[(x['name'],x) for x in alts]
        for k in horizons:
            runs={}
            for arm,chosen_arm in arms:
                spec=branch_spec(snap,k,arm);unit=out/'units'/unit_name(spec)
                if unit.exists():
                    z=read_json(unit)
                    require(z.get('unit_spec')==spec,'Existing checkpoint identity mismatch')
                    require(z.get('stage')==STAGE and z.get('result',{}).get('arm')==arm,
                            'Existing checkpoint corrupted')
                    require(z.get('checkpoint_source_sha256')==pins['r4a_snapshots_sha256'],
                            'Existing checkpoint changed source capture SHA')
                    require(z.get('plan')==plan,'Existing checkpoint plan differs from source selection')
                    result=z['result'];completed+=1
                else:
                    if a.max_new_units and new_units>=a.max_new_units:
                        print('R4C LIMIT reached; resume after increasing MAX_NEW_UNITS',flush=True)
                        write_summary(out,all_results,all_rewards,all_counts,finished=False)
                        return
                    result=run_one(v3,task,sc,seed,obj,snap,k,arm,chosen_arm,c3,road,legal,mf,
                                   plan['first_rng_after_original_C3_destroy_sha256'])
                    write_json(unit,{'stage':STAGE,'unit_spec':spec,'plan':plan,
                                     'result':result,'checkpoint_source_sha256':pins['r4a_snapshots_sha256']})
                    new_units+=1
                key=(tid,sid,int(seed),int(snap['stop_count']),arm)
                if key in prefix_by_arm:
                    earlier=prefix_by_arm[key]
                    require(result['events'][:len(earlier)]==earlier,
                            'K-step prefix diverged from shorter horizon for same arm')
                prefix_by_arm[key]=result['events']
                runs[arm]=result
                if arm=='SHAM_C3':compare_sham(runs['C3'],result)
                if arm not in ('C3','SHAM_C3'):
                    reward=normalize_reward(runs['C3'],result)
                    all_rewards.append({'unit_spec':spec,'first_action':result['first_action_id'],
                                        'c3_action':c3,**reward})
                all_results.append({'unit_spec':spec,'best_static':result['static_best_sec'],
                                    'best_road':result['road_best_sec'],
                                    'first_action':result['first_action_id'],
                                    'sha':result['best_route_sha256']})
            print('R4C',tid,sid,seed,'stop',snap['stop_count'],'K',k,
                  'C3 Road',round(runs['C3']['road_best_sec'],2),'alternatives',len(alts),flush=True)
    write_summary(out,all_results,all_rewards,all_counts,finished=True)
    print('R4C COMPLETE',str(out),'new',new_units,'reused',completed,
          'labels',len(all_rewards),flush=True)


def write_summary(out,all_results,rewards,plans,finished):
    from collections import defaultdict
    by=defaultdict(list)
    for r in rewards:by[str(r['unit_spec']['horizon'])].append(r)
    report={'stage':STAGE,'status':'COMPLETE' if finished else 'PARTIAL',
        'n_branch_results':len(all_results),'n_counterfactual_labels':len(rewards),
        'n_snapshot_start_points':len(plans),
        'n_sham_control_checks':len([x for x in all_results if x['unit_spec']['arm']=='SHAM_C3']),
        'per_horizon':{k:summarize(v) for k,v in sorted(by.items(),key=lambda p:int(p[0]))},
        'limits':['Development data: original July validation only, not blind independent test.',
                  'Fixed-iteration continuation K; does not reproduce wall-clock budget.',
                  'Candidates stratified by deterministic static sampler; no future road traffic used in policy.',
                  'RNG matched up to original first Section destroy; subsequent stochastic streams may diverge.',
                  'Alternative actions estimated using one captured RNG trajectory, not a confidence interval.']}
    write_json(out/'report.json',report)
    with (out/'counterfactual_labels.jsonl').open('w',encoding='utf8') as f:
        for r in rewards:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'branch_summary.jsonl').open('w',encoding='utf8') as f:
        for r in all_results:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'plans.jsonl').open('w',encoding='utf8') as f:
        for r in plans:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# R4-C Section连续搜索反事实（固定K步）','',
           f"状态：**{report['status']}**；快照起点：{len(plans)}；C3等价替换核查：{report['n_sham_control_checks']}；候选标签：{len(rewards)}",'',
           '| K | 标签数 | C3-候选Road成本差均值(s) | 正收益 | 严重损失（<-40s） |',
           '|---:|---:|---:|---:|---:|']
    for k,v in report['per_horizon'].items():
        lines.append(f"| {k} | {v['count']} | {v['mean_road_best_gain_sec']:+.3f} | {100*v['positive_frac']:.1f}% | {100*v['severe_harm_fraction']:.1f}% |")
    lines.extend(['','注意：这是真实C3内部状态、固定迭代长度下的首步Section反事实代理；不是原30秒墙钟收益。',''])
    (out/'report.md').write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir',required=True)
    p.add_argument('--r4b-dir',required=True)
    p.add_argument('--model-source',required=True)
    p.add_argument('--cost-dir',required=True)
    p.add_argument('--task-dir',required=True)
    p.add_argument('--od-cache-dir',required=True)
    p.add_argument('--output-dir')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--max-cases',type=int,default=0)
    p.add_argument('--max-snapshots-per-run',type=int,default=2)
    p.add_argument('--horizons',default='1,5,20,50')
    p.add_argument('--action-count',type=int,default=8)
    p.add_argument('--max-alternatives',type=int,default=8)
    p.add_argument('--sampler-seed',type=int,default=20261019)
    p.add_argument('--max-new-units',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not a.preflight_only and not a.output_dir:p.error('Require --output-dir for collection')
    main(a)
