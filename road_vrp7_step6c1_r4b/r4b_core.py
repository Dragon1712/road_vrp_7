"""R4-B: official ALNS 7.0.0 exact PRE-SELECT checkpoint continuation.

Reference path delegates to the ORIGINAL official ALNS.iterate. Only restored
path explicitly emulates the source-verified ALNS 7.0.0 loop. Fixed-iteration
K horizons are used, not fabricated 30s wall-clock counterfactuals.
"""
from __future__ import annotations
import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

STAGE='R4B-OFFICIAL-EXACT-CONTINUATION-AUDIT-v1'


def require(condition,reason):
    if not condition: raise ValueError(reason)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def validate_checkpoint(s,expected_max_iter=20000):
    from r4a_core import canonical_hash
    require(s.get('boundary')=='BEFORE_SELECT_AND_DESTROY','Wrong boundary')
    i=int(s['stop_count'])
    require(1<=i<=expected_max_iter,'stop_count invalid')
    require(int(s['next_iteration_one_based'])==i,'iteration counter mismatch')
    require(int(s['completed_updates_before_boundary'])==i-1,'update count mismatch')
    require(int(s['stop_max_iterations'])==expected_max_iter,'max iterations mismatch')
    require(int(s.get('real_selected_destroy_index',-1))==3,'Only actual Section pre-selection checkpoints')
    require(bool(s.get('snapshot_is_actual_section_decision',False)),'Not real Section snapshot')
    for key in ('current_routes','best_routes'):
        routes=s[key]
        require(isinstance(routes,list) and len(routes)==5 and all(isinstance(x,list) and x for x in routes),f'{key} invalid')
        ids=[int(c) for route in routes for c in route]
        require(len(ids)==len(set(ids)),'Duplicate customer in '+key)
        digest=canonical_hash(routes)
        want=s['current_route_sha256'] if key=='current_routes' else s['best_route_sha256']
        require(digest==want,key+' SHA mismatch')
    rng=s['rng_bit_generator_state']
    require(canonical_hash(rng)==s['rng_state_sha256'],'Snapshot RNG SHA mismatch')
    test_rng=np.random.default_rng();test_rng.bit_generator.state=copy.deepcopy(rng)
    require(len(s['roulette_destroy_weights'])==4 and len(s['roulette_repair_weights'])==2,'Weight shape mismatch')
    require(all(float(x)>0 and math.isfinite(float(x)) for x in s['roulette_destroy_weights']+s['roulette_repair_weights']),'Nonpositive weights')
    require(s['anneal_method']=='exponential','Unexpected SA method')
    require(float(s['anneal_temperature_before_accept'])>0,'Invalid temperature')
    return i


def signature(*,rng,selector,anneal,best,curr,count):
    from r4a_core import canonical_hash,jsonable,roads
    rng_state=jsonable(copy.deepcopy(rng.bit_generator.state))
    return {'stop_count':int(count),
        'current_route_sha256':canonical_hash(roads(curr)),
        'best_route_sha256':canonical_hash(roads(best)),
        'current_cost':float(curr.objective()),'best_cost':float(best.objective()),
        'rng_state_sha256':canonical_hash(rng_state),
        'roulette_destroy_weights':list(map(float,selector.destroy_weights)),
        'roulette_repair_weights':list(map(float,selector.repair_weights)),
        'anneal_temperature':float(anneal._temperature)}


def check_start(s,event):
    fields={'stop_count':s['stop_count'],
            'current_route_sha256':s['current_route_sha256'],
            'best_route_sha256':s['best_route_sha256'],
            'rng_state_sha256':s['rng_state_sha256']}
    for k,v in fields.items():
        require(event[k]==v,f'Original R4A snapshot does not match fresh frozen C3 ({k})')
    for x,y in [('roulette_destroy_weights','roulette_destroy_weights'),('roulette_repair_weights','roulette_repair_weights')]:
        require(np.array_equal(np.asarray(s[x],float),np.asarray(event[y],float)),f'Nonexact weights {x}')
    require(float(s['anneal_temperature_before_accept'])==event['anneal_temperature'],'Nonexact SA temperature')
    require(abs(float(s['current_surrogate_cost'])-event['current_cost'])<1e-8,'Current objective mismatch')
    require(abs(float(s['best_surrogate_cost'])-event['best_cost'])<1e-8,'Best objective mismatch')
    require(int(s['real_selected_destroy_index'])==int(event['destroy_index']),'Selected destroy mismatch')
    require(int(s['real_selected_repair_index'])==int(event['repair_index']),'Selected repair mismatch')


def verify_events(reference,resumed):
    require(len(reference)==len(resumed),'Trace length mismatch')
    for step,(a,b) in enumerate(zip(reference,resumed),1):
        require(a.keys()==b.keys(),f'Mismatched event keys at step {step}')
        for k in a:
            if a[k]!=b[k]:
                raise AssertionError(f'Step {step}: {k} diverged: reference={str(a[k])[:130]}, resume={str(b[k])[:130]}')
    return True


class OfficialReferenceObserver:
    """Records a sparse fixed set of original, uninterrupted ALNS transitions."""
    def __init__(self,selector,accept,stop,interested):
        from r4a_core import inspect_anneal
        self.selector=selector;self.stop=stop;self.anneal=inspect_anneal(accept)
        self.interested=set(interested);self.events={};self.rng=None
        self.calls=0;self.updates=0

    def __call__(self,rng,best,curr):
        self.calls+=1
        step=int(self.stop.count)
        if step in self.interested:
            self.events[step]=signature(rng=rng,selector=self.selector,anneal=self.anneal,best=best,curr=curr,count=step)
        pair=self.selector(rng,best,curr)
        if step in self.interested:
            self.events[step].update(destroy_index=int(pair[0]),repair_index=int(pair[1]))
        self.rng=rng
        return pair

    def update(self,cand,d_idx,r_idx,outcome):
        from r4a_core import canonical_hash,roads
        self.selector.update(cand,d_idx,r_idx,outcome)
        step=int(self.stop.count)
        if step in self.interested:
            self.events[step].update(candidate_route_sha256=canonical_hash(roads(cand)),
                outcome=int(outcome),
                anneal_temperature_after=float(self.anneal._temperature),
                roulette_destroy_weights_after=list(map(float,self.selector.destroy_weights)),
                roulette_repair_weights_after=list(map(float,self.selector.repair_weights)))
        self.updates+=1


def restored_iterator(engine,initial_solution,op_select,accept,stop,snapshot,k,**kwargs):
    """Restore real C3 operator closures into ALNS 7.0.0 manual K-step loop.

    Source mirrored from alns/ALNS.py v7.0.0, never changes destroy/repair,
    selection or acceptance. The very first boundary has ALREADY called stop.
    """
    from alns.Result import Result
    from alns.Statistics import Statistics
    from r4a_core import inspect_anneal,canonical_hash,roads
    from section_v2 import frozen
    from solvers import VRPState
    i=validate_checkpoint(snapshot,expected_max_iter=stop.max_iter)
    require(k>=1 and i+k-1<=stop.max_iter,'Horizon exceeds captured iteration budget')
    require(getattr(stop,'count',0)==0,'Fresh stop object expected')
    rng=engine._rng
    rng.bit_generator.state=copy.deepcopy(snapshot['rng_bit_generator_state'])
    op_select.destroy_weights[:]=np.asarray(snapshot['roulette_destroy_weights'],float)
    op_select.repair_weights[:]=np.asarray(snapshot['roulette_repair_weights'],float)
    anneal=inspect_anneal(accept)
    require(list(map(float,op_select.scores))==list(map(float,snapshot['roulette_scores'])),'Roulette scores changed')
    require(float(op_select.decay)==float(snapshot['roulette_decay']),'Roulette decay changed')
    for x,y in [('_start_temperature','anneal_start_temperature'),('_end_temperature','anneal_end_temperature'),('_step','anneal_step'),('_method','anneal_method')]:
        require(getattr(anneal,x)==snapshot[y],'Anneal config differs: '+x)
    anneal._temperature=float(snapshot['anneal_temperature_before_accept'])
    obj=initial_solution.objective_ref
    current=VRPState(frozen(snapshot['current_routes']),obj)
    best=VRPState(frozen(snapshot['best_routes']),obj)
    # No wall deadline in this FIXED-K continuation; time remaining cannot be
    # resumed exactly and must not create a false 30s continuation claim.
    stop.count=i;stop.start=time.perf_counter();stop.deadline=float('inf')
    stats=Statistics();stats.collect_objective(current.objective());stats.collect_runtime(time.perf_counter())
    events=[]
    for step in range(k):
        ev=signature(rng=rng,selector=op_select,anneal=anneal,best=best,curr=current,count=stop.count)
        d_idx,r_idx=op_select(rng,best,current)
        ev.update(destroy_index=int(d_idx),repair_index=int(r_idx))
        d_name,d_fun=engine.destroy_operators[d_idx]
        r_name,r_fun=engine.repair_operators[r_idx]
        partial=d_fun(current,rng,**kwargs)
        candidate=r_fun(partial,rng,**kwargs)
        best,current,outcome=engine._eval_cand(accept,best,current,candidate,**kwargs)
        op_select.update(candidate,d_idx,r_idx,outcome)
        ev.update(candidate_route_sha256=canonical_hash(roads(candidate)),
            outcome=int(outcome),anneal_temperature_after=float(anneal._temperature),
            roulette_destroy_weights_after=list(map(float,op_select.destroy_weights)),
            roulette_repair_weights_after=list(map(float,op_select.repair_weights)))
        events.append(ev)
        stats.collect_objective(current.objective())
        stats.collect_destroy_operator(d_name,outcome)
        stats.collect_repair_operator(r_name,outcome)
        stats.collect_runtime(time.perf_counter())
        if step<k-1: stop.count+=1  # original BothBudget's next successful call
    return Result(best,stats),events


def take_snapshots(all_snapshots,*,task_id,scenario_id,seed,max_states):
    matched=sorted((s for s in all_snapshots if s['task_id']==task_id and s['scenario_id']==scenario_id and int(s['seed'])==seed),key=lambda s:int(s['stop_count']))
    require(matched,'No checkpoints for run')
    matched=matched[:max_states]
    for s in matched:validate_checkpoint(s)
    require(len({s['stop_count'] for s in matched})==len(matched),'Duplicate checkpoint counters')
    return matched


def common_data_audit(capture_dir):
    """Pin supplied R4A checkpoint content; does not silently accept wrong role."""
    from r4a_core import sha256
    p=Path(capture_dir);mf=load_json(p/'manifest.json');rep=load_json(p/'report.json');prov=load_json(p/'provenance.json')
    require(rep.get('status')=='COMPLETE' and mf.get('arm')=='capture' and mf.get('section_only'),'Require complete real Section capture')
    require(mf.get('alns_version')=='7.0.0' and mf.get('split')=='validation','Wrong capture version/role')
    require(sha256(p/'metrics.jsonl')==prov['metrics_sha256'],'R4A metrics SHA differs')
    require(sha256(p/'snapshots.jsonl')==prov['snapshots_sha256'],'R4A snapshots SHA differs')
    require(sha256(p/'manifest.json')==prov['manifest_sha256'],'R4A manifest SHA differs')
    snap=[json.loads(line) for line in (p/'snapshots.jsonl').read_text().splitlines() if line.strip()]
    require(len(snap)==rep['snapshots'] and bool(snap),'Invalid snapshot count')
    for s in snap:validate_checkpoint(s,mf['max_iterations'])
    return mf,snap
