"""Non-invasive, pre-selection state capture for OFFICIAL ALNS 7.0.0.

The wrapper forwards exactly one selector call and each outcome update; it does
not copy/mutate the live generator or selector. Snapshots are recorded in RAM
and flushed after v3.solve() returns, *outside* the solver's search budget.
"""
from __future__ import annotations
import copy
import hashlib
import inspect
import json
import time
from contextlib import contextmanager
from collections import Counter
import numpy as np

STAGE = 'R4A-OFFICIAL-C3-PRESELECTION-SNAPSHOT-v1'


def jsonable(v):
    if isinstance(v, dict): return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v, (tuple,list)): return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, (str,int,float,bool)) or v is None: return v
    raise TypeError(f'Unsupported capture value: {type(v).__name__}')


def canonical_hash(v):
    return hashlib.sha256(json.dumps(jsonable(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def roads(state):
    """Snapshot immutable route tuples, not mutable live states."""
    return [[int(c) for c in route] for route in state.routes]


def inspect_anneal(accept):
    """Retrieve the ACTUAL v3.solve local anneal from its Acceptance.__call__ closure.

    We fail closed if upstream v3 changes; no fabricated temperature surrogate.
    """
    func=getattr(accept,'__call__',None)
    if func is None: raise RuntimeError('Acceptance missing __call__')
    values=inspect.getclosurevars(func).nonlocals
    obj=values.get('anneal')
    attrs=('_temperature','_start_temperature','_end_temperature','_step','_method')
    if obj is None or not all(hasattr(obj,a) for a in attrs):
        raise RuntimeError('Could not locate REAL v3 SimulatedAnnealing closure; inspect solver source')
    return obj


class SelectorObserver:
    """Duck-typed ALNS selection scheme proxy, transparently delegating."""
    def __init__(self, selector, accept, stop, save_at, *, max_snapshots=16, section_only=False):
        self.selector=selector
        self.accept=accept
        self.stop=stop
        self.anneal=inspect_anneal(accept)
        self.save_at=set(save_at)
        self.section_only=bool(section_only)
        self.completed_targets=set()
        self.max_snapshots=max_snapshots
        self.calls=0
        self.updates=0
        self.outcomes=Counter()
        self.snapshots=[]
        if not all(hasattr(selector,x) for x in ('destroy_weights','repair_weights','decay','scores','update')):
            raise RuntimeError('Expected official alns==7.0.0 RouletteWheel')

    def _capture(self,rng,best,curr):
        a=self.anneal
        b=roads(best);c=roads(curr)
        state=jsonable(copy.deepcopy(rng.bit_generator.state))
        w_d=np.asarray(self.selector.destroy_weights,dtype=float).tolist()
        w_r=np.asarray(self.selector.repair_weights,dtype=float).tolist()
        if not (len(w_d)==4 and len(w_r)==2):
            raise AssertionError('Frozen C3 expected 4 destroy, 2 repair operators')
        snap={
            'boundary':'BEFORE_SELECT_AND_DESTROY',
            'next_iteration_one_based':self.calls,
            'stop_count':int(self.stop.count),
            'stop_max_iterations':int(self.stop.max_iter),
            'stop_wall_budget_sec':float(self.stop.wall),
            'wall_elapsed_before_select_sec':max(0.,float(time.perf_counter()-self.stop.start)),
            'wall_remaining_before_select_sec':max(0.,float(self.stop.deadline-time.perf_counter())),
            'current_routes':c,'best_routes':b,
            'current_surrogate_cost':float(curr.objective()),
            'best_surrogate_cost':float(best.objective()),
            'current_route_sha256':canonical_hash(c),
            'best_route_sha256':canonical_hash(b),
            'rng_bit_generator_state':state,
            'rng_state_sha256':canonical_hash(state),
            'roulette_destroy_weights':w_d,
            'roulette_repair_weights':w_r,
            'roulette_decay':float(self.selector.decay),
            'roulette_scores':list(map(float,self.selector.scores)),
            'anneal_temperature_before_accept':float(a._temperature),
            'anneal_start_temperature':float(a._start_temperature),
            'anneal_end_temperature':float(a._end_temperature),
            'anneal_step':float(a._step),
            'anneal_method':str(a._method),
            'completed_updates_before_boundary':self.updates,
        }
        self.snapshots.append(snap)

    def __call__(self,rng,best,curr):
        self.calls+=1
        if not self.section_only:
            if self.calls in self.save_at and len(self.snapshots)<self.max_snapshots:
                self._capture(rng,best,curr)
            return self.selector(rng,best,curr)
        # Capture ONLY the first actual Section-selection boundary on/after
        # each milestone. To preserve the true pre-select RNG state, the
        # temporary snapshot is taken BEFORE selecting the operator. If the
        # original wheel selects a different operator we simply discard it.
        eligible=sorted(t for t in self.save_at
                        if t<=self.calls and t not in self.completed_targets)
        pending=eligible[0] if eligible and len(self.snapshots)<self.max_snapshots else None
        if pending is not None:self._capture(rng,best,curr)
        choice=self.selector(rng,best,curr)  # exactly one REAL RNG call
        if pending is not None:
            if int(choice[0])==3:  # frozen C3: random,worst,related,section
                self.snapshots[-1]['requested_milestone']=pending
                self.snapshots[-1]['real_selected_destroy_index']=int(choice[0])
                self.snapshots[-1]['real_selected_repair_index']=int(choice[1])
                self.snapshots[-1]['snapshot_is_actual_section_decision']=True
                self.completed_targets.add(pending)
            else:self.snapshots.pop()
        return choice

    def update(self,cand,d_idx,r_idx,outcome):
        self.selector.update(cand,d_idx,r_idx,outcome)
        self.updates+=1
        self.outcomes[str(int(outcome))]+=1

    def summary(self):
        return {'selector_calls':self.calls,'selector_updates':self.updates,
                'outcomes':dict(self.outcomes),'snapshot_count':len(self.snapshots)}


@contextmanager
def instrument_official_alns(save_at,max_snapshots=16,section_only=False):
    """Patch only ALNS.iterate temporarily; NEVER patch v3.solve or its operators.

    Caller must be single-threaded: this global class-level patch is intentionally
    scoped to ONE solver invocation and restored in finally.
    """
    from alns import ALNS
    original=ALNS.iterate
    observed=[]
    def intercepted(engine,initial_solution,op_select,accept,stop,**kwargs):
        wrapper=SelectorObserver(op_select,accept,stop,save_at,max_snapshots=max_snapshots,section_only=section_only)
        observed.append(wrapper)
        result=original(engine,initial_solution,wrapper,accept,stop,**kwargs)
        if wrapper.calls!=wrapper.updates:
            raise AssertionError('Selector proxy lost operator updates')
        return result
    ALNS.iterate=intercepted
    try:yield observed
    finally:ALNS.iterate=original


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,obj):
    from pathlib import Path
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)
