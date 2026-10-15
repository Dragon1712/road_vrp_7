#!/usr/bin/env python3
"""Comparable July-only CVRP heuristics; official ALNS uses N-Wouda/ALNS 7.0.0."""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
import numpy as np
from road_vrp import legal,loads

METHODS=('greedy','sa','vnd','official_alns')


def copied(routes):return [list(map(int,r)) for r in routes]


def greedy(task,obj,seed):
    rng=np.random.default_rng(seed)
    ids=[int(c['id']) for c in task['customers']]
    rng.shuffle(ids)
    k=int(task['vehicles']);r=[[] for _ in range(k)]
    # Seed each vehicle with a spatially separated customer where possible.
    ids.sort(key=lambda i:obj.od.static[0,obj.index[i]],reverse=True)
    for v in range(k):r[v].append(ids.pop(0))
    while ids:
        best=None
        for cid in ids:
            for v in range(k):
                route=r[v]
                if sum(obj.dem[x] for x in route)+obj.dem[cid]>obj.capacity:continue
                old=obj.route_cost(route)
                for pos in range(len(route)+1):
                    new=route[:pos]+[cid]+route[pos:]
                    delta=obj.route_cost(new)-old
                    key=(delta,cid,v,pos)
                    if best is None or key<best[0]:best=(key,cid,v,pos)
        if best is None:
            routes=copied(task['seed_routes']);legal(task,routes)
            return routes,{'fallback_seed_routes':True}
        _,cid,v,pos=best;r[v].insert(pos,cid);ids.remove(cid)
    legal(task,r)
    return r,{'fallback_seed_routes':False}


def perturb(routes,rng):
    r=copied(routes)
    action=int(rng.integers(3))
    if action==0:
        a=int(rng.integers(len(r)));b=int(rng.integers(len(r)))
        if r[a] and r[b]:
            i=int(rng.integers(len(r[a])));j=int(rng.integers(len(r[b])))
            r[a][i],r[b][j]=r[b][j],r[a][i]
    elif action==1:
        a=int(rng.integers(len(r)));b=int(rng.integers(len(r)))
        if len(r[a])>1:
            i=int(rng.integers(len(r[a])));cid=r[a].pop(i)
            j=int(rng.integers(len(r[b])+1));r[b].insert(j,cid)
    else:
        a=int(rng.integers(len(r)))
        if len(r[a])>2:
            i,j=sorted(rng.choice(len(r[a]),size=2,replace=False))
            r[a][i:j+1]=reversed(r[a][i:j+1])
    return r


def vnd(task,obj,initial,deadline,max_iter):
    r=copied(initial);best=obj.cost(r);iters=0;improved=0
    while iters<max_iter and time.perf_counter()<deadline:
        change=False
        # Fixed and deterministic neighborhoods: relocate, swap, 2-opt.
        for mode in ('relocate','swap','twoopt'):
            if change or time.perf_counter()>=deadline:break
            for a in range(len(r)):
                if change or time.perf_counter()>=deadline:break
                for i in range(len(r[a])):
                    if change or time.perf_counter()>=deadline:break
                    for b in range(len(r)) if mode!='twoopt' else (a,):
                        if change or time.perf_counter()>=deadline:break
                        for j in range(len(r[b])+(mode=='relocate')):
                            if time.perf_counter()>=deadline or iters>=max_iter:break
                            candidate=copied(r)
                            if mode=='relocate':
                                if a==b or len(r[a])==1:continue
                                cid=candidate[a].pop(i);candidate[b].insert(j,cid)
                            elif mode=='swap':
                                if a==b and i==j:continue
                                candidate[a][i],candidate[b][j]=candidate[b][j],candidate[a][i]
                            else:
                                if j<=i+1:continue
                                candidate[a][i:j+1]=reversed(candidate[a][i:j+1])
                            if not obj.feasible(candidate):continue
                            cost=obj.cost(candidate);iters+=1
                            if cost<best-1e-7:
                                r=candidate;best=cost;improved+=1;change=True;break
            if change:break
        if not change:break
    return r,iters,{'local_improvements':improved}


def sa(task,obj,initial,seed,wall,max_iter):
    rng=np.random.default_rng(seed)
    r=copied(initial);cost=obj.cost(r);best=cost;br=copied(r)
    start=time.perf_counter();deadline=start+wall;iters=accepted=0
    while iters<max_iter and time.perf_counter()<deadline:
        cand=perturb(r,rng);iters+=1
        if not obj.feasible(cand):continue
        val=obj.cost(cand)
        temp=max(1e-4, max(cost,1.)*.03*(1-iters/max_iter))
        if val<=cost or rng.random()<math.exp(max(-100.,(cost-val)/temp)):
            r=cand;cost=val;accepted+=1
        if val<best-1e-8:br=copied(cand);best=val
    return br,iters,{'accepted_moves':accepted,'search_wall_sec':time.perf_counter()-start}


def choose_destroy(name,routes,obj,rng,q):
    all_ids=[x for rt in routes for x in rt]
    q=min(q,len(all_ids)-len(routes)) # exact K must remain nonempty
    if q<1:return copied(routes),[]
    if name=='random':ordered=list(rng.permutation(all_ids))
    elif name=='related':
        anchor=int(rng.choice(all_ids));i=obj.index[anchor]
        ordered=sorted(all_ids,key=lambda x:obj.od.static[i,obj.index[x]]+obj.od.static[obj.index[x],i]
                       +20*abs(obj.dem[x]-obj.dem[anchor]))
    elif name=='worst':
        ranks=[]
        for rt in routes:
            if len(rt)<=1:continue
            old=obj.route_cost(rt)
            for x in rt:
                gain=old-obj.route_cost([y for y in rt if y!=x])
                ranks.append((gain+1e-6*rng.random(),x))
        ordered=[x for _,x in sorted(ranks,reverse=True)] + all_ids
    else:raise ValueError(name)
    working=copied(routes);removed=[]
    for x in ordered:
        if len(removed)>=q:break
        for rt in working:
            if x in rt and len(rt)>1:
                rt.remove(x);removed.append(int(x));break
    return working,removed


def insert_removed(partial,removed,obj,name,deadline=None):
    r=copied(partial);left=list(removed)
    while left:
        if deadline is not None and time.perf_counter()>=deadline:return None
        rankings=[]
        for x in left:
            options=[]
            for k,route in enumerate(r):
                if sum(obj.dem[y] for y in route)+obj.dem[x]>obj.capacity:continue
                old=obj.route_cost(route)
                for j in range(len(route)+1):
                    diff=obj.route_cost(route[:j]+[x]+route[j:])-old
                    options.append((diff,k,j))
            options.sort()
            if not options:continue
            if name=='greedy':rank=(options[0][0],x)
            else:
                rank=( -(options[min(len(options)-1,1)][0]-options[0][0]),options[0][0],x)
            rankings.append((rank,x,options[0]))
        if not rankings:return None
        _,x,(_,k,pos)=min(rankings,key=lambda z:z[0])
        r[k].insert(pos,x);left.remove(x)
    return r

@dataclass(frozen=True)
class VRPState:
    routes:tuple
    objective_ref:object
    removed:tuple=()
    previous:tuple|None=None
    def objective(self):
        if self.removed:raise RuntimeError('Cannot score a partial ALNS state')
        return self.objective_ref.cost(self.routes)
    def get_context(self):return None

class BothBudget:
    def __init__(self,iters,wall):self.max_iter=iters;self.wall=wall;self.count=0;self.start=None;self.deadline=float('inf');self.reason='not_started'
    def __call__(self,rng,best,current):
        if self.start is None:
            self.start=time.perf_counter();self.deadline=self.start+self.wall
        if self.count>=self.max_iter:self.reason='iteration_budget';return True
        if time.perf_counter()>=self.deadline:self.reason='wall_budget';return True
        self.count+=1;return False


def official_alns(task,obj,initial,seed,wall,max_iter):
    try:
        import alns
        from alns import ALNS
        from alns.accept import SimulatedAnnealing
        from alns.select import RouletteWheel
    except ImportError as exc:
        raise RuntimeError('Official ALNS is required: python -m pip install alns==7.0.0') from exc
    stop=BothBudget(max_iter,wall)
    rng=np.random.default_rng(seed)
    engine=ALNS(rng)
    usage={'destroy':{},'repair':{}}
    def make_d(name):
        def fn(s,rng):
            usage['destroy'][name]=usage['destroy'].get(name,0)+1
            q=int(rng.integers(max(2,len(obj.dem)//10),max(3,len(obj.dem)//4+1)))
            part,removed=choose_destroy(name,s.routes,obj,rng,q)
            return VRPState(tuple(tuple(r) for r in part),obj,tuple(removed),s.routes)
        return fn
    def make_r(name):
        def fn(s,rng):
            usage['repair'][name]=usage['repair'].get(name,0)+1
            cand=insert_removed(s.routes,s.removed,obj,name,deadline=stop.deadline)
            if cand is None or not obj.feasible(cand):cand=copied(s.previous)
            return VRPState(tuple(tuple(r) for r in cand),obj)
        return fn
    for name in ('random','worst','related'):engine.add_destroy_operator(make_d(name),name=name)
    for name in ('greedy','regret2'):engine.add_repair_operator(make_r(name),name=name)
    select=RouletteWheel(scores=[8.,4.,1.,0.],decay=.8,num_destroy=3,num_repair=2)
    start=VRPState(tuple(tuple(r) for r in initial),obj)
    accept=SimulatedAnnealing.autofit(init_obj=max(start.objective(),1e-6),worse=.05,accept_prob=.75,num_iters=max_iter)
    t=time.perf_counter()
    res=engine.iterate(start,select,accept,stop)
    best=copied(res.best_state.routes)
    if not obj.feasible(best):raise AssertionError('Official ALNS returned infeasible routes')
    return best,stop.count,{'search_wall_sec':time.perf_counter()-t,
                           'termination_reason':stop.reason,'operator_usage':usage,
                           'alns_version':getattr(alns,'__version__','not_reported')}


def solve(method,task,obj,seed=42,wall=5.,max_iter=5000):
    if method not in METHODS:raise ValueError('Unknown method: '+method)
    t0=time.perf_counter();initial,info=greedy(task,obj,seed)
    init_cost=obj.cost(initial);construction=time.perf_counter()-t0
    if method=='greedy':return {'routes':initial,'initial_cost':init_cost,'planned_cost':init_cost,
                                'iterations':0,'construct_wall_sec':construction,'search_wall_sec':0.,'diagnostics':info}
    t=time.perf_counter()
    if method=='sa':r,iters,diag=sa(task,obj,initial,seed,wall,max_iter)
    elif method=='vnd':r,iters,diag=vnd(task,obj,initial,t+wall,max_iter)
    else:r,iters,diag=official_alns(task,obj,initial,seed,wall,max_iter)
    search_wall=time.perf_counter()-t
    legal(task,r)
    cost=obj.cost(r)
    if cost>init_cost+1e-5:raise AssertionError('Best solution worse than initial')
    return {'routes':r,'initial_cost':init_cost,'planned_cost':cost,
            'iterations':iters,'construct_wall_sec':construction,
            'search_wall_sec':search_wall,'diagnostics':{**info,**diag}}
