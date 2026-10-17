#!/usr/bin/env python3
"""Contiguous customer Section ALNS operators for July TD-CVRP30.

Strict: Section removal always retains each truck's >=1 customer, repair
inserts WHOLE blocks (forward/reverse), never individualizes their customers.
Oracle uses hindsight OD; static mode never looks up OD.dynamic or Road.observed.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import time
import numpy as np


@dataclass(frozen=True)
class Block:
    customers: tuple[int, ...]
    original_route: int
    original_position: int


@dataclass(frozen=True)
class SectionState:
    routes: tuple[tuple[int,...], ...]
    objective_ref: object
    blocks: tuple[Block,...] = ()
    previous: tuple[tuple[int,...], ...] | None = None

    def objective(self):
        if self.blocks:
            raise RuntimeError('Partial SectionState must be repaired before scoring')
        return self.objective_ref.cost(self.routes)

    def get_context(self):return None


def copied(routes):return [list(map(int,r)) for r in routes]


def _normalised(values):
    """Robust unitless normalization per candidate set, no fixed scale domination."""
    a=np.asarray(values,dtype=float)
    if not len(a):return a
    if not np.isfinite(a).all():raise ValueError('Nonfinite Section score')
    low,high=np.quantile(a,[.10,.90])
    if high-low<1e-8:return np.zeros(len(a),dtype=float)
    return np.clip((a-low)/(high-low),0.,1.)


def feasible_partial(task,routes,blocks):
    k=int(task['vehicles'])
    if len(routes)!=k or any(not r for r in routes):return False
    dem={int(x['id']):int(x['demand']) for x in task['customers']}
    seen=[x for r in routes for x in r]+[x for b in blocks for x in b.customers]
    if sorted(seen)!=sorted(dem):return False
    return all(sum(dem[x] for x in r)<=int(task['capacity']) for r in routes)


class PathSupportProxy:
    """Offline OD observed support on STATIC shortest paths at slot departure.

    Not dynamic-path support and not direct per-OSM edge observations. Computed
    once per task+scenario+road hash and charged to precomputation. Never called
    for `static` planning. Can be disabled via support_weight=0.
    """
    def __init__(self,road,od,cache_dir=None):
        self.od=od
        nodes=[str(n) for n in od.nodes]
        knots=np.asarray(od.knot_seconds,float)
        key=json.dumps({'edge_sha':road.edge_file_hash,'cost_sha':road.cost_file_hash,
                        'nodes':nodes,'depart':float(od.start),'knots':knots.tolist()},sort_keys=True)
        digest=hashlib.sha256(key.encode()).hexdigest()
        out=(Path(cache_dir)/f'path_support_{digest}.npz') if cache_dir else None
        if out and out.exists():
            with np.load(out,allow_pickle=False) as z:
                self.support=z['support']
            if self.support.shape!=(len(knots),len(nodes),len(nodes)):
                raise ValueError('Corrupted path support cache')
            self.cached=True
            return
        self.cached=False
        n=len(nodes)
        masks=np.zeros((len(knots),n,n),dtype=np.float32)
        lengths=road.edges.length_m.to_numpy(float)
        # O(n) Dijkstra passes, rather than n^2 Dijkstra passes.
        for i,u in enumerate(nodes):
            distances,back=road.sssp(u,nodes,od.start,'static')
            for j,v in enumerate(nodes):
                if i==j:continue
                path=[];cur=v
                while cur!=u:
                    if cur not in back:raise ValueError(f'Unreachable OSM OD {u}->{v}')
                    prev,e=back[cur];path.append(int(e));cur=prev
                path=np.asarray(path,dtype=np.int64)
                if not len(path):continue
                den=lengths[path].sum()
                # snapshot at OD departure, not observed at each traversal time
                for m,lag in enumerate(knots):
                    slot=int((od.start+float(lag))//road.dt)
                    if not 0<=slot<len(road.times):raise ValueError('Support date outside traffic tensor')
                    masks[m,i,j]=float(np.sum(lengths[path]*road.observed[slot,path])/den)
        self.support=masks
        if out:
            out.parent.mkdir(parents=True,exist_ok=True)
            import tempfile
            fd,p=tempfile.mkstemp(prefix='support_',suffix='.npz',dir=out.parent)
            os.close(fd)
            try:
                np.savez_compressed(p,support=masks)
                os.replace(p,out)
            finally:
                if os.path.exists(p):os.unlink(p)

    def get(self,i,j,sec):
        if i==j:return 0.
        knot=self.od.knot_seconds
        return float(np.interp(min(max(sec,0.),knot[-1]),knot,self.support[:,i,j]))


class SectionSelector:
    def __init__(self,obj,*,support=None,traffic_weight=.45,support_weight=.15,
                 lengths=(2,3),lag_sec=1200.,section_probability=.30):
        self.obj=obj;self.support=support
        self.traffic_weight=float(traffic_weight)
        self.support_weight=float(support_weight)
        self.lengths=tuple(int(x) for x in lengths)
        self.lag=float(lag_sec)
        self.section_probability=float(section_probability)
        if not self.lengths or any(x<2 or x>4 for x in self.lengths):
            raise ValueError('Section lengths must be 2-4')
        if not (0<=self.traffic_weight<=1 and 0<=self.support_weight<=1
                and 0<=self.section_probability<=1):raise ValueError('Invalid weights')
        self.stats={'calls':0,'no_candidate':0,'traffic_changed':0,'traffic_compared':0,
                    'support_mean_sum':0.,'support_samples':0,'blocks_removed':0,
                    'customer_removed':0,'static_score_mean_sum':0.,'traffic_score_mean_sum':0.,
                    'score_samples':0,'multi_block_moves':0}

    def _trace(self,route):
        t=0.;prev=0;starts=[]
        for customer in route:
            starts.append(t)
            nxt=self.obj.index[int(customer)]
            t+=self.obj.od.travel(prev,nxt,t,self.obj.planner)
            t+=float(self.obj.task['service_seconds']);prev=nxt
        return starts

    def _traffic(self,route,pos,length,starts):
        if self.obj.planner!='oracle':return 0.,0.
        ods=self.obj.od
        # section-related entry, internal, exit arcs; propagate actual t along them
        pre=0 if pos==0 else self.obj.index[route[pos-1]]
        customers=route[pos:pos+length]
        nxt=0 if pos+length==len(route) else self.obj.index[route[pos+length]]
        seq=[pre]+[self.obj.index[i] for i in customers]+[nxt]
        t=float(starts[pos])
        values=[];supports=[]
        for x,y in zip(seq[:-1],seq[1:]):
            baseline=float(ods.static[x,y])
            if baseline<=1e-9:continue
            now=float(ods.travel(x,y,t,'oracle'))
            later=float(ods.travel(x,y,min(t+self.lag,ods.knot_seconds[-1]),'oracle'))
            # moderate bounded joint deviation + trend, no overflow/outliers
            values.append(min(3.,abs(math.log(max(now,1e-6)/baseline))) +
                          .5*min(3.,abs(later-now)/baseline))
            if self.support is not None:supports.append(self.support.get(x,y,t))
            t+=now
            if y!=0 and y!=nxt:t+=float(self.obj.task['service_seconds'])
        return (float(np.mean(values)) if values else 0.,
                float(np.mean(supports)) if supports else 0.)

    def candidates(self,routes,need_traffic=False):
        items=[]
        for k,r in enumerate(routes):
            if len(r)<3:continue
            arrivals=self._trace(r) if (need_traffic and self.obj.planner=='oracle') else None
            for length in self.lengths:
                for pos in range(len(r)-length+1):
                    if len(r)-length<1:continue
                    pre=0 if pos==0 else self.obj.index[r[pos-1]]
                    nxt=0 if pos+length==len(r) else self.obj.index[r[pos+length]]
                    seq=[pre]+[self.obj.index[x] for x in r[pos:pos+length]]+[nxt]
                    static_cost=sum(float(self.obj.od.static[a,b]) for a,b in zip(seq[:-1],seq[1:]))
                    traffic,support=self._traffic(r,pos,length,arrivals) if arrivals is not None else (0.,0.)
                    items.append({'route':k,'pos':pos,'size':length,
                                  'static':static_cost,'traffic':traffic,'support':support})
        return items

    def _ordered(self,items,mode,rng):
        count=len(items)
        stat=_normalised([x['static'] for x in items])
        traf=_normalised([x['traffic'] for x in items])
        sup=np.array([x['support'] for x in items],float)
        # Support only modulates the *dynamic* part; no bonus from support alone.
        feature=traf*(1-self.support_weight+self.support_weight*sup)
        if mode=='section_random':return list(map(int,rng.permutation(count)))
        if mode in ('section_static','section_static_probe') or self.obj.planner=='static':score=stat
        else:
            if mode=='section_shuffled':
                # preserve the traffic feature distribution but break candidate identity
                feature=rng.permutation(feature)
            elif mode!='section_traffic':raise ValueError('Unsupported Section mode '+mode)
            score=(1-self.traffic_weight)*stat+self.traffic_weight*feature
        jitter=rng.uniform(0.,1e-8,count)
        self.stats['score_samples']+=count
        self.stats['static_score_mean_sum']+=float(stat.sum())
        self.stats['traffic_score_mean_sum']+=float(traf.sum())
        self.stats['support_mean_sum']+=float(sup.sum())
        self.stats['support_samples']+=count
        if mode=='section_traffic' and self.obj.planner=='oracle':
            before=np.argsort(-(stat+jitter));after=np.argsort(-(score+jitter))
            self.stats['traffic_compared']+=1
            self.stats['traffic_changed']+=int(int(before[0])!=int(after[0]))
        return list(map(int,np.argsort(-(score+jitter))))

    def destroy(self,mode,routes,rng,q=None):
        self.stats['calls']+=1
        arr=copied(routes)
        need_traffic=(mode in ('section_traffic','section_shuffled','section_static_probe'))
        items=self.candidates(arr,need_traffic=need_traffic)
        if not items:
            self.stats['no_candidate']+=1
            return SectionState(tuple(map(tuple,arr)),self.obj,(),tuple(map(tuple,arr)))
        order=self._ordered(items,mode,rng)
        # weighted selection among best 20%, permits exploration
        top=max(1,int(math.ceil(len(order)*.2)))
        chosen=items[order[int(rng.integers(top))]]
        selected=[chosen]
        if rng.random()<self.section_probability:
            other=[idx for idx in order if items[idx]['route']!=chosen['route']]
            if other:
                selected.append(items[other[int(rng.integers(min(len(other),max(1,int(len(other)*.2)))))]])
        blocks=[]
        for c in selected:
            k=c['route'];i=c['pos'];length=c['size']
            blocks.append(Block(tuple(arr[k][i:i+length]),k,i))
        working=copied(arr)
        for b in sorted(blocks,key=lambda z:(z.original_route,z.original_position),reverse=True):
            del working[b.original_route][b.original_position:b.original_position+len(b.customers)]
        if not feasible_partial(self.obj.task,working,blocks):raise AssertionError('Section removal violates constraints')
        self.stats['blocks_removed']+=len(blocks)
        self.stats['customer_removed']+=sum(len(b.customers) for b in blocks)
        self.stats['multi_block_moves']+=int(len(blocks)==2)
        return SectionState(tuple(map(tuple,working)),self.obj,tuple(blocks),tuple(map(tuple,arr)))

    def diagnostics(self):
        d=dict(self.stats)
        d['traffic_change_rate']=d['traffic_changed']/max(d['traffic_compared'],1)
        d['mean_static_score']=d['static_score_mean_sum']/max(d['score_samples'],1)
        d['mean_traffic_score']=d['traffic_score_mean_sum']/max(d['score_samples'],1)
        d['mean_support_proxy']=d['support_mean_sum']/max(d['support_samples'],1)
        return d


def _insert_options(routes,block,obj):
    ids=block.customers;needs=sum(obj.dem[x] for x in ids)
    for k,rt in enumerate(routes):
        if sum(obj.dem[x] for x in rt)+needs>obj.capacity:continue
        old=obj.route_cost(rt)
        for orientation in (ids,ids[::-1]):
            for pos in range(len(rt)+1):
                new=rt[:pos]+list(orientation)+rt[pos:]
                delta=obj.route_cost(new)-old
                yield (float(delta),k,pos,tuple(orientation))


def _greedy_block_insert(partial,blocks,obj,*,repair_name='greedy',deadline=float('inf')):
    routes=copied(partial)
    remaining=list(blocks)
    while remaining:
        if time.perf_counter()>=deadline:return None
        possibilities=[]
        for j,block in enumerate(remaining):
            opts=sorted(_insert_options(routes,block,obj))
            if not opts:continue
            if repair_name=='regret2':
                loss=opts[1][0]-opts[0][0] if len(opts)>1 else 1e5
                rating=(-loss,opts[0][0],j)
            else:rating=(opts[0][0],j)
            possibilities.append((rating,j,opts[0]))
        if not possibilities:return None
        _,index,(delta,k,pos,orientation)=min(possibilities,key=lambda z:z[0])
        routes[k][pos:pos]=orientation
        remaining.pop(index)
    return routes


def _explicit_swap(original,blocks,obj,deadline):
    if len(blocks)!=2 or blocks[0].original_route==blocks[1].original_route:return None
    a,b=blocks
    first=copied(original)
    # Exchange the original blocks preserving the rest of both routes.
    for r,bl in sorted(((a.original_route,a),(b.original_route,b)),reverse=True):
        if first[r][bl.original_position:bl.original_position+len(bl.customers)]!=list(bl.customers):
            raise AssertionError('Swap source positions misaligned')
        del first[r][bl.original_position:bl.original_position+len(bl.customers)]
    if sum(obj.dem[x] for x in first[a.original_route])+sum(obj.dem[x] for x in b.customers)>obj.capacity:return None
    if sum(obj.dem[x] for x in first[b.original_route])+sum(obj.dem[x] for x in a.customers)>obj.capacity:return None
    best=None;value=float('inf')
    for ra in (a.customers,a.customers[::-1]):
        for rb in (b.customers,b.customers[::-1]):
            if time.perf_counter()>=deadline:return best
            cand=copied(first)
            cand[b.original_route][b.original_position:b.original_position]=ra
            cand[a.original_route][a.original_position:a.original_position]=rb
            cost=obj.cost(cand)
            if cost<value:value=cost;best=cand
    return best


def repair_section(state, obj, name, deadline=float('inf'), enable_swap=True):
    if not state.blocks:return copied(state.routes),{'strategy':'none','tried_swap':False}
    r=_greedy_block_insert(state.routes,state.blocks,obj,repair_name=name,deadline=deadline)
    best_cost=obj.cost(r) if r is not None else float('inf')
    diag={'strategy':'insert','tried_swap':False}
    if enable_swap and len(state.blocks)==2 and time.perf_counter()<deadline:
        swapped=_explicit_swap(state.previous,state.blocks,obj,deadline)
        diag['tried_swap']=True
        if swapped is not None and obj.cost(swapped)<best_cost:
            r=swapped;diag['strategy']='swap'
    if r is None or not obj.feasible(r):
        return copied(state.previous),{'strategy':'fallback','tried_swap':diag['tried_swap']}
    return r,diag
