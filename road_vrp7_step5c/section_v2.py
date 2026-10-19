#!/usr/bin/env python3
"""Section v2: strictly feasible block actions, top-K exact time-dependent scoring.

No implicit static symmetric/Euclidean shortcut is used as a FINAL scorer.
The cheap static scorer is only a ranking proxy for exact finalists.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
import numpy as np


def frozen(routes):return tuple(tuple(int(x) for x in r) for r in routes)
def mutable(routes):return [list(r) for r in routes]


def norm(values):
    v=np.asarray(values,float)
    if not len(v):return v
    if not np.isfinite(v).all():raise ValueError('Nonfinite section feature')
    a,b=np.quantile(v,[.1,.9])
    if b-a<1e-12:return np.zeros_like(v)
    return np.clip((v-a)/(b-a),0,1)


@dataclass(frozen=True)
class Block:
    ids:tuple[int,...]
    route:int
    pos:int


@dataclass(frozen=True)
class Partial:
    routes:tuple[tuple[int,...],...]
    obj:object
    blocks:tuple[Block,...]
    original:tuple[tuple[int,...],...]
    def objective(self):raise RuntimeError('Cannot score Section partial state')
    def get_context(self):return None


class Stats:
    def __init__(self):
        self.d={k:0 for k in ('destroy_calls','candidate_count','traffic_change','traffic_comparisons',
            'actual_choice_changed','actual_choice_compared','selected_support_sum','selected_support_n',
            'repair_calls','repair_candidates_screened','repair_candidates_exact','repair_completed',
            'repair_noop_rejected','repair_fallback','repair_deadline','swap_attempts','swap_selected',
            'changed','unchanged','improving','worsening','tied','accepted','accepted_changed','new_best')}
        self.d['destroy_sec']=0.;self.d['repair_sec']=0.
    def add(self,key,n=1):self.d[key]+=n
    def summary(self):
        x=dict(self.d)
        x['traffic_change_rate']=x['traffic_change']/max(x['traffic_comparisons'],1)
        x['actual_choice_change_rate']=x['actual_choice_changed']/max(x['actual_choice_compared'],1)
        x['avg_selected_support']=x['selected_support_sum']/max(x['selected_support_n'],1)
        x['mean_repair_ms']=1000*x['repair_sec']/max(x['repair_calls'],1)
        return x


class Selector:
    """Fixed candidate enumeration. Dynamic score gated by measured path support.

    Invariant: support=0 implies static rank and no future data is read in
    static planning mode. ``shuffled`` permutes the gated traffic increment
    between candidates but preserves its multiset.
    """
    def __init__(self,obj,*,support=None,traffic_weight=.45,block_sizes=(2,3),top_fraction=.20):
        self.obj=obj;self.support=support;self.weight=float(traffic_weight)
        self.sizes=tuple(block_sizes);self.top_fraction=float(top_fraction)
        if not 0<=self.weight<=1:raise ValueError('traffic weight must be in [0,1]')
        if not self.sizes or any(z<2 or z>5 for z in self.sizes):raise ValueError('invalid block sizes')
        if not 0<self.top_fraction<=1:raise ValueError('invalid top fraction')
        self.stats=Stats()
    def candidates(self,routes,traffic=False):
        o=self.obj;out=[]
        for route,r in enumerate(routes):
            for size in self.sizes:
                if len(r)-size<1:continue
                for pos in range(len(r)-size+1):
                    prev=0 if pos==0 else o.index[r[pos-1]]
                    ids=list(r[pos:pos+size]);next_i=0 if pos+size==len(r) else o.index[r[pos+size]]
                    seq=[prev]+[o.index[x] for x in ids]+[next_i]
                    stat=sum(float(o.od.static[i,j]) for i,j in zip(seq[:-1],seq[1:]))
                    tv=0.;sup=0.
                    if traffic and o.planner=='oracle' and self.support is not None:
                        # Exact start-of-edge times under the current route. Avoid using
                        # dynamic graph beyond the actual Section arc traversal.
                        t=0.;last=0
                        for cid in r[:pos]:
                            nxt=o.index[cid];t+=o.od.travel(last,nxt,t,'oracle')
                            t+=float(o.task['service_seconds']);last=nxt
                        deltas=[];supports=[]
                        for index,(i,j) in enumerate(zip(seq[:-1],seq[1:])):
                            c=float(o.od.travel(i,j,t,'oracle'))
                            ref=float(o.od.static[i,j])
                            if ref>1e-9:
                                future=float(o.od.travel(i,j,min(t+1200.,o.od.knot_seconds[-1]),'oracle'))
                                # congestion direction matters: only positive relative
                                # increase is rewarded. Trend is magnitude only.
                                deltas.append(min(3.,max(0.,(c-ref)/ref)) +
                                              .25*min(3.,abs(future-c)/ref))
                            supports.append(float(self.support.get(i,j,t)))
                            t+=c
                            if index<size:t+=float(o.task['service_seconds'])
                        tv=float(np.mean(deltas)) if deltas else 0.
                        sup=float(np.mean(supports)) if supports else 0.
                    out.append({'route':route,'pos':pos,'size':size,'static':stat,
                                'traffic':tv,'support':sup})
        return out
    def destroy(self,mode,routes,rng):
        t0=time.perf_counter();o=self.obj
        self.stats.add('destroy_calls')
        if mode not in ('static','gated','shuffled'):
            raise ValueError('Invalid selection mode: '+str(mode))
        original=frozen(routes)
        active=mode!='static' and o.planner=='oracle' and self.support is not None
        cs=self.candidates(routes,traffic=active)
        self.stats.add('candidate_count',len(cs))
        if not cs:
            self.stats.add('destroy_sec',time.perf_counter()-t0)
            return Partial(original,o,(),original)
        stat=norm([c['static'] for c in cs]);traf=norm([c['traffic'] for c in cs])
        gates=np.clip(np.array([c['support'] for c in cs],float),0,1)
        # Shuffle only TRAFFIC features across observed candidates; retain the
        # original Section's own reliability gate and static score.
        if mode=='shuffled' and active:
            eligible=np.flatnonzero(gates>0)
            shuffled=traf.copy()
            shuffled[eligible]=rng.permutation(traf[eligible])
            traf=shuffled
        inc=self.weight*gates*(traf-stat) if active else np.zeros(len(cs))
        # Support 0 -> score=static exactly, including the negative control.
        score=stat+inc
        # Use shared jitter for self-comparison (same RNG draw); dynamic choice
        # is defined as actual sampled move, not only a different top-ranked id.
        jitter=rng.uniform(0,1e-10,len(cs))
        reference=list(np.argsort(-(stat+jitter)))
        rank=list(np.argsort(-(score+jitter)))
        k=max(1,int(math.ceil(len(cs)*self.top_fraction)))
        pick=int(rng.integers(k))
        self.stats.add('traffic_comparisons',int(active))
        self.stats.add('traffic_change',int(active and rank[0]!=reference[0]))
        self.stats.add('actual_choice_compared',int(active))
        self.stats.add('actual_choice_changed',int(active and rank[pick]!=reference[pick]))
        c=cs[rank[pick]]
        self.stats.add('selected_support_sum',c['support']);self.stats.add('selected_support_n')
        block=Block(tuple(routes[c['route']][c['pos']:c['pos']+c['size']]),c['route'],c['pos'])
        p=mutable(original)
        del p[block.route][block.pos:block.pos+len(block.ids)]
        assert p[block.route], 'Exact K route emptied'
        self.stats.add('destroy_sec',time.perf_counter()-t0)
        return Partial(frozen(p),o,(block,),original)


def _static_route_cost(obj,r):
    prev=0;total=0.
    for x in r:
        j=obj.index[x];total+=float(obj.od.static[prev,j]);prev=j
    return total+float(obj.od.static[prev,0])


def _block_options(partial,block,obj,deadline,stats,nontrivial=True):
    """Generate capacity-feasible WHOLE-block insertions. Static cost = prefilter only."""
    routes=mutable(partial.routes);ids=block.ids
    demand=sum(obj.dem[x] for x in ids)
    result=[];visited=set()
    for k,r in enumerate(routes):
        if time.perf_counter()>=deadline:break
        if sum(obj.dem[x] for x in r)+demand>obj.capacity:continue
        old_static=_static_route_cost(obj,r)
        for orientation in (ids,ids[::-1]):
            for pos in range(len(r)+1):
                new=mutable(routes);new[k][pos:pos]=orientation
                signature=frozen(new)
                if signature in visited:continue
                visited.add(signature)
                if nontrivial and signature==partial.original:
                    stats.add('repair_noop_rejected');continue
                # Note: single Block means every candidate is a legal complete solution.
                proxy=_static_route_cost(obj,new[k])-old_static
                result.append((proxy,k,pos,tuple(orientation),signature,'relocate'))
                stats.add('repair_candidates_screened')
    result.sort(key=lambda p:(p[0],p[1],p[2],p[3]))
    return result


def _swap_options(partial,block,obj,deadline,stats):
    """Whole-block swap with a contiguous target sequence, preserving vehicle count."""
    originals=mutable(partial.original)
    ra=block.route;length=len(block.ids)
    olda=_static_route_cost(obj,originals[ra])
    moves=[];seen=set()
    a_dem=sum(obj.dem[x] for x in block.ids)
    for rb,route in enumerate(originals):
        if rb==ra:continue
        oldb=_static_route_cost(obj,route)
        # Avoid too-large pair explosion; target lengths 2..3, including reverse.
        for size in (2,3):
            for pos in range(len(route)-size+1):
                if time.perf_counter()>=deadline:return moves
                ids=route[pos:pos+size]
                b_dem=sum(obj.dem[x] for x in ids)
                if sum(obj.dem[x] for x in originals[ra])-a_dem+b_dem>obj.capacity:continue
                if sum(obj.dem[x] for x in route)-b_dem+a_dem>obj.capacity:continue
                for rev_a in (False,True):
                    for rev_b in (False,True):
                        out=mutable(originals)
                        out[ra][block.pos:block.pos+length]=list(reversed(ids)) if rev_b else ids
                        out[rb][pos:pos+size]=list(reversed(block.ids)) if rev_a else list(block.ids)
                        sig=frozen(out)
                        if sig in seen or sig==partial.original:continue
                        seen.add(sig)
                        delta=_static_route_cost(obj,out[ra])+_static_route_cost(obj,out[rb])-olda-oldb
                        moves.append((delta,rb,pos,tuple(ids),sig,'swap'))
                        stats.add('swap_attempts');stats.add('repair_candidates_screened')
    return moves


def repair(partial,obj,deadline,stats,top_k=12,nontrivial=True,enable_swap=False):
    """Whole-block relocations/reversals and optional two-route block swap.

    Every final candidate is evaluated by Objective.cost, which propagates
    directed travel times and service durations. The static proxy only sorts.
    """
    t0=time.perf_counter();stats.add('repair_calls')
    if len(partial.blocks)!=1:
        if partial.blocks:raise AssertionError('V2 requires exactly one selected Section')
        stats.add('repair_fallback');stats.add('repair_sec',time.perf_counter()-t0)
        return mutable(partial.original),'no_section'
    opts=_block_options(partial,partial.blocks[0],obj,deadline,stats,nontrivial=nontrivial)
    if enable_swap and time.perf_counter()<deadline:
        opts.extend(_swap_options(partial,partial.blocks[0],obj,deadline,stats))
        opts.sort(key=lambda p:(p[0],p[1],p[2],p[3]))
    if not opts:
        stats.add('repair_fallback');stats.add('repair_sec',time.perf_counter()-t0)
        return mutable(partial.original),'no_feasible_nontrivial'
    best=None;bestcost=float('inf');best_kind='relocate'
    for _,k,pos,orientation,signature,kind in (opts if top_k is None else opts[:int(top_k)]):
        if time.perf_counter()>=deadline:
            stats.add('repair_deadline');break
        candidate=mutable(signature)
        if not obj.feasible(candidate):raise AssertionError('Illegal completed candidate')
        # Objective fully propagates arrival time along directed OD cost!
        val=float(obj.cost(candidate));stats.add('repair_candidates_exact')
        if val<bestcost-1e-9:
            bestcost=val;best=candidate;best_kind=kind
    if best is None:
        stats.add('repair_fallback');stats.add('repair_sec',time.perf_counter()-t0)
        return mutable(partial.original),'deadline_fallback'
    assert frozen(best)!=partial.original or not nontrivial
    stats.add('repair_completed');stats.add('swap_selected',int(best_kind=='swap'));stats.add('repair_sec',time.perf_counter()-t0)
    return best,'moved'
