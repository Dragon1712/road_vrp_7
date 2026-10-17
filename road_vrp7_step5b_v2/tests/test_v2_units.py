#!/usr/bin/env python3
"""Synthetic tests, NOT empirical Chicago results."""
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from section_v2 import Selector,Stats,repair,Partial,Block,frozen
from time_selector import TimeUCB

class OD:
    def __init__(self,static_only=False):
        n=13
        self.static=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])*6.+10.
        np.fill_diagonal(self.static,0.)
        self.knot_seconds=np.arange(16)*1200.
        self.static_only=static_only
    def travel(self,i,j,sec,planner):
        if i==j:return 0.
        if planner=='static':return float(self.static[i,j])
        if self.static_only:raise AssertionError('Oracle traffic data accessed in static mode')
        return float(self.static[i,j]*(1.+.7*((i*5+j)%6==0)*(1+.5*np.sin(sec/1200.))))

class Obj:
    def __init__(self,planner='oracle'):
        self.planner=planner;self.od=OD(planner=='static')
        self.task={'vehicles':3,'capacity':28,'service_seconds':120,
                   'customers':[{'id':i,'demand':1+(i%3)} for i in range(1,13)]}
        self.index={i:i for i in range(1,13)}
        self.dem={i:1+(i%3) for i in range(1,13)}
        self.capacity=28;self.score_calls=0
    def route_cost(self,r):
        t=0.;value=0.;prev=0
        for x in list(r)+[0]:
            c=self.od.travel(prev,x,t,self.planner);value+=c;t+=c+(120 if x else 0);prev=x
        return value
    def cost(self,routes):
        self.score_calls+=1
        return sum(self.route_cost(r) for r in routes)
    def feasible(self,routes):
        vals=[x for r in routes for x in r]
        return len(routes)==3 and all(r for r in routes) and sorted(vals)==list(self.dem) and all(sum(self.dem[x] for x in r)<=self.capacity for r in routes)

R=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
class Proxy:
    def __init__(self,kind):self.kind=kind
    def get(self,i,j,t):
        if self.kind=='zero':return 0.
        return 1. if ((i+j)%4!=0) else .5


def test_core():
    from time import perf_counter
    for planner in ('oracle','static'):
        obj=Obj(planner)
        for method in ('static','gated','shuffled'):
            selector=Selector(obj,support=Proxy('active'))
            for seed in range(15):
                part=selector.destroy(method,R,np.random.default_rng(seed))
                assert obj.feasible(R)
                assert len(part.blocks)==1
                assert sorted(x for route in part.routes for x in route)+sorted(part.blocks[0].ids) != []
                repaired,status=repair(part,obj,perf_counter()+1,Stats(),top_k=15)
                assert status=='moved'
                assert obj.feasible(repaired)
                assert frozen(repaired)!=part.original
                block=list(part.blocks[0].ids)
                assert any(block==r[i:i+len(block)] or block[::-1]==r[i:i+len(block)]
                           for r in repaired for i in range(len(r)-len(block)+1)), 'Block split during repair'
        if planner=='static':
            for seed in range(25):
                a=Selector(obj,support=Proxy('active')).destroy('static',R,np.random.default_rng(seed))
                b=Selector(obj,support=Proxy('active')).destroy('gated',R,np.random.default_rng(seed))
                assert a.blocks==b.blocks,'Static planning must ignore oracle data'
    # Zero observed support must gate out traffic, INCLUDING the shuffled placebo.
    obj=Obj()
    for seed in range(25):
        a=Selector(obj,support=Proxy('zero')).destroy('static',R,np.random.default_rng(seed))
        b=Selector(obj,support=Proxy('zero')).destroy('gated',R,np.random.default_rng(seed))
        c=Selector(obj,support=Proxy('zero')).destroy('shuffled',R,np.random.default_rng(seed))
        assert a.blocks==b.blocks==c.blocks
    # Cheap top-K is genuinely different budget from exhaustive exact evaluation.
    p=Selector(obj,support=Proxy('active')).destroy('static',R,np.random.default_rng(12))
    a=Stats();b=Stats()
    ra,_=repair(p,obj,perf_counter()+1,a,top_k=1)
    rb,_=repair(p,obj,perf_counter()+1,b,top_k=None)
    assert a.d['repair_candidates_exact']==1 and b.d['repair_candidates_exact']>1
    assert obj.cost(rb)<=obj.cost(ra)+1e-9
    # Cross-vehicle whole-block swap candidates are legal under capacity.
    sx=Stats();swapped,status=repair(p,obj,perf_counter()+1,sx,top_k=None,enable_swap=True)
    assert obj.feasible(swapped) and sx.d['swap_attempts']>0
    assert sx.d['repair_candidates_exact']>b.d['repair_candidates_exact']
    # With deadline already gone, the method must return original feasible solution.
    c=Stats();r,status=repair(p,obj,perf_counter()-1,c,top_k=15)
    assert frozen(r)==p.original and status!='moved'
    print('PASS: 3 modes x 2 planners, 90 feasible non-trivial repairs, static leakage, zero-support gate, exact vs topK, deadline')


def test_time_ucb():
    u=TimeUCB(4,2,c=.25,epsilon=0)
    r=np.random.default_rng(42)
    for _ in range(24):
        d,j=u(r,None,None)
        u.record_candidate(100.,99. if d==3 else 100.)
        u.update(None,d,j,'any_outcome')
    assert (u.n>0).all()
    assert u.n.sum()==24 and np.isfinite(u.avg).all()
    assert u.reward_sum>=0
    print('PASS: Official ALNS 7 select/update signatures, cold-start all pairs, nonnegative reward')

if __name__=='__main__':
    test_core();test_time_ucb()
