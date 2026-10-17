#!/usr/bin/env python3
"""Offline Step5-B synthetic regression; actual official ALNS validated via server smoke.

Does NOT invent results on real July city graph. Stubs only Stage3/ALNS interfaces.
"""
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import types
from unittest.mock import patch
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from section_operators import (SectionSelector,PathSupportProxy,repair_section,
                               feasible_partial,SectionState,Block)

class ToyOD:
    def __init__(self,allow_dynamic=True):
        self.n=13
        ids=np.arange(self.n)
        self.static=np.abs(ids[:,None]-ids[None,:]).astype(float)*7+14
        np.fill_diagonal(self.static,0)
        self.knot_seconds=np.arange(16)*1200.
        self.allow_dynamic=allow_dynamic
        if allow_dynamic:
            self.dynamic=np.empty((len(self.knot_seconds),self.n,self.n),dtype=float)
            for k in range(len(self.knot_seconds)):
                factor=np.ones((self.n,self.n))
                for i in range(self.n):
                    factor[i,:]*=1. + (i%3)*.35+ .7*(i%4==0)*abs(np.sin(k*.7))
                self.dynamic[k]=self.static*factor
        else:self.dynamic=None
    def travel(self,i,j,t,planner):
        if i==j:return 0.
        if planner=='static':return float(self.static[i,j])
        if not self.allow_dynamic:raise AssertionError('Static planner tried to access future')
        k=max(0,min(int(t//1200),len(self.knot_seconds)-2))
        a=min(1.,max(0.,(t-self.knot_seconds[k])/1200))
        return float((1-a)*self.dynamic[k,i,j]+a*self.dynamic[k+1,i,j])

class ToyObj:
    def __init__(self,planner='oracle'):
        self.od=ToyOD(planner=='oracle');self.planner=planner
        self.index={i:i for i in range(1,13)}
        self.dem={i:(i%3)+1 for i in range(1,13)}
        self.task={'vehicles':3,'capacity':22,'service_seconds':120.,
                   'customers':[{'id':i,'demand':self.dem[i]} for i in range(1,13)]}
        self.capacity=22;self.score_calls=0
    def route_cost(self,route):
        t=0.;c=0.;prev=0
        for dest in list(route)+[0]:
            v=self.od.travel(prev,dest,t,self.planner)
            t+=v;c+=v;prev=dest
            if dest!=0:t+=120
        return c
    def cost(self,routes):
        self.score_calls+=1
        return sum(self.route_cost(r) for r in routes)
    def feasible(self,routes):
        items=[x for r in routes for x in r]
        return (len(routes)==3 and all(r for r in routes) and sorted(items)==list(self.dem)
                and all(sum(self.dem[x] for x in r)<=self.capacity for r in routes))

ROUTES=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]


def test_sections():
    for planner in ('oracle','static'):
        obj=ToyObj(planner)
        for name in ('section_random','section_static','section_static_probe','section_traffic','section_shuffled'):
            selector=SectionSelector(obj,traffic_weight=.8,support_weight=.1)
            for seed in (42,43,44,45,46):
                state=selector.destroy(name,ROUTES,np.random.default_rng(seed))
                assert isinstance(state,SectionState)
                assert len(state.blocks) in (1,2)
                assert all(len(b.customers) in (2,3) for b in state.blocks)
                assert feasible_partial(obj.task,state.routes,state.blocks)
                repaired,diag=repair_section(state,obj,'greedy',deadline=time.perf_counter()+2)
                assert obj.feasible(repaired),f'{planner} {name} {seed}'
                for b in state.blocks:
                    seq=list(b.customers)
                    assert any(seq==r[j:j+len(seq)] or seq[::-1]==r[j:j+len(seq)]
                               for r in repaired for j in range(len(r)-len(seq)+1)), 'Section was split'
            if planner=='static':
                assert selector.diagnostics()['mean_traffic_score']==0.
                assert selector.diagnostics()['traffic_compared']==0
        # explicit-block-exchange feasible / schema check
        selector=SectionSelector(obj,section_probability=1)
        s=selector.destroy('section_static',ROUTES,np.random.default_rng(13))
        assert len(s.blocks)==2
        repaired,diag=repair_section(s,obj,'regret2',deadline=time.perf_counter()+2)
        assert obj.feasible(repaired)


def test_traffic_changes_selection():
    obj=ToyObj('oracle')
    # Cost-matched probe must preserve the static decision with the same RNG.
    for seed in range(10):
        s=SectionSelector(obj).destroy('section_static',ROUTES,np.random.default_rng(seed))
        p=SectionSelector(obj).destroy('section_static_probe',ROUTES,np.random.default_rng(seed))
        assert s.blocks==p.blocks, 'Probe control changed static ranking'
    for seed in range(10):
        static_obj=ToyObj('static')
        s=SectionSelector(static_obj).destroy('section_static',ROUTES,np.random.default_rng(seed))
        t=SectionSelector(static_obj).destroy('section_traffic',ROUTES,np.random.default_rng(seed))
        assert s.blocks==t.blocks, 'Static mode touched dynamic selection'
    changed=0
    for seed in range(50):
        a=SectionSelector(obj,traffic_weight=.85)
        b=SectionSelector(obj,traffic_weight=.85)
        s=a.destroy('section_traffic',ROUTES,np.random.default_rng(seed))
        t=b.destroy('section_static',ROUTES,np.random.default_rng(seed))
        changed+=int(s.blocks!=t.blocks)
    assert changed>0, 'Strong dynamic cost must influence Section decisions'
    print(f'PASS traffic-selected blocks differ from static in {changed}/50 independent synthetic seeds')


def test_path_support():
    obj=ToyObj('oracle');n=obj.od.n
    class ToyRoad:
        edge_file_hash='abc';cost_file_hash='def';dt=600
        times=list(range(200));start=0
        def __init__(self):
            self.edges=pd.DataFrame({'length_m':np.ones(n*n)*100})
            self.observed=np.zeros((len(self.times),n*n),bool)
            self.observed[:,::2]=True
        def sssp(self,u,targets,depart,mode):
            u=int(u);targs=[int(x) for x in targets]
            d={str(v):100. for v in targs};d[str(u)]=depart
            back={str(v):(str(u),u*n+v) for v in targs if v!=u}
            return d,back
    road=ToyRoad()
    # toy node IDs match dense graph, static shortest path is a single-edge path
    od=types.SimpleNamespace(nodes=[str(i) for i in range(n)],knot_seconds=obj.od.knot_seconds,start=0.,n=n)
    with tempfile.TemporaryDirectory() as tmp:
        x=PathSupportProxy(road,od,tmp)
        y=PathSupportProxy(road,od,tmp)
        assert y.cached and not x.cached
        assert x.support.shape==(len(od.knot_seconds),n,n)
        assert (x.support>=0).all() and (x.support<=1).all()
        assert x.get(0,1,0) in (0.,1.)


def test_official_interface_with_fake():
    # We do NOT claim this substitutes for an official ALNS smoke test.
    source=types.ModuleType('road_vrp')
    def legal(task,routes):
        obj=ToyObj();assert obj.feasible(routes)
    source.legal=legal;source.Road=object;source.OD=object;source.Objective=object
    source.read_json=lambda p:json.loads(Path(p).read_text())
    source.write_json=lambda p,o:Path(p).write_text(json.dumps(o))
    source.sha256=lambda p:'dummy'
    benchmark=types.ModuleType('benchmark');benchmark.load_tasks=lambda p:([],[],{})
    solvers=types.ModuleType('solvers')
    solvers.copied=lambda r:[list(x) for x in r]
    solvers.greedy=lambda task,obj,seed:([list(x) for x in ROUTES],{'fallback_seed_routes':False})
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class VRPState:
        routes:tuple
        objective_ref:object
        removed:tuple=()
        previous:tuple|None=None
        def objective(self):return self.objective_ref.cost(self.routes)
    class BothBudget:
        def __init__(self,iters,wall):self.max_iter=iters;self.wall=wall;self.count=0;self.start=None;self.deadline=float('inf')
        def __call__(self,rng,best,current):
            if self.start is None:self.start=time.perf_counter();self.deadline=self.start+self.wall
            if self.count>=self.max_iter or time.perf_counter()>=self.deadline:return True
            self.count+=1;return False
    def choose_destroy(name,routes,obj,rng,q):
        partial=solvers.copied(routes)
        removed=[]
        for rt in partial:
            if len(rt)>1 and len(removed)<q:removed.append(rt.pop())
        return partial,removed
    def insert_removed(partial,removed,obj,name,deadline=float('inf')):
        x=solvers.copied(partial)
        for value in removed:x[0].append(value)
        if not obj.feasible(x):return None
        return x
    solvers.VRPState=VRPState;solvers.BothBudget=BothBudget
    solvers.choose_destroy=choose_destroy;solvers.insert_removed=insert_removed

    alns=types.ModuleType('alns')
    class FakeALNS:
        def __init__(self,rng):self.rng=rng;self.ds={};self.rs={}
        def add_destroy_operator(self,f,name):self.ds[name]=f
        def add_repair_operator(self,f,name):self.rs[name]=f
        def iterate(self,start,selector,accept,stop):
            current=best=start
            while not stop(self.rng,best,current):
                dname=list(self.ds)[stop.count%len(self.ds)]
                partial=self.ds[dname](current,self.rng)
                cand=self.rs['greedy'](partial,self.rng)
                if cand.objective()<best.objective():best=cand
                if accept(self.rng,best,current,cand):current=cand
            return types.SimpleNamespace(best_state=best)
    class SA:
        @staticmethod
        def autofit(**kwargs):return lambda rng,best,current,candidate:True
    class Roulette:
        def __init__(self,**kwargs):pass
    alns.ALNS=FakeALNS
    accept=types.ModuleType('alns.accept');accept.SimulatedAnnealing=SA
    select=types.ModuleType('alns.select');select.RouletteWheel=Roulette
    traffic=types.ModuleType('traffic_operators')
    class FakeTrafficDestroy:
        def __init__(self,obj):self.obj=obj
        def __call__(self,name,routes,rng,q):return choose_destroy('random',routes,self.obj,rng,q)
        def diagnostics(self):return {'calls':0}
    traffic.TrafficDestroy=FakeTrafficDestroy
    with patch.dict(sys.modules,{'road_vrp':source,'benchmark':benchmark,'solvers':solvers,
                                 'alns':alns,'alns.accept':accept,'alns.select':select,'traffic_operators':traffic}):
        import importlib.util
        spec=importlib.util.spec_from_file_location('stage5b_run',HERE/'run_experiment.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        for name in m.VARIANTS:
            obj=ToyObj('oracle')
            r=m.solve(name,obj.task,obj,42,.08,40,support=None)
            assert obj.feasible(r['routes'])
            assert r['search_wall_sec']>=0
            assert 'destroy' in r['audit'] and 'pairs' in r['audit']
        print('PASS 7/7 Section ALNS arms with fake ALNS API; real alns==7.0.0 NOT tested locally')


def main():
    test_sections()
    test_traffic_changes_selection()
    test_path_support()
    test_official_interface_with_fake()
    print('PASS Stage5-B synthetic Section block integrity, capacity, dynamic/static isolation, support-cache and fake API')
if __name__=='__main__':main()
