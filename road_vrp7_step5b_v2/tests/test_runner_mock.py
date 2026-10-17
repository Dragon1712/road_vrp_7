#!/usr/bin/env python3
"""Offline compatibility smoke of ALL C0-C6 branches against fake ALNS 7 protocol.

Fake official library tests wiring only, not real-ALNS compatibility or performance.
"""
import importlib.util
from pathlib import Path
import sys
import time
import types
from dataclasses import dataclass
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'tests'))
from test_v2_units import Obj,R

@dataclass(frozen=True)
class VState:
    routes:tuple
    objective_ref:object
    removed:tuple=()
    previous:tuple|None=None
    def objective(self):return self.objective_ref.cost(self.routes)
    def get_context(self):return None

class Budget:
    def __init__(self,n,sec):self.n=n;self.seconds=sec;self.count=0;self.deadline=float('inf');self.reason='not_started';self.first=True
    def __call__(self,rng,best,cur):
        if self.first:self.first=False;self.deadline=time.perf_counter()+self.seconds
        if self.count>=self.n or time.perf_counter()>=self.deadline:
            self.reason='finished';return True
        self.count+=1;return False

class OldPartial:
    def __init__(self,obj,original):
        self.obj=obj;self.previous=tuple(tuple(x) for x in original)
        copy=[list(r) for r in original]
        ids=tuple(copy[0][1:3]);del copy[0][1:3]
        self.routes=tuple(tuple(r) for r in copy)
        self.blocks=[types.SimpleNamespace(customers=ids,original_route=0,original_position=1)]

class OldSelector:
    def __init__(self,obj,**kwargs):self.obj=obj;self.calls=0
    def destroy(self,mode,routes,rng):
        self.calls+=1
        p=OldPartial(self.obj,routes)
        return p
    def diagnostics(self):return {'calls':self.calls}

def old_repair(part,obj,name,deadline,enable_swap):
    candidate=[list(x) for x in part.previous]
    return candidate,{'strategy':'old_reinsert'}

class FakeALNS:
    def __init__(self,rng):self.rng=rng;self.ds={};self.rs={};self.step=0
    def add_destroy_operator(self,fun,name):self.ds[name]=fun
    def add_repair_operator(self,fun,name):self.rs[name]=fun
    def iterate(self,start,selector,accept,stop):
        cur=best=start
        names=list(self.ds);repairs=list(self.rs)
        while not stop(self.rng,best,cur):
            d,r=selector(self.rng,best,cur)
            # Assure branch execution in mocked RouletteWheel.
            part=self.ds[names[d]](cur,self.rng)
            cand=self.rs[repairs[r]](part,self.rng)
            val=cand.objective();current_cost=cur.objective()
            if accept(self.rng,best,cur,cand):cur=cand
            if val<best.objective():best=cand
            selector.update(cand,d,r,'score')
            self.step+=1
        return types.SimpleNamespace(best_state=best)

class RW:
    def __init__(self,**kw):self.nd=kw['num_destroy'];self.nr=kw['num_repair'];self.count=0
    def __call__(self,rng,best,cur):
        # round-robin to exercise all destroy+repair combinations
        d=self.count%self.nd;r=self.count%self.nr;self.count+=1
        return d,r
    def update(self,candidate,d_idx,r_idx,outcome):pass

class SA:
    @staticmethod
    def autofit(**kw):return lambda rng,best,current,candidate:True


def make_modules():
    road=types.ModuleType('road_vrp')
    road.Road=road.OD=road.Objective=object
    def legal(task,routes):
        dummy=Obj();assert dummy.feasible(routes)
    road.legal=legal;road.read_json=lambda p:None
    road.write_json=lambda p,data:None;road.sha256=lambda p:'checksum'
    solvers=types.ModuleType('solvers');solvers.VRPState=VState;solvers.BothBudget=Budget
    solvers.copied=lambda x:[list(r) for r in x]
    solvers.greedy=lambda task,obj,seed:([list(r) for r in R],{})
    def chosen(n,routes,obj,rng,q):
        rows=[list(r) for r in routes]
        old=rows[0].pop(-1)
        return rows,[old]
    def inserted(routes,removed,obj,method,deadline=None):
        rows=[list(r) for r in routes]
        for id in removed:rows[0].append(id)
        return rows
    solvers.choose_destroy=chosen;solvers.insert_removed=inserted
    bench=types.ModuleType('benchmark');bench.load_tasks=lambda p:None
    old=types.ModuleType('section_operators')
    old.SectionSelector=OldSelector;old.SectionState=OldPartial;old.repair_section=old_repair
    old.PathSupportProxy=object
    alns=types.ModuleType('alns');alns.ALNS=FakeALNS
    accept=types.ModuleType('alns.accept');accept.SimulatedAnnealing=SA
    select=types.ModuleType('alns.select');select.RouletteWheel=RW
    return {'road_vrp':road,'solvers':solvers,'benchmark':bench,'section_operators':old,
            'alns':alns,'alns.accept':accept,'alns.select':select}


def main():
    with patch.dict(sys.modules,make_modules()):
        spec=importlib.util.spec_from_file_location('v2_runner',HERE/'run_experiment.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        runner.check_alns=lambda:None
        for name in runner.VARIANTS:
            o=Obj('oracle')
            fake_support=types.SimpleNamespace(get=lambda i,j,t:1. if (i+j)%3 else .1)
            s=runner.solve(name,o.task,o,seed=42,wall=1.0,max_iter=50,support=fake_support,top_k=10)
            assert o.feasible(s['routes']),name
            assert s['planned_cost']<=s['initial_cost']+1e-8
            assert s['iterations']==50
            assert 'pairs' in s['audit']
            if name not in ('C0_baseline',):assert s['stats']['destroy_calls']>0,name
            if name in ('C4_time_ucb','C5_gated','C6_shuffled'):
                assert s['selector']['pair_counts'],name
            print('PASS',name,'operators',s['audit']['destroy'].keys(),'sections',s['stats']['destroy_calls'])

if __name__=='__main__':main()
