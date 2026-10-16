#!/usr/bin/env python3
"""Dependency-light synthetic Step5-A tests, plus simulated official-ALNS protocol.

Real ALNS 7.0.0 and Chicago July end-to-end MUST be tested with `smoke`
on the user's server; do not interpret these tests as real data results.
"""
from __future__ import annotations
import math
from pathlib import Path
import sys
import types
from unittest.mock import patch
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from traffic_operators import route_trace,removal_components,TrafficDestroy,TrafficSignature

class ToyOD:
    def __init__(self,n,block=False):
        self.n=n
        self.static=np.asarray([[0. if i==j else 35+abs(i-j)*23+((i*3+j*2)%7)*9
                                  for j in range(n+1)] for i in range(n+1)],dtype=float)
        self.knot_seconds=np.arange(0,4801,600,dtype=float)
        t=self.knot_seconds
        self._dynamic=np.stack([self.static*(1.+.05*np.sin(k/2. + np.arange(n+1)[:,None]/3.)
                      +.10*np.cos(k/3.+np.arange(n+1)[None,:]/4.)) for k in range(len(t))])
        self._blocked=block
        self.dyn_access=0
    @property
    def dynamic(self):
        if self._blocked:raise AssertionError('Static solver leaked future dynamic traffic')
        self.dyn_access+=1
        return self._dynamic
    def travel(self,i,j,t,planner):
        if i==j:return 0.
        if planner=='static':return float(self.static[i,j])
        assert planner=='oracle'
        # test OD.travel also accesses true oracle only where expected
        vals=self.dynamic[:,i,j]
        return float(np.interp(t,self.knot_seconds,vals))

class ToyObjective:
    def __init__(self,planner='oracle',block=False):
        n=12
        self.task={'vehicles':3,'capacity':10,'service_seconds':90,
                   'customers':[{'id':i,'demand':1} for i in range(1,n+1)]}
        self.index={i:i for i in range(1,n+1)}
        self.dem={i:1 for i in range(1,n+1)}
        self.capacity=10
        self.planner=planner
        self.od=ToyOD(n,block=block)
        self.score_calls=0
    def route_cost(self,route):
        t=0.;travel=0.;prev=0
        for cid in [*route,None]:
            dst=0 if cid is None else self.index[int(cid)]
            leg=self.od.travel(prev,dst,t,self.planner)
            t+=leg;travel+=leg
            if cid is not None:t+=self.task['service_seconds']
            prev=dst
        return float(travel)
    def cost(self,routes):
        self.score_calls+=1
        return float(sum(self.route_cost(r) for r in routes))
    def feasible(self,routes):
        ids=[x for r in routes for x in r]
        return len(routes)==3 and all(r for r in routes) and sorted(ids)==list(range(1,13))

ROUTES=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]


def test_math():
    dynamic=ToyObjective('oracle')
    seen=[]
    for rt in ROUTES:
        a,d,c=route_trace(rt,dynamic)
        assert len(a)==len(d)==len(rt)
        assert math.isclose(c,dynamic.route_cost(rt),abs_tol=1e-6)
        components=removal_components(rt,dynamic)
        for cmp in components:
            expected=c-dynamic.route_cost([x for x in rt if x!=cmp.customer])
            assert math.isclose(expected,cmp.local_gain_sec+cmp.externality_sec,abs_tol=1e-6)
            seen.append(cmp.externality_sec)
        assert abs(components[-1].externality_sec)<1e-6
    assert any(abs(v)>1e-3 for v in seen),'Synthetic dynamic costs should have nonzero propagation'
    # Key novel-mechanism regression: Stage3 Worst ranks total savings;
    # Step5 TE-WR must make a DIFFERENT choice on a constructed instance.
    te=TrafficDestroy(dynamic)('te_wr',ROUTES,np.random.default_rng(42),3)[1]
    worst=sorted([x for rt in ROUTES for x in removal_components(rt,dynamic)],
                 key=lambda r:-r.total_gain_sec)[:3]
    assert set(te)!=set(x.customer for x in worst), 'TE-WR collapsed to Stage3 Worst'
    static=ToyObjective('static',block=True)
    for rt in ROUTES:
        assert all(abs(c.externality_sec)<1e-6 for c in removal_components(rt,static))
    assert static.od.dyn_access==0
    print('PASS exact local+externality decomposition and time-independent zero externality')


def test_destroy():
    for planner in ('static','oracle'):
        obj=ToyObjective(planner,block=(planner=='static'))
        engine=TrafficDestroy(obj)
        for label in ('te_wr','te_local','at_rr','at_timeonly'):
            r,removed=engine(label,ROUTES,np.random.default_rng(42),3)
            assert len(removed)==3 and len(set(removed))==3
            assert all(len(rt)>=1 for rt in r)
            assert sorted(removed+[x for rt in r for x in rt])==list(range(1,13))
            assert all(len(rt)<=10 for rt in r)
        assert obj.od.dyn_access==0 if planner=='static' else obj.od.dyn_access>0
        if planner=='static':
            assert engine.diagnostics()['te_static_fallback_calls']==1
            # For identical RNG and a STATIC planner, AT-RR must reduce to
            # the exact same ranking as the no-traffic ETA-only control.
            alt=TrafficDestroy(obj)('at_rr',ROUTES,np.random.default_rng(55),3)[1]
            plain=TrafficDestroy(obj)('at_timeonly',ROUTES,np.random.default_rng(55),3)[1]
            assert alt==plain, 'AT-RR static leaked features or changed base weights'
        assert engine.diagnostics()['call_counts']['te_wr']==1
    try:
        TrafficDestroy(ToyObjective())('invalid',ROUTES,np.random.default_rng(1),3)
    except ValueError:pass
    else:raise AssertionError('Missing unknown-operator rejection')
    print('PASS AT-RR & TE-WR capacity/coverage invariants, static information firewall')


def test_metadata_and_engine():
    # Inject synthetic Stage3 modules for protocol test. No Chicago code is replaced
    # or modified in user's repository; fake modules exist in this local interpreter.
    fake_road=types.ModuleType('road_vrp')
    fake_road.Road=type('Road',(),{})
    fake_road.OD=ToyOD
    fake_road.Objective=ToyObjective
    fake_road.legal=lambda task,routes:assert_legal(task,routes)
    fake_road.sha256=lambda p:'fake'
    fake_road.read_json=lambda p:{}
    fake_road.write_json=lambda p,d:None
    fake_solvers=types.ModuleType('solvers')
    fake_solvers.greedy=lambda task,obj,seed:([r[:] for r in ROUTES],{'fallback_seed_routes':False})
    fake_solvers.copied=lambda routes:[list(r) for r in routes]
    fake_solvers.choose_destroy=lambda label,routes,obj,rng,q: base_destroy(label,routes,rng,q)
    fake_solvers.insert_removed=lambda partial,removed,obj,label,deadline=None: base_repair(partial,removed,obj)
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class State:
        routes:tuple
        objective_ref:object
        removed:tuple=()
        previous:object=None
        def objective(self):return self.objective_ref.cost(self.routes)
    fake_solvers.VRPState=State
    class FakeBudget:
        def __init__(self,max_iter,wall):
            self.max_iter=max_iter;self.wall=wall;self.count=0
            self.deadline=float('inf');self.reason='not_started'
        def __call__(self,rng,best,current):
            import time
            if self.deadline==float('inf'):self.deadline=time.perf_counter()+self.wall
            if self.count>=self.max_iter:self.reason='iteration_budget';return True
            if time.perf_counter()>=self.deadline:self.reason='wall_budget';return True
            self.count+=1;return False
    fake_solvers.BothBudget=FakeBudget
    fake_bench=types.ModuleType('benchmark')
    fake_bench.load_tasks=lambda p:None
    fake_bench.frozen_scenarios=lambda *a:[]
    alns=types.ModuleType('alns')
    class ALNS:
        def __init__(self,rng):self.rng=rng;self.ds=[];self.rs=[]
        def add_destroy_operator(self,fn,name):self.ds.append(fn)
        def add_repair_operator(self,fn,name):self.rs.append(fn)
        def iterate(self,start,select,accept,stop):
            current=best=start
            while not stop(self.rng,best,current):
                d=self.ds[int(self.rng.integers(len(self.ds)))](current,self.rng)
                candidate=self.rs[int(self.rng.integers(len(self.rs)))](d,self.rng)
                if candidate.objective()<best.objective():best=candidate
                if accept(self.rng,best,current,candidate):current=candidate
            return types.SimpleNamespace(best_state=best)
    alns.ALNS=ALNS
    accept=types.ModuleType('alns.accept')
    class SA:
        @staticmethod
        def autofit(**kw):return lambda rng,best,current,candidate: candidate.objective()<=current.objective()
    accept.SimulatedAnnealing=SA
    select=types.ModuleType('alns.select')
    select.RouletteWheel=lambda **kw:object()
    with patch.dict(sys.modules,{'road_vrp':fake_road,'solvers':fake_solvers,
                                 'benchmark':fake_bench,'alns':alns,
                                 'alns.accept':accept,'alns.select':select}):
        import run_experiment
        with patch('run_experiment.assert_alns',return_value='7.0.0'):
            for variant in run_experiment.VARIANTS:
                obj=ToyObjective('oracle')
                result=run_experiment.solve_alns(variant,obj.task,obj,42,.02,40)
                assert_legal(obj.task,result['routes'])
                assert result['planned_cost']<=result['initial_cost']+1e-6
                assert result['iterations']<=40

        # Synthetic end-to-end analysis + independent Stage3 configuration audit.
        import csv,json,tempfile
        import analyze_step5a as analyzer
        analyzer.read_json=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
        analyzer.write_json=lambda p,d:Path(p).write_text(json.dumps(d),encoding='utf-8')
        analyze=analyzer.analyze
        with tempfile.TemporaryDirectory() as td:
            t=Path(td);s5=t/'step5';s3=t/'stage3';s5.mkdir();s3.mkdir()
            cfg={'split':'validation','variants':list(run_experiment.VARIANTS),
                 'planners':['static','oracle'],'seeds':[42],
                 'task_sha256':{'task_0000.json':'taskdigest'},
                 'scenario_ids':['2025-07-22_0800'],
                 'cost_file_sha256':'costdigest','edge_file_sha256':'edgedigest',
                 'tasks_manifest_sha256':'manifestdigest','wall_sec':5.0,
                 'max_iterations':5000,'od_step_min':20,'od_hours':5.0,
                 'expected_count':len(run_experiment.VARIANTS)*2}
            (s5/'run_config.json').write_text(json.dumps(cfg))
            (s5/'benchmark_summary.json').write_text(json.dumps({'status':'COMPLETE'}))
            cols=['task_id','scenario_id','solver','planner','seed','split','eligible',
                  'ex_post_road_travel_sec','search_wall_sec','iterations','score_calls',
                  'observed_edge_share','failed_repairs','te_nonzero_externality_calls',
                  'te_static_fallback_calls','extra_traffic_route_cost_calls']
            data=[];solutions=[]
            for planner in cfg['planners']:
                for i,variant in enumerate(cfg['variants']):
                    rec=dict(task_id='task_0000',scenario_id='2025-07-22_0800',
                             solver=variant,planner=planner,seed=42,split='validation',eligible=True,
                             ex_post_road_travel_sec=4000.-float(i),search_wall_sec=5.,
                             iterations=120,score_calls=500,observed_edge_share=.11,
                             failed_repairs=0,te_nonzero_externality_calls=2,
                             te_static_fallback_calls=0,extra_traffic_route_cost_calls=10)
                    data.append(rec)
                    solutions.append({k:rec[k] for k in run_experiment.KEY})
            with (s5/'metrics.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(data)
            (s5/'solutions.jsonl').write_text('\n'.join(json.dumps(x) for x in solutions)+'\n')
            basecfg={'split':'validation','cost_file_sha256':'costdigest','edge_file_sha256':'edgedigest',
                     'tasks_manifest_sha256':'manifestdigest','task_sha256':cfg['task_sha256'],
                     'wall_sec':5.,'iterations':5000,'od_step_min':20,'od_hours':5.}
            (s3/'run_config.json').write_text(json.dumps(basecfg))
            old=[dict(x,solver='official_alns') for x in data if x['solver']=='baseline']
            with (s3/'metrics.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(old)
            result=analyze(s5,t/'analysis',s3)
            assert result['experiment_complete']
            assert (t/'analysis'/'paired_gains.csv').exists()
            # Budget mismatch must not be silently reported as matched Stage3.
            basecfg['wall_sec']=1.
            (s3/'run_config.json').write_text(json.dumps(basecfg))
            try:analyze(s5,t/'mismatched',s3)
            except ValueError as e: assert 'configuration differ' in str(e)
            else:raise AssertionError('Stage3 budget audit failed')
    print('PASS strict paired analysis, full grid and mismatch rejection')

    print('PASS all 7 variants via synthetic ALNS API (not the real library)')


def assert_legal(task,routes):
    ids=sorted(x for rt in routes for x in rt)
    assert ids==list(range(1,13))
    assert len(routes)==3 and all(rt for rt in routes)
    assert all(len(rt)<=task['capacity'] for rt in routes)


def base_destroy(label,routes,rng,q):
    work=[list(r) for r in routes]
    pool=[x for rt in work for x in rt]
    if label=='random':order=list(rng.permutation(pool))
    else:order=pool
    removed=[]
    for x in order:
        if len(removed)>=q:break
        for rt in work:
            if x in rt and len(rt)>1:
                rt.remove(x);removed.append(x);break
    return work,removed


def base_repair(partial,removed,obj):
    routes=[list(r) for r in partial]
    for x in removed:
        r=min(routes,key=len)
        r.append(x)
    return routes

if __name__=='__main__':
    test_math();test_destroy();test_metadata_and_engine()
    print('STEP5-A SYNTHETIC TESTS COMPLETE. Run real `smoke` on alns==7.0.0 server.')
