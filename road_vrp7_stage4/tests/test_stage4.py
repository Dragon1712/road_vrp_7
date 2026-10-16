#!/usr/bin/env python3
"""Stage4 synthetic smoke + strict grid; official ALNS replaced by fake in this test.
Real library MUST be verified separately on the user's alns==7.0.0 server.
"""
import json
import math
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'road_vrp7_stage3'))
sys.path.insert(0,str(ROOT/'road_vrp7_stage4'))
import importlib.util
_src=ROOT/'road_vrp7_stage3'/'tests'/'test_stage3.py'
_spec=importlib.util.spec_from_file_location('stage3_fixture',_src)
_stage3_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_stage3_fixture)
fixture=_stage3_fixture.fixture
from road_vrp import Road,OD,Objective,legal,write_json,sha256
from tasks import make_tasks
from budget_experiment import WallTimeAnneal,time_temperature,solve_variant,run,read_checkpoint,KEY
from analyze_budget import analyze
from argparse import Namespace


class FakeEngine:
    def __init__(self,rng):self.rng=rng;self.destroy={};self.repair={}
    def add_destroy_operator(self,fn,name):self.destroy[name]=fn
    def add_repair_operator(self,fn,name):self.repair[name]=fn
    def iterate(self,start,select,accept,stop):
        current=best=start
        while not stop(self.rng,best,current):
            partial=self.destroy['random'](current,self.rng)
            candidate=self.repair['greedy'](partial,self.rng)
            if candidate.objective()<best.objective():best=candidate
            if accept(self.rng,best,current,candidate):current=candidate
        return types.SimpleNamespace(best_state=best)

class FakeRoulette:
    def __init__(self,**kw):pass


def check():
    a=time_temperature(100.,0.,5.)
    b=time_temperature(100.,2.5,5.)
    c=time_temperature(100.,5.,5.)
    assert a>b>c>0 and np.isclose(c/a,.005)
    assert math.isclose(math.exp(-(100*.05)/a),.75,rel_tol=1e-9)
    class Toy:
        def __init__(self,v):self.v=v
        def objective(self):return self.v
    rng=np.random.default_rng(1)
    sa=WallTimeAnneal(initial_obj=100,wall_sec=.1,start_sec=0)
    assert sa(rng,Toy(1),Toy(10),Toy(5))
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        cost,qc=fixture(root)
        road=Road(cost)
        task_dir=root/'tasks'
        make_tasks(road,scenario_quality=qc,output_dir=task_dir,num_tasks=10,seed=7)
        task=json.loads((task_dir/'task_0000.json').read_text())
        sc=next(s for s in task['scenarios'] if s['scenario_id']=='2025-07-22_0800')
        od=OD(road,task,sc,step_min=20,hours=5,cache_dir=root/'cache')
        for plan in ('static','oracle'):
            obj=Objective(task,od,plan)
            for variant in ('sa_timecool','vnd_multistart'):
                res=solve_variant(variant,task,obj,42,.035,500,.005)
                legal(task,res['routes'])
                assert res['planned_cost']<=res['initial_cost']+1e-6
                assert res['search_wall_sec']>=0
        alns=types.ModuleType('alns');alns.ALNS=FakeEngine
        selection=types.ModuleType('alns.select');selection.RouletteWheel=FakeRoulette
        with patch.dict(sys.modules,{'alns':alns,'alns.select':selection}):
            import importlib.metadata
            original=importlib.metadata.version
            with patch('importlib.metadata.version',side_effect=lambda package:
                       '7.0.0' if package=='alns' else original(package)):
                obj=Objective(task,od,'oracle')
                r=solve_variant('alns_timecool',task,obj,42,.03,100,.005)
                legal(task,r['routes'])
                assert r['planned_cost']<=r['initial_cost']+1e-6
                out=root/'stage4'
                args=Namespace(cost_dir=str(cost),task_dir=str(task_dir),od_cache_dir=str(root/'cache'),
                               output_dir=str(out),split='validation',variants='sa_timecool,vnd_multistart,alns_timecool',
                               planners='static,oracle',seeds='42',wall_sec=.03,max_iterations=100,
                               final_ratio=.005,od_step_min=20,od_hours=5.,max_tasks=1,
                               max_scenarios=1,resume=False)
                run(args)
                assert len(read_checkpoint(out))==6
                args.resume=True
                run(args)
                assert len(read_checkpoint(out))==6
                with (out/'metrics.csv').open(encoding='utf-8') as f:
                    assert len(f.readlines())==7
                # Reconstruct a synthetic complete Stage3 reference for strict pairing.
                ds=root/'stage3';ds.mkdir()
                arr=pd.read_csv(out/'metrics.csv')
                arr['solver']=arr.solver.replace({'sa_timecool':'sa','vnd_multistart':'vnd',
                                                  'alns_timecool':'official_alns'})
                arr.to_csv(ds/'metrics.csv',index=False)
                cfg4=json.loads((out/'run_config.json').read_text())
                cfg3={'split':'validation','cost_file_sha256':cfg4['cost_file_sha256'],
                      'edge_file_sha256':cfg4['edge_file_sha256'],
                      'tasks_manifest_sha256':cfg4['task_manifest_sha256'],
                      'task_sha256':cfg4['tasks'],
                      'selected_scenario_ids':cfg4['scenario_ids'],
                      'planners':cfg4['planners'],'seeds':cfg4['seeds'],
                      'methods':['sa','vnd','official_alns'],'expected_count':len(arr)}
                write_json(ds/'run_config.json',cfg3)
                result=analyze(out,ds,root/'analysis')
                assert result['verified_complete_grid']
                # Negative test: reject missing results.
                arr.iloc[:-1].to_csv(ds/'metrics.csv',index=False)
                try:analyze(out,ds,root/'bad_analysis')
                except ValueError as e: assert 'Incomplete' in str(e) or 'incomplete' in str(e)
                else:raise AssertionError('Missing Stage3 cell passed strict check')
                # Negative test: forbid test data in Stage4 ablations.
                args.split='test'
                try:run(args)
                except ValueError as e:assert 'validation-only' in str(e)
                else:raise AssertionError('Leakage guard failed')
    print('PASS Stage4 synthetic: time-temperature, 3 solvers (ALNS fake only), full grid, resume, paired audit, test-lock')
    print('NOTE Real official ALNS 7.0.0 integration still REQUIRES server smoke test.')

if __name__=='__main__':check()
