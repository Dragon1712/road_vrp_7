from __future__ import annotations
import unittest,sys,types,importlib.util
from pathlib import Path
from unittest.mock import patch
import numpy as np
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
import sv_core as s

class Dummy:
    def __init__(self,j,score=5.):self.j=j;self.score=score
    def select(self,actions,baseline,same=None,valid=None):
        return self.j,self.score,'testing'

class M0:
    def predict(self,x):
        n=len(x)
        return np.arange(n,dtype=float),np.ones(n)*.9,None

class Tests(unittest.TestCase):
    def test_selection_masks(self):
        a=np.ones((3,26),np.float32);b=np.zeros(26,np.float32)
        self.assertEqual(s.selection(Dummy(1),a,b,np.ones(3,bool),np.zeros(3,bool))[0],1)
        with self.assertRaises(ValueError):s.selection(Dummy(1),a,b,np.array([1,0,1],bool),np.zeros(3,bool))
        with self.assertRaises(ValueError):s.selection(Dummy(1),a,b,np.ones(3,bool),np.array([0,1,0],bool))
        self.assertEqual(s.selection(Dummy(None),a,b,np.ones(3,bool),np.zeros(3,bool))[0],-1)

    def test_m0_gate(self):
        x=np.zeros((3,26),np.float32);v=np.array([1,1,1],bool);z=np.zeros(3,bool)
        art={'model':M0(),'threshold':.5}
        self.assertEqual(s.m0_selection(art,x,v,z)[0],2)
        art['threshold']=1.
        self.assertEqual(s.m0_selection(art,x,v,z)[0],-1)
        art['threshold']=None
        self.assertEqual(s.m0_selection(art,x,v,z)[0],-1)

    def test_regret_and_oracle(self):
        acts={'C3':(-1,'C3',0.),'M0_PROXY':(2,'pick',1.),'R2_F2':(0,'pick',1.),'T2_HUBER':(-1,'gate',0.)}
        out=s.classify_row([20,-100,-50],np.ones(3,bool),np.zeros(3,bool),acts,{'decision_id':'a'})
        d={r['arm']:r for r in out}
        self.assertEqual(d['C3']['gain_sec'],0)
        self.assertEqual(d['R2_F2']['gain_sec'],20)
        self.assertEqual(d['R2_F2']['regret_sec'],0)
        self.assertEqual(d['M0_PROXY']['regret_sec'],70)
        self.assertTrue(d['M0_PROXY']['harm_gt40'])

    def test_pick(self):
        rows=[]
        for n in range(12):
            for arm in ('R2_F2','T2_HUBER'):
                rows.append(dict(decision_id=f'ID{n}',arm=arm,selected_index=(0 if arm=='R2_F2' else n%2),
                                 role='internal_test',task_id='a',scenario_id='b'))
        x=s.pick_states(rows,6,42)
        self.assertEqual(len(x),6);self.assertEqual(x,s.pick_states(rows,6,42))
        self.assertEqual(len(set(x)),6)

    def test_paired(self):
        rows=[]
        for h in (5,20):
            for arm,val in [('C3',10),('M0_PROXY',13),('R2_F2',8),('T2_HUBER',9)]:
                rows.append(dict(decision_id='z',horizon=h,replicate=0,arm=arm,final_td_od_cost=val))
        d=s.paired_summary(rows,'final_td_od_cost')
        self.assertEqual(d['5']['R2_F2']['mean_gain_vs_C3_sec'],2)
        self.assertEqual(d['20']['M0_PROXY']['mean_gain_vs_C3_sec'],-3)
        with self.assertRaises(ValueError):s.paired_summary(rows[:-1],'final_td_od_cost')

    def test_code_horizon_and_repair_mock(self):
        # Import runnable module with isolated test doubles (no Chicago asset needed).
        for n in ['road_vrp','section_v2','section_dataset','r2_core','run_experiment']:
            if n in sys.modules:del sys.modules[n]
        fake={n:types.ModuleType(n) for n in ['road_vrp','section_v2','section_dataset','r2_core','run_experiment']}
        fake['road_vrp'].Road=object;fake['road_vrp'].OD=object;fake['road_vrp'].Objective=object
        fake['road_vrp'].legal=lambda t,r:True;fake['road_vrp'].read_json=lambda p:{}
        fake['section_v2'].Selector=object
        fake['section_dataset'].select_actions=lambda *a:[]
        def fake_repair(obj,route,c,top,deadline,enable):
            return [list(v) for v in route], 'moved',.0001,True,{}
        fake['section_dataset'].repair_for=fake_repair
        fake['r2_core'].state_key=lambda *a:'sample';fake['r2_core'].matches=lambda a,b:a is b
        fake['run_experiment'].greedy=lambda *a:([[-1]],None)
        def solve(name,task,obj,seed,wall,h,*,top_k,anneal_iterations):
            solution,_=fake['run_experiment'].greedy(task,obj,seed)
            return {'routes':solution,'iterations':h,'search_wall_sec':.001,'termination_reason':'iteration_budget'}
        fake['run_experiment'].solve=solve
        with patch.dict(sys.modules,fake):
            spec=importlib.util.spec_from_file_location('rollout_sv_test',P/'rollout_sv.py')
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            with mod.fixed_greedy_start([[1,2],[3,4]]):
                self.assertEqual(mod.v3.greedy({},None,42)[0],[[1,2],[3,4]])
            self.assertEqual(mod.v3.greedy({},None,42)[0],[[-1]])
            self.assertEqual(mod.stable_seed('x',5,0),mod.stable_seed('x',20,0))
            self.assertNotEqual(mod.stable_seed('x',5,0),mod.stable_seed('x',5,1))
            result=mod.search_restart({},None,[[1,2]],44,5,30)
            self.assertEqual(result['iterations'],5)
            self.assertEqual(result['routes'],[[1,2]])

if __name__=='__main__':unittest.main()
