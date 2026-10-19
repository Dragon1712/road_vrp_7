#!/usr/bin/env python3
"""Synthetic tests. On the project server actual stage3/section_v2 are imported.
Outside the project, a small mock tests the collector logic, not ALNS integration.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np

folder=Path(__file__).resolve().parents[1]
root=folder.parent
sys.path.insert(0,str(folder))
actual_stage=(root/'road_vrp7_stage3/road_vrp.py').is_file()
if not actual_stage:
    fake_road=types.ModuleType('road_vrp')
    fake_road.Road=fake_road.OD=fake_road.Objective=object
    fake_road.legal=lambda task,routes:True
    fake_road.read_json=lambda p:json.loads(Path(p).read_text())
    fake_road.write_json=lambda p,o:Path(p).write_text(json.dumps(o))
    fake_road.sha256=lambda p:'checksum-synthetic'
    sys.modules['road_vrp']=fake_road
    fake_section=types.ModuleType('section_v2')
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Block:
        ids:tuple;route:int;pos:int
    @dataclass(frozen=True)
    class Partial:
        routes:tuple;obj:object;blocks:tuple;original:tuple
    class Stats:
        def __init__(self):self.d={'repair_deadline':0,'repair_candidates_exact':0}
        def summary(self):return self.d
    fake_section.Block=Block;fake_section.Partial=Partial;fake_section.Selector=object
    fake_section.Stats=Stats
    fake_section.frozen=lambda x:tuple(tuple(y) for y in x)
    fake_section.mutable=lambda x:[list(y) for y in x]
    def fake_repair(partial,obj,deadline,stats,top_k,nontrivial,enable_swap):
        # Build two feasible nontrivial repairs from the SAME partial state.
        original=[list(x) for x in partial.original]
        base=partial.blocks[0]
        x=[list(q) for q in partial.routes]
        y=[list(q) for q in partial.routes]
        x[0][0:0]=list(base.ids)
        y[1][0:0]=list(base.ids)
        moves=[p for p in [x,y] if tuple(map(tuple,p))!=partial.original]
        best=min(moves,key=obj.cost)
        stats.d['repair_candidates_exact']=len(moves)
        return best,'moved'
    fake_section.repair=fake_repair
    sys.modules['section_v2']=fake_section

import section_dataset as ds
import analyze_step6a as az

class TestStep6A(unittest.TestCase):
    def test_stratified_determinism_and_balance(self):
        cs=[{'static':i} for i in range(32)]
        a=ds.select_actions(cs,8,'state-123',42)
        b=ds.select_actions(cs,8,'state-123',42)
        self.assertEqual(a,b)
        self.assertEqual(len(a),8)
        self.assertEqual(len({idx for idx,_ in a}),8)
        vals=[idx for idx,_ in a]
        self.assertTrue(any(x<11 for x in vals))
        self.assertTrue(any(11<=x<22 for x in vals))
        self.assertTrue(any(x>=22 for x in vals))

    def test_state_sampler_unique_and_group_coverage(self):
        states=[]
        for task in ['task_0000','task_0001','task_0002']:
            for scene in ['s1','s2']:
                for i in range(10):
                    states.append({'task_id':task,'scenario_id':scene,'state_id':task+scene+str(i)})
        a=ds.pick_balanced_states(states,18,2026)
        b=ds.pick_balanced_states(list(reversed(states)),18,2026)
        self.assertEqual(a,b)
        self.assertEqual(len({x['state_id'] for x in a}),18)
        self.assertEqual(len(set((x['task_id'],x['scenario_id']) for x in a)),6)

    def test_snapshot_dedup_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'run_config.json').write_text(json.dumps({'stage':'step5b-v3',
                'data_role':'July development only','tasks':{'task_0000.json':'x'},
                'scenarios':['2025-07-22_0800']}))
            (p/'benchmark_summary.json').write_text(json.dumps({'status':'COMPLETE','actual_count':1,'expected_count':1}))
            base={'task_id':'task_0000','scenario_id':'2025-07-22_0800',
                  'solver':'C3_topk','planner':'oracle','seed':42,
                  'snapshots':[{'status':'reached','routes':[[1,2],[3,4]],'target_sec':5},
                               {'status':'reached','routes':[[1,2],[3,4]],'target_sec':15}]}
            (p/'snapshot_records.jsonl').write_text(json.dumps(base)+'\n')
            got,_,_=ds.load_snapshot_states(p,('C3_topk',),target_times=(5.,15.))
            self.assertEqual(len(got),1)
            self.assertEqual(got[0]['snapshot_sec'],5.)
            self.assertEqual(got[0]['routes'],[[1,2],[3,4]])

    def test_no_future_observation_and_masked_mean(self):
        class FakeRoad:
            def __init__(self):
                self.dt=60.
                self.times=np.arange(6)
                self.static=np.array([10.,20.,30.])
                self.observed=np.array([
                    [0,0,0], [1,0,0], [0,1,0],
                    [1,1,1], [1,1,1], [1,1,1]],dtype=bool)
            def seconds(self,dt):return 180.
            def path(self,a,b,t,mode='static'):
                return 10.,[0,1]
        road=FakeRoad()
        raw=np.ones((6,3))*999999
        raw[1,0]=15.;raw[2,1]=30.
        # index 3 is exactly at departure; 999999 must NOT be used.
        obj=ds.HistoricalCorridorFeatures(road,raw,2)
        od=types.SimpleNamespace(task={'task_id':'t1'},nodes=['0','1'],start=180.)
        obj.enter('t1',od,'date')
        self.assertEqual(obj.index_anchor,2)
        feat=obj.path_features(0,1)
        self.assertEqual(feat['covered_fraction'],[.5,.5])
        self.assertEqual(feat['observed_rel_cost'],[1.5,1.5])
        self.assertTrue(feat['has_any_observation'])

    def test_unobserved_is_not_a_speed(self):
        class Road:
            dt=60.;times=np.arange(8);static=np.array([10.])
            observed=np.zeros((8,1),dtype=bool)
            def seconds(self,x):return 300.
            def path(self,a,b,t,mode='static'):return 10.,[0]
        hist=ds.HistoricalCorridorFeatures(Road(),np.ones((8,1))*900,4)
        od=types.SimpleNamespace(task={'task_id':'task'},nodes=['a','b'],start=300.)
        hist.enter('task',od,'when')
        val=hist.path_features(0,1)
        self.assertEqual(val['observed_rel_cost'],[0.,0.,0.,0.])
        self.assertEqual(val['covered_fraction'],[0.,0.,0.,0.])

    def test_checkpoint_fail_closed_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'new'
            cfg={'actions_per_state':4,'version':1}
            got=ds.resume_states(output,cfg,resume=False)
            self.assertEqual(got,set())
            ds.write_line(output/'state_records.jsonl',{'state_id':'s1','candidates':[]})
            self.assertEqual(ds.resume_states(output,cfg,resume=True),{'s1'})
            with self.assertRaises(ValueError):
                ds.resume_states(output,{'actions_per_state':5,'version':1},resume=True)
            with self.assertRaises(FileExistsError):
                ds.resume_states(output,cfg,resume=False)

    def test_spearman_and_tied_ranks(self):
        self.assertAlmostEqual(az.spearman([1,2,3],[3,2,1]),-1.)
        self.assertAlmostEqual(az.spearman([1,2,3],[1,2,3]),1.)
        self.assertIsNone(az.spearman([2,2,2],[1,2,3]))
        self.assertEqual(az.ranks([4,1,4]).tolist(),[2.5,1.,2.5])

    def test_audit_candidate_identity(self):
        r={'original_td_cost_sec':100.,'routes':[[1,2,3],[4,5,6]],
           'candidates':[{'candidate_id':'id1','route_index':0,'position':1,'size':2,
                'block_ids':[2,3],'history_observed_share':.4,
                'history_coverage_by_slot':[.2,.4],
                'history_rel_cost_by_slot':[1.,1.1], 'label_valid':True,
                'td_repaired_routes':[[1,4,5],[2,3,6]],
                'static_repaired_routes':[[1,2],[3,4,5,6]],
                'td_gain_sec':8.,'static_repair_td_gain_sec':5.,
                'td_advantage_over_static_repair_sec':3.,
                'static_gain_same_td_move_sec':-2.,'static_repair_static_gain_sec':1.},
                {'candidate_id':'id2','route_index':1,'position':1,'size':2,
                 'block_ids':[5,6],'history_observed_share':0.,
                 'history_coverage_by_slot':[0.,0.],
                 'history_rel_cost_by_slot':[0.,0.], 'label_valid':False,
                 'td_gain_sec':None,'static_gain_same_td_move_sec':None}]}
        az.audit_one_state(r,{'actions_per_state':8,'lookback_slots':2})
        r['candidates'][0]['td_advantage_over_static_repair_sec']=-1.
        with self.assertRaises(ValueError):az.audit_one_state(r,{'actions_per_state':8,'lookback_slots':2})

    def test_end_to_end_analysis_and_safe_export(self):
        import export_step6a as ex
        def sample(cid, route, pos, block, gain, sgain, st_td_gain, td_routes, st_routes):
            return {'candidate_id':cid,'candidate_index':0,'route_index':route,
                'position':pos,'size':2,'block_ids':block,'block_demand':2,
                'route_load':3,'vehicle_capacity':5,
                'static_section_cost_sec':14.,'route_length':3,
                'relative_position':.5,'history_observed_share':.25,
                'history_observed_rel_cost':1.25,'history_has_observation':True,
                'history_coverage_by_slot':[.5,0.],
                'history_rel_cost_by_slot':[1.25,0.],
                'label_valid':True,'td_gain_sec':gain,
                'static_gain_same_td_move_sec':sgain,
                'static_repair_static_gain_sec':sgain,
                'static_repair_td_gain_sec':st_td_gain,
                'td_advantage_over_static_repair_sec':gain-st_td_gain,
                'td_repaired_routes':td_routes,'static_repaired_routes':st_routes}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'run_config.json').write_text(json.dumps({
               'stage':'step6a-offline-v1','actions_per_state':8,'lookback_slots':2}))
            (p/'benchmark_summary.json').write_text(json.dumps({
               'status':'COMPLETE','actual_states':1,'expected_states':1}))
            r={'state_id':'state123','task_id':'task_0000',
               'scenario_id':'2025-07-22_0800','source_solver':'C3_topk',
               'seed':42,'snapshot_sec':5,'date':'2025-07-22','task_role':'train',
               'departure_hour_local':8,'departure_weekday_local':1,
               'routes':[[1,2,3],[4,5,6]],
               'original_static_cost_sec':110.,'original_td_cost_sec':100.,
               'candidates':[
                   sample('id1',0,1,[2,3],8.,-2.,5.,
                          [[1,4,5],[2,3,6]],[[1,2],[3,4,5,6]]),
                   sample('id2',1,1,[5,6],-2.,5.,-4.,
                          [[1,5,3],[4,2,6]],[[1,2,3,5],[4,6]])]}
            (p/'state_records.jsonl').write_text(json.dumps(r)+'\n')
            ans=az.analyze(p)
            self.assertEqual(ans['states'],1)
            self.assertEqual(ans['valid_labels'],2)
            self.assertEqual(ans['states_static_top1_differs_from_td_top1'],1)
            self.assertAlmostEqual(ans['mean_oracle_vs_static_selection_regret_sec'],10.)
            info=ex.export(p)
            X=np.load(p/'analysis/features_X.npz')['X']
            y=np.load(p/'analysis/labels_y.npz')['y']
            self.assertEqual(list(X.shape),info['shape_X'])
            self.assertEqual(y.shape,(2,5))
            self.assertEqual(y[:,0].tolist(),[8.,-2.])
            self.assertNotIn('td_gain_sec',info['feature_names'])
            self.assertNotIn('source_solver',info['feature_names'])
            self.assertTrue((p/'analysis/groups.csv').exists())

    @unittest.skipIf(not actual_stage,'requires real stage3 and Section package')
    def test_real_section_integration(self):
        class OD:
            static=np.array([[0,10,20,10,20,10,15],
                             [10,0,10,12,15,8,10],
                             [20,10,0,10,15,15,5],
                             [10,12,10,0,8,15,10],
                             [20,15,15,8,0,8,8],
                             [10,8,15,15,8,0,6],
                             [15,10,5,10,8,6,0]],dtype=float)
            knot_seconds=np.array([0.,600.,1200.,1800.,2400.])
            def travel(self,i,j,t,planner):
                v=float(self.static[i,j]);return v*(1.4 if planner=='oracle' and i==5 else 1.)
        customers=[{'id':i,'node':str(i),'demand':1} for i in range(1,7)]
        task={'task_id':'task','vehicles':2,'capacity':4,
              'customers':customers,'service_seconds':1,'require_all_vehicles':True}
        td=ds.Objective(task,OD(),'oracle')
        st=ds.Objective(task,td.od,'static')
        routes=[[1,2,3],[4,5,6]]
        c={'route':0,'pos':0,'size':2}
        ans=ds.evaluate_section(routes,c,td,st,12,4.,True)
        self.assertTrue(ans['label_valid'])
        self.assertEqual(sorted([i for r in ans['td_repaired_routes'] for i in r]),[1,2,3,4,5,6])
        self.assertGreaterEqual(ans['td_advantage_over_static_repair_sec'],-1e-5)

if __name__=='__main__':
    unittest.main(verbosity=2)
