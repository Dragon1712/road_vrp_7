import json,tempfile,sys,unittest
from pathlib import Path
from unittest import mock
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from vq_core import (uniform_train_selection,paired_features,restart_seed,row_labels,
                     select_by_score,decision_metrics,sha256,STAGE,LABELS)
import train_vq

class CoreTest(unittest.TestCase):
    def test_task_only_sample_determinism(self):
        groups=[{'task_id':f'task_{i//10:04d}','decision_id':f'd{i}'} for i in range(80)]
        a=uniform_train_selection(groups,{f'task_{i:04d}' for i in range(4)},20,42)
        b=uniform_train_selection(groups,{f'task_{i:04d}' for i in range(4)},20,42)
        self.assertEqual(a,b);self.assertEqual(len(set(a)),20)
        self.assertEqual({groups[x]['task_id'] for x in a},{f'task_{i:04d}' for i in range(4)})
        self.assertRaises(ValueError,uniform_train_selection,groups,{'task_0000'},1000,42)
    def test_paired_features(self):
        x=paired_features(np.array([[1,3],[3,4]]),np.array([2,3]))
        np.testing.assert_array_equal(x,np.array([[1,3,2,3,-1,0],[3,4,2,3,1,1]]))
    def test_seed_independent_horizon(self):
        self.assertEqual(restart_seed('xyz',0),restart_seed('xyz',0))
        self.assertNotEqual(restart_seed('xyz',0),restart_seed('xyz',1))
    def test_value_labels(self):
        rows=[]
        for rep in range(3):
            rows += [{'decision_id':'d','horizon':20,'replicate':rep,'action_index':-1,'final_road':100+rep},
                     {'decision_id':'d','horizon':20,'replicate':rep,'action_index':0,'final_road':90+(rep%2)*20+rep}]
        a=row_labels(rows,'road');self.assertAlmostEqual(a['d',20,0]['mean_gain'],10-20/3)
        self.assertEqual(a['d',20,-1]['sd_gain'],0)
        self.assertRaises(ValueError,row_labels,rows+[rows[0]])
    def test_selective_gate(self):
        s=np.array([10.,12.,99.]);mask=[1,1,0]
        self.assertEqual(select_by_score(s,mask),1)
        self.assertEqual(select_by_score(s,mask,lam=1,std=[0,20,0]),0)
        self.assertEqual(select_by_score(s,mask,threshold=50),-1)
    def test_decision_metrics(self):
        r=decision_metrics([0,-1],[[20,-50],[30,40]],[[True,True],[True,True]],['task0','task1'])
        self.assertEqual(r['n_states'],2);self.assertEqual(r['top1_hit'],.5)
    def test_same_repair_route_exact_control(self):
        import types
        stubs={name:types.ModuleType(name) for name in ('rollout_sv','road_vrp','section_dataset','r2_core')}
        stubs['rollout_sv'].reconstruct=lambda *x: None
        stubs['rollout_sv'].search_restart=lambda *x: None
        for name in ('Road','OD','Objective','legal','read_json'):setattr(stubs['road_vrp'],name,lambda *x:None)
        stubs['section_dataset'].repair_for=lambda *x:None
        stubs['r2_core'].matches=lambda *x:False
        with mock.patch.dict(sys.modules,stubs):
            import importlib
            collect_vq=importlib.import_module('collect_vq')
        same=[ [1,2], [3,4] ];other=[[1,3],[2,4]]
        info={-1:{'routes':same,'action_id':[0,0,2],'one_step_gain':0},
              0:{'routes':same,'action_id':[0,1,2],'one_step_gain':0},
              1:{'routes':other,'action_id':[1,1,2],'one_step_gain':5}}
        def fake_search(task,static,starting_routes,seed,h,wall,anneal,top_k):
            return {'routes':starting_routes,'iterations':h,'search_wall_sec':.01}
        class FakeCost:
            def cost(self,routes):return float(sum(sum(r) for r in routes))
        class FakeRoad:
            def replay(self,task,routes,departure):return {'ex_post_road_travel_sec':float(sum(sum(r) for r in routes))}
        with mock.patch.object(collect_vq,'search_restart',side_effect=fake_search) as s:
            rows=collect_vq.make_unit('d',None,{'departure_local':'2025-07-25'},FakeCost(),FakeCost(),FakeRoad(),same,
                    {'task_id':'t','scenario_id':'s'},info,20,0,42,12,30,True)
        self.assertEqual(s.call_count,2)
        self.assertEqual(rows[0]['final_road'],rows[1]['final_road'])
        self.assertEqual({r['action_index'] for r in rows},{-1,0,1})

class SyntheticPipeline(unittest.TestCase):
    def test_train_loto_no_original_validation_or_test(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);src=base/'r2';col=base/'collect';dst=base/'out'
            src.mkdir();col.mkdir()
            rng=np.random.default_rng(3)
            groups=[{'decision_id':f'd{i}','task_id':f'task_{i//2:04d}',
                  'scenario_id':'s','date':'2025-07-23'} for i in range(16)]
            X=rng.normal(size=(16,8,44)).astype('float32');B=rng.normal(size=(16,44)).astype('float32')
            Y=rng.normal(size=(16,8)).astype('float32')*10;BY=np.zeros(16,dtype='float32')
            valid=np.zeros((16,8),bool);valid[:,:2]=True;same=np.zeros((16,8),bool)
            np.savez(src/'paired.npz',X=X,B=B,Y=Y,BY=BY,valid=valid,same=same)
            (src/'groups.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in groups))
            (src/'manifest.json').write_text(json.dumps({'stage':'step6c1r2-real-C3-labelled-v1',
                    'pair_sha256':sha256(src/'paired.npz'),'groups_sha256':sha256(src/'groups.jsonl')}))
            selection=[f'd{i}' for i in range(8)]
            (col/'manifest.json').write_text(json.dumps({'stage':STAGE,'selected_decision_ids':selection,
                      'horizons':[20],'replicates':3,'label_sha256':sha256(src/'paired.npz'),
                      'group_sha256':sha256(src/'groups.jsonl')}))
            rows=[]
            for di in selection:
                idx=int(di[1:]);ground=[float(3*X[idx,j,0]+4*j) for j in range(2)]
                for rep in range(3):
                    for j in [-1,0,1]:
                        rows.append({'decision_id':di,'horizon':20,'replicate':rep,
                            'action_index':j,'final_road':1000+rep-(ground[j] if j!=-1 else 0)})
            (col/'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (col/'report.json').write_text(json.dumps({'stage':LABELS,'status':'COMPLETE','n_restart_action_rows':len(rows)}))
            from argparse import Namespace
            cfg=Namespace(dataset_dir=str(src),collection_dir=str(col),output_dir=str(dst),seed=42)
            train_vq.run(cfg)
            summary=json.loads((dst/'report.json').read_text())
            self.assertEqual(summary['n_train_tasks'],4)
            self.assertTrue(summary['no_internal_test_or_validation_data_used'])
            self.assertEqual(len((dst/'fold_decisions.jsonl').read_text().splitlines()),8*5)
            self.assertEqual(set(summary['results']['20']['per_heldout_task']),{f'task_{i:04d}' for i in range(4)})

if __name__=='__main__':unittest.main()
