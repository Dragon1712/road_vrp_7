from __future__ import annotations
import csv,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import ModuleType,SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
B2=ROOT/'road_vrp7_step6b2';B1=ROOT/'road_vrp7_step6b1_v2';R=ROOT/'road_vrp7_step6b2_r'
for p in (B2,B1,R):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from features import FEATURE_MODES,B2_SUFFIX,TRAFFIC,RELIABILITY
from dataset import sha256,load
from train_b2_r import study


def module_by_path(p,name):
    sp=importlib.util.spec_from_file_location(name,p)
    mod=importlib.util.module_from_spec(sp);sys.modules[name]=mod
    sp.loader.exec_module(mod)
    return mod

class DummyObj:
    def __init__(self):
        self.od=SimpleNamespace(static=np.asarray([[0 if i==j else 10+5*i+3*j for j in range(7)] for i in range(7)],float))
        self.index={i:i for i in range(7)};self.dem={i:1 for i in range(7)}
        self.capacity=12;self.task={'service_seconds':25};self.planner='static'
    def cost(self,routes):
        return sum(self.od.static[a,b] for r in routes for a,b in zip([0]+list(r),list(r)+[0]))

class TestB2RTraining(unittest.TestCase):
    def test_f5_report_is_task_only(self):
        fixture=module_by_path(B1/'tests/test_v2.py','_b2r_v2_fixture')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture.fixture(root/'frozen');d=load(source)
            f=root/'features';f.mkdir()
            np.savez_compressed(f/'boundary_X.npz',X=np.random.default_rng(17).normal(size=(d.n,len(B2_SUFFIX))))
            (f/'run_config.json').write_text(json.dumps({'stage':'step6b2-features-v1',
             'source_sha256':{k:sha256(v) for k,v in d.source_files.items()},'feature_names':list(B2_SUFFIX)}))
            (f/'summary.json').write_text(json.dumps({'status':'COMPLETE',
             'feature_npz_sha256':sha256(f/'boundary_X.npz')}))
            out=root/'trained';study(source,f,out,['task','date'],42,5,('P2','P4'),('F0','F2','F3','F4','F5'))
            task=json.loads((out/'task/report.json').read_text())
            date=json.loads((out/'date/report.json').read_text())
            self.assertIn('P2_F5',task)
            self.assertNotIn('P2_F5',date)
            self.assertNotIn('P4_F5',date)
            self.assertEqual(set(task),set(json.loads((out/'summary.json').read_text())['actual_methods_by_protocol']['task']))
            self.assertEqual(set(date),set(json.loads((out/'summary.json').read_text())['actual_methods_by_protocol']['date']))
            self.assertEqual(len(date),8)

    def test_frozen_traffic_permutation_no_refit(self):
        from frozen_permutation import probe
        fixture=module_by_path(B1/'tests/test_v2.py','_b2r_v2_fixture_perturb')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture.fixture(root/'frozen');d=load(source)
            f=root/'features';f.mkdir()
            np.savez_compressed(f/'boundary_X.npz',X=np.random.default_rng(18).normal(size=(d.n,len(B2_SUFFIX))))
            (f/'run_config.json').write_text(json.dumps({'stage':'step6b2-features-v1',
             'source_sha256':{k:sha256(v) for k,v in d.source_files.items()},'feature_names':list(B2_SUFFIX)}))
            (f/'summary.json').write_text(json.dumps({'status':'COMPLETE',
             'feature_npz_sha256':sha256(f/'boundary_X.npz')}))
            trained=root/'train';study(source,f,trained,['task'],42,6,('P2',),('F3','F4'))
            out=root/'perm.json'
            result=probe(source,f,trained,out,('P2',),('F3','F4'),(100,101))
            self.assertEqual(len(result),2+(2*2)+(3*2))
            self.assertTrue(out.exists())
            self.assertEqual({r['model'] for r in result},{'P2_F3','P2_F4'})
            self.assertEqual([r['permutation'] for r in result].count('identity'),2)
            with self.assertRaises(FileExistsError):probe(source,f,trained,out,('P2',),('F3',),(100,))

class TestB2RHook(unittest.TestCase):
    def setUp(self):
        names=('section_dataset','section_v2','road_vrp','benchmark','run_experiment')
        self.original={n:sys.modules.get(n) for n in names}
        sec=ModuleType('section_v2')
        class Selector:
            def destroy(self,*a):return 'ORIGINAL_C3'
        class Stats:
            def __init__(self):self.val={}
            def add(self,key,n=1):self.val[key]=self.val.get(key,0)+n
        sec.Selector=Selector;sec.Stats=Stats
        sec.Block=lambda ids,r,p:SimpleNamespace(ids=ids,route=r,pos=p)
        sec.Partial=lambda routes,obj,blocks,original:SimpleNamespace(routes=routes,obj=obj,blocks=blocks,original=original)
        sec.frozen=lambda r:tuple(tuple(x) for x in r)
        sec.mutable=lambda r:[list(x) for x in r]
        sys.modules['section_v2']=sec
        ds=ModuleType('section_dataset')
        ds.HistoricalCorridorFeatures=object;ds.read_raw_cost=lambda p:None
        ds.select_actions=lambda cs,k,sid,seed:list(enumerate(cs))[:k]
        sys.modules['section_dataset']=ds
        road=ModuleType('road_vrp')
        for n in ('Road','OD','Objective'):setattr(road,n,object)
        road.legal=lambda *x:None;road.read_json=lambda _:{}
        sys.modules['road_vrp']=road
        bench=ModuleType('benchmark');bench.load_tasks=lambda _:None
        sys.modules['benchmark']=bench
        v3=ModuleType('run_experiment');v3.NewSelector=Selector
        class Budget:
            def __init__(self,iters,wall):self.deadline=float('inf')
        v3.BothBudget=Budget;v3.check_alns=lambda:None
        sys.modules['run_experiment']=v3
        self.sec=sec;self.selector=Selector();self.selector.stats=Stats()
        self.selector.candidates=lambda routes,traffic=False:[{'route':0,'pos':0,'size':2,'static':150.},
            {'route':1,'pos':1,'size':2,'static':180.}]
        self.script=module_by_path(R/'benchmark_30s_r.py','_b2r_hook_test')
        self.obj=DummyObj();self.routes=[[1,2,3],[4,5,6]]
    def tearDown(self):
        for n,v in self.original.items():
            if v is None:sys.modules.pop(n,None)
            else:sys.modules[n]=v

    def artifact(self,threshold=0):
        class Predict:
            def __init__(self):self.calls=0
            def predict(self,X):
                self.calls+=1
                s=np.arange(len(X),dtype=float)+10.
                return s,s,None
        return {'mode':'F0','threshold':threshold,'model':Predict()}
    def handler(self,a,**kwargs):
        return self.script.OnlineSectionR(a,self.obj,None,'t','sc','2025-07-22 08:00',**kwargs)

    def test_cadence_gate_fallback_and_hook_restore(self):
        handler=self.handler(self.artifact(1e9),score_every=4,max_scoring_sec=2.)
        old=self.sec.Selector.destroy
        with self.script.temporary_hook(handler):
            for _ in range(8):
                r=self.selector.destroy('static',self.routes,np.random.default_rng(42))
                self.assertEqual(r,'ORIGINAL_C3')
        self.assertIs(self.sec.Selector.destroy,old)
        self.assertEqual(handler.stats['r_cadence_fallbacks'],6)
        self.assertEqual(handler.stats['r_gate_fallbacks'],2)
        self.assertEqual(handler.stats['r_model_eval_calls'],1)  # gate cache
        self.assertEqual(handler.stats['r_noop_fallbacks'],0)
        self.assertEqual(handler.stats['r_c3_fallbacks'],8)

    def test_valid_learned_and_state_cache(self):
        art=self.artifact(-100)
        handler=self.handler(art,score_every=1,max_scoring_sec=2.)
        with self.script.temporary_hook(handler):
            for _ in range(3):
                out=self.selector.destroy('static',self.routes,np.random.default_rng(42))
                self.assertEqual(len(out.blocks),1)
        self.assertEqual(art['model'].calls,1)
        self.assertEqual(handler.stats['r_model_actions'],3)
        self.assertEqual(handler.stats['r_cache_hits'],2)
        self.assertEqual(handler.stats['r_scoring_calls'],1)
        self.assertEqual(handler.stats['r_c3_fallbacks'],0)

    def test_budget_zero_always_original(self):
        h=self.handler(self.artifact(),score_every=1,max_scoring_sec=0.)
        with self.script.temporary_hook(h):
            for i in range(5):self.assertEqual(self.selector.destroy('static',self.routes,np.random.default_rng(i)),'ORIGINAL_C3')
        self.assertEqual(h.stats['r_budget_fallbacks'],5)
        self.assertEqual(h.stats['r_model_eval_calls'],0)

    def test_cheap_prefilter_avoids_model(self):
        art=self.artifact()
        h=self.handler(art,score_every=1,min_static_fraction=100.)
        with self.script.temporary_hook(h):
            for _ in range(3):self.assertEqual(self.selector.destroy('static',self.routes,np.random.default_rng(3)),'ORIGINAL_C3')
        self.assertEqual(h.stats['r_model_eval_calls'],0)
        self.assertEqual(h.stats['r_cheap_fallbacks'],3)

class TestReportRepair(unittest.TestCase):
    def test_copy_only_old_f5_date_pollution(self):
        from repair_f5_report import repair
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'frozen';(study/'date').mkdir(parents=True)
            (study/'task').mkdir(parents=True)
            (study/'manifest.json').write_text(json.dumps({'stage':'step6b2-ablation-v1'}))
            task={'P2_F0':{'gain':10},'P2_F5':{'gain':30}}
            date={'P2_F0':{'gain':12},'P2_F5':{'gain':30}}
            (study/'task/report.json').write_text(json.dumps(task))
            (study/'date/report.json').write_text(json.dumps(date))
            before=(study/'date/report.json').read_bytes()
            r=repair(study,root/'corrected')
            self.assertEqual(r['removed_invalid_date_entries'],['P2_F5'])
            self.assertEqual(json.loads((root/'corrected/date_report_corrected.json').read_text()),{'P2_F0':{'gain':12}})
            self.assertEqual((study/'date/report.json').read_bytes(),before)
            with self.assertRaises(FileExistsError):repair(study,root/'corrected')

class TestMatchedSummary(unittest.TestCase):
    def test_strict_keys_and_sign(self):
        from analyze_r import compare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest={'source_sha256':'s','road_sha256':'r','edge_sha256':'e','solver_sha256':'v',
              'section_sha256':'j','split':'validation','wall_sec':30,'max_iterations':20000,
              'anneal_iterations':5000,'top_k':12,'seeds':[42], 'task_ids':['t'], 'scenario_ids':['s']}
            for arm,cost in [('base',100.),('treat',80.)]:
                d=root/arm;d.mkdir()
                (d/'manifest.json').write_text(json.dumps(manifest))
                (d/'summary.json').write_text(json.dumps({'status':'COMPLETE','expected_records':1}))
                with (d/'metrics.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['task_id','scenario_id','date','seed','ex_post_road_travel_sec','iterations'])
                    w.writeheader();w.writerow({'task_id':'t','scenario_id':'s','date':'2025-07-22',
                                                   'seed':42,'ex_post_road_travel_sec':cost,'iterations':10})
            r=compare(root/'base',root/'treat',root/'diff.json')
            self.assertEqual(r['mean_road_improvement_sec'],20.)
            with self.assertRaises(FileExistsError):compare(root/'base',root/'treat',root/'diff.json')

if __name__=='__main__':unittest.main()
