from __future__ import annotations
import sys,unittest,json,tempfile
from pathlib import Path
from types import SimpleNamespace,ModuleType
import numpy as np
BASE=Path(__file__).resolve().parents[1]
B1=BASE.parent/'road_vrp7_step6b1_v2'
sys.path.insert(0,str(BASE));sys.path.insert(1,str(B1))
from features import (CORE,GEOMETRY,TRAFFIC,RELIABILITY,OLD_HIST,B2_SUFFIX,FEATURE_MODES,
                      core_online,extract_boundary,_static_delta)
from train_b2 import get_features,shuffle_other_date,study
from dataset import sha256

class DummyObjective:
    def __init__(self):
        # depot index=0, customers 1..6, use asymmetric costs
        self.od=SimpleNamespace(static=np.asarray([[0 if i==j else (i*4+j*3+10) for j in range(7)] for i in range(7)],float))
        self.index={i:i for i in range(7)}
        self.dem={i:1 for i in range(7)}
        self.capacity=12
        self.task={'service_seconds':25}
    def cost(self,routes):
        s=0.
        for route in routes:
            seq=[0]+list(route)+[0]
            s+=sum(self.od.static[a,b] for a,b in zip(seq[:-1],seq[1:]))
        return s

class DummyHist:
    n_slots=6
    def path_features(self,i,j):
        # Some real obs, some missing; never all-future.
        cov=.0 if (i+j)%3==0 else .6
        return {'edge_count':1,'covered_fraction':[cov]*6,
                'observed_rel_cost':[1.2+.01*k for k in range(6)],'has_any_observation':cov>0}

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.routes=[[1,2,3],[4,5,6]]
        self.obj=DummyObjective()
        self.before=dict(sys.modules)
        sec=ModuleType('section_v2')
        class Stats:
            def add(self,*args):pass
        sec.Stats=Stats
        # return option: insert the [2,3] block into route 1 at pos1 (no swap)
        def opts(part,block,obj,deadline,stats,nontrivial=True):
            return [(20.,1,1,tuple(block.ids),None,'relocate')]
        sec._block_options=opts
        sec.Block=lambda ids,r,p:SimpleNamespace(ids=ids,route=r,pos=p)
        sec.Partial=lambda routes,obj,blocks,original:SimpleNamespace(routes=routes,obj=obj,blocks=blocks,original=original)
        sec.frozen=lambda r:tuple(tuple(x) for x in r)
        sec.mutable=lambda r:[list(x) for x in r]
        sys.modules['section_v2']=sec
        sd=ModuleType('section_dataset')
        sd.make_partial=lambda routes,c,obj:SimpleNamespace(routes=((1,),(4,5,6)),blocks=(SimpleNamespace(ids=(2,3)),),original=self.routes)
        sys.modules['section_dataset']=sd
    def tearDown(self):
        for k in ('section_v2','section_dataset'):
            if k in self.before:sys.modules[k]=self.before[k]
            else:sys.modules.pop(k,None)
    def test_feature_groups_no_overlap(self):
        self.assertEqual(len(FEATURE_MODES['F0']),12)
        self.assertEqual(len(FEATURE_MODES['F1']),27)
        self.assertTrue(set(GEOMETRY).isdisjoint(TRAFFIC))
        self.assertTrue(set(TRAFFIC).isdisjoint(RELIABILITY))
    def test_proxy_removal_insertion(self):
        c={'route':0,'pos':1,'size':2}
        x=extract_boundary(self.routes,c,self.obj,DummyHist())
        self.assertEqual(set(x),set(B2_SUFFIX))
        self.assertEqual(x['target_route_index'],1.)
        self.assertEqual(x['same_route'],0.)
        self.assertTrue(np.isfinite(list(x.values())).all())
        self.assertAlmostEqual(x['proxy_static_net_gain_sec'],
                               x['remove_static_saved_sec']-x['insert_proxy_static_added_sec']+
                               x['orientation_internal_delta_static_sec'])
        self.assertAlmostEqual(x['proxy_history_net_gain_sec'],
                               x['remove_history_saved_sec']-x['insert_history_added_sec']+
                               x['orientation_internal_delta_history_sec'])
        self.assertTrue(0<=x['removal_observed_share']<=1)
    def test_core_consistency(self):
        c={'route':0,'pos':1,'size':2}
        x=core_online(self.routes,c,self.obj,8,2)
        self.assertEqual(set(x),set(CORE))
        self.assertEqual(x['route_length'],3)

class TestShuffle(unittest.TestCase):
    def test_within_partition_other_date(self):
        names=list(FEATURE_MODES['F4']);N=4*2*3
        x=np.tile(np.arange(len(names)),(N,1)).astype(float)
        groups=[]
        for day in range(4):
            for st in range(2):
                for act in range(3):
                    groups.append({'state_id':f's{day}-{st}','task_id':'task_A','date':f'2025-07-{22+day}'})
                    x[len(groups)-1,names.index('remove_history_saved_sec')]=day+100
        mask=np.array([i<12 for i in range(N)])
        y,info=shuffle_other_date(x,names,groups,mask,42)
        self.assertEqual(info['donor_fraction'],1)
        self.assertTrue(np.array_equal(x[~mask],y[~mask]))
        self.assertTrue(np.array_equal(x[:,names.index('remove_static_saved_sec')],
                                       y[:,names.index('remove_static_saved_sec')]))
        self.assertFalse(np.array_equal(x[mask][:,names.index('remove_history_saved_sec')],
                                         y[mask][:,names.index('remove_history_saved_sec')]))
    def test_single_date_fails_closed(self):
        x=np.ones((4,len(FEATURE_MODES['F4'])))
        g=[{'state_id':str(i),'task_id':'task_A','date':'2025-07-25'} for i in range(4)]
        with self.assertRaisesRegex(ValueError,'ONE DATE'):
            shuffle_other_date(x,list(FEATURE_MODES['F4']),g,np.ones(4,bool),42)

class TestSmokeModel(unittest.TestCase):
    def test_online_schema_and_threshold_models(self):
        from models import Predictor
        names=list(FEATURE_MODES['F3'])
        X=np.random.default_rng(9).normal(size=(64,len(names)))
        y=np.linspace(-20,40,64)
        groups=[{'state_id':f's{i//8}','task_id':f't{i//16}',
                 'date':f'2025-07-{22+i//16}'} for i in range(64)]
        for kind in ('reg','two'):
            z=Predictor(kind,42,8,names,'task').fit(X,y,groups)
            rank,gate,_=z.predict(X[:4]);self.assertTrue(np.isfinite(rank).all())
            self.assertTrue(np.isfinite(gate).all())

if __name__=='__main__':unittest.main()

class TestSyntheticEndToEnd(unittest.TestCase):
    def test_f0_f5_p2_p4_and_model_reload(self):
        from importlib.util import spec_from_file_location,module_from_spec
        path=B1/'tests/test_v2.py'
        sp=spec_from_file_location('_b1_fixture_b2',path)
        mod=module_from_spec(sp);sp.loader.exec_module(mod)
        from dataset import load
        import joblib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=mod.fixture(root/'frozen')
            d=load(source)
            aug=root/'features';aug.mkdir()
            values=np.random.default_rng(17).normal(size=(d.n,len(B2_SUFFIX)))
            np.savez_compressed(aug/'boundary_X.npz',X=values)
            source_hash={k:sha256(v) for k,v in d.source_files.items()}
            (aug/'run_config.json').write_text(json.dumps({'stage':'step6b2-features-v1',
                 'source_sha256':source_hash,'feature_names':list(B2_SUFFIX)}))
            (aug/'summary.json').write_text(json.dumps({'status':'COMPLETE',
                 'feature_npz_sha256':sha256(aug/'boundary_X.npz')}))
            result=root/'trained'
            study(source,aug,result,['task','date'],42,8,('P2','P4'),('F0','F1','F2','F3','F4','F5'))
            self.assertTrue((result/'task/P2_F5_model.joblib').exists())
            self.assertFalse((result/'date/P2_F5_model.joblib').exists())
            self.assertTrue((result/'date/P4_F4_model.joblib').exists())
            obj=joblib.load(result/'task/P2_F4_model.joblib')
            self.assertEqual(obj['training_label'],'static_repair_td_gain_sec')
            self.assertEqual(obj['feature_names'],list(FEATURE_MODES['F4']))
            X,_=get_features(d,values,'F4')
            self.assertTrue(np.isfinite(obj['model'].predict(X[:8])[0]).all())
            with self.assertRaises(FileExistsError):
                study(source,aug,result,['task'],42,8,('P2',),('F0',))

class TestOnlineHook(unittest.TestCase):
    def test_patch_reversible_and_inference_is_actionable(self):
        import importlib.util
        from models import Predictor
        original={n:sys.modules.get(n) for n in ('section_dataset','section_v2','road_vrp','benchmark','run_experiment')}
        try:
            sec=ModuleType('section_v2')
            class Selector:
                def destroy(self,*a):return 'ORIGINAL'
            class Stats:
                def __init__(self):self.val={}
                def add(self,key,x=1):self.val[key]=self.val.get(key,0)+x
            sec.Selector=Selector
            sec.Stats=Stats
            sec.Block=lambda ids,r,p:SimpleNamespace(ids=ids,route=r,pos=p)
            sec.Partial=lambda routes,obj,blocks,original:SimpleNamespace(routes=routes,obj=obj,blocks=blocks,original=original)
            sec.frozen=lambda routes:tuple(tuple(r) for r in routes)
            sec.mutable=lambda routes:[list(r) for r in routes]
            sys.modules['section_v2']=sec
            ds=ModuleType('section_dataset')
            ds.HistoricalCorridorFeatures=object
            ds.read_raw_cost=lambda _:None
            ds.select_actions=lambda candidates,k,sid,seed:list(enumerate(candidates))[:k]
            sys.modules['section_dataset']=ds
            road=ModuleType('road_vrp')
            for n in ('Road','OD','Objective'):setattr(road,n,object)
            road.legal=lambda *args:None
            road.read_json=lambda _:{}
            sys.modules['road_vrp']=road
            benchmark=ModuleType('benchmark');benchmark.load_tasks=lambda _:None
            sys.modules['benchmark']=benchmark
            core=ModuleType('run_experiment')
            core.NewSelector=Selector
            class Budget:
                def __init__(self,iters,wall):self.deadline=float('inf')
            core.BothBudget=Budget;core.check_alns=lambda:None
            sys.modules['run_experiment']=core
            spec=importlib.util.spec_from_file_location('_b2_online_test',BASE/'benchmark_30s.py')
            online=importlib.util.module_from_spec(spec);spec.loader.exec_module(online)
            obj=DummyObjective();obj.planner='static'
            base=np.random.default_rng(3).normal(size=(48,len(CORE)))
            y=np.linspace(-10,10,len(base))
            model=Predictor('reg',42,8,list(CORE),'task').fit(base,y,
              [{'state_id':f's{i//8}','task_id':f't{i//16}','date':'2025-07-22'} for i in range(48)])
            artifact={'model':model,'mode':'F0','threshold':-1000.}
            handler=online.OnlineSection(artifact,obj,None,'task_0001','sc_0800','2025-07-22 08:00')
            selector=Selector();selector.stats=Stats()
            selector.candidates=lambda routes,traffic=False:[{'route':0,'pos':0,'size':2,'static':15.},
                                                             {'route':1,'pos':1,'size':2,'static':18.}]
            old=Selector.destroy
            with online.temporary_hook(handler):
                self.assertIsNot(Selector.destroy,old)
                solution=selector.destroy('static',[[1,2,3],[4,5,6]],np.random.default_rng(42))
                self.assertEqual(len(solution.blocks),1)
                self.assertEqual(sum(len(r) for r in solution.routes),4)
            self.assertIs(Selector.destroy,old)
            self.assertIs(core.BothBudget,Budget)
            self.assertGreater(handler.stats['actions_scored'],0)
            # No-op gating returns a legitimate empty-block C3 partial.
            artifact['threshold']=float('inf')
            with online.temporary_hook(handler):
                empty=selector.destroy('static',[[1,2,3],[4,5,6]],np.random.default_rng(42))
                self.assertFalse(empty.blocks)
        finally:
            for name,old in original.items():
                if old is None:sys.modules.pop(name,None)
                else:sys.modules[name]=old

class TestPairedRoadComparison(unittest.TestCase):
    def test_road_gain_positive_for_lower_treatment_cost(self):
        from analyze_30s import compare
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';new=root/'new';base.mkdir();new.mkdir()
            mf={'source_sha256':'frozen','road_sha256':'cost','edge_sha256':'graph',
                'solver_sha256':'v3','section_sha256':'v3section','wall_sec':30,
                'max_iterations':20000,'anneal_iterations':5000,'top_k':12,
                'seeds':[42],'task_ids':['task_0008'],'scenario_ids':['sc1'],'split':'validation'}
            for d,score in ((base,100.),(new,90.)):
                (d/'manifest.json').write_text(json.dumps(mf))
                (d/'summary.json').write_text(json.dumps({'status':'COMPLETE','expected_records':1}))
                with (d/'metrics.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['task_id','scenario_id','seed','ex_post_road_travel_sec',
                         'iterations','section_feature_wall_sec','section_inference_wall_sec'])
                    w.writeheader();w.writerow({'task_id':'task_0008','scenario_id':'sc1','seed':42,
                         'ex_post_road_travel_sec':score,'iterations':50,
                         'section_feature_wall_sec':0.02,'section_inference_wall_sec':0.002})
            output=root/'compare.json'
            compare(base,new,output)
            result=json.loads(output.read_text())
            self.assertEqual(result['mean_road_travel_improvement_sec'],10.)
            self.assertEqual(result['wins'],1)
            with self.assertRaises(FileExistsError):compare(base,new,output)
