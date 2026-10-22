"""Mocked online adapter branch test (NOT frozen road-network integration)."""
from __future__ import annotations
import copy,importlib.util,sys,types,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class Stats:
    def __init__(self):self.d={'destroy_calls':0,'candidate_count':0,'destroy_sec':0.}
    def add(self,k,n=1):self.d[k]=self.d.get(k,0)+n
class Selector:
    def __init__(self):self.stats=Stats()
    def candidates(self,routes,traffic=False):
        return [{'route':0,'pos':j,'size':2,'static':float(10+j)} for j in range(8)]

class Block:
    def __init__(self,c):self.route,self.pos,self.ids=c['route'],c['pos'],(c['pos']+1,c['pos']+2)

def c3(selector,mode,routes,rng):
    selector.stats.add('destroy_calls');selector.stats.add('candidate_count',8)
    candidate=selector.candidates(routes)[int(rng.integers(8))]
    return SimpleNamespace(blocks=(Block(candidate),),action=candidate)

class FakeModel:
    def __init__(self,defer=False):self.defer=defer
    def select(self,X,B,same):
        if self.defer:return None,0.,'reject'
        j=int(np.flatnonzero(~same)[0]);return j,22.,'override_C3'

class OnlineBase:
    def __init__(self,artifact,obj,hist,task_id,scenario_id,departure_local,**kw):
        self.artifact=artifact;self.obj=obj;self.hist=hist;self.tid=task_id;self.scid=scenario_id
        self.action_count=kw.get('action_count',8);self.sampler_seed=kw.get('sampler_seed',1)
        self.score_every=1;self.max_scoring_sec=2.;self.scoring_elapsed=0.;self.fallback_elapsed=0.;self.budget=None
        self.stats={k:0 for k in ('calls','budget_short_count','r_scoring_calls','r_scoring_seconds',
          'r_model_actions','r_model_eval_calls','actions_scored','feature_wall_sec','inference_wall_sec',
          'gate_rejected','no_proxy_count','r_gate_fallbacks','r_c3_fallbacks','r_budget_fallbacks',
          'r_cadence_fallbacks','r_cheap_fallbacks')}
    def form(self,routes,c,base):return np.asarray([base,c['pos'],c['size']]+[0.]*23,np.float32)
    def _fallback(self,selector,mode,routes,rng,reason):
        self.stats['r_c3_fallbacks']+=1
        self.stats['r_'+reason+'_fallbacks']+=1
        return c3(selector,mode,routes,rng)


def import_stub():
    modules={}
    bench=types.ModuleType('benchmark_30s_r');bench.OnlineSectionR=OnlineBase
    bench.FROZEN_C3_DESTROY=c3
    bench.load_artifact=lambda *args:None
    bench.no_section=lambda routes,obj:SimpleNamespace(blocks=())
    bench.state_identifier=lambda t,s,r:'state'
    bench.make_candidate_partial=lambda routes,c,obj:SimpleNamespace(blocks=(Block(c),),action=c)
    modules['benchmark_30s_r']=bench
    sec=types.ModuleType('section_dataset');sec.select_actions=lambda cs,k,sid,seed:list(enumerate(cs[:k]))
    modules['section_dataset']=sec
    sec2=types.ModuleType('section_v2');sec2.Stats=Stats;modules['section_v2']=sec2
    ds=types.ModuleType('dataset');ds.sha256=lambda x:'sha';modules['dataset']=ds
    saved={k:sys.modules.get(k) for k in modules}
    try:
        sys.modules.update(modules)
        name='_online_r2_mock'
        spec=importlib.util.spec_from_file_location(name,ROOT/'online_r2.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        return module,saved
    finally:
        for k,v in saved.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v

class TestOnlineR2(unittest.TestCase):
    def test_shadow_defer_commits_same_real_C3(self):
        m,_=import_stub();rng=np.random.default_rng(42);baseline=np.random.default_rng(42)
        selector=Selector();expected=c3(Selector(),'static',[[1,2,3]],baseline)
        obj=SimpleNamespace(planner='static',cost=lambda x:123.)
        h=m.R2Section({'model':FakeModel(defer=True)},obj,None,'t','s','2025-07-02')
        actual=h.destroy(selector,'static',[[1,2,3]],rng)
        self.assertEqual(actual.action,expected.action)
        self.assertEqual(int(rng.integers(100)),int(baseline.integers(100)))
        self.assertEqual(selector.stats.d['destroy_calls'],1)
        self.assertEqual(h.stats['r_model_actions'],0)
        self.assertEqual(h.stats['r_c3_fallbacks'],1)

    def test_real_override_only_different_than_C3(self):
        m,_=import_stub();rng=np.random.default_rng(42);baseline=np.random.default_rng(42)
        expected=c3(Selector(),'static',[[1,2,3]],baseline)
        selector=Selector();obj=SimpleNamespace(planner='static',cost=lambda x:123.)
        h=m.R2Section({'model':FakeModel(defer=False)},obj,None,'t','s','2025-07-02')
        actual=h.destroy(selector,'static',[[1,2,3]],rng)
        self.assertNotEqual(actual.action,expected.action)
        self.assertEqual(int(rng.integers(100)),int(baseline.integers(100)))
        self.assertEqual(h.stats['r2_effective_overrides'],1)
        self.assertEqual(h.stats['r_c3_fallbacks'],0)
        self.assertEqual(selector.stats.d['destroy_calls'],1)

if __name__=='__main__':unittest.main()
