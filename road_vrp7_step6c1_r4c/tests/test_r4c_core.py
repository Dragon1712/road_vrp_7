from __future__ import annotations
import copy
import json
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r4c_core import (candidate_identity,partial_identity,make_partial,validate_sham,
    select_alternatives,branch_spec,unit_name,inspect_events,compare_sham,
    normalize_reward,summarize)
from counterfactual_r4c import first_section_substitution,write_summary

@contextmanager
def mock_section():
    """Pure stand-in with original Section draw consuming an RNG integer."""
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Block:
        ids:tuple
        route:int
        pos:int
    @dataclass(frozen=True)
    class Partial:
        routes:tuple
        obj:object
        blocks:tuple
        original:tuple
    def frozen(x):return tuple(tuple(int(c) for c in r) for r in x)
    def mutable(x):return [list(r) for r in x]
    class Selector:
        def __init__(self,obj):self.obj=obj
        def destroy(self,mode,routes,rng):
            assert mode=='static'
            j=int(rng.integers(2))
            r=frozen(routes);b=Block(tuple(r[0][j:j+2]),0,j)
            new=mutable(r);del new[0][j:j+2]
            return Partial(frozen(new),self.obj,(b,),r)
    fake=types.ModuleType('section_v2')
    fake.Block=Block;fake.Partial=Partial;fake.frozen=frozen;fake.mutable=mutable;fake.Selector=Selector
    with patch.dict(sys.modules,{'section_v2':fake}):yield fake

class TestR4CCore(unittest.TestCase):
    def setUp(self):
        self.routes=[[1,2,3,4],[5,6,7],[8,9,10]]
        self.s={'task_id':'task_1','scenario_id':'sc1','seed':60,'stop_count':505,
                'real_selected_repair_index':0,'current_route_sha256':'shaC',
                'best_route_sha256':'shaB','rng_state_sha256':'rng'}

    def test_partial_construct_and_sham(self):
        with mock_section() as m:
            rng=np.random.default_rng(5);obj=object();sec=m.Selector(obj)
            baseline=sec.destroy('static',self.routes,rng)
            j=partial_identity(baseline)
            restored=make_partial(self.routes,{'route':j[0],'pos':j[1],'size':j[2]},obj)
            self.assertTrue(validate_sham(baseline,restored))
            self.assertEqual(baseline,restored)

    def test_sham_preserves_rng_draws(self):
        with mock_section() as m:
            obj=object();rng0=np.random.default_rng(19);rng1=np.random.default_rng(19)
            sec=m.Selector(obj)
            original=sec.destroy('static',self.routes,rng0)
            with first_section_substitution('SHAM_C3',None,partial_identity(original)) as audit:
                sham=sec.destroy('static',self.routes,rng1)
            self.assertEqual(original,sham)
            self.assertEqual(rng0.bit_generator.state,rng1.bit_generator.state)
            self.assertTrue(audit['sham_partial_equal'])
            self.assertEqual(audit['substitution_calls'],1)

    def test_alternative_changes_only_partial_not_rng(self):
        with mock_section() as m:
            obj=object();rng0=np.random.default_rng(2);rng1=np.random.default_rng(2)
            sec=m.Selector(obj);orig=sec.destroy('static',self.routes,rng0)
            chosen={'name':'A077','route':0,'pos':0,'size':2}
            if partial_identity(orig)==(0,0,2):chosen['pos']=1
            with first_section_substitution('A077',chosen,partial_identity(orig)) as audit:
                different=sec.destroy('static',self.routes,rng1)
            self.assertNotEqual(orig.routes,different.routes)
            self.assertEqual(rng0.bit_generator.state,rng1.bit_generator.state)
            self.assertEqual(audit['first_chosen_section'],[0,chosen['pos'],2])

    def test_fail_on_wrong_shadow_C3(self):
        with mock_section() as m:
            obj=object();rng=np.random.default_rng(8)
            with self.assertRaises(ValueError):
                with first_section_substitution('SHAM_C3',None,(2,8,3)):
                    m.Selector(obj).destroy('static',self.routes,rng)
            # patch always restored even on exception
            self.assertEqual(m.Selector.__dict__['destroy'].__name__,'destroy')

    def test_deterministic_candidate_sampling(self):
        cs=[{'route':0,'pos':i,'size':2} for i in range(5)]
        sampled=[(1,cs[1]),(1,cs[1]),(2,cs[2]),(0,cs[0]),(4,cs[4])]
        self.assertEqual([x['name'] for x in select_alternatives(cs,sampled,(0,2,2),3)],
                         ['A001','A000','A004'])
        with self.assertRaises(ValueError):select_alternatives(cs,[(100,cs[0])],(0,2,2),2)

    def test_snapshot_and_event_gate(self):
        events=[{'stop_count':505+i,'current_route_sha256':'shaC',
                 'best_route_sha256':'shaB','rng_state_sha256':'rng',
                 'destroy_index':3 if i==0 else 0,'repair_index':0} for i in range(3)]
        self.assertTrue(inspect_events(events,self.s,3))
        broken=copy.deepcopy(events);broken[1]['stop_count']=999
        with self.assertRaises(ValueError):inspect_events(broken,self.s,3)

    def test_sham_trace_comparator(self):
        x={'events':[{'a':1},{'a':2}],'end_signature':{'state':5},'first_action_id':[0,1,2],
           'best_route_sha256':'a','current_route_sha256':'b',
           'road_best_sec':2.,'road_current_sec':2.5,'static_best_sec':2.,'static_current_sec':3.}
        self.assertTrue(compare_sham(x,copy.deepcopy(x)))
        bad=copy.deepcopy(x);bad['events'][0]['a']=7
        with self.assertRaises(ValueError):compare_sham(x,bad)

    def test_rewards_and_statistics(self):
        p={'task_id':'a','scenario_id':'b','seed':5,'stop_count':4}
        b={'snapshot_spec':p,'horizon':20,'arm':'C3','road_best_sec':100.,
           'road_current_sec':110.,'static_best_sec':90.,'static_current_sec':105.,
           'best_route_sha256':'x','current_route_sha256':'y'}
        a={**b,'arm':'A002','road_best_sec':150.,'road_current_sec':99.,
           'static_best_sec':80.,'static_current_sec':95.,'best_route_sha256':'z'}
        y=normalize_reward(b,a)
        self.assertEqual(y['road_best_gain_sec'],-50.)
        self.assertEqual(y['road_current_gain_sec'],11.)
        self.assertTrue(y['is_severe_harm'])
        self.assertEqual(summarize([y,y])['severe_harm_fraction'],1.)
        with self.assertRaises(ValueError):normalize_reward(a,b)

    def test_unit_identity_and_reporting(self):
        s=branch_spec(self.s,5,'A003')
        self.assertNotEqual(unit_name(s),unit_name(branch_spec(self.s,20,'A003')))
        with __import__('tempfile').TemporaryDirectory() as tmp:
            out=Path(tmp)
            write_summary(out,[{'unit_spec':s}],
               [{'unit_spec':s,'road_best_gain_sec':1.,'is_severe_harm':False}],
               [{'task_id':'x'}],True)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'COMPLETE')
            self.assertEqual(report['n_counterfactual_labels'],1)

if __name__=='__main__':unittest.main()
