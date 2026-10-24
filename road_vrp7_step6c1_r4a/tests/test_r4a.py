import copy
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from r4a_core import SelectorObserver,canonical_hash,inspect_anneal,STAGE,write_json
from compare_r4a import summary_pair

class State:
    def __init__(self,v):self.routes=((1,2),(3,4));self.v=float(v)
    def objective(self):return self.v
class Wheel:
    def __init__(self):
        self.destroy_weights=np.ones(4)
        self.repair_weights=np.ones(2)
        self.decay=.8;self.scores=[8,4,1,0];self.updates=0
    def __call__(self,rng,best,curr):return int(rng.integers(4)),int(rng.integers(2))
    def update(self,cand,d_idx,r_idx,outcome):self.updates+=1
class Anneal:
    def __init__(self):
        self._temperature=100.
        self._start_temperature=100.
        self._end_temperature=1.
        self._step=.9
        self._method='exponential'
class Stop:
    count=1
    start=0.
    deadline=30.
    max_iter=2000
    wall=30.

def make_accept():
    anneal=Anneal()
    class Acceptance:
        def __call__(self,*args):return anneal._temperature>0
    return Acceptance()

class TestR4A(unittest.TestCase):
    def test_closure_exact_anneal(self):self.assertEqual(inspect_anneal(make_accept())._temperature,100)
    def test_observer_does_not_advance_rng(self):
        a,b=np.random.default_rng(42),np.random.default_rng(42)
        w=Wheel();obs=SelectorObserver(w,make_accept(),Stop(),[1])
        result=obs(a,State(12),State(14));expected=w(b,State(12),State(14))
        self.assertEqual(result,expected)
        self.assertEqual(canonical_hash(a.bit_generator.state),canonical_hash(b.bit_generator.state))
        self.assertEqual(obs.snapshots[0]['next_iteration_one_based'],1)
        self.assertEqual(obs.snapshots[0]['completed_updates_before_boundary'],0)
        self.assertEqual(obs.snapshots[0]['roulette_destroy_weights'],[1.]*4)
        self.assertEqual(obs.snapshots[0]['anneal_temperature_before_accept'],100.)
    def test_update_propagation(self):
        w=Wheel();o=SelectorObserver(w,make_accept(),Stop(),[1]);o.update(State(0),2,1,3)
        self.assertEqual(w.updates,1);self.assertEqual(o.updates,1)
    def test_section_event_before_selection_rng(self):
        class PredictedWheel(Wheel):
            def __init__(self):
                super().__init__();self.k=0
            def __call__(self,rng,best,curr):
                self.k+=1
                return (3 if self.k==3 else 2),int(rng.integers(2))
        a,b=np.random.default_rng(10),np.random.default_rng(10)
        w=PredictedWheel();o=SelectorObserver(w,make_accept(),Stop(),[1],section_only=True)
        for j in range(1,4):
            Stop.count=j
            result=o(a,State(3),State(4))
            b.integers(2)  # original Wheel uses exactly one random draw
        self.assertEqual(canonical_hash(a.bit_generator.state),canonical_hash(b.bit_generator.state))
        self.assertEqual(o.snapshots[0]['next_iteration_one_based'],3)
        self.assertEqual(o.snapshots[0]['requested_milestone'],1)
        self.assertEqual(o.snapshots[0]['real_selected_destroy_index'],3)
    def test_copy_routes(self):
        r=State(2);o=SelectorObserver(Wheel(),make_accept(),Stop(),[1]);o(np.random.default_rng(1),r,r)
        self.assertEqual(o.snapshots[0]['current_routes'],[[1,2],[3,4]])
        self.assertEqual(o.snapshots[0]['current_route_sha256'],canonical_hash([[1,2],[3,4]]))
    def _pair(self,base,*,alter=False,iteration_limited=True):
        paths=[]
        for arm in ('plain','capture'):
            path=Path(base)/arm;path.mkdir()
            mf={k:'fixed' for k in ('task_ids','scenario_ids','seeds','search_wall_sec','max_iterations','anneal_iterations','top_k','source_sha256','road_sha256','edge_sha256','solver_sha256','section_sha256','alns_version','split')}
            mf.update({'stage':STAGE,'arm':arm})
            write_json(path/'manifest.json',mf)
            write_json(path/'report.json',{'status':'COMPLETE'})
            a={'task_id':'task_0008','scenario_id':'2025-07-22_0800','seed':60,'iterations':2000 if iteration_limited else 1700,
                'search_wall_sec':10.,'termination_reason':'iteration_budget' if iteration_limited else 'wall_budget',
                'best_route_sha256':'abcd' if arm=='plain' or not alter else 'defg',
                'road_cost_sec':3890. if arm=='plain' or not alter else 3893.,'captured':0}
            (path/'metrics.jsonl').write_text(json.dumps(a)+'\n')
            s={'task_id':'task_0008','scenario_id':'2025-07-22_0800','seed':60,
               'rng_bit_generator_state':{'state':5},'current_routes':[[1],[2]],'best_routes':[[1],[2]],
               'completed_updates_before_boundary':99,'next_iteration_one_based':100,'stop_count':100}
            (path/'snapshots.jsonl').write_text(json.dumps(s)+'\n' if arm=='capture' else '')
            paths.append(path)
        return paths
    def test_exact_pair_audit(self):
        with TemporaryDirectory() as d:
            a,b=self._pair(d);r=summary_pair(a,b,require_exact=True)
            self.assertEqual(r['strong_determinism_passes'],1)
    def test_exact_mismatch_fails_closed(self):
        with TemporaryDirectory() as d:
            a,b=self._pair(d,alter=True)
            with self.assertRaises(AssertionError):summary_pair(a,b,require_exact=True)
    def test_wall_limited_cannot_prove_exactness(self):
        with TemporaryDirectory() as d:
            a,b=self._pair(d,iteration_limited=False)
            r=summary_pair(a,b)
            self.assertEqual(r['strong_determinism_passes'],0)
    def test_source_identity_mismatch(self):
        with TemporaryDirectory() as d:
            a,b=self._pair(d);x=json.loads((b/'manifest.json').read_text());x['road_sha256']='changed';write_json(b/'manifest.json',x)
            with self.assertRaises(ValueError):summary_pair(a,b)
    def test_iterate_hook_restore_and_noninterference(self):
        from r4a_core import instrument_official_alns
        fake_mod=types.ModuleType('alns')
        class ALNS:
            def iterate(self, init, selector, accept, stop):
                rng=np.random.default_rng(34)
                for j in range(4):
                    stop.count=j+1
                    d,r=selector(rng,init,init)
                    selector.update(init,d,r,1)
                return 7
        fake_mod.ALNS=ALNS
        old=sys.modules.get('alns')
        sys.modules['alns']=fake_mod
        try:
            f=ALNS.iterate
            with instrument_official_alns([1,3]) as instances:
                z=ALNS().iterate(State(1),Wheel(),make_accept(),Stop())
                self.assertEqual(z,7)
            self.assertIs(ALNS.iterate,f)
            self.assertEqual(len(instances),1)
            self.assertEqual(instances[0].updates,4)
            self.assertEqual(len(instances[0].snapshots),2)
        finally:
            if old is None:sys.modules.pop('alns',None)
            else:sys.modules['alns']=old

if __name__=='__main__':unittest.main()
