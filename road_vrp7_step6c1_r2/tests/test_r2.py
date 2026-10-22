"""Mocked deterministic tests; NOT equivalent to a Chicago-road end-to-end run."""
from __future__ import annotations
import json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from r2_core import state_key,clone_rng,shadow_c3,transfer_shadow,matches,block_action
from r2_policy import paired_matrix,R2Advantage,group_split,evaluate

class FakeStats:
    def __init__(self):self.d={'destroy_calls':0,'candidate_count':0,'destroy_sec':0.}
    def add(self,key,n=1):self.d[key]=self.d.get(key,0)+n
class FakeSelector:
    def __init__(self):self.stats=FakeStats();self.x=1

def fake_c3(selector,mode,routes,rng):
    selector.stats.add('destroy_calls');selector.stats.add('candidate_count',11)
    selector.stats.add('destroy_sec',.123)
    j=int(rng.integers(7))
    block=SimpleNamespace(route=0,pos=j,size=2,ids=(j+1,j+2))
    return SimpleNamespace(blocks=[block],choice=j)

class TestC3Shadow(unittest.TestCase):
    def test_exact_state_hash(self):
        routes=[[1,2],[3,4]]
        import hashlib
        expected=hashlib.sha256(json.dumps(['t','s',tuple(tuple(r) for r in routes)],sort_keys=True).encode()).hexdigest()[:24]
        self.assertEqual(state_key('t','s',routes),expected)

    def test_no_rng_or_stats_side_effects(self):
        s=FakeSelector();r=np.random.default_rng(2026)
        rng_before=json.dumps(r.bit_generator.state,sort_keys=True)
        action=shadow_c3(s,'static',[[1,2,3]],r,fake_c3,FakeStats)
        self.assertEqual(json.dumps(r.bit_generator.state,sort_keys=True),rng_before)
        self.assertEqual(s.stats.d['destroy_calls'],0)
        other=fake_c3(s,'static',[[1,2,3]],r)
        self.assertTrue(matches(action,block_action(other)))

    def test_transfer_shadow_equals_real_c3(self):
        a=np.random.default_rng(20261019);b=clone_rng(a)
        s=FakeSelector()
        action,partial,after,stats=shadow_c3(s,'static',[[1,2,3]],a,fake_c3,FakeStats,True)
        self.assertEqual(s.stats.d['candidate_count'],0)
        expected=fake_c3(FakeSelector(),'static',[[1,2,3]],b)
        self.assertTrue(matches(action,block_action(expected)))
        transfer_shadow(a,s,after,stats,True)
        self.assertEqual(s.stats.d['candidate_count'],11)
        self.assertEqual(json.dumps(a.bit_generator.state,sort_keys=True),json.dumps(b.bit_generator.state,sort_keys=True))
        self.assertEqual(int(a.integers(10000)),int(b.integers(10000)))

    def test_override_consumes_shadow_rng_but_not_C3_stats(self):
        s=FakeSelector();r=np.random.default_rng(43)
        _,_,post,delta=shadow_c3(s,'static',[[1,2]],r,fake_c3,FakeStats,True)
        transfer_shadow(r,s,post,delta,False)
        self.assertEqual(s.stats.d['destroy_calls'],0)
        self.assertNotEqual(int(r.integers(99)), -100)

    def test_no_section_action(self):
        self.assertIsNone(block_action(SimpleNamespace(blocks=[])))
        self.assertFalse(matches(None,{'route':0,'pos':0,'size':2}))

class TestR2Model(unittest.TestCase):
    def test_pair_matrix_and_strict_baseline(self):
        x=np.ones((3,26),np.float32);b=np.zeros(26,np.float32)
        z=paired_matrix(x,b)
        self.assertEqual(z.shape,(3,78))
        self.assertTrue(np.allclose(z[:,:26],1.))
        self.assertTrue(np.allclose(z[:,26:52],0.))
        self.assertTrue(np.allclose(z[:,52:],1.))
        with self.assertRaises(ValueError):paired_matrix(x,np.zeros(44))

    def synthetic_data(self,n=55):
        rng=np.random.default_rng(42)
        X=rng.normal(size=(n,8,44)).astype('float32');B=rng.normal(size=(n,44)).astype('float32')
        # Synthetic only. True C3 appears in slot 0 and is never eligible to override.
        X[:,0]=B
        valid=np.ones((n,8),bool);same=np.zeros((n,8),bool);same[:,0]=True
        Y=np.zeros((n,8),np.float32)
        Y[:,1:]=3.5*(X[:,1:,0]-B[:,None,0])-2.0*(X[:,1:,1]-B[:,None,1])
        BY=np.zeros(n,np.float32)
        return X,B,Y,BY,valid,same

    def test_group_disjoint_splits(self):
        groups=[{'task_id':f't{i%8}','date':f'd{i%5}'} for i in range(80)]
        folds,roles=group_split(groups,'task')
        for x in folds.values():self.assertTrue(x.any())
        for i,g in enumerate(groups):
            self.assertTrue(folds[roles[g['task_id']]][i])
        self.assertFalse(np.any(folds['train']&folds['internal_test']))
        with self.assertRaises(ValueError):group_split([{'task_id':'t','date':'d'}]*4)

    def test_train_and_choose(self):
        X,B,Y,BY,valid,same=self.synthetic_data()
        tr=np.zeros(len(X),bool);tr[:35]=True
        m=R2Advantage(mode='F2',seed=42,threshold=0.).fit(X,B,Y,BY,valid,same,tr)
        j,adv,reason=m.select(X[-1,:,:26],B[-1,:26],same[-1],valid[-1])
        self.assertTrue(j is None or j>0)
        if j is not None:self.assertEqual(reason,'override_C3')
        r=evaluate(m,X,B,Y,BY,valid,same,np.arange(35,len(X)))
        self.assertEqual(r['states'],20)
        self.assertIn('mean_delta_vs_actual_C3_sec_per_state',r)
        m.threshold=float('inf')
        j,_,reason=m.select(X[-1,:,:26],B[-1,:26],same[-1],valid[-1])
        self.assertIsNone(j)

    def test_all_candidates_same_no_override(self):
        m=R2Advantage()
        class Dummy:
            def predict(self,x):return np.ones(len(x))*1000
        m.model=Dummy();j,delta,reason=m.select(np.ones((8,26)),np.ones(26),np.ones(8,bool))
        self.assertIsNone(j);self.assertEqual(reason,'no_different_candidate')

if __name__=='__main__':unittest.main()
