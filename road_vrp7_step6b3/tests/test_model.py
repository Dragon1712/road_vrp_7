import sys,unittest
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from model import (FEATURES,STRUCTURE,TRAFFIC,RELIABILITY,from_full,
      ResidualFusionPredictor,group_oof_anchor,TinyFiLMResidual,torch)

class FakeAnchor:
    def predict(self,x):
        score=30 + x[:,14]*.2 + x[:,11]*.1
        return score,score,None

def make_rows(n=144,seed=88):
    g=np.random.default_rng(seed)
    x=g.normal(size=(n,len(FEATURES)))
    names=list(FEATURES)
    for key in ('removal_observed_share','insertion_observed_share'):
        x[:,names.index(key)]=g.uniform(.05,.9,n)
    for k in ('removal_all_missing','insertion_all_missing'):
        x[:,names.index(k)]=0.
    x[:,names.index('insert_proxy_static_added_sec')]=g.uniform(12,160,n)
    x[:,names.index('remove_static_saved_sec')]=g.uniform(15,165,n)
    return x

class TestB3(unittest.TestCase):
    def test_schema_and_boundary_gate(self):
        x=make_rows()
        s,t,m,q=from_full(x)
        self.assertEqual(s.shape,(144,26));self.assertEqual(t.shape,(144,10))
        self.assertEqual(m.shape,(144,8));self.assertTrue((q>=0).all() and (q<=1).all())
        x[0,list(FEATURES).index('insertion_observed_share')]=0.
        self.assertEqual(from_full(x)[-1][0],0.)
        with self.assertRaises(ValueError):from_full(x[:,:-1])
        x[0,0]=np.nan
        with self.assertRaises(ValueError):from_full(x)

    def test_m0_is_exact_frozen_anchor(self):
        x=make_rows();a=FakeAnchor();model=ResidualFusionPredictor(a,'M0')
        model.prepare(x,np.zeros(len(x)))
        z,gate,_=model.predict(x)
        expected=a.predict(x[:,:26])[0]
        np.testing.assert_allclose(z,expected)
        np.testing.assert_allclose(gate,z)

    def test_zero_data_reverts_to_static(self):
        x=make_rows();x[:,list(FEATURES).index('removal_observed_share')]=0
        x[:,list(FEATURES).index('insertion_observed_share')]=0
        y=np.random.default_rng(3).normal(size=len(x))
        for method in ('M2','M3','M4'):
            model=ResidualFusionPredictor(FakeAnchor(),method,seed=21)
            model.prepare(x,y,epochs=3)
            model.alpha=1.5
            z,_,_=model.predict(x)
            np.testing.assert_allclose(z,FakeAnchor().predict(x[:,:26])[0],atol=1e-5,
                                       err_msg=method)

    def test_group_oof(self):
        x=make_rows(96)
        rng=np.random.default_rng(13)
        y=x[:,14]+rng.normal(size=96)
        g=[{'task_id':f'task_{i//16}','date':f'2025-07-{22+i//48:02d}'} for i in range(96)]
        a=group_oof_anchor(x[:,:26],y,g,'task',seed=2,trees=10)
        self.assertEqual(a.shape,(96,));self.assertTrue(np.isfinite(a).all())
        b=group_oof_anchor(x[:,:26],y,g,'date',seed=2,trees=10)
        self.assertTrue(np.isfinite(b).all())
        g=[dict(task_id='same',date='same') for _ in range(96)]
        with self.assertRaises(ValueError):group_oof_anchor(x[:,:26],y,g,'task')

    def test_film_variation_and_serialization(self):
        x=make_rows(120);res=np.random.default_rng(3).normal(0,14,120)
        for method in ('M1','M2','M3','M4'):
            m=ResidualFusionPredictor(FakeAnchor(),method,seed=5)
            m.prepare(x,res,epochs=4)
            m.alpha=0.5
            z,gate,_=m.predict(x)
            self.assertTrue(np.isfinite(z).all())
            np.testing.assert_array_equal(z,gate)
            import joblib,tempfile
            with tempfile.TemporaryDirectory() as folder:
                file=Path(folder)/'test.joblib';joblib.dump(m,file)
                loaded=joblib.load(file);np.testing.assert_allclose(loaded.predict(x)[0],z,rtol=1e-5)

    def test_film_identity_initialization(self):
        net=TinyFiLMResidual()
        s=torch.randn(4,26);t=torch.randn(4,18)
        with torch.no_grad():z=net(s,t)
        self.assertTrue(torch.all(z==0))

if __name__=='__main__':unittest.main()
