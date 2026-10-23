import sys,unittest,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).absolute().parents[2]
for name in ('road_vrp7_step6c1_r3_tricks','road_vrp7_step6c1_r2'):
    sys.path.insert(0,str(ROOT/name))
from r3_models import prepare_group,R3Advantage,make_blend
from decision_metrics import evaluate,select_gate


def make(n=48,seed=42):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n,8,44)).astype(np.float32)
    B=rng.normal(size=(n,44)).astype(np.float32)
    gain=25*X[:,:,8]-12*X[:,:,9]+10*X[:,:,12]+rng.normal(size=(n,8))*3
    Y=gain.astype(np.float32);BY=np.zeros(n,np.float32)
    same=np.zeros((n,8),bool);same[:,7]=True
    valid=np.ones((n,8),bool);valid[:,6]=False
    return X,B,Y,BY,valid,same

class FixedScorer:
    def __init__(self,v):self.v=np.asarray(v,float)
    def predict_delta(self,a,b):return self.v[:len(a)]

class TestR3(unittest.TestCase):
    def test_group_c3_anchor_and_mask(self):
        X,B,Y,BY,valid,same=make()
        features,y,m=prepare_group(X,B,Y,BY,valid,same,[0,1])
        self.assertEqual(features.shape,(2,9,78));self.assertTrue(np.all(m[:,8]));self.assertFalse(m[0,7]);self.assertFalse(m[0,6]);self.assertTrue(np.all(y[:,8]==0))

    def test_synthetic_catboost_huber(self):
        X,B,Y,BY,valid,same=make()
        mask=np.arange(len(X))<35
        for algo in ('catboost_rmse','catboost_huber'):
            a=R3Advantage(mode='F2',algorithm=algo)
            a.fit(X,B,Y,BY,valid,same,mask)
            p=a.predict_delta(X[36,:,:26],B[36,:26])
            self.assertEqual(p.shape,(8,));self.assertTrue(np.isfinite(p).all())
            stats=evaluate(a,X,B,Y,BY,valid,same,np.arange(35,48))
            self.assertGreaterEqual(stats['mean_regret_sec'],0)

    def test_pairwise_and_weighted_forward(self):
        X,B,Y,BY,valid,same=make()
        mask=np.arange(len(X))<35
        for algo in ('pairwise','weighted_pairwise'):
            a=R3Advantage(mode='F2',algorithm=algo,seed=42)
            a.fit(X,B,Y,BY,valid,same,mask,epochs=4)
            scores=a.predict_delta(X[37,:,:26],B[37,:26])
            self.assertTrue(np.isfinite(scores).all());self.assertEqual(scores.shape,(8,))

    def test_margin_rejects_unstable_rank(self):
        b=np.zeros(26,np.float32);x=np.zeros((3,26),np.float32)
        a=R3Advantage(algorithm='blend',components=[FixedScorer([20.,19.,-5.]),FixedScorer([20.,19.,-5.])])
        a.threshold=0.;a.margin=0.;self.assertEqual(a.select(x,b)[0],0)
        a.margin=2.;self.assertIsNone(a.select(x,b)[0])
        a.margin=0.;a.threshold=21.;self.assertIsNone(a.select(x,b)[0])

    def test_defer_rejects_same_and_invalid(self):
        x=np.zeros((3,26),np.float32);b=np.zeros(26,np.float32)
        a=R3Advantage(algorithm='blend',components=[FixedScorer([999.,-1.,-2.]),FixedScorer([999.,-1.,-2.])])
        a.threshold=0.;a.margin=0.
        self.assertIsNone(a.select(x,b,same=np.array([1,0,0]))[0])
        self.assertIsNone(a.select(x,b,valid=np.array([0,0,0]))[0])

    def test_validation_gate_no_test_input(self):
        X,B,Y,BY,valid,same=make()
        a=R3Advantage(mode='F2',algorithm='catboost_rmse').fit(X,B,Y,BY,valid,same,np.arange(48)<25)
        train_ids=np.arange(25,35)
        b=copy.deepcopy(a)
        a,grid,best=select_gate(a,X,B,Y,BY,valid,same,train_ids,margins=[0,10])
        Y2=Y.copy();Y2[35:]=1e6
        b,grid2,best2=select_gate(b,X,B,Y2,BY,valid,same,train_ids,margins=[0,10])
        self.assertEqual(best['threshold'],best2['threshold']);self.assertEqual(best['margin'],best2['margin'])

if __name__=='__main__':unittest.main()
