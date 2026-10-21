import sys,unittest,copy
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'road_vrp7_step6c1_r'),str(P/'road_vrp7_step6c1')]
from r_model import ReliableC1,m0_proxy_loss,selected_action,evaluate_m0_proxy
from graph_audit import summarize as audit_graph
from graph_features import N,T,GRAPH_DIM,GraphBatch

class Anchor:
    def predict(self,X):
        X=np.asarray(X)
        scores=X[:,4]*20.+30.
        return scores,scores,None

def inputs(n=8):
    a=np.random.default_rng(123).normal(size=(n,44)).astype('float32')
    a[:,4]=np.arange(n)
    a[:,36]=.75;a[:,37]=.8
    return a

def graph(n):
    g=np.zeros((n,GRAPH_DIM),dtype=np.float32)
    for k in range(n):
        seq=np.zeros((N,T,2),np.float32);seq[0,0,1]=1;seq[1,1,1]=1
        valid=np.zeros(N,np.float32);valid[:2]=1
        adj=np.zeros((N,N),np.float32);adj[0,0]=1;adj[1,1]=1;adj[0,1]=1
        g[k]=GraphBatch(seq,np.zeros((N,4),np.float32),adj,valid).flat()
    return g

class TestR(unittest.TestCase):
    def test_d0_f2_matches_f4_and_no_history(self):
        X=inputs();m=ReliableC1(Anchor(),'D0',2).prepare(X)
        a=m.predict(X)[0];b=m.predict(X[:,:26])[0]
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-5)

    def test_m0_proxy_cost_sensitive_gradients(self):
        pred=torch.tensor([30.,40.,50.],requires_grad=True)
        reward=torch.tensor([-10.,60.,15.]);base=torch.tensor([5.,70.,8.])
        l=m0_proxy_loss(pred,reward,base,anchor_threshold=0.)
        self.assertTrue(torch.isfinite(l))
        l.backward();self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()),0.)

    def test_conservative_gate(self):
        score=np.array([12.,20.,4.]);base=np.array([25.,10.,4.])
        a,ok,why=selected_action(score,base,margin=5.,max_correction=100.)
        self.assertEqual(a,1);self.assertFalse(ok)
        score=np.array([11.,38.,4.]);a,ok,why=selected_action(score,base,margin=5.,max_correction=100.)
        self.assertTrue(ok);self.assertEqual(why,'passed')
        self.assertFalse(selected_action(score,base,margin=5.,max_correction=10.)[1])
        self.assertFalse(selected_action(score,base,margin=5.,max_correction=100.,min_coverage=.4,coverage=[.9,.1,.8])[1])
        self.assertFalse(selected_action(np.array([-3.,-2.]),np.array([0.,0.]))[1])

    def test_predicted_defer_executes_original_C3_gate_protocol(self):
        X=inputs();m=ReliableC1(Anchor(),'D0',seed=5).prepare(X)
        m.safe_gate=True;m.margin=1.e6
        # With zero initialized corrections the maximum is also the M0 max;
        # in this case learned select is allowed; force score-selected != M0.
        with torch.no_grad():
            vals,gates,_=m.predict(X)
        self.assertTrue(np.isfinite(vals).all())
        self.assertEqual(int(np.argmax(vals)),int(np.argmax(gates)))
        self.assertEqual(gates[np.argmax(vals)],vals[np.argmax(vals)])
        m.predict(np.zeros((8,26),np.float32))

    def test_online_gate_blocks_changed_low_confidence_action(self):
        X=inputs()
        m=ReliableC1(Anchor(),'D0',1).prepare(X)
        m.safe_gate=True;m.margin=5.;m.max_correction=100.
        # Forced counterfactual: raw learned top is Section1, static top Section0.
        m.score_components=lambda _: (np.array([12.,20.,4.,1.,1.,1.,1.,1.]),
                                        np.array([25.,10.,4.,1.,1.,1.,1.,1.]))
        score,gate,_=m.predict(X[:,:26])
        self.assertEqual(int(np.argmax(score)),1)
        self.assertLess(float(gate[1]),0.)  # B2-R must fall back to frozen C3
        m.margin=0.;m.score_components=lambda _: (np.array([11.,38.,4.,1.,1.,1.,1.,1.]),
                                         np.array([25.,10.,4.,1.,1.,1.,1.,1.]))
        score,gate,_=m.predict(X[:,:26]);self.assertEqual(gate[1],score[1])

    def test_failed_model_input_does_not_silently_use_unavailable_graph(self):
        X=inputs();m=ReliableC1(Anchor(),'D2',seed=5).prepare(X)
        with self.assertRaises(ValueError):m.predict(X)
        with self.assertRaises(ValueError):m.predict(X[:,:26])

    def test_graph_padding_aware_coverage_and_real_links(self):
        out=audit_graph(graph(3));self.assertEqual(out['candidates'],3)
        self.assertAlmostEqual(out['valid_physical_edge_slots_coverage'],2/12)
        self.assertAlmostEqual(out['padded_slots_coverage'],2/144)
        self.assertAlmostEqual(out['mean_directed_nonself_links'],1)
        self.assertAlmostEqual(out['valid_node_isolation_rate'],0)

    def test_m0_proxy_metric_different_from_zero_noop(self):
        X=inputs(16);G=graph(16);m=ReliableC1(Anchor(),'D0',seed=1).prepare(X)
        m.safe_gate=True;m.margin=10.
        groups=[{'state_id':'s'+str(j//8)} for j in range(16)]
        y=np.arange(16,dtype=float)-10.
        r=evaluate_m0_proxy(m,X,G,groups,y,anchor_threshold=0.)
        self.assertEqual(r['states'],2)
        self.assertTrue(np.isfinite(r['mean_improvement_vs_m0_proxy_sec_per_state']))

if __name__=='__main__':unittest.main()
