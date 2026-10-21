import sys,unittest,types,importlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from graph_features import N,T,GRAPH_DIM,GraphBatch,encode_raw_graph,unpack
from c1_model import C1Model,DirectedMaskedEncoder,state_loss

class FakeAnchor:
    def predict(self,X):
        x=np.asarray(X)
        z=10.+x[:,4]*2.+x[:,13]*.05
        return z,z,None

def fake_x(rows,seed=12):
    g=np.random.default_rng(seed)
    x=g.normal(size=(rows,44)).astype(np.float32)
    x[:,36]=g.random(rows)
    x[:,37]=g.random(rows)
    return x

def graph_matrix(rows):
    z=np.zeros((rows,GRAPH_DIM),np.float32)
    for i in range(rows):
        seq=np.zeros((N,T,2),np.float32)
        seq[0,1:4,0]=.35+i*.001;seq[0,1:4,1]=1.
        meta=np.zeros((N,4),np.float32);meta[0,0]=1.
        adj=np.zeros((N,N),np.float32);adj[0,0]=1.
        valid=np.zeros(N,np.float32);valid[0]=1
        z[i]=GraphBatch(seq,meta,adj,valid).flat()
    return z

class TestGraph(unittest.TestCase):
    def test_graph_schema(self):
        z=graph_matrix(5)
        self.assertEqual(z.shape,(5,GRAPH_DIM))
        seq,meta,adj,valid=unpack(z)
        self.assertEqual(seq.shape,(5,N,T,2))
        self.assertEqual(adj.shape,(5,N,N))
        self.assertEqual(float(seq[0,0,1,1]),1.)

    def test_strict_preddeparture_osm_direction(self):
        edges=pd.DataFrame(dict(osm_u=['a','b','c','x'],osm_v=['b','c','d','y']))
        road=SimpleNamespace(observed=np.zeros((12,4),bool),static=np.ones(4)*10.,
                edges=edges,dt=10.,seconds=lambda t:float(t),
                path=lambda a,b,_,mode:(30,[0,1,2]) if (a,b)==('a','d') else (10,[3]))
        road.observed[7,0]=True
        road.observed[9,1]=True
        road.observed[10,2]=True # CURRENT/FUTURE: must NEVER enter the output.
        raw=np.ones((12,4))*10
        raw[7,0]=13
        raw[9,1]=16
        raw[10,2]=99
        hist=SimpleNamespace(index_anchor=9,n_slots=6,road=road,raw=raw,
             key=('t',100),od=SimpleNamespace(nodes=['a','d','x'],start=100,task={'task_id':'t'}),paths={})
        b=encode_raw_graph([(0,1,1,0)],hist)
        self.assertEqual(b.valid.sum(),3)
        self.assertEqual(b.seq[:,:,1].sum(),2)
        self.assertAlmostEqual(b.seq[0,3,0],.3)
        self.assertAlmostEqual(b.seq[1,5,0],.6)
        self.assertEqual(float(b.seq[2,:,0].sum()),0.)
        self.assertEqual(b.adj[0,1],1.)
        self.assertEqual(b.adj[1,0],0.)

    def test_masked_temporal_attention_changes_with_observed_values(self):
        enc=DirectedMaskedEncoder()
        G=unpack(graph_matrix(2))
        tensors=tuple(torch.from_numpy(v.copy()) for v in G)
        with torch.no_grad():
            a,h=enc(*tensors)
            self.assertEqual(a.shape,(2,24))
            self.assertTrue(torch.isfinite(a).all())
        seq=tensors[0].clone();seq[...,1]=0
        with torch.no_grad():
            b,h=enc(seq,*tensors[1:])
        self.assertTrue(torch.allclose(b,torch.zeros_like(b),atol=1e-6))

def import_trainer_with_stubs():
    """Stub upstream signatures; real repo dependencies are not mounted locally."""
    modules={}
    dataset=types.ModuleType('dataset')
    dataset.load=lambda *args:None
    dataset.masks=lambda *args:None
    dataset.select_rows=lambda data,mask:[g for g,ok in zip(data.groups,mask) if ok]
    dataset.PRIMARY='static_repair_td_gain_sec';dataset.ORACLE='td_gain_sec'
    dataset.sha256=lambda p:'fakehash'
    dataset.grouped_indices=lambda groups:{g['state_id']:[] for g in groups}
    modules['dataset']=dataset
    tb=types.ModuleType('train_b2');tb.read_extra=lambda *args:None;tb.get_features=lambda *args:None
    modules['train_b2']=tb
    ff=types.ModuleType('features');ff.FEATURE_MODES={'F4':tuple(range(44))};modules['features']=ff
    t=types.ModuleType('train')
    def choose(y,oracle,scores,gate,groups,threshold):
        states={}
        for i,g in enumerate(groups):states.setdefault(g['state_id'],[]).append(i)
        result=[]
        for ids in states.values():
            k=max(ids,key=lambda i:scores[i])
            result.append({'reward':float(y[k]) if threshold is not None and gate[k]>=threshold else 0.})
        return result
    t.choose=choose
    t.summarize=lambda records:{'mean_signed_gain_sec':float(np.mean([r['reward'] for r in records]))}
    t.test_metrics=lambda *args:{}
    t.csv_write=lambda *args:None
    modules['train']=t
    b=types.ModuleType('train_b3');b.read_anchor=lambda *args:None;modules['train_b3']=b
    saved={k:sys.modules.get(k) for k in modules}
    try:
        sys.modules.update(modules)
        if 'train_c1' in sys.modules:del sys.modules['train_c1']
        trainer=importlib.import_module('train_c1')
        return trainer,saved
    finally:
        for k,v in saved.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v

class TestDelegate(unittest.TestCase):
    def test_variants_predict_and_state_loss(self):
        x=fake_x(8);g=graph_matrix(8)
        for variant in ('D0','D1','D2T','D2'):
            model=C1Model(FakeAnchor(),variant,seed=5).prepare(x)
            data=np.concatenate([x,g],axis=1) if variant in ('D2','D2T') else x
            score,gate,_=model.predict(data)
            self.assertEqual(score.shape,(8,))
            self.assertTrue(np.isfinite(score).all())
            z=torch.from_numpy(score.astype(np.float32))
            target=torch.linspace(-40.,40.,8)
            loss=state_loss(z,target)
            self.assertTrue(torch.isfinite(loss))

    def test_state_group_splitting(self):
        trainer,_=import_trainer_with_stubs()
        by_state=trainer.by_state
        groups=[{'state_id':f's{i//8}','task_id':'t','date':'day'} for i in range(48)]
        mask=np.array([True]*32+[False]*16)
        self.assertEqual(len(by_state(groups,mask)),4)
        mask[3]=False
        with self.assertRaises(ValueError):by_state(groups,mask)

    def test_supervised_end_to_end_synthetic(self):
        trainer,_=import_trainer_with_stubs()
        train_one,score_groups=trainer.train_one,trainer.score_groups
        x=fake_x(48);g=graph_matrix(48)
        y=np.asarray([v*2. for v in x[:,14]],np.float32)
        groups=[{'state_id':f's{i//8}','task_id':f't{i//8}','date':f'd{i//8}'} for i in range(48)]
        tr=np.array([i<32 for i in range(48)],bool)
        val=np.array([32<=i<40 for i in range(48)],bool)
        model=C1Model(FakeAnchor(),'D2',seed=5).prepare(x[tr])
        hist,pre,epoch=train_one(model,x,g,y,groups,tr,val,seed=5,epochs=2,pretrain_epochs=1)
        self.assertEqual(len(hist),2)
        self.assertEqual(len(pre),1)
        self.assertIn(epoch,[1,2])
        score=score_groups(model,x,g,groups,val)
        self.assertEqual(len(score),8)
        self.assertTrue(np.isfinite(score).all())

if __name__=='__main__':unittest.main()
