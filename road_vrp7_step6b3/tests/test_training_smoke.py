"""Stubbed upstream-integration test: uses same upstream function signatures.

This is NOT a claim that the real Chicago tensors were available locally.
"""
import csv,importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from model import FEATURES,STRUCTURE
from test_model import FakeAnchor,make_rows

class Data:
    def __init__(self,X,Y,groups,src):
        self.X=X;self.Y=Y;self.groups=groups;self.source=Path(src)
        self.names=list(FEATURES);self.labels=['static_repair_td_gain_sec','td_gain_sec']
        self.n=len(X)
    def label(self,k):return self.Y[:,self.labels.index(k)]


def prepare_stubs(d,extra):
    dataset=types.ModuleType('dataset')
    dataset.PRIMARY='static_repair_td_gain_sec';dataset.ORACLE='td_gain_sec'
    dataset.load=lambda p:d
    dataset.masks=lambda ds,p:{role:np.array([g[p+'_role']==role for g in ds.groups]) for role in ('train','val','internal_test')}
    dataset.select_rows=lambda ds,mask:[g for g,x in zip(ds.groups,mask) if x]
    dataset.sha256=lambda p:'testhash-'+Path(p).name
    tb=types.ModuleType('train_b2')
    tb.read_extra=lambda ds,dir:(extra,{'stage':'step6b2-features-v1',
      'source_sha256':{'X':'testhash-X'},'feature_names':list(FEATURES)[12:]},'boundaryhash')
    tb.get_features=lambda ds,ex,mode:(ds.X,list(FEATURES))
    ft=types.ModuleType('features');ft.FEATURE_MODES={'F4':tuple(FEATURES)}
    base=types.ModuleType('train')
    def choose(y,ora,rank,gate,g,threshold):
        states={}
        for i,row in enumerate(g):states.setdefault(row['state_id'],[]).append(i)
        rec=[]
        for sid,ids in states.items():
            idx=max(ids,key=lambda v:rank[v]);on=threshold is not None and gate[idx]>=threshold
            rec.append({'state_id':sid,'task_id':g[idx]['task_id'],'date':g[idx]['date'],
              'selected_id':g[idx]['sample_id'],'execute':int(on),
              'gain':float(y[idx]) if on else 0.,'oracle_bound':max(0.,max(y[j] for j in ids))})
        return rec
    base.choose=choose
    base.summarize=lambda rows:{'states':len(rows),'mean_signed_gain_sec':float(np.mean([x['gain'] for x in rows])),
                                 'executed_states':sum(x['execute'] for x in rows)}
    base.test_metrics=lambda y,r,p,g,kind:{'MAE':float(np.mean(np.abs(y-r)))}
    def csv_write(path,rows):
        with Path(path).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    base.csv_write=csv_write
    return dict(dataset=dataset,train_b2=tb,features=ft,train=base)

class TestSmoke(unittest.TestCase):
    def test_all_methods_protocols(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);X=make_rows(n=24*8);rng=np.random.default_rng(1)
            rows=[];ys=[]
            for task in range(6):
                for date in range(4):
                    for section in range(8):
                        idx=len(rows)
                        score=float(12*X[idx,14]+4*X[idx,27]+rng.normal()*9)
                        ys.append([score,score+abs(rng.normal())])
                        rows.append({'task_id':f'task_{task:04d}','state_id':f'st_{task}_{date}',
                          'date':f'2025-07-{22+date}',
                          'sample_id':f'{task}_{date}_{section}',
                          'task_role':'train' if task<4 else 'val' if task==4 else 'internal_test',
                          'date_role':'train' if date<2 else 'val' if date==2 else 'internal_test'})
            d=Data(X,np.array(ys),rows,root)
            mods=prepare_stubs(d,np.zeros((len(X),18)))
            saved={k:sys.modules.get(k) for k in mods}
            try:
                sys.modules.update(mods)
                spec=importlib.util.spec_from_file_location('b3_train_mock',P/'train_b3.py')
                b3=importlib.util.module_from_spec(spec);spec.loader.exec_module(b3)
                anchor=root/'anchor'
                for p in ('task','date'):
                    fold=anchor/p;fold.mkdir(parents=True)
                    import joblib
                    joblib.dump({'method':'P2','mode':'F2','protocol':p,
                      'training_label':'static_repair_td_gain_sec',
                      'feature_names':list(STRUCTURE),
                      'source_sha256':{'X':'testhash-X'},
                      'boundary_X_sha256':'boundaryhash',
                      'threshold':-20.,'model':FakeAnchor()},fold/'P2_F2_model.joblib')
                output=root/'results'
                result=b3.run(root,root,anchor,output,('task','date'),
                              ('M0','M1','M2','M3','M4'),seed=42,epochs=3,trees=8)
                self.assertEqual(set(result),{'task','date'})
                for p in ('task','date'):
                    self.assertEqual(set(result[p]),set(('M0','M1','M2','M3','M4')))
                    for mode in result[p]:
                        a=result[p][mode]['alpha_selected_on_validation']
                        self.assertIn(a,b3.ALPHAS)
                        self.assertTrue((output/p/f'{mode}_model.joblib').exists())
                    self.assertEqual(result[p]['M0']['alpha_selected_on_validation'],0.)
                self.assertTrue((output/'summary.json').is_file())
                with self.assertRaises(FileExistsError):b3.run(root,root,anchor,output,('task',),('M0',))
            finally:
                for key,val in saved.items():
                    if val is None:sys.modules.pop(key,None)
                    else:sys.modules[key]=val

if __name__=='__main__':unittest.main()
