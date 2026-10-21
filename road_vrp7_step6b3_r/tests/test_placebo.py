"""Self-contained upstream-stubbed smoke test for placebo experiment wiring.

This is NOT a substitute for real Chicago data or a torch model artifact.
"""
import sys,tempfile,unittest,json,csv,types,importlib.util
from pathlib import Path
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1]

FEATURES=(tuple('struct_%02d'%i for i in range(26))+
          tuple('traffic_%02d'%i for i in range(10))+
          tuple('mask_%02d'%i for i in range(8)))


class FakeModel:
    alpha=.1
    def predict(self,arr):
        scores=arr[:,0]+self.alpha*arr[:,26]*arr[:,36]
        return scores,scores,None


class FakeData:
    def __init__(self):
        rows=[];X=[];Y=[]
        for task in ('task_0008','task_0009'):
            for date in ('2025-07-24','2025-07-25'):
                for section in range(2):
                    i=len(X);sid=f'{task}-{date}'
                    rows.append({'sample_id':f'{sid}-{section}','state_id':sid,
                                 'task_id':task,'date':date})
                    x=np.zeros(44);x[0]=float(section);x[26]=5+section+(date[-2:]=='25')*2
                    x[36:44]=.7
                    X.append(x);Y.append([2+section,3+section])
        self.X=np.asarray(X);self.Y=np.asarray(Y);self.groups=rows
    def label(self,kind):
        return self.Y[:,0] if kind=='static_repair_td_gain_sec' else self.Y[:,1]


def stubs(d):
    mdl=types.ModuleType('model')
    mdl.FEATURES=FEATURES;mdl.STRUCTURE=FEATURES[:26]
    mdl.TRAFFIC=FEATURES[26:36];mdl.RELIABILITY=FEATURES[36:]
    def from_full(X,names=FEATURES):return X[:,:26],X[:,26:36],X[:,36:],np.minimum(X[:,36],X[:,37])
    mdl.from_full=from_full
    ft=types.ModuleType('features');ft.FEATURE_MODES={'F4':FEATURES}
    tb=types.ModuleType('train_b2');tb.read_extra=lambda *args:(None,{'source_sha256':{'X':'ok'}},'fhash')
    tb.get_features=lambda *args:(d.X,FEATURES)
    dataset=types.ModuleType('dataset');dataset.PRIMARY='static_repair_td_gain_sec';dataset.ORACLE='td_gain_sec'
    dataset.load=lambda *args:d
    dataset.masks=lambda *args:{'internal_test':np.ones(len(d.X),bool)}
    dataset.select_rows=lambda ds,mask:[g for j,g in enumerate(ds.groups) if mask[j]]
    dataset.sha256=lambda *args:'dummy'
    train=types.ModuleType('train')
    def choose(y,ora,scores,gates,groups,threshold):
        by={}
        for j,g in enumerate(groups):by.setdefault(g['state_id'],[]).append(j)
        return [{'reward':float(y[max(ix,key=lambda a:scores[a])])} for ix in by.values()]
    train.choose=choose
    train.summarize=lambda rows:{'mean_signed_gain_sec':float(np.mean([r['reward'] for r in rows])),
        'executed_states':len(rows),'nonpositive_executions':0,'states':len(rows)}
    fr=types.ModuleType('frozen_permutation')
    def shuffle(X,names,groups,mask,seed):
        out=X.copy();by={}
        for j,g in enumerate(groups):by.setdefault(g['state_id'],[]).append(j)
        for state,ids in by.items():
            task,tail=state.split('-2025-',1)
            date='2025-'+tail
            donor=by[f'{task}-'+('2025-07-25' if date=='2025-07-24' else '2025-07-24')]
            out[np.ix_(ids,range(26,44))]=X[np.ix_(donor,range(26,44))]
        return out,{'donor_fraction':1.,'donor_mapped_states':len(by),'states':len(by)}
    fr.shuffle_same_task_date=shuffle
    return {'model':mdl,'features':ft,'train_b2':tb,'dataset':dataset,
            'train':train,'frozen_permutation':fr}


class TestPlacebo(unittest.TestCase):
    def test_mock_model_frozen_perm_and_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);study=root/'b3_anchor_film_seed42_v1';(study/'task').mkdir(parents=True)
            (study/'summary.json').write_text(json.dumps({'status':'COMPLETE'}))
            (study/'manifest.json').write_text(json.dumps({
                'seed':42,'stage':'step6b3-anchored-film-v1',
                'feature_npz_sha256':'fhash','source_sha256':{'X':'ok'}}))
            d=FakeData();model=FakeModel()
            artifact={'b3_method':'M3','mode':'F4','method':'P2','protocol':'task',
                'feature_names':list(FEATURES),'boundary_X_sha256':'fhash',
                'alpha':.1,'threshold':-20,'model':model}
            joblib.dump(artifact,study/'task/M3_model.joblib')
            with (study/'task/M3_test_candidates.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['sample_id','model_score_sec'])
                w.writeheader()
                predictions=model.predict(d.X)[0]
                w.writerows({'sample_id':r['sample_id'],'model_score_sec':float(predictions[i])}
                            for i,r in enumerate(d.groups))
            mods=stubs(d);old={k:sys.modules.get(k) for k in mods}
            try:
                sys.modules.update(mods)
                spec=importlib.util.spec_from_file_location('b3r_placebo_mock',ROOT/'placebo_b3r.py')
                module=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                result=module.run(root,root,root,root/'out',
                        (42,),('M3',),(100,),('traffic_and_mask',))
                self.assertEqual(result['results'][0]['shuffle'],'identity')
                self.assertEqual(result['results'][1]['shuffle'],'traffic_and_mask')
                self.assertTrue((root/'out'/'placebo.json').is_file())
            finally:
                for name,val in old.items():
                    if val is None:sys.modules.pop(name,None)
                    else:sys.modules[name]=val
if __name__=='__main__':unittest.main()
