import os,sys,unittest,tempfile,csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r3_vq6'))
# The test can exercise online predictions without assuming whole Github repo is on the
# local artifact-building computer; the server will load the frozen real r2_policy.py.
try:
    import r2_policy
except ImportError:
    import types
    from dataclasses import dataclass
    m=types.ModuleType('r2_policy')
    def paired_matrix(a,b):
        a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
        return np.concatenate([a,np.broadcast_to(b,a.shape),a-b],axis=1)
    @dataclass
    class R2Advantage:
        mode:str='F2';seed:int=42;threshold:float=20.;min_effective_actions:int=1;model:object=None
        def select(self,a,b,same=None,valid=None):
            s=self.predict_delta(a,b);v=np.ones(len(s),bool)
            if same is not None:v &=~np.asarray(same,bool)
            if valid is not None:v &=np.asarray(valid,bool)
            if not v.any():return None,float('-inf'),'no_different_candidate'
            j=int(np.argmax(np.where(v,s,-np.inf)))
            return (j,float(s[j]),'override_C3') if s[j]>self.threshold else (None,float(s[j]),'defer')
    m.R2Advantage=R2Advantage;m.paired_matrix=paired_matrix
    sys.modules['r2_policy']=m
from vq6_policy import VQ6Policy
from compare_vq6 import paired_report
from sklearn.ensemble import ExtraTreesRegressor

class PolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(11)
        X=rng.normal(size=(36,78)).astype(np.float32)
        cls.reg=ExtraTreesRegressor(n_estimators=10,random_state=42).fit(X,X[:,0]*4+2)
        cls.regres=ExtraTreesRegressor(n_estimators=10,random_state=24).fit(X,np.full(36,10.))
        cls.x=rng.normal(size=(8,26)).astype(np.float32)
        cls.b=rng.normal(size=(26,)).astype(np.float32)
    def test_f2_78dim_and_alpha(self):
        base=VQ6Policy(kind='E0_MATCHED',mode='F2',model=self.reg,threshold=0.)
        res=VQ6Policy(kind='RESID_OOF50',mode='F2',model=self.reg,residual_model=self.regres,alpha=.5,threshold=0.)
        p=base.predict_delta(self.x,self.b);q=res.predict_delta(self.x,self.b)
        self.assertEqual(p.shape,(8,));np.testing.assert_allclose(q-p,5.,atol=1e-6)
    def test_invalid_f2_rejected(self):
        base=VQ6Policy(kind='E0_MATCHED',mode='F2',model=self.reg)
        with self.assertRaises(ValueError):base.predict_delta(self.x[:,:25],self.b)
    def test_c3_mask(self):
        base=VQ6Policy(kind='E0_MATCHED',mode='F2',model=self.reg,threshold=-999.)
        j,_,_=base.select(self.x,self.b,same=np.ones(8,bool))
        self.assertIsNone(j)
    def test_full_scoring_defer(self):
        p=VQ6Policy(kind='DEFER_RESID',mode='F2',model=self.reg,residual_model=self.regres,alpha=.5,threshold=0.)
        j,score,why=p.select(self.x,self.b)
        self.assertIsNone(j);self.assertTrue(np.isfinite(score));self.assertIn('defer',why)
    def test_invalid_residual_schema(self):
        x=VQ6Policy(kind='RESID_OOF50',model=self.reg,alpha=.25)
        with self.assertRaises(ValueError):x.predict_delta(self.x,self.b)

class PairsTest(unittest.TestCase):
    def write_arm(self,root,name,costs,effective=0):
        p=root/name;p.mkdir()
        mf={'source_sha256':'s','road_sha256':'r','edge_sha256':'e','solver_sha256':'sol',
            'section_sha256':'sect','wall_sec':30,'max_iterations':20000,
            'anneal_iterations':5000,'top_k':12,'seeds':[60,61],
            'task_ids':['task_0008'],'scenario_ids':['scenario'],'split':'validation',
            'arm':'C3' if name=='C3' else 'learned',
            'model_sha256':None if name=='C3' else 'hash'+name,
            'action_count':8,'sampler_seed':20261019,'r_score_every':8,
            'r_max_scoring_sec':2,'r_fallback':'c3'}
        (p/'manifest.json').write_text(json.dumps(mf))
        (p/'summary.json').write_text(json.dumps({'status':'COMPLETE','records':2,'expected_records':2}))
        rows=[]
        for seed,c in zip([60,61],costs):
            rows.append({'task_id':'task_0008','scenario_id':'scenario','seed':seed,'ex_post_road_travel_sec':c,
                         'iterations':20000,'search_wall_sec':30.,'r_scoring_seconds':.3,
                         'r_model_actions':effective,'r_model_eval_calls':3})
        with (p/'metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        if name!='C3':
            (p/'vq6_provenance.json').write_text(json.dumps({'stage':'R3-VQ6-online-v1',
                'kind':name,'model_sha256':'hash'+name}))
            (p/'r2_diagnostics.json').write_text(json.dumps({'runs':[
                {'r2_effective_overrides':effective} for _ in range(2)]}))
        return p
    def test_paired_and_negative_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths={}
            for name,costs,eff in [('C3',[100,100],0),('DEFER_RESID',[101,102],0),
                                   ('E0_MATCHED',[95,98],2),('RESID_OOF50',[94,97],2)]:
                paths[name]=self.write_arm(root,name,costs,eff)
            out=paired_report(paths)
            self.assertEqual(out['n_pairs'],2)
            self.assertAlmostEqual(out['summaries']['RESID_OOF50_vs_C3']['mean_road_improvement_sec'],4.5)
            self.assertAlmostEqual(out['summaries']['RESID_OOF50_vs_E0_MATCHED']['mean_road_improvement_sec'],1.)
    def test_mismatched_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths={}
            for name in ('C3','DEFER_RESID','E0_MATCHED','RESID_OOF50'):
                paths[name]=self.write_arm(root,name,[100,100])
            fn=paths['RESID_OOF50']/'manifest.json';o=json.loads(fn.read_text());o['wall_sec']=20;fn.write_text(json.dumps(o))
            with self.assertRaises(ValueError):paired_report(paths)
    def test_defer_override_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths={}
            for name in ('C3','DEFER_RESID','E0_MATCHED','RESID_OOF50'):
                paths[name]=self.write_arm(root,name,[100,100],int(name=='DEFER_RESID'))
            with self.assertRaises(ValueError):paired_report(paths)

if __name__=='__main__':unittest.main()
