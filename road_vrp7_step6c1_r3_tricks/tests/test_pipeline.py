import sys,os,types,json,hashlib,tempfile,unittest
from pathlib import Path
from argparse import Namespace
import numpy as np,joblib
ROOT=Path(__file__).absolute().parents[2]
for name in ('road_vrp7_step6c1_r3_tricks','road_vrp7_step6c1_r2'):
    sys.path.insert(0,str(ROOT/name))
from r2_policy import R2Advantage
from r3_models import R3Advantage
from train_r3 import train,sha


class Integration(unittest.TestCase):
    def test_full_training_pipeline_synthetic(self):
        # Synthetic test: never a claim about real Chicago data.
        rng=np.random.default_rng(73)
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);ds=r/'label';ds.mkdir();src=r/'source';(src/'analysis').mkdir(parents=True)
            cfg={'source_cost_hash':'unit-road','source_edge_hash':'unit-edge'}
            (src/'run_config.json').write_text(json.dumps(cfg));(src/'analysis/groups.csv').write_text('id\n1\n')
            np.savez_compressed(src/'analysis/features_X.npz',X=np.zeros((1,26)))
            n=64;X=rng.normal(size=(n,8,44)).astype(np.float32);B=rng.normal(size=(n,44)).astype(np.float32)
            Y=(X[:,:,8]*50-X[:,:,9]*15+rng.normal(size=(n,8))*2).astype(np.float32)
            BY=rng.normal(size=n).astype(np.float32);valid=np.ones((n,8),bool);same=np.zeros((n,8),bool);same[:,7]=True
            np.savez_compressed(ds/'paired.npz',X=X,B=B,Y=Y,BY=BY,valid=valid,same=same)
            groups=[{'task_id':f'task_{i//8:04d}','date':'2025-07-02','scenario_id':f's{i//8}',
                      'seed':42,'state_id':str(i)} for i in range(n)]
            (ds/'groups.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in groups))
            (ds/'manifest.json').write_text(json.dumps({'stage':'step6c1r2-real-C3-labelled-v1',
                'pair_sha256':sha(ds/'paired.npz'),'groups_sha256':sha(ds/'groups.jsonl'),
                'road_sha256':'unit-road','edge_sha256':'unit-edge'}))
            orig=R2Advantage(mode='F2',seed=42).fit(X,B,Y,BY,valid,same,np.arange(n)<32)
            baseline=r/'r2_baseline.joblib';joblib.dump({'stage':'step6c1r2-c3-advantage-train-v1',
                'mode':'F2','model':orig,'r2_dataset_sha256':sha(ds/'paired.npz'),'protocol':'task'},baseline)
            # Lightweight stand-in for the frozen feature schema; actual server uses B2 features.py.
            prev=sys.modules.get('features');f=types.ModuleType('features');f.FEATURE_MODES={'F2':tuple(f'f{i}' for i in range(26))};sys.modules['features']=f
            try:
                a=Namespace(output_dir=str(r/'output'),dataset_dir=str(ds),model_source=str(src),
                    baseline_artifact=str(baseline),protocol='task',seed=42,epochs=3)
                train(a)
                self.assertEqual(json.loads((r/'output/summary.json').read_text())['status'],'COMPLETE')
                report=json.loads((r/'output/report.json').read_text())
                self.assertIn('T0_R2_F2',report)
                for arm in ('T1_CB_RMSE','T2_CB_HUBER','T3_PAIRWISE','T4_WEIGHTED_PAIR',
                    'T5_HUBER_MARGIN','T6_WEIGHTED_MARGIN','T7_BLEND_MARGIN'):
                    path=r/'output'/f'{arm}.joblib'
                    self.assertTrue(path.is_file());a=joblib.load(path)
                    self.assertIsInstance(a['model'],R3Advantage)
                    self.assertEqual(a['r3_arm'],arm)
                    self.assertTrue(np.isfinite(a['model'].predict_delta(X[60,:,:26],B[60,:26])).all())
            finally:
                if prev is None:sys.modules.pop('features',None)
                else:sys.modules['features']=prev

if __name__=='__main__':unittest.main()
