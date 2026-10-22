"""Synthetic full training pipeline; upstream dataset and feature schema stubbed.

Only real Chicago dataset/run can validate the frozen ALNS hooks.
"""
from __future__ import annotations
import hashlib,importlib,json,sys,tempfile,types,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class TrainPipeline(unittest.TestCase):
    def test_synthetic_end_to_end_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'labels';data.mkdir();source=root/'source';source.mkdir()
            n=90;rng=np.random.default_rng(17)
            X=rng.normal(size=(n,8,44)).astype('float32')
            B=rng.normal(size=(n,44)).astype('float32')
            same=np.zeros((n,8),bool);same[:,0]=True;X[:,0]=B
            Y=np.zeros((n,8),np.float32)
            Y[:,1:]=X[:,1:,0]*11.-B[:,None,0]*3.
            BY=np.zeros(n,np.float32);valid=np.ones((n,8),bool)
            np.savez_compressed(data/'paired.npz',X=X,B=B,Y=Y,BY=BY,valid=valid,same=same)
            groups=[{'decision_id':str(i),'task_id':f'task_{i//9:02d}','date':f'date_{i//18:02d}'} for i in range(n)]
            (data/'groups.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in groups))
            def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
            (data/'manifest.json').write_text(json.dumps({'stage':'step6c1r2-real-C3-labelled-v1',
                'pair_sha256':sha(data/'paired.npz'),'groups_sha256':sha(data/'groups.jsonl'),
                'road_sha256':'road','edge_sha256':'edge'}))
            (source/'run_config.json').write_text(json.dumps({'source_cost_hash':'road','source_edge_hash':'edge'}))
            (source/'analysis').mkdir();(source/'analysis/groups.csv').write_text('test\n')
            (source/'analysis/features_X.npz').write_bytes(b'stub')
            saved={x:sys.modules.get(x) for x in ('dataset','features','train_c3')}
            try:
                fake_dataset=types.ModuleType('dataset');fake_dataset.sha256=sha
                fake_features=types.ModuleType('features');fake_features.FEATURE_MODES={'F2':[str(i) for i in range(26)],
                    'F4':[str(i) for i in range(44)]}
                sys.modules['dataset']=fake_dataset;sys.modules['features']=fake_features
                sys.modules.pop('train_c3',None)
                trainer=importlib.import_module('train_c3')
                trainer.train(types.SimpleNamespace(output_dir=root/'trained',dataset_dir=data,
                    seed=43,protocol='task',modes='F2',model_source=str(source)))
                self.assertEqual(json.loads((root/'trained/summary.json').read_text())['status'],'COMPLETE')
                self.assertTrue((root/'trained/R2_F2.joblib').exists())
                report=json.loads((root/'trained/report.json').read_text())
                self.assertIn('internal_test',report['F2']['scores'])
            finally:
                for name,module in saved.items():
                    if module is None:sys.modules.pop(name,None)
                    else:sys.modules[name]=module

if __name__=='__main__':unittest.main()
