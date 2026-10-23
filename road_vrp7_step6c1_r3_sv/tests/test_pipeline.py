from __future__ import annotations
import importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sv_core as core

class Model:
    def __init__(self,select_index=0):self.index=select_index
    def select(self,actions,baseline,same=None,valid=None):
        return self.index,10.,'candidate'

class DummyM0:
    def predict(self,x):return np.arange(len(x),dtype=float),np.ones(len(x)),None

class Pipeline(unittest.TestCase):
    def test_same_state_roundtrip_synthetic(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);d=p/'data';d.mkdir();out=p/'output'
            # Eight independent tasks make train/validation/internal_test folds.
            groups=[{'decision_id':f'id{i}','state_id':f'S{i}',
                     'task_id':f'task_{i:04d}','scenario_id':'2025-07-20_0800',
                     'seed':42,'section_call':8,'date':'2025-07-20'} for i in range(8)]
            for z in groups:z['task_id']='task_'+z['task_id'].split('_')[1]
            (d/'groups.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in groups))
            x=np.zeros((8,8,44),np.float32)
            b=np.zeros((8,44),np.float32)
            y=np.zeros((8,8),np.float32);y[:,0]=10.;y[:,1]=-30.;y[:,2]=20.
            by=np.zeros(8,np.float32);valid=np.ones((8,8),bool);same=np.zeros((8,8),bool)
            np.savez_compressed(d/'paired.npz',X=x,B=b,Y=y,BY=by,valid=valid,same=same)
            (d/'manifest.json').write_text(json.dumps({'stage':'step6c1r2-real-C3-labelled-v1',
                'pair_sha256':core.sha256(d/'paired.npz'),'groups_sha256':core.sha256(d/'groups.jsonl')}))
            files=[p/f'file{i}.joblib' for i in range(3)]
            for f in files:f.write_bytes(b'mock only')
            artifacts={str(files[0]):{'stage':'step6c1r2-c3-advantage-train-v1','mode':'F2',
                  'model':Model(0),'r2_dataset_sha256':core.sha256(d/'paired.npz')},
                 str(files[1]):{'stage':'step6c1r3t-tricks-training-v1',
                    'r3_arm':'T2_CB_HUBER','mode':'F2','model':Model(2),
                    'r2_dataset_sha256':core.sha256(d/'paired.npz')},
                 str(files[2]):{'mode':'F2','model':DummyM0(),'threshold':.5}}
            dummy=types.ModuleType('r2_policy')
            def group_split(groups,protocol):
                keys=sorted(set(g['task_id'] for g in groups))
                roles={k:('train' if i<4 else 'validation' if i<6 else 'internal_test') for i,k in enumerate(keys)}
                return None,roles
            dummy.group_split=group_split
            with patch.dict(sys.modules,{'r2_policy':dummy}):
                spec=importlib.util.spec_from_file_location('same_state_standalone_test',ROOT/'same_state.py')
                mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
                # Regression: legacy M0 P2_F2_model.joblib unpickles models.Predictor.
                # The legacy model's module folder must be importable even when
                # invoking this script directly from an arbitrary cwd.
                self.assertIn(str(ROOT.parent/'road_vrp7_step6b1_v2'),sys.path)
                Args=types.SimpleNamespace(dataset_dir=str(d),r2_model=str(files[0]),
                    huber_model=str(files[1]),m0_model=str(files[2]),output_dir=str(out),
                    max_pilot_states=2,pilot_seed=42,pilot_role='internal_test')
                with patch.object(mod.joblib,'load',side_effect=lambda f:artifacts[str(f)]):mod.run(Args)
            rep=json.loads((out/'report.json').read_text())
            self.assertEqual(rep['n_states'],8)
            self.assertEqual(rep['n_rows'],8*4)
            self.assertAlmostEqual(rep['by_role']['internal_test']['T2_HUBER']['avg_gain_sec'],20.)
            self.assertAlmostEqual(rep['by_role']['internal_test']['R2_F2']['avg_gain_sec'],10.)
            self.assertAlmostEqual(rep['by_role']['internal_test']['M0_PROXY']['avg_gain_sec'],0.)
            ids=json.loads((out/'pilot_states.json').read_text())['selected_decision_ids']
            self.assertEqual(len(ids),2)
            with self.assertRaises(FileExistsError):
                with patch.object(mod.joblib,'load',side_effect=lambda f:artifacts[str(f)]):mod.run(Args)

if __name__=='__main__':unittest.main()
