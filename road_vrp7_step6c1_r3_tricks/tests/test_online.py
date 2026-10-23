import unittest,tempfile,sys,json,types,hashlib,importlib.util
from pathlib import Path
from unittest.mock import patch
import joblib
ROOT=Path(__file__).absolute().parents[2]
for name in ('road_vrp7_step6c1_r3_tricks','road_vrp7_step6c1_r2'):
    p=str(ROOT/name)
    if p not in sys.path:sys.path.insert(0,p)
from r3_models import R3Advantage

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.original_online=sys.modules.get('online_r2')
        self.original_dataset=sys.modules.get('dataset')
        self.r2=types.ModuleType('online_r2')
        self.r2.ORIGINAL_LOADER=lambda *args:self.artifact
        self.r2.checked_load=lambda *args: {'stage':'step6c1r2-c3-advantage-train-v1','model':'base'}
        self.r2.main=lambda:None
        self.ds=types.ModuleType('dataset')
        self.ds.sha256=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        sys.modules['online_r2']=self.r2;sys.modules['dataset']=self.ds
        name='r3_online_test_module';sys.modules.pop(name,None)
        spec=importlib.util.spec_from_file_location(name,ROOT/'road_vrp7_step6c1_r3_tricks/online_r3.py')
        self.module=importlib.util.module_from_spec(spec);sys.modules[name]=self.module
        spec.loader.exec_module(self.module)
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
        (self.path/'paired.npz').write_bytes(b'dataset')
        self.artifact={'stage':'step6c1r3t-tricks-training-v1','model':R3Advantage(mode='F2'),
            'protocol':'task','r2_dataset_sha256':self.ds.sha256(self.path/'paired.npz'),
            'r2_source':str(self.path),'r3_arm':'T2_CB_HUBER',
            'r3_training_code_sha256':{p:self.ds.sha256(ROOT/'road_vrp7_step6c1_r3_tricks'/p)
                 for p in ('r3_models.py','decision_metrics.py','train_r3.py')}}
    def tearDown(self):
        self.tmp.cleanup();sys.modules.pop('r3_online_test_module',None)
        if self.original_online is None:sys.modules.pop('online_r2',None)
        else:sys.modules['online_r2']=self.original_online
        if self.original_dataset is None:sys.modules.pop('dataset',None)
        else:sys.modules['dataset']=self.original_dataset
    def test_valid_artifact(self):
        self.assertIs(self.module.checked_load('x','F2','P2','task','source'),self.artifact)
    def test_code_tamper_rejected(self):
        self.artifact['r3_training_code_sha256']['r3_models.py']='0'*64
        with self.assertRaises(ValueError):self.module.checked_load('x','F2','P2','task','source')
    def test_wrong_mode_rejected(self):
        with self.assertRaises(ValueError):self.module.checked_load('x','F4','P2','task','source')
    def test_wrong_dataset_rejected(self):
        self.artifact['r2_dataset_sha256']='0'*64
        with self.assertRaises(ValueError):self.module.checked_load('x','F2','P2','task','source')
    def test_r2_passthrough(self):
        self.artifact['stage']='step6c1r2-c3-advantage-train-v1'
        self.assertEqual(self.module.checked_load('x','F2','P2','task','source')['model'],'base')
    def test_output_dir_forwarding(self):
        self.artifact['stage']='step6c1r3t-tricks-training-v1'
        f=self.path/'model.joblib';joblib.dump(self.artifact,f)
        out=self.path/'pilot'
        def fake_main():
            self.assertIn('--output-dir',sys.argv)
            self.assertEqual(sys.argv[sys.argv.index('--output-dir')+1],str(out))
            self.assertIn('--model-path',sys.argv)
            out.mkdir()
        self.r2.main=fake_main
        with patch.object(sys,'argv',['online_r3.py','--output-dir',str(out),
                         '--model-path',str(f),'--control','learned']):
            self.module.main()
        p=json.loads((out/'r3_provenance.json').read_text())
        self.assertEqual(p['arm'],'T2_CB_HUBER')

if __name__=='__main__':unittest.main()
