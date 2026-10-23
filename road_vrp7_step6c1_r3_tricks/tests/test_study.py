import sys,csv,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r3_tricks'))
import study_r3

class StudyTests(unittest.TestCase):
    def config(self,engine='R3'):
        return {'arm':'T2_CB_HUBER' if engine=='R3' else 'C3', 'engine':engine,
          'model_path':'x.joblib' if engine=='R3' else None,'model_mode':'F2',
          'source_path':'source_dir','seeds':[60,61], 'split':'validation',
          'cache_size':0,'source_sha256':'src','frozen_solver_sha256':'solver',
          'frozen_selector_sha256':'section','model_sha256':'model',
          'wall_sec':30,'max_iterations':20000,'top_k':12}
    def test_no_change_solver_command(self):
        conf=self.config('R3');cmd=study_r3.command(conf,'task_0008','2025-07-25_0800',Path('tmp'))
        self.assertEqual(cmd[1],'road_vrp7_step6c1_r3_tricks/online_r3.py')
        self.assertEqual(cmd[cmd.index('--wall-sec')+1],'30')
        self.assertEqual(cmd[cmd.index('--cache-size')+1],'0')
        self.assertEqual(cmd[cmd.index('--scenario-ids')+1],'2025-07-25_0800')
    def test_C3_uses_original_adapter(self):
        conf=self.config('C3');cmd=study_r3.command(conf,'task_0008','2025-07-25_0800',Path('tmp'))
        self.assertEqual(cmd[1],'road_vrp7_step6b2_r/benchmark_30s_r.py')
        self.assertEqual(cmd[cmd.index('--control')+1],'C3')
    def test_shard_complete_and_bad_model_hash(self):
        c=self.config('R3')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'summary.json').write_text(json.dumps({'status':'COMPLETE','records':2}))
            (p/'manifest.json').write_text(json.dumps({'source_sha256':'src','solver_sha256':'solver',
                'section_sha256':'section','model_sha256':'model','wall_sec':30,
                'max_iterations':20000,'top_k':12}))
            with (p/'metrics.csv').open('w') as f:
                f.write('task_id,scenario_id,seed,ex_post_road_travel_sec\n')
                f.write('task_0008,2025-07-25_0800,60,4000\n')
                f.write('task_0008,2025-07-25_0800,61,3900\n')
            self.assertTrue(study_r3.check_shard(p,'task_0008','2025-07-25_0800',[60,61],c))
            c['model_sha256']='tampered'
            with self.assertRaises(ValueError):study_r3.check_shard(p,'task_0008','2025-07-25_0800',[60,61],c)
    def test_resume_no_partial(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(study_r3.check_shard(Path(d),'t','s',[42],self.config()))

if __name__=='__main__':unittest.main()
