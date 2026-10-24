import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4f'))
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4c'))
from r4f_core import LONG,dump
from r4c_core import unit_name
import reuse_r4f

class ReuseTests(unittest.TestCase):
    def make(self,root):
        src,dst=Path(root)/'src',Path(root)/'dst';(src/'units').mkdir(parents=True);(dst/'units').mkdir(parents=True)
        a={'stage':LONG,'status':'COMPLETE','max_cases':4,'selected_snapshots':[['t','s'+str(i),60,505] for i in range(8)],
           'horizons':[100,200,500], 'source_capture_sha256':'a','source_r4b_sha256':'b',
           'r4c_manifest_sha256':'c','r4c_report_sha256':'d','long_proof_report_sha256':'e',
           'long_proof_rows_sha256':'f','max_snapshots_per_run':2,'action_count':8,'max_alternatives':8,
           'sampler_seed':20261019,'code_sha256':{'r4e_core.py':'z'},
           'evaluation':'FIXED_ITERATION_K_NOT_30S_WALL_CLOCK',
           'data_role':'INSPECTED_JULY_VALIDATION_DEVELOPMENT_ONLY'}
        b={**a,'status':'RUNNING','max_cases':0,'selected_snapshots':a['selected_snapshots']+[['t','p'+str(i),61,505] for i in range(8)]}
        dump(src/'manifest.json',a);dump(dst/'manifest.json',b)
        spec={'task_id':'t','scenario_id':'s0','seed':60,'stop_count':505,'horizon':100,'arm':'C3'}
        dump(src/'units'/unit_name(spec),{'stage':LONG,'unit_spec':spec,'plan':{'id':123},'result':{'arm':'C3'},
            'r4c_manifest_sha256':'c','long_proof_rows_sha256':'f'})
        dump(src/'report.json',{'n_branch_results':1})
        return src,dst

    def test_unit_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,t=self.make(tmp)
            reuse_r4f.run(type('Args',(),{'source_dir':s,'target_dir':t})())
            self.assertEqual(len(list((t/'units').glob('*.json'))),1)
            self.assertEqual(json.loads((t/'r4f_checkpoint_reuse.json').read_text())['reused_copied'],1)

    def test_unit_reuse_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,t=self.make(tmp)
            a=type('Args',(),{'source_dir':s,'target_dir':t})()
            reuse_r4f.run(a);reuse_r4f.run(a)
            self.assertEqual(json.loads((t/'r4f_checkpoint_reuse.json').read_text())['already_present'],1)

    def test_reject_pin_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,t=self.make(tmp)
            m=json.loads((t/'manifest.json').read_text());m['sampler_seed']=3;dump(t/'manifest.json',m)
            with self.assertRaisesRegex(ValueError,'Reuse pin'):
                reuse_r4f.run(type('Args',(),{'source_dir':s,'target_dir':t})())

    def test_reject_unit_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,t=self.make(tmp)
            a=type('Args',(),{'source_dir':s,'target_dir':t})()
            reuse_r4f.run(a)
            f=next((t/'units').glob('*.json'));f.write_text('changed')
            with self.assertRaisesRegex(ValueError,'checkpoint differs'):
                reuse_r4f.run(a)

if __name__=='__main__':unittest.main()
