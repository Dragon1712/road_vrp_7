"""Local synthetic end-to-end and adversarial guards, no Chicago/ALNS needed."""
import json,sys,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import r4j_core as core
import audit_r4j

SID=['task_0002','2025-07-02_0800',62,1018]

def fixture(tmp):
    root=Path(tmp);c=root/'capture';p=root/'plain';a=root/'audit';x=root/'prefix';y=root/'cf'
    for d in (c,p,a,x,y):d.mkdir()
    cm={'stage':core.CAP_STAGE,'arm':'capture','split':'train','task_ids':['task_0002'],
        'scenario_ids':['2025-07-02_0800'],'seeds':[62],'max_iterations':20000,'search_wall_sec':30,
        'section_only':True}
    pm={**cm,'arm':'plain'}
    snap={'task_id':SID[0],'scenario_id':SID[1],'seed':SID[2],'stop_count':SID[3],
          'boundary':'BEFORE_SELECT_AND_DESTROY','snapshot_is_actual_section_decision':True,
          'real_selected_destroy_index':3,
          'current_route_sha256':'curr','best_route_sha256':'best','rng_state_sha256':'rng',
          'roulette_destroy_weights':[1.,2.,3.,4.],
          'roulette_repair_weights':[1.,2.],
          'anneal_temperature_before_accept':200.1234567890123}
    base={'task_id':SID[0],'scenario_id':SID[1],'seed':SID[2],
          'iterations':20000,'best_route_sha256':'bb','road_cost_sec':3000.,'search_wall_sec':26.,'captured':0}
    for d,m,rows,snaps in ((c,cm,[dict(base,captured=1)],[snap]),(p,pm,[base],[])):
        core.write(d/'manifest.json',m)
        core.write(d/'report.json',{'stage':core.CAP_STAGE,'status':'COMPLETE',
                                    'records':1,'snapshots':len(snaps)})
        (d/'metrics.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in rows))
        (d/'snapshots.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in snaps))
        core.write(d/'provenance.json',{
            'manifest_sha256':core.digest(d/'manifest.json'),
            'metrics_sha256':core.digest(d/'metrics.jsonl'),
            'snapshots_sha256':core.digest(d/'snapshots.jsonl')})
    core.write(a/'report.json',{'stage':'R4G-TRAIN-NONINTERFERENCE-AUDIT-v1',
       'status':'COMPLETE','pairs':1,'strict_pass':1,'route_road_equal':1,'failure_reason_counts':{}})
    pref=[{'task_id':SID[0],'scenario_id':SID[1],'seed':SID[2],
           'snapshot_stop_count':SID[3],'horizon':k,'verified_steps':k,'exact':True} for k in (1,50,200,500)]
    (x/'paired_rows.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in pref))
    (x/'excluded_windows.json').write_text('[]\n')
    core.write(x/'report.json',{'stage':core.PREFIX_STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
       'exact_matches':4,'n_comparisons':4,'n_cases':1,
       'original_capture_snapshots_sha256':core.digest(c/'snapshots.jsonl')})
    core.write(x/'manifest.json',{'stage':core.PREFIX_STAGE,'data_role':'TRAIN_ONLY',
       'source_sha256':{'capture_snapshots':core.digest(c/'snapshots.jsonl')}})
    branches=[];labels=[]
    for k in core.HORIZONS:
        controls=[('C3',1000.,'base'),('SHAM_C3',1000.,'base'),('A005',999. if k==500 else 1000.,'alt')]
        for arm,val,rsha in controls:
            branches.append({'unit_spec':dict(zip(('task_id','scenario_id','seed','stop_count'),SID),horizon=k,arm=arm),
                   'best_road_sec':val,'best_route_sha256':rsha})
        labels.append({'unit_spec':dict(zip(('task_id','scenario_id','seed','stop_count'),SID),horizon=k,arm='A005'),
                       'road_best_gain_sec':1. if k==500 else 0.})
    (y/'branch_summary.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in branches))
    (y/'labels.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in labels))
    (y/'plans.jsonl').write_text(json.dumps({'snapshot_id':SID,'candidate_count':12,'n_alternatives':1})+'\n')
    core.write(y/'report.json',{'stage':core.CF_STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
         'n_labels':3,'n_branches':9,'n_scenario_snapshots':1,'n_sham_checks':3})
    core.write(y/'manifest.json',{'stage':core.CF_STAGE,'data_role':'TRAIN_ONLY','horizons':[50,200,500],
       'require_all_k':True,'selected_snapshot_ids':[SID],
       'source_sha256':{'capture_snapshots':core.digest(c/'snapshots.jsonl'),
             'prefix_report':core.digest(x/'report.json'),'prefix_paired_rows':core.digest(x/'paired_rows.jsonl')}})
    return c,p,a,x,y

class R4JTests(unittest.TestCase):
    def test_physical_exact_temperature_not_rounded(self):
        s={'task_id':'t','seed':62,'stop_count':1018,'current_route_sha256':'a',
           'best_route_sha256':'b','rng_state_sha256':'c','roulette_destroy_weights':[1.,1.,1.,1.],
           'roulette_repair_weights':[1.,1.], 'anneal_temperature_before_accept':233.0771455604309}
        z=dict(s,anneal_temperature_before_accept=233.0771455604079)
        self.assertNotEqual(core.physical(s),core.physical(z))
        self.assertEqual(core.physical(s),core.physical(dict(s,scenario_id='2025-07-09_1700')))
    def test_end_to_end_audit_and_output(self):
        with tempfile.TemporaryDirectory() as td:
            c,p,a,x,y=fixture(td)
            bundle=core.verify_capture(c,p,a)
            prefix=core.verify_prefix(x,bundle)
            cf=core.verify_cf(y,bundle,prefix)
            result=core.analyze(cf,bundle)
            self.assertEqual(result['per_horizon']['500']['early']['positive'],1)
            self.assertEqual(result['n_exact_physical_states'],1)
            self.assertEqual(result['source_strict_pairs'],1)
            self.assertIn('Train',core.markdown(result).replace('TRAIN','Train'))
            opts=type('X',(),{'capture_dir':str(c),'plain_dir':str(p),'audit_dir':str(a),
                'prefix_dir':str(x),'cf_dir':str(y),'old_cf_dir':None,
                'output_dir':str(Path(td)/'r4j-report'),'phase':'cf','forbid_seeds':'60,61'})()
            audit_r4j.run(opts)
            self.assertTrue((Path(td)/'r4j-report'/'provenance.json').exists())
    def test_reused_original_seed_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            c,p,a,x,y=fixture(td)
            z=core.read(c/'manifest.json');z['seeds']=[60];core.write(c/'manifest.json',z)
            with self.assertRaisesRegex(ValueError,'reused original seed|grid differs'):
                core.verify_capture(c,p,a)
    def test_manifest_tamper_fails_SHA(self):
        with tempfile.TemporaryDirectory() as td:
            c,p,a,x,y=fixture(td)
            z=core.read(c/'manifest.json');z['max_iterations']=21000;core.write(c/'manifest.json',z)
            with self.assertRaises(ValueError):core.verify_capture(c,p,a)
    def test_terminal_road_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            c,p,a,x,y=fixture(td)
            z=core.read(a/'report.json');z['route_road_equal']=0;core.write(a/'report.json',z)
            with self.assertRaisesRegex(ValueError,'Final C3 route/Road'):core.verify_capture(c,p,a)
    def test_prefix_comparison_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            c,p,a,x,y=fixture(td);bun=core.verify_capture(c,p,a)
            file=x/'paired_rows.jsonl';rows=core.readl(file);rows[0]['exact']=False
            file.write_text(''.join(json.dumps(z)+'\n' for z in rows))
            with self.assertRaisesRegex(ValueError,'per-event'):core.verify_prefix(x,bun)
    def test_prefix_snapshot_sha_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            c,p,a,x,y=fixture(td);bun=core.verify_capture(c,p,a)
            z=core.read(x/'report.json');z['original_capture_snapshots_sha256']='x';core.write(x/'report.json',z)
            with self.assertRaisesRegex(ValueError,'source SHA'):core.verify_prefix(x,bun)
    def test_cf_sham_changed_fails(self):
        with tempfile.TemporaryDirectory() as td:
            c,p,a,x,y=fixture(td);bun=core.verify_capture(c,p,a);prefix=core.verify_prefix(x,bun)
            file=y/'branch_summary.jsonl';data=core.readl(file);data[1]['best_road_sec']=1100
            file.write_text(''.join(json.dumps(z)+'\n' for z in data))
            with self.assertRaisesRegex(ValueError,'SHAM'):core.verify_cf(y,bun,prefix)
    def test_cf_labels_inconsistent_fails(self):
        with tempfile.TemporaryDirectory() as td:
            c,p,a,x,y=fixture(td);bun=core.verify_capture(c,p,a);prefix=core.verify_prefix(x,bun)
            file=y/'labels.jsonl';data=core.readl(file);data[2]['road_best_gain_sec']=900
            file.write_text(''.join(json.dumps(z)+'\n' for z in data))
            with self.assertRaisesRegex(ValueError,'Wrong gain'):core.verify_cf(y,bun,prefix)
    def test_cf_missing_k_fails(self):
        with tempfile.TemporaryDirectory() as td:
            c,p,a,x,y=fixture(td);bun=core.verify_capture(c,p,a);prefix=core.verify_prefix(x,bun)
            file=y/'labels.jsonl';data=core.readl(file)[:-1];file.write_text(''.join(json.dumps(z)+'\n' for z in data))
            rep=core.read(y/'report.json');rep['n_labels']=2;core.write(y/'report.json',rep)
            with self.assertRaisesRegex(ValueError,'horizons|missing'):core.verify_cf(y,bun,prefix)
    def test_off_role_cf_fails(self):
        with tempfile.TemporaryDirectory() as td:
            c,p,a,x,y=fixture(td);bun=core.verify_capture(c,p,a);prefix=core.verify_prefix(x,bun)
            rep=core.read(y/'report.json');rep['data_role']='validation';core.write(y/'report.json',rep)
            with self.assertRaisesRegex(ValueError,'not Train'):core.verify_cf(y,bun,prefix)
    def test_source_meta_end_summary(self):
        with tempfile.TemporaryDirectory() as td:
            c,p,a,x,y=fixture(td);bun=core.verify_capture(c,p,a);prefix=core.verify_prefix(x,bun)
            cf=core.verify_cf(y,bun,prefix)
            result=core.analyze(cf,bun)
            self.assertEqual(result['per_horizon']['50']['early']['zero'],1)
            self.assertEqual(result['per_task']['500']['task_0002']['positive'],1)

if __name__=='__main__':unittest.main()
