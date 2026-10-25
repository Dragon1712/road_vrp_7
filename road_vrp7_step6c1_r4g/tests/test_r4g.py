import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
for d in ('road_vrp7_step6c1_r4g','road_vrp7_step6c1_r4a','road_vrp7_step6c1_r4b'):
    sys.path.insert(0,str(ROOT/d))
from r4g_core import (validate_scenario_role,select_phase_snapshots,physical_id,paired_audit,load_capture,STAGE)
from r4a_core import sha256,write_json
from r4b_core import require

class TestR4G(unittest.TestCase):
    def setUp(self):
        self.task={'scenarios':[{'scenario_id':'train','split':'train','eligible':True},
                {'scenario_id':'validation','split':'validation','eligible':True},
                {'scenario_id':'test','split':'test','eligible':True},
                {'scenario_id':'ineligible','split':'train','eligible':False}]}
    def test_train_only(self):self.assertTrue(validate_scenario_role(self.task,['train']))
    def test_no_validation(self):
        with self.assertRaisesRegex(ValueError,'Forbidden'):validate_scenario_role(self.task,['validation'])
    def test_no_test(self):
        with self.assertRaisesRegex(ValueError,'Forbidden'):validate_scenario_role(self.task,['test'])
    def test_no_ineligible(self):
        with self.assertRaisesRegex(ValueError,'Forbidden'):validate_scenario_role(self.task,['ineligible'])
    def test_no_unknown(self):
        with self.assertRaisesRegex(ValueError,'Unknown'):validate_scenario_role(self.task,['missing'])
    def test_phase_stratification(self):
        x=[{'stop_count':i} for i in [10,20,30,40,50,60]]
        self.assertEqual([z['stop_count'] for z in select_phase_snapshots(x,3)],[10,30,60])
    def test_phase_duplicates_fail(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):select_phase_snapshots([{'stop_count':1}]*2)
    def test_physical_cross_scenario(self):
        s={'task_id':'task_0000','seed':60,'stop_count':100,
           'current_route_sha256':'a','best_route_sha256':'b','rng_state_sha256':'c',
           'roulette_destroy_weights':[1,2,3],'roulette_repair_weights':[4,5],
           'anneal_temperature_before_accept':23.}
        self.assertEqual(physical_id(dict(s,scenario_id='2025-07-02_0800')),
                         physical_id(dict(s,scenario_id='2025-07-09_1700')))
    def fake(self,arm):
        mf={'stage':STAGE,'split':'train','arm':arm,'task_ids':['task_0000'],
            'scenario_ids':['2025-07-02_0800'],'seeds':[60],
            'max_iterations':2000,'search_wall_sec':30,'anneal_iterations':5000,'top_k':12,
            'source_sha256':'s','road_sha256':'r','edge_sha256':'e',
            'solver_sha256':'v','section_sha256':'ss','section_only':arm=='capture'}
        r={'task_id':'task_0000','scenario_id':'2025-07-02_0800','seed':60,
           'iterations':2000,'best_route_sha256':'abc','road_cost_sec':100.,
           'search_wall_sec':3,'captured':1 if arm=='capture' else 0}
        s={'task_id':'task_0000','scenario_id':'2025-07-02_0800','seed':60,'stop_count':100,
           'current_route_sha256':'a','best_route_sha256':'b','rng_state_sha256':'c',
           'roulette_destroy_weights':[1]*4,'roulette_repair_weights':[1]*2,
           'anneal_temperature_before_accept':23.}
        return mf,[r],[s] if arm=='capture' else []
    def test_paired_strict(self):
        z=paired_audit(self.fake('plain'),self.fake('capture'))
        self.assertEqual((z['pairs'],z['strict_pass'],z['n_distinct_physical_states']),(1,1,1))
    def test_paired_noninterference_failure(self):
        p=self.fake('plain');c=self.fake('capture');c[1][0]['best_route_sha256']='changed'
        z=paired_audit(p,c);self.assertEqual(z['strict_pass'],0)
    def test_paired_bad_roles(self):
        p=self.fake('plain');c=self.fake('capture');c[0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'Pair source'):paired_audit(p,c)
    def test_roundtrip_provenance_reject(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);mf,rows,ss=self.fake('plain')
            write_json(p/'manifest.json',mf);write_json(p/'report.json',{'stage':STAGE,'status':'COMPLETE','records':1,'snapshots':0})
            (p/'metrics.jsonl').write_text(json.dumps(rows[0])+'\n');(p/'snapshots.jsonl').write_text('')
            write_json(p/'provenance.json',{k:sha256(p/f) for k,f in [('manifest_sha256','manifest.json'),
                ('metrics_sha256','metrics.jsonl'),('snapshots_sha256','snapshots.jsonl')]})
            self.assertEqual(len(load_capture(p,'plain')[1]),1)
            with (p/'metrics.jsonl').open('a') as f:f.write(' \n')
            with self.assertRaisesRegex(ValueError,'SHA'):load_capture(p,'plain')
    def test_require_failclosed(self):
        with self.assertRaises(ValueError):require(False,'fail')

if __name__=='__main__': unittest.main()
