import json,sys,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from r4f_core import (LONG,SHORT,sha,dump,audit_sources,audit_data,analyze,
                      require,ident,sign,make_markdown)


def create_fixture(root,scenarios=('0800',)):
    d=Path(root);long=d/'long';short=d/'short';long.mkdir();short.mkdir()
    src=[['task_0008','2025-07-22_'+sc,60,505] for sc in scenarios]
    lm={'stage':LONG,'status':'COMPLETE','horizons':[100,200,500],
        'selected_snapshots':src,'data_role':'INSPECTED_JULY_VALIDATION_DEVELOPMENT_ONLY',
        'r4c_manifest_sha256':'','r4c_report_sha256':''}
    sm={'stage':SHORT,'evaluation':'FIXED_ITERATION_K_NOT_WALL_CLOCK_30SEC'}
    sr={'stage':SHORT,'status':'COMPLETE','n_counterfactual_labels':len(src),
        'n_branch_results':len(src)*3}
    dump(short/'manifest.json',sm);dump(short/'report.json',sr)
    lm['r4c_manifest_sha256']=sha(short/'manifest.json');lm['r4c_report_sha256']=sha(short/'report.json')
    l=[];b=[];s=[];sb=[]
    for spec in src:
        tid,scenario,seed,stop=spec
        phy=[tid,seed,stop,'a'*64,'b'*64,'c'*64]
        for h in [100,200,500]:
            baseline={100:1000.,200:900.,500:800.}[h]
            alternate={100:990.,200:850.,500:820.}[h]
            for arm,cost,route in [('C3',baseline,'B'+str(h)),('SHAM_C3',baseline,'B'+str(h)),('A001',alternate,'A'+str(h))]:
                b.append({'unit_spec':{'task_id':tid,'scenario_id':scenario,'seed':seed,'stop_count':stop,'horizon':h,'arm':arm},
                    'best_road_sec':cost,'current_road_sec':cost,'best_route_sha256':route})
            l.append({'unit_spec':{'task_id':tid,'scenario_id':scenario,'seed':seed,'stop_count':stop,'horizon':h,'arm':'A001'},
                'physical_id':phy,'best_road_gain_sec':baseline-alternate,
                'best_static_gain_sec':baseline-alternate,'same_best_routes':False,
                'first_action_id':[0,1,2],'original_c3_action_id':[1,1,2],
                'first_accepted':True,'first_outcome':2})
        for arm,cost,route in [('C3',1100.,'B50'),('SHAM_C3',1100.,'B50'),('A001',1100.,'B50')]:
            sb.append({'unit_spec':{'task_id':tid,'scenario_id':scenario,'seed':seed,'stop_count':stop,'horizon':50,'arm':arm},
                       'best_road':cost,'best_static':1000.,'sha':route})
        s.append({'unit_spec':{'task_id':tid,'scenario_id':scenario,'seed':seed,'stop_count':stop,'horizon':50,'arm':'A001'},
                  'road_best_gain_sec':0.,'static_best_gain_sec':0.,'same_best_routes':True,
                  'first_action':[0,1,2],'c3_action':[1,1,2]})
    lr={'stage':LONG,'status':'COMPLETE','horizons':[100,200,500],
        'n_candidate_horizon_labels':len(l),'n_branch_results':len(b),
        'n_snapshots':len(src),'n_physical_states':1,'n_sham_verified':len(src)*3,
        'n_prior_k50_prefix_verified':len(b)}
    dump(long/'manifest.json',lm);dump(long/'report.json',lr)
    for folder,filename,records in [(long,'labels.jsonl',l),(long,'branches.jsonl',b),
                                    (short,'counterfactual_labels.jsonl',s),(short,'branch_summary.jsonl',sb)]:
        (folder/filename).write_text(''.join(json.dumps(x)+'\n' for x in records))
    return long,short


def load_fixture(long,short):
    data,pins=audit_sources(long,short)
    m,phy,road=audit_data(*data[:2],*data[2:])
    return analyze(m,phy,road)

class LongHorizonTests(unittest.TestCase):
    def test_sign_threshold(self):
        self.assertEqual([sign(x) for x in (-1.,0.,1.)],['-','0','+'])

    def test_ident_accepts_json_list(self):
        self.assertEqual(ident(['task_1','sc',42,120]),('task_1','sc',42,120))

    def test_reversal_and_baseline_catch_up_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            l,s=create_fixture(tmp)
            a,t,o=load_fixture(l,s)
            self.assertEqual(a['per_horizon']['200']['mean_road_gain_sec'],50.)
            self.assertEqual(a['per_horizon']['500']['mean_road_gain_sec'],-20.)
            self.assertEqual(a['transitions']['200_to_500']['sign_counts']['+→-'],1)
            rec=[x for x in t if x['from_K']==200][0]
            self.assertEqual(rec['C3_road_progress_sec'],100)
            self.assertEqual(rec['alternative_road_progress_sec'],30)
            self.assertEqual(rec['gain_change_sec'],-70)
            self.assertIn('统计边界',make_markdown({'analysis':a}))

    def test_duplicate_scenarios_are_one_physical_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            l,s=create_fixture(tmp,('0800','1700'))
            a,_,_=load_fixture(l,s)
            self.assertEqual(a['n_scenario_snapshots'],2)
            self.assertEqual(a['n_unique_physical_states'],1)
            self.assertEqual(a['transitions']['200_to_500']['positive_to_negative_distinct_physical'],1)

    def test_corrupt_label_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            l,s=create_fixture(tmp)
            p=l/'labels.jsonl';z=[json.loads(k) for k in p.read_text().splitlines()];z[1]['best_road_gain_sec']+=5
            p.write_text(''.join(json.dumps(x)+'\n' for x in z))
            with self.assertRaisesRegex(ValueError,'Road label'):
                load_fixture(l,s)

    def test_corrupt_sham_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            l,s=create_fixture(tmp)
            p=l/'branches.jsonl';z=[json.loads(k) for k in p.read_text().splitlines()];z[1]['best_route_sha256']='attack'
            p.write_text(''.join(json.dumps(x)+'\n' for x in z))
            with self.assertRaisesRegex(ValueError,'SHAM'):
                load_fixture(l,s)

    def test_invalid_source_sha_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            l,s=create_fixture(tmp)
            m=json.loads((l/'manifest.json').read_text());m['r4c_report_sha256']='0'*64;dump(l/'manifest.json',m)
            with self.assertRaisesRegex(ValueError,'SHA'):
                load_fixture(l,s)

    def test_missing_horizon_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            l,s=create_fixture(tmp)
            p=l/'labels.jsonl';z=[json.loads(k) for k in p.read_text().splitlines()][:-1]
            p.write_text(''.join(json.dumps(x)+'\n' for x in z))
            with self.assertRaisesRegex(ValueError,'row count'):
                load_fixture(l,s)

    def test_first_action_changed_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            l,s=create_fixture(tmp)
            p=l/'labels.jsonl';z=[json.loads(k) for k in p.read_text().splitlines()];z[-1]['first_action_id']=[3,2,2]
            p.write_text(''.join(json.dumps(x)+'\n' for x in z))
            with self.assertRaisesRegex(ValueError,'First action'):
                load_fixture(l,s)

    def test_longitudinal_branch_cost_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            l,s=create_fixture(tmp)
            p=l/'branches.jsonl';z=[json.loads(k) for k in p.read_text().splitlines()];z[-1]['best_road_sec']+=11
            p.write_text(''.join(json.dumps(x)+'\n' for x in z))
            with self.assertRaisesRegex(ValueError,'Road label'):
                load_fixture(l,s)

if __name__=='__main__':unittest.main()
