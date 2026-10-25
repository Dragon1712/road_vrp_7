"""Synthetic golden path, negative controls, partial coverage and source SHA tests."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from r4k_core import *
from audit_r4k import run


def fixture(tmp, partial=False):
    tmp=Path(tmp);cp=tmp/'capture';pf=tmp/'prefix';cf=tmp/'cf'
    for p in (cp,pf,cf,cf/'units'):p.mkdir()
    s=('task_0002','2025-07-02_0800',62,118)
    source={'task_id':s[0],'scenario_id':s[1],'seed':s[2],'stop_count':s[3],
       'current_route_sha256':'cur','best_route_sha256':'best','rng_state_sha256':'rng',
       'roulette_destroy_weights':[1.,2.,3.], 'roulette_repair_weights':[1.,3.],
       'anneal_temperature_before_accept':10.,'real_selected_repair_index':0}
    cap={'stage':CAPTURE_STAGE,'split':'train','arm':'capture','task_ids':['task_0002'],
         'scenario_ids':[s[1]],'seeds':[62],'max_iterations':20000}
    save(cp/'manifest.json',cap);save(cp/'report.json',{'status':'COMPLETE','snapshots':1})
    (cp/'snapshots.jsonl').write_text(json.dumps(source)+'\n')
    (cp/'metrics.jsonl').write_text('[]\n');save(cp/'provenance.json',{'schema':'test'})
    rows=[{'task_id':s[0],'scenario_id':s[1],'seed':s[2],
          'snapshot_stop_count':s[3],'horizon':k,'verified_steps':k,'exact':True} for k in KS]
    (pf/'paired_rows.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in rows))
    save(pf/'report.json',{'stage':PREFIX_STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
         'n_cases':1,'n_comparisons':3,'exact_matches':3,'original_capture_snapshots_sha256':hash_file(cp/'snapshots.jsonl')})
    save(pf/'manifest.json',{'stage':PREFIX_STAGE,'data_role':'TRAIN_ONLY'})
    save(pf/'excluded_windows.json',[])
    cfmanifest={'stage':CF_STAGE,'data_role':'TRAIN_ONLY',
         'horizons':list(KS),'require_all_k':True,'eval_semantics':'FIXED_K_NOT_REMAINING_30S',
         'max_alternatives':8,'sampler_seed':20261019,'selected_snapshot_ids':[list(s)],
         'source_sha256':{k:hash_file(v) for k,v in {
           'prefix_report':pf/'report.json','prefix_manifest':pf/'manifest.json',
           'prefix_paired_rows':pf/'paired_rows.jsonl','prefix_excluded_windows':pf/'excluded_windows.json',
           'capture_manifest':cp/'manifest.json','capture_provenance':cp/'provenance.json',
           'capture_snapshots':cp/'snapshots.jsonl','capture_metrics':cp/'metrics.jsonl'
         }.items()}}
    save(cf/'manifest.json',cfmanifest)
    plan={'c3_identity':[0,0,2],'alternatives':[{'name':'A001','route':0,'pos':1,'size':2}],
          'sampling_seed':20261019}
    def output(k,arm):
        price={'C3':{50:1000.,200:900.,500:800.},
               'SHAM_C3':{50:1000.,200:900.,500:800.},
               'A001':{50:990.,200:850.,500:820.}}[arm][k]
        events=[]
        for i in range(k):
            events.append({'stop_count':s[3]+i,'current_route_sha256':'cur',
                'best_route_sha256':'best','rng_state_sha256':'rng','destroy_index':3,
                'repair_index':0,'outcome':'A' if arm=='A001' else 'C3'})
        chosen=[0,1,2] if arm=='A001' else [0,0,2]
        return {'snapshot_spec':{'task_id':s[0],'scenario_id':s[1],'seed':s[2],'stop_count':s[3]},
               'horizon':k,'arm':arm,'first_action_id':chosen,'first_original_c3_action_id':[0,0,2],
               'events':events,'end_signature':{'route':'a' if arm=='A001' else 'base'},
               'road_best_sec':price,'road_current_sec':price,'static_best_sec':price,
               'static_current_sec':price,'best_route_sha256':'a' if arm=='A001' else 'base',
               'current_route_sha256':'a' if arm=='A001' else 'base',
               'first_section_audit':{'substitution_calls':0 if arm=='C3' else 1,
                    'original_destroy_draws_preserved':True,'sham_partial_equal':arm=='SHAM_C3'}}
    existing=[]
    for k in KS:
        for arm in ['C3','SHAM_C3','A001']:
            if partial and k==500 and arm=='A001':continue
            spec={'task_id':s[0],'scenario_id':s[1],'seed':s[2],'stop_count':s[3],
                  'horizon':k,'arm':arm}
            res=output(k,arm)
            save(cf/'units'/official_unit_filename(spec),{'stage':CF_STAGE,'unit_spec':spec,
                 'plan':plan,'result':res,
                 'source_manifest_sha256':cfmanifest['source_sha256']['prefix_manifest'],
                 'source_snapshots_sha256':cfmanifest['source_sha256']['capture_snapshots']})
            existing.append((spec,res))
    if not partial:
        save(cf/'report.json',{'stage':CF_STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
             'n_branches':len(existing),'n_labels':3,'n_scenario_snapshots':1,'n_sham_checks':3})
        with (cf/'labels.jsonl').open('w') as f:
            for spec,res in existing:
                if spec['arm']!='A001':continue
                base=output(spec['horizon'],'C3')
                gain=base['road_best_sec']-res['road_best_sec']
                f.write(json.dumps({'unit_spec':spec,'road_best_gain_sec':gain,
                    'road_current_gain_sec':gain,'static_best_gain_sec':gain})+'\n')
    return cp,pf,cf


class R4KTests(unittest.TestCase):
    def audit(self,cp,pf,cf,out,full=False):
        return run(Namespace(cf_dir=cf,prefix_dir=pf,capture_dir=cp,output_dir=out,require_complete=full))

    def test_complete_reversal_and_negative_controls(self):
        with tempfile.TemporaryDirectory() as d:
            c,p,f=fixture(d)
            self.audit(c,p,f,Path(d)/'r',True)
            rep=read(Path(d)/'r'/'report.json')
            self.assertEqual(rep['status'],'SOURCE_COMPLETE')
            self.assertEqual(rep['analysis']['per_horizon']['200']['positive'],1)
            self.assertEqual(rep['analysis']['per_horizon']['500']['negative'],1)
            self.assertEqual(rep['analysis']['horizon_transitions']['200_to_500']['positive_to_negative'],1)
    def test_partial_does_not_count_as_zero(self):
        with tempfile.TemporaryDirectory() as d:
            c,p,f=fixture(d,True)
            self.audit(c,p,f,Path(d)/'r')
            r=read(Path(d)/'r'/'report.json')
            self.assertEqual(r['status'],'PARTIAL_COVERAGE_ONLY')
            self.assertEqual(r['analysis']['per_horizon']['50']['action_labels'],0)
            self.assertEqual(r['analysis']['incomplete_scenario_snapshots'],1)
            self.assertTrue(read(Path(d)/'r'/'missing_units.json'))
    def test_partial_forbids_complete_claim(self):
        with tempfile.TemporaryDirectory() as d:
            c,p,f=fixture(d,True)
            with self.assertRaisesRegex(ValueError,'PARTIAL'):
                self.audit(c,p,f,Path(d)/'r',True)
    def test_sham_trace_corruption_fails(self):
        with tempfile.TemporaryDirectory() as d:
            c,p,f=fixture(d)
            spec={'task_id':'task_0002','scenario_id':'2025-07-02_0800','seed':62,
                  'stop_count':118,'horizon':50,'arm':'SHAM_C3'}
            u=f/'units'/official_unit_filename(spec);z=read(u);z['result']['events'][10]['outcome']='tampered';save(u,z)
            with self.assertRaisesRegex(ValueError,'SHAM'):
                self.audit(c,p,f,Path(d)/'r')
    def test_bad_proof_sha_fails(self):
        with tempfile.TemporaryDirectory() as d:
            c,p,f=fixture(d)
            with (p/'paired_rows.jsonl').open('a') as h:h.write('\n')
            with self.assertRaisesRegex(ValueError,'SHA pin'):
                self.audit(c,p,f,Path(d)/'r')
    def test_bad_plan_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            c,p,f=fixture(d)
            spec={'task_id':'task_0002','scenario_id':'2025-07-02_0800','seed':62,
                  'stop_count':118,'horizon':50,'arm':'A001'}
            u=f/'units'/official_unit_filename(spec);z=read(u);z['plan']['sampling_seed']=0;save(u,z)
            with self.assertRaisesRegex(ValueError,'Sampling seed'):
                self.audit(c,p,f,Path(d)/'r')
    def test_bad_source_role_fails(self):
        with tempfile.TemporaryDirectory() as d:
            c,p,f=fixture(d)
            cap=read(c/'manifest.json');cap['split']='validation';save(c/'manifest.json',cap)
            with self.assertRaisesRegex(ValueError,'role mismatch'):
                self.audit(c,p,f,Path(d)/'r')
    def test_old_result_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            c,p,f=fixture(d);out=Path(d)/'r'
            self.audit(c,p,f,out)
            with self.assertRaisesRegex(ValueError,'NEW output'):
                self.audit(c,p,f,out)
    def test_official_unit_digest_key_order_invariant(self):
        x={'task_id':'task_0000','scenario_id':'sc','seed':61,'stop_count':13,'horizon':50,'arm':'C3'}
        self.assertEqual(official_unit_filename(x),official_unit_filename(dict(reversed(list(x.items())))))
    def test_exact_temp_and_coarse_search_start(self):
        x={'task_id':'t','seed':61,'stop_count':13,'current_route_sha256':'a','best_route_sha256':'b',
           'rng_state_sha256':'c','roulette_destroy_weights':[1,2],
           'roulette_repair_weights':[2,3],'anneal_temperature_before_accept':1.0}
        y=dict(x,anneal_temperature_before_accept=1.0+1e-12)
        a,b=state_phys(x);c,d=state_phys(y)
        self.assertNotEqual(a,c);self.assertEqual(b,d)

if __name__=='__main__':unittest.main()
