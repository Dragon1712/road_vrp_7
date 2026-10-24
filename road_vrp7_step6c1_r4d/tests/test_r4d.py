from __future__ import annotations
import copy
import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT=pathlib.Path(__file__).resolve().parents[2]
for name in ('road_vrp7_step6c1_r4d','road_vrp7_step6c1_r4c'):
    sys.path.insert(0,str(ROOT/name))
from r4d_core import analyze_records,summarize,validate_inputs,sha256,STAGE
from r4c_core import branch_spec,unit_name,normalize_reward,STAGE as R4C_STAGE
from audit_r4d import audit


def fixture():
    sn=dict(task_id='task_0008',scenario_id='2025-07-22_0800',seed=60,stop_count=505)
    results={};branches=[];labels=[]
    for k in (1,5,20,50):
        for arm in ('C3','SHAM_C3','A007'):
            alt=arm=='A007'
            event=[]
            for i in range(k):
                current='curr-ALT' if alt and i>0 else 'curr-C3'
                event.append({'stop_count':505+i,'destroy_index':3 if i==0 else 0,
                  'repair_index':0,'candidate_route_sha256':('different' if alt else 'same') if i==0 else 'later',
                  'current_route_sha256':'start' if i==0 else current,
                  'outcome':2 if alt and i==0 else 3,'anneal_temperature':100-i})
            if alt and k==1: current_hash='curr-ALT'
            else:current_hash='curr-ALT' if alt else 'curr-C3'
            best_hash='best-ALT' if alt and k>=20 else 'best-C3'
            spec={**sn,'horizon':k,'arm':arm}
            res={'snapshot_spec':sn,'horizon':k,'arm':arm,
                 'first_action_id':[1,1,2] if alt else [0,0,2],
                 'first_original_c3_action_id':[0,0,2],
                 'first_frozen_operator_pair':[3,0],
                 'events':event,
                 'end_signature':{'best_route_sha256':best_hash,'current_route_sha256':current_hash},
                 'best_route_sha256':best_hash,'current_route_sha256':current_hash,
                 'road_best_sec':90. if alt and k>=20 else 100.,
                 'road_current_sec':110. if alt else 120.,
                 'static_best_sec':80. if alt and k>=20 else 100.,
                 'static_current_sec':105. if alt else 120.,}
            key=('task_0008','2025-07-22_0800',60,505,k,arm)
            results[key]=res
            branches.append({'unit_spec':spec,'first_action':res['first_action_id'],
               'best_road':res['road_best_sec'],'best_static':res['static_best_sec'],
               'sha':res['best_route_sha256']})
            if alt:labels.append({'unit_spec':spec,'first_action':[1,1,2],
                'c3_action':[0,0,2],**normalize_reward(results[('task_0008','2025-07-22_0800',60,505,k,'C3')],res)})
    # Ensure optional comparison of SHAM is strict. Keep the exact same data as C3.
    snapshots=[{**sn,'current_route_sha256':'initial','best_route_sha256':'oldbest','rng_state_sha256':'seedstate'}]
    return labels,branches,results,snapshots


class R4DTests(unittest.TestCase):
    def test_pass_and_funnel(self):
        labels,branches,results,snaps=fixture()
        rows,starts=analyze_records(labels,branches,results)
        rep,ss=summarize(rows,snaps)
        self.assertEqual(len(rows),4)
        self.assertEqual(rep['per_horizon']['1']['road_best_nonzero'],0)
        self.assertEqual(rep['per_horizon']['20']['road_best_positive'],1)
        self.assertEqual(rep['per_horizon']['50']['terminal_current_changed'],1)
        self.assertEqual(rep['per_horizon']['5']['first_accepted'],1)
        self.assertEqual(rep['n_distinct_preselection_search_states'],1)
        self.assertEqual(len(ss),4)

    def test_sham_mutation_rejected(self):
        labs,bs,units,_=fixture()
        units[('task_0008','2025-07-22_0800',60,505,20,'SHAM_C3')]['events'][3]['outcome']=0
        with self.assertRaises(ValueError):analyze_records(labs,bs,units)

    def test_reward_mutation_rejected(self):
        labs,bs,units,_=fixture()
        labs[-1]['road_best_gain_sec']+=2
        with self.assertRaises(ValueError):analyze_records(labs,bs,units)

    def test_prefix_mutation_rejected(self):
        labs,bs,units,_=fixture()
        units[('task_0008','2025-07-22_0800',60,505,50,'A007')]['events'][3]['candidate_route_sha256']='tampered'
        with self.assertRaises(ValueError):analyze_records(labs,bs,units)

    def test_missing_candidate_rejected(self):
        labs,bs,units,_=fixture()
        with self.assertRaises(ValueError):analyze_records(labs[:-1],bs,units)

    def test_repeated_scenario_grouping(self):
        labs,bs,units,s=fixture()
        s.append({**s[0],'scenario_id':'2025-07-22_1700'})
        rows,_=analyze_records(labs,bs,units)
        report,_=summarize(rows,s)
        self.assertEqual(report['n_snapshots'],2)
        self.assertEqual(report['n_distinct_preselection_search_states'],1)
        self.assertEqual(report['n_physical_states_repeated_across_scenarios'],1)

    def make_disk(self,root):
        labs,bs,units,snaps=fixture()
        p=root/'r4c';(p/'units').mkdir(parents=True)
        sha={'counterfactual_r4c.py':sha256(ROOT/'road_vrp7_step6c1_r4c/counterfactual_r4c.py'),
             'r4c_core.py':sha256(ROOT/'road_vrp7_step6c1_r4c/r4c_core.py')}
        r4a=root/'r4a';r4b=root/'r4b';r4a.mkdir();r4b.mkdir()
        (r4a/'snapshots.jsonl').write_text('\n'.join(json.dumps(x) for x in snaps)+'\n')
        for name in ('report.json','manifest.json','paired_rows.jsonl'):
            (r4b/name).write_text('{}\n')
        mf={'stage':R4C_STAGE,'evaluation':'FIXED_ITERATION_K_NOT_WALL_CLOCK_30SEC',
            'data_role':'INSPECTED_JULY_VALIDATION_DEVELOPMENT_ONLY','horizons':[1,5,20,50],
            'max_alternatives':8,'code_sha256':sha,
            'r4a_snapshots_sha256':sha256(r4a/'snapshots.jsonl'),
            'r4b_report_sha256':sha256(r4b/'report.json'),
            'r4b_manifest_sha256':sha256(r4b/'manifest.json'),
            'r4b_paired_rows_sha256':sha256(r4b/'paired_rows.jsonl')}
        (p/'manifest.json').write_text(json.dumps(mf))
        (p/'report.json').write_text(json.dumps({'stage':R4C_STAGE,'status':'COMPLETE',
           'n_counterfactual_labels':len(labs),'n_branch_results':len(bs),
           'n_snapshot_start_points':1,'n_sham_control_checks':4}))
        for fn,rows in [('counterfactual_labels.jsonl',labs),('branch_summary.jsonl',bs),
                        ('plans.jsonl',[dict(task_id='task_0008',scenario_id='2025-07-22_0800',seed=60,stop_count=505)])]:
            (p/fn).write_text('\n'.join(json.dumps(z) for z in rows)+'\n')
        plan={'c3_identity':[0,0,2],
             'alternatives':[{'name':'A007','action_index':7,'route':1,'pos':1,'size':2}]}
        for k,v in units.items():
            spec=dict(zip(('task_id','scenario_id','seed','stop_count','horizon','arm'),k))
            file=p/'units'/unit_name(spec)
            file.write_text(json.dumps({'stage':R4C_STAGE,'unit_spec':spec,
                'plan':plan,'result':v,'checkpoint_source_sha256':mf['r4a_snapshots_sha256']}))
        return p,r4a,r4b

    def test_complete_readonly_disk_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a,b=self.make_disk(pathlib.Path(tmp))
            args=SimpleNamespace(r4c_dir=str(p),r4c_source_dir=str(ROOT/'road_vrp7_step6c1_r4c'),
                                 capture_dir=str(a),r4b_dir=str(b),output_dir=str(pathlib.Path(tmp)/'output'))
            rep=audit(args)
            self.assertEqual(rep['n_validated_full_units'],12)
            self.assertEqual(rep['per_horizon']['50']['road_best_positive'],1)
            self.assertTrue(pathlib.Path(args.output_dir,'report.md').exists())
            self.assertTrue(pathlib.Path(args.output_dir,'provenance.json').exists())

    def test_source_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a,b=self.make_disk(pathlib.Path(tmp))
            mf=json.loads((p/'manifest.json').read_text());mf['code_sha256']['r4c_core.py']='0'*64
            (p/'manifest.json').write_text(json.dumps(mf))
            with self.assertRaises(ValueError):validate_inputs(p,source_dir=ROOT/'road_vrp7_step6c1_r4c')

    def test_tampered_unit_spec_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a,b=self.make_disk(pathlib.Path(tmp))
            unit=next((p/'units').glob('*.json'))
            data=json.loads(unit.read_text());data['result']['first_original_c3_action_id']=[7,7,7]
            unit.write_text(json.dumps(data))
            args=SimpleNamespace(r4c_dir=str(p),r4c_source_dir=str(ROOT/'road_vrp7_step6c1_r4c'),
                capture_dir=str(a),r4b_dir=str(b),output_dir=str(pathlib.Path(tmp)/'out'))
            with self.assertRaises(ValueError):audit(args)

if __name__=='__main__':unittest.main()
