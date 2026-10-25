import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from r4h_core import (PREFIX_STAGE,CAPTURE_STAGE,sha,load,save,jsonl,parse_ks,
                      verified_allowlist,select_train_snapshots,case_round_robin,
                      paired_reward,snap_id)


def fixture(base, role='train'):
    d=Path(base);cap=d/'capture';proof=d/'proof';cap.mkdir();proof.mkdir()
    source=[];scenarios=['2025-07-02_0800','2025-07-09_1700']
    for sid in scenarios:
        for stop in (100,19805):
            source.append({'task_id':'task_0000','scenario_id':sid,'seed':60,
                'stop_count':stop,'rng_state_sha256':'same','current_route_sha256':'curr',
                'best_route_sha256':'best', 'roulette_destroy_weights':[1.,2.],
                'roulette_repair_weights':[1.,2.], 'anneal_temperature_before_accept':2.})
    save(cap/'manifest.json',{'stage':CAPTURE_STAGE,'split':role,'arm':'capture',
       'max_iterations':20000,'search_wall_sec':30,'task_ids':['task_0000'],
       'scenario_ids':scenarios,'seeds':[60]})
    (cap/'snapshots.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in source))
    (cap/'metrics.jsonl').write_text('row\n')
    save(cap/'provenance.json',{'manifest_sha256':sha(cap/'manifest.json'),
      'snapshots_sha256':sha(cap/'snapshots.jsonl'), 'metrics_sha256':sha(cap/'metrics.jsonl')})
    case_plan=[];paired=[];excluded=[]
    for sid in scenarios:
        windows=[]
        for stop in (100,19805):
            ks=[50,200,500] if stop==100 else [50]
            windows.append([stop,ks])
            for k in ks:
                paired.append({'task_id':'task_0000','scenario_id':sid,'seed':60,
                     'snapshot_stop_count':stop,'horizon':k,'verified_steps':k,'exact':True})
            for k in set((50,200,500))-set(ks):
                excluded.append({'task_id':'task_0000','scenario_id':sid,'seed':60,
                  'stop_count':stop,'horizon':k,'reason':'SOURCE_30S_PREFIX_TOO_SHORT'})
        case_plan.append(['task_0000',sid,60,windows])
    (proof/'paired_rows.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in paired))
    save(proof/'excluded_windows.json',excluded)
    save(proof/'manifest.json',{'stage':PREFIX_STAGE,'status':'CONFIGURED',
         'data_role':'TRAIN_ONLY','source_max_iterations':20000,'source_wall_sec':30,
         'case_plan':case_plan,
         'source_sha256':{'capture_snapshots':sha(cap/'snapshots.jsonl'),
          'capture_manifest':sha(cap/'manifest.json'),'capture_metrics':sha(cap/'metrics.jsonl'),
          'capture_provenance':sha(cap/'provenance.json')}})
    save(proof/'report.json',{'stage':PREFIX_STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
         'n_cases':2,'eligible_snapshots':4, 'exact_matches':len(paired),'n_comparisons':len(paired),
         'original_capture_snapshots_sha256':sha(cap/'snapshots.jsonl'),
         'per_horizon':{'50':{'exact':4},'200':{'exact':2},'500':{'exact':2}},
         'excluded_horizon_count':len(excluded)})
    return proof,cap


class ProofGates(unittest.TestCase):
    def test_actual_manifest_configured_is_valid(self):
        with tempfile.TemporaryDirectory() as d:
            p,c=fixture(d);a,s,r,m,cap=verified_allowlist(p,c,[50,200,500]);self.assertEqual(len(a),8)
            selected=select_train_snapshots(a,s,[50,200,500],0,0)
            self.assertEqual(len(selected),2)
            self.assertEqual({key[3] for key,_,_ in selected},{100})
    def test_train_capture_only(self):
        with tempfile.TemporaryDirectory() as d:
            p,c=fixture(d,'validation')
            with self.assertRaisesRegex(ValueError,'strictly TRAIN'):verified_allowlist(p,c,[50,200,500])
    def test_tampered_source_sha_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,c=fixture(d);(c/'snapshots.jsonl').write_text((c/'snapshots.jsonl').read_text()+' \n')
            with self.assertRaisesRegex(ValueError,'different capture'):verified_allowlist(p,c,[50,200,500])
    def test_missing_proof_event_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,c=fixture(d);rows=jsonl(p/'paired_rows.jsonl');rows[-1]['exact']=False
            (p/'paired_rows.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaisesRegex(ValueError,'Unverified'):verified_allowlist(p,c,[50,200,500])
    def test_excluded_horizon_not_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            p,c=fixture(d);a,s,_,_,_=verified_allowlist(p,c,[50,200,500]);q=select_train_snapshots(a,s,[50,200,500],0,0,False)
            self.assertEqual(len(q),4)
    def test_per_task_round_robin(self):
        k=[('task_0000','one',60),('task_0000','two',61),('task_0001','one',60),('task_0002','one',60)]
        self.assertEqual([z[0] for z in case_round_robin(k)][:3],['task_0000','task_0001','task_0002'])
    def test_conflicting_counter_rejected(self):
        with self.assertRaisesRegex(ValueError,'Conflicting'):snap_id({'task_id':'a','scenario_id':'b','seed':1,'stop_count':2,'snapshot_stop_count':3})
    def test_reward_definition(self):
        b={'arm':'C3','snapshot_spec':{'task_id':'T','scenario_id':'S','seed':0,'stop_count':100},
          'horizon':500,'road_best_sec':100,'road_current_sec':130,
          'static_best_sec':80,'static_current_sec':120,
          'best_route_sha256':'b','current_route_sha256':'c'}
        a={**b,'arm':'A001','road_best_sec':115,'road_current_sec':125,
            'static_best_sec':75,'static_current_sec':115,'best_route_sha256':'a'}
        res=paired_reward(b,a)
        self.assertEqual(res['road_best_gain_sec'],-15)
        self.assertEqual(res['road_current_gain_sec'],5)
        self.assertEqual(res['static_best_gain_sec'],5)
        self.assertFalse(res['same_best_routes'])
    def test_horizon_rejection(self):
        self.assertEqual(parse_ks('500,200,200,50'),[50,200,500])
        with self.assertRaisesRegex(ValueError,'subset'):parse_ks('1,50')
    def test_early_mid_late(self):
        with tempfile.TemporaryDirectory() as d:
            p,c=fixture(d);v,s,_,_,_=verified_allowlist(p,c,[50,200,500]);r=select_train_snapshots(v,s,[50],1,1)
            self.assertEqual(len(r),1)

if __name__=='__main__':unittest.main()
