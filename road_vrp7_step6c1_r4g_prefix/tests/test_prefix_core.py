import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prefix_core import (parse_horizons,plan_cases,summary,validate_unit,selected_manifest,
                         compare_prefix_events,write_json,read_json)


def sample_data():
    plain=[];capture=[];snaps=[]
    for i,stopcap in enumerate([20000,16000]):
        key={'task_id':'task_000'+str(i),'scenario_id':'2025-07-02_0800','seed':60}
        a={**key,'iterations':stopcap-100 if i else stopcap,
           'best_route_sha256':'route'+str(i),'road_cost_sec':3900.}
        b={**key,'iterations':stopcap,
           'best_route_sha256':'route'+str(i),'road_cost_sec':3900.}
        plain.append(a);capture.append(b)
        for v in ([1000,19805] if i==0 else [1000,15700]):
            snaps.append({**key,'stop_count':v,'rng_state_sha256':'rng'})
    return plain,capture,snaps

class TestPlan(unittest.TestCase):
    def test_K_range_and_counts(self):
        p,c,s=sample_data()
        plan,excluded=plan_cases(p,c,s,[1,50,200,500])
        self.assertEqual(len(plan),2)
        self.assertEqual(len(excluded),3)
        self.assertEqual(sum(len(w['horizons']) for v in plan.values() for w in v),13)
        self.assertEqual([w['max_horizon'] for w in plan[next(iter(plan))]],[500,50])
        self.assertEqual(sorted({z['horizon'] for z in excluded}),[200,500])

    def test_empty_horizon_rejected(self):
        with self.assertRaises(ValueError):parse_horizons('')
        with self.assertRaises(ValueError):parse_horizons('501')
        self.assertEqual(parse_horizons('200,1,200,50'),[1,50,200])

    def test_reject_changed_route(self):
        p,c,s=sample_data();c[0]['best_route_sha256']='different'
        with self.assertRaisesRegex(ValueError,'TERMINAL mismatch'):plan_cases(p,c,s,[1])

    def test_reject_changed_road(self):
        p,c,s=sample_data();c[0]['road_cost_sec']=3901.
        with self.assertRaisesRegex(ValueError,'TERMINAL mismatch'):plan_cases(p,c,s,[1])

    def test_reject_missing_capture_case(self):
        p,c,s=sample_data();s=[x for x in s if x['task_id']!='task_0001']
        with self.assertRaisesRegex(ValueError,'no Section snapshot'):plan_cases(p,c,s,[1])

    def test_reject_duplicate_snapshot_iteration(self):
        p,c,s=sample_data();s.append(dict(s[0]))
        with self.assertRaisesRegex(ValueError,'Duplicate checkpoint'):plan_cases(p,c,s,[1])

    def test_case_cap(self):
        p,c,s=sample_data()
        plan,exc=plan_cases(p,c,s,[1,50,200,500],max_cases=1,max_snapshots=1)
        self.assertEqual(len(plan),1)
        self.assertEqual(sum(len(w['horizons']) for v in plan.values() for w in v),4)
        self.assertEqual(len(exc),0)

    def test_saved_unit_source_and_row_verification(self):
        p,c,s=sample_data()
        plan,ex=plan_cases(p,c,s,[1,50,200,500],max_cases=1,max_snapshots=1)
        key,wins=next(iter(plan.items()))
        unit={'case':list(key),'source_snapshots_sha256':'source', 'settings_sha256':'settings',
              'rows':[{**dict(zip(['task_id','scenario_id','seed'],key)),
                      'snapshot_stop_count':1000,'horizon':k,'verified_steps':k,'exact':True}
                      for k in [1,50,200,500]]}
        self.assertEqual(len(validate_unit(unit,key,wins,'source','settings')),4)
        bad=json.loads(json.dumps(unit));bad['rows'][1]['exact']=False
        with self.assertRaisesRegex(ValueError,'unverified'):validate_unit(bad,key,wins,'source','settings')
        with self.assertRaisesRegex(ValueError,'source SHA'):validate_unit(unit,key,wins,'other','settings')

    def test_strict_trace_prefix_comparison(self):
        a=[{'step':i,'rng':str(i)} for i in range(500)]
        calls=[]
        def exact(x,y):
            calls.append(len(x))
            if x!=y:raise AssertionError('divergence')
        self.assertEqual(compare_prefix_events(a,list(a),[1,50,200,500],exact),4)
        self.assertEqual(calls,[1,50,200,500])
        b=list(a);b[128]={'step':128,'rng':'bad'}
        with self.assertRaises(AssertionError):compare_prefix_events(a,b,[1,50,200,500],exact)

    def test_summary_no_fake_pass(self):
        p,c,s=sample_data();plan,ex=plan_cases(p,c,s,[1,50])
        rows=[{'horizon':k,'exact':True} for vs in plan.values() for w in vs for k in w['horizons']]
        rpt=summary(plan,ex,rows,{'strict_pass':0})
        self.assertEqual(rpt['n_comparisons'],8)
        self.assertEqual(rpt['per_horizon']['50']['exact'],4)
        with self.assertRaisesRegex(ValueError,'Not all'):summary(plan,ex,rows[:-1],{})

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.json'
            write_json(path,{'a':'中文'})
            self.assertEqual(read_json(path),{'a':'中文'})
            self.assertFalse((Path(temp)/'data.json.tmp').exists())

if __name__=='__main__': unittest.main()
