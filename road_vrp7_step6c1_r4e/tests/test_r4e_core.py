import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r4e_core import (HORIZONS,STAGE_VERIFY,STAGE_CF,horizons,snapshot_id,
    physical_id,first_divergence,diagnose,aggregate,sha,read,write,read_lines,
    verify_r4b_evidence,verified_long_evidence)


def event(best='B',current='C',cand='X',outcome=2):
    return {'best_route_sha256':best,'current_route_sha256':current,
            'candidate_route_sha256':cand,'rng_state_sha256':'R',
            'destroy_index':3,'repair_index':0,'outcome':outcome}


def state(best='B',current='C'):
    return {'best_route_sha256':best,'current_route_sha256':current}


def branch(arm,events,end=None,k=None):
    return {'arm':arm,'snapshot_spec':{'task_id':'T0','scenario_id':'M','seed':60,'stop_count':5},
            'horizon':k or len(events),'events':events,
            'end_signature':end or state(), 'best_route_sha256':(end or state())['best_route_sha256'],
            'current_route_sha256':(end or state())['current_route_sha256'],
            'road_best_sec':100.0,'road_current_sec':150.0,
            'static_best_sec':90.0,'static_current_sec':130.0}


class R4ECore(unittest.TestCase):
    def test_horizons_strict(self):
        self.assertEqual(horizons('500,100,100,200'),(100,200,500))
        with self.assertRaises(ValueError):horizons('20,500')
        with self.assertRaises(ValueError):horizons('')
        with self.assertRaises(ValueError):horizons('100,501')

    def test_first_divergence_includes_last_post_step(self):
        a=branch('C3',[event(),event(),event()],state())
        b=branch('A01',[event(),event(),event()],state(best='DIFFERENT'))
        self.assertEqual(first_divergence(a,b,'best_route_sha256'),3)
        b['events'][1]['best_route_sha256']='DIFFERENT'
        self.assertEqual(first_divergence(a,b,'best_route_sha256'),1)
        b['events'][1]['best_route_sha256']='B'
        b['end_signature']['best_route_sha256']='B'
        self.assertIsNone(first_divergence(a,b,'best_route_sha256'))

    def test_accept_and_change_funnel(self):
        a=branch('C3',[event(),event(),event()])
        b=branch('A01',[event(cand='Y',outcome=1),event(current='X'),event()],state(current='X'))
        b['road_current_sec']=125.0
        x=diagnose(a,b)
        self.assertTrue(x['first_candidate_changed'])
        self.assertTrue(x['first_accepted'])
        self.assertEqual(x['first_current_divergence_step'],1)
        self.assertIsNone(x['first_best_divergence_step'])
        self.assertEqual(x['best_road_gain_sec'],0.0)
        self.assertEqual(x['current_road_gain_sec'],25.0)

    def test_rejected_action_not_called_accepted(self):
        a=branch('C3',[event(outcome=3)])
        b=branch('A01',[event(cand='Y',outcome=3)])
        self.assertFalse(diagnose(a,b)['first_accepted'])

    def test_unpaired_rng_fails_closed(self):
        a=branch('C3',[event()]);b=branch('A01',[event()])
        b['events'][0]['rng_state_sha256']='OTHER'
        with self.assertRaisesRegex(ValueError,'Unpaired'):diagnose(a,b)

    def test_mismatched_same_routes_road_cost_rejected(self):
        a=branch('C3',[event()]);b=branch('A01',[event()])
        b['road_best_sec']=98
        with self.assertRaisesRegex(ValueError,'Same best route'):diagnose(a,b)

    def test_aggregate_no_false_independence(self):
        spec={'task_id':'T0','scenario_id':'M0','seed':60,'stop_count':100,'horizon':100,'arm':'A01'}
        physical=['T0',60,100,'C','B','R']
        r={'unit_spec':spec,'physical_id':physical,'best_road_gain_sec':12.0,
            'first_accepted':True,'first_candidate_changed':True,
            'same_best_routes':False,'same_current_routes':False,
            'first_best_divergence_step':30}
        s={**r,'unit_spec':{**spec,'scenario_id':'M1','arm':'A01'},'best_road_gain_sec':-2.0}
        z=aggregate([r,s],(100,))['100']
        self.assertEqual(z['n_actions'],2)
        self.assertEqual(z['n_scenario_snapshots'],2)
        self.assertEqual(z['n_distinct_physical_search_states'],1)
        self.assertEqual(z['oracle_positive_scenario_snapshots'],1)
        self.assertEqual(z['positive_best'],1)

    def test_snapshot_id_accepts_real_r4b_schema(self):
        # Actual GitHub R4-B paired_rows.jsonl uses snapshot_stop_count.
        real_r4b_row = {'task_id': 'task_0008',
            'scenario_id': '2025-07-22_0800', 'seed': 60,
            'snapshot_stop_count': 505, 'horizon': 1,
            'verified_steps': 1, 'exact': True}
        self.assertEqual(snapshot_id(real_r4b_row),
            ('task_0008', '2025-07-22_0800', 60, 505))
        r4a_snapshot = {**real_r4b_row, 'stop_count': 505}
        self.assertEqual(snapshot_id(r4a_snapshot), snapshot_id(real_r4b_row))
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            snapshot_id({**r4a_snapshot, 'stop_count': 506})
        with self.assertRaisesRegex(ValueError, 'Missing'):
            snapshot_id({'task_id':'t', 'scenario_id':'s', 'seed':60})

    def test_r4b_proof_requires_all_16_by_4(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'source').mkdir();(d/'r4b').mkdir()
            (d/'source'/'snapshots.jsonl').write_text('[]\n')
            sh=sha(d/'source'/'snapshots.jsonl')
            write(d/'r4b'/'report.json',{'stage':'R4B-OFFICIAL-EXACT-CONTINUATION-AUDIT-v1',
                'status':'COMPLETE','n_comparisons':64,'exact_matches':64,'source_snapshot_sha256':sh})
            write(d/'r4b'/'manifest.json',{'source_capture_snapshot_sha256':sh})
            rows=[]
            for i in range(16):
                for k in (1,5,20,50):
                    rows.append({'task_id':'task','scenario_id':str(i),'seed':42,
                        'snapshot_stop_count':100,'horizon':k,'verified_steps':k,'exact':True})
            with (d/'r4b'/'paired_rows.jsonl').open('w') as f:
                for r in rows:f.write(json.dumps(r)+'\n')
            self.assertEqual(len(verify_r4b_evidence(d/'r4b',d/'source')),16)
            rows[-1]['exact']=False
            with (d/'r4b'/'paired_rows.jsonl').open('w') as f:
                for r in rows:f.write(json.dumps(r)+'\n')
            with self.assertRaisesRegex(ValueError,'step not verified'):verify_r4b_evidence(d/'r4b',d/'source')

    def test_long_proof_rejects_source_changes_and_missing_500(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            for name in ('source','r4b','proof'):(d/name).mkdir()
            (d/'source'/'snapshots.jsonl').write_text('original')
            (d/'r4b'/'report.json').write_text('original R4B')
            (d/'r4b'/'paired_rows.jsonl').write_text('rows')
            pins={'source_capture_sha256':sha(d/'source'/'snapshots.jsonl'),
                'source_r4b_report_sha256':sha(d/'r4b'/'report.json'),
                'source_r4b_rows_sha256':sha(d/'r4b'/'paired_rows.jsonl')}
            write(d/'proof'/'manifest.json',{'stage':STAGE_VERIFY,'status':'COMPLETE',**pins})
            write(d/'proof'/'report.json',{'stage':STAGE_VERIFY,'status':'COMPLETE',
                'n_comparisons':2,'exact_matches':2,'horizons':[100,200]})
            with (d/'proof'/'paired_rows.jsonl').open('w') as f:
                for k in (100,200):f.write(json.dumps({'task_id':'T','scenario_id':'S','seed':42,
                    'stop_count':10,'horizon':k,'verified_steps':k,'exact':True})+'\n')
            with self.assertRaisesRegex(ValueError,'Incomplete long proof K coverage'):
                verified_long_evidence(d/'proof',d/'source',d/'r4b',(100,200,500))
            (d/'source'/'snapshots.jsonl').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'Capture mismatch'):
                verified_long_evidence(d/'proof',d/'source',d/'r4b',(100,200))

    def test_hash_and_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.json';write(p,{'z':1,'a':[2,3]})
            self.assertEqual(read(p)['a'],[2,3]);self.assertEqual(len(sha(p)),64)


if __name__=='__main__':unittest.main()
