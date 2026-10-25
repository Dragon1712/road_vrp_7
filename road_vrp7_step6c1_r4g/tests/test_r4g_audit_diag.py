"""Regress strict-audit failure when real 20k/30sec TRAIN search hits wall cap."""
import json,sys,tempfile,unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
for folder in ('road_vrp7_step6c1_r4g','road_vrp7_step6c1_r4a'):
    sys.path.insert(0,str(ROOT/folder))
from r4g_core import STAGE,paired_audit
import audit_r4g

class AuditFailureDiagnostics(unittest.TestCase):
    def pair(self):
        def each(arm):
            manifest={'stage':STAGE,'split':'train','arm':arm,
                'task_ids':['task_0000'],'scenario_ids':['2025-07-02_0800'], 'seeds':[60],
                'search_wall_sec':30,'max_iterations':20000,'anneal_iterations':5000,'top_k':12,
                'source_sha256':'src','road_sha256':'road','edge_sha256':'edge',
                'solver_sha256':'solve','section_sha256':'section'}
            row={'task_id':'task_0000','scenario_id':'2025-07-02_0800','seed':60,
                'iterations':20000,'termination_reason':'iteration_budget',
                'best_route_sha256':'abc','road_cost_sec':3900.,'search_wall_sec':25.,
                'captured':1 if arm=='capture' else 0}
            snap={'task_id':'task_0000','scenario_id':'2025-07-02_0800','seed':60,'stop_count':100,
              'current_route_sha256':'curr','best_route_sha256':'best','rng_state_sha256':'rng',
              'roulette_destroy_weights':[1]*4,'roulette_repair_weights':[1]*2,
              'anneal_temperature_before_accept':22.}
            return (manifest,[row],[snap] if arm=='capture' else [])
        return each('plain'),each('capture')
    def test_wall_cap_but_same_routes_not_verified(self):
        plain,capture=self.pair()
        capture[1][0]['iterations']=18000
        capture[1][0]['termination_reason']='wall_budget'
        report=paired_audit(plain,capture)
        self.assertEqual(report['strict_verdict'],'NOT_VERIFIED')
        self.assertEqual(report['route_road_equal'],1)
        self.assertEqual(report['failure_reason_counts'],{'ITERATION_CAP_NOT_MET':1,'ITERATION_COUNT_DIFFERS':1})
        self.assertFalse(report['details'][0]['strong_pass'])
    def test_same_iter_but_route_changed_not_verified(self):
        plain,capture=self.pair()
        capture[1][0]['best_route_sha256']='different'
        capture[1][0]['road_cost_sec']=3970.
        report=paired_audit(plain,capture)
        self.assertTrue(report['details'][0]['iteration_limited_both'])
        self.assertEqual(set(report['details'][0]['failure_reasons']),{'FINAL_ROUTE_MISMATCH','ROAD_COST_MISMATCH'})
    def test_strict_failure_writes_report_before_raising(self):
        plain,capture=self.pair()
        capture[1][0]['iterations']=19900
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            p=base/'plain';c=base/'capture';p.mkdir();c.mkdir()
            for folder in (p,c):
                for file in ('manifest.json','metrics.jsonl','snapshots.jsonl'):(folder/file).write_text('stub')
            def loader(path,role): return plain if role=='plain' else capture
            with patch.object(audit_r4g,'load_capture',loader):
                with self.assertRaisesRegex(ValueError,'report.md'):
                    audit_r4g.run(Namespace(plain=str(p),capture=str(c),output_dir=str(base/'out'),require_strict=True))
            actual=json.loads((base/'out'/'report.json').read_text())
            self.assertEqual(actual['strict_verdict'],'NOT_VERIFIED')
            self.assertIn('ITERATION_CAP_NOT_MET',(base/'out'/'report.md').read_text())
    def test_diagnostic_report_success_without_strict_does_not_claim_proof(self):
        plain,capture=self.pair()
        capture[1][0]['iterations']=19900
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            p=base/'plain';c=base/'capture';p.mkdir();c.mkdir()
            for folder in (p,c):
                for file in ('manifest.json','metrics.jsonl','snapshots.jsonl'):(folder/file).write_text('stub')
            with patch.object(audit_r4g,'load_capture',lambda path,role:plain if role=='plain' else capture):
                audit_r4g.run(Namespace(plain=str(p),capture=str(c),output_dir=str(base/'diag'),require_strict=False))
            actual=json.loads((base/'diag'/'report.json').read_text())
            self.assertFalse(actual['can_claim_fixed_iteration_noninterference'])
    def test_full_iteration_equal_routes_verified(self):
        plain,capture=self.pair()
        rep=paired_audit(plain,capture)
        self.assertEqual(rep['strict_verdict'],'VERIFIED')
        self.assertEqual(rep['failure_reason_counts'],{})
if __name__=='__main__':unittest.main()
