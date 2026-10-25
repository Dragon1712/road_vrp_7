"""Full R4-H collector logic with synthetic ALNS/Road stubs.

These do NOT replace real official ALNS+Chicago testing on the server.
"""
import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import collect_train_cf as run
from r4h_core import sha
from test_r4h_core import fixture


def stub_solver(break_sham=False):
    def spec(snapshot,k,arm):
        return {'task_id':snapshot['task_id'],'scenario_id':snapshot['scenario_id'],
                'seed':snapshot['seed'],'stop_count':snapshot['stop_count'],
                'horizon':k,'arm':arm}
    def filename(z):
        return hashlib.sha256(json.dumps(z,sort_keys=True).encode()).hexdigest()[:32]+'.json'
    def shadow(s,obj,action_count,seed,max_alts):
        return {'c3_identity':[0,0,2],
            'alternatives':[{'name':'A007','route':1,'pos':0,'size':2}],
            'candidate_count':10,'first_rng_after_original_C3_destroy_sha256':'z'}
    def compute(v3,task,sc,seed,obj,s,k,arm,opt,c3,road,legal,mf,rng):
        base={'C3':100.,'SHAM_C3':100.,'A007':95. if k<500 else 120.}[arm]
        if break_sham and arm=='SHAM_C3':base=101.
        info={'task_id':s['task_id'],'scenario_id':s['scenario_id'],
              'seed':s['seed'],'stop_count':s['stop_count']}
        return {'arm':arm,'snapshot_spec':info,'horizon':k,'events':list(range(k)),
            'first_original_c3_action_id':[0,0,2],
            'first_action_id':[0,0,2] if arm in ('C3','SHAM_C3') else [1,0,2],
            'road_best_sec':base,'road_current_sec':base+10,'static_best_sec':base-1,
            'static_current_sec':base+9,'best_route_sha256':str(base),
            'current_route_sha256':str(base+10)}
    def compare_sham(base,sham):
        if any(base[x]!=sham[x] for x in ('events','best_route_sha256','road_best_sec','road_current_sec')):
            raise ValueError('SHAM mismatch')
    c=types.SimpleNamespace(branch_spec=spec,unit_name=filename,compare_sham=compare_sham)
    r=types.SimpleNamespace(shadow_action=shadow,run_one=compute)
    return r,c


class Runner(unittest.TestCase):
    def do(self,break_sham=False,partial=False):
        with tempfile.TemporaryDirectory() as d:
            proof,cap=fixture(d)
            rows,smap,p_rep,p_m,cap_m=run.verified_allowlist(proof,cap,[50,200,500])
            r4c,core=stub_solver(break_sham)
            mf={'solver_sha256':'solver','section_sha256':'section'}
            cfg={'od_step_min':20,'od_hours':4}
            taskmap={'task_0000':{'task_id':'task_0000',
                     'scenarios':[{'scenario_id':'2025-07-02_0800'},
                                  {'scenario_id':'2025-07-09_1700'}]}}
            def pre(a):
                return (mf,[],taskmap,None,lambda *a,**kw:object(),
                        lambda *a:object(),lambda *a:None,object(),cfg,r4c,core)
            def builder(a,ks,selected,p,pman,cman):
                return {'stage':'TEST','source_sha256':{
                   'prefix_manifest':sha(Path(a.prefix_dir)/'manifest.json'),
                   'capture_snapshots':sha(Path(a.capture_dir)/'snapshots.jsonl'),
                   'frozen_solver':'solver','frozen_section':'section'},
                   'selected_snapshot_ids':[list(x[0]) for x in selected]}
            class FakeCheck:
                @staticmethod
                def validate_checkpoint(s,expected_max_iter):
                    assert expected_max_iter==20000
            fake_a=types.ModuleType('r4a_core')
            fake_a.canonical_hash=lambda x:json.dumps(x,sort_keys=True)
            args=SimpleNamespace(prefix_dir=str(proof),capture_dir=str(cap),
                   output_dir=str(Path(d)/'out'),mode='collect',
                   horizons='50,200,500',max_cases=1,max_snapshots_per_case=1,
                   action_count=8,max_alternatives=1,sampler_seed=42,
                   max_new_units=3 if partial else 0,resume=False,od_cache_dir='none')
            with (patch.object(run,'preflight',pre),
                  patch.object(run,'build',builder),
                  patch.dict(sys.modules,{'r4b_core':FakeCheck,
                                          'r4a_core':fake_a})):
                if break_sham:
                    with self.assertRaisesRegex(ValueError,'SHAM mismatch'):
                        run.execute(args)
                    self.assertFalse((Path(args.output_dir)/'report.json').exists())
                    return
                run.execute(args)
                if partial:
                    self.assertTrue((Path(args.output_dir)/'partial.json').exists())
                    self.assertFalse((Path(args.output_dir)/'report.json').exists())
                    args.resume=True;args.max_new_units=0
                    run.execute(args)
                rep=json.loads((Path(args.output_dir)/'report.json').read_text())
                self.assertEqual(rep['n_branches'],9)
                self.assertEqual(rep['n_labels'],3)
                self.assertEqual(rep['n_sham_checks'],3)
                self.assertEqual(rep['per_horizon']['50']['positive'],1)
                self.assertEqual(rep['per_horizon']['500']['negative'],1)
                self.assertEqual(rep['n_distinct_physical_states'],1)
                self.assertEqual(len(list((Path(args.output_dir)/'units').glob('*.json'))),9)

    def test_complete_mocked_smoke(self):self.do()
    def test_partial_and_resumed_smoke(self):self.do(partial=True)
    def test_negative_control_fails_closed(self):self.do(break_sham=True)

if __name__=='__main__':unittest.main()
