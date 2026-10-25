"""Synthetic controller tests: mocks ONLY the frozen ALNS wrappers, not audits."""
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import verify_prefix as verifier

class TestWrapper(unittest.TestCase):
    def build(self,mutate=False):
        reference=types.ModuleType('verify_r4b')
        audited=types.ModuleType('r4b_core')
        role=types.ModuleType('r4g_core')
        role.validate_scenario_role=lambda task,sids: None
        @contextmanager
        def recording(windows):
            ds={i:{'step':i,'token':'ok'} for start,n in windows for i in range(start,start+n)}
            yield [SimpleNamespace(events=ds)]
        @contextmanager
        def exact(snapshot,k,holder):
            start=snapshot['stop_count']
            trace=[{'step':i,'token':'ok'} for i in range(start,start+k)]
            if mutate: trace[100]['token']='wrong'
            holder.append(trace)
            yield
        reference.reference_recording=recording
        reference.exact_resume=exact
        audited.require=lambda condition,reason:None if condition else (_ for _ in ()).throw(ValueError(reason))
        def check_start(s,e):
            if e['step']!=s['stop_count']:raise AssertionError('start mismatch')
        audited.check_start=check_start
        def compare(a,b):
            if a!=b:raise AssertionError('step mismatch')
        audited.verify_events=compare
        class OD:
            def __init__(self,*args,**kwargs):pass
        class Objective:
            def __init__(self,*args,**kwargs):pass
        def solve(*args,**kwargs):return {'iterations':20000}
        snap={'task_id':'task_0000','scenario_id':'2025-07-02_0800','seed':60,
              'stop_count':1000,'rng_state_sha256':'snapshotRNG'}
        ctx=({'max_iterations':20000,'top_k':12,'anneal_iterations':5000,'search_wall_sec':30},
             [],[],[snap],{'task_0000':{'scenarios':[{'scenario_id':'2025-07-02_0800'}]}},
             object(),OD,Objective,SimpleNamespace(solve=solve),
             {'od_step_min':20,'od_hours':5.}, {'pairs':1})
        args=SimpleNamespace(reference_wall_sec=90,od_cache_dir='cache',
                             capture_dir='unused')
        return {'verify_r4b':reference,'r4b_core':audited,'r4g_core':role},args,snap,ctx

    def test_longest_continuation_proves_all_shorter_k(self):
        modules,args,snap,context=self.build()
        with patch.dict(sys.modules,modules), patch.object(verifier,'sha256',return_value='source'):
            result=verifier.verify_case(args,('task_0000','2025-07-02_0800',60),
                  [{'snapshot':snap,'horizons':[1,50,200,500], 'max_horizon':500,
                    'source_min_iterations':20000}],context,'settings')
        self.assertEqual([r['verified_steps'] for r in result['rows']],[1,50,200,500])
        self.assertEqual(result['original_reference_iterations'],20000)

    def test_first_event_divergence_never_marked_exact(self):
        modules,args,snap,context=self.build(mutate=True)
        with patch.dict(sys.modules,modules), patch.object(verifier,'sha256',return_value='source'):
            with self.assertRaisesRegex(AssertionError,'step mismatch'):
                verifier.verify_case(args,('task_0000','2025-07-02_0800',60),
                  [{'snapshot':snap,'horizons':[1,50,200,500], 'max_horizon':500,
                    'source_min_iterations':20000}],context,'settings')

if __name__=='__main__':unittest.main()
