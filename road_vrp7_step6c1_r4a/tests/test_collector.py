"""R4A v1.1 real collector control-flow regressions (pure stub environment).

In particular, the v1 plain branch must call v3.solve BEFORE using ans.
No Chicago files or official ALNS installation are required for these tests.
"""
import json
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import collect_r4a as collector
from compare_r4a import summary_pair


class TestCollectorArms(unittest.TestCase):
    def test_plain_and_capture_reach_complete_and_pair_exact(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            scenario = {'scenario_id':'2025-07-22_0800', 'split':'validation',
                        'eligible':True, 'departure_local':'2025-07-22 08:00'}
            task = {'task_id':'task_0008', 'scenarios':[scenario]}

            class Road:
                edge_file_hash = 'edgedigest'
                cost_file_hash = 'costdigest'
                def __init__(self, path): pass
                def replay(self, task, routes, departure):
                    return {'ex_post_road_travel_sec':4200.}

            class OD:
                def __init__(self, *args, **kwargs):pass

            class Objective:
                def __init__(self, *args, **kwargs):pass

            road_mod = types.ModuleType('road_vrp')
            road_mod.Road = Road
            road_mod.OD = OD
            road_mod.Objective = Objective
            road_mod.legal = lambda task, routes: None
            road_mod.read_json = lambda path: {
                'source_cost_hash':'costdigest', 'source_edge_hash':'edgedigest',
                'code_sha256':{
                    'road_vrp7_step5b_v3/section_v2.py':'sha',
                    'road_vrp7_stage3/road_vrp.py':'sha'
                }, 'od_step_min':20, 'od_hours':5
            }
            benchmark = types.ModuleType('benchmark')
            benchmark.load_tasks = lambda path: (
                [root/'task_0008.json'], [task],
                {'cost_edge_index_sha256':'edgedigest',
                 'cost_edge_npz_sha256':'costdigest'}
            )
            run_mod = types.ModuleType('run_experiment')
            def fake_solve(solver, task, objective, seed, wall, max_iterations,
                           *, top_k, anneal_iterations):
                calls.append((solver,seed,wall,max_iterations,top_k,anneal_iterations))
                return {'routes':[[1,2],[3,4]],'iterations':max_iterations,
                        'termination_reason':'iteration_budget',
                        'search_wall_sec':1., 'planned_cost':4100.}
            run_mod.solve = fake_solve
            fake_modules = {'road_vrp':road_mod, 'benchmark':benchmark,
                            'run_experiment':run_mod}

            class Observed:
                snapshots = [{
                    'boundary':'BEFORE_SELECT_AND_DESTROY',
                    'next_iteration_one_based':1,
                    'stop_count':1,
                    'completed_updates_before_boundary':0,
                    'rng_bit_generator_state':{'bit_generator':'PCG64','state':{'state':7,'inc':3}},
                    'current_routes':[[1,2],[3,4]],
                    'best_routes':[[1,2],[3,4]],
                }]
                def summary(self):
                    return {'selector_calls':2,'selector_updates':2,'snapshot_count':1}

            @contextmanager
            def stub_instrument(*args,**kwargs):
                yield [Observed()]

            with (patch.dict(sys.modules, fake_modules),
                  patch.object(collector,'ROOT',root),
                  patch.object(collector,'sha256',return_value='sha'),
                  patch.object(collector,'instrument_official_alns',stub_instrument),
                  patch('importlib.metadata.version',return_value='7.0.0')):
                for arm in ('plain','capture'):
                    args = SimpleNamespace(
                        arm=arm,output_dir=str(root/arm),model_source=str(root/'src'),
                        cost_dir=str(root/'cost'),task_dir=str(root/'task'),
                        od_cache_dir=str(root/'od'),task_ids='task_0008',
                        scenario_ids='2025-07-22_0800',seeds='60',wall_sec=30,
                        max_iterations=2,anneal_iterations=5000,top_k=12,
                        snapshots='1',max_snapshots=16,section_only=False)
                    collector.main(args)

            self.assertEqual(len(calls),2, 'Both arms MUST invoke frozen C3 solve')
            for call in calls:
                self.assertEqual(call,('C3_topk',60,30,2,12,5000))
            plain = root/'plain'
            capture = root/'capture'
            self.assertEqual(json.loads((plain/'report.json').read_text())['status'],'COMPLETE')
            self.assertEqual((plain/'snapshots.jsonl').read_text(),'')
            self.assertEqual(json.loads((capture/'report.json').read_text())['snapshots'],1)
            pair = summary_pair(plain,capture,require_exact=True)
            self.assertEqual(pair['strong_determinism_passes'],1)
            self.assertEqual(pair['total_snapshots'],1)



if __name__=='__main__':unittest.main()
