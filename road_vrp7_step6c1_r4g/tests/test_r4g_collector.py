"""Collector entrypoint regression: TRAIN smoke must hash R4-A's real core path.

Runs both collector arms with stubbed road/solver modules, so no installed
ALNS package or Chicago files are required. This catches errors BEFORE
out.mkdir and checks that actual manifest/provenance files are produced.
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'road_vrp7_step6c1_r4g'))
import collect_train_r4g as collector


class FakeRoad:
    edge_file_hash = 'edge-sha'
    cost_file_hash = 'cost-sha'

    def __init__(self, *args, **kwargs):
        pass

    def replay(self, task, routes, departure):
        return {'ex_post_road_travel_sec': 123.5}


class FakeOD:
    def __init__(self, *args, **kwargs):
        pass


class FakeObjective:
    def __init__(self, *args, **kwargs):
        pass


class FakeObserver:
    def __init__(self):
        self.snapshots = [{'boundary': 'BEFORE_SELECT_AND_DESTROY', 'stop_count': 300}]

    def summary(self):
        return {'selector_calls': 2000, 'selector_updates': 2000, 'snapshot_count': 1}


@contextmanager
def instrument(*args, **kwargs):
    yield [FakeObserver()]


def solve(*args, **kwargs):
    return {'routes': [[1], [2], [3], [4], [5]], 'iterations': 2000,
            'search_wall_sec': 1.2, 'termination_reason': 'iteration_budget',
            'planned_cost': 100.0}


class TestR4GCollectorSmoke(unittest.TestCase):
    def run_arm(self, arm):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            calls = []
            source = {'source_cost_hash': 'cost-sha', 'source_edge_hash': 'edge-sha',
                      'od_step_min': 20, 'od_hours': 5.0,
                      'code_sha256': {
                          'road_vrp7_step5b_v3/section_v2.py': 'pinned-section',
                          'road_vrp7_stage3/road_vrp.py': 'pinned-road'}}
            scenario = {'scenario_id': '2025-07-02_0800', 'split': 'train',
                        'eligible': True, 'departure_local': '2025-07-02 08:00'}
            task = {'task_id': 'task_0000', 'scenarios': [scenario]}

            def fake_sha(path):
                p = Path(path)
                calls.append(p)
                if p == base / 'road_vrp7_step6c1_r4a' / 'r4a_core.py':
                    return 'correct-r4a-core-sha'
                if p.name == 'section_v2.py':
                    return 'pinned-section'
                if p.name == 'road_vrp.py':
                    return 'pinned-road'
                if p.is_file():
                    from r4a_core import sha256
                    return sha256(p)
                return 'mock-source-sha'

            road_mod = types.ModuleType('road_vrp')
            road_mod.Road, road_mod.OD = FakeRoad, FakeOD
            road_mod.Objective = FakeObjective
            road_mod.legal = lambda *a: None
            road_mod.read_json = lambda *a: source
            benchmark_mod = types.ModuleType('benchmark')
            benchmark_mod.load_tasks = lambda *a: ([], [task], {
                'cost_edge_index_sha256': 'edge-sha',
                'cost_edge_npz_sha256': 'cost-sha'})
            solver_mod = types.ModuleType('run_experiment')
            solver_mod.solve = solve
            args = SimpleNamespace(
                arm=arm, output_dir=str(base / 'output'),
                model_source=str(base / 'models'), cost_dir=str(base / 'cost'),
                task_dir=str(base / 'tasks'), od_cache_dir=str(base / 'cache'),
                task_ids='task_0000', scenario_ids='2025-07-02_0800', seeds='60',
                wall_sec=30, max_iterations=2000, anneal_iterations=5000,
                top_k=12, snapshots='250,700,1200,1700',
                max_snapshots=16, section_only=True)
            with patch.object(collector, 'ROOT', base), \
                 patch.object(collector, 'sha256', side_effect=fake_sha), \
                 patch.object(collector, 'instrument_official_alns', instrument), \
                 patch('importlib.metadata.version', return_value='7.0.0'), \
                 patch.dict(sys.modules, {'road_vrp': road_mod,
                                          'benchmark': benchmark_mod,
                                          'run_experiment': solver_mod}):
                collector.main(args)
            self.assertIn(base / 'road_vrp7_step6c1_r4a' / 'r4a_core.py', calls)
            self.assertNotIn(base / 'road_vrp7_step6c1_r4g' / 'r4a_core.py', calls)
            manifest = json.loads((base / 'output' / 'manifest.json').read_text())
            self.assertEqual(manifest['core_sha256'], 'correct-r4a-core-sha')
            self.assertEqual(manifest['split'], 'train')
            self.assertEqual(manifest['arm'], arm)
            self.assertEqual(json.loads((base / 'output' / 'report.json').read_text())['status'], 'COMPLETE')
            self.assertTrue((base / 'output' / 'provenance.json').exists())

    def test_plain_smoke_manifest_uses_real_r4a_path(self):
        self.run_arm('plain')

    def test_capture_smoke_manifest_uses_real_r4a_path(self):
        self.run_arm('capture')


if __name__ == '__main__':
    unittest.main()
