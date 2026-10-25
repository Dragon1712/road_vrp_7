"""Subprocess regression checks for previously leaked generic TASK_IDS et al."""
import os
import subprocess
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'road_vrp7_step6c1_r4g'/'run_r4g.sh'
GENERIC={'TASK_IDS':'task_0008,task_0009',
         'SCENARIO_IDS':'2025-07-22_0800',
         'SEEDS':'60,61,62',
         'MAX_ITERATIONS':'20000',
         'SNAPSHOTS':'4000,8000,12000'}

class TestShellIsolation(unittest.TestCase):
    def cmd(self,extra=None):
        env=os.environ.copy()
        for k in list(env):
            if k.startswith('R4G_') or k in GENERIC:
                env.pop(k)
        env.update(GENERIC)
        env.update(extra or {})
        return subprocess.run(['bash',str(SCRIPT),'show-config'],
                              cwd=ROOT,env=env,text=True,capture_output=True)

    def test_inherited_validation_defaults_to_train_smoke(self):
        r=self.cmd()
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('TASK_IDS=task_0000\n',r.stdout)
        self.assertIn('SCENARIO_IDS=2025-07-02_0800\n',r.stdout)
        self.assertIn('SEEDS=60\n',r.stdout)
        self.assertIn('MAX_ITERATIONS=2000\n',r.stdout)
        self.assertIn('Ignoring inherited generic TASK_IDS',r.stderr)

    def test_pilot_only_train_even_with_generic_leak(self):
        r=self.cmd({'R4G_VIEW_MODE':'pilot'})
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('TASK_IDS=task_0000,task_0001,task_0002,task_0003\n',r.stdout)
        self.assertIn('SCENARIO_IDS=2025-07-02_0800,2025-07-09_1700\n',r.stdout)
        self.assertIn('MAX_ITERATIONS=20000\n',r.stdout)

    def test_scoped_overrides_work(self):
        r=self.cmd({'R4G_TASK_IDS':'task_0002','R4G_SCENARIO_IDS':'2025-07-09_1700',
                    'R4G_SEEDS':'62,63','R4G_MAX_ITERATIONS':'1500'})
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('TASK_IDS=task_0002\n',r.stdout)
        self.assertIn('SCENARIO_IDS=2025-07-09_1700\n',r.stdout)
        self.assertIn('SEEDS=62,63\n',r.stdout)
        self.assertIn('MAX_ITERATIONS=1500\n',r.stdout)

    def test_scoped_validation_override_fails_closed(self):
        r=self.cmd({'R4G_TASK_IDS':'task_0008'})
        self.assertNotEqual(r.returncode,0)
        self.assertIn('Non-TRAIN/unauthorized',r.stderr)

if __name__=='__main__': unittest.main()
