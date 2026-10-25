"""Launcher validation checks without invoking ALNS or data packages."""
import os,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'road_vrp7_step6c1_r4j'/'run_r4j.sh'

class ShellGuards(unittest.TestCase):
    def cmd(self,**overrides):
        env=os.environ.copy()
        for k in list(env):
            if k.startswith('R4J_'):env.pop(k)
        env.update(overrides)
        return subprocess.run(['bash',str(SCRIPT),'plan'],cwd=ROOT,env=env,
                              capture_output=True,text=True)
    def test_new_seeds_and_train_defaults(self):
        r=self.cmd(TASK_IDS='task_0008,task_0009',SCENARIO_IDS='2025-07-22_0800',SEEDS='60,61')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('New seeds=62,63',r.stdout)
        self.assertIn('task_0000,task_0001,task_0002,task_0003',r.stdout)
        self.assertIn('Pilot case count: 16',r.stdout)
    def test_disallow_old_seeds(self):
        r=self.cmd(R4J_SEEDS='60,61')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('NEW seeds',r.stderr)
    def test_disallow_validation_task(self):
        r=self.cmd(R4J_TASK_IDS='task_0008')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('whitelist',r.stderr)
    def test_disallow_validation_date(self):
        r=self.cmd(R4J_SCENARIOS='2025-07-22_0800')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('TRAIN July',r.stderr)
    def test_extra_seed_batch_counts(self):
        r=self.cmd(R4J_SEEDS='64,65')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('Pilot case count: 16',r.stdout)

if __name__=='__main__':unittest.main()
