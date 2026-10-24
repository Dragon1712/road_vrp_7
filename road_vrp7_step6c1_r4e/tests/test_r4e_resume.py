import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r4e_core import snapshot_id, write, read

class TestR4EResume(unittest.TestCase):
    def test_json_checkpoint_identity_roundtrip(self):
        s={'task_id':'task_0008','scenario_id':'2025-07-22_0800','seed':60,'stop_count':505}
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'unit.json'
            write(f, {'snapshots':[list(snapshot_id(s))], 'horizons':[100,200,500]})
            result=read(f)
            self.assertEqual(result['snapshots'],[list(snapshot_id(s))])
            self.assertNotEqual(result['snapshots'],[snapshot_id(s)])

if __name__=='__main__':unittest.main()
