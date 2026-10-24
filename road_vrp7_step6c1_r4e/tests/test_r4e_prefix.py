import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r4e_core import read,write,sha
from collect_long import verify_prior_k50
from r4c_core import branch_spec,unit_name,STAGE

class TestR4EPrefix(unittest.TestCase):
    def test_r4c_50_step_prefix_positive_and_negative(self):
        with tempfile.TemporaryDirectory() as t:
            source=Path(t);(source/'units').mkdir()
            s={'task_id':'task_0008','scenario_id':'s','seed':60,'stop_count':505}
            spec=branch_spec(s,50,'A002')
            old={'stage':STAGE,'unit_spec':spec,
                 'result':{'arm':'A002','events':[{'i':i} for i in range(50)],'first_action_id':[0,1,3]}}
            write(source/'units'/unit_name(spec),old)
            current={'arm':'A002','events':[{'i':i} for i in range(100)],'first_action_id':[0,1,3]}
            self.assertTrue(verify_prior_k50(source,s,'A002',current))
            current['events'][48]['i']=999
            with self.assertRaisesRegex(ValueError,'changed original R4C'):verify_prior_k50(source,s,'A002',current)

    def test_prior_prefix_requires_real_unit(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'missing'):
                verify_prior_k50(t,{'task_id':'t','scenario_id':'s','seed':1,'stop_count':1},'A00',
                                  {'events':[], 'first_action_id':[0,0,2]})

if __name__=='__main__':unittest.main()
