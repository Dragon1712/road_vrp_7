import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from select_states import choose,load_metadata,create
from analyze_values import analyze,clustered_interval

class SV2Tests(unittest.TestCase):
    def test_outcome_blind_repeatable_and_unique(self):
        g={f'id_{i}':{'task_id':f'task_{i%2}','scenario_id':f'sc_{i%3}'} for i in range(60)}
        a=choose(g,24,201);self.assertEqual(a,choose(g,24,201))
        self.assertEqual(len(set(a)),24);self.assertNotEqual(a,choose(g,24,202))
    def test_balanced_cohort(self):
        g={f'id_{i}':{'task_id':f'task_{i%2}','scenario_id':'sc0'} for i in range(50)}
        a=choose(g,14,7,'balanced');self.assertEqual(len(a),14)
        self.assertEqual(sum(g[k]['task_id']=='task_0' for k in a),7)
    def test_bad_n(self):
        with self.assertRaises(ValueError):choose({'a':{}},2,1)
    def test_interval_determinism(self):
        self.assertEqual(clustered_interval([1,2,3],5,100),clustered_interval([1,2,3],5,100))
    def test_output_selection_copy_and_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'src';src.mkdir();tgt=Path(td)/'tgt'
            (src/'manifest.json').write_text(json.dumps({'stage':'R3-SV-one-step-v1'}))
            (src/'report.json').write_text(json.dumps({'status':'COMPLETE'}))
            (src/'pilot_states.json').write_text(json.dumps({'selected_decision_ids':['old']}))
            rows=[]
            for i in range(6):
                for a in ('C3','M0_PROXY','R2_F2','T2_HUBER'):
                    rows.append({'decision_id':f'i{i}','role':'internal_test','task_id':f'task_{i%2}',
                      'scenario_id':'sc1','date':'2025-01-01','seed':42,'arm':a,'gain_sec':10*i})
            (src/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            report=create(src,tgt,3,10)
            self.assertEqual(report['n_selected'],3)
            self.assertEqual(len(json.loads((tgt/'pilot_states.json').read_text())['selected_decision_ids']),3)
            with self.assertRaises(FileExistsError):create(src,tgt,3,10)
    def test_rollout_state_means_and_same_action_placebo(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'step';roll=Path(td)/'roll';src.mkdir();roll.mkdir()
            ids=['s1','s2'];plans={'selected_decision_ids':ids,'sampling':'outcome_blind_uniform','role':'internal_test'}
            (src/'pilot_states.json').write_text(json.dumps(plans))
            (src/'decisions.jsonl').write_text(''.join(json.dumps({'decision_id':di,'arm':arm,
              'gain_sec':0. if arm in ('C3','R2_F2') else 10.,'selected_index':-1 if arm in ('C3','R2_F2') else 0,
              'task_id':f'task_{di}','scenario_id':'sc0'})+'\n'
                    for di in ids for arm in ('C3','M0_PROXY','R2_F2','T2_HUBER')))
            from analyze_values import sha
            (roll/'manifest.json').write_text(json.dumps({'stage':'R3-SV-paired-restart-v1',
              'one_step_sha256':sha(src/'decisions.jsonl'),'pilot_sha256':sha(src/'pilot_states.json'),
              'states':ids,'horizons':[5,20],'replicates':2,'road_replay':True}))
            (roll/'report.json').write_text(json.dumps({'status':'COMPLETE'}))
            rows=[]
            for di in ids:
                for h in (5,20):
                    for rep in range(2):
                        for a in ('C3','M0_PROXY','R2_F2','T2_HUBER'):
                            base=a in ('C3','R2_F2')
                            rows.append({'decision_id':di,'horizon':h,'replicate':rep,'arm':a,
                              'rollout_seed':rep+1,'initial_td_od_cost':100. if base else 90.,
                              'final_td_od_cost':100. if base else 80.+rep,
                              'final_ex_post_road_travel_sec':200. if base else 175.+rep,
                              'action_identity':[0,0,2] if base else [1,0,2]})
            (roll/'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            report,labels=analyze(roll,src,100,42)
            self.assertEqual(report['n_runs'],32)
            self.assertEqual(report['identical_action_placebo']['violations'],0)
            self.assertAlmostEqual(report['metrics']['td_od']['5']['T2_HUBER']['mean_h_gain_sec'],19.5)
            self.assertEqual(report['metrics']['td_od']['5']['R2_F2']['mean_h_gain_sec'],0)
            self.assertAlmostEqual(report['metrics']['road_replay']['20']['M0_PROXY']['mean_h_gain_sec'],24.5)
            self.assertEqual(len(labels),2*3*2*2)
    def test_incomplete_pairs_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'src';roll=Path(td)/'roll';src.mkdir();roll.mkdir()
            (src/'pilot_states.json').write_text(json.dumps({'selected_decision_ids':['a']}))
            (src/'decisions.jsonl').write_text(json.dumps({'decision_id':'a','arm':'C3'})+'\n')
            from analyze_values import sha
            (roll/'manifest.json').write_text(json.dumps({'stage':'R3-SV-paired-restart-v1',
                 'one_step_sha256':sha(src/'decisions.jsonl'),'pilot_sha256':sha(src/'pilot_states.json'),
                 'states':['a'],'horizons':[5],'replicates':1}))
            (roll/'report.json').write_text(json.dumps({'status':'COMPLETE'}))
            (roll/'rollouts.jsonl').write_text('')
            with self.assertRaises(ValueError):analyze(roll,src)

if __name__=='__main__':unittest.main()
