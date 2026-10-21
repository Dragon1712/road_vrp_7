import json,sys,tempfile,csv,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_b3r import compare_states,audit,effective_action


def make_row(sid,sample,execute,raw,task='task_0008'):
    return dict(state_id=sid,task_id=task,date='2025-07-25',scenario_id='2025-07-25_0800',
        selected_candidate_id=sample,execute=str(int(execute)),score='1.5',
        chosen_static_repair_td_gain_sec=str(raw),
        realized_signed_gain_sec=str(raw if execute else 0.))


class TestAudit(unittest.TestCase):
    def test_noop_candidate_changes_do_not_count_as_action(self):
        a=[make_row('s1','a',False,-1),make_row('s2','a',True,2)]
        b=[make_row('s1','b',False,-3),make_row('s2','b',True,3)]
        rows=compare_states(a,b,'task',42)
        self.assertEqual(rows[0]['fake_change_both_noop'],1)
        self.assertEqual(rows[0]['changed_effective_action'],0)
        self.assertEqual(rows[1]['changed_effective_action'],1)
        self.assertEqual(rows[1]['delta_gain_sec'],1.)

    def test_wrong_reward_caught(self):
        a=[make_row('s1','a',False,-1)];b=[make_row('s1','b',False,-2)]
        b[0]['realized_signed_gain_sec']='11'
        with self.assertRaises(ValueError):compare_states(a,b)

    def test_real_reports_ingested(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'b3_anchor_film_seed42_v1'
            (base/'task').mkdir(parents=True)
            (base/'manifest.json').write_text(json.dumps({'stage':'step6b3-anchored-film-v1','seed':42,
                'source_sha256':{'X':'abc'},'feature_npz_sha256':'bnd'}))
            (base/'summary.json').write_text(json.dumps({'status':'COMPLETE','protocols':['task']}))
            rows0=[make_row('s1','a',False,-1),make_row('s2','a',True,2)]
            rows3=[make_row('s1','b',False,-2),make_row('s2','b',True,3)]
            for key,rows in [('M0',rows0),('M3',rows3)]:
                with (base/'task'/f'{key}_test_states.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            report={m:{'test_policy':{'states':2,'mean_signed_gain_sec':v},
                       'validation_policy':{'mean_signed_gain_sec':v},
                       'alpha_selected_on_validation':0. if m=='M0' else .1}
                    for m,v in [('M0',1.),('M3',1.5)]}
            (base/'task'/'report.json').write_text(json.dumps(report))
            result=audit(root,root/'output',[42],methods=('M0','M3'),protocols=('task',))
            item=next(x for x in result['per_seed'] if x['method']=='M3')
            self.assertEqual(item['effective_action_changed'],1)
            self.assertEqual(item['both_noop_top1_changed'],1)
            self.assertAlmostEqual(item['mean_gain_diff_sec_per_state'],.5)
            with self.assertRaises(FileExistsError):audit(root,root/'output',[42],methods=('M0','M3'),protocols=('task',))
if __name__=='__main__':unittest.main()
