from __future__ import annotations
import math,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4o'))
import r4o_core as m
import probe_r4o as cli


def snap(task='task_0000',seed=60,stop=100,sc='2025-07-02_0800'):
    routes=[[11,12,13,14,15],[21,22,23,24],[31,32,33,34]]
    return {'task_id':task,'scenario_id':sc,'seed':seed,'stop_count':stop,
       'current_routes':routes,'best_routes':[[11,13,12,14,15],[21,23,22,24],[31,32,33,34]],
       'current_surrogate_cost':4400.,'best_surrogate_cost':4200.,
       'wall_remaining_before_select_sec':24.,'anneal_temperature_before_accept':200.,'anneal_start_temperature':300.}


def record(s,arm,act,y):
    k=m.sid(s)
    # Original 15 feature dimensions, no future values
    f=[1.,k[-1]/20000.,1.,.8,.1,.7,.3,.3,float(act[0])/2.,float(act[1])/4.,float(act[2]==3),0.,.2,.5,1.]
    return {'snap':k,'arm':arm,'task':k[0],'source':'original_s60s61' if k[2] in (60,61) else 'new_s62s63',
       'physical':(k[0],k[2],k[3]),'features':f,'targets':{'k50':y,'k200':y,'k500':y,'robust':y},
       'feature_sets':{z:m.augment(f,s,act,z) for z in m.FEATURE_MODES},'gate_features':m.state_gate_features(s)}


def make(n_tasks=4):
    rec=[]
    for t in range(n_tasks):
        for seed in (60,61,62,63):
            for step in (101,510):
                s=snap(f'task_{t:04d}',seed,step)
                for i,a in enumerate([(0,1,2),(0,2,2),(1,1,3),(2,0,2)]):
                    gain=([10.,-4.,0.,-5.] if t%2==0 else [-2.,5.,0.,-7.])[i]
                    if seed>=62:gain+=.5
                    rec.append(record(s,f'A{i:03d}',a,gain))
    return rec


class TestR4OStructure(unittest.TestCase):
    def test_all_feature_sets_finite(self):
        s=snap()
        for a in [(0,0,2),(0,3,2),(1,1,3),(2,2,2)]:
            for mode,size in [('baseline15',15),('route_context',38),('route_context_interact',50)]:
                vals=m.augment([1.]*15,s,a,mode)
                self.assertEqual(len(vals),size)
                self.assertTrue(all(math.isfinite(v) for v in vals))
    def test_customer_id_magnitude_is_not_a_feature(self):
        s=snap();u=m.route_context(s,(0,1,2))
        other={**s,'current_routes':[[q+1000 for q in r] for r in s['current_routes']],
               'best_routes':[[q+1000 for q in r] for r in s['best_routes']]}
        self.assertEqual(u,m.route_context(other,(0,1,2)))
    def test_reject_illegal_section(self):
        with self.assertRaisesRegex(ValueError,'Illegal Section'):
            m.route_context(snap(),(0,5,2))
    def test_no_future_rewards_enter_context(self):
        s=snap();a=(1,0,2);one=m.augment([.1]*15,s,a,'route_context')
        different={**s,'current_surrogate_cost':4600.} # structural part is unaffected
        self.assertEqual(one,m.augment([.1]*15,different,a,'route_context'))
    def test_join_rejects_missing_arm(self):
        s=snap();recs=[record(s,'A001',(0,1,2),10.)]
        with self.assertRaisesRegex(ValueError,'Missing candidate action'):
            m.attach_context(recs,[s],[],{}, {})
    def test_join_rejects_duplicated_snapshots(self):
        s=snap();recs=[record(s,'A001',(0,1,2),10.)]
        with self.assertRaisesRegex(ValueError,'Duplicate source'):
            m.attach_context(recs,[s,s],[],{(m.sid(s),'A001'):[0,1,2]}, {})
    def test_grouped_pairwise_fit(self):
        rec=make()
        for mode in m.FEATURE_MODES:
            a=m.fit_pairwise(rec,mode)
            self.assertTrue(a['fitted'])
            self.assertGreater(a['pairs'],0)
            self.assertTrue(all(math.isfinite(v) for v in a['weights']))
    def test_gate_fit(self):
        rec=make()
        for r in rec:
            if r['snap'][3]==510:r['targets']['k500']=-1.
        x=m.fit_gate(rec)
        self.assertTrue(x['fitted'])
        prob=m.gate_prob(x,rec[0])
        self.assertTrue(0<=prob<=1)
    def test_no_positive_gate_fails_closed(self):
        rec=make()
        for r in rec:r['targets']={k:-1. for k in r['targets']}
        self.assertFalse(m.fit_gate(rec)['fitted'])
    def test_oracle_policy_declared(self):
        self.assertTrue(set(m.ORACLE_POLICIES)<=set(m.POLICIES))
        rec=make();gate=m.fit_gate(rec);rank=m.fit_pairwise(rec,'baseline15')
        ridge=[0.]*15
        pred=m.predict_decisions(rec[:16],rank,gate,ridge,'baseline15','sampled_oracle')
        self.assertTrue(all(p['gain_k500_sec']>=0 for p in pred))
        self.assertTrue(all(p['oracle_conditioned'] for p in pred))
    def test_task_holdout_and_seed_holdout_run(self):
        import r4l_core
        rec=make()
        fs,rows=m.evaluate(rec,r4l_core.make_splits(rec))
        self.assertEqual(len(fs),6*3*2*len(m.POLICIES))
        self.assertEqual(fs[0]['policy'],'C3')
        self.assertTrue(any(z['policy']=='learned_gate_pairwise' for z in fs))
    def test_train_only_shuffling(self):
        rec=make()
        z=m.shuffle_labels(rec,123)
        self.assertEqual(len(z),len(rec))
        self.assertEqual(sorted(r['targets']['k500'] for r in rec),sorted(r['targets']['k500'] for r in z))
        self.assertEqual([r['features'] for r in rec],[r['features'] for r in z])
    def test_explicit_leakage_rejected(self):
        rec=make();a=[r for r in rec if r['task']=='task_0000'];b=[r for r in rec if r['task']=='task_0000']
        with self.assertRaisesRegex(ValueError,'Snapshot split leakage'):
            m.evaluate(rec,[('bad',a,b,'TASK_LOTO')])
    def test_render_belongs_to_train(self):
        d={'n_scenario_snapshots':2,'n_actions':8,'n_coarse_search_starts':2,'folds':[]}
        self.assertIn('Train-only',cli.render(d))


if __name__=='__main__':unittest.main()
