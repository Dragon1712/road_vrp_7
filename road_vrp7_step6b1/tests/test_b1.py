from __future__ import annotations
import csv,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
from dataset import load,group_masks,Data,PRIMARY,ORACLE
from audit_static_repair import analyze
from train_b1 import (run,policy,policy_summary,select_threshold,ranking_metrics,
                     Predictor,method_report)


FEATURES=['original_static_cost_sec','departure_hour_local','departure_weekday_local',
 'route_index','position','size','block_demand','route_load','vehicle_capacity',
 'static_section_cost_sec','route_length','relative_position',
 'history_observed_share','history_observed_rel_cost','history_has_observation'] + \
 [f'history_coverage_tminus_{i}' for i in range(6,0,-1)]+ \
 [f'history_rel_cost_tminus_{i}' for i in range(6,0,-1)]
LABELS=['td_gain_sec','static_gain_same_td_move_sec','static_repair_static_gain_sec',
        'static_repair_td_gain_sec','td_advantage_over_static_repair_sec']


def make_data(folder):
    folder=Path(folder); a=folder/'analysis';a.mkdir(parents=True)
    rng=np.random.default_rng(123)
    xs=[];ys=[];rows=[]
    for ti in range(4):
        for dt in range(4):
            for state in range(3):
                state_id=f's{ti}-{dt}-{state}'
                for cand in range(8):
                    x=rng.normal(size=len(FEATURES))
                    x[0]=3000+10*ti+8*state
                    x[1]=8+3*(dt%2);x[2]=dt
                    x[8]=40.;x[9]=3.0+cand
                    x[12]=.3+.1*dt;x[14]=1.
                    score=(cand-4)*8+15*x[3]+8*x[5]+5*x[12]-4*ti+5*dt
                    true=float(score+10*rng.normal())
                    oracle=true+float(rng.uniform(0.,10.))
                    label=[oracle,oracle*.9,true+8,true,oracle-true]
                    xs.append(x);ys.append(label)
                    rows.append({'sample_id':f'{state_id}-{cand}','state_id':state_id,
                         'task_id':f'task_{ti:04d}','scenario_id':f'day_{dt}',
                         'date':f'2025-07-{22+dt:02d}',
                         'task_role':('train' if ti<2 else 'val' if ti==2 else 'internal_test'),
                         'date_role':('train' if dt<2 else 'val' if dt==2 else 'internal_test'),
                         'source_solver':'C3_topk','seed':'42','snapshot_sec':'5'})
    np.savez_compressed(a/'features_X.npz',X=np.asarray(xs,dtype=np.float32))
    np.savez_compressed(a/'labels_y.npz',y=np.asarray(ys,dtype=np.float32))
    with (a/'groups.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a/'dataset_schema.json').write_text(json.dumps({'feature_names':FEATURES,
      'label_names':LABELS,'shape_X':[len(xs),len(FEATURES)],
      'shape_y':[len(ys),len(LABELS)]}))
    (folder/'run_config.json').write_text(json.dumps({'stage':'step6a-offline-v2'}))
    (folder/'benchmark_summary.json').write_text(json.dumps({'status':'COMPLETE'}))
    return folder

class TestStep6B1(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src=make_data(Path(self.tmp.name)/'source')
        self.d=load(self.src)

    def test_load_and_feature_ablation(self):
        self.assertEqual(self.d.X.shape,(384,27))
        self.assertEqual(len(self.d.features('core')[1]),10)
        self.assertEqual(len(self.d.features('static')[1]),12)
        self.assertEqual(len(self.d.features('traffic')[1]),27)
        self.assertEqual(int((self.d.label(PRIMARY)>0).sum())>0,True)

    def test_group_separation(self):
        for typ in ['task','date']:
            roles=group_masks(self.d,typ)
            self.assertTrue(all(mask.sum()>0 for mask in roles.values()))
            self.assertEqual(sum(mask.sum() for mask in roles.values()),self.d.n)
        self.d.groups[0]['task_role']='internal_test'
        with self.assertRaisesRegex(ValueError,'Leaked'):group_masks(self.d,'task')

    def test_label_leakage(self):
        s=json.loads((self.src/'analysis/dataset_schema.json').read_text())
        s['feature_names'][0]='td_gain_sec'
        (self.src/'analysis/dataset_schema.json').write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError,'leakage'):load(self.src)

    def test_oracle_audit(self):
        r=analyze(self.d)
        self.assertGreaterEqual(r['oracle_td_repair_positive_candidates'],r['static_repair_td_positive_candidates'])
        self.assertGreaterEqual(r['mean_best_repair_gap_sec'],-1e-4)

    def test_exact_policy_noop_and_negative(self):
        y=np.array([-20.,12.,-2.,-10.]);o=y+3
        rows=[{'state_id':s,'task_id':'x','date':'2025-07-22','scenario_id':'sc',
             'sample_id':f'id-{i}'} for i,s in enumerate(('s1','s1','s2','s2'))]
        score=np.array([0.,1.,2.,0.])
        no=policy(y,o,score,score,rows,None)
        self.assertEqual(policy_summary(no)['mean_signed_gain_sec'],0)
        yes=policy(y,o,score,np.ones(4),rows,0.)
        self.assertAlmostEqual(policy_summary(yes)['total_signed_gain_sec'],10.)
        self.assertEqual(policy_summary(yes)['nonpositive_executions'],1)
        self.assertAlmostEqual(sum(r['static_repair_td_gain_sec'] for r in yes),10.)

    def test_validation_threshold_no_test(self):
        roles=group_masks(self.d,'task')
        X=self.d.features('static')[0]
        m=Predictor('reg',42,10).fit(X[roles['train']],self.d.label(PRIMARY)[roles['train']],
                                        [g for i,g in enumerate(self.d.groups) if roles['train'][i]])
        score,gate,_=m.predict(X[roles['val']]);vrows=[g for i,g in enumerate(self.d.groups) if roles['val'][i]]
        v=self.d.label(PRIMARY)[roles['val']];u=self.d.label(ORACLE)[roles['val']]
        before=select_threshold(v,u,score,gate,vrows,'reg')[0]
        # Changing *test* labels cannot change validation threshold.
        self.d.Y[roles['internal_test'],self.d.labels.index(PRIMARY)] += 9999
        after=select_threshold(v,u,score,gate,vrows,'reg')[0]
        self.assertEqual(before,after)

    def test_model_fit_pairwise_two_part(self):
        roles=group_masks(self.d,'date')
        X=self.d.features('static')[0][roles['train']]
        y=self.d.label(PRIMARY)[roles['train']]
        groups=[g for i,g in enumerate(self.d.groups) if roles['train'][i]]
        for kind in ['cls','reg','pair','two']:
            model=Predictor(kind,42,10).fit(X,y,groups)
            rank,gate,p=model.predict(X[:10])
            self.assertTrue(np.isfinite(rank).all())
            self.assertTrue(np.isfinite(gate).all())

    def test_smoke_end_to_end(self):
        o=Path(self.tmp.name)/'output'
        methods=['P1_class_static_time','P2_reg_static_time','P3_pairwise_static_time',
                 'P4_two_part_static_time','P5_reg_history']
        reports=run(self.src,o,('task','date'),seed=42,trees=10,methods=methods)
        self.assertEqual(len(reports),2)
        for p in ('task','date'):
            r=json.loads((o/p/'report.json').read_text())
            self.assertEqual(set(r['methods']),set(methods))
            for m in methods:
                with (o/p/f'{m}_test_states.csv').open(newline='') as f:
                    self.assertGreater(len(list(csv.DictReader(f))),0)
                with (o/p/f'{m}_test_candidates.csv').open(newline='') as f:
                    self.assertGreater(len(list(csv.DictReader(f))),0)
        self.assertTrue((o/'summary.json').is_file())
        with self.assertRaises(FileExistsError):run(self.src,o,('task',),seed=42,trees=10,methods=methods)

if __name__=='__main__':unittest.main()
