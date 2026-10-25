import json,sys,tempfile,unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4n'))
import r4n_core as mod
import probe_r4n


def synthetic():
    records=[]
    n_snapshots=0
    for tid in range(4):
        for seed in (60,61,62,63):
            for step in range(7 if tid==0 else 6):
                stop=100+step*340
                for sid in ('2025-07-02_0800','2025-07-09_1700'):
                    snap=(f'task_{tid:04d}',sid,seed,stop)
                    physical=(snap[0],seed,stop,'route','best','rng','dw','rw',float(stop)+(.00001 if sid.endswith('1700') else 0))
                    n_arms=8 if n_snapshots<172 else 7
                    n_snapshots+=1
                    for arm in [f'A{k:03d}' for k in range(n_arms)]:
                        y=(30. if arm=='A000' and tid in (0,2) and stop<800 else
                           -45. if arm=='A001' and tid in (1,3) and stop<900 else 0.)
                        f=[1.,stop/20000.,float(stop<=2000),.6,0.2,.15,.3,.5,tid/3.,
                           float(arm=='A001'),float(arm=='A002'),0.,0.25,.30,0.]
                        records.append({'snap':snap,'arm':arm,'task':snap[0],'physical':physical,
                          'features':f,'targets':{'k50':0.,'k200':y/2,'k500':y,'robust':min(y/2,y)},
                          'source':'old' if seed<=61 else 'new'})
    return records


class TestR4N(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=synthetic()

    def test_splits_and_coarse_state_correlation(self):
        folds=mod.r4l.make_splits(self.data)
        self.assertEqual(len(folds),6)
        self.assertEqual(len(mod.group(self.data)),200)
        self.assertEqual(len({tuple(z['physical'][:-1]) for z in self.data}),100)
        for name,tr,held,typ in folds:
            self.assertFalse({x['snap'] for x in tr}&{x['snap'] for x in held})
            if typ=='TASK_LOTO':self.assertFalse({r['task'] for r in tr}&{r['task'] for r in held})
            else:self.assertFalse({r['snap'][2] for r in tr}&{r['snap'][2] for r in held})

    def test_pairwise_predictor_and_gate_are_deterministic(self):
        train=[z for z in self.data if z['task']!='task_0003']
        m1=mod.fit_rank(train);m2=mod.fit_rank(train)
        self.assertEqual(m1,m2)
        self.assertTrue(m1['fitted']);self.assertGreater(m1['pair_count'],0)
        w=mod.r4i.ridge_predictors(train,target='k500',strength=100.)
        held=[z for z in self.data if z['task']=='task_0003']
        for policy in mod.POLICIES:
            res=mod.choose(held,m1,w,policy)
            self.assertEqual(len(res),48)
            self.assertTrue(all(x['oracle_uses_future_labels']==(policy in mod.ORACLE_POLICIES) for x in res))
            self.assertTrue(all(x['selected_arm']=='C3' for x in res) if policy=='C3' else True)

    def test_no_label_leakage_into_pairwise_features(self):
        tr=[z for z in self.data if z['task']!='task_0003']
        te=[z for z in self.data if z['task']=='task_0003']
        model=mod.fit_rank(tr)
        altered=[dict(z, targets={'k500':1e9}) for z in te]
        self.assertEqual(model,mod.fit_rank(tr))
        self.assertEqual([mod.rank_score(model,z) for z in te],
                         [mod.rank_score(model,z) for z in altered])

    def test_shuffle_null_preserves_task_marginals_and_is_deterministic(self):
        a=mod.shuffle_train_targets(self.data,42)
        b=mod.shuffle_train_targets(self.data,42)
        self.assertEqual(a,b)
        for tid in range(4):
            key=f'task_{tid:04d}'
            self.assertEqual(Counter(z['targets']['k500'] for z in self.data if z['task']==key),
                             Counter(z['targets']['k500'] for z in a if z['task']==key))
        self.assertEqual(self.data[0]['targets']['k500'],30.)

    def test_no_informative_pairs_fails_closed(self):
        xx=[dict(x,targets={**x['targets'],'k500':0.}) for x in self.data]
        model=mod.fit_rank(xx)
        self.assertFalse(model['fitted'])
        w=[0.]*len(mod.r4i.FEATURES)
        for policy in ('pairwise_forced_early','pairwise_with_ridge_gate','oracle_gate_pairwise_rank'):
            rows=mod.choose(xx,model,w,policy)
            self.assertEqual(sum(r['selected_arm']!='C3' for r in rows),0)

    def test_oracle_upper_bound_and_score_consistency(self):
        tr=[z for z in self.data if z['task']!='task_0003'];held=[z for z in self.data if z['task']=='task_0003']
        model=mod.fit_rank(tr);ridge=mod.r4i.ridge_predictors(tr,target='k500')
        z=mod.choose(held,model,ridge,'sampled_oracle_early')
        stat=mod.score_summary(z)
        self.assertTrue(all(x['gain_k500_sec']>=-mod.EPS for x in z))
        self.assertAlmostEqual(stat['mean_gain_sec'],stat['sampled_oracle_mean_sec'])
        self.assertEqual(stat['positive']+stat['negative']+stat['zero'],stat['n_snapshots'])

    def test_full_synthetic_evaluation(self):
        folds,preds=mod.evaluate(self.data)
        self.assertEqual(len(folds),6*2*2*len(mod.POLICIES))
        self.assertEqual(len(preds),sum(x['n_snapshots'] for x in folds))
        self.assertEqual(len({z['split'] for z in folds}),6)
        for x in folds:
            self.assertEqual(x['n_snapshots'],x['positive']+x['negative']+x['zero'])
            self.assertLessEqual(x['delegations'],x['n_snapshots'])
            self.assertTrue(x['sampled_oracle_mean_sec']>=-mod.EPS)
        self.assertIn('Oracle',mod.markdown({'data_summary':{'actions':len(self.data),'snapshots':200,'coarse_starts':100},'folds':folds}))

    def test_runner_fail_closed_source_and_output_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'r4m';src.mkdir();out=root/'output'
            args=type('A',(),{})();args.mode='probe';args.output_dir=str(out);args.r4m_result=str(src)
            for k in ('r4l_result','old_cf','old_capture','old_prefix','new_r4k','new_cf','new_capture','new_prefix'):
                setattr(args,k,str(root/k))
            pins={'r4l_report_sha256':'old'}
            mod.save(src/'report.json',{'stage':mod.r4m.STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
                         'n_configs':120,'reference_report_sha256':'old'})
            (src/'fold_predictions.jsonl').write_text('a\n')
            hashes={k:mod.sha(mod.ROOT/'road_vrp7_step6c1_r4m'/k) for k in ('r4m_core.py','probe_r4m.py')}
            mod.save(src/'provenance.json',{'stage':mod.r4m.STAGE,'report_sha256':mod.sha(src/'report.json'),
                       'predictions_sha256':mod.sha(src/'fold_predictions.jsonl'),
                       'source_sha256':pins,'code_sha256':hashes})
            with patch.object(probe_r4n.c.r4m,'source_preflight',return_value=(self.data,pins)):
                probe_r4n.run(args)
                self.assertEqual(mod.read(out/'report.json')['status'],'COMPLETE')
                self.assertTrue((out/'fold_predictions.jsonl').is_file())
                with self.assertRaisesRegex(ValueError,'independent'):probe_r4n.run(args)
                (src/'report.json').write_text('{"tampered":true}')
                args.output_dir=str(root/'other')
                with self.assertRaisesRegex(ValueError,'R4-M report'):probe_r4n.run(args)

if __name__=='__main__':unittest.main()
