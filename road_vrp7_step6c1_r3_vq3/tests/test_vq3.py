"""Offline synthetic and leakage tests. Never claim Chicago online improvements."""
from __future__ import annotations
from argparse import Namespace
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT/'road_vrp7_step6c1_r3_vq3',ROOT/'road_vrp7_step6c1_r3_vq2/tests',
             ROOT/'road_vrp7_step6c1_r3_vq2',ROOT/'road_vrp7_step6c1_r3_vq',ROOT/'road_vrp7_step6c1_r2'):
    sys.path.insert(0,str(path))
from vq3_core import (POLICIES,STAGE,candidate_baseline_feature,train_pairrank,
     HarmPredictor,crossfit_calibrated_harm,prob_diagnostics,risk_gate,coverage_matched_tau,
     pairwise_accuracy)
import train_vq3

class PureTests(unittest.TestCase):
    def test_real_c3_anchor_is_not_zero(self):
        x=np.arange(78,dtype=float)
        b=candidate_baseline_feature(x)
        self.assertEqual(b.shape,(78,))
        np.testing.assert_array_equal(b[:26],x[26:52]);np.testing.assert_array_equal(b[26:52],x[26:52])
        np.testing.assert_array_equal(b[52:],0.)

    def test_ranker_anchors_c3_and_separates_linear_label(self):
        rng=np.random.default_rng(1);rows=[]
        for st in range(12):
            b=rng.normal(size=26)
            for a in range(3):
                d=rng.normal(size=26);y=20.*d[0]-7.*d[1]+0.5*a
                x=np.concatenate((b+d,b,d))
                rows.append({'task':'t','decision':f's{st}','action':a,'x':x,'mean':y})
        for weighted in (False,True):
            clf=train_pairrank(rows,weighted=weighted,seed=42)
            self.assertEqual(clf.training_states,12)
            self.assertTrue(clf.training_pairs>30)
            x=rows[0]['x'];baseline=candidate_baseline_feature(x)
            self.assertAlmostEqual(float(clf.predict(np.stack([baseline]))[0]),0.,places=10)

    def test_risk_oof_disjoint_calibration(self):
        rng=np.random.default_rng(10);rr=[]
        for task in ['t1','t2','t3']:
            for i in range(24):
                x=rng.normal(size=78)
                rr.append({'task':task,'decision':f'{task}_{i}','action':0,'x':x,
                           'mean':-70. if i%4==0 else 25.,'weight':1.})
        model,oof,diag=crossfit_calibrated_harm(rr,{'t1','t2','t3'},seed=42,trees=20)
        self.assertEqual(len(oof),len(rr))
        self.assertEqual(diag['n_harm_candidates'],18)
        for fold in diag['calibration_folds']:
            self.assertNotIn(fold['oof_task'],fold['fit_tasks'])
        probabilities=model.predict(np.stack([z['x'] for z in rr[:3]]))
        self.assertTrue(((probabilities>=0)&(probabilities<=1)).all())
        self.assertTrue(np.isfinite(diag['brier']))

    def test_harm_classifier_constant_class(self):
        x=np.zeros(78);rr=[{'x':x,'mean':4,'weight':1.} for _ in range(15)]
        p=HarmPredictor(20,1).fit(rr).predict(np.stack([x]))
        self.assertLess(p[0],1e-4)

    def test_risk_gate_and_score_matching(self):
        self.assertEqual(risk_gate([4,5],[1,1],[.1,.9],.2),-1)
        self.assertEqual(risk_gate([4,5],[1,1],[.1,.9],.95),1)
        self.assertEqual(risk_gate([-4,-5],[1,1],[.1,.1],.2),-1)
        self.assertEqual(risk_gate([4,5],[1,0],[.1,.9],.2),0)
        tau=coverage_matched_tau([1,2,3,4],.5)
        self.assertAlmostEqual(np.mean(np.array([1,2,3,4])>tau),.5)
        self.assertEqual(coverage_matched_tau([-5,-3],0),0.)

    def test_prob_metrics_and_pairwise(self):
        z=prob_diagnostics([0,0,1,1],[.1,.2,.8,.9])
        self.assertGreater(z['roc_auc'],.99);self.assertLess(z['brier'],.1)
        self.assertAlmostEqual(pairwise_accuracy([([1,2],[10,20],[True,True])]),1.)
        self.assertAlmostEqual(pairwise_accuracy([([2,1],[10,20],[True,True])]),2/3)

class AggregationTests(unittest.TestCase):
    def test_aggregate_rejects_mismatched_source_and_seeds(self):
        from aggregate_vq3 import aggregate
        from vq2_core import write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dirs=[]
            for seed in (42,43):
                p=root/str(seed);p.mkdir();dirs.append(p)
                metric={k:0.1 for k in ('mean_gain_sec','mean_regret_sec','override_rate','top1_hit',
                                          'harm_gt40_per_state','harm_gt40_per_override')}
                write_json(p/'report.json',{'stage':STAGE,'status':'COMPLETE','seed':seed,'trees':160,
                   'n_selected_states':48,'horizons':[20],'policies':list(POLICIES),
                   'macro_task_metrics':{'20':{name:metric for name in POLICIES}}})
                write_json(p/'provenance.json',{'collection_manifest_sha256':'a','collection_rollouts_sha256':'b',
                   'dataset_paired_sha256':'c','dataset_groups_sha256':'d','code_sha256':{'vq3':'original'}})
            z=aggregate(dirs,root/'summary')
            self.assertIn('PAIR_RANK',z['20'])
            with self.assertRaises(FileExistsError):aggregate(dirs,root/'summary')
            p=dirs[1];pro=json.loads((p/'provenance.json').read_text());pro['dataset_groups_sha256']='BAD'
            write_json(p/'provenance.json',pro)
            with self.assertRaises(ValueError):aggregate(dirs,root/'summary_bad')

class SyntheticEndToEnd(unittest.TestCase):
    def test_full_task_loto_train_and_baseline_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            from test_vq2 import SyntheticPipeline
            root=Path(tmp);src,col,selected=SyntheticPipeline()._make_data(root)
            # Use 1 H to keep the synthetic integration test inexpensive.
            manifest=json.loads((col/'manifest.json').read_text());manifest['horizons']=[20]
            (col/'manifest.json').write_text(json.dumps(manifest))
            rows=[json.loads(s) for s in (col/'rollouts.jsonl').read_text().splitlines() if s.strip()]
            rows=[r for r in rows if r['horizon']==20]
            (col/'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            report=json.loads((col/'report.json').read_text());report['n_restart_action_rows']=len(rows)
            (col/'report.json').write_text(json.dumps(report))
            out=root/'vq3'
            train_vq3.run(Namespace(dataset_dir=str(src),collection_dir=str(col),output_dir=str(out),seed=42,trees=20))
            d=json.loads((out/'report.json').read_text())
            self.assertEqual(d['status'],'COMPLETE');self.assertEqual(d['stage'],STAGE)
            self.assertEqual(len((out/'fold_decisions.jsonl').read_text().splitlines()),len(selected)*len(POLICIES))
            self.assertEqual(len((out/'candidate_predictions.jsonl').read_text().splitlines()),len(selected)*3)
            for held,z in d['folds']['20'].items():
                info=z['counts'];self.assertNotIn(held,info['train_tasks'])
                self.assertEqual(info['n_matched_one_step_actions'],info['n_multi_step_actions'])
                for inner in info['risk_model_oof']['calibration_folds']:
                    self.assertNotIn(inner['oof_task'],inner['fit_tasks'])
                for inner in info['coverage_control']['folds']:
                    self.assertNotIn(inner['oof_task'],inner['fit_tasks'])
                self.assertTrue(info['exactly_matched_candidate_ids_and_weights'])
            # VQ2 reference is unchanged by the new risk/ranking variants.
            import train_vq2
            other=root/'vq2'
            train_vq2.run(Namespace(dataset_dir=str(src),collection_dir=str(col),output_dir=str(other),seed=42,trees=20))
            frozen=json.loads((other/'report.json').read_text())
            for name in ['E0_FULL','E0_MATCHED','VQ_MEAN','RESID_OOF25','RESID_OOF50']:
                a=d['macro_task_metrics']['20'][name]['mean_gain_sec']
                b=frozen['macro_task_metrics']['20'][name]['mean_gain_sec']
                self.assertAlmostEqual(a,b,places=9,msg=name)
            with self.assertRaises(FileExistsError):
                train_vq3.run(Namespace(dataset_dir=str(src),collection_dir=str(col),output_dir=str(out),seed=42,trees=20))

if __name__=='__main__':unittest.main()
