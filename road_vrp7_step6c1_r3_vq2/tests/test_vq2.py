"""Synthetic and leakage tests. No real road replay claims are made here."""
from __future__ import annotations
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from vq2_core import (fit_extra,select,oof_residual_training,one_step_rows,
                      train_arrays,metrics_for_decisions,diagnostic_for_candidates,sha256,STAGE,METHODS)
import train_vq2

class PureTests(unittest.TestCase):
    def test_selector_c3_and_mask(self):
        self.assertEqual(select([9,99,2],[1,0,1]),0)
        self.assertEqual(select([-1,-2],[1,1]),-1)
        self.assertEqual(select([1,5],[0,0]),-1)
        with self.assertRaises(ValueError):select([1],[0,1])

    def test_match_sample_weights_and_models(self):
        x=np.random.default_rng(0).normal(size=(50,4))
        y=2*x[:,0]-x[:,2]
        model=fit_extra(x,y,seed=42,trees=20,weights=np.ones(50))
        self.assertEqual(model.predict(x[:2]).shape,(2,))

    def test_oof_excludes_own_task(self):
        rng=np.random.default_rng(0)
        one=[];mult=[]
        for t in range(3):
            for i in range(9):
                z={'task':f't{t}','decision':f'{t}_{i}','action':0,'x':rng.normal(size=6),
                    'y':float(i+t),'mean':float(i+t+5),'sd':2.,'weight':1/9}
                one.append(z);mult.append(z)
        out,info=oof_residual_training(one,mult,{'t0','t1','t2'},seed=42,trees=20)
        self.assertEqual(len(out),27)
        for z in info:self.assertNotIn(z['oof_task'],z['fit_tasks'])
        self.assertTrue(all(np.isfinite(z['residual']) for z in out))

    def test_metrics_harms(self):
        r=metrics_for_decisions([{'selected_index':-1,'realized_mean_road_gain':0.,'regret':20.},
             {'selected_index':1,'realized_mean_road_gain':-100.,'regret':130.}])
        self.assertEqual(r['overrides'],1)
        self.assertAlmostEqual(r['harm_gt40_per_override'],1.)
        self.assertAlmostEqual(r['mean_gain_sec'],-50.)

    def test_train_rejects_small(self):
        with self.assertRaises(ValueError):fit_extra([[1,2]],[1],seed=42)

class SyntheticPipeline(unittest.TestCase):
    def _make_data(self,base):
        src=base/'r2';col=base/'collection';src.mkdir();col.mkdir()
        rng=np.random.default_rng(123)
        groups=[]
        for task in range(8):
            n=4 if task<4 else 2
            for state in range(n):
                groups.append({'decision_id':f'd{task:02d}_{state:02d}',
                  'task_id':f'task_{task:04d}','scenario_id':'s',
                  'date':f'2025-07-{22+(state%3):02d}'})
        n=len(groups);X=rng.normal(size=(n,8,44)).astype('float32');B=rng.normal(size=(n,44)).astype('float32')
        Y=(5*X[:,:,0]+8*X[:,:,1]).astype('float32');BY=np.zeros(n,'float32')
        valid=np.zeros((n,8),bool);valid[:,:4]=True;same=np.zeros((n,8),bool)
        # invalid/same actions have deliberately misleading labels
        same[:,3]=True;Y[:,3]=9000;Y[:,4:]=9999
        # Non-original-train labels deliberately poisonous; loaders must ignore them.
        for i,g in enumerate(groups):
            if int(g['task_id'].split('_')[1])>=4:Y[i,:]=99999
        np.savez(src/'paired.npz',X=X,B=B,Y=Y,BY=BY,valid=valid,same=same)
        (src/'groups.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in groups))
        (src/'manifest.json').write_text(json.dumps({'stage':'step6c1r2-real-C3-labelled-v1',
            'pair_sha256':sha256(src/'paired.npz'),'groups_sha256':sha256(src/'groups.jsonl')}))
        selection=[g['decision_id'] for g in groups if int(g['task_id'].split('_')[1])<4]
        (col/'manifest.json').write_text(json.dumps({'stage':'R3-VQ-train-only-restart-v1',
            'selected_decision_ids':selection,'horizons':[20,50],'replicates':3,
            'label_sha256':sha256(src/'paired.npz'),'group_sha256':sha256(src/'groups.jsonl')}))
        rows=[]
        lookup={g['decision_id']:i for i,g in enumerate(groups)}
        for di in selection:
            i=lookup[di]
            for h in [20,50]:
                for rep in range(3):
                    basecost=1000+rep*3
                    rows.append({'decision_id':di,'horizon':h,'replicate':rep,
                         'action_index':-1,'final_road':basecost})
                    for j in [0,1,2]:
                        gain=float(3*X[i,j,0]+2*X[i,j,1])+(h/20)*4-2*j
                        # make noise shared baseline/common random, plus action-specific variation
                        perturb=([3,-2,-1][rep])*(j+1)
                        rows.append({'decision_id':di,'horizon':h,'replicate':rep,
                             'action_index':j,'final_road':basecost-gain-perturb})
        (col/'rollouts.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in rows))
        (col/'report.json').write_text(json.dumps({'stage':'R3-VQ-labels-v1',
                          'status':'COMPLETE','n_restart_action_rows':len(rows)}))
        return src,col,selection

    def test_full_loto_and_stable_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);src,col,selection=self._make_data(base)
            from argparse import Namespace
            out=base/'out'
            train_vq2.run(Namespace(dataset_dir=str(src),collection_dir=str(col),output_dir=str(out),seed=42,trees=160))
            rep=json.loads((out/'report.json').read_text())
            self.assertEqual(rep['stage'],STAGE)
            self.assertEqual(rep['n_selected_states'],16)
            self.assertEqual(len((out/'fold_decisions.jsonl').read_text().splitlines()),16*2*len(METHODS))
            self.assertEqual(len((out/'candidate_predictions.jsonl').read_text().splitlines()),16*2*3)
            for h in ('20','50'):
                for held,z in rep['folds'][h].items():
                    c=z['counts']
                    self.assertNotIn(held,c['train_tasks'])
                    self.assertEqual(c['n_E0_MATCHED_one_step_actions'],c['n_VQ_MEAN_actions'])
                    self.assertEqual(c['n_E0_MATCHED_one_step_actions'],36)
                    self.assertEqual(c['n_E0_FULL_one_step_actions'],36)
                    for info in c['residual_inner_task_oof']:
                        self.assertNotIn(info['oof_task'],info['fit_tasks'])
                self.assertIn('RESID_OOF50',rep['macro_task_metrics'][h])
            # Critical code audit: E0_FULL and VQ_MEAN reproduce the corresponding
            # frozen R3-VQ v1 policies exactly (same data and ExtraTrees RNG).
            import train_vq
            original=base/'old_vq'
            train_vq.run(Namespace(dataset_dir=str(src),collection_dir=str(col),output_dir=str(original),seed=42))
            historical=json.loads((original/'report.json').read_text())
            for h in ('20','50'):
                for held in rep['folds'][h]:
                    current=rep['folds'][h][held]['policies']
                    old=historical['results'][h]['per_heldout_task'][held]['metrics']
                    self.assertAlmostEqual(current['E0_FULL']['mean_gain_sec'],old['ONE_STEP_ET']['mean_gain'])
                    self.assertAlmostEqual(current['VQ_MEAN']['mean_gain_sec'],old['VQ_MEAN']['mean_gain'])
            with self.assertRaises(FileExistsError):
                train_vq2.run(Namespace(dataset_dir=str(src),collection_dir=str(col),output_dir=str(out),seed=42,trees=160))

class AggregateTests(unittest.TestCase):
    def test_seed_aggregation_pins(self):
        from aggregate_vq2 import aggregate
        from vq2_core import write_json
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);studies=[]
            for seed in [42,43]:
                directory=base/f'seed{seed}';directory.mkdir()
                r={'stage':STAGE,'status':'COMPLETE','seed':seed,'n_selected_states':48,'n_original_train_tasks':4,
                   'horizons':[20],'trees':160,'policies':list(METHODS),
                   'macro_task_metrics':{'20':{m:{k:0.2 for k in ('mean_gain_sec','mean_regret_sec','top1_hit',
                   'override_rate','harm_gt40_per_state','harm_gt40_per_override')} for m in METHODS}}}
                write_json(directory/'report.json',r)
                write_json(directory/'provenance.json',{
                   'collection_manifest_sha256':'a','collection_rollouts_sha256':'b',
                   'dataset_paired_sha256':'c','dataset_groups_sha256':'d','code_sha256':{'x':'abc'}})
                studies.append(directory)
            result=aggregate(studies,base/'aggregate')
            self.assertEqual(result['20']['VQ_MEAN']['mean_gain_sec']['mean_over_training_seeds'],.2)
            with self.assertRaises(FileExistsError):aggregate(studies,base/'aggregate')
            broken=json.loads((studies[1]/'provenance.json').read_text());broken['dataset_groups_sha256']='bad'
            write_json(studies[1]/'provenance.json',broken)
            with self.assertRaises(ValueError):aggregate(studies,base/'aggregate-bad')

if __name__=='__main__':unittest.main()
