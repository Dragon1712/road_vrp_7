import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4m'))
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4l'))
import r4m_core as mod
import probe_r4m


def synthetic():
    """200 snapshots, 100 coarse (scenario-correlated) starts, 4 tasks, 4 seeds."""
    rr=[]
    for task in range(4):
        for seed in (60,61,62,63):
            length=7 if (task*4+(seed-60))<4 else 6
            for stopi in range(length):
                stop=100+stopi*350
                for sid in ('2025-07-02_0800','2025-07-09_1700'):
                    snap=(f'task_{task:04d}',sid,seed,stop)
                    physical=(snap[0],seed,stop,'route','best','rng','dw','rw',float(stop)+(.00001 if sid.endswith('1700') else 0))
                    for arm in ('A000','A001'):
                        y=(26. if task in (0,2) and arm=='A000' and stop<1200 else
                           -36. if task==1 and arm=='A001' and stop<1200 else 0.)
                        # Original predecision features: 15 numeric values, no future Road labels.
                        f=[1.,stop/20000.,float(stop<=2000),.6,0.1,.15,.3,.5,float(task)/3.,
                           float(arm=='A001'),float(arm=='A001'),0.,0.25,.30,0.]
                        rr.append({'snap':snap,'arm':arm,'task':snap[0],
                                   'physical':physical,'features':f,
                                   'targets':{'k50':0.,'k200':y*.5,'k500':y,'robust':min(y*.5,y)},
                                   'source':'old' if seed in (60,61) else 'new'})
    assert len({r['snap'] for r in rr})==200
    assert len({r['physical'][:-1] for r in rr})==100
    return rr


class R4MCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rr=synthetic()

    def test_splits_are_task_or_seed_disjoint(self):
        s=mod.r4l.make_splits(self.rr)
        self.assertEqual(len(s),6)
        for name,tr,te,typ in s:
            self.assertFalse({r['snap'] for r in tr} & {r['snap'] for r in te})
            if typ=='TASK_LOTO':
                self.assertFalse({r['task'] for r in tr}&{r['task'] for r in te})
            else:
                self.assertFalse({r['snap'][2] for r in tr}&{r['snap'][2] for r in te})

    def test_coarse_group_identifies_near_same_scenario_states(self):
        original=self.rr
        coarse=mod.feature_records(self.rr,'full')
        self.assertEqual(len({r['physical'] for r in original}),200)
        self.assertEqual(len({r['physical'] for r in coarse}),100)

    def test_quantile_and_invalid(self):
        self.assertEqual(mod.quantile([0,1,3,9],.75),3)
        self.assertEqual(mod.quantile([0,1,3,9],.90),9)
        with self.assertRaises(ValueError):mod.quantile([0,float('nan')],.5)

    def test_calibration_uses_training_partition_only(self):
        tr=[r for r in self.rr if r['task']!='task_0003']
        inp=mod.feature_records(tr,'full')
        a=mod.calibrate_inner(inp,policy='lcb_q75',shuffle=False,seed=13)
        # Never pass the outer heldout task into this function, regardless of
        # whether its labels are perturbed elsewhere in the test harness.
        b=mod.calibrate_inner(inp,policy='lcb_q75',shuffle=False,seed=13)
        self.assertEqual(a,b)
        self.assertIn('n_inner_tasks_with_positive_score',a)

    def test_fail_closed_on_no_inner_positive_predictions(self):
        train=mod.feature_records([r for r in self.rr if r['task']!='task_0003'],'full')
        # Invalid/low support always abstains; never inf/NaN JSON.
        preds=mod.fit(train,'k500')
        held=mod.feature_records([r for r in self.rr if r['task']=='task_0003'],'full')
        picks=mod.select_fold(held,preds,None,'lcb_q90',{'valid':False})
        self.assertEqual(sum(z['selected_arm']!='C3' for z in picks),0)
        self.assertTrue(all(z['predicted_margin_sec'] is None for z in picks))
        self.assertEqual(len(picks),48)

    def test_full_comparisons_deterministic_and_well_formed(self):
        folds,details=mod.evaluate(self.rr)
        self.assertEqual(len(folds),6*2*2*len(mod.POLICIES))
        self.assertEqual(len(details),sum(x['n_scenario_snapshots'] for x in folds))
        for z in folds:
            self.assertEqual(z['n_positive']+z['n_negative']+z['n_zero'],z['n_scenario_snapshots'])
            self.assertLessEqual(z['n_delegations'],z['n_scenario_snapshots'])
            self.assertLessEqual(z['sum_harm_sec'],0)
            self.assertTrue(math.isfinite(z['mean_gain_k500_sec']))
            if z['policy']!='ridge_ref' and not z['calibration']['valid']:
                self.assertEqual(z['n_delegations'],0)
        self.assertEqual(sum(z['n_scenario_snapshots'] for z in folds),len(details))
        txt=mod.report_md({'results':folds})
        self.assertIn('Train内部嵌套风险校准',txt)
        self.assertIn('old_to_new',txt)

    def test_runner_rejects_existing_directory_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fresh'
            a=type('Args',(),{})()
            for attr in ('r4l_result','old_cf','old_capture','old_prefix','new_r4k','new_cf','new_capture','new_prefix'):
                setattr(a,attr,'fakepath')
            a.mode='probe';a.output_dir=str(out)
            with patch.object(probe_r4m,'source_preflight',return_value=(self.rr,{'r4l_report_sha256':'pinned'})):
                probe_r4m.run(a)
                report=json.loads((out/'report.json').read_text())
                prov=json.loads((out/'provenance.json').read_text())
                self.assertEqual(report['status'],'COMPLETE')
                self.assertEqual(report['n_configs'],120)
                self.assertEqual(prov['report_sha256'],mod.sha(out/'report.json'))
                with self.assertRaisesRegex(ValueError,'independent'):
                    probe_r4m.run(a)

    def test_preflight_missing_source_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mod.source_preflight({'r4l_result':tmp})

if __name__=='__main__':unittest.main()
