"""Causal historical-source and feature isolation tests for R4-Q."""
import sys,tempfile,unittest,csv,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
for n in ('road_vrp7_step6c1_r4q','road_vrp7_step6c1_r4p','road_vrp7_step6c1_r4o','road_vrp7_step6c1_r4l','road_vrp7_step6c1_r4n','road_vrp7_step6c1_r4m','road_vrp7_step6c1_r4i','road_vrp7_step6c1_r4k'):
    sys.path.insert(0,str(ROOT/n))
import r4q_core as q
import r4p_core as p

class TestTrafficLag(unittest.TestCase):
    def times(self):
        dates=[datetime(2025,7,1,8),datetime(2025,7,1,8,10),datetime(2025,7,2,8),datetime(2025,7,8,17),datetime(2025,7,9,17)]
        return np.array([np.datetime64(x,'ns').astype(np.int64) for x in dates],dtype=np.int64)

    def test_exact_lag24_july2(self):
        self.assertEqual(q.historical_slot(self.times(),'2025-07-02 08:00:00'),(0,'2025-07-01 08:00:00'))

    def test_exact_lag24_july9(self):
        self.assertEqual(q.historical_slot(self.times(),'2025-07-09 17:00:00'),(3,'2025-07-08 17:00:00'))

    def test_missing_lag_refused(self):
        with self.assertRaisesRegex(ValueError,'No exact'):
            q.historical_slot(self.times(),'2025-07-02 09:00:00')

    def test_stage2_start_boundary_resolves_static_only(self):
        # Real Stage2 starts 2025-07-02 00:00, not July 1.
        ns=np.array([np.datetime64(x,'ns').astype(np.int64) for x in
             ('2025-07-02T00:00:00','2025-07-02T08:00:00',
              '2025-07-08T17:00:00','2025-07-09T17:00:00')], dtype=np.int64)
        early=q.resolve_historical_source(ns,'2025-07-02 08:00:00')
        self.assertFalse(early['available'])
        self.assertIsNone(early['historical_slot'])
        self.assertIsNone(early['slot_index'])
        self.assertEqual(early['requested_historical_slot'],'2025-07-01 08:00:00')
        self.assertEqual(early['fallback_reason'],'BEFORE_DATA_START')
        late=q.resolve_historical_source(ns,'2025-07-09 17:00:00')
        self.assertTrue(late['available'])
        self.assertEqual(late['historical_slot'],'2025-07-08 17:00:00')

    def test_internal_missing_slot_still_fails_closed(self):
        ns=np.array([np.datetime64(x,'ns').astype(np.int64) for x in
             ('2025-07-02T00:00:00','2025-07-08T17:00:00',
              '2025-07-08T17:20:00','2025-07-09T17:00:00')], dtype=np.int64)
        with self.assertRaisesRegex(ValueError,'No exact'):
            q.resolve_historical_source(ns,'2025-07-09 17:10:00')
        with self.assertRaisesRegex(ValueError,'No exact'):
            q.resolve_historical_source(ns,'2025-07-11 17:00:00')

    def test_full_source_fallback_no_future_leak(self):
        with tempfile.TemporaryDirectory() as d:
            self.source_files(d)
            clocks=['2025-07-02T00:00:00','2025-07-02T08:00:00',
                    '2025-07-08T17:00:00','2025-07-09T17:00:00',
                    '2025-07-09T17:10:00']
            ns=np.array([np.datetime64(x,'ns').astype(np.int64) for x in clocks],dtype=np.int64)
            raw=np.ones((5,6),np.float32)*10
            obs=np.zeros((5,6),dtype=np.bool_)
            raw[1,0]=99  # July 2 same-day future must NEVER enter July 2 features
            raw[2,0]=50  # July 8 past 24h observation for July 9
            obs[1,0]=True;obs[2,0]=True
            np.savez_compressed(Path(d)/'edge_costs.npz',travel_sec_raw=raw,
                chicago_observed_mask=obs,slot_local_naive_ns=ns,
                is_mapped_edge=np.array([True,False,False,False,False,False]))
            tasks={}
            for i in range(4):
                tasks[f'task_{i:04d}']={'task_id':f'task_{i:04d}',
                    'depot_node':'0','customers':[{'id':1,'node':'1'},{'id':2,'node':'2'}],
                    'scenarios':[{'scenario_id':sid,'departure_local':time,'split':'train','eligible':True}
                      for sid,time in [('2025-07-02_0800','2025-07-02 08:00:00'),
                                       ('2025-07-09_1700','2025-07-09 17:00:00')]]}
            statics,lagged,meta=q.build_context(tasks,d)
            a=meta['2025-07-02_0800'];b=meta['2025-07-09_1700']
            self.assertFalse(a['available']);self.assertTrue(b['available'])
            self.assertEqual(a['observed_edges'],0)
            self.assertEqual(a['observed_mapped_fraction'],0)
            self.assertIsNone(a['historical_slot'])
            self.assertEqual(b['historical_slot'],'2025-07-08 17:00:00')
            self.assertEqual(b['observed_edges'],1)
            for task in tasks:
                np.testing.assert_array_equal(statics[task]['dist'],lagged[task,'2025-07-02_0800']['dist'])
            self.assertNotEqual(lagged['task_0000','2025-07-09_1700']['dist'][0,1],
                                statics['task_0000']['dist'][0,1])
            # Distort future observations; missing-history scenario MUST stay static.
            raw[1,0]=999;raw[3,0]=999
            np.savez_compressed(Path(d)/'edge_costs.npz',travel_sec_raw=raw,
                chicago_observed_mask=obs,slot_local_naive_ns=ns,
                is_mapped_edge=np.array([True,False,False,False,False,False]))
            again,lag2,_=q.build_context(tasks,d)
            for task in tasks:
                np.testing.assert_array_equal(lag2[task,'2025-07-02_0800']['dist'],again[task]['dist'])
                np.testing.assert_array_equal(lag2[task,'2025-07-09_1700']['dist'],lagged[task,'2025-07-09_1700']['dist'])

    def test_nonmonotone_rejected(self):
        with self.assertRaisesRegex(ValueError,'nonmonotonic'):
            q.historical_slot(self.times()[::-1],'2025-07-02 08:00:00')

    def test_no_lookahead_lag_override(self):
        with self.assertRaisesRegex(ValueError,'frozen'):
            q.historical_slot(self.times(),'2025-07-02 08:00:00',lag_hours=0)

    def test_features_no_label(self):
        v=q.fused_features([.1]*38,[.2]*12,[.3]*12,.6,'lag24_concat')
        self.assertEqual(len(v),63)
        self.assertEqual(len(q.fused_features([.1]*38,[.2]*12,[.3]*12,.6,'lag24_conditioned')),75)
        self.assertEqual(len(q.fused_features([.1]*38,[.2]*12,[.3]*12,.6,'static_concat')),50)
        self.assertEqual(len(q.fused_features([.1]*38,[.2]*12,[.3]*12,.6,'route_context')),38)
        self.assertAlmostEqual(v[-1],.6)
        with self.assertRaises(ValueError):q.fused_features([.1]*38,[.2]*12,[.3]*12,1.1,'lag24_concat')

    def test_history_stratification_never_counts_oracle(self):
        meta={'2025-07-02_0800':{'available':False},'2025-07-09_1700':{'available':True}}
        def row(scenario,gain,policy='learned_gate_pairwise',shuffle=False):
            return {'evaluation':{'split':'hold_task_task_0000','mode':'lag24_concat',
                 'shuffle':shuffle,'policy':policy},
                 'snapshot_id':['task_0000',scenario,62,100],
                 'selected_arm':'A001' if gain else 'C3',
                 'gain_k500_sec':gain}
        r=q.stratify_by_history([
            row('2025-07-02_0800',0.),row('2025-07-09_1700',3.),
            row('2025-07-09_1700',-6.),
            row('2025-07-09_1700',100.,'sampled_oracle'),
            row('2025-07-09_1700',100.,shuffle=True)],meta)
        self.assertEqual(len(r),2)
        by={z['scenario_id']:z for z in r}
        self.assertFalse(by['2025-07-02_0800']['history_available'])
        self.assertEqual(by['2025-07-02_0800']['n_delegations'],0)
        self.assertEqual(by['2025-07-09_1700']['mean_gain_sec'],-1.5)
        self.assertEqual(by['2025-07-09_1700']['positive'],1)
        self.assertEqual(by['2025-07-09_1700']['negative'],1)

    def test_graph_directed(self):
        g=q.graph_from_weights([('a','b'),('b','a')],[3.,8.])
        d=p.dijkstra(g,'a',{'b'})
        self.assertEqual(d['b'],3.)
        self.assertEqual(p.dijkstra(g,'b',{'a'})['a'],8.)

    def source_files(self, root):
        root=Path(root)
        with (root/'edge_index.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['edge_idx','osm_u','osm_v','static_travel_sec'])
            for i,(a,b,c) in enumerate([(0,1,10),(1,2,10),(2,0,10),(0,2,40),(1,0,30),(2,1,30)]):
                w.writerow([i,a,b,c])
        times=self.times()
        raw=np.ones((len(times),6),dtype=np.float32)*10
        raw[0,0]=20;raw[3,0]=50
        obs=np.zeros(raw.shape,dtype=np.bool_);obs[0,0]=True;obs[3,0]=True
        np.savez_compressed(root/'edge_costs.npz',travel_sec_raw=raw,
          chicago_observed_mask=obs,slot_local_naive_ns=times,
          is_mapped_edge=np.array([True,False,False,False,False,False]))
        return times,raw,obs

    def test_raw_mask_static_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            _,raw,obs=self.source_files(d)
            edges,static,xx,mask,clock,mapped=q.road_arrays(Path(d)/'edge_index.csv',Path(d)/'edge_costs.npz')
            pidx,_=q.historical_slot(clock,'2025-07-02 08:00:00')
            weight=np.where(mask[pidx],xx[pidx],static)
            self.assertEqual(weight[0],20.)
            self.assertEqual(weight[1],10.)
            self.assertFalse(mask[pidx,1])

    def test_future_traffic_changes_do_not_change_historical_features(self):
        with tempfile.TemporaryDirectory() as d:
            _,raw,obs=self.source_files(d)
            def selected():
                e,s,v,m,t,mp=q.road_arrays(Path(d)/'edge_index.csv',Path(d)/'edge_costs.npz')
                idx,_=q.historical_slot(t,'2025-07-02 08:00:00')
                return np.where(m[idx],v[idx],s)
            original=selected()
            raw[2,:]=9999  # same-day target observations should be invisible
            raw[4,:]=9999  # later date is also invisible
            np.savez_compressed(Path(d)/'edge_costs.npz',travel_sec_raw=raw,
                chicago_observed_mask=obs,slot_local_naive_ns=self.times(),
                is_mapped_edge=np.array([True,False,False,False,False,False]))
            np.testing.assert_array_equal(original,selected())

    def test_corrupt_edges_refused(self):
        with tempfile.TemporaryDirectory() as d:
            self.source_files(d)
            pth=Path(d)/'edge_index.csv'
            s=pth.read_text();pth.write_text(s.replace('0,0,1,10','7,0,1,10'))
            with self.assertRaisesRegex(ValueError,'Edge order'):
                q.road_arrays(pth,Path(d)/'edge_costs.npz')

    def test_task_od_observed_day_diff(self):
        with tempfile.TemporaryDirectory() as d:
            self.source_files(d)
            tasks={}
            for i in range(4):
                tasks[f'task_{i:04d}']={
                    'task_id':f'task_{i:04d}', 'depot_node':'0',
                    'customers':[{'id':1,'node':'1'},{'id':2,'node':'2'}],
                    'scenarios':[{'scenario_id':sid,'departure_local':clock,
                        'split':'train','eligible':True} for sid,clock in
                          [('2025-07-02_0800','2025-07-02 08:00:00'),
                           ('2025-07-09_1700','2025-07-09 17:00:00')]]}
            base,lag,meta=q.build_context(tasks,d)
            self.assertEqual(len(lag),8)
            self.assertEqual(len(base),4)
            self.assertEqual(meta['2025-07-02_0800']['historical_slot'],'2025-07-01 08:00:00')
            self.assertEqual(meta['2025-07-09_1700']['observed_edges'],1)
            self.assertEqual(meta['2025-07-09_1700']['observed_mapped_fraction'],1.0)
            self.assertEqual(lag['task_0000','2025-07-02_0800']['dist'].shape,(3,3))
            self.assertNotEqual(lag['task_0000','2025-07-02_0800']['dist'][0,1],
                                lag['task_0000','2025-07-09_1700']['dist'][0,1])

    def test_train_role_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.source_files(d)
            tasks={f'task_{i:04d}':{'task_id':f'task_{i:04d}','depot_node':'0',
                'customers':[{'id':1,'node':'1'}], 'scenarios':[
                  {'scenario_id':'2025-07-02_0800','departure_local':'2025-07-02 08:00:00',
                   'split':'validation','eligible':True}]} for i in range(4)}
            with self.assertRaisesRegex(ValueError,'NonTrain'):
                q.build_context(tasks,d,scenarios=('2025-07-02_0800',))

if __name__=='__main__':unittest.main()

class TestSyntheticEvaluation(unittest.TestCase):
    def test_four_task_holdout_pipeline(self):
        import r4l_core as l
        import r4o_core as o
        rec=[]
        for taskidx in range(4):
            task=f'task_{taskidx:04d}'
            for seed in (60,61,62,63):
                for scen in ('2025-07-02_0800','2025-07-09_1700'):
                    sid=(task,scen,seed,1000)
                    for a in range(2):
                        base=[1.]+[.1*(j+1)+.04*a+.03*taskidx+.01*(seed-60) for j in range(37)]
                        sf=[.03*(j+1) for j in range(12)]
                        lf=[.02*(j+1)+.01*(seed-60) for j in range(12)]
                        modes={m:q.fused_features(base,sf,lf,.4,m) for m in q.MODES}
                        # Informative positives/negatives but not trivial all-constant labels.
                        gain=20. if (taskidx+seed+a)%3==0 else (-4. if a else 0.)
                        rec.append({'snap':sid,'task':task,'source':('original_s60s61' if seed<62 else 'new_s62s63'),
                            'arm':f'A{a:03d}', 'physical':(task,seed,1000),
                            'features':[1.]+base[1:15],
                            'feature_sets':modes,
                            'gate_features':[1.,.05*(taskidx+1),.01*(seed-60),.1]*2+[1.,.2],
                            'targets':{'k50':gain*.7,'k200':gain*.9,'k500':gain,'robust':min(gain*.9,gain)}})
        # Gate has exactly 10 dimensions (and is state-constant).
        self.assertEqual(len(rec[0]['gate_features']),10)
        folds,detail=q.evaluate(rec,l.make_splits(rec),o)
        self.assertEqual(len(folds),240)
        self.assertEqual(len(detail),2560)
        self.assertEqual(o.FEATURE_MODES,('baseline15','route_context','route_context_interact'))
        self.assertFalse(any(not z['oracle_conditioned'] and z['policy']=='sampled_oracle' for z in folds))
