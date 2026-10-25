import json,sys,tempfile,unittest
from pathlib import Path
D=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(D))
from r4i_core import (FEATURES,need,read,readl,save,key,physics,_capture_weight_hash,predecision_features,
    ridge_predictors,predict,decision_audit,validate_dataset,group_report,records_from_dataset,
    to_markdown)

class R4ISynthetic(unittest.TestCase):
    def snap(self,task,scenario,seed,stop):
        return {'task_id':task,'scenario_id':scenario,'seed':seed,'stop_count':stop,
           'snapshot_is_actual_section_decision':True,'real_selected_destroy_index':3,
           'real_selected_repair_index':1,'current_routes':[[10,11,12,13,14],[20,21,22,23,24]],
           'current_surrogate_cost':1100.,'best_surrogate_cost':1000.,
           'stop_max_iterations':20000,'wall_remaining_before_select_sec':21.,
           'anneal_temperature_before_accept':40.,'anneal_start_temperature':400.,
           'current_route_sha256':f'c{task}{seed}{stop}',
           'best_route_sha256':f'b{task}{seed}{stop}',
           'rng_state_sha256':f'r{task}{seed}{stop}',
           'roulette_destroy_weights':[1,2,3,4], 'roulette_repair_weights':[1,2]}
    def fixture(self):
        snaps=[];plans=[];labels=[];branches=[];ids=[]
        for t in range(4):
            for sc in ('2025-07-02_0800','2025-07-09_1700'):
                for stop in (1000,5000):
                    task=f'task_{t:04d}';s=self.snap(task,sc,60,stop);snaps.append(s)
                    k=(task,sc,60,stop);ids.append(list(k))
                    plans.append({'snapshot_id':list(k),'n_alternatives':2,'candidate_count':46,'c3_identity':[1,0,2]})
                    for h in (50,200,500):
                        for arm,cost,route in [('C3',1000.,'c3'),('SHAM_C3',1000.,'c3'),('A001',995. if stop==1000 and t in (2,3) else 1005.,'x'),('A002',1005.,'y')]:
                            spec={'task_id':task,'scenario_id':sc,'seed':60,'stop_count':stop,'horizon':h,'arm':arm}
                            branches.append({'unit_spec':spec,'best_road_sec':cost,'best_route_sha256':route})
                            if arm.startswith('A'):
                                act=[0,1,2] if arm=='A001' else [1,1,3]
                                labels.append({'unit_spec':spec,'first_action_id':act,'original_c3_action_id':[1,0,2],
                                    'road_best_gain_sec':1000.-cost,'same_best_routes':False})
        report={'n_distinct_physical_states':8,'n_scenario_snapshots':16}
        return report,{'selected_snapshot_ids':ids},labels,branches,plans,snaps
    def test_dataset_joins_and_signal(self):
        r,m,l,b,p,s=self.fixture()
        selected,snapmap,labmap,feats=validate_dataset(r,m,l,b,p,s)
        self.assertEqual(len(selected),16)
        self.assertEqual(len(feats),32)
        self.assertEqual(len(feats[next(iter(feats))]),len(FEATURES))
        signal=group_report(l,selected,snapmap,labmap)
        self.assertEqual(signal['n_physical_states'],8)
        self.assertEqual(signal['per_horizon']['500']['by_phase']['early_<=2000']['positive'],4)
        self.assertEqual(signal['per_horizon']['500']['by_phase']['later_>2000']['positive'],0)
    def test_no_action_horizon_mixing(self):
        r,m,l,b,p,s=self.fixture()
        l[-1]['road_best_gain_sec']=999
        with self.assertRaisesRegex(ValueError,'Reward/branch'):validate_dataset(r,m,l,b,p,s)
    def test_sham_mutation_rejected(self):
        r,m,l,b,p,s=self.fixture()
        b[1]['best_route_sha256']='tampered'
        with self.assertRaisesRegex(ValueError,'Baseline/SHAM'):validate_dataset(r,m,l,b,p,s)
    def test_holdout_is_whole_task(self):
        r,m,l,b,p,s=self.fixture();sel,sm,lm,f=validate_dataset(r,m,l,b,p,s)
        records=records_from_dataset(sel,sm,lm,f)
        folds,details=decision_audit(records)
        self.assertEqual(len(folds),4*2*2*4)
        self.assertEqual({x['heldout_train_task'] for x in folds}, {f'task_{t:04d}' for t in range(4)})
        for x in folds:
            self.assertNotIn(x['heldout_train_task'],x['training_tasks'])
        self.assertIn('TRAIN',to_markdown({'signal':group_report(l,sel,sm,lm),'folds':folds}))
    def test_source_feature_cannot_use_future_or_road(self):
        s=self.snap('task_0000','2025-07-02_0800',60,1000)
        a=predecision_features(s,{'candidate_count':40,'c3_identity':[0,0,2]},[0,1,2])
        s['road_best_gain_sec']=99999999.;s['future_outcome']=1;s['scenario_id']='some_other_date'
        self.assertEqual(a,predecision_features(s,{'candidate_count':40,'c3_identity':[0,0,2]},[0,1,2]))
    def test_bad_section_geometry_rejected(self):
        s=self.snap('task_0000','2025-07-02_0800',60,1000)
        with self.assertRaisesRegex(ValueError,'Invalid section'):predecision_features(s,{'candidate_count':50,'c3_identity':[0,0,2]},[0,999,2])
    def test_ridge_negative_control_determinism(self):
        r,m,l,b,p,s=self.fixture();sel,sm,lm,f=validate_dataset(r,m,l,b,p,s)
        z=records_from_dataset(sel,sm,lm,f)
        a=ridge_predictors(z,'k500',seed=5,shuffle_labels=True)
        b=ridge_predictors(z,'k500',seed=5,shuffle_labels=True)
        self.assertEqual(a,b)
        self.assertTrue(all(abs(x)<1e8 for x in a))
    def test_full_precision_anneal_preserves_r4h_groups(self):
        # Actual R4-G snapshots have scenario-level SA differences near 1e-11;
        # R4-H kept them distinct; R4-I v1 rounded both to 9 places.
        s=self.snap('task_0000','2025-07-02_0800',60,1006)
        s['anneal_temperature_before_accept']=233.0771455604309
        t={**s,'scenario_id':'2025-07-09_1700',
           'anneal_temperature_before_accept':233.0771455604079}
        self.assertEqual(round(s['anneal_temperature_before_accept'],9),
                         round(t['anneal_temperature_before_accept'],9))
        self.assertNotEqual(physics(s),physics(t))

    def test_r4h_exact_state_cardinality_not_relaxed(self):
        rep,man,labs,branches,plans,snaps=self.fixture()
        # Scenario-specific tiny floating point differences are distinct in
        # R4-H's full-precision physical identity, despite identical routes/RNG.
        for s in snaps:
            if s['scenario_id']=='2025-07-09_1700':
                s['anneal_temperature_before_accept']+=1e-10
        rep['n_distinct_physical_states']=16
        selected,smap,labmap,features=validate_dataset(rep,man,labs,branches,plans,snaps)
        self.assertEqual(len({physics(smap[k]) for k in selected}),16)
        self.assertEqual(group_report(labs,selected,smap,labmap)['n_physical_states'],16)
        rep['n_distinct_physical_states']=8
        with self.assertRaisesRegex(ValueError,'Physical state count disagrees'):
            validate_dataset(rep,man,labs,branches,plans,snaps)

    def test_capture_weights_use_json_canonical_hash_without_casting(self):
        import hashlib,json
        raw=[1,2.0,3.25]
        expected=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()
        self.assertEqual(_capture_weight_hash(raw),expected)
        self.assertNotEqual(_capture_weight_hash([1,2]),
                            _capture_weight_hash([1.0,2.0]))

    def test_invalid_weight_and_temperature_rejected(self):
        s=self.snap('task_0000','2025-07-02_0800',60,1000)
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            physics({**s,'anneal_temperature_before_accept':float('nan')})
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            physics({**s,'roulette_destroy_weights':[float('inf')]})

    def test_source_physical_identity(self):
        s=self.snap('task_0000','2025-07-02_0800',60,1000)
        self.assertEqual(physics(s),physics({**s,'scenario_id':'2025-07-09_1700'}))
    def test_plain_io(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'file.json';save(file,{'foo':1})
            self.assertEqual(read(file)['foo'],1)
    def test_bad_snapshot_identity_rejected(self):
        with self.assertRaisesRegex(ValueError,'Bad snapshot'):key(['task','scenario',60])
if __name__=='__main__':unittest.main()
