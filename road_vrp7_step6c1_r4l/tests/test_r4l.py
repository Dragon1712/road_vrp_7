"""R4-L regressions including full synthetic 128 snapshots x 1007 alternatives."""
from __future__ import annotations
import sys,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'road_vrp7_step6c1_r4l'))
import r4l_core as c

class Tests(unittest.TestCase):
    def test_mask_does_not_read_targets(self):
        r={'features':[float(x) for x in range(15)],'physical':tuple(range(9)),
           'targets':{'k500':123456}}
        x=c.with_mask([r],'geometry')[0]
        self.assertEqual(x['targets'],r['targets'])
        self.assertEqual(x['features'][1:6],[0.]*5)
        self.assertEqual(x['physical'],tuple(range(8)))
    def test_split_task_seed_group_separation(self):
        rec=[]
        for t in range(4):
            for seed in (60,61,62,63):
                rec.append({'snap':(f'task_{t:04d}','2025-07-02_0800',seed,100),
                            'task':f'task_{t:04d}'})
        spl=c.make_splits(rec)
        self.assertEqual(len(spl),6)
        for name,train,test,typ in spl:
            self.assertFalse({r['snap'] for r in train}&{r['snap'] for r in test})
        self.assertRaisesRegex(ValueError,'seed',c.make_splits,[r for r in rec if r['snap'][2]!=63])
    def test_abstain_and_gate(self):
        records=[]
        for stop,gain in ((100,12.),(4000,23.)):
            k=('task_0000','2025-07-02_0800',60,stop)
            records.append({'snap':k,'arm':'A001','features':[1.]+[0.]*14,
                'targets':{'k500':gain},'physical':(0,)})
        picks=c.choices_for_model(records,[10.]+[0.]*14,0.,'early2000')
        self.assertEqual([p['actual_gain_k500_sec'] for p in picks],[12.,0.])
        self.assertEqual(c.metrics(picks)['n_delegations'],1)
        picks2=c.choices_for_model(records,[10.]+[0.]*14,15.,'none')
        self.assertTrue(all(p['selected_arm']=='C3' for p in picks2))
    def test_paired_horizons_and_shuffled_control(self):
        records=[]
        for t in range(4):
            for seed in (60,61,62,63):
                for a in range(3):
                    feat=[1.,(seed-60)/3.,t/3.,float(a)/2.]+[0.]*11
                    records.append({'snap':(f'task_{t:04d}','2025-07-02_0800',seed,100),
                       'task':f'task_{t:04d}','arm':f'A{a}',
                       'features':feat,'physical':(t,seed),
                       'targets':{'k50':0.,'k200':5.*a,'k500':(-1)**a*3.*a,
                                  'robust':min(5.*a,(-1)**a*3.*a)}})
        folds,details=c.evaluate(records)
        self.assertEqual(len(folds),6*2*2*2*2*3)
        self.assertTrue(all(f['n_snapshots']>0 for f in folds))
        self.assertTrue(any(f['shuffle_train_labels'] for f in folds))
        self.assertEqual(set(f['split_type'] for f in folds),{'TASK_LOTO','SEED_PAIR_TRANSFER'})
        self.assertTrue(details)
    def create_new_source(self,root):
        kdir=root/'r4k';cdir=root/'capture';cfdir=root/'cf';pdir=root/'prefix'
        for q in (kdir,cdir,cfdir,pdir,cfdir/'units'):q.mkdir(parents=True)
        snaps=[];labs=[];selected=[]
        narm=0
        for t in range(4):
            for scen_i,scen in enumerate(('2025-07-02_0800','2025-07-09_1700')):
                for seed in (62,63):
                    for step in range(8):
                        stop=100+step*200
                        k=(f'task_{t:04d}',scen,seed,stop)
                        selected.append(list(k))
                        s={'task_id':k[0],'scenario_id':k[1],'seed':seed,'stop_count':stop,
                         'current_routes':[list(range(1,11)) for _ in range(5)],
                         'current_route_sha256':'c'+str(t), 'best_route_sha256':'b'+str(t),
                         'rng_state_sha256':'r'+str(seed),
                         'roulette_destroy_weights':[1.,2.,3.,4.], 'roulette_repair_weights':[1.,2.],
                         'anneal_temperature_before_accept':200.0+scen_i*1e-9,
                         'anneal_start_temperature':1000.,'wall_remaining_before_select_sec':28.,
                         'current_surrogate_cost':4000.,'best_surrogate_cost':3800.,
                         'real_selected_repair_index':1}
                        snaps.append(s)
                        count=8 if len(selected)<=111 else 7
                        choices=[{'name':f'A{i:03d}','route':0,'pos':i,'size':2} for i in range(count)]
                        plan={'c3_identity':[1,0,2],'alternatives':choices,'candidate_count':46}
                        for arm in choices:
                            name=arm['name'];narm+=1
                            for h in (50,200,500):
                                labs.append({'snapshot_id':list(k),'task_id':k[0],
                                  'scenario_id':scen,'seed':seed,'stop_count':stop,
                                  'arm':name,'horizon':h,
                                  'road_best_gain_sec':0.,'static_best_gain_sec':0.,'road_current_gain_sec':0.})
                            spec=c.unit_spec(k,50,name)
                            unit={'stage':'R4H-TRAIN-SECTION-COUNTERFACTUAL-v1',
                                'unit_spec':spec,'plan':plan,
                                'result':{'arm':name,'horizon':50,'first_action_id':[0,arm['pos'],2],
                                  'first_original_c3_action_id':[1,0,2]},
                                'source_manifest_sha256':None,'source_snapshots_sha256':None}
                            c.write(c.unit_file(cfdir,k,50,name),unit)
        self.assertEqual(len(snaps),128);self.assertEqual(narm,1007)
        c.write(cdir/'manifest.json',{'arm':'capture','split':'train','section_only':True,
           'task_ids':[f'task_{i:04d}' for i in range(4)],'seeds':[62,63]})
        c.dump_rows(cdir/'snapshots.jsonl',snaps)
        for name,data in [('provenance.json',{}),('metrics.jsonl','')]:
            if name.endswith('.json'):c.write(cdir/name,data)
            else:(cdir/name).write_text(data)
        c.write(pdir/'report.json',{'status':'COMPLETE','exact_matches':512,'n_comparisons':512,
                        'original_capture_snapshots_sha256':c.sha(cdir/'snapshots.jsonl')})
        c.write(pdir/'manifest.json',{})
        (pdir/'paired_rows.jsonl').write_text('');c.write(pdir/'excluded_windows.json',[])
        c.write(cfdir/'manifest.json',{'data_role':'TRAIN_ONLY','horizons':[50,200,500],
           'selected_snapshot_ids':selected,
           'source_sha256':{'capture_snapshots':c.sha(cdir/'snapshots.jsonl')}})
        c.write(cfdir/'report.json',{'status':'COMPLETE'})
        sp={
          'prefix_report':pdir/'report.json','prefix_manifest':pdir/'manifest.json',
          'prefix_paired_rows':pdir/'paired_rows.jsonl','prefix_excluded_windows':pdir/'excluded_windows.json',
          'capture_manifest':cdir/'manifest.json','capture_provenance':cdir/'provenance.json',
          'capture_snapshots':cdir/'snapshots.jsonl','capture_metrics':cdir/'metrics.jsonl',
          'cf_manifest':cfdir/'manifest.json','cf_report':cfdir/'report.json'}
        pins={name:c.sha(path) for name,path in sp.items()}
        for f in (cfdir/'units').glob('*.json'):
            u=c.read(f);u['source_manifest_sha256']=pins['prefix_manifest'];u['source_snapshots_sha256']=pins['capture_snapshots'];c.write(f,u)
        rep={'stage':c.audit.STAGE,'status':'SOURCE_COMPLETE','data_role':'TRAIN_ONLY',
             'analysis':{'planned_scenario_snapshots':128,'completed_scenario_snapshots':128,
             'incomplete_scenario_snapshots':0,'existing_unit_files':3789,
             'complete_coarse_search_starts':64,
             'per_horizon':{str(h):{'action_labels':1007,'positive':0,'negative':0,'zero':1007} for h in (50,200,500)}},
             'source_sha256':pins}
        c.write(kdir/'report.json',rep);c.write(kdir/'provenance.json',
                        {'stage':c.audit.STAGE,'report_sha256':c.sha(kdir/'report.json'),
                         'source_sha256':pins})
        c.dump_rows(kdir/'audited_labels.jsonl',labs)
        return kdir,cdir,cfdir,pdir
    def test_full_1007_units_audited_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            k,capture,cf,p=self.create_new_source(Path(d))
            records,meta=c.read_new(k,capture,cf,p)
            self.assertEqual((len(records),meta['n_snapshots']), (1007,128))
            self.assertEqual(meta['exact_physical_states'],128)
            self.assertEqual(len(set(r['physical'][:-1] for r in records)),64)
            # Source is immutable, fail closed even when the report was COMPLETE.
            (capture/'metrics.jsonl').write_text('modified')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                c.read_new(k,capture,cf,p)
    def test_probe_end_to_end_new_source_and_old_train_stubs(self):
        import probe_r4l
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            k,capture,cf,p=self.create_new_source(root)
            oldrec=[]
            for t in range(4):
                for seed in (60,61):
                    oldrec.append({'snap':(f'task_{t:04d}','2025-07-02_0800',seed,1001),
                        'arm':'A001','task':f'task_{t:04d}',
                        'physical':(f'task_{t:04d}',seed,1001,'c','b','r','d','rp',float(seed)),
                        'features':[1.,.05,1.,1.,.01,.2,.2,.5,.2,.25,0.,0.,.3,.4,1.],
                        'targets':{'k50':0.,'k200':0.,'k500':0.,'robust':0.},
                        'source':'original_s60s61'})
            out=root/'output'
            args=SimpleNamespace(output_dir=str(out),old_cf_dir='not-used',
                old_capture_dir='not-used',old_prefix_dir='not-used',
                new_r4k_dir=str(k),new_capture_dir=str(capture),new_cf_dir=str(cf),
                new_prefix_dir=str(p))
            with patch.object(probe_r4l,'read_old',return_value=(oldrec,{'source_sha256':{}})):
                probe_r4l.execute(args)
            rep=c.read(out/'report.json')
            self.assertEqual(rep['status'],'COMPLETE')
            self.assertEqual(rep['summary']['data_signal']['new']['n_candidate_actions'],1007)
            self.assertEqual(rep['summary']['data_signal']['combined']['n_candidate_actions'],1015)
            self.assertEqual(rep['summary']['n_evaluation_configs'],6*2*2*2*2*3)
            self.assertTrue((out/'fold_predictions.jsonl').is_file())
            with self.assertRaisesRegex(ValueError,'independent output'):
                with patch.object(probe_r4l,'read_old',return_value=(oldrec,{'source_sha256':{}})):
                    probe_r4l.execute(args)
    def test_incomplete_r4k_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            k,capture,cf,p=self.create_new_source(Path(d))
            r=c.read(k/'report.json');r['status']='PARTIAL_COVERAGE_ONLY'
            c.write(k/'report.json',r)
            with self.assertRaisesRegex(ValueError,'SOURCE_COMPLETE'):
                c.read_new(k,capture,cf,p)

if __name__=='__main__':unittest.main()
