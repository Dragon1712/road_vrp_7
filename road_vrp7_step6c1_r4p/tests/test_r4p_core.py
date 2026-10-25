from __future__ import annotations
import csv,math,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4p'))
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4o'))
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4l'))
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4i'))
import r4p_core as p
import r4o_core as o
from importlib.util import spec_from_file_location,module_from_spec
sp=spec_from_file_location('r4o_synthetic_test',ROOT/'road_vrp7_step6c1_r4o/tests/test_r4o_core.py')
t=module_from_spec(sp);sp.loader.exec_module(t)


def small_graph():
    # All ordered pairs including depot, deliberately asymmetric.
    z={'DEPOT':{},'A':{},'B':{},'C':{},'D':{}}
    for src in z:
        for dst in z:
            if src!=dst:z[src][dst]=float(1+len(dst)+(src>dst)*2)
    return {src:list(d.items()) for src,d in z.items()}


def mini_task():
    return {'task_id':'task_0000','depot_node':'DEPOT',
            'customers':[{'id':11,'node':'A'},{'id':12,'node':'B'},{'id':13,'node':'C'},{'id':14,'node':'D'}]}


def mini_snap():
    return {'task_id':'task_0000','scenario_id':'2025-07-02_0800','seed':60,'stop_count':101,
       'current_routes':[[11,12,13,14]],'best_routes':[[11,13,12,14]]}


class Tests(unittest.TestCase):
    def test_graph_csv_only_reads_static_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'edges.csv'
            with f.open('w',newline='') as dst:
                w=csv.DictWriter(dst,fieldnames=['osm_u','osm_v','static_travel_sec','travel_sec_fifo','road_future_label'])
                w.writeheader()
                w.writerow({'osm_u':'DEPOT','osm_v':'A','static_travel_sec':3,'travel_sec_fifo':'999999','road_future_label':'9999'})
                w.writerow({'osm_u':'A','osm_v':'DEPOT','static_travel_sec':4,'travel_sec_fifo':'NAN','road_future_label':'10000'})
            g,n=p.graph_csv(f)
            self.assertEqual(n,2)
            self.assertEqual(p.dijkstra(g,'DEPOT',{'A'})['A'],3)
    def test_directionality(self):
        g={'x':[('y',2.)],'y':[('x',8.)]}
        self.assertEqual(p.dijkstra(g,'x',{'y'})['y'],2)
        self.assertEqual(p.dijkstra(g,'y',{'x'})['x'],8)
    def test_od_and_features_finite(self):
        z=p.task_od(mini_task(),small_graph());s=mini_snap()
        for a in [(0,0,2),(0,1,2),(0,2,2),(0,1,3)]:
            f=p.static_features(s,a,z)
            self.assertEqual(len(f),12)
            self.assertTrue(all(math.isfinite(v) for v in f))
    def test_bad_section_raises(self):
        with self.assertRaisesRegex(ValueError,'Illegal Section'):
            p.static_features(mini_snap(),(0,3,3),p.task_od(mini_task(),small_graph()))
    def test_no_nontrain_task(self):
        with self.assertRaisesRegex(ValueError,'Non TRAIN'):
            p.task_od({**mini_task(),'task_id':'task_0008'},small_graph())
    def test_modes_lengths_and_no_label_access(self):
        base=[.02*i for i in range(38)];road=[.01*i for i in range(12)]
        self.assertEqual([len(p.fuse(base,road,m)) for m in p.MODES],[38,50,62])
        self.assertEqual(p.fuse(base,road,'route_context'),base)
    def test_bilinear_interaction_changes_with_road(self):
        base=[1.]*38;a=[1.]*12;b=[2.]*12
        self.assertNotEqual(p.fuse(base,a,'road_conditioned'),p.fuse(base,b,'road_conditioned'))
    def test_invalid_numeric_refused(self):
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            p.fuse([1.]*38,[float('nan')]+[1.]*11,'road_concat')
    def test_static_preserves_customer_id_relations_not_magnitude(self):
        task=mini_task();g=small_graph();s=mini_snap();a=(0,1,2)
        v=p.static_features(s,a,p.task_od(task,g))
        nt={**task,'customers':[{**c,'id':c['id']+1000} for c in task['customers']]}
        ns={**s,'current_routes':[[x+1000 for x in r] for r in s['current_routes']],
            'best_routes':[[x+1000 for x in r] for r in s['best_routes']]}
        self.assertEqual(v,p.static_features(ns,a,p.task_od(nt,g)))
    def test_real_r4o_synthetic_loto_with_road_augmentation(self):
        import r4l_core
        rec=t.make()
        for r in rec:
            base=r['feature_sets']['route_context']
            r['feature_sets']={mode:p.fuse(base,[.1*(i+1) for i in range(12)],mode) for mode in p.MODES}
        folds,details=p.evaluate(rec,r4l_core.make_splits(rec),o)
        self.assertEqual(len(folds),6*3*2*5)
        self.assertGreater(len(details),0)
        self.assertTrue(any(x['policy']=='learned_gate_pairwise' for x in folds))
        self.assertTrue(all(x['n_snapshots']>0 for x in folds))
    def test_snapshot_leakage_rejected(self):
        rec=t.make()
        for r in rec:
            base=r['feature_sets']['route_context']
            r['feature_sets']={mode:p.fuse(base,[1.]*12,mode) for mode in p.MODES}
        a=[r for r in rec if r['task']=='task_0000']
        with self.assertRaisesRegex(ValueError,'Snapshot leakage'):
            p.evaluate(rec,[('bad',a,a,'TASK_LOTO')],o)
    def test_core_mode_restored(self):
        self.assertEqual(o.FEATURE_MODES,('baseline15','route_context','route_context_interact'))


if __name__=='__main__':unittest.main()
