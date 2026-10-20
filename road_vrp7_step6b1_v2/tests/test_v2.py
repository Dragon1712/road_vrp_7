from __future__ import annotations
import csv,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import joblib
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from dataset import load,masks,route_overlap,PRIMARY,ORACLE
from models import Predictor,METHODS,interaction_matrix
from train import run,choose,summarize,tune_threshold,test_metrics
from aggregate import read_states,analyze as aggregate

NAMES=['original_static_cost_sec','departure_hour_local','departure_weekday_local',
'route_index','position','size','block_demand','route_load','vehicle_capacity',
'static_section_cost_sec','route_length','relative_position',
'history_observed_share','history_observed_rel_cost','history_has_observation']+[
 f'history_coverage_tminus_{i}' for i in range(6,0,-1)]+[
 f'history_rel_cost_tminus_{i}' for i in range(6,0,-1)]
LABELS=['td_gain_sec','static_gain_same_td_move_sec','static_repair_static_gain_sec',
 'static_repair_td_gain_sec','td_advantage_over_static_repair_sec']


def fixture(path):
    path=Path(path);(path/'analysis').mkdir(parents=True)
    rng=np.random.default_rng(10)
    xs=[];ys=[];groups=[];raw=[]
    for task in range(8):
      for day in range(4):
       for state in range(2):
        sid=f's-{task}-{day}-{state}'
        raw.append({'state_id':sid,'task_id':f'task_{task:04d}',
          'date':f'2025-07-{22+day}','routes':[[1,2,3],[4,5,6]] if state==0 else [[1,3,2],[4,5,6]]})
        for action in range(8):
          x=rng.normal(size=len(NAMES));x[0]=3000+15*state
          x[1]=8 if day%2==0 else 17;x[2]=day;x[8]=45.;x[9]=action+4
          x[12]=rng.uniform(.05,.65)
          true=float((action-3)*12+5*x[3]+12*x[12]+rng.normal()*4-5*task+3*day)
          oracle=true+rng.uniform(.1,20)
          xs.append(x);ys.append([oracle,oracle*.8,true+4,true,oracle-true])
          groups.append({'sample_id':f'{sid}-{action}','state_id':sid,
            'task_id':f'task_{task:04d}','scenario_id':f'day_{day}',
            'date':f'2025-07-{22+day}',
            'task_role':'train' if task<5 else 'val' if task<7 else 'internal_test',
            'date_role':'train' if day<2 else 'val' if day==2 else 'internal_test',
            'seed':'42','source_solver':'C3_topk','snapshot_sec':'5'})
    np.savez_compressed(path/'analysis/features_X.npz',X=np.asarray(xs,dtype='f4'))
    np.savez_compressed(path/'analysis/labels_y.npz',y=np.asarray(ys,dtype='f4'))
    schema={'feature_names':NAMES,'label_names':LABELS,
      'shape_X':[len(xs),len(NAMES)],'shape_y':[len(ys),len(LABELS)]}
    (path/'analysis/dataset_schema.json').write_text(json.dumps(schema))
    with (path/'analysis/groups.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(groups[0]));w.writeheader();w.writerows(groups)
    (path/'run_config.json').write_text(json.dumps({'stage':'step6a-offline-v2','source_solvers':['C3_topk']}))
    (path/'benchmark_summary.json').write_text(json.dumps({'status':'COMPLETE',
     'actual_states':len(raw),'expected_max_actions':len(xs)}))
    with (path/'state_records.jsonl').open('w') as f:
      for r in raw:f.write(json.dumps(r)+'\n')
    return path


class Tests(unittest.TestCase):
 def setUp(self):
    self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
    self.src=fixture(Path(self.temp.name)/'input')
    self.d=load(self.src)
 def test_valid_splits_and_feature_modes(self):
    self.assertEqual(self.d.X.shape[1],27)
    self.assertEqual([self.d.features(x)[0].shape[1] for x in ['core','static','traffic']],[10,12,27])
    for protocol in ('task','date'):
      roles=masks(self.d,protocol)
      self.assertTrue(all(v.any() for v in roles.values()))
    self.assertEqual(route_overlap(self.d)['status'],'CHECKED')
 def test_target_tamper_rejected(self):
    f=self.src/'analysis/labels_y.npz'
    with np.load(f) as z:Y=z['y']
    Y[0,0]=Y[0,3]-5
    np.savez_compressed(f,y=Y)
    with self.assertRaisesRegex(ValueError,'identity violated'):load(self.src)
 def test_duplicate_source_id_rejected(self):
    f=self.src/'analysis/groups.csv'
    with f.open(newline='') as fh:rows=list(csv.DictReader(fh))
    rows[1]['sample_id']=rows[0]['sample_id']
    with f.open('w',newline='') as fh:
      w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with self.assertRaisesRegex(ValueError,'Duplicate candidate'):load(self.src)
 def test_noop_negative_and_executed_only(self):
    rows=[{'state_id':i,'task_id':'t','date':'d','scenario_id':'s','sample_id':str(k)}
          for k,i in enumerate(['a','a','b','b'])]
    y=np.array([5,-10,-7,-8.]);o=y+3
    sc=np.array([4,0,6,2]);gate=np.ones(4)
    a=choose(y,o,sc,gate,rows,0.)
    z=summarize(a)
    self.assertEqual(z['executed_states'],2)
    self.assertEqual(z['nonpositive_executions'],1)
    self.assertAlmostEqual(z['total_signed_gain_sec'],-2)
    self.assertAlmostEqual(z['mean_oracle_repair_upgrade_executed_only_sec'],3)
    a=choose(y,o,sc,gate,rows,None)
    self.assertIsNone(summarize(a)['mean_oracle_repair_upgrade_executed_only_sec'])
    self.assertEqual(summarize(a)['mean_signed_gain_sec'],0)
 def test_pairwise_metrics_are_not_global_auc(self):
    rows=[{'state_id':i} for i in ('a','a','a','b','b','b')]
    z=test_metrics(np.array([2.,-1.,0.,1.,-2.,0.]),np.array([1.,-1.,0.,1.,-1.,0.]),
                   np.array([.8,.1,.5,.8,.1,.5]),rows,'pair')
    self.assertIsNone(z['global_ranking_PR_AUC'])
    self.assertGreater(z['state_top1_hit_rate'],0)
    self.assertIsNotNone(z['gate_positive_PR_AUC'])
 def test_interaction_changes_with_context(self):
    features=['original_static_cost_sec','departure_hour_local','departure_weekday_local','position']
    xs=[];ys=[];groups=[]
    for st in range(120):
      hour=8 if st%2==0 else 17
      for k in (0,1):
        xs.append([3000.,hour,2.,float(k)])
        ys.append((1 if hour==8 else -1)*(2*k-1)*20.)
        groups.append({'state_id':f's{st}'})
    X=np.array(xs,dtype=float);y=np.array(ys,dtype=float)
    self.assertGreater(np.abs(interaction_matrix(X,features)[0]-interaction_matrix(X,features)[2]).sum(),0.)
    mdl=Predictor('pair_inter',11,8,features,'task').fit(X,y,groups)
    scores,_,_=mdl.predict(X)
    self.assertGreater(scores[1]-scores[0],0)
    self.assertLess(scores[3]-scores[2],0)
 def test_all_methods_and_model_reload(self):
    dest=Path(self.temp.name)/'all_models'
    run(self.src,dest,('task','date'),seed=42,trees=8,methods=list(METHODS))
    for proto in ('task','date'):
      report=json.loads((dest/proto/'report.json').read_text())
      self.assertEqual(set(report['methods']),set(METHODS))
      for k in METHODS:
        self.assertTrue((dest/proto/f'{k}_model.joblib').exists())
        with (dest/proto/f'{k}_test_states.csv').open() as f:
          self.assertTrue(list(csv.DictReader(f)))
      cal=report['methods']['P11_two_part_calibrated_static']['group_training_calibration']
      self.assertEqual(cal['method'],'group_oof_platt')
      self.assertGreaterEqual(cal['n_groups'],2)
    artifact=joblib.load(dest/'task/P11_two_part_calibrated_static_model.joblib')
    feat,names=self.d.features(artifact['feature_mode'])
    self.assertEqual(names,artifact['feature_names'])
    ranking,gate,p=artifact['model'].predict(feat[:5])
    self.assertTrue(np.isfinite(ranking).all())
    self.assertTrue(np.isfinite(gate).all())
    with self.assertRaises(FileExistsError):run(self.src,dest,('task',),seed=42,trees=8,methods=['P1_class_static_time'])
 def test_aggregate_identity_and_duplicate(self):
    dest=Path(self.temp.name)/'few'
    run(self.src,dest,('task',),seed=42,trees=8,
        methods=['P1_class_static_time','P8_class_history'])
    out=aggregate([dest],'P1_class_static_time','P8_class_history')
    self.assertEqual(len(out),1)
    with self.assertRaisesRegex(ValueError,'Repeated training seed'):
      aggregate([dest,dest],'P1_class_static_time','P8_class_history')
    f=dest/'task/P1_class_static_time_test_states.csv'
    with f.open(newline='') as fh:rr=list(csv.DictReader(fh))
    rr.append(dict(rr[0]))
    with f.open('w',newline='') as fh:
      w=csv.DictWriter(fh,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    with self.assertRaisesRegex(ValueError,'Duplicate state_id'):
      aggregate([dest],'P1_class_static_time','P8_class_history')

if __name__=='__main__':unittest.main(verbosity=2)
