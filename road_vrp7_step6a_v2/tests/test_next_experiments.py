#!/usr/bin/env python3
"""No Chicago tensor required: synthetic probes of two new experiment branches."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, pathlib, sys, tempfile, types
import numpy as np

HERE=pathlib.Path(__file__).resolve()
ROOT=HERE.parents[2]

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def test_b0():
    b=load(ROOT/'road_vrp7_step6b0'/'train_baselines.py','step6b0_test')
    with tempfile.TemporaryDirectory() as t:
        base=pathlib.Path(t)/'data';(base/'analysis').mkdir(parents=True)
        names=['original_static_cost_sec','static_section_cost_sec','history_observed_share']
        labels=['td_gain_sec']
        rng=np.random.default_rng(2);rows=[];X=[];Y=[];k=0
        for task in range(10):
            for day in range(4):
                for state in range(6):
                    for action in range(5):
                        static=float(rng.normal());traffic=float(rng.random())
                        # Both static+traffic matter; make positive examples for all splits.
                        label=12. if static + 2*traffic + rng.normal(0,.3)>1.45 else -15.
                        X.append([10.,static,traffic]);Y.append([label])
                        rows.append({'sample_id':f's{k}','state_id':f's{task}_{day}_{state}',
                           'task_id':f'task_{task}','scenario_id':f'day_{day}',
                           'date':f'2025-07-{22+day}',
                           'task_role':'train' if task<6 else 'val' if task<8 else 'internal_test',
                           'date_role':'train' if day<2 else 'val' if day==2 else 'internal_test',
                           'source_solver':'C3','seed':42,'snapshot_sec':5})
                        k+=1
        np.savez_compressed(base/'analysis/features_X.npz',X=np.array(X,dtype='f4'))
        np.savez_compressed(base/'analysis/labels_y.npz',y=np.array(Y,dtype='f4'))
        # Keep the variable name EXACTLY the production exporter uses ('y').
        (base/'analysis/dataset_schema.json').write_text(json.dumps({'feature_names':names,'label_names':labels}))
        with (base/'analysis/groups.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        output=pathlib.Path(t)/'output';b.run(base,output,42,('task','date'))
        assert json.loads((output/'config.json').read_text())['n']==1200
        res=json.loads((output/'results.json').read_text())
        assert all(x['methods']['B2_static_plus_history_RF']['test_policy']['states']>0 for x in res)
        assert all(x['methods']['B2_static_plus_history_RF']['test_PR_AUC'] is not None for x in res)
        try:b.run(base,output,42,('task','date'))
        except FileExistsError:pass
        else:raise AssertionError('Overwriting output not blocked')
        # Deliberate cross-fold state duplication is rejected.
        swapped=[dict(x) for x in rows]
        swapped[-1]['state_id']=swapped[0]['state_id']
        try:b.verify_group_split(swapped,'task')
        except ValueError:pass
        else:raise AssertionError('State fold leakage not blocked')
    print('PASS: B0 full synthetic fit/eval, role isolation and output immutability')


def test_v2_loader():
    v=load(ROOT/'road_vrp7_step6a_v2'/'label_current.py','step6a_current_loader_test')
    with tempfile.TemporaryDirectory() as t:
        folder=pathlib.Path(t)
        config={'stage':'step6a-current-source-v1','planner':'static',
            'tasks':{'task_0000.json':'s'},'scenarios':['day0']}
        (folder/'run_config.json').write_text(json.dumps(config))
        (folder/'benchmark_summary.json').write_text(json.dumps({'status':'COMPLETE','source_planner':'static','expected_runs':1}))
        lines=[{'key':['task_0000','day0',42],'planner':'static','source_solver':'C3_topk',
            'sampled_states':[{'target_sec':5,'elapsed_sec':2.0,
               'routes':[[1,2],[3]],'nonbest':True,'gap_before_sec':4.}]},]
        (folder/'current_states.jsonl').write_text('\n'.join(json.dumps(x) for x in lines)+'\n')
        st,cfg,path=v.load_current_states(folder,['C3_topk'],target_times=[5.])
        assert len(st)==1 and st[0]['source_planner']=='static' and st[0]['captured_nonbest']
        assert st[0]['routes']==[[1,2],[3]]
        config['planner']='oracle';(folder/'run_config.json').write_text(json.dumps(config))
        try:v.load_current_states(folder,['C3_topk'])
        except ValueError:pass
        else:raise AssertionError('Oracle source not blocked')
    print('PASS: A-v2 honest CURRENT loader, source planner leakage checks')


def test_accepted_collector():
    # Stub only external imports to test the bounded, priority-aware callback offline.
    road=types.ModuleType('road_vrp')
    for n in ('Road','OD','Objective'):
        setattr(road,n,type(n,(),{}))
    road.legal=lambda *a:True
    road.read_json=lambda p:json.loads(pathlib.Path(p).read_text())
    road.write_json=lambda p,x:pathlib.Path(p).write_text(json.dumps(x))
    road.sha256=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
    benchmark=types.ModuleType('benchmark');benchmark.load_tasks=lambda p:([],[],{})
    sys.modules['road_vrp']=road;sys.modules['benchmark']=benchmark
    cap=load(ROOT/'road_vrp7_step6a_v2'/'capture_current.py','step6a_capture_test')
    sink=cap.AcceptedCollector(30,5)
    sink.observe([[1],[2]],100,100,0.1)
    sink.observe([[2],[1]],105,100,1.0)
    sink.observe([[3],[2]],110,100,2.0)
    sink.observe([[3],[1]],110,100,5.1)
    sink.observe([[3],[1]],110,100,31.)
    assert len(sink.get())==2 and sink.get()[0]['nonbest']
    assert sink.get()[0]['routes']==[[2],[1]] and sink.get()[1]['target_sec']==10
    print('PASS: A-v2 accepted-state bounded sampling and non-best priority')

if __name__=='__main__':
    test_b0();test_v2_loader();test_accepted_collector()
    print('ALL SYNTHETIC TESTS PASSED')
