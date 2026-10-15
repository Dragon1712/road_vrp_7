#!/usr/bin/env python3
"""End-to-end synthetic July road fixture + guarded result pairing; no old repo.

Official ALNS real runtime tested only if alns==7.0.0 is installed.
"""
import sys
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from road_vrp import Road,OD,Objective,legal,write_json
from tasks import make_tasks
from solvers import METHODS,solve
from benchmark import run,inspect_resume
from analyze import analyze
from argparse import Namespace


def fixture(base):
    root=Path(base);cost=root/'july_v1'/'dynamic_costs';cost.mkdir(parents=True)
    nodes=[f'{x},{y}' for x in range(7) for y in range(7)]
    edges=[]
    for x in range(7):
        for y in range(7):
            for dx,dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                nx,ny=x+dx,y+dy
                if 0<=nx<7 and 0<=ny<7:edges.append((f'{x},{y}',f'{nx},{ny}','0'))
    edges.sort()
    e=pd.DataFrame(edges,columns=['osm_u','osm_v','osm_key'])
    e.insert(0,'edge_idx',np.arange(len(e)))
    e['static_travel_sec']=25.;e['length_m']=100.
    # 6 mapped arcs, near all nodes within ~600 m.
    e['mapped_segment_id']='';e['mapped_path_fraction']=np.nan
    e.to_csv(cost/'edge_index.csv',index=False)
    times=pd.date_range('2025-07-02',periods=30*144,freq='10min')
    pd.DataFrame({'time_local':times.strftime('%Y-%m-%d %H:%M:%S')}).to_csv(cost/'time_index.csv',index=False)
    tt=np.full((len(times),len(e)),25.,dtype=np.float32)
    mask=np.zeros(tt.shape,dtype=bool)
    mapped=np.zeros(len(e),dtype=bool)
    for i in range(len(e)):
        if i%13==0:
            mapped[i]=True
            mask[:,i]=True
            tt[:,i]=45. if i%2 else 34.
    arr=np.maximum.accumulate(tt.astype(float)+np.arange(len(times))[:,None]*600,axis=0)
    fifo=(arr-np.arange(len(times))[:,None]*600).astype(np.float32)
    np.savez_compressed(cost/'edge_costs.npz',travel_sec_raw=tt,travel_sec_fifo=fifo,
                        chicago_observed_mask=mask,is_mapped_edge=mapped,
                        slot_seconds=600.,slot_local_naive_ns=times.asi8)
    write_json(cost.parent/'stage2_validation.json',{'validation':'PASS'})
    qc=[]
    for date in pd.date_range('2025-07-02','2025-07-31',freq='D'):
        split='train' if date.day<=21 else ('validation' if date.day<=25 else 'test')
        for hm in ('08:00','17:00'):
            qc.append({'scenario_id':date.strftime('%Y-%m-%d')+'_'+hm.replace(':',''),
                       'departure_local':date.strftime('%Y-%m-%d')+' '+hm+':00',
                       'date':date.strftime('%Y-%m-%d'),'split':split,
                       'eligible':not(date.day==27),'edge_mean':.7,
                       'rejection_reasons':'low_test_day' if date.day==27 else ''})
    qcp=root/'scenario_quality.csv';pd.DataFrame(qc).to_csv(qcp,index=False)
    return cost,qcp


def run_test():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        cost,qc=fixture(root)
        road=Road(cost)
        assert len(road.edges)>100
        taskdir=root/'tasks'
        manifest=make_tasks(road,scenario_quality=qc,output_dir=taskdir,num_tasks=10,seed=7)
        assert manifest['splits']=={'train':40,'validation':8,'test':12}
        task=json.loads((taskdir/'task_0000.json').read_text())
        assert legal(task,task['seed_routes'])
        sc=next(s for s in task['scenarios'] if s['scenario_id']=='2025-07-22_0800')
        cache=root/'od_cache'
        od=OD(road,task,sc,step_min=30,hours=5,cache_dir=cache)
        assert not od.cache_hit
        od2=OD(road,task,sc,step_min=30,hours=5,cache_dir=cache)
        assert od2.cache_hit
        for planner in ('static','oracle'):
            for solver in ('greedy','sa','vnd'):
                obj=Objective(task,od,planner)
                res=solve(solver,task,obj,seed=42,wall=.15,max_iter=80)
                assert legal(task,res['routes'])
                ex=road.replay(task,res['routes'],sc['departure_local'])
                assert ex['ex_post_road_travel_sec']>0
                assert ex['num_vehicles_used']==5
                assert sum(ex['vehicle_loads'])==sum(c['demand'] for c in task['customers'])
                print('PASS solver',solver,planner,round(ex['ex_post_road_travel_sec'],2),flush=True)
        out=root/'benchmark'
        args=Namespace(cost_dir=str(cost),task_dir=str(taskdir),output_dir=str(out),od_cache_dir=str(cache),
                       split='validation',include_ineligible=False,methods='greedy,sa,vnd',planners='static,oracle',
                       seeds='42',max_tasks=1,max_scenarios=1,wall_sec=.1,max_iterations=50,
                       od_step_min=30,od_hours=5,resume=False,scenario_ids='')
        run(args)
        assert len(inspect_resume(out))==6
        s=analyze(out/'metrics.csv',out/'run_config.json',root/'analysis')
        assert s['grid_records']==6
        a=Namespace(**vars(args));a.resume=True
        run(a)
        assert len(inspect_resume(out))==6
        # Negative control: delete a row to prove strict COMPLETE-grid checking.
        raw=pd.read_csv(out/'metrics.csv');raw.iloc[:-1].to_csv(root/'incomplete.csv',index=False)
        try:analyze(root/'incomplete.csv',out/'run_config.json',root/'bad_analysis')
        except ValueError as exc:assert 'Incomplete result grid' in str(exc)
        else:raise AssertionError('Analyzer accepted incomplete results')
        print('PASS strict result-grid negative control')
        # Ensure omitted future day is not silently treated as eligible.
        assert sum(s['split']=='test' and s['eligible'] for s in task['scenarios'])==10
        try:
            import importlib.metadata
            assert importlib.metadata.version('alns')=='7.0.0'
        except (ImportError,AssertionError,importlib.metadata.PackageNotFoundError):
            print('SKIP official ALNS integration: install alns==7.0.0 and rerun on server')
        else:
            obj=Objective(task,od,'oracle')
            res=solve('official_alns',task,obj,seed=42,wall=.2,max_iter=25)
            assert legal(task,res['routes'])
            print('PASS official ALNS 7.0.0 integration')
    print('PASS stage3 synthetic July fixture, baseline methods, resume, paired analysis')

if __name__=='__main__':run_test()
