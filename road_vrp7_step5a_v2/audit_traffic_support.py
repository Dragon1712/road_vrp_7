#!/usr/bin/env python3
"""Offline, independent OD/path observation-support audit for Step5-A AT-RR.

This does not participate in search. It samples the frozen seed_routes to explain
which AT-RR OD-derived traffic features have genuinely observed Chicago backing.
A supported OSM edge is itself an allocation of a monitored corridor speed,
not a per-edge direct speed observation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_stage3'))
from road_vrp import Road,OD,Objective,write_json
from benchmark import load_tasks,frozen_scenarios
from traffic_operators import TrafficSignature,route_trace


def audit(args):
    road=Road(args.cost_dir)
    _,tasks,_=load_tasks(args.task_dir)
    if args.max_tasks:tasks=tasks[:args.max_tasks]
    rows=[];totals=[]
    for task in tasks:
        for scenario in frozen_scenarios(task,'validation',False,args.max_scenarios):
            obj=Objective(task,OD(road,task,scenario,step_min=20,hours=5,cache_dir=args.od_cache_dir),'oracle')
            arrival={}
            for rt in task['seed_routes']:
                arr,_,_=route_trace(rt,obj)
                arrival.update(zip(rt,arr))
            sig=TrafficSignature(obj)
            start=road.seconds(scenario['departure_local'])
            node_by_id={int(c['id']):str(c['node']) for c in task['customers']}
            for x in sig.ids:
                t=arrival[x]
                level,trend=sig.features(x,t)
                i=obj.index[x]
                for y in sig.neighbors[x]:
                    j=obj.index[y]
                    static=float(obj.od.static[i,j]); oracle=float(obj.od.travel(i,j,t,'oracle'))
                    try:dur,path=road.path(node_by_id[x], node_by_id[y],start+t,'oracle')
                    except (ValueError,KeyError) as exc:
                        raise RuntimeError(f'Cannot road-evaluate OD {task["task_id"]}:{x}->{y}') from exc
                    elapsed=start+t
                    flags=[]
                    for eid in path:
                        index=max(0,min(int(elapsed//road.dt),len(road.times)-1))
                        flags.append(int(road.observed[index,eid]))
                        elapsed+=road.dynamic_cost(eid,elapsed)
                    rows.append({'task_id':task['task_id'],'scenario_id':scenario['scenario_id'],
                                 'customer':x,'neighbor':y,'arrival_relative_sec':t,
                                 'OD_dynamic_over_static':oracle/static if static else np.nan,
                                 'feature_level':level,'feature_trend':trend,
                                 'road_path_edges':len(path),
                                 'observed_supported_edges':sum(flags),
                                 'observed_supported_share':sum(flags)/len(flags) if flags else 0.,
                                 'any_monitored_support':bool(sum(flags)),
                                 'exact_road_path_sec':dur})
            print('Audited',task['task_id'],scenario['scenario_id'],flush=True)
    if not rows:raise ValueError('No eligible scenarios')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out/'od_traffic_support.csv',index=False)
    summary={'validation_only':True,'count_od_pairs':len(df),
             'sampled_task_scenarios':int(df[['task_id','scenario_id']].drop_duplicates().shape[0]),
             'pairs_with_any_observed_support':int(df.any_monitored_support.sum()),
             'share_pairs_with_any_observed_support':float(df.any_monitored_support.mean()),
             'mean_path_observed_supported_share':float(df.observed_supported_share.mean()),
             'median_path_observed_supported_share':float(df.observed_supported_share.median()),
             'ratio_mean':float(df.OD_dynamic_over_static.mean()),
             'ratio_std':float(df.OD_dynamic_over_static.std()),
             'relative_signal_high_deviation_share':float((df.OD_dynamic_over_static.sub(1).abs()>.05).mean()),
             'semantics':'OD features use full historical future (oracle). The observed-backed edge flag denotes corridor-imputed edge costs, not direct per-OSM-edge measurements. All percentages conditional on seed_routes and selected customer neighbors.'}
    write_json(out/'support_summary.json',summary)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--output-dir',default='runs/chicago_july_v1/step5a_v2/feature_support')
    p.add_argument('--max-tasks',type=int,default=2)
    p.add_argument('--max-scenarios',type=int,default=2)
    audit(p.parse_args())

if __name__=='__main__':main()
