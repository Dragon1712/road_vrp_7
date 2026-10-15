#!/usr/bin/env python3
"""Generate reproducible 30-customer 5-truck synthetic orders on the July road.
Orders are SYNTHETIC; Chicago CSV does not contain delivery demand or GPS trips.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from road_vrp import Road,legal,sha256,write_json


def scenarios_from_qc(path,include_ineligible=True):
    df=pd.read_csv(path,keep_default_na=False)
    required={'scenario_id','departure_local','date','split','eligible','edge_mean'}
    if not required.issubset(df.columns):raise ValueError('QC scenario_quality missing '+str(required-set(df.columns)))
    dates=pd.to_datetime(df.date,errors='raise')
    if dates.min()!=pd.Timestamp('2025-07-02') or dates.max()!=pd.Timestamp('2025-07-31'):
        raise ValueError('QC dates outside July 2-31 or incomplete period')
    if df.duplicated('scenario_id').any() or len(df)!=60:raise ValueError('Expected 60 unique July morning/evening scenarios')
    expected_times={f'{d:%Y-%m-%d}_{h}' for d in pd.date_range('2025-07-02','2025-07-31',freq='D') for h in ('0800','1700')}
    if set(df.scenario_id)!=expected_times:raise ValueError('QC is missing/duplicating a required July morning/evening time slot')
    expected_splits={'train':('2025-07-02','2025-07-21'),'validation':('2025-07-22','2025-07-25'),'test':('2025-07-26','2025-07-31')}
    for _,r in df.iterrows():
        if not any(a<=r.date<=b and r.split==s for s,(a,b) in expected_splits.items()):raise ValueError('Non-frozen split for '+r.scenario_id)
        if r.departure_local[:10]!=r.date:raise ValueError('QC departure day mismatch')
    rows=[]
    for r in df.itertuples():
        eligible=str(r.eligible).lower() in ('true','1','yes')
        if not include_ineligible and not eligible:continue
        rows.append({'scenario_id':str(r.scenario_id),'departure_local':str(r.departure_local),
                     'date':str(r.date),'split':str(r.split),'eligible':eligible,
                     'edge_observation_ratio':float(r.edge_mean),'rejection_reasons':str(r.rejection_reasons) if hasattr(r,'rejection_reasons') else ''})
    return rows


def make_tasks(road, *, scenario_quality, output_dir, num_tasks=10, num_customers=30,
               vehicles=5,capacity=45,seed=20261015,corridor_fraction=.67,radius_m=600):
    output_dir=Path(output_dir)
    if output_dir.exists() and list(output_dir.iterdir()):raise FileExistsError('Refusing to overwrite nonempty task directory: '+str(output_dir))
    if num_tasks<10 or num_customers<vehicles or vehicles<1 or capacity<1 or not 0<=corridor_fraction<=1:
        raise ValueError('Invalid experiment size; require 10+ tasks and corridor_fraction within [0,1]')
    scenarios=scenarios_from_qc(scenario_quality)
    g=nx.DiGraph();lengths=road.edges.length_m.to_numpy(float)
    for e,r in enumerate(road.edges.itertuples()):
        w=lengths[e]
        if not g.has_edge(r.osm_u,r.osm_v) or w<g[r.osm_u][r.osm_v]['weight']:
            g.add_edge(r.osm_u,r.osm_v,weight=float(w))
    scc=max(nx.strongly_connected_components(g),key=len)
    sub=g.subgraph(scc).copy()
    if len(sub)<num_customers+1:raise ValueError('Largest road SCC too small')
    candidates=sorted(scc)
    roots=sorted(set(road.mapping_nodes)&scc)
    if not roots:raise ValueError('No mapped corridors in largest SCC')
    und=sub.to_undirected()
    corr_d=nx.multi_source_dijkstra_path_length(und,roots,cutoff=float(radius_m),weight='weight')
    near=sorted(set(corr_d))
    if len(near)<min(num_customers+1,int(num_customers*corridor_fraction)):raise ValueError('Not enough mapped-road nearby nodes; increase radius')
    # Static centrality proxy: depot is a node of the largest SCC near mapped edges,
    # chosen deterministically. NetworkX betweenness on 2K vertices is expensive.
    depot=roots[len(roots)//2]
    near=[x for x in near if x!=depot];far=[x for x in candidates if x!=depot]
    num_near=int(round(num_customers*corridor_fraction))
    rng=np.random.default_rng(seed)
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'source':'road_vrp_7 July stage2 only','orders':'synthetic','seed':seed,
              'parameters':{'num_tasks':num_tasks,'num_customers':num_customers,
                            'vehicles':vehicles,'capacity':capacity,'corridor_fraction':corridor_fraction,
                            'radius_m':radius_m},
              'cost_dir':str(road.folder),'cost_edge_index_sha256':sha256(road.folder/'edge_index.csv'),
              'cost_edge_npz_sha256':sha256(road.folder/'edge_costs.npz'),
              'qc_sha256':sha256(scenario_quality),'splits':{a:sum(x['split']==a for x in scenarios) for a in ('train','validation','test')},
              'scenario_count':len(scenarios),'task_ids':[], 'largest_scc_nodes':len(sub),
              'mapped_near_candidate_nodes':len(near)}
    try:
        for task_i in range(num_tasks):
            picked=list(rng.choice(near,size=num_near,replace=False))
            free=[x for x in far if x not in picked]
            picked+=list(rng.choice(free,size=num_customers-num_near,replace=False))
            rng.shuffle(picked)
            # Six customers per truck for the standard 30/5 scenario; reject-demand
            # sampling ensures the seed is feasible without changing customer IDs.
            for trial in range(100000):
                demands=rng.integers(2,10,size=num_customers).astype(int)
                groups=np.array_split(np.arange(num_customers),vehicles)
                if all(int(demands[g].sum())<=capacity for g in groups):break
            else:raise RuntimeError('Unable to sample feasible loads')
            seed_routes=[[int(j)+1 for j in g] for g in groups]
            task={'task_id':f'task_{task_i:04d}','depot_node':str(depot),
                  'customers':[{'id':i+1,'node':str(picked[i]),'demand':int(demands[i])}
                               for i in range(num_customers)],
                  'vehicles':vehicles,'capacity':capacity,'service_seconds':120,
                  'require_all_vehicles':True,'seed_routes':seed_routes,'scenarios':scenarios,
                  'order_source':'synthetic - no delivery trajectories in Chicago traffic tracker',
                  'generation_seed':seed,'task_index':task_i}
            legal(task,seed_routes)
            p=output_dir/(task['task_id']+'.json')
            write_json(p,task);manifest['task_ids'].append(task['task_id'])
        write_json(output_dir/'manifest.json',manifest)
    except Exception:
        # Leave diagnostics but never present a partial manifest as success.
        (output_dir/'GENERATION_FAILED.txt').write_text('Incomplete generation. Delete this folder before rerun.\n')
        raise
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--scenario-quality',default='runs/chicago_july_v1/data_qc/scenario_quality.csv')
    p.add_argument('--output-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--num-tasks',type=int,default=10)
    p.add_argument('--num-customers',type=int,default=30)
    p.add_argument('--vehicles',type=int,default=5)
    p.add_argument('--capacity',type=int,default=45)
    p.add_argument('--seed',type=int,default=20261015)
    p.add_argument('--corridor-fraction',type=float,default=.67)
    p.add_argument('--radius-m',type=float,default=600)
    a=p.parse_args()
    res=make_tasks(Road(a.cost_dir),scenario_quality=a.scenario_quality,output_dir=a.output_dir,
                   num_tasks=a.num_tasks,num_customers=a.num_customers,vehicles=a.vehicles,
                   capacity=a.capacity,seed=a.seed,corridor_fraction=a.corridor_fraction,radius_m=a.radius_m)
    print(json.dumps(res,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
