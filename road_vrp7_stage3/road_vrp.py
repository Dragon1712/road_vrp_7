#!/usr/bin/env python3
"""July stage3 TD-CVRP primitives. Pure Python/numpy/pandas; no old repository imports.

Road cost labels are mixed historical Chicago corridor observations plus static
unobserved fallback. Oracle planning is retrospective ONLY.
"""
from __future__ import annotations
import hashlib
import heapq
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

class Road:
    def __init__(self,cost_dir):
        self.folder=Path(cost_dir)
        self.edges=pd.read_csv(self.folder/'edge_index.csv',dtype={'osm_u':str,'osm_v':str,'osm_key':str},keep_default_na=False)
        self.times=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(self.folder/'time_index.csv').time_local))
        if not self.times.is_monotonic_increasing or self.times.has_duplicates or len(self.times)<2:raise ValueError('Invalid time index')
        self.start=self.times[0]
        self.dt=float((self.times[1]-self.times[0]).total_seconds())
        if self.dt<=0 or not np.all(np.diff(self.times.asi8)==int(self.dt*1e9)):raise ValueError('Nonuniform time grid')
        with np.load(self.folder/'edge_costs.npz',allow_pickle=False) as z:
            self.fifo=z['travel_sec_fifo']  # Float32 stage2 tensor avoids extra 190MB copy
            raw_shape=z['travel_sec_raw'].shape
            self.observed=z['chicago_observed_mask'].astype(bool)
            self.mapped=z['is_mapped_edge'].astype(bool)
        n=len(self.edges); t=len(self.times)
        if self.fifo.shape!=(t,n) or raw_shape!=(t,n) or self.observed.shape!=(t,n) or self.mapped.shape!=(n,):raise ValueError('Stage2 tensor shape mismatch')
        if self.edges.edge_idx.tolist()!=list(range(n)) or self.edges.duplicated(['osm_u','osm_v','osm_key']).any():raise ValueError('Invalid edge_idx or duplicate road keys')
        if not np.all(np.isfinite(self.fifo)) or np.min(self.fifo)<=0 or np.any(self.observed[:,~self.mapped]):raise ValueError('Invalid FIFO or observed mask')
        self.static=self.edges.static_travel_sec.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(self.static)) or np.min(self.static)<=0:raise ValueError('Invalid static costs')
        self.adj={};self.vertices=set()
        for row in self.edges.itertuples():
            self.adj.setdefault(row.osm_u,[]).append((row.osm_v,int(row.edge_idx)))
            self.vertices.update([row.osm_u,row.osm_v])
        self.mapping_nodes=set(self.edges.loc[self.mapped,'osm_u'].astype(str))|set(self.edges.loc[self.mapped,'osm_v'].astype(str))
        self._cache={}
        self.cost_file_hash=sha256(self.folder/'edge_costs.npz')
        self.edge_file_hash=sha256(self.folder/'edge_index.csv')
    def seconds(self,date):
        return (pd.Timestamp(date)-self.start).total_seconds()
    def dynamic_cost(self,edge,t):
        if t<0 or t>=len(self.times)*self.dt:raise ValueError(f'TD road time beyond July window: {t}')
        k=min(int(t//self.dt),len(self.times)-1)
        if k==len(self.times)-1:return float(self.fifo[k,edge])
        alpha=(t-k*self.dt)/self.dt
        # interpolate ARRIVAL, not time-dependent speed or separate FIFO adjusted data
        a0=k*self.dt+self.fifo[k,edge]
        a1=(k+1)*self.dt+self.fifo[k+1,edge]
        return max(0.,float((1-alpha)*a0+alpha*a1-t))
    def _shortest(self,origin,depart,targets,mode):
        if origin not in self.vertices:raise ValueError('Road origin not in OSM graph: '+str(origin))
        remaining=set(targets)-{origin}
        d={origin:float(depart)};back={}; pq=[(float(depart),origin)]
        while pq and remaining:
            at,u=heapq.heappop(pq)
            if at>d[u]+1e-8:continue
            remaining.discard(u)
            for v,e in self.adj.get(u,()):
                c=self.static[e] if mode=='static' else self.dynamic_cost(e,at)
                nt=at+c
                if nt<d.get(v,float('inf'))-1e-8:
                    d[v]=nt;back[v]=(u,e)
                    heapq.heappush(pq,(nt,v))
        if remaining:raise ValueError(f'Unreachable road targets from {origin}: {list(remaining)[:4]}')
        return d,back
    def sssp(self,origin,targets,depart,mode='oracle'):
        return self._shortest(str(origin),float(depart),{str(x) for x in targets},'static' if mode=='static' else 'oracle')
    def path(self,src,dst,at,mode='oracle'):
        src=str(src);dst=str(dst)
        if src==dst:return 0.,[]
        d,prev=self.sssp(src,[dst],at,mode)
        edges=[];u=dst
        while u!=src:
            p,e=prev[u];edges.append(e);u=p
        edges.reverse()
        return float(d[dst]-at),edges
    def replay(self,task,routes,depart):
        legal(task,routes)
        begin=self.seconds(depart)
        total=0.;vehicle=[];num_edges=0;num_obs=0
        nodes={int(c['id']):str(c['node']) for c in task['customers']}
        depot=str(task['depot_node'])
        for route in routes:
            t=begin;at_node=depot;dist=0.;obs=0;cnt=0
            for cid in list(route)+[None]:
                dest=depot if cid is None else nodes[cid]
                dur,path=self.path(at_node,dest,t,'oracle')
                for edge in path:
                    slot=min(int(t//self.dt),len(self.times)-1)
                    obs+=int(self.observed[slot,edge]);cnt+=1
                    t+=self.dynamic_cost(edge,t)
                dist+=dur;at_node=dest
                if cid is not None:t+=float(task['service_seconds'])
            total+=dist; num_edges+=cnt;num_obs+=obs
            vehicle.append({'road_travel_sec':float(dist),'elapsed_sec':float(t-begin),
                            'observed_edge_traversals':obs,'edge_traversals':cnt})
        return {'ex_post_road_travel_sec':float(total),
                'ex_post_total_vehicle_elapsed_sec':float(sum(v['elapsed_sec'] for v in vehicle)),
                'observed_edge_share':num_obs/max(num_edges,1),
                'num_vehicles_used':sum(bool(r) for r in routes),'vehicle_loads':loads(task,routes),
                'vehicle_details':vehicle}


def loads(task,routes):
    dem={int(c['id']):int(c['demand']) for c in task['customers']}
    return [sum(dem[i] for i in r) for r in routes]

def legal(task,routes):
    if len(routes)!=int(task['vehicles']):raise ValueError('Wrong route count')
    actual=[int(x) for route in routes for x in route]
    expected=[int(x['id']) for x in task['customers']]
    if sorted(actual)!=sorted(expected):raise ValueError('Duplicated/missing/unknown customers')
    if task.get('require_all_vehicles',True) and any(len(r)==0 for r in routes):raise ValueError('Empty route with exact K rule')
    l=loads(task,routes)
    if any(x>int(task['capacity']) for x in l):raise ValueError(f'Over capacity: {l}')
    return True

class OD:
    def __init__(self,road,task,scenario,*,step_min=20,hours=5,cache_dir=None):
        self.road=road;self.task=task;self.scenario=scenario
        self.nodes=[str(task['depot_node'])]+[str(c['node']) for c in task['customers']]
        self.n=len(self.nodes)
        self.start=road.seconds(scenario['departure_local'])
        self.knot_seconds=np.arange(0,int(round(hours*60/step_min))+1,dtype=np.float64)*(step_min*60)
        if (self.start+self.knot_seconds[-1]+3600 > len(road.times)*road.dt):raise ValueError('Scenario near data end; need horizon+buffer')
        cache_dir=Path(cache_dir) if cache_dir else None
        key_data={'cost_sha256':road.cost_file_hash,
                  'edge_sha256':road.edge_file_hash,
                  'task_nodes':self.nodes,'departure':scenario['departure_local'],
                  'step_min':step_min,'hours':hours}
        # Hash is based on BOTH graph cost bytes and the task nodes, not file mtimes.
        key=hashlib.sha256(json.dumps(key_data,sort_keys=True).encode()).hexdigest()
        self.cache_hit=False
        cache=(cache_dir/f'{key}.npz') if cache_dir else None
        if cache is not None and cache.exists():
            with np.load(cache,allow_pickle=False) as z:
                self.static=z['static'].astype(np.float64)
                self.dynamic=z['dynamic'].astype(np.float64)
                if self.dynamic.shape!=(len(self.knot_seconds),self.n,self.n) or self.static.shape!=(self.n,self.n):raise ValueError('Corrupt OD cache')
            self.cache_hit=True
            return
        self.static=np.zeros((self.n,self.n),dtype=np.float64)
        self.dynamic=np.zeros((len(self.knot_seconds),self.n,self.n),dtype=np.float64)
        target=set(self.nodes)
        for i,node in enumerate(self.nodes):
            ds,_=road.sssp(node,target,self.start,'static')
            for j,t in enumerate(self.nodes):self.static[i,j]=ds[t]-self.start
        for k,dt in enumerate(self.knot_seconds):
            at=self.start+dt
            for i,node in enumerate(self.nodes):
                ds,_=road.sssp(node,target,at,'oracle')
                for j,t in enumerate(self.nodes):self.dynamic[k,i,j]=ds[t]-at
        if not np.all(np.isfinite(self.dynamic)) or np.min(self.dynamic)<-1e-8:raise ValueError('Invalid OD')
        if cache:
            cache_dir.mkdir(parents=True,exist_ok=True)
            import os,tempfile
            fd,name=tempfile.mkstemp(prefix='od_',suffix='.npz',dir=cache_dir)
            os.close(fd)
            try:
                np.savez_compressed(name,static=self.static,dynamic=self.dynamic)
                os.replace(name,cache)
            finally:
                if os.path.exists(name):os.unlink(name)
    def travel(self,i,j,sec,planner):
        if i==j:return 0.
        if planner=='static':return float(self.static[i,j])
        if planner!='oracle':raise ValueError(planner)
        if sec<0:raise ValueError('Negative relative travel time')
        if sec>self.knot_seconds[-1]+1e-6:raise ValueError('VRP plan exceeds OD horizon; increase --od-hours')
        hi=min(int(np.searchsorted(self.knot_seconds,sec,side='right')),len(self.knot_seconds)-1)
        lo=max(0,hi-1)
        if hi==lo:return float(self.dynamic[lo,i,j])
        a=(sec-self.knot_seconds[lo])/(self.knot_seconds[hi]-self.knot_seconds[lo])
        return float((1-a)*self.dynamic[lo,i,j]+a*self.dynamic[hi,i,j])

class Objective:
    def __init__(self,task,od,planner):
        self.task=task;self.od=od;self.planner=planner
        self.index={int(c['id']):i+1 for i,c in enumerate(task['customers'])}
        self.dem={int(c['id']):int(c['demand']) for c in task['customers']}
        self.capacity=int(task['capacity']);self.score_calls=0
    def route_cost(self,route):
        t=0.;travel=0.;prev=0
        for cid in list(route)+[None]:
            end=0 if cid is None else self.index[int(cid)]
            c=self.od.travel(prev,end,t,self.planner)
            t+=c;travel+=c
            if cid is not None:t+=float(self.task['service_seconds'])
            prev=end
        return float(travel)
    def cost(self,routes):
        self.score_calls+=1
        return float(sum(self.route_cost(r) for r in routes))
    def feasible(self,routes,full=True):
        if len(routes)!=int(self.task['vehicles']):return False
        actual=[x for route in routes for x in route]
        if len(actual)!=len(set(actual)) or any(x not in self.dem for x in actual):return False
        if full and set(actual)!=set(self.dem):return False
        if full and self.task.get('require_all_vehicles',True) and any(not r for r in routes):return False
        return all(sum(self.dem[x] for x in r)<=self.capacity for r in routes)
