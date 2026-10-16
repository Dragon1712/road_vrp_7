#!/usr/bin/env python3
"""Step5-A traffic-aware destroy operators for July TD-CVRP30.

Compatible with road_vrp7_stage3 Objective/OD and solvers.choose_destroy.
No geospatial ground truth is assumed. The oracle planner uses retrospective
historical OD labels ONLY, never deployment-time predictions.

TE-WR specifically ranks DOWNSTREAM time externality (total removal gain
minus local adjacent-arc saving), not total removal gain (Stage3 'worst').
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
import numpy as np


@dataclass(frozen=True)
class RemovalComponent:
    customer: int
    local_gain_sec: float
    externality_sec: float
    total_gain_sec: float


def route_trace(route, obj):
    """Return (arrival times, times after service, old route travel cost).

    Time coordinates are seconds since scenario departure.  The depot has index 0.
    This matches Stage3's Objective.route_cost and excludes service from travel.
    """
    t=0.;travel=0.;prev=0
    arrives=[];departures=[]
    for customer in route:
        nxt=obj.index[int(customer)]
        leg=obj.od.travel(prev,nxt,t,obj.planner)
        t+=leg;travel+=leg
        arrives.append(float(t))
        t+=float(obj.task['service_seconds'])
        departures.append(float(t))
        prev=nxt
    travel+=obj.od.travel(prev,0,t,obj.planner)
    return arrives,departures,float(travel)


def removal_components(route,obj):
    """Exact additive decomposition of Stage3 surrogate *travel* saving.

    total_gain = C(route)-C(route without customer)
    local_gain = c(prev,x;t_prev)+c(x,next;t_x_leave)-c(prev,next;t_prev)
    externality = total_gain-local_gain.

    When x is final customer, downstream externality is zero. For time-
    independent costs all externalities should be zero (floating tolerance).
    Do not use ex-post road replay here: that would leak its high-cost signal
    into the solver and invalidate comparison to Stage3 OD objective.
    """
    route=list(map(int,route))
    if len(route)<=1:return []
    arrivals,departures,old=route_trace(route,obj)
    # Verify against inherited cost; catches accidental time-accounting drift.
    inherited=float(obj.route_cost(route))
    if not math.isclose(old,inherited,rel_tol=1e-9,abs_tol=1e-6):
        raise AssertionError('Step5 route trace differs from Stage3 route_cost')
    result=[]
    for i,x in enumerate(route):
        previous=0 if i==0 else obj.index[route[i-1]]
        following=0 if i+1==len(route) else obj.index[route[i+1]]
        before=0. if i==0 else departures[i-1]
        local_old=(obj.od.travel(previous,obj.index[x],before,obj.planner)+
                   obj.od.travel(obj.index[x],following,departures[i],obj.planner))
        local_new=obj.od.travel(previous,following,before,obj.planner)
        local=float(local_old-local_new)
        after=float(obj.route_cost(route[:i]+route[i+1:]))
        total=float(old-after)
        ext=float(total-local)
        if i==len(route)-1 and abs(ext)>1e-6:
            raise AssertionError('Unexpected externality on final customer')
        result.append(RemovalComponent(x,local,ext,total))
    return result


def _remove_order(routes,ordered,q):
    """Same exact-K and partial-route feasibility rules as Stage3 destroy."""
    working=[list(map(int,r)) for r in routes]
    n=sum(map(len,working))
    q=min(int(q),n-len(working))
    if q<1:return working,[]
    removed=[]
    for x in ordered:
        if len(removed)>=q:break
        for rt in working:
            if x in rt and len(rt)>1:
                rt.remove(x);removed.append(int(x));break
    if len(removed)!=q:raise AssertionError(f'Unable to remove {q} customers with nonempty routes')
    return working,removed


class TrafficSignature:
    """Lagged congestion level/trend over 4 STATIC nearest OD neighbors.

    Lazy precomputation occurs only when AT-RR is selected, INSIDE search time.
    Base normalization relies on OD.static.  No dynamic tensor access is made
    for the static planner.  OD.dynamic is 20min sampled, mostly OSM fallbacks;
    signatures are NOT equivalent to fully observed traffic.
    """
    def __init__(self,obj,n_neighbors=4,lag_sec=1200.):
        self.obj=obj;self.lag=float(lag_sec)
        self.ids=sorted(obj.dem)
        self.nidx={x:obj.index[x] for x in self.ids}
        self.neighbors={}
        for x in self.ids:
            i=obj.index[x]
            options=[(float(obj.od.static[i,obj.index[y]]+obj.od.static[obj.index[y],i]),int(y))
                     for y in self.ids if y!=x and
                     float(obj.od.static[i,obj.index[y]]+obj.od.static[obj.index[y],i])>0]
            self.neighbors[x]=[y for _,y in sorted(options)[:n_neighbors]]
        self.levels=None
        if obj.planner=='oracle':
            # Use only explicitly authorized hindsight-Oracle planner.
            dynamic=obj.od.dynamic
            levels={}
            for x in self.ids:
                i=obj.index[x]
                ratios=[]
                for y in self.neighbors[x]:
                    j=obj.index[y]
                    static=float(obj.od.static[i,j])
                    if static>1e-10:
                        ratios.append(np.maximum(0.01,np.asarray(dynamic[:,i,j],dtype=float))/static)
                levels[x]=np.mean(ratios,axis=0) if ratios else np.ones(len(obj.od.knot_seconds))
            self.levels=levels
        elif obj.planner!='static':
            raise ValueError('Unknown planner '+str(obj.planner))

    def _level(self,x,t):
        if self.obj.planner=='static':return 1.
        knots=self.obj.od.knot_seconds
        return float(np.interp(min(max(float(t),0.),float(knots[-1])),knots,self.levels[x]))

    def features(self,x,arrive):
        if self.obj.planner=='static':return 1.,0.
        now=self._level(x,arrive)
        later=self._level(x,arrive+self.lag)
        return now,(later-now)


class TrafficDestroy:
    """Per-(task, scenario, planner) operator engine: no global mutable state."""
    def __init__(self,obj):
        self.obj=obj
        self._profiles=None
        self.calls={}
        self.selected_externality_sum=0.
        self.selected_local_sum=0.
        self.selected_externality_count=0
        self.nonzero_externality_calls=0
        self.static_fallback_calls=0
        self.extra_route_cost_calls=0
        # Mechanism effectiveness (not solution quality): matched-shadow rankings.
        self.at_shadow_compared=0
        self.at_shadow_changed=0
        self.at_shadow_jaccard_sum=0.
        self.at_feature_totals={'distance':0.,'eta':0.,'demand':0.,'level':0.,'trend':0.}
        self.at_feature_items=0
        self.te_shadow_compared=0
        self.te_shadow_changed=0
        self.te_shadow_jaccard_sum=0.
        self.te_positive_count=0
        self.te_negative_count=0

    def __call__(self,name,routes,rng,q):
        self.calls[name]=self.calls.get(name,0)+1
        all_ids=[int(x) for rt in routes for x in rt]
        if not all_ids:return [list(r) for r in routes],[]
        if name in ('te_wr','te_local','te_positive','te_static_local'):
            # Stage3 dynamic objective for the *original* methods. Static-local
            # is a strict non-traffic comparison: no dynamic OD queries.
            if name=='te_static_local':
                ranks={}
                for route in routes:
                    if len(route)<=1:continue
                    for j,x in enumerate(route):
                        prev=0 if j==0 else self.obj.index[int(route[j-1])]
                        idx=self.obj.index[int(x)]
                        nxt=0 if j+1==len(route) else self.obj.index[int(route[j+1])]
                        costs=self.obj.od.static
                        ranks[int(x)]=float(costs[prev,idx]+costs[idx,nxt]-costs[prev,nxt])
                allowed=list(ranks)
                jit={x:float(rng.random()) for x in allowed}
                order=sorted(allowed,key=lambda x:(-ranks[x],jit[x]))
                order+=all_ids
                return _remove_order(routes,order,q)
            self.extra_route_cost_calls+=sum(len(r)+1 for r in routes if len(r)>1)
            components={d.customer:d for r in routes for d in removal_components(r,self.obj)}
            allowed=list(components)
            jitter={x:float(rng.random()) for x in allowed}
            if name=='te_wr':
                if self.obj.planner=='static':
                    self.static_fallback_calls+=1
                    order=list(map(int,rng.permutation(allowed)))
                else:
                    nonzero=any(abs(d.externality_sec)>1e-6 for d in components.values())
                    if nonzero:self.nonzero_externality_calls+=1
                    def impact(x):
                        e=components[x].externality_sec
                        return abs(e)+max(e,0.)
                    order=sorted(allowed,key=lambda x:(-impact(x),jitter[x]))
                    if not nonzero:order=list(map(int,rng.permutation(allowed)))
            elif name=='te_positive':
                # Diagnostic alternative: only positive downstream relief matters.
                order=sorted(allowed,key=lambda x:(-max(components[x].externality_sec,0.),jitter[x]))
            else:
                order=sorted(allowed,key=lambda x:(-components[x].local_gain_sec,jitter[x]))
            # Compare under the SAME state and count with Stage3 total-worst.
            if name in ('te_wr','te_positive') and self.obj.planner=='oracle':
                base=sorted(allowed,key=lambda x:(-components[x].total_gain_sec,jitter[x]))
                main=_remove_order(routes,order+all_ids,q)[1]
                alt=_remove_order(routes,base+all_ids,q)[1]
                union=set(main)|set(alt)
                self.te_shadow_compared+=1
                self.te_shadow_changed+=int(set(main)!=set(alt))
                self.te_shadow_jaccard_sum+=len(set(main)&set(alt))/max(1,len(union))
            order+=all_ids
            partial,removed=_remove_order(routes,order,q)
            for x in removed:
                if x in components:
                    e=float(components[x].externality_sec)
                    self.selected_externality_sum+=e
                    self.selected_local_sum+=float(components[x].local_gain_sec)
                    self.selected_externality_count+=1
                    if e>1e-6:self.te_positive_count+=1
                    elif e< -1e-6:self.te_negative_count+=1
            return partial,removed
        if name in ('at_rr','at_timeonly','at_space','at_misaligned'):
            if self._profiles is None and name not in ('at_space','at_timeonly'):
                self._profiles=TrafficSignature(self.obj)
            arrivals={}
            # This is the CURRENT solver arrival time, not GPS measurement.
            if name!='at_space':
                for r in routes:
                    arrival,_,_=route_trace(r,self.obj)
                    arrivals.update(zip((int(x) for x in r),arrival))
            anchor=int(rng.choice(all_ids));ix=self.obj.index[anchor]
            values=[float(self.obj.od.static[ix,self.obj.index[x]]+
                          self.obj.od.static[self.obj.index[x],ix]) for x in all_ids if x!=anchor]
            distance_scale=max(float(np.median(values)),1.) if values else 1.
            am,at=(1.,0.) if name in ('at_space','at_timeonly') else self._profiles.features(anchor,arrivals[anchor])
            noise={x:1e-10*float(rng.random()) for x in all_ids}
            items=[]
            for x in all_ids:
                i=self.obj.index[x]
                d=float(self.obj.od.static[ix,i]+self.obj.od.static[i,ix])/distance_scale
                eta=0. if name=='at_space' else abs(float(arrivals[x]-arrivals[anchor]))/1200.
                demand=abs(self.obj.dem[x]-self.obj.dem[anchor])/max(1,self.obj.capacity)
                m,t=(1.,0.) if name in ('at_space','at_timeonly') else self._profiles.features(x,arrivals[x])
                if name=='at_misaligned' and self.obj.planner=='oracle':
                    # Misalign customer traffic identity deterministically (not
                    # temporal future access on STATIC mode). Negative control.
                    ids=sorted(self.obj.dem);other=ids[(ids.index(x)+len(ids)//2)%len(ids)]
                    m,t=self._profiles.features(other,arrivals[x])
                    ma,ta=self._profiles.features(ids[(ids.index(anchor)+len(ids)//2)%len(ids)],arrivals[anchor])
                else:ma,ta=am,at
                pieces={'distance':.35*d,'eta':.30*eta,'demand':.10*demand,
                        'level':.15*abs(m-ma),'trend':.10*abs(t-ta)}
                base=pieces['distance']+pieces['eta']+pieces['demand']
                score=base
                if name in ('at_rr','at_misaligned') and self.obj.planner=='oracle':
                    score+=pieces['level']+pieces['trend']
                if name=='at_rr' and self.obj.planner=='oracle':
                    for key,v in pieces.items():self.at_feature_totals[key]+=v
                    self.at_feature_items+=1
                items.append((x,base,score,noise[x]))
            order=[x for x,_,_,_ in sorted(items,key=lambda z:(z[2]+z[3],z[0]))]
            if name=='at_rr' and self.obj.planner=='oracle':
                control=[x for x,_,_,_ in sorted(items,key=lambda z:(z[1]+z[3],z[0]))]
                main=_remove_order(routes,order,q)[1]
                alt=_remove_order(routes,control,q)[1]
                self.at_shadow_compared+=1
                self.at_shadow_changed+=int(set(main)!=set(alt))
                self.at_shadow_jaccard_sum+=len(set(main)&set(alt))/max(1,len(set(main)|set(alt)))
            return _remove_order(routes,order,q)
        raise ValueError('Unknown traffic destroy operator '+name)

    def diagnostics(self):
        n=self.selected_externality_count
        return {'call_counts':dict(self.calls),
                'te_nonzero_externality_calls':self.nonzero_externality_calls,
                'te_static_fallback_calls':self.static_fallback_calls,
                'selected_mean_externality_sec':self.selected_externality_sum/n if n else 0.,
                'selected_mean_local_gain_sec':self.selected_local_sum/n if n else 0.,
                'selected_components':n,
                'traffic_signature_generated':self._profiles is not None,
                'extra_traffic_route_cost_calls':self.extra_route_cost_calls,
                'at_shadow_compared':self.at_shadow_compared,
                'at_shadow_changed':self.at_shadow_changed,
                'at_shadow_change_rate':self.at_shadow_changed/max(1,self.at_shadow_compared),
                'at_shadow_mean_jaccard':self.at_shadow_jaccard_sum/max(1,self.at_shadow_compared),
                'at_feature_mean_parts':{k:v/max(1,self.at_feature_items) for k,v in self.at_feature_totals.items()},
                'te_shadow_compared':self.te_shadow_compared,
                'te_shadow_changed':self.te_shadow_changed,
                'te_shadow_change_rate':self.te_shadow_changed/max(1,self.te_shadow_compared),
                'te_shadow_mean_jaccard':self.te_shadow_jaccard_sum/max(1,self.te_shadow_compared),
                'selected_positive_externality':self.te_positive_count,
                'selected_negative_externality':self.te_negative_count}
