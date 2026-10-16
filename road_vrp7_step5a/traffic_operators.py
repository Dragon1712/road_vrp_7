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

    def __call__(self,name,routes,rng,q):
        self.calls[name]=self.calls.get(name,0)+1
        all_ids=[int(x) for rt in routes for x in rt]
        if not all_ids:return [list(r) for r in routes],[]
        if name in ('te_wr','te_local'):
            # This extra route-cost work is not counted by Stage3's obj.score_calls.
            # Track it explicitly for fair computation-cost interpretation.
            self.extra_route_cost_calls+=sum(len(r)+1 for r in routes if len(r)>1)
            components={d.customer:d for r in routes for d in removal_components(r,self.obj)}
            # Exclude single-customer routes from ranking; never empty a vehicle.
            allowed=list(components)
            if name=='te_wr':
                if self.obj.planner=='static':
                    # DO NOT use oracle traffic to rank a static plan: null control.
                    self.static_fallback_calls+=1
                    order=list(map(int,rng.permutation(allowed)))
                else:
                    nonzero=any(abs(d.externality_sec)>1e-6 for d in components.values())
                    if nonzero:self.nonzero_externality_calls+=1
                    # Rank the magnitude of the DOWNSTREAM externality. Positive
                    # improvements receive 2x weight, but large negative effects
                    # are still explored (avoids always removing last customers
                    # whose externality is identically zero). No local/total gain.
                    def impact(x):
                        e=components[x].externality_sec
                        return abs(e)+max(e,0.)
                    order=sorted(allowed,key=lambda x:(-impact(x),float(rng.random())))
                    if not nonzero:
                        order=list(map(int,rng.permutation(allowed)))
            else:
                order=sorted(allowed,key=lambda x:(-components[x].local_gain_sec,float(rng.random())))
            # Stage3 appends all IDs to ensure q removal if necessary.
            order+=all_ids
            partial,removed=_remove_order(routes,order,q)
            for x in removed:
                if x in components:
                    self.selected_externality_sum+=float(components[x].externality_sec)
                    self.selected_local_sum+=float(components[x].local_gain_sec)
                    self.selected_externality_count+=1
            return partial,removed
        if name in ('at_rr','at_timeonly'):
            if self._profiles is None:self._profiles=TrafficSignature(self.obj)
            arrivals={}
            for r in routes:
                arrival,_,_=route_trace(r,self.obj)
                arrivals.update(zip((int(x) for x in r),arrival))
            anchor=int(rng.choice(all_ids))
            ix=self.obj.index[anchor]
            # Use STATIC neighborhood scaling to make coefficients transferable.
            values=[float(self.obj.od.static[ix,self.obj.index[x]]+
                          self.obj.od.static[self.obj.index[x],ix]) for x in all_ids if x!=anchor]
            distance_scale=max(float(np.median(values)),1.) if values else 1.
            am,at=self._profiles.features(anchor,arrivals[anchor])
            ranks=[]
            for x in all_ids:
                i=self.obj.index[x]
                d=(float(self.obj.od.static[ix,i]+self.obj.od.static[i,ix])/distance_scale)
                delta_t=abs(float(arrivals[x]-arrivals[anchor]))/1200.
                delta_dem=abs(self.obj.dem[x]-self.obj.dem[anchor])/max(1,self.obj.capacity)
                if name=='at_timeonly' or self.obj.planner=='static':
                    # Arrival-time only control, no OD.dynamic lookups.
                    # Same base coefficients as AT-RR, minus only the
                    # two true traffic terms: isolates their contribution.
                    related=.35*d+.30*delta_t+.10*delta_dem
                else:
                    m,t=self._profiles.features(x,arrivals[x])
                    related=(.35*d+.30*delta_t+.10*delta_dem+
                             .15*abs(m-am)+.10*abs(t-at))
                ranks.append((related+1e-10*rng.random(),x))
            order=[x for _,x in sorted(ranks)]
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
                'extra_traffic_route_cost_calls':self.extra_route_cost_calls}
