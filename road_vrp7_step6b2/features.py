"""Section-removal and STATIC proxy insertion-boundary features.

NO Oracle/TD evaluation, no labels and no future observations. The insertion target
is the first ranked *relocation* candidate returned by the SAME frozen v3
static-proxy option generator. This is NOT the realized Static Repair position,
which may be a swap and is purposely NOT read from the counterfactual record.
"""
from __future__ import annotations
import math, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for p in ('road_vrp7_step5b_v3','road_vrp7_step6a','road_vrp7_step6b1_v2','road_vrp7_stage3'):
    x=str(ROOT/p)
    if x not in sys.path:sys.path.insert(0,x)

CORE=('original_static_cost_sec','departure_hour_local','departure_weekday_local',
      'route_index','position','size','block_demand','route_load','vehicle_capacity',
      'static_section_cost_sec','route_length','relative_position')
OLD_HIST=('history_observed_share','history_observed_rel_cost','history_has_observation')+tuple(f'history_coverage_tminus_{i}' for i in range(6,0,-1))+tuple(f'history_rel_cost_tminus_{i}' for i in range(6,0,-1))
GEOMETRY=('remove_static_saved_sec','insert_proxy_static_added_sec','proxy_static_net_gain_sec',
          'target_route_index','target_position','target_route_length',
          'target_route_load','target_route_slack','same_route','source_arrival_static_sec',
          'target_arrival_static_sec','arrival_offset_static_sec','target_orientation_reversed','orientation_internal_delta_static_sec')
TRAFFIC=('remove_history_saved_sec','insert_history_added_sec','proxy_history_net_gain_sec',
         'removal_boundary_rel_cost','insertion_boundary_rel_cost',
         'removal_history_latest_rel','insertion_history_latest_rel',
         'removal_history_trend','insertion_history_trend','orientation_internal_delta_history_sec')
RELIABILITY=('removal_observed_share','insertion_observed_share','removal_observed_fraction',
             'insertion_observed_fraction','removal_all_missing','insertion_all_missing',
             'boundary_observed_min','boundary_observed_gap')
B2_SUFFIX=GEOMETRY+TRAFFIC+RELIABILITY
FEATURE_MODES={
 'F0': CORE,
 'F1': CORE+OLD_HIST,
 'F2': CORE+GEOMETRY,
 'F3': CORE+GEOMETRY+TRAFFIC,
 'F4': CORE+GEOMETRY+TRAFFIC+RELIABILITY,
 'F5': CORE+GEOMETRY+TRAFFIC+RELIABILITY,
}

def _od(obj,cid):return 0 if cid==0 else int(obj.index[int(cid)])

def _arc_static(obj,u,v):return float(obj.od.static[_od(obj,u),_od(obj,v)])

def _directed_path(obj,customers):
    return [0]+list(map(int,customers))+[0]

def _prefix_static(obj,route,pos):
    if not 0<=pos<=len(route):raise ValueError('Invalid prefix')
    t=0.; last=0
    for cid in route[:pos]:
        t+=_arc_static(obj,last,cid)+float(obj.task['service_seconds']);last=cid
    return t

def _summary(hist,obj,edges):
    """Expected-edge cost uses static fallback when NO real historical coverage.

    Difference of costs is computed on the SAME directed boundary arcs, so it
    respects asymmetry. Missing values are never filled with future observations.
    """
    slots=hist.n_slots
    weighted=np.zeros(slots,float);observed=np.zeros(slots,float)
    totals=np.zeros(slots,float);edge_n=0
    latest=[];time_means=[];per_edge_cov=[]
    for u,v,sign in edges:
        i,j=_od(obj,u),_od(obj,v)
        st=_arc_static(obj,u,v)
        info=hist.path_features(i,j)
        cov=np.asarray(info['covered_fraction'],float)
        rel=np.asarray(info['observed_rel_cost'],float)
        if len(cov)!=slots or len(rel)!=slots:raise ValueError('Historical slot mismatch')
        # Ratio from REAL edges only, uncovered part takes STATIC fallback.
        ratio=1.+cov*(rel-1.)
        totals+=float(sign)*st*ratio
        w=max(0.,int(info['edge_count']))
        weighted+=max(w,1)*cov*rel
        observed+=max(w,1)*cov
        edge_n+=max(w,1)
        per_edge_cov.append(float(np.mean(cov)))
        latest.append(float(ratio[-1]));time_means.append(float(np.mean(ratio)))
    if not edges:raise ValueError('Empty boundary set')
    # The weighted signed difference may be negative; this is intentional.
    mean=float(np.mean(totals)); coverage=float(np.mean(observed/max(edge_n,1)))
    obs_fraction=float(np.mean([x>0 for x in per_edge_cov]))
    # Signed arcs cannot be used as weights for coverage or reliability.
    return {'delta':mean,'rel':float(np.mean(time_means)),
            'latest':float(np.mean(latest)),
            'trend':float(totals[-1]-totals[0]),
            'coverage':coverage,'fraction':obs_fraction,'missing':float(coverage==0.),
            'edge_min':min(per_edge_cov) if per_edge_cov else 0.}

def _static_delta(obj,edges):
    return sum(s*_arc_static(obj,u,v) for u,v,s in edges)

def _remove_edges(src,p,size):
    before=src[p-1] if p else 0
    after=src[p+size] if p+size<len(src) else 0
    first=src[p];last=src[p+size-1]
    # Internal Section arcs are common to removal/insertion, and cancel in
    # same-orientation relocation. Reversal is explicitly marked separately.
    return [(before,first,1),(last,after,1),(before,after,-1)]

def _insert_edges(target,pos,oriented):
    u=target[pos-1] if pos else 0
    v=target[pos] if pos<len(target) else 0
    return [(u,oriented[0],1),(oriented[-1],v,1),(u,v,-1)]

def extract_boundary(routes,candidate,obj,hist=None):
    """Run at training feature export AND online solve with identical inputs.

    `hist` is optional for F2-only cases. Never uses TD Repair or ground truth.
    If there is no relocation proxy, return None (caller must fail/skip explicitly).
    """
    from section_v2 import Stats,_block_options
    from section_dataset import make_partial
    routes=[list(map(int,r)) for r in routes]
    r,p,n=(int(candidate[k]) for k in ('route','pos','size'))
    if r<0 or r>=len(routes) or p<0 or n<2 or p+n>len(routes[r]) or len(routes[r])<=n:
        raise ValueError('Invalid Section geometry')
    partial=make_partial(routes,candidate,obj)
    opts=_block_options(partial,partial.blocks[0],obj,float('inf'),Stats(),nontrivial=True)
    if not opts:return None
    _,k,target_pos,orientation,_,move_type=opts[0]
    if move_type!='relocate':raise AssertionError('Expected relocation proxy only')
    dest=list(partial.routes[k]);orient=list(orientation)
    removed=_remove_edges(routes[r],p,n)
    inserted=_insert_edges(dest,target_pos,orient)
    rem=_static_delta(obj,removed);ins=_static_delta(obj,inserted)
    # Directed internal arcs change when the block is reversed.
    original_block=list(routes[r][p:p+n])
    internal_delta_static=sum(_arc_static(obj,u,v) for u,v in zip(original_block[:-1],original_block[1:]))-sum(
        _arc_static(obj,u,v) for u,v in zip(orient[:-1],orient[1:]))
    # `rem` is static saved cost; `ins` is static added cost.
    src_arrive=_prefix_static(obj,routes[r],p)
    dst_arrive=_prefix_static(obj,dest,target_pos)
    slack=float(obj.capacity-sum(obj.dem[x] for x in dest))
    output={'remove_static_saved_sec':rem,'insert_proxy_static_added_sec':ins,
            'proxy_static_net_gain_sec':rem-ins+internal_delta_static,
            'target_route_index':float(k),'target_position':float(target_pos),
            'target_route_length':float(len(dest)),
            'target_route_load':float(obj.capacity-slack), 'target_route_slack':slack,
            'same_route':float(k==r),
            'source_arrival_static_sec':src_arrive,
            'target_arrival_static_sec':dst_arrive,
            'arrival_offset_static_sec':dst_arrive-src_arrive,
            'target_orientation_reversed':float(tuple(orient)!=tuple(routes[r][p:p+n])),
            'orientation_internal_delta_static_sec':internal_delta_static}
    if hist is None:
        return output
    a=_summary(hist,obj,removed);b=_summary(hist,obj,inserted)
    internal_history=_summary(hist,obj,[(u,v,1) for u,v in zip(original_block[:-1],original_block[1:])] +
                           [(u,v,-1) for u,v in zip(orient[:-1],orient[1:])])['delta']
    output.update({'remove_history_saved_sec':a['delta'],
          'insert_history_added_sec':b['delta'],
          'proxy_history_net_gain_sec':a['delta']-b['delta']+internal_history,
          'removal_boundary_rel_cost':a['rel'],'insertion_boundary_rel_cost':b['rel'],
          'removal_history_latest_rel':a['latest'],'insertion_history_latest_rel':b['latest'],
          'removal_history_trend':a['trend'],'insertion_history_trend':b['trend'],
          'orientation_internal_delta_history_sec':internal_history,
          'removal_observed_share':a['coverage'],'insertion_observed_share':b['coverage'],
          'removal_observed_fraction':a['fraction'],'insertion_observed_fraction':b['fraction'],
          'removal_all_missing':a['missing'],'insertion_all_missing':b['missing'],
          'boundary_observed_min':min(a['edge_min'],b['edge_min']),
          'boundary_observed_gap':a['coverage']-b['coverage']})
    if not np.isfinite(list(output.values())).all():raise ValueError('Nonfinite new features')
    return output

def core_online(routes,c,obj,hour,weekday,base_static_cost=None):
    r,p,n=[int(c[k]) for k in ('route','pos','size')]
    ids=routes[r][p:p+n];before=0 if p==0 else int(routes[r][p-1]);after=0 if p+n==len(routes[r]) else int(routes[r][p+n])
    seq=[before]+list(ids)+[after]
    section_cost=sum(_arc_static(obj,a,b) for a,b in zip(seq[:-1],seq[1:]))
    return {'original_static_cost_sec':float(obj.cost(routes)) if base_static_cost is None else float(base_static_cost),
        'departure_hour_local':float(hour),'departure_weekday_local':float(weekday),
        'route_index':float(r),'position':float(p),'size':float(n),
        'block_demand':float(sum(obj.dem[x] for x in ids)),
        'route_load':float(sum(obj.dem[x] for x in routes[r])),
        'vehicle_capacity':float(obj.capacity),
        'static_section_cost_sec':float(section_cost),'route_length':float(len(routes[r])),
        'relative_position':float(p/max(1,len(routes[r])-1))}

def old_hist_online(routes,c,obj,hist):
    a=hist.section_history(routes,c,obj)
    L=hist.n_slots
    if L!=6:raise ValueError('B2 currently requires six historical slots')
    x={'history_observed_share':float(a['history_observed_share']),
       'history_observed_rel_cost':float(a['history_observed_rel_cost']),
       'history_has_observation':float(a['history_has_observation'])}
    x.update({f'history_coverage_tminus_{L-i}':float(v) for i,v in enumerate(a['history_coverage_by_slot'])})
    x.update({f'history_rel_cost_tminus_{L-i}':float(v) for i,v in enumerate(a['history_rel_cost_by_slot'])})
    return x
