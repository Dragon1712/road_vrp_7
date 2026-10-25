"""R4-P: leakage-safe STATIC road-network modality for Section value ranking.

Road modality: directed shortest-path distances from edge_index.csv/static_travel_sec.
Never touch travel_sec_fifo / edge_costs.npz / hindsight Road.replay for features.
The 'conditioned' mode is explicit bilinear (FiLM-style) interactions; NOT a
trained neural FiLM module and MUST NOT be represented as such.
"""
from __future__ import annotations
import csv, heapq, hashlib, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

STAGE='R4P-STATIC-ROAD-SECTION-FUSION-v1'
MODES=('route_context','road_concat','road_conditioned')
KS=(50,200,500)
EPS=1e-8
N_ROAD=12


def need(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for s in iter(lambda:f.read(1024*1024),b''):h.update(s)
    return h.hexdigest()


def load_json(path):return json.loads(Path(path).read_text(encoding='utf8'))


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(path)


def graph_csv(path):
    """Build directed static graph only; no temporal/hindsight fields loaded."""
    graph=defaultdict(list);count=0
    with Path(path).open(newline='',encoding='utf8') as f:
        rdr=csv.DictReader(f)
        need({'osm_u','osm_v','static_travel_sec'}<=set(rdr.fieldnames or []),'Missing static edge columns')
        for row in rdr:
            c=float(row['static_travel_sec'])
            need(math.isfinite(c) and c>0,'Nonpositive/nonfinite static edge')
            u=str(row['osm_u']);v=str(row['osm_v'])
            graph[u].append((v,c));count+=1
    need(count>0,'Empty road network')
    return graph,count


def dijkstra(graph,src,targets):
    target=set(targets);distance={src:0.};heap=[(0.,src)]
    while heap and target:
        cost,u=heapq.heappop(heap)
        if cost>distance[u]+1e-9:continue
        target.discard(u)
        for v,d in graph.get(u,[]):
            nxt=cost+d
            if nxt<distance.get(v,float('inf'))-1e-9:
                distance[v]=nxt;heapq.heappush(heap,(nxt,v))
    need(not target,'Disconnected road nodes: '+str(list(target)[:4]))
    return distance


def task_od(task,graph):
    """Customer ID mapping is not a feature; only directed static path costs."""
    need(str(task['task_id']) in {f'task_{i:04d}' for i in range(4)},'Non TRAIN task')
    customers=task['customers']; ids=[int(z['id']) for z in customers]
    need(len(set(ids))==len(ids),'Repeated customers')
    nodes=[str(task['depot_node'])]+[str(z['node']) for z in customers]
    # Stable unique node order, accommodates co-located customers.
    targets=set(nodes);dist=np.zeros((len(nodes),len(nodes)),dtype=np.float64)
    for idx,source in enumerate(nodes):
        d=dijkstra(graph,source,targets)
        for j,target in enumerate(nodes):dist[idx,j]=d[target]
    need(np.isfinite(dist).all() and (dist>=0).all(),'Invalid static OD')
    return {'id_to_idx':{v:i+1 for i,v in enumerate(ids)},'dist':dist}


def route_cost(route,static):
    ix=static['id_to_idx'];d=static['dist'];idx=[0]+[ix[int(z)] for z in route]+[0]
    return sum(float(d[a,b]) for a,b in zip(idx[:-1],idx[1:]))


def static_features(snapshot,action,static):
    routes=snapshot['current_routes'];best=snapshot['best_routes']
    i,p,z=map(int,action);need(0<=i<len(routes),'Illegal Section route')
    route=routes[i];need(z in (2,3) and 0<=p<=len(route)-z,'Illegal Section position')
    ix=static['id_to_idx'];d=static['dist'];segment=route[p:p+z]
    def idx(c):return 0 if c is None else ix[int(c)]
    left=route[p-1] if p else None
    right=route[p+z] if p+z<len(route) else None
    seq=[left]+segment+[right]
    arcs=[float(d[idx(a),idx(b)]) for a,b in zip(seq[:-1],seq[1:])]
    by=float(d[idx(left),idx(right)])
    current=route_cost(route,static);oldbest=route_cost(best[i],static)
    all_current=[route_cost(r,static) for r in routes]
    mean_total=sum(all_current)/len(all_current)
    inner=sum(arcs[1:-1]);total=sum(arcs)
    saving=total-by
    scale=max(1.,mean_total)
    features=[
      math.log1p(arcs[0])/10., math.log1p(arcs[-1])/10.,
      math.log1p(inner)/10.,math.log1p(by)/10.,
      max(-2.,min(2.,saving/scale)),
      max(0.,min(2.,total/scale)),
      max(0.,min(2.,current/scale)),
      max(-2.,min(2.,(current-oldbest)/scale)),
      min(2.,arcs[0]/max(1.,arcs[-1])),
      min(2.,arcs[-1]/max(1.,arcs[0])),
      min(2.,current/max(1.,sum(all_current))),
      min(2.,inner/max(1.,total))]
    need(len(features)==N_ROAD and all(math.isfinite(x) for x in features),'Bad static road features')
    return features


def fuse(base,road,mode):
    """An ablation: route-only; concatenation; fixed feature-cross map.

    Last mode is *FiLM-style bilinear feature interaction*, not learned FiLM.
    Weight learning occurs in the subsequently fitted pairwise ridge model.
    """
    need(mode in MODES and len(base)==38 and len(road)==N_ROAD,'Bad fusion mode/input')
    need(all(math.isfinite(x) for x in base+road),'Non-finite feature')
    if mode=='route_context':return list(base)
    if mode=='road_concat':return list(base)+list(road)
    # Modulation of 12 route features by the matching 12 static-road features.
    # Interaction coefficients are fitted solely on each outer training group.
    route_ix=(8,9,10,11,14,17,20,23,25,26,32,36)
    return list(base)+list(road)+[float(base[j]*road[k]) for k,j in enumerate(route_ix)]


def attach(records,static_map):
    res=[]
    for r in records:
        key=tuple(r['snap']);t=key[0]
        need(t in static_map,'Missing Train static distance matrix')
        need('action' in r and 'snapshot' in r,'Missing preselection evidence')
        s=r['snapshot'];need(tuple([s['task_id'],s['scenario_id'],int(s['seed']),int(s['stop_count'])])==key,'Snapshot identity mismatch')
        road=static_features(s,r['action'],static_map[t]);base=r['feature_sets']['route_context']
        mods={m:fuse(base,road,m) for m in MODES}
        res.append({**r,'feature_sets':mods,'static_road_features':road})
    need(len(res)==1572,'Unexpected row count')
    return res


def evaluate(records,splits,original_core):
    """Use exact R4-O ranking/gate/evaluation, changing only ranking features."""
    # Additional modes are local to this process; no upstream files modified.
    prior=original_core.FEATURE_MODES
    original_core.FEATURE_MODES=MODES
    folds=[];details=[]
    try:
        for split_idx,(name,tr,held,typ) in enumerate(splits):
            need(not ({tuple(z['snap']) for z in tr}&{tuple(z['snap']) for z in held}),'Snapshot leakage')
            if typ=='TASK_LOTO':need(not ({z['task'] for z in tr}&{z['task'] for z in held}),'Task leakage')
            else:need(not ({z['snap'][2] for z in tr}&{z['snap'][2] for z in held}),'Seed leakage')
            for mode in MODES:
                for shuffled in (False,True):
                    train=original_core.shuffle_labels(tr,20261025+split_idx*127) if shuffled else tr
                    rank=original_core.fit_pairwise(train,mode)
                    gate=original_core.fit_gate(train)
                    from r4i_core import ridge_predictors
                    ridge=ridge_predictors(train,'k500',strength=100.,shuffle_labels=False)
                    for policy in ('C3','pairwise_ridge_gate','learned_gate_pairwise','oracle_gate_pairwise','sampled_oracle'):
                        rows=original_core.predict_decisions(held,rank,gate,ridge,mode,policy)
                        scores=original_core.metrics(rows)
                        folds.append({'split':name,'split_type':typ,'feature_mode':mode,
                           'shuffle_train_labels':shuffled,'policy':policy,
                           'oracle_conditioned':policy in original_core.ORACLE_POLICIES,
                           'rank_pairs':rank['pairs'],'rank_fitted':rank['fitted'],
                           'gate_fitted':gate['fitted'],**scores})
                        for x in rows:details.append({'evaluation':{'split':name,'mode':mode,'shuffle':shuffled,'policy':policy},**x})
    finally:original_core.FEATURE_MODES=prior
    return folds,details


def render(report):
    o=['# R4-P STATIC Road Section Fusion (Train-only)','',
       '**Not dynamic traffic and NOT a trained neural FiLM.** K=500 fixed-step offline diagnostic only.','',
       'Mode: route_context / road_concat / road_conditioned; only static directed OSM shortest paths enter features.','',
       '| Holdout | Road features | Learned gate + Pairwise gain (sec/snapshot) | Delegations | Positive / Negative |',
       '|---|---|---:|---:|---|']
    for x in report['folds']:
        if not x['shuffle_train_labels'] and x['policy']=='learned_gate_pairwise':
            o.append(f"| {x['split']} | {x['feature_mode']} | {x['mean_gain_sec']:+.3f} | {x['delegations']} | {x['positive']}/{x['negative']} |")
    o.extend(['','- Each outer Train task/seed split is fitted separately; no heldout label or preprocessing optimization.',
       '- Oracle-conditioned rows are diagnostic upper bounds and are never deployable.',
       '- Static road-network distance is a distinct geographical modality, **not** measured live traffic.',
       '- Random/shuffled training label control and baseline route-only mode must be inspected.',
       '- No real 30-second ALNS and no independent Validation/Test.',''])
    return '\n'.join(o)
