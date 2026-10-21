"""Directed OSM road-edge observations for B2's STATIC relocation proxy.

Only strictly earlier historical RAW observations are read. No TD label/replay.
These features are a sampled, directed road-edge graph, NOT the full official SPIN model.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

T = 6
N = 24
META = 4
GRAPH_DIM = N*T*2 + N*META + N*N + N

@dataclass
class GraphBatch:
    seq: np.ndarray     # [N,T,2]: clipped observed deviation; real mask
    meta: np.ndarray    # [N,4]: log static sec, removal/insertion, arc sign, path index
    adj: np.ndarray     # [N,N]: directed OSM edge adjacency, plus diag on valid nodes
    valid: np.ndarray   # [N]

    def flat(self):
        return np.concatenate([self.seq.ravel(), self.meta.ravel(), self.adj.ravel(),self.valid.ravel()]).astype(np.float32)


def unpack(arr):
    import numpy as np
    a=np.asarray(arr,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]!=GRAPH_DIM:raise ValueError(f'bad graph dimension: expected {GRAPH_DIM}')
    p=0
    seq=a[:,p:p+N*T*2].reshape(-1,N,T,2);p+=N*T*2
    meta=a[:,p:p+N*META].reshape(-1,N,META);p+=N*META
    adj=a[:,p:p+N*N].reshape(-1,N,N);p+=N*N
    valid=a[:,p:p+N]
    return seq,meta,adj,valid


def _indices(length,k):
    if length <= k:return np.arange(length,dtype=np.int64)
    # Deterministic, endpoint-inclusive sampling of a static OSM path.
    return np.unique(np.rint(np.linspace(0,length-1,k)).astype(np.int64))


def encode_raw_graph(arc_specs,hist,max_edges_per_arc=4):
    """arc_specs is [(od_from,od_to,sign,is_insert),...] in B2 ordering.
    arc_specs includes the shortcut with sign=-1; it must remain as a node
    attribute rather than being silently erased.
    """
    if hist.index_anchor is None:raise ValueError('Call hist.enter() before graph extraction')
    if hist.n_slots!=T:raise ValueError('Only frozen six-slot traffic is supported')
    if max_edges_per_arc<1:raise ValueError('positive samples/arc required')
    road=hist.road
    slots=np.arange(hist.index_anchor-T+1,hist.index_anchor+1)
    if int(np.max(slots))>=hist.index_anchor+1 or int(np.max(slots))*road.dt>=road.seconds(hist.key[1]):
        raise AssertionError('Future or current observation leaked')
    edge_list=[]
    for u,v,sgn,is_insert in arc_specs:
        # Arc endpoint IDs are already indices in the frozen OD matrix.
        i,j=int(u),int(v)
        if i==j: continue
        key=(hist.od.task['task_id'],i,j)
        if key not in hist.paths:
            _,p=road.path(hist.od.nodes[i],hist.od.nodes[j],hist.od.start,mode='static')
            hist.paths[key]=np.asarray(p,dtype=int)
        path=np.asarray(hist.paths[key],dtype=int)
        selected=_indices(len(path),max_edges_per_arc)
        for ind in selected:
            if len(edge_list)>=N:break
            edge_list.append((int(path[ind]),int(sgn),int(is_insert),float(ind/max(1,len(path)-1))))
        if len(edge_list)>=N:break
    seq=np.zeros((N,T,2),dtype=np.float32)
    meta=np.zeros((N,META),dtype=np.float32)
    adj=np.zeros((N,N),dtype=np.float32)
    valid=np.zeros(N,dtype=np.float32)
    if not edge_list:return GraphBatch(seq,meta,adj,valid)
    ids=np.asarray([x[0] for x in edge_list],dtype=int)
    obs=road.observed[np.ix_(slots,ids)].T.astype(np.float32)
    raw=hist.raw[np.ix_(slots,ids)].T.astype(np.float64)
    stat=road.static[ids].reshape(-1,1)
    ratios=np.divide(raw,stat, out=np.zeros_like(raw),where=stat>0)
    observed_delta=np.where(obs>0,np.clip(ratios-1.,-5.,5.),0.)
    count=len(ids);valid[:count]=1.
    seq[:count,:,0]=observed_delta.astype(np.float32)
    seq[:count,:,1]=obs
    meta[:count,0]=np.log1p(stat[:,0])/6.0
    meta[:count,1]=[v[2] for v in edge_list]
    meta[:count,2]=[v[1] for v in edge_list]
    meta[:count,3]=[v[3] for v in edge_list]
    # True directed OSM adjacency: road edge i's destination equals edge j's origin.
    src=road.edges.osm_u.iloc[ids].astype(str).to_numpy()
    dst=road.edges.osm_v.iloc[ids].astype(str).to_numpy()
    adj[:count,:count]=(dst[:,None]==src[None,:]).astype(np.float32)
    adj[np.arange(count),np.arange(count)]=1.
    return GraphBatch(seq,meta,adj,valid)


def candidate_graph(routes,c,obj,hist):
    """Identical STATIC first-relocation insertion proxy to B2.features.extract_boundary."""
    from section_v2 import Stats,_block_options
    from section_dataset import make_partial
    from features import _remove_edges,_insert_edges
    src=[list(map(int,r)) for r in routes]
    p=make_partial(src,c,obj)
    opts=_block_options(p,p.blocks[0],obj,float('inf'),Stats(),nontrivial=True)
    if not opts: return GraphBatch(np.zeros((N,T,2),np.float32),np.zeros((N,META),np.float32),
                                   np.zeros((N,N),np.float32),np.zeros(N,np.float32))
    _,k,target_pos,orientation,_,move_type=opts[0]
    if move_type!='relocate':raise AssertionError('B2 only uses first relocation')
    rem=_remove_edges(src[int(c['route'])],int(c['pos']),int(c['size']))
    ins=_insert_edges(list(p.routes[k]),target_pos,list(orientation))
    def od_id(x):return 0 if int(x)==0 else int(obj.index[int(x)])
    arcs=[(od_id(u),od_id(v),sgn,0) for u,v,sgn in rem]
    arcs += [(od_id(u),od_id(v),sgn,1) for u,v,sgn in ins]
    return encode_raw_graph(arcs,hist)
