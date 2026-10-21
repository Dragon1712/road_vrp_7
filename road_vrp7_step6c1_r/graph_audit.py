#!/usr/bin/env python3
"""Graph representation audit: physical directed edges, padding-aware sparsity.

No labels used. No model training. Analyze connectivity before adding SPIN/GAT.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1'))
from graph_features import unpack,N,T,GRAPH_DIM


def summarize(G):
    seq,meta,adj,valid=unpack(G)
    binary=valid>0.5
    obs=seq[...,1]>0.5
    counts=binary.sum(1)
    # Directed, off-diagonal physical adjacency (ignore padding & self-loops).
    edge_mask=binary[:,:,None]&binary[:,None,:]
    adj=np.asarray(adj>0.5)&edge_mask
    d=np.arange(N);adj[:,d,d]=False
    connections=adj.sum((1,2))
    deg=adj.sum(1)+adj.sum(2)
    isolation=(deg==0)&binary
    valid_slots=int(binary.sum())*T
    return {'candidates':len(G),'padded_slots_coverage':float(obs.mean()),
     'valid_physical_edge_slots_coverage':float(obs.sum()/max(1,valid_slots)),
     'mean_valid_edges':float(counts.mean()),'median_valid_edges':float(np.median(counts)),
     'mean_directed_nonself_links':float(connections.mean()),
     'fraction_graphs_without_nonself_links':float(np.mean(connections==0)),
     'valid_node_isolation_rate':float(isolation.sum()/max(1,binary.sum())),
     'fraction_graphs_all_missing_history':float(np.mean((obs&binary[:,:,None]).sum((1,2))==0)),
     'mean_observations_per_valid_edge':float(obs.sum()/max(1,binary.sum())),
     'note':'Nonself links are direct sampled OSM edge adjacency; gap paths not inferred.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph-dir',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    src=Path(a.graph_dir)/'graph_X.npz';out=Path(a.output_dir)
    if out.exists():raise FileExistsError(out)
    if not src.is_file():raise FileNotFoundError(src)
    with np.load(src,allow_pickle=False) as z:G=z['X']
    if G.ndim!=2 or G.shape[1]!=GRAPH_DIM:raise ValueError('Graph dim incompatible')
    res=summarize(G);out.mkdir(parents=True,exist_ok=False)
    (out/'graph_audit.json').write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(res,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
