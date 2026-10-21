#!/usr/bin/env python3
"""Trustworthy C1-R 30-sec B2-R adapter. Original ALNS files are unchanged."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for rel in ('road_vrp7_step6c1_r','road_vrp7_step6c1','road_vrp7_step6b2_r','road_vrp7_step6b1_v2','road_vrp7_step6b2','road_vrp7_step6a','road_vrp7_step5b_v3','road_vrp7_stage3'):
    p=str(ROOT/rel)
    if p not in sys.path:sys.path.insert(0,p)
import c1_model,r_model  # ensure original + new classes can be safely unpickled from local artifacts
import benchmark_30s_r as base
from graph_features import candidate_graph,GRAPH_DIM
from dataset import sha256

FROZEN_LOADER=base.load_artifact
GRAPH_STUDY=None

class C1RSection(base.OnlineSectionR):
    def form(self,routes,c,base_cost):
        values=super().form(routes,c,base_cost)
        if values is None:return None
        if self.artifact['c1_variant'] not in ('D2','D2T'):return values
        if self.hist is None:raise ValueError('Graph requires predeparture raw history')
        return np.concatenate((values,candidate_graph(routes,c,self.obj,self.hist).flat()))

def checked_load(path,mode,method,protocol,source):
    a=FROZEN_LOADER(path,mode,method,protocol,source)
    if a.get('stage')!='step6c1r-anchored-selective-defer-v1':raise ValueError('Wrong C1-R artifact')
    variant=a.get('c1_variant');want='F2' if variant=='D0' else 'F4'
    if method!='P2' or mode!=want:raise ValueError('Incorrect C1-R mode')
    if not isinstance(a.get('model'),r_model.ReliableC1):raise ValueError('Wrong predictor class')
    if a['model'].variant!=variant or a['model'].safe_gate!=(a.get('r_arm')=='FULL_DEFER_SAFE'):
        raise ValueError('Artifact policy settings mismatch')
    if a['graph_dim']!=GRAPH_DIM:raise ValueError('Graph schema mismatch')
    # D0 (F2) and D1 (F4) do not read graph data at inference.
    # Avoid unrelated graph-file IO for these ablations.
    if variant in ('D2T','D2'):
        if not GRAPH_STUDY or sha256(Path(GRAPH_STUDY)/'graph_X.npz')!=a['graph_sha256']:
            raise ValueError('Graph training source changed')
    return a

def main():
    global GRAPH_STUDY
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--graph-dir',required=True)
    config,forward=p.parse_known_args();GRAPH_STUDY=config.graph_dir
    q=argparse.ArgumentParser(add_help=False);q.add_argument('--output-dir',required=True)
    folder,_=q.parse_known_args(forward)
    if '--control' not in forward or forward[forward.index('--control')+1]!='learned':
        raise ValueError('Run frozen C3/M0/Random through official B2-R')
    if '--model-path' not in forward:raise ValueError('Missing model path')
    model_path=Path(forward[forward.index('--model-path')+1])
    old_class,old_loader,old_argv=base.OnlineSectionR,base.load_artifact,sys.argv
    base.OnlineSectionR=C1RSection;base.load_artifact=checked_load
    try:
        sys.argv=['benchmark_30s_r.py']+forward
        base.main()
    finally:
        sys.argv=old_argv;base.OnlineSectionR=old_class;base.load_artifact=old_loader
    graph_meta=Path(GRAPH_STUDY)/'summary.json'
    graph_sha=json.loads(graph_meta.read_text()).get('graph_sha256') if graph_meta.is_file() else None
    report={'stage':'step6c1r-online-v1','model_sha256':sha256(model_path),
      'code_sha256':{s:sha256(Path(__file__).with_name(s)) for s in ('r_model.py','online_c1r.py')},
      'graph_training_declared_sha256':graph_sha,
      'warnings':['30s search is also limited by max-iterations.',
       'Online rejection falls back to original C3; offline M0 proxy is not C3.',
       'History and OD setup measured outside SEARCH budget according to frozen B2-R protocol.']}
    (Path(folder.output_dir)/'c1r_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
