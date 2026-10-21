#!/usr/bin/env python3
"""Use frozen B2-R 30-second official ALNS, patch only instance Section scorer.

D2 adds true predeparture directed OSM masked graphs. D0/D1 use F4 only.
All feature/inference operations are timed INSIDE the same B2-R 30s budget.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for rel in ('road_vrp7_step6c1','road_vrp7_step6b2_r'):
    p=str(ROOT/rel)
    if p not in sys.path:sys.path.insert(0,p)
import c1_model  # noqa; required before trusted joblib.load
import benchmark_30s_r as base
from graph_features import candidate_graph,GRAPH_DIM
from dataset import sha256

FROZEN_R=base.OnlineSectionR
FROZEN_LOADER=base.load_artifact
GRAPH_STUDY=None

class C1Section(FROZEN_R):
    def form(self,routes,c,base_cost):
        f4=super().form(routes,c,base_cost)
        if f4 is None:return None
        if self.artifact['c1_variant'] not in ('D2','D2T'):return f4
        if self.hist is None:raise ValueError('D2 requires original raw history')
        graph=candidate_graph(routes,c,self.obj,self.hist).flat()
        import numpy as np
        return np.concatenate((f4,graph))


def checked_load(path,mode,method,protocol,source):
    a=FROZEN_LOADER(path,mode,method,protocol,source)
    if a.get('stage')!='step6c1-delegation-v1' or a.get('c1_variant') not in ('D0','D1','D2T','D2'):
        raise ValueError('Not a verified C1 artifact')
    if a.get('graph_dim')!=GRAPH_DIM or mode!='F4' or method!='P2':
        raise ValueError('Bad C1 graph/feature schema')
    if not GRAPH_STUDY or sha256(Path(GRAPH_STUDY)/'graph_X.npz')!=a['graph_sha256']:
        raise ValueError('Online cannot prove graph training source')
    return a


def main():
    global GRAPH_STUDY
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--graph-dir',required=True)
    args,forward=ap.parse_known_args()
    GRAPH_STUDY=args.graph_dir
    if not (Path(GRAPH_STUDY)/'summary.json').exists():raise ValueError('Graph export incomplete')
    output_parser=argparse.ArgumentParser(add_help=False)
    output_parser.add_argument('--output-dir',required=True)
    out,_=output_parser.parse_known_args(forward)
    if '--control' not in forward or forward[forward.index('--control')+1]!='learned':
        raise ValueError('C1 online requires learned treatment; run M0/Random using frozen B2-R')
    original_argv=sys.argv
    original_handler=base.OnlineSectionR
    original_loader=base.load_artifact
    base.OnlineSectionR=C1Section
    base.load_artifact=checked_load
    try:
        sys.argv=['benchmark_30s_r.py']+forward
        base.main()
    finally:
        sys.argv=original_argv
        base.OnlineSectionR=original_handler
        base.load_artifact=original_loader
    # B2's manifest is preserved unedited; supplementary provenance pins C1 code.
    model_path=Path(forward[forward.index('--model-path')+1])
    metadata={'stage':'step6c1-online-wrapper-v1',
              'model_sha256':sha256(model_path),'graph_training_sha256':sha256(Path(GRAPH_STUDY)/'graph_X.npz'),
              'code_sha256':{p:sha256(Path(__file__).with_name(p)) for p in
                             ('c1_model.py','graph_features.py','online_c1.py')},
              'notes':['Frozen B2-R solver, same sampler, cadence, C3 fallback and 30s budget.',
                       'The B2 legacy model_code_sha256 field identifies the B1 scorer interface; use this sidecar for C1.',
                       'OD precompute and initial historical loading excluded from SEARCH budget as in B2-R.']}
    (Path(out.output_dir)/'c1_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':main()
