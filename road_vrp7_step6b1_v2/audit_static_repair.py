#!/usr/bin/env python3
"""Static vs Oracle Repair retrospective offline label audit."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from dataset import load,DEFAULT_SOURCE,PRIMARY,ORACLE,grouped_indices,route_overlap


def analyze(d):
    a=d.label(PRIMARY);b=d.label(ORACLE)
    pools=grouped_indices(d.groups)
    stat_best=np.array([max(0.,float(a[ix].max())) for ix in pools.values()])
    oracle_best=np.array([max(0.,float(b[ix].max())) for ix in pools.values()])
    if np.any(oracle_best+2e-3<stat_best):raise ValueError('Contradictory top action Oracle bound')
    return {'candidates':d.n,'states':len(pools),'target':PRIMARY,
            'static_repair_td_positive_candidates':int(np.sum(a>1e-7)),
            'oracle_td_repair_positive_candidates':int(np.sum(b>1e-7)),
            'static_repair_td_positive_rate':float(np.mean(a>1e-7)),
            'oracle_td_repair_positive_rate':float(np.mean(b>1e-7)),
            'states_with_static_repair_improvement':int(np.sum(stat_best>1e-7)),
            'states_with_oracle_repair_improvement':int(np.sum(oracle_best>1e-7)),
            'mean_best_static_repair_noop_gain_sec':float(np.mean(stat_best)),
            'mean_best_oracle_repair_noop_gain_sec':float(np.mean(oracle_best)),
            'mean_best_repair_gap_sec':float(np.mean(oracle_best-stat_best)),
            'mean_candidate_oracle_repair_advantage_sec':float(np.mean(b-a)),
            'fraction_strict_oracle_repair_advantage':float(np.mean((b-a)>1e-7)),
            'route_reuse':route_overlap(d),
            'warning':'Historical TD objective retrospective score on static-repaired routes, NOT deployable real-road ground truth.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',default=DEFAULT_SOURCE,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    result=analyze(load(a.source_dir))
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
