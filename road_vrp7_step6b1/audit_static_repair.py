#!/usr/bin/env python3
"""Compare fixed candidate pools under Oracle-TD Repair vs static Repair.

All costs are retrospective TD objective values, not Road.replay metrics.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from dataset import DEFAULT_SOURCE, PRIMARY, ORACLE, load, states_index


def analyze(data):
    td=data.label(ORACLE); st=data.label(PRIMARY)
    n=data.n
    assert n
    idx=states_index(data.groups)
    gain_or=[];gain_st=[];miss=[];td_wins=0;st_wins=0;same_best=0
    for rows in idx.values():
        t=float(max(0.,np.max(td[rows])));s=float(max(0.,np.max(st[rows])))
        gain_or.append(t);gain_st.append(s);miss.append(t-s)
        td_wins+=int(t>1e-7);st_wins+=int(s>1e-7)
        same_best+=int(int(rows[np.argmax(td[rows])])==int(rows[np.argmax(st[rows])]))
    if any(v< -1e-4 for v in miss):raise ValueError('Oracle TD repair must upper bound static repair')
    pos=lambda x:int(np.sum(x>1e-7))
    results={
        'source':str(data.source),'states':len(idx),'candidates':n,
        'comparison':'Static Repair vs Oracle TD Repair on identical source states and section candidates',
        'static_repair_td_positive_candidates':pos(st),
        'oracle_td_repair_positive_candidates':pos(td),
        'static_repair_td_positive_rate':float(np.mean(st>1e-7)),
        'oracle_td_repair_positive_rate':float(np.mean(td>1e-7)),
        'oracle_positive_but_static_nonpositive':int(np.sum((td>1e-7)&(st<=1e-7))),
        'states_with_static_repair_improvement':st_wins,
        'states_with_oracle_repair_improvement':td_wins,
        'mean_static_repair_signed_gain_sec':float(np.mean(st)),
        'mean_oracle_repair_signed_gain_sec':float(np.mean(td)),
        'mean_oracle_minus_static_repair_adv_sec':float(np.mean(td-st)),
        'fraction_oracle_repair_strictly_better':float(np.mean(td-st>1e-7)),
        'mean_best_static_repair_noop_gain_sec':float(np.mean(gain_st)),
        'mean_best_oracle_repair_noop_gain_sec':float(np.mean(gain_or)),
        'mean_best_repair_gap_sec':float(np.mean(miss)),
        'best_section_agreement_rate':same_best/len(idx),
        'evaluation':'Retrospective Oracle TD planning-objective replay; NOT real-road replay; multiple candidate repairs are evaluated offline and must not be treated as free during ALNS.'}
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError('Use an independent NEW output directory')
    r=analyze(load(a.source_dir))
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'static_repair_audit.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(r,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
