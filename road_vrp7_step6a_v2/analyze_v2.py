#!/usr/bin/env python3
"""Run frozen v1 statistical audit/export with Step6-A v2 stage checking."""
import argparse, csv, importlib.util, json, sys
from collections import Counter
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load_copy(src,name):
    # Run v1 scripts with a SINGLE stage-name change; v1 files remain frozen.
    text=src.read_text(encoding='utf-8')
    old="step6a-offline-v1"
    if text.count(old)<1: raise RuntimeError('No frozen stage anchor')
    src2=text.replace(old, 'step6a-offline-v2')
    ns={'__file__':str(src), '__name__':name}
    exec(compile(src2,str(src),'exec'),ns)
    return ns

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);a=p.parse_args()
    audit=load_copy(ROOT/'road_vrp7_step6a'/'analyze_step6a.py','step6a_v2_analyzer')
    exp=load_copy(ROOT/'road_vrp7_step6a'/'export_step6a.py','step6a_v2_exporter')
    audit['analyze'](a.experiment_dir)
    exp['export'](a.experiment_dir)
    folder=Path(a.experiment_dir)
    with (folder/'analysis/state_ranking_diagnostics.csv').open(newline='',encoding='utf-8') as f:
        d=list(csv.DictReader(f))
    from_source=Counter()
    with (folder/'state_records.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r=json.loads(line)
                from_source['states']+=1
                from_source['nonbest']+=int(r.get('captured_nonbest',False))
    positive=0;extra=[];oracle=[];static=[];topchange=0
    by_source=Counter()
    for r in d:
        o=float(r['oracle_top1_gain_sec']);s=float(r['static_selected_td_gain_sec'])
        oracle.append(max(0.,o));static.append(max(0.,s))
        diff=max(0.,o)-max(0.,s)
        extra.append(diff)
        positive+=int(o>1e-7)
        topchange+=int(r['top1_changed'])
        by_source[str(r.get('source_solver',''))]+=1
    if any(x< -1e-7 for x in extra): raise AssertionError('Negative no-op regret')
    out={'stage':'step6a-current-label-audit-v1', 'states':len(d),
         'source_states':from_source['states'],
         'captured_nonbest_states':from_source['nonbest'],
         'captured_nonbest_fraction':from_source['nonbest']/max(1,from_source['states']),
         'states_with_useful_oracle_action':positive,
         'useful_state_rate':positive/max(1,len(d)),
         'top1_disagreement_rate':topchange/max(1,len(d)),
         'oracle_noop_mean_gain_sec':float(np.mean(oracle)) if oracle else None,
         'static_noop_mean_gain_sec':float(np.mean(static)) if static else None,
         'mean_noop_decision_regret_sec':float(np.mean(extra)) if extra else None,
         'no_op_decision_regret_positive_states':sum(x>1e-7 for x in extra),
         'warning':'Only a diagnostic within the sampled actions and hindsight labels; not learned policy performance.'}
    (folder/'analysis/current_state_diagnostics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
