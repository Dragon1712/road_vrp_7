#!/usr/bin/env python3
"""Export strict, whitelist-only numeric features and hindsight utility labels.

Group metadata is stored in a SEPARATE file, never in the feature tensor.
No random row split; task and date splits are explicit independent diagnostics.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


STATES = ('original_static_cost_sec','departure_hour_local','departure_weekday_local')
SCALAR = ('route_index','position','size','block_demand','route_load',
          'vehicle_capacity','static_section_cost_sec','route_length','relative_position',
          'history_observed_share','history_observed_rel_cost','history_has_observation')
LABELS = ('td_gain_sec','static_gain_same_td_move_sec',
          'static_repair_static_gain_sec','static_repair_td_gain_sec',
          'td_advantage_over_static_repair_sec')


def export(experiment_dir):
    folder=Path(experiment_dir)
    cfg=json.loads((folder/'run_config.json').read_text())
    summary=json.loads((folder/'benchmark_summary.json').read_text())
    contract=json.loads((Path(__file__).parent/'feature_schema.json').read_text())
    if cfg.get('stage')!='step6a-offline-v1' or summary.get('status')!='COMPLETE':
        raise ValueError('Step6-A collector must be complete')
    L=int(cfg['lookback_slots'])
    names=list(STATES)+list(SCALAR)+[f'history_coverage_tminus_{L-i}' for i in range(L)]+[
        f'history_rel_cost_tminus_{L-i}' for i in range(L)]
    whitelist=set(contract['state_features'])|set(contract['candidate_features'])
    if not set(STATES).issubset(whitelist) or not set(SCALAR).issubset(whitelist):
        raise ValueError('Feature used without whitelist authorization')
    if set(names)&set(contract['NEVER_AS_FEATURE']):
        raise ValueError('Label leaked into features')
    states=[];valid_records=[];dates=set()
    with (folder/'state_records.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);states.append(r);dates.add(r['date'])
    if len(states)!=summary['expected_states']:raise ValueError('Missing states')
    sorted_dates=sorted(dates)
    date_role={d:('train' if i<int(np.ceil(len(sorted_dates)*.5)) else
                  'val' if i<int(np.ceil(len(sorted_dates)*.75)) else 'internal_test')
               for i,d in enumerate(sorted_dates)}
    xs=[];ys=[];groups=[]
    for r in states:
        for c in r['candidates']:
            if not c['label_valid']:
                continue
            flat=[float(r[k]) for k in STATES]
            flat+=[float(c[k]) for k in SCALAR]
            flat+=[float(z) for z in c['history_coverage_by_slot']]
            flat+=[float(z) for z in c['history_rel_cost_by_slot']]
            if len(flat)!=len(names):raise ValueError('Feature vector dimension mismatch')
            label=[float(c[k]) for k in LABELS]
            if not np.isfinite(flat).all() or not np.isfinite(label).all():
                raise ValueError('Nonfinite feature or label')
            xs.append(flat);ys.append(label)
            groups.append({'sample_id':c['candidate_id'],'state_id':r['state_id'],
                           'task_id':r['task_id'],'scenario_id':r['scenario_id'],
                           'date':r['date'],'task_role':r['task_role'],
                           'date_role':date_role[r['date']],
                           'source_solver':r['source_solver'],
                           'seed':r['seed'], 'snapshot_sec':r['snapshot_sec']})
    out=folder/'analysis';out.mkdir(exist_ok=True)
    X=np.array(xs,dtype=np.float32).reshape(-1,len(names))
    y=np.array(ys,dtype=np.float32).reshape(-1,len(LABELS))
    np.savez_compressed(out/'features_X.npz',X=X)
    np.savez_compressed(out/'labels_y.npz',y=y)
    cols=list(groups[0]) if groups else ['sample_id','state_id','task_id','scenario_id','date','task_role','date_role','source_solver','seed','snapshot_sec']
    with (out/'groups.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(groups)
    info={'feature_contract':contract['schema'],'feature_names':names,'label_names':LABELS,
          'shape_X':list(X.shape),'shape_y':list(y.shape),
          'task_roles':'train/val/internal_test by sorted tasks; valid only inside July development',
          'date_roles':'independent train/val/internal_test by chronological day; never mix evaluation protocols',
          'warning':'True prospective deployment needs non-Oracle state generation. Labels are retrospective Oracle only.'}
    (out/'dataset_schema.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'PASS exported X={X.shape}, y={y.shape}, groups={len(groups)} to {out}')
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir',required=True)
    export(p.parse_args().experiment_dir)

if __name__=='__main__':main()
