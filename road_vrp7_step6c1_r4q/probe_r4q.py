#!/usr/bin/env python3
"""R4-Q reproducible Train-only lagged (not contemporaneous) road learning probe."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for name in ('road_vrp7_step6c1_r4q','road_vrp7_step6c1_r4p',
             'road_vrp7_step6c1_r4o','road_vrp7_step6c1_r4n',
             'road_vrp7_step6c1_r4m','road_vrp7_step6c1_r4l',
             'road_vrp7_step6c1_r4i','road_vrp7_step6c1_r4k'):
    s=str(ROOT/name)
    if s not in sys.path:sys.path.insert(0,s)
import probe_r4p as p4p
import probe_r4o as p4o
import r4p_core as pc
import r4o_core as oc
import r4l_core as lc
import r4q_core as qc


def check(a):
    records,tasks,pins=p4p.validate_inputs(a)
    cost=Path(a.cost_dir)
    raw=cost/'edge_costs.npz';time=cost/'time_index.csv'
    qc.need(raw.is_file() and time.is_file(),'Stage2 NPZ/time_index missing')
    prior=qc.load_json(Path(a.old_capture)/'manifest.json')
    recent=qc.load_json(Path(a.new_capture)/'manifest.json')
    h=qc.sha(raw)
    qc.need(h==prior['road_sha256']==recent['road_sha256'],'Stage2 traffic NPZ differs from frozen Capture')
    qc.need(len(records)==1572 and set(tasks)=={f'task_{i:04d}' for i in range(4)},'Unexpected Train data')
    evidence={**pins,'edge_costs_npz':h,'time_index_csv':qc.sha(time),
              'r4p_core_sha256':qc.sha(ROOT/'road_vrp7_step6c1_r4p/r4p_core.py'),
              'r4o_core_sha256':qc.sha(ROOT/'road_vrp7_step6c1_r4o/r4o_core.py')}
    # Check time-index textual table is exactly aligned with the npz integer ns axis.
    import csv
    with time.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    qc.need(rows and all('time_local' in r for r in rows),'Time_index file malformed')
    with np.load(raw,allow_pickle=False) as z:
        ns=z['slot_local_naive_ns']
    text=np.array([np.datetime64(r['time_local'],'ns').astype(np.int64) for r in rows],dtype=np.int64)
    qc.need(len(text)==len(ns) and np.array_equal(text,ns),'Stage2 time axis CSV differs from NPZ axis')
    for scen in ('2025-07-02_0800','2025-07-09_1700'):
        arr=[s['departure_local'] for t in tasks.values() for s in t['scenarios'] if s['scenario_id']==scen]
        qc.need(len(arr)==4 and len(set(arr))==1,'Missing matching Train scenario clock')
        src=qc.resolve_historical_source(ns,arr[0])
        if scen=='2025-07-09_1700':
            qc.need(src['available'],'July 9 must have exact past-24h history')
        else:
            qc.need((not src['available'] and src['fallback_reason']=='BEFORE_DATA_START')
                    or src['available'], 'Invalid July 2 historical fallback')
        print(f"[R4Q] {scen}: requested={src['requested_historical_slot']} "
              f"available={src['available']} used={src['historical_slot']} "
              f"fallback={src['fallback_reason']}",flush=True)
    return records,tasks,evidence


def execute(a):
    out=Path(a.output_dir) if a.output_dir else None
    if a.mode=='probe':qc.need(out is not None and not out.exists(),'Use a NEW independent STUDY_NAME')
    records,tasks,pins=check(a)
    print('[R4Q] SOURCE PASS: 1572 Train actions, 200 snapshots; past-only source or explicitly marked static fallback',flush=True)
    if a.mode=='preflight':return
    old_snaps,new_snaps,old_actions,new_actions=p4o.action_maps(a,records)
    expanded=oc.attach_context(records,old_snaps,new_snaps,old_actions,new_actions)
    snapmap={oc.sid(s):s for s in old_snaps+new_snaps}
    actionmap={**old_actions,**new_actions}
    joined=[]
    for r in expanded:
        k=tuple(r['snap'])
        joined.append({**r,'snapshot':snapmap[k],'action':actionmap[k,r['arm']]})
    statics,lagged,lag_metadata=qc.build_context(tasks,a.cost_dir)
    fused=qc.attach(joined,statics,lagged,lag_metadata,pc)
    splits=lc.make_splits(fused)
    qc.need(len(splits)==6,'Changed Train grouping')
    fold,details=qc.evaluate(fused,splits,oc)
    provenance={**pins,'code':{
        'r4q_core.py':qc.sha(Path(__file__).with_name('r4q_core.py')),
        'probe_r4q.py':qc.sha(__file__)}}
    doc={'stage':qc.STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
      'eval_semantics':'HISTORICAL_PAST_24H_OFFLINE_FIXED_K500_NOT_REALTIME_OR_30S_ONLINE',
      'source_git_commit_reviewed':'423d638a7a06be6db8da6fb231277c69dffa9c8e',
      'n_actions':len(fused),'n_snapshots':len(snapmap),
      'feature_modes':{'route_context':38,'static_concat':50,
                       'lag24_concat':63,'lag24_conditioned':75},
      'historical_source':lag_metadata,
      'history_stratified':qc.stratify_by_history(details,lag_metadata),
      'historical_availability_protocol':'Exact past 24h required if within Stage2 index; static-only fallback if requested slot predates index start',
      'source_sha256':provenance,
      'model':'Frozen R4-O weighted Pairwise Ridge + Gate; NOT learned FiLM or contrastive',
      'splits':[name for name,_,_,_ in splits],
      'policies':['C3','pairwise_ridge_gate','learned_gate_pairwise','oracle_gate_pairwise','sampled_oracle'],
      'oracle_policies_non_deployable':['oracle_gate_pairwise','sampled_oracle'],
      'folds':fold,
      'limitations':['Only Train 4 tasks with two scenario dates and four seeds',
         'Lagged historical traffic proxy, not prediction of target-day real-time traffic',
         'Sparse monitored corridors and static missing fallback; inspect coverage',
         'Historical observations are raw Stage2 values at fixed previous-day clock when available; July 2 static-only fallback',
         'No 30s ALNS performance claims and no independent Validation/Test',
         'Conditional interactions are fixed FiLM-style features, NOT trainable FiLM']}
    out.mkdir(parents=True,exist_ok=False)
    qc.save(out/'report.json',doc)
    (out/'report.md').write_text(qc.report_md(doc),encoding='utf-8')
    with (out/'fold_predictions.jsonl').open('w',encoding='utf-8') as f:
        for x in details:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
    # Save derived matrices plus timestamps to validate that only historical data entered model.
    odpath=out/'historical_od';odpath.mkdir(exist_ok=False)
    for (task,scenario),z in sorted(lagged.items()):
        np.savez_compressed(odpath/f'{task}_{scenario}_lag24.npz',
                            distance_lag24=z['dist'], customer_ids=np.asarray(
                                [cid for cid,i in sorted(z['id_to_idx'].items(),key=lambda kv:kv[1])],dtype=np.int64),
                            source_slot=lag_metadata[scenario]['historical_slot'] or '',
                            requested_source_slot=lag_metadata[scenario]['requested_historical_slot'],
                            history_available=np.bool_(lag_metadata[scenario]['available']))
    qc.save(out/'provenance.json',{'stage':qc.STAGE,
       'report_sha256':qc.sha(out/'report.json'),
       'predictions_sha256':qc.sha(out/'fold_predictions.jsonl'),
       'source_sha256':provenance,
       'historical_od_sha256':{p.name:qc.sha(p) for p in sorted(odpath.glob('*.npz'))}})
    print('R4-Q COMPLETE',out,'folds',len(fold),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','probe'),default='probe')
    p.add_argument('--r4o-result',default='runs/chicago_july_v1/step6c1_r4o/r4o_struct_rank_gate_train_v1')
    p.add_argument('--r4n-result',default='runs/chicago_july_v1/step6c1_r4n/r4n_rank_gate_train_v1')
    p.add_argument('--r4m-result',default='runs/chicago_july_v1/step6c1_r4m/r4m_nested_risk_train_v1')
    p.add_argument('--r4l-result',default='runs/chicago_july_v1/step6c1_r4l/r4l_seed_transfer_loto_train_v1')
    p.add_argument('--old-cf',default='runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1')
    p.add_argument('--old-capture',default='runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture')
    p.add_argument('--old-prefix',default='runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1')
    p.add_argument('--new-r4k',default='runs/chicago_july_v1/step6c1_r4k/r4k_full_128audited_v1')
    p.add_argument('--new-cf',default='runs/chicago_july_v1/step6c1_r4h/r4j_train_early_s62s63_v1_cf_full')
    p.add_argument('--new-capture',default='runs/chicago_july_v1/step6c1_r4g/r4j_train_early_s62s63_v1_capture')
    p.add_argument('--new-prefix',default='runs/chicago_july_v1/step6c1_r4g_prefix/r4j_train_early_s62s63_v1_prefix_full')
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--output-dir')
    a=p.parse_args()
    if a.mode=='probe' and not a.output_dir:p.error('--output-dir required')
    execute(a)
if __name__=='__main__':main()
