#!/usr/bin/env python3
"""Evaluate static-graph road Section modality using already-pinned Train R4-O labels."""
from __future__ import annotations
import argparse, json, sys
import numpy as np
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r4o','road_vrp7_step6c1_r4n','road_vrp7_step6c1_r4m',
               'road_vrp7_step6c1_r4l','road_vrp7_step6c1_r4i','road_vrp7_step6c1_r4k'):
    path=str(ROOT/folder)
    if path not in sys.path:sys.path.insert(0,path)
import probe_r4o as p4o
import r4o_core as o
import r4l_core as l
import r4p_core as core


def validate_inputs(args):
    # R4-O itself revalidates all R4-L, M, N and source Train capture pins.
    records,pins,extra=p4o.source_records(args)
    d=Path(args.r4o_result)
    rep=core.load_json(d/'report.json')
    prov=core.load_json(d/'provenance.json')
    core.need(rep.get('stage')==o.STAGE and rep.get('status')=='COMPLETE' and
              rep.get('data_role')=='TRAIN_ONLY' and rep.get('n_actions')==1572,'Incorrect R4-O source report')
    core.need(prov.get('report_sha256')==core.sha(d/'report.json') and
              prov.get('predictions_sha256')==core.sha(d/'fold_predictions.jsonl'), 'R4-O provenance mismatch')
    core.need(prov.get('source_sha256')==rep.get('source_sha256'),'R4-O source SHA mismatch')
    core.need(all(core.sha(ROOT/'road_vrp7_step6c1_r4o'/name)==v for name,v in prov['code_sha256'].items()),'R4-O code changed')
    cost=Path(args.cost_dir)/'edge_index.csv'
    taskdir=Path(args.task_dir)
    core.need(cost.is_file() and (taskdir/'manifest.json').is_file(),'Missing source road static graph or task manifest')
    cap1=core.load_json(Path(args.old_capture)/'manifest.json')
    cap2=core.load_json(Path(args.new_capture)/'manifest.json')
    mf=core.load_json(taskdir/'manifest.json')
    edghash=core.sha(cost)
    core.need(cap1['split']==cap2['split']=='train' and cap1['edge_sha256']==cap2['edge_sha256']==edghash
              and mf['cost_edge_index_sha256']==edghash,'Static graph SHA disagrees with captured source')
    tasks={}
    for i in range(4):
        t=taskdir/f'task_{i:04d}.json'
        core.need(t.is_file(),'Missing Train task '+str(t))
        z=core.load_json(t)
        core.need(z['task_id']==f'task_{i:04d}','Wrong task identity')
        # Original experiments use ONLY these Train scenarios.
        for sid in ('2025-07-02_0800','2025-07-09_1700'):
            items=[s for s in z['scenarios'] if s['scenario_id']==sid]
            core.need(len(items)==1 and items[0]['split']=='train' and items[0]['eligible'] is True,
                      'Non-Train/ineligible task scenario')
        tasks[z['task_id']]=z
    core.need(set(z['task'] for z in records)==set(tasks),'Unexpected Train record tasks')
    evidence={'r4o_report':core.sha(d/'report.json'),'r4o_predictions':core.sha(d/'fold_predictions.jsonl'),
        'r4o_provenance':core.sha(d/'provenance.json'), 'static_edge_index':edghash,
        'task_manifest':core.sha(taskdir/'manifest.json'),
        'task_json':{task:core.sha(taskdir/(task+'.json')) for task in sorted(tasks)},
        'old_capture_manifest':core.sha(Path(args.old_capture)/'manifest.json'),
        'new_capture_manifest':core.sha(Path(args.new_capture)/'manifest.json'),
        'original_source_hashes':extra}
    return records,tasks,evidence


def execute(a):
    out=Path(a.output_dir) if a.output_dir else None
    if a.mode=='probe':core.need(out is not None and not out.exists(),'New independent STUDY_NAME required')
    records,tasks,evidence=validate_inputs(a)
    print('R4-P preflight passed: TRAIN 1572 actions/200 snapshots, R4O pinned, static edge SHA verified',flush=True)
    if a.mode=='preflight':return
    old_snaps,new_snaps,old_actions,new_actions=p4o.action_maps(a,records)
    enrich=o.attach_context(records,old_snaps,new_snaps,old_actions,new_actions)
    smap={o.sid(s):s for s in old_snaps+new_snaps}
    actions={**old_actions,**new_actions}
    packed=[]
    for r in enrich:
        k=tuple(r['snap']);packed.append({**r,'snapshot':smap[k],'action':actions[k,r['arm']]})
    graph,edges=core.graph_csv(Path(a.cost_dir)/'edge_index.csv')
    statics={t:core.task_od(obj,graph) for t,obj in sorted(tasks.items())}
    # Make sure all customer IDs in the captured solutions are the source task IDs.
    for k,s in smap.items():
        wanted=set(statics[k[0]]['id_to_idx'])
        actual=[int(cid) for rr in s['current_routes'] for cid in rr]
        core.need(set(actual)==wanted and len(actual)==len(wanted), 'Static OD/capture customer IDs disagree')
    fused=core.attach(packed,statics)
    split=l.make_splits(fused)
    core.need(len(split)==6,'Train splits changed')
    folds,details=core.evaluate(fused,split,o)
    core.need(len(folds)==6*3*2*5,'Wrong fold count')
    provenance={**evidence,'r4p_code':{'r4p_core.py':core.sha(Path(__file__).with_name('r4p_core.py')),
                                       'probe_r4p.py':core.sha(__file__)}}
    doc={'stage':core.STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
        'eval_semantics':'STATIC_ROAD_GRAPH_OFFLINE_FIXED_K500_NO_ONLINE_ALNS',
        'source_commit_reviewed':'453437f25aeda6c0c2a77d2b49516d8b504c30bc',
        'n_actions':len(fused),'n_snapshots':len(smap), 'static_edges':edges,
        'features':{'route_context':38,'road_concat':50,'road_conditioned':62},
        'modality':'static road shortest-path distance ONLY; no dynamic traffic, FIFO NPZ, Oracle, or Road.replay features',
        'model':'R4-O weighted pairwise ridge with fixed static-road cross features; NOT neural FiLM',
        'splits':[name for name,_,_,_ in split],
        'policies':['C3','pairwise_ridge_gate','learned_gate_pairwise','oracle_gate_pairwise','sampled_oracle'],
        'oracle_policies_non_deployable':['oracle_gate_pairwise','sampled_oracle'],
        'source_sha256':provenance,'folds':folds,
        'limitations':['Only 4 Train tasks, not independent validation','Static graph is not real-time traffic',
          'Ridge conditional interactions are NOT official FiLM or contrastive learning',
          'No 30s online ALNS improvement claim','No heldout selection of mode/hyperparameters']}
    out.mkdir(parents=True,exist_ok=False)
    core.save(out/'report.json',doc)
    (out/'report.md').write_text(core.render(doc),encoding='utf8')
    with (out/'fold_predictions.jsonl').open('w',encoding='utf8') as f:
        for z in details:f.write(json.dumps(z,ensure_ascii=False,allow_nan=False)+'\n')
    # Save ONLY static distance matrices; never store or read dynamic/hindsight costs.
    static_dir=out/'static_od';static_dir.mkdir(exist_ok=False)
    for t,z in statics.items():
        npz=static_dir/(t+'.npz')
        customer_ids=[cid for cid,idx in sorted(z['id_to_idx'].items(),key=lambda kv:kv[1])]
        np.savez_compressed(npz,distance_static=z['dist'],customer_ids=np.array(customer_ids,dtype=np.int64))
    core.save(out/'provenance.json',{'stage':core.STAGE,
         'report_sha256':core.sha(out/'report.json'),
         'predictions_sha256':core.sha(out/'fold_predictions.jsonl'),
         'source_sha256':provenance,
         'static_od_sha256':{p.name:core.sha(p) for p in sorted(static_dir.glob('*.npz'))}})
    print('R4-P COMPLETE',out,'folds',len(folds),flush=True)


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
