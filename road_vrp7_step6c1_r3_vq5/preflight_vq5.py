#!/usr/bin/env python3
"""Read-only VQ5 preflight: source SHA, fixed panel, original Train-role separation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for f in ('road_vrp7_step6c1_r3_vq5','road_vrp7_step6c1_r3_vq',
          'road_vrp7_step6c1_r2'):
    p=str(ROOT/f)
    if p not in sys.path:sys.path.insert(0,p)
from vq5_core import read_json,sha256,check_balanced_prefix
from r2_policy import group_split

def check(collection,dataset,sizes):
    c=Path(collection);d=Path(dataset)
    mf=read_json(c/'manifest.json');rep=read_json(c/'report.json');src=read_json(d/'manifest.json')
    if mf.get('stage')!='R3-VQ-train-only-restart-v1' or rep.get('status')!='COMPLETE' or rep.get('stage')!='R3-VQ-labels-v1':
        raise ValueError('Incomplete/wrong VQ4 collection')
    if src.get('stage')!='step6c1r2-real-C3-labelled-v1':raise ValueError('Wrong original R2 labels')
    if sha256(d/'paired.npz')!=src['pair_sha256'] or sha256(d/'groups.jsonl')!=src['groups_sha256']:
        raise ValueError('Changed R2 original labels')
    if src['pair_sha256']!=mf['label_sha256'] or src['groups_sha256']!=mf['group_sha256']:
        raise ValueError('Collection/data SHA mismatch')
    groups=[json.loads(x) for x in (d/'groups.jsonl').read_text().splitlines() if x.strip()]
    _,roles=group_split(groups,'task');lookup={z['decision_id']:i for i,z in enumerate(groups)}
    ids=mf['selected_decision_ids']
    tasks=check_balanced_prefix(ids,groups,lookup,roles,sizes)
    if len(ids)!=rep['n_selected_states'] or set(map(int,mf['horizons']))!={20,50} or int(mf['replicates'])!=3:
        raise ValueError('Unexpected panel, H or replicate config')
    return {'status':'OK','n_states':len(ids),'sizes':list(sizes),'task_ids':tasks,
            'per_heldout_task_eval_states':len(ids)//4,
            'train_states_per_fold':[3*n//4 for n in sizes],
            'read_only':True,'source_sha256':sha256(c/'manifest.json')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection-dir',required=True);p.add_argument('--dataset-dir',required=True);p.add_argument('--sizes',default='48,96,144,192');a=p.parse_args()
    print(json.dumps(check(a.collection_dir,a.dataset_dir,tuple(map(int,a.sizes.split(',')))),indent=2))
