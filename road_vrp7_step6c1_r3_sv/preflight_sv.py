#!/usr/bin/env python3
"""Fast server-side file and artifact sanity check before expensive rollouts."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for module_dir in ('road_vrp7_step6c1_r3_sv','road_vrp7_step6c1_r3_tricks',
                   'road_vrp7_step6c1_r2','road_vrp7_step6b1_v2'):
    if str(ROOT/module_dir) not in sys.path:sys.path.insert(0,str(ROOT/module_dir))
from sv_core import sha256


def check(label_dir,r2,huber,m0):
    d=Path(label_dir)
    needed=[d/'manifest.json',d/'paired.npz',d/'groups.jsonl',Path(r2),Path(huber),Path(m0),
       ROOT/'road_vrp7_step5b_v3/run_experiment.py',ROOT/'road_vrp7_step5b_v3/section_v2.py',
       ROOT/'road_vrp7_stage3/road_vrp.py']
    missing=[str(x) for x in needed if not x.is_file()]
    if missing:raise FileNotFoundError('Missing required files:\n'+'\n'.join(missing))
    meta=json.loads((d/'manifest.json').read_text())
    if meta['stage']!='step6c1r2-real-C3-labelled-v1' or sha256(d/'paired.npz')!=meta['pair_sha256'] or sha256(d/'groups.jsonl')!=meta['groups_sha256']:
        raise ValueError('Failed C3 labels manifest SHA check')
    cap=Path(meta['source_dir'])
    if not (cap/'states.jsonl').is_file():raise FileNotFoundError('Missing R2 capture file '+str(cap/'states.jsonl'))
    if sha256(cap/'states.jsonl')!=meta['capture_state_sha256']:
        raise ValueError('R2 capture SHA changed')
    import joblib
    old,new,static=(joblib.load(r2),joblib.load(huber),joblib.load(m0))
    if old.get('stage')!='step6c1r2-c3-advantage-train-v1' or new.get('stage')!='step6c1r3t-tricks-training-v1' or new.get('r3_arm')!='T2_CB_HUBER':
        raise ValueError('Wrong R2 / T2 model identity')
    if new.get('r2_dataset_sha256')!=meta['pair_sha256'] or old.get('r2_dataset_sha256')!=meta['pair_sha256']:
        raise ValueError('R2/T2 label sources inconsistent')
    if static.get('mode')!='F2':raise ValueError('Expected M0 P2/F2 artifact')
    details={'status':'READY','c3_label_sha':meta['pair_sha256'],
            'capture_sha':meta['capture_state_sha256'],
            'model_shas':{'r2':sha256(r2),'huber':sha256(huber),'m0':sha256(m0)},
            'capture_count':sum(1 for line in (cap/'states.jsonl').open() if line.strip()),
            'label_groups_count':sum(1 for line in (d/'groups.jsonl').open() if line.strip())}
    print(json.dumps(details,indent=2,ensure_ascii=False));return details

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',default='runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1')
    p.add_argument('--r2-model',default='runs/chicago_july_v1/step6c1_r2/r2_task_seed42_v1/R2_F2.joblib')
    p.add_argument('--huber-model',default='runs/chicago_july_v1/step6c1_r3_tricks/r3_seed42_v1/T2_CB_HUBER.joblib')
    p.add_argument('--m0-model',default='runs/chicago_july_v1/step6b2/b2_full_seed42_v1/task/P2_F2_model.joblib')
    args=p.parse_args()
    check(args.dataset_dir,args.r2_model,args.huber_model,args.m0_model)
