#!/usr/bin/env python3
"""Deploy VQ6 using the original R2 shadow-C3 hook; no edits to frozen ALNS.

Real-RNG C3 action is committed on fallback; all model costs are inside 30 sec.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parents[1]
for f in ('road_vrp7_step6c1_r3_vq6','road_vrp7_step6c1_r2',
          'road_vrp7_step6b2_r','road_vrp7_step6b1_v2',
          'road_vrp7_step6b2','road_vrp7_step6a',
          'road_vrp7_step5b_v3','road_vrp7_stage3'):
    p=str(ROOT/f)
    if p not in sys.path:sys.path.insert(0,p)
import online_r2 as r2
from vq6_policy import VQ6Policy,KINDS
from dataset import sha256


def checked_load(path,mode,method,protocol,source):
    # Frozen base model-source, feature-schema and road consistency checks.
    art=r2.ORIGINAL_LOADER(path,mode,method,protocol,source)
    if art.get('stage')!='R3-VQ6-ONLINE-FROZEN-TRAIN-v1' or not isinstance(art.get('model'),VQ6Policy):
        raise ValueError('Not a trained VQ6 model')
    model=art['model']
    if (mode,method,protocol)!=('F2','P2','task') or model.mode!='F2' or model.threshold!=0.:
        raise ValueError('VQ6 frozen online contract mismatch')
    if art.get('vq6_kind') not in KINDS or model.kind!=art['vq6_kind']:
        raise ValueError('Wrong model arm')
    if art.get('training_horizon')!=50:raise ValueError('Initial online pilot requires H=50 training label')
    expected=art.get('training_code_sha256',{})
    if set(expected)!={'train_vq6.py','vq6_policy.py'}:raise ValueError('Training-code pin incomplete')
    for name,digest in expected.items():
        if sha256(Path(__file__).with_name(name))!=digest:
            raise ValueError('VQ6 training code changed: '+name)
    if sha256(Path(art['r2_source'])/'paired.npz')!=art['r2_dataset_sha256']:
        raise ValueError('Original R2 labels changed')
    collection=Path(art['vq_collection_dir'])
    for name in ('manifest.json','rollouts.jsonl'):
        key='vq_collection_'+('manifest' if name.startswith('manifest') else 'rollouts')+'_sha256'
        if sha256(collection/name)!=art[key]:raise ValueError('VQ4 collected labels mutated: '+name)
    return art


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--output-dir',required=True)
    a,forward=p.parse_known_args()
    if '--control' not in forward or forward[forward.index('--control')+1]!='learned':
        raise ValueError('VQ6 wrapper accepts only learned arms; C3 uses frozen original B2-R')
    if '--model-path' not in forward:raise ValueError('Missing VQ6 model')
    old_loader,oldargv=r2.checked_load,sys.argv
    try:
        r2.checked_load=checked_load
        sys.argv=['online_r2.py','--output-dir',a.output_dir]+forward
        r2.main()
    finally:
        r2.checked_load=old_loader
        sys.argv=oldargv
    model_path=Path(forward[forward.index('--model-path')+1]);art=joblib.load(model_path)
    (Path(a.output_dir)/'vq6_provenance.json').write_text(json.dumps({
        'stage':'R3-VQ6-online-v1','kind':art['vq6_kind'],
        'model_sha256':sha256(model_path),'online_code_sha256':sha256(Path(__file__)),
        'training_code_sha256':art['training_code_sha256'],
        'meaning':'Frozen real-C3 shadow, all feature/prediction walltime inside 30s',
        'limit':'Online chosen actions alter trajectory and are not comparable to exact H-step restart Q'},indent=2)+'\n')

if __name__=='__main__':main()
