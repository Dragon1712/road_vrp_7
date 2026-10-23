#!/usr/bin/env python3
"""Deploy R3 predictors through the unchanged official R2 shadow-C3 online engine.

No search operator, static repair, RNG or wall-budget changes.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r3_tricks','road_vrp7_step6c1_r2',
               'road_vrp7_step6b2_r','road_vrp7_step6b1_v2','road_vrp7_step6b2',
               'road_vrp7_step6a','road_vrp7_step5b_v3','road_vrp7_stage3'):
    p=str(ROOT/folder)
    if p not in sys.path:sys.path.insert(0,p)
import online_r2 as r2
from r3_models import R3Advantage
from r2_policy import R2Advantage
from dataset import sha256

ORIGINAL_R2_CHECKED_LOAD=r2.checked_load


def checked_load(path,mode,method,protocol,source):
    artifact=r2.ORIGINAL_LOADER(path,mode,method,protocol,source)
    model=artifact.get('model')
    if artifact.get('stage')=='step6c1r2-c3-advantage-train-v1':
        # Explicit frozen original R2 control (not a new trained R3 model).
        return ORIGINAL_R2_CHECKED_LOAD(path,mode,method,protocol,source)
    if artifact.get('stage')!='step6c1r3t-tricks-training-v1':
        raise ValueError('Only frozen R2 or R3 first-batch models may be used')
    if not isinstance(model,R3Advantage) or not isinstance(model,R2Advantage) or model.mode!='F2' or mode!='F2':
        raise ValueError('R3 predictor type or F2 schema mismatch')
    if method!='P2' or protocol!=artifact['protocol']:
        raise ValueError('Wrong method/protocol')
    for name,digest in artifact.get('r3_training_code_sha256',{}).items():
        if name not in ('r3_models.py','decision_metrics.py','train_r3.py'):
            raise ValueError('Unexpected R3 pinned file')
        if sha256(Path(__file__).with_name(name))!=digest:
            raise ValueError(f'R3 code changed since training ({name}); retrain in a NEW study dir')
    expected=artifact.get('r3_training_code_sha256',{})
    if set(expected)!=set(('r3_models.py','decision_metrics.py','train_r3.py')):
        raise ValueError('Incomplete R3 code manifest')
    if sha256(Path(artifact['r2_source'])/'paired.npz')!=artifact['r2_dataset_sha256']:
        raise ValueError('R3 trained label dataset changed')
    if artifact.get('r3_arm') not in ('T1_CB_RMSE','T2_CB_HUBER','T3_PAIRWISE',
        'T4_WEIGHTED_PAIR','T5_HUBER_MARGIN','T6_WEIGHTED_MARGIN','T7_BLEND_MARGIN'):
        raise ValueError('Unknown R3 model arm')
    return artifact


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--output-dir',required=True)
    args,forward=p.parse_known_args()
    old_checked_load=r2.checked_load
    old_argv=sys.argv
    try:
        r2.checked_load=checked_load
        sys.argv=['online_r2.py','--output-dir',args.output_dir]+forward
        r2.main()
    finally:
        r2.checked_load=old_checked_load
        sys.argv=old_argv
    import joblib
    model_path=Path(forward[forward.index('--model-path')+1])
    art=joblib.load(model_path)
    (Path(args.output_dir)/'r3_provenance.json').write_text(json.dumps({
        'stage':'step6c1r3t-online-v1','arm':art.get('r3_arm','T0_R2_F2'),
        'model_sha256':sha256(model_path),
        'code_sha256':sha256(Path(__file__)),
        'train_source_sha256':art.get('r3_training_code_sha256'),
        'note':'Real frozen C3 shadow action; no future TD inputs; within original 30-sec solver budget.'
    },indent=2)+'\n')

if __name__=='__main__':main()
