#!/usr/bin/env python3
"""Step6-B3-A: anchored residual traffic + SPIN-inspired coverage + FiLM.

All feature/label source files reused from frozen Step6-A v2 and Step6-B2.
Training uses train split only; alpha tuned on validation split only;
internal_test scores never affect hyperparameters or threshold. The base P2-F2
artifact is independently trained and FROZEN. Alpha=0 is allowed (safe
validation selection fallback), so failure to beat M0 is an actual finding.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1]
for rel in ('road_vrp7_step6b1_v2','road_vrp7_step6b2'):
    x=str(ROOT/rel)
    if x not in sys.path:sys.path.insert(0,x)
from dataset import load,masks,select_rows,sha256,PRIMARY,ORACLE
from train_b2 import read_extra,get_features
from features import FEATURE_MODES
from train import choose,summarize,test_metrics,csv_write
from model import FEATURES,STRUCTURE,TRAFFIC,RELIABILITY,ResidualFusionPredictor,group_oof_anchor,from_full

DEFAULT_SOURCE=Path('runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
DEFAULT_FEATURES=Path('runs/chicago_july_v1/step6b2/features_boundary_v1')
METHODS=('M0','M1','M2','M3','M4')
ALPHAS=(0.,0.1,0.25,0.5,1.,1.5)


def read_anchor(study,protocol,source_mf,boundary_sha,seed):
    path=Path(study)/protocol/'P2_F2_model.joblib'
    if not path.is_file():raise FileNotFoundError(path)
    a=joblib.load(path)   # only load model artifacts from YOUR trusted training runs.
    if a.get('method')!='P2' or a.get('mode')!='F2' or a.get('protocol')!=protocol:
        raise ValueError('Anchor not a P2-F2 model for this split')
    if 'seed' in a and int(a['seed'])!=int(seed):
        raise ValueError('Frozen anchor training seed mismatch')
    if a.get('training_label')!=PRIMARY or a.get('feature_names')!=list(STRUCTURE):
        raise ValueError('Anchor is incompatible with expected feature or labels')
    if a.get('source_sha256')!=source_mf or a.get('boundary_X_sha256')!=boundary_sha:
        raise ValueError('Anchor source features differ from current data')
    return a,path


def optimize_alpha(model,Xval,y,ora,g,threshold):
    # Only validation rows, NO test threshold search. Frozen P2-F2 threshold
    # is held fixed for all arms to isolate the traffic fusion changes.
    previous=model.alpha
    best=None
    for alpha in ALPHAS:
        model.alpha=alpha
        rank,gate,_=model.predict(Xval)
        report=summarize(choose(y,ora,rank,gate,g,threshold))
        # Conservative tie break: prefer LESS correction to static anchor.
        key=(report['mean_signed_gain_sec'],-alpha)
        if best is None or key>best[0]:best=(key,alpha,report)
    model.alpha=best[1]
    return best[1],best[2]


def preflight(d,extra,manifest):
    if tuple(FEATURE_MODES['F4'])!=tuple(FEATURES):
        raise ValueError('B2 F4 feature order changed')
    X,n=get_features(d,extra,'F4')
    if list(n)!=list(FEATURES) or X.shape!=(d.n,44):raise ValueError('B2 features differ from B3 schema')
    if manifest['stage']!='step6b2-features-v1':raise ValueError('Wrong boundary extraction')
    return X


def run(source,features,anchor_study,output,protocols=('task','date'),methods=METHODS,
        seed=42,epochs=55,trees=120):
    source,features,anchor_study,output=map(Path,(source,features,anchor_study,output))
    if output.exists():raise FileExistsError('Independent NEW result directory required: '+str(output))
    if not methods or len(set(methods))!=len(methods) or any(m not in METHODS for m in methods):
        raise ValueError('Invalid requested methods')
    d=load(source)
    extra,mf,xhash=read_extra(d,features)
    X=preflight(d,extra,mf)
    y=d.label(PRIMARY); oracle=d.label(ORACLE)
    split_map={p:masks(d,p) for p in protocols}
    anchored={p:read_anchor(anchor_study,p,mf['source_sha256'],xhash,seed) for p in protocols}
    output.mkdir(parents=True,exist_ok=False)
    # Portable provenance includes SOURCE + F2 trained anchor hashes, not files themselves.
    provenance={'stage':'step6b3-anchored-film-v1', 'research_target':'Static Repair counterfactual TD gain',
        'source':str(source),'features':str(features),'anchor_study':str(anchor_study),
        'source_sha256':mf['source_sha256'], 'feature_npz_sha256':xhash,
        'anchor_sha256':{p:sha256(path) for p,(_,path) in anchored.items()},
        'code_sha256':{n:sha256(Path(__file__).with_name(n)) for n in ('model.py','train_b3.py')},
        'film_reference':'literature/code/film-master/vr/models/filmed_net.py',
        'film_license_notice':'FiLM official CC BY-NC 4.0; mathematical idea reimplemented independently, original code not redistributed.',
        'spin_reference':'literature/code/spin-main/spin/models/spin.py',
        'model_distinction':'Coverage gate is SPIN-inspired, NOT official SPIN GNN.',
        'methods':list(methods),'protocols':list(protocols),'seed':seed,'epochs':epochs,
        'baseline_trees_for_group_OOF':trees,'alphas':list(ALPHAS),
        'warnings':['Same July development data as previous B1/B2 tuning; internal_test is not a blind test.',
        'Static Repair labels are historically evaluated with TD objective, not road.replay.',
        'Anchor P2-F2 validation threshold is frozen and shared by all B3 methods.',
        'Feature extraction and FiLM inference must be benchmarked inside 30s online ALNS.',
        'Group OOF residual target uses refitted RF on training folds; final anchor is original frozen P2-F2.']}
    (output/'manifest.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    reports={}
    for protocol,fold in split_map.items():
        tr=fold['train'];va=fold['val'];te=fold['internal_test']
        an,an_path=anchored[protocol]
        base_thr=an['threshold']
        coverage=from_full(X,FEATURES)[-1]
        coverage_report={role:{'n_candidates':int(mask.sum()),
           'gate_zero_rate':float(np.mean(coverage[mask]<=1e-12)),
           'mean_gate':float(np.mean(coverage[mask]))} for role,mask in fold.items()}
        if base_thr is not None and not np.isfinite(base_thr):raise ValueError('Bad anchor threshold')
        folder=output/protocol;folder.mkdir()
        # Fit OOF baseline by task/date group on *training* rows only.
        Xstruct=X[:,[list(FEATURES).index(n) for n in STRUCTURE]]
        train_g=select_rows(d,tr)
        oof=group_oof_anchor(Xstruct[tr],y[tr],train_g,protocol,seed,trees)
        residual_target=y[tr]-oof
        metrics={}
        for method in methods:
            model=ResidualFusionPredictor(an['model'],method,FEATURES,seed)
            model.prepare(X[tr],residual_target,epochs=epochs)
            alpha,vr=optimize_alpha(model,X[va],y[va],oracle[va],select_rows(d,va),base_thr)
            rank_t,gate_t,_=model.predict(X[te])
            records=choose(y[te],oracle[te],rank_t,gate_t,select_rows(d,te),base_thr)
            record=summarize(records)
            tmetrics=test_metrics(y[te],rank_t,None,select_rows(d,te),'reg')
            artifact={'model':model,'mode':'F4','method':'P2','protocol':protocol,
              'b3_method':method,'b3_version':'step6b3-anchored-film-v1','online_compatible':True,
              'feature_names':list(FEATURES),'threshold':base_thr,'alpha':alpha,
              'anchor_sha256':sha256(an_path),'training_label':PRIMARY,
              'source_sha256':mf['source_sha256'],'boundary_X_sha256':xhash,
              'seed':seed,'film_reference':provenance['film_reference']}
            # B2-R online can consume the artifact via --model-mode F4 --model-method P2.
            # M0 should normally be deployed using original P2-F2 to avoid F4 history overhead.
            joblib.dump(artifact,folder/f'{method}_model.joblib',compress=3)
            predictions=[]
            for row,sc,yy,bb in zip(select_rows(d,te),rank_t,y[te],model._anchor(X[te])):
                predictions.append({'sample_id':row['sample_id'],'state_id':row['state_id'],
                 'task_id':row['task_id'],'date':row['date'], 'model_score_sec':float(sc),
                 'static_anchor_score_sec':float(bb),'target_static_repair_td_gain_sec':float(yy),
                 'delta_model_vs_anchor_sec':float(sc-bb)})
            csv_write(folder/f'{method}_test_candidates.csv',predictions)
            csv_write(folder/f'{method}_test_states.csv',records)
            metrics[method]={'alpha_selected_on_validation':alpha,'frozen_threshold':base_thr,
              'validation_policy':vr,'test_policy':record,'test_metrics':tmetrics,
              'anchor_sha256':sha256(an_path),'model_file':f'{method}_model.joblib',
              'label':'static_repair_td_gain_sec'}
            print(f'{protocol:4s} {method:3s} alpha={alpha:<5g} test signed gain: {record["mean_signed_gain_sec"]:.3f} sec/state',flush=True)
        (folder/'reliability_diagnostics.json').write_text(
          json.dumps(coverage_report,ensure_ascii=False,indent=2)+'\n')
        (folder/'report.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        reports[protocol]=metrics
    (output/'summary.json').write_text(json.dumps({'status':'COMPLETE','protocols':list(protocols),
      'method_names':list(methods)},indent=2)+'\n')
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--features-dir',type=Path,default=DEFAULT_FEATURES)
    p.add_argument('--anchor-study',type=Path,default=Path('runs/chicago_july_v1/step6b2/b2_full_seed42_v1'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--protocol',choices=('task','date','both'),default='both')
    p.add_argument('--methods',default=','.join(METHODS))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=55)
    p.add_argument('--trees',type=int,default=120)
    a=p.parse_args()
    if a.epochs<1 or a.trees<5:p.error('invalid training budget')
    run(a.source_dir,a.features_dir,a.anchor_study,a.output_dir,
      ('task','date') if a.protocol=='both' else (a.protocol,),
      tuple(t.strip() for t in a.methods.split(',')),a.seed,a.epochs,a.trees)
if __name__=='__main__':main()
