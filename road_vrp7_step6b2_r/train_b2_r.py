#!/usr/bin/env python3
"""Strict F0-F5 × P2/P4 ablation on Step6-A v2, no test-set tuning."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path as _Path
_B2_DIR=_Path(__file__).resolve().parents[1]/'road_vrp7_step6b2'
if str(_B2_DIR) not in sys.path:sys.path.insert(0,str(_B2_DIR))
from pathlib import Path
from collections import defaultdict
import numpy as np
import joblib
from features import ROOT,CORE,OLD_HIST,B2_SUFFIX,FEATURE_MODES,TRAFFIC,RELIABILITY
from dataset import load,masks,select_rows,sha256,PRIMARY,ORACLE
from models import Predictor
from train import choose,summarize,tune_threshold,test_metrics,csv_write

METHODS={'P2':'reg','P4':'two'}

def read_extra(d,folder):
    folder=Path(folder)
    mf=json.loads((folder/'run_config.json').read_text(encoding='utf-8'))
    su=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if mf.get('stage')!='step6b2-features-v1' or su.get('status')!='COMPLETE':
        raise ValueError('Step6-B2 feature export incomplete or wrong version')
    for key,path in d.source_files.items():
        if sha256(path)!=mf['source_sha256'][key]:raise ValueError('Changed frozen source '+key)
    path=folder/'boundary_X.npz'
    if sha256(path)!=su['feature_npz_sha256']:raise ValueError('Boundary NPZ checksum differs')
    with np.load(path,allow_pickle=False) as z:x=np.asarray(z['X'],dtype=float)
    if x.shape!=(d.n,len(B2_SUFFIX)) or mf['feature_names']!=list(B2_SUFFIX):
        raise ValueError('Feature labels/shape differ from schema')
    if not np.isfinite(x).all():raise ValueError('Nonfinite augmented features')
    return x,mf,sha256(path)

def get_features(d,extra,mode):
    if mode not in FEATURE_MODES:raise ValueError(mode)
    # OLD history explicitly used only in F1; F2-F5 evaluate boundary information.
    names=FEATURE_MODES[mode]
    b={n:d.X[:,i] for i,n in enumerate(d.names)}
    b.update({n:extra[:,i] for i,n in enumerate(B2_SUFFIX)})
    if any(n not in b for n in names):raise ValueError('Missing feature')
    return np.column_stack([b[n] for n in names]),list(names)

def shuffle_other_date(X,names,groups,mask,seed):
    """F5: same-task, other-date donor, WHOLE state-wise Section history swap.

    Features/labels never cross train-val-test boundaries. Keep all static
    geometry fixed. The outcome is an explicit NEGATIVE CONTROL, not a model.
    """
    ids=np.flatnonzero(mask)
    selected=[groups[i] for i in ids]
    by_state=defaultdict(list)
    for local,g in enumerate(selected):by_state[g['state_id']].append(local)
    dates={s:selected[idx[0]]['date'] for s,idx in by_state.items()}
    tasks={s:selected[idx[0]]['task_id'] for s,idx in by_state.items()}
    if len(set(dates.values()))<2:
        raise ValueError('F5 other-date shuffle invalid for split with just ONE DATE')
    out=X.copy();rng=np.random.default_rng(seed)
    indices=[names.index(n) for n in TRAFFIC+RELIABILITY]
    transferred=0;all_states=0
    for sid,positions in sorted(by_state.items()):
        all_states+=1
        donors=[key for key,loc in by_state.items()
                if tasks[key]==tasks[sid] and dates[key]!=dates[sid] and len(loc)==len(positions)]
        if not donors:continue
        donor=by_state[donors[int(rng.integers(len(donors)))]]
        # Sort by original position/order within frozen state; do not use labels.
        out[np.ix_(ids[positions],indices)]=X[np.ix_(ids[donor],indices)]
        transferred+=1
    return out,{'states':all_states,'donor_mapped_states':transferred,
                'donor_fraction':transferred/max(all_states,1),
                'within_split_only':True,'same_task_other_date':True}

def study(source,features,output,protocols,seed=42,trees=120,methods=('P2','P4'),modes=('F0','F1','F2','F3','F4','F5')):
    source,features,output=map(Path,(source,features,output))
    if output.exists():raise FileExistsError('Use an independent NEW training study: '+str(output))
    if len(set(methods))!=len(methods) or any(x not in METHODS for x in methods):raise ValueError('Invalid method')
    if len(set(modes))!=len(modes) or any(x not in FEATURE_MODES for x in modes):raise ValueError('Invalid feature mode')
    d=load(source);extra,mf,xhash=read_extra(d,features)
    for protocol in protocols:masks(d,protocol)
    output.mkdir(parents=True,exist_ok=False)
    manifest={'stage':'step6b2r-ablation-v1',
      'code_sha256':{path.name:sha256(path) for path in (Path(__file__),
          ROOT/'road_vrp7_step6b2/features.py',
          ROOT/'road_vrp7_step6b1_v2/models.py',
          ROOT/'road_vrp7_step6b1_v2/dataset.py',
          ROOT/'road_vrp7_step6b1_v2/train.py')},
      'seed':seed,'trees':trees,'methods':list(methods),
      'modes':list(modes),'protocols':list(protocols),'features_source':str(features),
      'frozen_source_sha256':mf['source_sha256'],'boundary_X_sha256':xhash,
      'source_label':PRIMARY,'label_scope':'retrospective TD cost of Static Repair',
      'notes':['F5 only valid for task protocol with multiple dates in EACH split.',
      'No test set labels are used for threshold selection.',
      'Repeated training seeds reuse identical held-out tasks/dates; descriptive only.']}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    all_reports={};y=d.label(PRIMARY);ora=d.label(ORACLE)
    for protocol in protocols:
        reports={}  # critical: never leak task-protocol F5 into date report.
        masks_=masks(d,protocol);folder=output/protocol;folder.mkdir()
        for mode in modes:
            if mode=='F5' and protocol!='task':continue
            X,names=get_features(d,extra,mode)
            # Test-of-meaning: F5 changes traffic assignment in each partition.
            permutation={}
            if mode=='F5':
                for role,mask in masks_.items():
                    X,permutation[role]=shuffle_other_date(X,names,d.groups,mask,
                                                            seed+{'train':11,'val':19,'internal_test':29}[role])
                    if permutation[role]['donor_fraction']<.5:
                        raise ValueError('F5 insufficient cross-date donors: '+str(permutation[role]))
            for method in methods:
                name=method+'_'+mode;kind=METHODS[method]
                tr=masks_['train'];va=masks_['val'];te=masks_['internal_test']
                model=Predictor(kind,seed,trees,names,protocol)
                model.fit(X[tr],y[tr],select_rows(d,tr))
                rank_v,gate_v,_=model.predict(X[va])
                th,vreport=tune_threshold(y[va],ora[va],rank_v,gate_v,select_rows(d,va),kind)
                rank_t,gate_t,prob_t=model.predict(X[te])
                rows=choose(y[te],ora[te],rank_t,gate_t,select_rows(d,te),th)
                csv_write(folder/f'{name}_test_states.csv',rows)
                group_test=select_rows(d,te)
                csv_write(folder/f'{name}_test_candidates.csv',[{'sample_id':g['sample_id'],
                    'state_id':g['state_id'],'task_id':g['task_id'],'date':g['date'],
                    'rank_score':float(r),'gate_score':float(gt),
                    'realized_static_repair_td_gain_sec':float(t)}
                    for g,r,gt,t in zip(group_test,rank_t,gate_t,y[te])])
                artifact={'model':model,'method':method,'mode':mode,'feature_names':names,
                    'seed':seed,'protocol':protocol,'threshold':th,'source_sha256':mf['source_sha256'],
                    'boundary_X_sha256':xhash,'training_label':PRIMARY,
                    'online_compatible':mode!='F5'}
                joblib.dump(artifact,folder/f'{name}_model.joblib',compress=3)
                reports[name]={'mode':mode,'kind':kind,'features':names,'validation_threshold':th,
                    'validation_policy':vreport,'test_policy':summarize(rows),
                    'test_metrics':test_metrics(y[te],rank_t,prob_t,group_test,kind),
                    'permutation':permutation if mode=='F5' else None}
                print(protocol,name,'signed_gain_sec/state',round(reports[name]['test_policy']['mean_signed_gain_sec'],3),flush=True)
        (folder/'report.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        all_reports[protocol]=list(reports)
    (output/'summary.json').write_text(json.dumps({'status':'COMPLETE','protocols':list(protocols),
                  'method_count':len(methods),'requested_mode_count':len(modes),
                  'actual_methods_by_protocol':all_reports},indent=2)+'\n')
    return all_reports

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--features-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--protocol',choices=('task','date','both'),default='both')
    p.add_argument('--methods',default='P2,P4');p.add_argument('--modes',default='F0,F1,F2,F3,F4,F5')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--trees',type=int,default=120)
    a=p.parse_args()
    study(a.source_dir,a.features_dir,a.output_dir,['task','date'] if a.protocol=='both' else [a.protocol],
          a.seed,a.trees,a.methods.split(','),a.modes.split(','))
if __name__=='__main__':main()
