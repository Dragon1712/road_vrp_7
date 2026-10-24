"""Pure VQ2 helpers. All outcomes are retrospective, never online model inputs."""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

STAGE='R3-VQ2-FAIR-LOTO-v1'
METHODS=('C3','E0_FULL','E0_MATCHED','VQ_MEAN',
         'BLEND25','BLEND50','RESID_OOF25','RESID_OOF50','RESID_OOF100',
         'VQ_LCB10','VQ_LCB25')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def fit_extra(rows,targets,*,seed,trees=160,min_leaf=3,max_depth=12,weights=None):
    if len(rows)<12 or len(rows)!=len(targets):raise ValueError('Too few rows or length mismatch')
    x=np.asarray(rows,dtype=np.float32);y=np.asarray(targets,dtype=float)
    if x.ndim!=2 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Bad training numeric data')
    m=ExtraTreesRegressor(n_estimators=trees,max_depth=max_depth,min_samples_leaf=min_leaf,
          max_features=.8,n_jobs=1,random_state=int(seed))
    m.fit(x,y,sample_weight=weights)
    return m

def select(scores, mask,threshold=0.):
    x=np.asarray(scores,float);v=np.asarray(mask,bool)
    if x.shape!=v.shape or x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Scores/mask mismatch')
    if not v.any():return -1
    j=int(np.argmax(np.where(v,x,-np.inf)))
    return j if float(x[j])>float(threshold) else -1

def one_step_rows(X,B,Y,BY,valid,same,groups,roles,taskset,features_fn):
    """Returns list of records with one row per VALID DISTINCT action, train tasks only."""
    records=[]
    for i,g in enumerate(groups):
        task=str(g['task_id'])
        if task not in taskset:continue
        if roles[task]!='train':raise AssertionError('Attempt to access nontrain one-step labels')
        indices=np.flatnonzero(valid[i]&(~same[i]))
        if not len(indices):continue
        ff=features_fn(X[i,:,:26],B[i,:26])
        for j in indices:
            records.append({'task':task,'decision':g['decision_id'],'action':int(j),
                'x':ff[j],'y':float(Y[i,j]-BY[i]),'weight':1./len(indices)})
    return records

def rows_for_selected(selection,groups,lookup,X,B,Y,BY,valid,same,measured,h,features_fn):
    """Measured all candidate labels at the same selected states (no C3 duplicate)."""
    records=[]
    for di in selection:
        i=lookup[di];g=groups[i]
        idx=np.flatnonzero(valid[i]&(~same[i]))
        ff=features_fn(X[i,:,:26],B[i,:26])
        for j in idx:
            z=measured[(di,int(h),int(j))]
            if z['sd_gain'] is None or not np.isfinite(z['mean_gain']) or not np.isfinite(z['sd_gain']):
                raise ValueError('Missing multi-step uncertainty label')
            records.append({'task':str(g['task_id']),'decision':di,'action':int(j),
                'x':ff[j],'one':float(Y[i,j]-BY[i]),'mean':float(z['mean_gain']),
                'sd':float(z['sd_gain']),'weight':1./len(idx)})
    return records

def train_arrays(records,target,taskset=None):
    rows=[z for z in records if taskset is None or z['task'] in taskset]
    if not rows:raise ValueError('Empty train subset')
    return np.stack([r['x'] for r in rows]),np.asarray([r[target] for r in rows],float),np.asarray([r['weight'] for r in rows],float)

def oof_residual_training(one_step_records,multi_records,train_tasks,*,seed,trees):
    """Task-disjoint OOF baseline; residual target = H-mean minus OOF ONE-STEP model.

    For each selected training task, train OOF baseline on OTHER train tasks only.
    The outer held-out evaluation task is excluded from ALL fits by caller.
    """
    result=[];fits=[]
    for k,task in enumerate(sorted(train_tasks)):
        other=set(train_tasks)-{task}
        if len(other)<2:raise ValueError('Need >=3 training tasks for inner leave-one-task-out')
        if any(z['task']==task for z in one_step_records if z['task'] in other):raise AssertionError('Inner leakage')
        x,y,w=train_arrays(one_step_records,'y',other)
        model=fit_extra(x,y,seed=seed+1000+k,trees=trees)
        held=[z for z in multi_records if z['task']==task]
        if not held:raise ValueError('Selected state missing on OOF task')
        pp=model.predict(np.stack([z['x'] for z in held]))
        for z,pred in zip(held,pp):
            item=dict(z);item['residual']=float(z['mean']-pred);item['oof_step_pred']=float(pred)
            result.append(item)
        fits.append({'oof_task':task,'fit_tasks':sorted(other),'n_one_step_training_actions':len(y),
                    'n_selected_actions':len(held)})
    if len(result)!=len(multi_records):raise ValueError('OOF residual count mismatch')
    return result,fits

def metrics_for_decisions(rows):
    if not rows:raise ValueError('Empty evaluation rows')
    gain=np.array([r['realized_mean_road_gain'] for r in rows],float)
    regret=np.array([r['regret'] for r in rows],float)
    override=np.array([r['selected_index']!=-1 for r in rows],bool)
    return {'n_states':len(rows),'mean_gain_sec':float(gain.mean()),
            'mean_regret_sec':float(regret.mean()),'top1_hit':float(np.mean(regret<=1e-6)),
            'override_rate':float(override.mean()),'overrides':int(override.sum()),
            'negative_overrides':int(((gain < -1e-6)&override).sum()),
            'harm_gt40_per_state':float(np.mean(gain < -40.)),
            'harm_gt40_per_override':float(np.sum((gain< -40.)&override)/max(1,int(override.sum()))),
            'mean_excess_harm_gt40':float(np.maximum(-gain-40.,0).mean())}

def diagnostic_for_candidates(rows):
    if not rows:raise ValueError('Empty candidate predictions')
    truth=np.array([z['realized_mean_road_gain'] for z in rows]);one=np.array([z['one_step_gain'] for z in rows]);
    sd=np.array([z['realized_restart_sd'] for z in rows]);sp=np.array([z['predicted_restart_sd'] for z in rows]);
    out={'n_candidate_actions':len(rows),'mean_true_restart_sd_sec':float(sd.mean()),
         'mean_pred_restart_sd_sec':float(sp.mean()),'sd_mae_sec':float(np.mean(np.abs(sd-sp))),
         'fraction_true_sd_gt_abs_gain':float(np.mean(sd>np.abs(truth))),
         'one_step_vs_multistep_sign_flip_frac':float(np.mean(one*truth<0))}
    for n,field in [('E0_FULL','pred_e0_full'),('E0_MATCHED','pred_e0_matched'),
                    ('VQ_MEAN','pred_vq_mean'),('RESID_OOF50','pred_resid_oof50')]:
        p=np.array([z[field] for z in rows]);err=p-truth
        out[n]={'mae_sec':float(np.mean(np.abs(err))),'rmse_sec':float(np.sqrt(np.mean(err**2))),
                'bias_sec':float(err.mean()),'positive_sign_accuracy':float(np.mean((p>0)==(truth>0)))}
    err=np.array([z['pred_vq_mean'] for z in rows])-truth
    # Descriptive only: model error vs predicted stochastic restart SD is NOT a valid confidence interval.
    out['frac_abs_VQ_prediction_error_within_predicted_restart_sd']=float(np.mean(np.abs(err)<=sp))
    return out

def task_macro(per_task):
    keys=('mean_gain_sec','mean_regret_sec','top1_hit','override_rate',
          'harm_gt40_per_state','harm_gt40_per_override','mean_excess_harm_gt40')
    return {m:{k:float(np.mean([per_task[t][m][k] for t in per_task])) for k in keys}
            for m in METHODS}
