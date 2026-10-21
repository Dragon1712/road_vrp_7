#!/usr/bin/env python3
"""C1 offline group-wise delegate training, frozen RF anchor and raw road graphs.

D0: static neural delegate, D1: F4 tabular history, D2: masked directed raw OSM graph.
Uses only train data for parameter updates; val for early-stopping; internal_test
is evaluated once after checkpoint selection. Not a blind external test.
"""
from __future__ import annotations
import argparse,csv,copy,hashlib,json,os,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import joblib
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1','road_vrp7_step6b1_v2','road_vrp7_step6b2','road_vrp7_step6b3'):
    p=str(ROOT/folder)
    if p not in sys.path:sys.path.insert(0,p)
from dataset import load,masks,select_rows,PRIMARY,ORACLE,sha256,grouped_indices
from train_b2 import read_extra,get_features
from features import FEATURE_MODES
from train import choose,summarize,test_metrics,csv_write
from train_b3 import read_anchor
from graph_features import GRAPH_DIM,unpack
from c1_model import C1Model,state_loss

SOURCE='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1'
FEATS='runs/chicago_july_v1/step6b2/features_boundary_v1'
GRAPH='runs/chicago_july_v1/step6c1/c1_graph_full_v1'


def by_state(groups,mask):
    selected=np.flatnonzero(mask)
    by=defaultdict(list)
    for idx in selected:by[groups[idx]['state_id']].append(int(idx))
    if not by:raise ValueError('Empty split')
    # Existing Step6A v2 has exactly 8 valid candidates per state. Fail closed.
    n={len(v) for v in by.values()}
    if n!={8}:raise ValueError(f'Expected 8 valid actions/state, found group sizes {n}')
    for state,ind in by.items():
        if len({groups[i]['task_id'] for i in ind})!=1 or len({groups[i]['date'] for i in ind})!=1:
            raise ValueError('Mixed task/date within a candidate set')
    return list(by.values())


def score_groups(model,X,G,groups,mask):
    result=np.full(len(X),np.nan,dtype=np.float64)
    for inds in by_state(groups,mask):
        data=np.concatenate([X[inds],G[inds]],axis=1) if model.variant in ('D2','D2T') else X[inds]
        result[inds]=model.predict(data)[0]
    if not np.isfinite(result[mask]).all():raise ValueError('Incomplete predictions')
    return result[mask]


def pretrain_graph(net,G,train_ids,seed,epochs=5,batch=96):
    if net.graph is None or epochs<=0:return []
    rng=np.random.default_rng(seed+111)
    graph=tuple(torch.from_numpy(a.copy()) for a in unpack(G[train_ids]))
    if net.variant=='D2T':
        seq,meta,adj,valid=graph
        graph=(seq,meta,torch.eye(24).unsqueeze(0)*valid.unsqueeze(-1),valid)
    opt=torch.optim.AdamW(net.graph.parameters(),lr=0.0007,weight_decay=.002)
    log=[]
    net.graph.train()
    for ep in range(epochs):
        loss_values=[]
        for inds in np.array_split(rng.permutation(len(train_ids)),max(1,int(np.ceil(len(train_ids)/batch)))):
            seq,meta,adj,valid=(a[inds] for a in graph)
            original=seq[...,1]>0.5
            held=(torch.rand(original.shape)<0.28)&original
            remaining=original&(~held)
            # Only test reconstructions with at least one supporting observed slot.
            active=held.any(-1)&remaining.any(-1)&(valid>0.5)
            if not active.any():continue
            masked_seq=seq.clone();masked_seq[...,0][held]=0.;masked_seq[...,1][held]=0.
            _,node=net.graph(masked_seq,meta,adj,valid)
            target=(seq[...,0]*held.float()).sum(-1)/(held.sum(-1).clamp(min=1))
            pred=net.graph.node_out(node).squeeze(-1)
            loss=torch.nn.functional.smooth_l1_loss(pred[active],target[active],beta=.5)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.graph.parameters(),1.)
            opt.step();loss_values.append(loss.item())
        log.append(float(np.mean(loss_values)) if loss_values else None)
        print('PRETRAIN sparse masked graph',ep+1,'loss',log[-1],flush=True)
    net.graph.eval()
    return log


def train_one(model,X,G,Y,groups,train_mask,val_mask,seed,epochs=30,lr=.0015,
              pretrain_epochs=4,decision_weight=1.0,train_oof_anchor=None):
    if model.variant in ('D2','D2T'):
        prelog=pretrain_graph(model.net,G,np.flatnonzero(train_mask),seed,pretrain_epochs)
    else:prelog=[]
    tr=by_state(groups,train_mask)
    opt=torch.optim.AdamW(model.net.parameters(),lr=lr,weight_decay=0.02)
    scaler_s=model.static_scaler.transform(X[:,:26]).astype(np.float32)
    scaler_t=model.traffic_scaler.transform(X[:,26:]).astype(np.float32)
    scaler_t[:,10]=np.clip(X[:,36],0,1);scaler_t[:,11]=np.clip(X[:,37],0,1)
    ss=torch.from_numpy(scaler_s);tt=torch.from_numpy(scaler_t)
    gg=tuple(torch.from_numpy(a.copy()) for a in unpack(G)) if model.variant in ('D2','D2T') else None
    if gg is not None and model.variant=='D2T':
        seq,meta,adj,valid=gg
        gg=(seq,meta,torch.eye(24).unsqueeze(0)*valid.unsqueeze(-1),valid)
    yy=torch.from_numpy(np.asarray(Y,np.float32))
    # Supervised decision score includes original frozen static RF as a non-trainable anchor.
    anchor_raw=model.get_anchor(X).copy()
    if train_oof_anchor is not None:
        oof=np.asarray(train_oof_anchor,dtype=np.float32)
        if oof.shape!=(int(train_mask.sum()),) or not np.isfinite(oof).all():
            raise ValueError('OOF anchor must match training rows only')
        anchor_raw[train_mask]=oof
    anchors=torch.from_numpy(anchor_raw)
    rng=np.random.default_rng(seed)
    best=None;best_state=None;history=[]
    val_grp=select_rows_dummy(groups,val_mask)
    # Epoch is selected on validation gain ONLY; test is never inspected.
    for ep in range(epochs):
        model.net.train()
        order=rng.permutation(len(tr))
        losses=[]
        for start in range(0,len(tr),8):
            batch=[tr[j] for j in order[start:start+8]]
            inds=torch.tensor(batch,dtype=torch.long)
            st=ss[inds];traf=tt[inds]
            graph_arg=None if gg is None else tuple(x[inds] for x in gg)
            corrections=model.net(st,traf,graph_arg)
            preds=anchors[inds]+corrections
            loss=torch.stack([state_loss(preds[k],yy[inds][k],decision_weight=decision_weight)
                              for k in range(len(batch))]).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.net.parameters(),2.)
            opt.step();losses.append(float(loss.detach()))
        model.net.eval()
        vscore=score_groups(model,X,G,groups,val_mask)
        rr=summarize(choose(Y[val_mask],Y[val_mask],vscore,vscore,val_grp,0.))
        val_gain=float(rr['mean_signed_gain_sec'])
        history.append({'epoch':ep+1,'training_loss':float(np.mean(losses)),'validation_signed_gain_sec_per_state':val_gain})
        key=(round(val_gain,8),-(ep+1))
        if best is None or key>best:
            best=key;best_state=copy.deepcopy(model.net.state_dict());best_ep=ep+1
        if (ep+1)%5==0 or ep==0:print(model.variant,'epoch',ep+1,'loss',round(np.mean(losses),5),'validation gain',round(val_gain,3),flush=True)
    model.net.load_state_dict(best_state);model.net.eval()
    return history,prelog,best_ep


def select_rows_dummy(groups,mask):return [g for g,take in zip(groups,mask) if take]


def run(a):
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('New independent output directory required: '+str(out))
    data=load(a.source_dir)
    extra,mf,xhash=read_extra(data,a.features_dir)
    X,names=get_features(data,extra,'F4')
    if len(names)!=44 or tuple(names)!=tuple(FEATURE_MODES['F4']):raise ValueError('F4 incompatible')
    graphroot=Path(a.graph_dir)
    summary=json.loads((graphroot/'summary.json').read_text())
    manifest=json.loads((graphroot/'manifest.json').read_text())
    if summary['status']!='COMPLETE' or summary['graph_sha256']!=sha256(graphroot/'graph_X.npz'):
        raise ValueError('Unverified raw graph export')
    if manifest['b2_boundary_sha256']!=xhash or manifest['source_sha256']!=mf['source_sha256']:
        raise ValueError('Graph/label F4 source mismatch')
    with np.load(graphroot/'graph_X.npz',allow_pickle=False) as z:G=np.asarray(z['X'],dtype=np.float32)
    if G.shape!=(data.n,GRAPH_DIM) or not np.isfinite(G).all():raise ValueError('Invalid graph shape')
    y=data.label(PRIMARY)
    ora=data.label(ORACLE)
    protos=['task','date'] if a.protocol=='both' else [a.protocol]
    variants=a.variants.split(',')
    if not variants or len(set(variants))!=len(variants) or any(x not in ('D0','D1','D2T','D2') for x in variants):
        raise ValueError('D0/D1/D2T/D2 variants required')
    out.mkdir(parents=True,exist_ok=False)
    mfout={'stage':'step6c1-delegation-v1','seed':a.seed,'source':a.source_dir,'graph_study':a.graph_dir,
           'anchor_study':a.anchor_study,'source_sha256':mf['source_sha256'],
           'boundary_X_sha256':xhash,'graph_sha256':summary['graph_sha256'],
           'code_sha256':{p:sha256(Path(__file__).with_name(p)) for p in ('c1_model.py','graph_features.py','train_c1.py')},
           'protocols':protos,'variants':variants,'epochs':a.epochs,'pretrain_epochs':a.pretrain_epochs,
           'decision_weight':a.decision_weight,'threshold':0.0,'training_target':PRIMARY,
           'static_anchor_train_mode':'group_oof_on_train; frozen full RF at val/test/online',
           'warnings':['July development: internal_test is NOT an independent blind test.',
             'D2 is a lightweight SPIN-inspired raw-edge graph encoder, NOT official SPIN.',
             'No SPCLT contrastive learning in C1.',
             'Offline no-op has zero gain, online rejection falls back to C3; they differ.',
             'Task/date splits have very few truly independent groups; repeated seeds are not independent.']}
    (out/'manifest.json').write_text(json.dumps(mfout,ensure_ascii=False,indent=2)+'\n')
    all_reports={}
    for protocol in protos:
        fold=masks(data,protocol)
        anchor,apath=read_anchor(a.anchor_study,protocol,mf['source_sha256'],xhash,a.seed)
        folder=out/protocol;folder.mkdir()
        from model import group_oof_anchor as frozen_group_oof_anchor  # existing B3 helper
        train_rows=select_rows(data,fold['train'])
        oof=frozen_group_oof_anchor(X[fold['train'],:26],y[fold['train']],train_rows,
                                    protocol,a.seed,120)
        in_sample=anchor['model'].predict(X[fold['train'],:26])[0]
        report={'_anchor_diagnostics':{
            'oof_train_mae_sec':float(np.mean(abs(oof-y[fold['train']]))),
            'full_rf_train_mae_sec':float(np.mean(abs(in_sample-y[fold['train']]))),
            'mean_oof_minus_full_sec':float(np.mean(oof-in_sample)),
            'std_oof_minus_full_sec':float(np.std(oof-in_sample))}}

        for variant in variants:
            model=C1Model(anchor['model'],variant,a.seed).prepare(X[fold['train']])
            # Group OOF anchor supervises TRAIN only; original frozen RF handles VAL/TEST/ONLINE.
            hist,prelog,epoch=train_one(model,X,G,y,data.groups,fold['train'],fold['val'],a.seed,
                    epochs=a.epochs,lr=a.learning_rate,pretrain_epochs=a.pretrain_epochs,
                    decision_weight=a.decision_weight,train_oof_anchor=oof)
            test_score=score_groups(model,X,G,data.groups,fold['internal_test'])
            te_groups=select_rows(data,fold['internal_test'])
            policy=choose(y[fold['internal_test']],ora[fold['internal_test']],
                          test_score,test_score,te_groups,0.)
            metric=summarize(policy)
            tmetric=test_metrics(y[fold['internal_test']],test_score,None,te_groups,'reg')
            artifact={'model':model,'mode':'F4','method':'P2','protocol':protocol,
                'online_compatible':True,'feature_names':list(names),'threshold':0.,
                'c1_variant':variant,'graph_dim':GRAPH_DIM,'seed':a.seed,
                'training_label':PRIMARY,'source_sha256':mf['source_sha256'],
                'boundary_X_sha256':xhash,'graph_sha256':summary['graph_sha256'],
                'anchor_sha256':sha256(apath),'stage':'step6c1-delegation-v1',
                'train_anchor_oof':True}
            joblib.dump(artifact,folder/f'{variant}_model.joblib',compress=3)
            csv_write(folder/f'{variant}_test_states.csv',policy)
            report[variant]={'test_policy':metric,'test_metrics':tmetric,
                        'best_validation_epoch':epoch,'pretrain_losses':prelog,
                        'model_sha256':sha256(folder/f'{variant}_model.joblib')}
            (folder/f'{variant}_history.json').write_text(json.dumps(hist,indent=2)+'\n')
            print('C1',protocol,variant,'signed gain',round(metric['mean_signed_gain_sec'],3),'best_epoch',epoch,flush=True)
        # Original static anchor is untouched, reported as an essential baseline.
        base=model.anchor.predict(X[fold['internal_test'],:26])[0]
        original=choose(y[fold['internal_test']],ora[fold['internal_test']],base,base,
                        select_rows(data,fold['internal_test']),anchor['threshold'])
        report['M0']={'test_policy':summarize(original),'frozen_threshold':anchor['threshold']}
        (folder/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        all_reports[protocol]=report
    (out/'summary.json').write_text(json.dumps({'status':'COMPLETE','protocols':protos,'variants':variants},indent=2)+'\n')
    return all_reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',default=SOURCE)
    p.add_argument('--features-dir',default=FEATS)
    p.add_argument('--graph-dir',default=GRAPH)
    p.add_argument('--anchor-study',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--protocol',choices=('task','date','both'),default='both')
    p.add_argument('--variants',default='D0,D1,D2T,D2')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--pretrain-epochs',type=int,default=4)
    p.add_argument('--learning-rate',type=float,default=.0015)
    p.add_argument('--decision-weight',type=float,default=1.0)
    a=p.parse_args()
    if a.epochs<1 or a.pretrain_epochs<0 or a.learning_rate<=0:raise ValueError('Invalid training hyperparams')
    run(a)
if __name__=='__main__':main()
