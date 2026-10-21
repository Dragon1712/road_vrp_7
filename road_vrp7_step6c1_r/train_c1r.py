#!/usr/bin/env python3
"""C1-R: controlled C1 training anchor/loss study with cautious validation gating.

READ ONLY consumes original C1 graph, frozen C1/B2 data + P2-F2 anchor.
R0 OOF_ZERO = reproduce C1 loss with OOF training anchor.
R1 FULL_ZERO = isolate full frozen P2-F2 anchor in training.
R2 FULL_DEFER = M0-proxy defer action as fixed alternative in decision loss.
R3 FULL_DEFER_SAFE = same R2 network, *validation-only* conservative gate.
Actual ONLINE rejection is C3, not the offline M0 proxy.
"""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import joblib
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r','road_vrp7_step6c1','road_vrp7_step6b1_v2','road_vrp7_step6b2','road_vrp7_step6b3'):
    p=str(ROOT/folder)
    if p not in sys.path:sys.path.insert(0,p)
from train_c1 import by_state,pretrain_graph,score_groups,select_rows_dummy
from dataset import load,masks,select_rows,PRIMARY,ORACLE,sha256
from train_b2 import read_extra,get_features
from features import FEATURE_MODES
from train_b3 import read_anchor
from train import choose,summarize,test_metrics,csv_write
from graph_features import GRAPH_DIM,unpack
from c1_model import state_loss
from r_model import ReliableC1,m0_proxy_loss,evaluate_m0_proxy,selected_action

DEFAULT_SOURCE='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1'
DEFAULT_FEATURES='runs/chicago_july_v1/step6b2/features_boundary_v1'
DEFAULT_GRAPH='runs/chicago_july_v1/step6c1/c1_graph_full_v1'
ARMS=('OOF_ZERO','FULL_ZERO','FULL_DEFER','FULL_DEFER_SAFE')


def score_groups_with_gates(model,X,G,groups,mask):
    score=np.full(len(X),np.nan,dtype=np.float64)
    gate=np.full(len(X),np.nan,dtype=np.float64)
    for inds in by_state(groups,mask):
        xx=np.concatenate((X[inds],G[inds]),axis=1) if model.variant in ('D2','D2T') else X[inds]
        s,g,_=model.predict(xx)
        score[inds]=s;gate[inds]=g
    if not np.isfinite(score[mask]).all() or not np.isfinite(gate[mask]).all():
        raise ValueError('Incomplete state-level prediction/gating')
    return score[mask],gate[mask]


def train_one_r(model,X,G,Y,groups,train_mask,val_mask,seed,epochs,lr,pretrain_epochs,
                mode,full_anchor,train_oof=None,anchor_threshold=0.):
    if mode not in ('OOF_ZERO','FULL_ZERO','FULL_DEFER'):raise ValueError(mode)
    if model.variant in ('D2','D2T'):
        prelog=pretrain_graph(model.net,G,np.flatnonzero(train_mask),seed,pretrain_epochs)
    else:prelog=[]
    tr=by_state(groups,train_mask)
    opt=torch.optim.AdamW(model.net.parameters(),lr=lr,weight_decay=0.02)
    s=model.static_scaler.transform(X[:,:26]).astype(np.float32)
    t=model.traffic_scaler.transform(X[:,26:]).astype(np.float32)
    t[:,10]=np.clip(X[:,36],0,1);t[:,11]=np.clip(X[:,37],0,1)
    ss=torch.from_numpy(s);tt=torch.from_numpy(t)
    gg=tuple(torch.from_numpy(a.copy()) for a in unpack(G)) if model.variant in ('D2','D2T') else None
    if gg is not None and model.variant=='D2T':
        seq,meta,adj,valid=gg
        gg=(seq,meta,torch.eye(24).unsqueeze(0)*valid.unsqueeze(-1),valid)
    yy=torch.from_numpy(np.asarray(Y,np.float32))
    anchor_training=np.asarray(full_anchor,np.float32).copy()
    if mode=='OOF_ZERO':
        if train_oof is None or len(train_oof)!=int(train_mask.sum()):raise ValueError('OOF train anchor required')
        anchor_training[train_mask]=np.asarray(train_oof,np.float32)
    aa=torch.from_numpy(anchor_training)
    frozen_full=torch.from_numpy(np.asarray(full_anchor,np.float32))
    rng=np.random.default_rng(seed)
    best=(-float('inf'),0);best_state=None;best_ep=None;history=[]
    val_rows=select_rows_dummy(groups,val_mask)
    for ep in range(1,epochs+1):
        model.net.train();losses=[]
        order=rng.permutation(len(tr))
        for start in range(0,len(tr),8):
            indices=torch.as_tensor([tr[j] for j in order[start:start+8]],dtype=torch.long)
            arg=None if gg is None else tuple(a[indices] for a in gg)
            corrections=model.net(ss[indices],tt[indices],arg)
            scores=aa[indices]+corrections
            batch=[]
            for j in range(len(indices)):
                if mode=='FULL_DEFER':
                    loss=m0_proxy_loss(scores[j],yy[indices][j],frozen_full[indices][j],anchor_threshold)
                else:
                    loss=state_loss(scores[j],yy[indices][j])
                batch.append(loss)
            l=torch.stack(batch).mean()
            opt.zero_grad();l.backward();torch.nn.utils.clip_grad_norm_(model.net.parameters(),2.)
            opt.step();losses.append(float(l.detach()))
        model.net.eval()
        vscore=score_groups(model,X,G,groups,val_mask)
        # Validation selection is identical across training arms; no test access.
        vr=summarize(choose(Y[val_mask],Y[val_mask],vscore,vscore,val_rows,0.))
        value=float(vr['mean_signed_gain_sec'])
        history.append({'epoch':ep,'train_loss':float(np.mean(losses)),
                        'validation_zero_noop_gain_sec_per_state':value})
        key=(round(value,8),-ep)
        if key>best:best=key;best_ep=ep;best_state=copy.deepcopy(model.net.state_dict())
        if ep==1 or ep%5==0:print(model.variant,mode,'epoch',ep,'train_loss',round(np.mean(losses),4),'val_gain',round(value,3),flush=True)
    model.net.load_state_dict(best_state);model.net.eval()
    return history,prelog,best_ep


def fit_gate_validation(model,X,G,groups,y,anchor_threshold,mask):
    """Choose risk gate on validation only; cache neural scores ONCE per state.

    This avoids repeatedly running an expensive graph encoder for all 60 grid
    configurations. The labels are used only for validation gate selection.
    """
    cached=[]
    for inds in by_state(groups,mask):
        xx=X[inds]
        inp=np.concatenate((xx,G[inds]),axis=1) if model.variant in ('D2','D2T') else xx
        scores,anchor=model.score_components(inp)
        j=int(np.argmax(anchor))
        base_reward=float(y[inds[j]]) if anchor[j]>=anchor_threshold else 0.
        cover=None if model.variant=='D0' else np.minimum(np.clip(xx[:,36],0,1),np.clip(xx[:,37],0,1))
        cached.append((scores,anchor,np.asarray(y[inds],float),base_reward,cover))
    grid=[]
    covs=(0.,) if model.variant=='D0' else (0.,.05,.2)
    for cap in (20.,40.,80.,160.):
        for margin in (0.,5.,10.,20.,40.):
            for coverage in covs:
                rewards=[];baseline=[];decisions=[]
                for scores,anchor,true_reward,base,cover in cached:
                    chosen,ok,_=selected_action(scores,anchor,margin,cap,coverage,cover)
                    baseline.append(base)
                    decisions.append(int(ok))
                    rewards.append(float(true_reward[chosen]) if ok else base)
                delta=float(np.mean(np.asarray(rewards)-baseline))
                grid.append({'margin':margin,'max_correction':cap,'min_coverage':coverage,
                  'validation_m0_proxy':{'states':len(cached),
                    'mean_improvement_vs_m0_proxy_sec_per_state':delta,
                    'mean_signed_gain_sec_per_state':float(np.mean(rewards)),
                    'mean_m0_proxy_gain_sec_per_state':float(np.mean(baseline)),
                    'delegation_rate':float(np.mean(decisions))}})
    # Same-risk utility ties prefer fewer interventions, more margin, less correction.
    chosen=max(grid,key=lambda v:(round(v['validation_m0_proxy']['mean_improvement_vs_m0_proxy_sec_per_state'],8),
                 -v['validation_m0_proxy']['delegation_rate'],v['margin'],
                 -v['max_correction'],v['min_coverage']))
    model.safe_gate=True;model.margin=chosen['margin']
    model.max_correction=chosen['max_correction'];model.min_coverage=chosen['min_coverage']
    return chosen


def run(a):
    output=Path(a.output_dir)
    if output.exists():raise FileExistsError('Use a NEW independent output directory: '+str(output))
    arms=tuple(x.strip() for x in a.arms.split(','))
    variants=tuple(x.strip() for x in a.variants.split(','))
    if not arms or len(set(arms))!=len(arms) or any(x not in ARMS for x in arms):raise ValueError('Bad arms')
    if not variants or len(set(variants))!=len(variants) or any(x not in ('D0','D1','D2T','D2') for x in variants):raise ValueError('Bad variants')
    d=load(a.source_dir)
    extra,mf,xhash=read_extra(d,a.features_dir)
    X,names=get_features(d,extra,'F4')
    if tuple(names)!=tuple(FEATURE_MODES['F4']) or X.shape!=(d.n,44):raise ValueError('Unexpected F4 schema')
    if tuple(FEATURE_MODES['F2'])!=tuple(names[:26]):raise ValueError('Frozen F2!=F4 prefix')
    graphroot=Path(a.graph_dir)
    summary=json.loads((graphroot/'summary.json').read_text())
    graphmanifest=json.loads((graphroot/'manifest.json').read_text())
    if summary['status']!='COMPLETE' or summary['graph_sha256']!=sha256(graphroot/'graph_X.npz'):
        raise ValueError('Unverified original C1 graph')
    if graphmanifest['b2_boundary_sha256']!=xhash or graphmanifest['source_sha256']!=mf['source_sha256']:
        raise ValueError('Graph/source hash mismatch')
    with np.load(graphroot/'graph_X.npz',allow_pickle=False) as z:G=np.asarray(z['X'],dtype=np.float32)
    if G.shape!=(d.n,GRAPH_DIM) or not np.isfinite(G).all():raise ValueError('Invalid graph')
    y=d.label(PRIMARY);ora=d.label(ORACLE)
    protos=('task','date') if a.protocol=='both' else (a.protocol,)
    output.mkdir(parents=True,exist_ok=False)
    provenance={'stage':'step6c1r-anchored-selective-defer-v1','source':a.source_dir,
        'features':a.features_dir,'graph_study':a.graph_dir,'anchor_study':a.anchor_study,
        'seed':a.seed,'arms':arms,'variants':variants,'protocols':protos,'epochs':a.epochs,
        'pretrain_epochs':a.pretrain_epochs,'train_label':PRIMARY,'source_sha256':mf['source_sha256'],
        'graph_sha256':summary['graph_sha256'],'F4_sha256':xhash,
        'code_sha256':{p:sha256(Path(__file__).with_name(p)) for p in ('r_model.py','train_c1r.py')},
        'research_sources':{
         'cost_sensitive_defer':'literature/code/learn-to-defer-master',
         'baseline_bootstrap':'literature/code/SPIBB-master/spibb.py',
         'routing_subproblem_delegate':'literature/code/learning-to-delegate-main/supervised.py',
         'masked_traffic_pretrain':'literature/code/spin-main/spin/models/spin.py'},
        'critical_notes':['July previously used for model development; not blind.',
           'FULL train uses frozen RF in-sample anchor; helps alignment but may overfit.',
           'FULL_DEFER offline default is frozen M0 within sampled K; actual online fallback is C3.',
           'Gate is tuned on validation ONLY and does not offer SPIBB theoretical safety.',
           'No SPCLT integrated; graph representation is unchanged from original C1.']}
    (output/'manifest.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    summary_by_proto={}
    for protocol in protos:
        fold=masks(d,protocol)
        anchor,apath=read_anchor(a.anchor_study,protocol,mf['source_sha256'],xhash,a.seed)
        from model import group_oof_anchor
        oof=group_oof_anchor(X[fold['train'],:26],y[fold['train']],select_rows(d,fold['train']),protocol,a.seed,120)
        frozen_full=anchor['model'].predict(X[:,:26])[0]
        directory=output/protocol;directory.mkdir()
        report={'_anchor_diagnostics': {
            'mean_oof_minus_full_sec':float(np.mean(oof-frozen_full[fold['train']])),
            'std_oof_minus_full_sec':float(np.std(oof-frozen_full[fold['train']])),
            'train_OOF_MAE':float(np.mean(np.abs(y[fold['train']]-oof))),
            'train_fullRF_MAE':float(np.mean(np.abs(y[fold['train']]-frozen_full[fold['train']])))} }
        tr_groups=select_rows(d,fold['internal_test']);val_groups=select_rows(d,fold['val'])
        for variant in variants:
            cache={}
            for arm in arms:
                base_arm='FULL_DEFER' if arm=='FULL_DEFER_SAFE' else arm
                if base_arm not in cache:
                    model=ReliableC1(anchor['model'],variant,a.seed).prepare(X[fold['train']])
                    model.train_anchor_mode='oof' if base_arm=='OOF_ZERO' else 'full'
                    model.loss_mode='m0_proxy_defer' if base_arm=='FULL_DEFER' else 'zero_noop'
                    hist,prelog,epoch=train_one_r(model,X,G,y,d.groups,fold['train'],fold['val'],
                       a.seed,a.epochs,a.learning_rate,a.pretrain_epochs,base_arm,
                       frozen_full,oof,anchor['threshold'])
                    cache[base_arm]=(model,hist,prelog,epoch)
                trained,hist,prelog,epoch=cache[base_arm]
                model=copy.deepcopy(trained) if arm=='FULL_DEFER_SAFE' else trained
                gate_cfg=None
                if arm=='FULL_DEFER_SAFE':
                    gate_cfg=fit_gate_validation(model,X,G,d.groups,y,anchor['threshold'],fold['val'])
                # Keep original zero-noop metric separately; proxy fallback is a different offline diagnostic.
                test_score,test_gate=score_groups_with_gates(model,X,G,d.groups,fold['internal_test'])
                policy=choose(y[fold['internal_test']],ora[fold['internal_test']],
                   test_score,test_gate,tr_groups,0.)
                mkey=f'{variant}_{arm}'
                from r_model import evaluate_m0_proxy
                m0proxy=evaluate_m0_proxy(model,X[fold['internal_test']],G[fold['internal_test']],
                                         tr_groups,y[fold['internal_test']],anchor['threshold'])
                file_mode='F2' if variant=='D0' else 'F4'
                artifact={'model':model,'mode':file_mode,'method':'P2','protocol':protocol,
                    'online_compatible':True,'feature_names':list(FEATURE_MODES[file_mode]),
                    'threshold':0.,'stage':'step6c1r-anchored-selective-defer-v1',
                    'c1_variant':variant,'r_arm':arm,'graph_dim':GRAPH_DIM,'seed':a.seed,
                    'training_label':PRIMARY,'source_sha256':mf['source_sha256'],
                    'boundary_X_sha256':xhash,'graph_sha256':summary['graph_sha256'],
                    'anchor_sha256':sha256(apath),'m0_anchor_threshold':anchor['threshold'],
                    'gate_validation_config':gate_cfg}
                joblib.dump(artifact,directory/f'{mkey}_model.joblib',compress=3)
                csv_write(directory/f'{mkey}_test_states_zero_noop.csv',policy)
                report[mkey]={'test_zero_noop':summarize(policy),'test_M0_proxy_fallback':m0proxy,
                  'best_validation_epoch':epoch,'gate_validation_only':gate_cfg,
                  'pretrain_losses':prelog,'model_sha256':sha256(directory/f'{mkey}_model.joblib'),
                  'caution':'Zero-noop and M0-proxy are separate retrospective offline conventions; real online fallback=C3.'}
                (directory/f'{mkey}_history.json').write_text(json.dumps(hist,indent=2)+'\n')
                print(protocol,mkey,'test_zero',round(report[mkey]['test_zero_noop']['mean_signed_gain_sec'],3),
                    'M0proxy_delta',round(m0proxy['mean_improvement_vs_m0_proxy_sec_per_state'],3),flush=True)
        base=choose(y[fold['internal_test']],ora[fold['internal_test']],
           frozen_full[fold['internal_test']],frozen_full[fold['internal_test']],tr_groups,anchor['threshold'])
        report['M0']={'test_policy':summarize(base),'frozen_threshold':anchor['threshold']}
        (directory/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        summary_by_proto[protocol]={'arms':len(report)-2,'status':'COMPLETE'}
    (output/'summary.json').write_text(json.dumps({'status':'COMPLETE','protocols':summary_by_proto},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',default=DEFAULT_SOURCE)
    p.add_argument('--features-dir',default=DEFAULT_FEATURES);p.add_argument('--graph-dir',default=DEFAULT_GRAPH)
    p.add_argument('--anchor-study',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--protocol',choices=('task','date','both'),default='both')
    p.add_argument('--variants',default='D0,D1,D2T,D2');p.add_argument('--arms',default=','.join(ARMS))
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--pretrain-epochs',type=int,default=4);p.add_argument('--learning-rate',type=float,default=.0015)
    a=p.parse_args()
    if a.epochs<1 or a.pretrain_epochs<0 or a.learning_rate<=0:p.error('Bad budget')
    run(a)
if __name__=='__main__':main()
