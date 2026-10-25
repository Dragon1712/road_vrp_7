"""R4-O: TRAIN-only structural Section ranking + independent state gate.

No online ALNS, no ex-post OD inputs, no Validation/Test. All feature values
are from original R4-G preselection snapshots and candidate geometry.
"""
from __future__ import annotations
from collections import defaultdict,Counter
import hashlib, json, math, random
import numpy as np

STAGE='R4O-TRAIN-STRUCTURAL-RANK-GATE-v1'
EPS=1e-8
FEATURE_MODES=('baseline15','route_context','route_context_interact')
POLICIES=('C3','ridge_ref','pairwise_ridge_gate','learned_gate_pairwise',
          'oracle_gate_pairwise','learned_gate_oracle_rank','sampled_oracle')
ORACLE_POLICIES=('oracle_gate_pairwise','learned_gate_oracle_rank','sampled_oracle')


def need(flag,msg):
    if not flag:raise ValueError(msg)


def sid(x):
    if isinstance(x,(list,tuple)):
        need(len(x)==4,'Wrong snapshot key length')
        return str(x[0]),str(x[1]),int(x[2]),int(x[3])
    return str(x['task_id']),str(x['scenario_id']),int(x['seed']),int(x['stop_count'])


def groups(records):
    out=defaultdict(list)
    for r in records:out[tuple(r['snap'])].append(r)
    need(all(1<=len(v)<=8 for v in out.values()),'Invalid candidate count')
    return dict(sorted(out.items()))


def _route_edges(r):
    return {(a,b) for a,b in zip([None]+list(r),list(r)+[None])}


def route_context(s, action):
    """Route topology only. No future candidate/label or customer ID magnitude.

    Exact customer IDs are used solely for equality/membership relations.
    They are NEVER used as numeric features and no Road.replay cost is read.
    """
    routes=s['current_routes'];best=s['best_routes']
    need(isinstance(routes,list) and isinstance(best,list) and len(routes)==len(best) and len(routes)>0,'Bad route snapshot')
    route,pos,size=(int(v) for v in action)
    need(0<=route<len(routes) and size in (2,3) and 0<=pos<=len(routes[route])-size,'Illegal Section geometry')
    rr=routes[route];brr=best[route]
    all_nodes=[c for r in routes for c in r];all_best=[c for r in best for c in r]
    need(len(all_nodes)==len(set(all_nodes)) and set(all_nodes)==set(all_best),'Best/current customer mismatch')
    n=len(rr); n_all=max(1,len(all_nodes)); seg=rr[pos:pos+size]
    best_location={v:(i,j) for i,r in enumerate(best) for j,v in enumerate(r)}
    best_edges=set().union(*(_route_edges(r) for r in best))
    seg_arcs={(rr[j],rr[j+1]) for j in range(pos,pos+size-1)}
    left=rr[pos-1] if pos else None
    right=rr[pos+size] if pos+size<n else None
    section_edges={(left,seg[0]),(seg[-1],right)}
    lengths=[len(r) for r in routes];length_best=[len(r) for r in best]
    within=sum(best_location[z][0]==route for z in seg)/size
    best_routes_in_segment=len({best_location[z][0] for z in seg})
    best_segment_positions=[best_location[z][1] for z in seg]
    order=sum(best_segment_positions[j+1]==best_segment_positions[j]+1 and best_location[seg[j]][0]==best_location[seg[j+1]][0] for j in range(size-1))/max(1,size-1)
    best_local_edge=sum(e in best_edges for e in section_edges)/2.0
    other=[len(r) for i,r in enumerate(routes) if i!=route]
    avg_other=sum(other)/len(other) if other else n
    mean_len=sum(lengths)/len(lengths)
    ss=[
      size/n, n/n_all, pos/max(1,n-1), (n-pos-size)/max(1,n-1),
      float(pos==0),float(pos+size==n),
      float(pos<=1),float(pos+size>=n-1),
      n/max(1,max(lengths)), min(2.,n/max(1,mean_len)),
      sum(v>n for v in other)/max(1,len(other)), min(2.,n/max(1,avg_other)),
      abs(n-len(brr))/max(1,n),
      min(1.,abs(len(rr)-len(brr))/max(1,n_all)),
      within, len({best_location[z][0] for z in seg})/size,
      order, sum(e in best_edges for e in seg_arcs)/max(1,len(seg_arcs)),
      best_local_edge,
      sum(best_location[z][0]!=route for z in seg)/size,
      float(best_routes_in_segment>1),
      float((left,seg[0]) in best_edges),float((seg[-1],right) in best_edges),
    ]
    need(len(ss)==23 and all(math.isfinite(v) for v in ss),'Invalid contextual features')
    return [float(x) for x in ss]


def augment(base,snap,act,mode):
    need(mode in FEATURE_MODES,'Unknown feature mode')
    v=[float(x) for x in base]
    need(len(v)==15 and all(math.isfinite(x) for x in v),'Bad original R4-I feature vector')
    if mode=='baseline15':return v
    new=route_context(snap,act)
    f=v+new
    if mode=='route_context_interact':
        # Deterministic, fixed prior interactions: no heldout fitting of feature maps.
        ix=[(0,14),(2,16),(3,18),(4,15),(8,17),(10,20),(11,18),(14,22),
            (16,17),(18,19),(5,13),(20,21)]
        f+=[new[a]*new[b] for a,b in ix]
    return f


def state_gate_features(s):
    """Only global preselection state, intentionally action-independent."""
    routes=s['current_routes'];lengths=[len(x) for x in routes];best=s['best_routes']
    n=sum(lengths);need(n>0 and lengths and len(best)==len(routes),'Bad state routes')
    cur=float(s['current_surrogate_cost']);best_cost=float(s['best_surrogate_cost'])
    temp=float(s['anneal_temperature_before_accept'])
    start=float(s['anneal_start_temperature'])
    remain=float(s['wall_remaining_before_select_sec'])
    vec=[1.,float(s['stop_count'])/20000.,min(1.,max(0.,remain)/30.),
         max(0.,cur-best_cost)/max(1.,abs(best_cost)),
         max(0.,temp)/max(1.,abs(start)),
         len(routes)/10., max(lengths)/n,min(lengths)/n,
         sum((len(x)-n/len(routes))**2 for x in routes)/max(1.,n*n),
         sum(len(a)!=len(b) for a,b in zip(routes,best))/len(routes)]
    need(all(math.isfinite(x) for x in vec),'Invalid state gate features')
    return vec


def attach_context(records,old_snaps,new_snaps,old_actions,new_actions):
    """Strict predecision source join, catching any missing action identity."""
    oldmap={sid(s):s for s in old_snaps};newmap={sid(s):s for s in new_snaps}
    need(len(oldmap)==len(old_snaps) and len(newmap)==len(new_snaps),'Duplicate source snapshots')
    overlap=set(oldmap)&set(newmap);need(not overlap,'Old/new source snapshot overlap')
    amap={**old_actions,**new_actions};out=[]
    for r in records:
        k=tuple(r['snap']);a=str(r['arm']); snap=(oldmap if r['source']=='original_s60s61' else newmap).get(k)
        need(snap is not None,'Missing original predecision snapshot')
        need((k,a) in amap,'Missing candidate action identity')
        act=tuple(int(x) for x in amap[k,a]);need(len(act)==3,'Bad action tuple')
        extra={mode:augment(r['features'],snap,act,mode) for mode in FEATURE_MODES}
        out.append({**r,'feature_sets':extra,'gate_features':state_gate_features(snap)})
    need(len(out)==len(records),'Context row count differs')
    for k,rr in groups(out).items():
        need(all(z['gate_features']==rr[0]['gate_features'] for z in rr),'Gate features vary by action')
    return out


def shuffle_labels(train,seed):
    """Break association at WHOLE STATE granularity, preserving action vectors.

    For each task, permute complete action reward vectors only between snapshots
    having the same number of candidate actions, retaining within-state rank signal.
    This is a stringent negative control against learned state->action matching.
    """
    rng=random.Random(seed);gs=groups(train);out={}
    by=defaultdict(list)
    for k,rows in gs.items():by[k[0],len(rows)].append(k)
    for _,keys in sorted(by.items()):
        keys=sorted(keys);donors=keys[:];rng.shuffle(donors)
        for receiver,donor in zip(keys,donors):
            values=[dict(r['targets']) for r in gs[donor]]
            rng.shuffle(values)
            rec=sorted(gs[receiver],key=lambda r:r['arm'])
            for r,t in zip(rec,values):out[(receiver,r['arm'])]=t
    return [{**r,'targets':dict(out[(tuple(r['snap']),r['arm'])])} for r in train]


def fit_pairwise(train,mode,lam=1.0):
    need(train and mode in FEATURE_MODES,'Bad ranking training sample')
    gs=groups(train);n=len(train[0]['feature_sets'][mode])-1
    X=[];Y=[];W=[];task_starts=defaultdict(set);coarse_count=Counter()
    for k in gs:
        coarse=(k[0],k[2],k[3]);task_starts[k[0]].add(coarse);coarse_count[coarse]+=1
    for k,recs in gs.items():
        if k[3]>2000:continue
        diffs=[]
        for i in range(len(recs)):
            for j in range(i+1,len(recs)):
                yi=float(recs[i]['targets']['k500']);yj=float(recs[j]['targets']['k500'])
                if abs(yi-yj)<EPS:continue
                d=np.asarray(recs[i]['feature_sets'][mode][1:],dtype=float)-np.asarray(recs[j]['feature_sets'][mode][1:],dtype=float)
                if np.max(np.abs(d))<1e-12:continue
                diffs.append((d,float(np.clip(yi-yj,-250,250))))
        if not diffs:continue
        coarse=(k[0],k[2],k[3]);w=1.0/(len(task_starts)*len(task_starts[k[0]])*coarse_count[coarse]*len(diffs))
        for d,y in diffs:X.append(d);Y.append(y);W.append(w)
    if not X:return {'fitted':False,'weights':[0.]*n,'pairs':0,'states':0}
    X=np.vstack(X);y=np.asarray(Y);w=np.asarray(W)
    cov=X.T@(X*w[:,None])+lam*np.eye(n)
    b=X.T@(w*y)
    coef=np.linalg.solve(cov,b)
    need(np.isfinite(coef).all(),'Rank weights nonfinite')
    return {'fitted':True,'weights':coef.tolist(),'pairs':len(Y),'states':len({tuple(r['snap']) for r in train if r['snap'][3]<=2000 and any(abs(z['targets']['k500'])>EPS for z in gs[tuple(r['snap'])])})}


def rank_score(m,r,mode):
    return float(np.dot(m['weights'],r['feature_sets'][mode][1:]))


def _sigmoid(x):
    x=min(50.,max(-50.,x))
    return 1/(1+math.exp(-x))


def fit_gate(train,l2=.8,steps=450,lr=.4):
    """Weighted sigmoid loss with task/coarse-start balancing, train partitions only."""
    gs=groups(train);data=[];count=Counter();task_starts=defaultdict(set)
    for k in gs:
        if k[3]<=2000:task_starts[k[0]].add((k[0],k[2],k[3]))
    for k,recs in gs.items():
        if k[3]>2000:continue
        co=(k[0],k[2],k[3]);count[co]+=1
        y=int(any(z['targets']['k500']>EPS for z in recs))
        data.append((k,np.asarray(recs[0]['gate_features'],float),y))
    need(data,'No early gate state')
    labels=[y for _,_,y in data]
    if not (any(labels) and not all(labels)):
        return {'fitted':False,'reason':'one_class','weights':[0.]*len(data[0][1]),'training_states':len(data),'positive_states':sum(labels)}
    weights=[]
    for k,f,y in data:
        co=(k[0],k[2],k[3]);weights.append(1/max(1,len(task_starts[k[0]]))/count[co])
    w=np.asarray(weights);w=w/w.sum()
    X=np.vstack([f for k,f,y in data]);Y=np.asarray(labels,dtype=float)
    # Balance classes for the ranking/recognition diagnostic. This does NOT
    # calibrate deployment risk; probability 0.5 is an exploratory threshold.
    frac=float(w@Y);w=w*np.where(Y>0,.5/frac,.5/(1-frac));w/=w.sum()
    beta=np.zeros(X.shape[1]);beta[0]=math.log(max(1e-4,frac)/max(1e-4,1-frac))
    penalty=np.ones(len(beta));penalty[0]=0.
    for _ in range(steps):
        z=np.clip(X@beta,-40,40);p=1/(1+np.exp(-z))
        grad=X.T@(w*(p-Y))+l2*beta*penalty
        beta-=lr*grad
    need(np.isfinite(beta).all(),'Nonfinite gate coefficients')
    return {'fitted':True,'reason':'ok','weights':beta.tolist(),'training_states':len(data),'positive_states':sum(labels)}


def gate_prob(model,rec):
    return _sigmoid(float(np.dot(model['weights'],rec['gate_features']))) if model['fitted'] else 0.


def predict_decisions(held,rank_model,gate_model,ridge_model,mode,policy):
    need(policy in POLICIES,'Unknown policy')
    out=[]
    for k,arms in groups(held).items():
        early=k[3]<=2000
        rank=max(arms,key=lambda r:(rank_score(rank_model,r,mode),r['arm']))
        ridge=max(arms,key=lambda r:(float(np.dot(ridge_model,r['features'])),r['arm']))
        ridge_open=early and float(np.dot(ridge_model,ridge['features']))>0
        prob=gate_prob(gate_model,arms[0])
        gate_open=bool(early and gate_model['fitted'] and prob>=.5)
        oracle_best=max(arms,key=lambda r:(r['targets']['k500'],r['arm']))
        oracle_open=early and oracle_best['targets']['k500']>EPS
        if policy=='C3':picked=None
        elif policy=='ridge_ref':picked=ridge if ridge_open else None
        elif policy=='pairwise_ridge_gate':picked=rank if ridge_open and rank_model['fitted'] else None
        elif policy=='learned_gate_pairwise':picked=rank if gate_open and rank_model['fitted'] else None
        elif policy=='oracle_gate_pairwise':picked=rank if oracle_open and rank_model['fitted'] else None
        elif policy=='learned_gate_oracle_rank':picked=oracle_best if gate_open and oracle_best['targets']['k500']>EPS else None
        elif policy=='sampled_oracle':picked=oracle_best if oracle_open else None
        else:raise AssertionError(policy)
        gain=float(picked['targets']['k500']) if picked is not None else 0.
        out.append({'snapshot_id':list(k),'selected_arm':picked['arm'] if picked else 'C3',
             'gain_k500_sec':gain,'oracle_positive':bool(oracle_open),
             'oracle_gain_sec':max(0.,float(oracle_best['targets']['k500'])) if early else 0.,
             'top1_positive':bool(rank['targets']['k500']>EPS) if early and rank_model['fitted'] else None,
             'top1_gain_sec':float(rank['targets']['k500']) if early and rank_model['fitted'] else None,
             'random_hit_probability':sum(z['targets']['k500']>EPS for z in arms)/len(arms) if early else 0.,
             'random_expected_sec':sum(z['targets']['k500'] for z in arms)/len(arms) if early else 0.,
             'gate_probability':prob,'gate_open':gate_open,'ridge_gate_open':ridge_open,
             'oracle_conditioned':policy in ORACLE_POLICIES,'rank_model_fitted':rank_model['fitted']})
    return out


def metrics(rows):
    need(rows,'No heldout states')
    g=[v['gain_k500_sec'] for v in rows];pos=[r for r in rows if r['oracle_positive']]
    return {'n_snapshots':len(rows),'delegations':sum(r['selected_arm']!='C3' for r in rows),
       'positive':sum(x>EPS for x in g),'negative':sum(x<-EPS for x in g),
       'mean_gain_sec':sum(g)/len(g),'total_gain_sec':sum(g),
       'sum_harm_sec':sum(min(0.,x) for x in g),
       'positive_oracle_states':len(pos),
       'top1_positive_hits':sum(r['top1_positive'] is True for r in pos),
       'random_expected_hits':sum(r['random_hit_probability'] for r in pos),
       'random_expected_gain_sec':sum(r['random_expected_sec'] for r in pos)/len(pos) if pos else None,
       'oracle_mean_gain_sec':sum(r['oracle_gain_sec'] for r in rows)/len(rows),
       'gate_open':sum(r['gate_open'] for r in rows)}


def evaluate(records,original_splits):
    out=[];details=[]
    for split_i,(name,tr,te,typ) in enumerate(original_splits):
        need(not ({tuple(x['snap']) for x in tr}&{tuple(x['snap']) for x in te}),'Snapshot split leakage')
        if typ=='TASK_LOTO':need(not ({x['task'] for x in tr}&{x['task'] for x in te}),'Task split leakage')
        else:need(not ({x['snap'][2] for x in tr}&{x['snap'][2] for x in te}),'Seed split leakage')
        for mode in FEATURE_MODES:
            for shuffled in (False,True):
                train=shuffle_labels(tr,20261025+split_i*127) if shuffled else tr
                rank=fit_pairwise(train,mode)
                gate=fit_gate(train)
                # Keep reference Ridge unchanged; shuffle only *training* labels
                # for negative control, never outer heldout labels.
                from r4i_core import ridge_predictors
                ref=ridge_predictors(train,'k500',strength=100.,shuffle_labels=False)
                for policy in POLICIES:
                    rows=predict_decisions(te,rank,gate,ref,mode,policy)
                    out.append({'split':name,'split_type':typ,'feature_mode':mode,
                         'shuffle_train_labels':shuffled,'policy':policy,
                         'oracle_conditioned':policy in ORACLE_POLICIES,
                         'rank_pairs':rank['pairs'],'rank_fitted':rank['fitted'],
                         'gate_fitted':gate['fitted'],'gate_train_states':gate['training_states'],
                         **metrics(rows)})
                    for item in rows:details.append({'evaluation':{'split':name,'mode':mode,'shuffle':shuffled,'policy':policy},**item})
    return out,details
