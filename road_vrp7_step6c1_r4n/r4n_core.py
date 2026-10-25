"""R4-N: Section action RANK vs state GATE decomposition, Train-only.

The sole deployable-baseline proxy here is Ridge/C3 gate + pairwise rank. Other
oracle-conditioned policies are explicitly NON-DEPLOYABLE diagnostics. No ALNS.
"""
from __future__ import annotations
from collections import defaultdict,Counter
from pathlib import Path
import hashlib,json,math,random,sys

ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r4m','road_vrp7_step6c1_r4l','road_vrp7_step6c1_r4i'):
    q=str(ROOT/folder)
    if q not in sys.path:sys.path.insert(0,q)
import r4m_core as r4m
import r4l_core as r4l
import r4i_core as r4i

STAGE='R4N-TRAIN-SECTION-RANK-VS-GATE-v1'
EPS=1e-8
POLICIES=('C3','ridge_ref','pairwise_forced_early', 'pairwise_with_ridge_gate',
          'oracle_gate_pairwise_rank','ridge_gate_oracle_rank','sampled_oracle_early')
ORACLE_POLICIES=('oracle_gate_pairwise_rank','ridge_gate_oracle_rank','sampled_oracle_early')


def need(ok,msg):
    if not ok:raise ValueError(msg)

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8');t.replace(p)

def group(recs):
    groups=defaultdict(list)
    for r in recs:groups[tuple(r['snap'])].append(r)
    need(all(1<=len(a)<=8 for a in groups.values()),'Bad candidate count')
    return dict(sorted(groups.items()))

def shuffle_train_targets(records,seed):
    """Break feature-target relation only inside each TRAIN task; retain task prevalence.

    This is a negative control, not a candidate bootstrap and never touches heldout.
    """
    rng=random.Random(seed);bytask=defaultdict(list)
    for i,r in enumerate(records):bytask[r['task']].append(i)
    out=[dict(r,targets=dict(r['targets'])) for r in records]
    for task in sorted(bytask):
        ix=bytask[task];vals=[records[i]['targets']['k500'] for i in ix]
        rng.shuffle(vals)
        for i,y in zip(ix,vals):out[i]['targets']['k500']=float(y)
    return out

def fit_rank(train,*,regularization=1.0):
    """Pairwise ridge on within-snapshot feature differences and K500 gains.

    Ignore equal-label pairs, never import sample rewards into feature vectors.
    Weight coarse physical search starts equally inside each task. The intercept
    cancels in a within-state difference, so solve only non-bias coefficients.
    """
    need(train,'Empty training partition')
    grouped=group(train);n=len(r4i.FEATURES)-1
    tasks=sorted({r['task'] for r in train});need(tasks,'Empty tasks')
    task_coarse=defaultdict(set);coarse_shots=Counter()
    for sid in grouped:
        c=(sid[0],sid[2],sid[3])
        task_coarse[sid[0]].add(c);coarse_shots[c]+=1
    mat=[[0.]*n for _ in range(n)];vec=[0.]*n
    pair_count=0;informative_states=0
    for sid,acts in grouped.items():
        if sid[3]>2000:continue
        diffs=[]
        for i in range(len(acts)):
            for j in range(i+1,len(acts)):
                yi=float(acts[i]['targets']['k500']);yj=float(acts[j]['targets']['k500'])
                if abs(yi-yj)<=EPS:continue
                diff=[float(x)-float(y) for x,y in zip(acts[i]['features'][1:],acts[j]['features'][1:])]
                if max(abs(z) for z in diff)<1e-12:continue
                diffs.append((diff,max(-250.,min(250.,yi-yj))))
        if not diffs:continue
        informative_states+=1
        coarse=(sid[0],sid[2],sid[3])
        w=1./(len(tasks)*len(task_coarse[sid[0]])*coarse_shots[coarse]*len(diffs))
        for d,delta in diffs:
            pair_count+=1
            for i in range(n):
                vec[i]+=w*d[i]*delta
                for j in range(n):mat[i][j]+=w*d[i]*d[j]
    if pair_count==0:
        return {'weights':[0.]*n,'pair_count':0,'informative_snapshots':0,'fitted':False}
    for i in range(n):mat[i][i]+=regularization
    beta=r4i._linear_system(mat,vec)
    need(all(math.isfinite(v) for v in beta),'Nonfinite ranking weights')
    return {'weights':beta,'pair_count':pair_count,'informative_snapshots':informative_states,'fitted':True}

def rank_score(model,rec):
    return sum(w*float(x) for w,x in zip(model['weights'],rec['features'][1:]))

def choose(held,rank_model,ridge,policy):
    need(policy in POLICIES,'Unknown policy')
    out=[]
    for sid,arms in group(held).items():
        early=sid[3]<=2000
        pairwise=max(arms,key=lambda r:(rank_score(rank_model,r),r['arm']))
        ref=max(arms,key=lambda r:(r4i.predict(ridge,r['features']),r['arm']))
        ridge_gate=bool(early and r4i.predict(ridge,ref['features'])>0.)
        best_oracle=max(arms,key=lambda r:(r['targets']['k500'],r['arm']))
        pos_oracle=bool(early and best_oracle['targets']['k500']>EPS)
        if policy=='C3':chosen=None
        elif policy=='ridge_ref':chosen=ref if ridge_gate else None
        elif policy=='pairwise_forced_early':chosen=pairwise if early and rank_model['fitted'] else None
        elif policy=='pairwise_with_ridge_gate':chosen=pairwise if ridge_gate and rank_model['fitted'] else None
        elif policy=='oracle_gate_pairwise_rank':chosen=pairwise if pos_oracle and rank_model['fitted'] else None
        elif policy=='ridge_gate_oracle_rank':chosen=best_oracle if ridge_gate and best_oracle['targets']['k500']>EPS else None
        elif policy=='sampled_oracle_early':chosen=best_oracle if pos_oracle else None
        else:raise AssertionError(policy)
        gain=float(chosen['targets']['k500']) if chosen else 0.
        need(math.isfinite(gain),'Nonfinite label')
        out.append({'snapshot_id':list(sid),'selected_arm':chosen['arm'] if chosen else 'C3',
                    'gain_k500_sec':gain,'positive_oracle_state':pos_oracle,
                    'oracle_best_gain_sec':max(0.,float(best_oracle['targets']['k500'])) if early else 0.,
                    'rank_top_positive':(pairwise['targets']['k500']>EPS) if early and rank_model['fitted'] else None,
                    'ridge_gate':ridge_gate,'early':early,
                    'oracle_uses_future_labels':policy in ORACLE_POLICIES,
                    'ranking_model_fitted':rank_model['fitted']})
    return out

def score_summary(rows):
    n=len(rows);need(n>0,'Empty heldout set')
    gains=[z['gain_k500_sec'] for z in rows]
    oracle_n=sum(z['positive_oracle_state'] for z in rows)
    oracle_hit=sum(z['positive_oracle_state'] and z['gain_k500_sec']>EPS for z in rows)
    rank_n=sum(z['early'] and z['positive_oracle_state'] and z['ranking_model_fitted'] for z in rows)
    rank_hit=sum(z['early'] and z['positive_oracle_state'] and z['rank_top_positive'] is True for z in rows)
    return {'n_snapshots':n,'n_early':sum(z['early'] for z in rows),
            'delegations':sum(z['selected_arm']!='C3' for z in rows),
            'positive':sum(v>EPS for v in gains),'negative':sum(v<-EPS for v in gains),
            'zero':sum(abs(v)<=EPS for v in gains),
            'total_gain_sec':sum(gains),'mean_gain_sec':sum(gains)/n,
            'sum_harm_sec':sum(min(0.,v) for v in gains),
            'oracle_positive_states':oracle_n,'positive_oracle_recall':oracle_hit/oracle_n if oracle_n else None,
            'rank_top1_hits_on_opportunities':rank_hit,
            'rank_top1_opportunities':rank_n,
            'rank_top1_recall_on_opportunities':rank_hit/rank_n if rank_n else None,
            'sampled_oracle_mean_sec':sum(z['oracle_best_gain_sec'] for z in rows)/n}

def evaluate(records):
    folds=[];details=[]
    for split_idx,(name,tr,te,kind) in enumerate(r4l.make_splits(records)):
        need(not ({r['snap'] for r in tr}&{r['snap'] for r in te}),'Snapshot leakage')
        if kind=='TASK_LOTO':need(not ({r['task'] for r in tr}&{r['task'] for r in te}),'Task leakage')
        else:need(not ({r['snap'][2] for r in tr}&{r['snap'][2] for r in te}),'Seed leakage')
        for feat in ('full','geometry'):
            training=r4l.with_mask(tr,feat)
            held=r4l.with_mask(te,feat)
            for null in (False,True):
                seed=20261026+split_idx*37+int(null)
                data=shuffle_train_targets(training,seed) if null else training
                ridge=r4i.ridge_predictors(data,target='k500',strength=100.)
                rank=fit_rank(data)
                for policy in POLICIES:
                    rows=choose(held,rank,ridge,policy)
                    # Oracle policies are diagnostic only, never deployment candidates.
                    f={'split':name,'split_type':kind,'feature_set':feat,
                       'shuffled_training_labels':null,'policy':policy,
                       'oracle_conditioned':policy in ORACLE_POLICIES,
                       'training_tasks':sorted({r['task'] for r in tr}),
                       'heldout_tasks':sorted({r['task'] for r in te}),
                       'training_seeds':sorted({r['snap'][2] for r in tr}),
                       'heldout_seeds':sorted({r['snap'][2] for r in te}),
                       'rank_pairs':rank['pair_count'],'rank_informative_snapshots':rank['informative_snapshots'],
                       **score_summary(rows)}
                    folds.append(f)
                    details.extend({'evaluation':{'split':name,'feature_set':feat,'shuffle':null,'policy':policy},**r} for r in rows)
    return folds,details

def markdown(result):
    folds=result['folds'];ss=result['data_summary']
    out=['# R4-N Train-only Section动作排序 vs 状态委托门控解耦诊断','',
      f"数据：{ss['actions']}动作，{ss['snapshots']}场景快照，{ss['coarse_starts']}个粗粒度搜索起点；Train-only，无新的ALNS调用。",'',
      '## 主比较：全特征、非Shuffle；外层仅评分','',
      '| 外层留组 | 策略 | 委托 | 正/负 | K500均值(s) | 正例状态Top-1命中/机会 |',
      '|---|---|---:|---:|---:|---:|']
    for x in folds:
        if x['feature_set']!='full' or x['shuffled_training_labels']:continue
        if x['policy'] in ('C3','ridge_ref','pairwise_with_ridge_gate','oracle_gate_pairwise_rank','sampled_oracle_early'):
            out.append(f"| {x['split']} | {x['policy']}{' (*)' if x['oracle_conditioned'] else ''} | {x['delegations']} | {x['positive']}/{x['negative']} | {x['mean_gain_sec']:+.3f} | {x['rank_top1_hits_on_opportunities']}/{x['rank_top1_opportunities']} |")
    out+=['','(*) **Oracle gate / oracle policy使用留出数据的未来收益标签，绝不可上线；仅用于分解错误来源。**','',
      '## 分析约束','',
      '- Ridge门控与Pairwise排序的组合也仅是Train内部固定K离线评分，尚未包含在线推理成本。',
      '- Pairwise强制委托是有风险的诊断策略，不能当作推荐部署方案。',
      '- 所有外层划分依原R4-L任务留一及种子组隔离；同一状态候选不跨组。',
      '- Shuffle仅在外层训练任务内打乱标签；特征不含未来Road收益。',
      '- 样本仍来自4个Train任务；不能将此结果当作独立Validation/Test或30秒在线增益。','']
    return '\n'.join(out)
