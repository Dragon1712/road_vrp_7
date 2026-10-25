"""R4-Q: STRICT historical lag-24h traffic at decision time, never same-day hindsight.

Uses original Stage2 raw (per-slot observation, static fallback), not FIFO Road
replay or contemporaneous observations. Historical slot must be strictly earlier
than decision departure by a full 24 hours.  No Validation/Test tasks admitted.
"""
from __future__ import annotations
import csv, hashlib, json, math, heapq
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

STAGE = 'R4Q-TRAIN-PAST-24H-ROAD-FUSION-v1'
MODES = ('route_context', 'static_concat', 'lag24_concat', 'lag24_conditioned')
LAG_HOURS = 24
STATIC_N = 12
EXTRA_N = 13   # lagged 12 OD context features + observed edge support fraction

def need(flag, reason):
    if not flag: raise ValueError(reason)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()

def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, data):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)

def parse_departure(v):
    need(isinstance(v,str) and len(v)>=16 and not any(x in v for x in ('Z','+')), 'Departure must be local naive datetime')
    t=datetime.fromisoformat(v)
    need(t.tzinfo is None,'Timezone-aware departure not allowed')
    return t

def historical_slot(slot_ns, departure, *, lag_hours=LAG_HOURS):
    """Fetch EXACT prior wall-clock slot. Fail closed if missing or future."""
    need(lag_hours==24,'R4Q protocol frozen to lag 24h')
    d=parse_departure(departure)
    target=d-timedelta(hours=lag_hours)
    target_ns=int(np.datetime64(target,'ns').astype(np.int64))
    ns=np.asarray(slot_ns,dtype=np.int64)
    need(ns.ndim==1 and len(ns)>1 and np.all(np.diff(ns)>0),'Bad/nonmonotonic time index')
    pos=int(np.searchsorted(ns,target_ns))
    need(pos<len(ns) and int(ns[pos])==target_ns,'No exact 24h-past observation slot')
    need(target<d,'Traffic source is not past')
    return pos, target.isoformat(sep=' ')

def resolve_historical_source(slot_ns, departure):
    """Strict lag24 if present; STATIC ONLY if the required slot predates Stage2.

    Never substitute another nearby date. Missing *internal* or later slots are
    data-integrity errors. The requested source timestamp remains recorded even
    when no historical observation is available.
    """
    d=parse_departure(departure)
    requested=d-timedelta(hours=LAG_HOURS)
    target_ns=int(np.datetime64(requested,'ns').astype(np.int64))
    ns=np.asarray(slot_ns,dtype=np.int64)
    need(ns.ndim==1 and len(ns)>1 and np.all(np.diff(ns)>0),
         'Bad/nonmonotonic time index')
    need(requested<d,'Historical time must precede departure')
    if target_ns<int(ns[0]):
        return {'available':False,'slot_index':None,
                'requested_historical_slot':requested.isoformat(sep=' '),
                'historical_slot':None,'fallback_reason':'BEFORE_DATA_START'}
    # If the target lies inside the coverage but is missing, or lies after the
    # index, do NOT substitute a different observation or silently fall back.
    pos,source_date=historical_slot(ns,departure)
    return {'available':True,'slot_index':pos,
            'requested_historical_slot':requested.isoformat(sep=' '),
            'historical_slot':source_date,'fallback_reason':None}


def road_arrays(csv_path, npz_path):
    """Read Stage2 raw slot costs/observations; never dynamic FIFO.

    Memory note: Stage2 compressed NPZ loading creates raw and mask arrays once.
    """
    edges=[]; static=[]
    with Path(csv_path).open(newline='',encoding='utf-8') as f:
        rdr=csv.DictReader(f)
        need({'edge_idx','osm_u','osm_v','static_travel_sec'}<=set(rdr.fieldnames or []),'Missing edge fields')
        for i,row in enumerate(rdr):
            need(int(row['edge_idx'])==i,'Edge order differs from npz edge axis')
            w=float(row['static_travel_sec']);need(math.isfinite(w) and w>0,'Invalid static weight')
            edges.append((str(row['osm_u']),str(row['osm_v'])));static.append(w)
    need(edges,'Empty edge index')
    with np.load(npz_path,allow_pickle=False) as z:
        need({'travel_sec_raw','chicago_observed_mask','slot_local_naive_ns','is_mapped_edge'}<=set(z.files), 'Missing raw/mask/time/mapped in Stage2')
        raw=z['travel_sec_raw']
        obs=z['chicago_observed_mask']
        times=z['slot_local_naive_ns']
        mapped=z['is_mapped_edge']
    need(raw.ndim==2 and raw.shape==obs.shape and raw.shape[1]==len(edges),'Raw/edge/mask shape mismatch')
    need(len(times)==raw.shape[0] and obs.dtype==np.bool_ and mapped.shape==(len(edges),) and mapped.dtype==np.bool_ and bool(mapped.any()), 'Observed mask/time/mapping shape invalid')
    need(not bool(obs[:,~mapped].any()),'Observations on unmapped road edges')
    need(np.isfinite(raw).all() and (raw>0).all(),'Raw traffic invalid')
    return edges,np.asarray(static,dtype=float),raw,obs,times,mapped

def graph_from_weights(edges, weights):
    need(len(edges)==len(weights),'Edge/weight length mismatch')
    out=defaultdict(list)
    for (u,v),w in zip(edges,weights):
        f=float(w);need(math.isfinite(f) and f>0,'Invalid graph weight')
        out[u].append((v,f))
    return out

def task_od(task, graph):
    import r4p_core as p
    return p.task_od(task,graph)

def build_context(tasks, cost_dir, *, scenarios=('2025-07-02_0800','2025-07-09_1700')):
    """Two exact Train dates. Return static OD plus lag24 OD per (task, scenario)."""
    cost_dir=Path(cost_dir)
    edges,static,raw,obs,slots,mapped=road_arrays(cost_dir/'edge_index.csv',cost_dir/'edge_costs.npz')
    need(set(tasks)=={f'task_{i:04d}' for i in range(4)},'Only original four Train tasks')
    static_graph=graph_from_weights(edges,static)
    statics={k:task_od(v,static_graph) for k,v in sorted(tasks.items())}
    lagged={};meta={}
    for scenario in scenarios:
        choices=[]
        for task in tasks.values():
            choices.extend([z for z in task['scenarios'] if z['scenario_id']==scenario])
        need(len(choices)==len(tasks) and all(z['split']=='train' and z['eligible'] is True for z in choices),'NonTrain or missing scenario')
        departures={z['departure_local'] for z in choices}
        need(len(departures)==1,'Scenario clocks differ between tasks')
        departure=next(iter(departures))
        source=resolve_historical_source(slots,departure)
        if source['available']:
            pos=source['slot_index']
            mask=obs[pos]
            w=np.where(mask,raw[pos],static).astype(np.float64)
        else:
            # The *entire* 24h history is unavailable (e.g. July 1 before
            # Stage2 July 2 start); do not use July 2's contemporaneous data.
            mask=np.zeros(len(edges),dtype=np.bool_)
            w=static.copy()
        graph=graph_from_weights(edges,w)
        for k,task in sorted(tasks.items()):lagged[k,scenario]=task_od(task,graph)
        meta[scenario]={**source,'departure':departure,
                        'observed_edges':int(mask.sum()),
                        'total_edges':len(edges),'mapped_edges':int(mapped.sum()),
                        'observed_fraction':float(np.mean(mask)),
                        'observed_mapped_fraction':float(np.mean(mask[mapped])),
                        'lag_hours':24,
                        'traffic_feature_semantics':('ACTUAL_PREVIOUS_DAY_RAW' if source['available'] else 'STATIC_FALLBACK_NO_HISTORY')}
        if not source['available']:
            for task in tasks:
                need(np.array_equal(lagged[task,scenario]['dist'],statics[task]['dist']),
                     'Static fallback OD differs from static baseline')
    return statics,lagged,meta

def fused_features(base, static_features, lag_features, observed_fraction, mode):
    """All features are predecision; no label/future input anywhere."""
    need(mode in MODES,'Unknown mode')
    b=list(map(float,base));a=list(map(float,static_features));h=list(map(float,lag_features))
    c=float(observed_fraction)
    need(len(b)==38 and len(a)==12 and len(h)==12 and 0<=c<=1,'Feature shapes/support invalid')
    need(all(math.isfinite(v) for v in b+a+h+[c]),'Nonfinite features')
    if mode=='route_context':return b
    if mode=='static_concat':return b+a
    if mode=='lag24_concat':return b+a+h+[c]
    # FiLM-inspired fixed cross terms only; NOT official trainable FiLM.
    ix=(8,9,10,11,14,17,20,23,25,26,32,36)
    return b+a+h+[c]+[b[j]*(h[k]-a[k]) for k,j in enumerate(ix)]

def attach(recs, statics, lagged, meta, pcore):
    out=[]
    for r in recs:
        k=tuple(r['snap']);task,scenario=k[:2]
        need((task,scenario) in lagged and task in statics and scenario in meta,'Unrecognized Train snapshot')
        s=r['snapshot'];a=r['action']
        need((s['task_id'],s['scenario_id'],int(s['seed']),int(s['stop_count']))==k,'Snapshot/action key mismatch')
        base=r['feature_sets']['route_context']
        sf=pcore.static_features(s,a,statics[task])
        lf=pcore.static_features(s,a,lagged[task,scenario])
        modes={m:fused_features(base,sf,lf,meta[scenario]['observed_mapped_fraction'],m) for m in MODES}
        out.append({**r,'feature_sets':modes})
    need(len(out)==1572,'Expected 1572 original Train candidate actions')
    return out

def evaluate(records,splits, ocore):
    """Same paired-ridge and gate as R4-O/P, with only feature mode changed."""
    from r4i_core import ridge_predictors
    prev=ocore.FEATURE_MODES
    ocore.FEATURE_MODES=MODES
    folds=[];details=[]
    try:
        for idx,(name,tr,held,typ) in enumerate(splits):
            need(not set(tuple(x['snap']) for x in tr)&set(tuple(x['snap']) for x in held),'Snapshot leakage')
            if typ=='TASK_LOTO':need(not ({x['task'] for x in tr}&{x['task'] for x in held}),'Task leakage')
            else:need(not ({x['snap'][2] for x in tr}&{x['snap'][2] for x in held}),'Seed leakage')
            for mode in MODES:
                for shuffle in (False,True):
                    training=ocore.shuffle_labels(tr,20261025+idx*127) if shuffle else tr
                    rank=ocore.fit_pairwise(training,mode)
                    gate=ocore.fit_gate(training)
                    ridge=ridge_predictors(training,'k500',strength=100.,shuffle_labels=False)
                    for policy in ('C3','pairwise_ridge_gate','learned_gate_pairwise','oracle_gate_pairwise','sampled_oracle'):
                        rows=ocore.predict_decisions(held,rank,gate,ridge,mode,policy)
                        fold={'split':name,'split_type':typ,'feature_mode':mode,
                              'shuffle_train_labels':shuffle,'policy':policy,
                              'oracle_conditioned':policy in ocore.ORACLE_POLICIES,
                              'rank_pairs':rank['pairs'],'rank_fitted':rank['fitted'],
                              'gate_fitted':gate['fitted'],**ocore.metrics(rows)}
                        folds.append(fold)
                        details.extend({'evaluation':{'split':name,'mode':mode,'shuffle':shuffle,'policy':policy},**z} for z in rows)
    finally:ocore.FEATURE_MODES=prev
    need(len(folds)==6*4*2*5,'Unexpected fold count')
    return folds,details

def stratify_by_history(details, meta):
    """Only observational policy outcomes, separated by source availability.

    Uses labels solely for posthoc scoring; the source flag is a fixed property
    of the scenario and is not added to the model inputs. Never count Oracle
    decisions as learned-model successes.
    """
    grouped=defaultdict(list)
    for z in details:
        e=z['evaluation']
        if e['shuffle'] or e['policy']!='learned_gate_pairwise':continue
        scenario=z['snapshot_id'][1]
        need(scenario in meta,'Unknown scenario in model decision detail')
        k=(e['split'],e['mode'],scenario)
        grouped[k].append(z)
    result=[]
    for (split,mode,scenario),rr in sorted(grouped.items()):
        gains=[float(z['gain_k500_sec']) for z in rr]
        need(all(math.isfinite(x) for x in gains),'Nonfinite stratified rewards')
        source=meta[scenario]
        result.append({'split':split,'feature_mode':mode,'scenario_id':scenario,
            'history_available':bool(source['available']),
            'n_snapshots':len(rr),
            'n_delegations':sum(z['selected_arm']!='C3' for z in rr),
            'positive':sum(g>1e-8 for g in gains),
            'negative':sum(g<-1e-8 for g in gains),
            'mean_gain_sec':sum(gains)/len(gains),
            'total_gain_sec':sum(gains)})
    return result


def report_md(report):
    lines=['# R4-Q Train-only 24h滞后交通模态消融','',
      '**仅使用出发前24小时已观测的Stage2原始历史旅行时间；不是实时交通预测或神经FiLM。**','',
      '| 留组 | 特征 | learned_gate+pairwise K500收益(秒/快照) | 委托 | 正/负 |',
      '|---|---|---:|---:|---|']
    for z in report['folds']:
        if not z['shuffle_train_labels'] and z['policy']=='learned_gate_pairwise':
            lines.append(f"| {z['split']} | {z['feature_mode']} | {z['mean_gain_sec']:+.3f} | {z['delegations']} | {z['positive']}/{z['negative']} |")
    lines.extend(['','## 历史观测支持度','',
      '| Train场景 | 出发时刻 | 请求的24h时槽 | 真实时槽可用 | 实际使用时槽 | 全路网观测比例 | 已映射边观测比例 |',
      '|---|---|---|---|---|---:|---:|'])
    for sc,meta in sorted(report['historical_source'].items()):
        lines.append(f"| {sc} | {meta['departure']} | {meta['requested_historical_slot']} | {meta['available']} | {meta['historical_slot'] or '无（静态回退）'} | {meta['observed_fraction']:.3%} | {meta['observed_mapped_fraction']:.3%} |")
    lines.extend(['','## 按实际历史可用性分层的模型结果（非Shuffle learned_gate_pairwise）','',
       '| 留组 | 特征 | 场景 | 历史可用 | 快照 | 委托 | 正/负 | 平均K500收益(s) |',
       '|---|---|---|---|---:|---:|---|---:|'])
    for z in report['history_stratified']:
        lines.append(f"| {z['split']} | {z['feature_mode']} | {z['scenario_id']} | {z['history_available']} | {z['n_snapshots']} | {z['n_delegations']} | {z['positive']}/{z['negative']} | {z['mean_gain_sec']:+.3f} |")
    lines.extend(['','**关键限制：** 7月2日缺少7月1日24h历史，完整使用静态回退；只有7月9日真实使用24h历史。因此整体200快照的差异不得全部归因于动态交通。模型仍是R4-O加权Pairwise Ridge与状态门控；K500离线Road收益不是30秒在线改善。',''])
    return '\n'.join(lines)
