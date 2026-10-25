"""R4-L: OLD+NEW TRAIN-only counterfactual decision analysis; no online ALNS claims.

Reuses unchanged r4i_core for predecision features / deterministic weighted ridge,
and r4k_core for official R4-C unit filenames. No intervention is executed.
"""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
from collections import Counter, defaultdict
ROOT=Path(__file__).resolve().parents[1]
for name in ('road_vrp7_step6c1_r4i','road_vrp7_step6c1_r4k'):
    q=str(ROOT/name)
    if q not in sys.path:sys.path.insert(0,q)
import r4i_core as old
import r4k_core as audit

STAGE='R4L-TRAIN-CROSS-SEED-CONSERVATIVE-UTILITY-v1'
H=(50,200,500)
EPS=1e-8
THRESHOLDS=(0.,5.,15.)

def need(x,msg):
    if not x:raise ValueError(msg)

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def rows(p):
    with Path(p).open(encoding='utf8') as f:return [json.loads(s) for s in f if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):h.update(buf)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');t.replace(p)
def dump_rows(p,rr):
    with Path(p).open('w',encoding='utf8') as f:
        for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def snapshot_id(s):return (str(s['task_id']),str(s['scenario_id']),int(s['seed']),int(s['stop_count']))
def unit_spec(key,h,arm):return dict(zip(('task_id','scenario_id','seed','stop_count'),key),horizon=h,arm=arm)
def unit_file(cfdir,key,h,arm):return Path(cfdir)/'units'/audit.official_unit_filename(unit_spec(key,h,arm))

def read_old(hdir,cdir,pdir):
    """Read pinned original R4-I evidence; require full R4-H and 298-step proof."""
    hr,hm,labs,branches,plans,snaps,pins=old.source_audit(hdir,cdir,pdir)
    selected,smap,labelmap,features=old.validate_dataset(hr,hm,labs,branches,plans,snaps)
    rec=old.records_from_dataset(selected,smap,labelmap,features)
    need(len(rec)==565 and len(selected)==72 and set(r['snap'][2] for r in rec)=={60,61},
         'Expected frozen old R4-H 72/565 sample (60/61)')
    for r in rec:r['source']='original_s60s61'
    return rec,{'source_sha256':pins,'n_snapshots':len(selected),'n_candidate_actions':len(rec),
                'exact_physical_states':len({old.physics(smap[k]) for k in selected})}

def read_new(kdir,cdir,cfdir,pdir):
    """Reconstruct predecision features from C3/alternative first action at K50.

    R4-K already audited full K event/control traces. For speed, do not reread
    3789 full event traces; independently pin source hashes and first action
    identity of every alternative K50 unit. No action label is a feature.
    """
    kdir,cdir,cfdir,pdir=map(Path,(kdir,cdir,cfdir,pdir))
    rep=read(kdir/'report.json');prov=read(kdir/'provenance.json')
    need(rep.get('stage')==audit.STAGE and rep.get('status')=='SOURCE_COMPLETE' and
         rep.get('data_role')=='TRAIN_ONLY','R4-K full audit not SOURCE_COMPLETE TRAIN')
    need(prov.get('stage')==audit.STAGE and prov['report_sha256']==sha(kdir/'report.json'),
         'R4-K report/provenance SHA mismatch')
    need(prov['source_sha256']==rep['source_sha256'],'R4-K provenance pins disagree')
    a=rep['analysis']
    need(a['planned_scenario_snapshots']==a['completed_scenario_snapshots']==128 and
         a['incomplete_scenario_snapshots']==0 and a['existing_unit_files']==3789 and
         a['complete_coarse_search_starts']==64,'Not original fully audited R4-J Train-early study')
    source_paths={
       'prefix_report':pdir/'report.json','prefix_manifest':pdir/'manifest.json',
       'prefix_paired_rows':pdir/'paired_rows.jsonl','prefix_excluded_windows':pdir/'excluded_windows.json',
       'capture_manifest':cdir/'manifest.json','capture_provenance':cdir/'provenance.json',
       'capture_snapshots':cdir/'snapshots.jsonl','capture_metrics':cdir/'metrics.jsonl',
       'cf_manifest':cfdir/'manifest.json','cf_report':cfdir/'report.json'
    }
    for field,path in source_paths.items():
        need(field in rep['source_sha256'] and path.is_file() and
             sha(path)==rep['source_sha256'][field],f'R4-K upstream SHA mismatch or missing: {field} {path}')
    cap=read(cdir/'manifest.json');pref=read(pdir/'report.json');cf=read(cfdir/'manifest.json');cfr=read(cfdir/'report.json')
    need(cap['arm']=='capture' and cap['split']=='train' and cap['section_only'] is True and
         set(cap['task_ids'])=={f'task_{i:04d}' for i in range(4)} and set(map(int,cap['seeds']))=={62,63},
         'New capture is not independent Train new-seed grid')
    need(pref['status']=='COMPLETE' and pref['exact_matches']==pref['n_comparisons']==512 and
         pref['original_capture_snapshots_sha256']==sha(cdir/'snapshots.jsonl'),
         'New prefix proof missing/inconsistent')
    need(cf['data_role']=='TRAIN_ONLY' and cf['horizons']==[50,200,500] and
         cf['source_sha256']['capture_snapshots']==sha(cdir/'snapshots.jsonl') and
         cfr['status']=='COMPLETE','New Train CF stage incomplete')
    snaps=rows(cdir/'snapshots.jsonl');smap={snapshot_id(s):s for s in snaps}
    need(len(smap)==len(snaps)==128,'New source snapshots malformed/duplicated')
    allow={tuple(x) for x in cf['selected_snapshot_ids']}
    need(set(smap)==allow,'New CF/full snapshot selection mismatch')
    lbl=rows(kdir/'audited_labels.jsonl')
    need(len(lbl)==sum(a['per_horizon'][str(k)]['action_labels'] for k in H)==3021,
         'New R4-K audited labels count mismatch')
    by=defaultdict(dict)
    counts=Counter()
    for z in lbl:
        k=tuple(z['snapshot_id']);h=int(z['horizon']);arm=str(z['arm']);v=float(z['road_best_gain_sec'])
        need(k in smap and h in H and arm not in ('C3','SHAM_C3') and math.isfinite(v),
             'Invalid new R4-K action label')
        need((h,arm) not in by[k],'Duplicate new label action/horizon')
        need((str(z['task_id']),str(z['scenario_id']),int(z['seed']),int(z['stop_count']))==k,
             'R4-K label task identity mismatch')
        by[k][h,arm]=v;counts[h]+=1
    for h in H:
        z=a['per_horizon'][str(h)]
        ys=[v for arms in by.values() for (hh,arm),v in arms.items() if hh==h]
        need(len(ys)==z['action_labels'] and
             sum(v>EPS for v in ys)==z['positive'] and
             sum(v<-EPS for v in ys)==z['negative'] and
             sum(abs(v)<=EPS for v in ys)==z['zero'],
             'New R4-K histogram differs from per-action rewards')
    records=[]
    for k in sorted(by):
        s=smap[k];z=by[k]
        arms={arm for h,arm in z}
        need(all(set(h for h,an in z if an==arm)==set(H) for arm in arms),
             'Missing new horizon for an action')
        for arm in sorted(arms):
            name=unit_file(cfdir,k,50,arm)
            need(name.is_file(),f'No audited K50 unit: {name}')
            unit=read(name);sp=unit['unit_spec'];res=unit['result'];plan=unit['plan']
            need(sp==unit_spec(k,50,arm) and res['arm']==arm and res['horizon']==50 and
                 unit['source_manifest_sha256']==rep['source_sha256']['prefix_manifest'] and
                 unit['source_snapshots_sha256']==rep['source_sha256']['capture_snapshots'],
                 'K50 unit identity/provenance mismatch')
            choice=next((v for v in plan['alternatives'] if v['name']==arm),None)
            need(choice is not None and res['first_action_id']==[choice['route'],choice['pos'],choice['size']]
                 and res['first_original_c3_action_id']==plan['c3_identity'],
                 'Action geometry/C3 changed')
            targets={f'k{h}':z[h,arm] for h in H}
            targets['robust']=min(targets['k200'],targets['k500'])
            records.append({'snap':k,'arm':arm,'task':k[0],
                'physical':old.physics(s), 'features':old.predecision_features(s,plan,res['first_action_id']),
                'targets':targets,'source':'new_s62s63'})
    need(len(records)==1007 and len(by)==128,'New R4-K complete action features mismatch')
    return records,{'source_sha256':{k:sha(v) for k,v in source_paths.items()},
                    'r4k_report_sha256':sha(kdir/'report.json'),
                    'r4k_audited_labels_sha256':sha(kdir/'audited_labels.jsonl'),
                    'n_snapshots':len(by),'n_candidate_actions':len(records),
                    'exact_physical_states':len({old.physics(s) for s in smap.values()})}

def group_records(recs):
    d=defaultdict(list)
    for r in recs:
        d[r['snap']].append(r)
    need(all(1<=len(v)<=8 for v in d.values()),'Bad candidate count')
    return d

def make_splits(records):
    tasks=sorted({r['task'] for r in records})
    need(tasks==[f'task_{i:04d}' for i in range(4)],'Unexpected Train task coverage')
    seeds=sorted({r['snap'][2] for r in records})
    need(seeds==[60,61,62,63],'Source seed groups unexpected')
    splits=[]
    for t in tasks:
        splits.append(('hold_task_'+t,[r for r in records if r['task']!=t],
                       [r for r in records if r['task']==t],'TASK_LOTO'))
    for src,target,label in [({60,61},{62,63},'old_to_new'),({62,63},{60,61},'new_to_old')]:
        splits.append((label,[r for r in records if r['snap'][2] in src],
                           [r for r in records if r['snap'][2] in target],'SEED_PAIR_TRANSFER'))
    for name,tr,te,protocol in splits:
        need(tr and te,'Empty evaluation split')
        need(not (set(r['snap'] for r in tr)&set(r['snap'] for r in te)),
             'TRAIN/HELDOUT overlap in decision states')
        if protocol=='TASK_LOTO':need(not ({r['task'] for r in tr}&{r['task'] for r in te}),
                                      'TASK group leakage')
        else:need(not ({r['snap'][2] for r in tr}&{r['snap'][2] for r in te}),
                  'SEED group leakage')
    return splits

def with_mask(recs,which):
    # For geometry-only, strip search progress and state quality. Keep action
    # geometry/C3 relationship and the Section operator's known repair slot.
    allowed={0,6,7,8,9,10,11,12,13,14} if which=='geometry' else set(range(len(old.FEATURES)))
    return [{**r,'features':[v if j in allowed else 0. for j,v in enumerate(r['features'])],
             'physical':r['physical'][:-1]} for r in recs]

def choices_for_model(held,weight,threshold,gate):
    by=group_records(held);choices=[]
    for k,alts in sorted(by.items()):
        scores=[(old.predict(weight,r['features']),r['arm'],r) for r in alts]
        best=max(scores,key=lambda x:(x[0],x[1]));score=best[0]
        choose=best[2] if score>threshold and (gate!='early2000' or k[3]<=2000) else None
        gain=float(choose['targets']['k500']) if choose else 0.
        oro=max([0.]+[float(r['targets']['k500']) for r in alts])
        choices.append({'snapshot_id':list(k),'selected_arm':choose['arm'] if choose else 'C3',
           'predicted_margin_sec':float(score),'actual_gain_k500_sec':gain,'oracle_gain_k500_sec':oro,
           'random_expected_k500_sec':sum(float(r['targets']['k500']) for r in alts)/(len(alts)+1),
           'n_alternatives':len(alts),'early':k[3]<=2000})
    return choices

def metrics(picks):
    gains=[r['actual_gain_k500_sec'] for r in picks];positive_oracle=[r for r in picks if r['oracle_gain_k500_sec']>EPS]
    delegated=[r for r in picks if r['selected_arm']!='C3']
    return {'n_snapshots':len(picks),'n_delegations':len(delegated),
       'n_positive':sum(x>EPS for x in gains),'n_negative':sum(x<-EPS for x in gains),
       'n_zero':sum(abs(x)<=EPS for x in gains),'mean_gain_k500_sec':sum(gains)/len(gains),
       'total_gain_k500_sec':sum(gains),
       'worst_harm_sec':min(gains),'sum_harm_sec':sum(min(0.,x) for x in gains),
       'oracle_positive_snapshots':len(positive_oracle),
       'oracle_mean_gain_k500_sec':sum(r['oracle_gain_k500_sec'] for r in picks)/len(picks),
       'random_expected_mean_k500_sec':sum(r['random_expected_k500_sec'] for r in picks)/len(picks),
       'positive_oracle_recall':sum(r['actual_gain_k500_sec']>EPS for r in positive_oracle)/len(positive_oracle) if positive_oracle else None}

def evaluate(records):
    folds=[];details=[]
    for split_i,(split,tr,te,protocol) in enumerate(make_splits(records)):
        for ablation in ('full','geometry'):
            training=with_mask(tr,ablation);held=with_mask(te,ablation)
            for target in ('k500','robust'):
                for shuffled in (False,True):
                    weights=old.ridge_predictors(training,target=target,strength=100.0,
                                                 shuffle_labels=shuffled,seed=20261025+split_i)
                    for gate in ('none','early2000'):
                        for thr in THRESHOLDS:
                            picks=choices_for_model(held,weights,thr,gate)
                            result={'split':split,'split_type':protocol,'feature_set':ablation,
                               'prediction_target':target,'shuffle_train_labels':shuffled,
                               'gate':gate,'threshold_sec':thr,
                               'train_seeds':sorted({r['snap'][2] for r in tr}),
                               'heldout_seeds':sorted({r['snap'][2] for r in te}),
                               'train_tasks':sorted({r['task'] for r in tr}),
                               'heldout_tasks':sorted({r['task'] for r in te}),
                               **metrics(picks)}
                            folds.append(result)
                            for p in picks:details.append({'evaluation':{'split':split,'feature_set':ablation,
                                      'target':target,'shuffled':shuffled,'gate':gate,'threshold':thr},**p})
    return folds,details

def make_summary(old_rec,new_rec,folds):
    ss={}
    for tag,rr in [('old',old_rec),('new',new_rec),('combined',old_rec+new_rec)]:
        by=group_records(rr)
        per={}
        for h in H:
            vals=[float(r['targets'][f'k{h}']) for r in rr]
            per[str(h)]={'labels':len(vals),'positive':sum(v>EPS for v in vals),
                'negative':sum(v<-EPS for v in vals),'zero':sum(abs(v)<=EPS for v in vals),
                'positive_rate':sum(v>EPS for v in vals)/len(vals)}
        ss[tag]={'n_candidate_actions':len(rr),'n_scenario_snapshots':len(by),
            'n_exact_physical_states':len(set(tuple(r['physical']) for r in rr)),
            'n_coarse_physical_starts':len(set(tuple(r['physical'][:-1]) for r in rr)),
            'per_horizon':per}
    return {'data_signal':ss,'n_evaluation_configs':len(folds),
        'fold_types':dict(Counter(r['split_type'] for r in folds)),
        'limits':['TRAIN exploratory only: four tasks, four seeds, two scenarios per seed',
                  'R4-K audited fixed-K counterfactual best Road gain is not exact 30s online gain',
                  'No score/threshold tuned on independent Validation or Test',
                  'Oracle is a posthoc upper bound on sampled alternatives only',
                  'Task and seed split observations still have correlated cross-scenario searches',
                  'No claim of out-of-distribution generalization from four tasks']}

def markdown(report,folds):
    s=report['summary']['data_signal']
    out=['# R4-L Train-only跨任务、跨种子保守委托诊断','',
      f"数据：原R4-H {s['old']['n_candidate_actions']}动作 + 新R4-K {s['new']['n_candidate_actions']}动作；全部 {s['combined']['n_scenario_snapshots']}场景状态。",'',
      '| 来源 | K500正/负/零 | 场景状态 | 粗粒度搜索起点 |',
      '|---|---:|---:|---:|']
    for name in ('old','new','combined'):
        z=s[name];v=z['per_horizon']['500']
        out.append(f"| {name} | {v['positive']}/{v['negative']}/{v['zero']} | {z['n_scenario_snapshots']} | {z['n_coarse_physical_starts']} |")
    out.extend(['','## 主对照：全特征、非打乱、K500目标、0秒门槛','',
        '| 留组 | 委托 | 正/负 | 平均K500收益(s) | Oracle均值(s) |',
        '|---|---:|---:|---:|---:|'])
    for f in folds:
        if (f['feature_set']=='full' and f['prediction_target']=='k500' and not f['shuffle_train_labels']
             and f['gate']=='early2000' and f['threshold_sec']==0.):
            out.append(f"| {f['split']} | {f['n_delegations']} | {f['n_positive']}/{f['n_negative']} | {f['mean_gain_k500_sec']:+.3f} | {f['oracle_mean_gain_k500_sec']:+.3f} |")
    out+=['','### 审计和限制','',
      '- 拟合仅使用Train；任务留一和种子留组仍属于Train内部探索。',
      '- 所有预测只用Section决策前状态、Section几何特征；未来反事实Road收益仅作为标签。',
      '- 始终C3的收益为0；SHAM由R4-H/R4-K上游严格控制；Oracle不可部署。',
      '- 任务留一仅4折；不能据此生成稳健显著性结论或挑选最终门槛。',
      '- 不包含推理开销或真实30秒在线预算；进入Validation必须另行冻结并执行实测。','']
    return '\n'.join(out)
