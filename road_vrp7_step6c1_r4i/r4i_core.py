"""R4-I TRAIN-only utility learnability: frozen labels, leakage-free features.

This file intentionally contains no ALNS, road oracle, or Validation labels.
Cross-task tests here are EXPLORATORY TRAIN-only, not an independent test.
"""
from __future__ import annotations
from collections import defaultdict,Counter
import hashlib,json,math,random
from pathlib import Path

STAGE='R4I-TRAIN-SIGNAL-AND-CONSERVATIVE-UTILITY-PROBE-v1'
EPS=1e-8
H=(50,200,500)
FEATURES=('bias','iter_frac','iter_early','wall_remain_frac','gap_ratio','anneal_frac',
          'route_len_frac','n_routes_frac','route_index_frac','position_frac','section_size3',
          'same_route_c3','distance_from_c3','candidate_count_frac','repair_index1')

def need(ok,msg):
    if not ok:raise ValueError(msg)

def sha(path):
    obj=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):obj.update(b)
    return obj.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def readl(path):
    with Path(path).open('r',encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(path)

def key(x):
    if isinstance(x,dict):return str(x['task_id']),str(x['scenario_id']),int(x['seed']),int(x['stop_count'])
    need(isinstance(x,(tuple,list)) and len(x)==4,'Bad snapshot ID')
    return str(x[0]),str(x[1]),int(x[2]),int(x[3])

def armkey(x):
    z=x['unit_spec'];return key(z)+(int(z['horizon']),str(z['arm']))

def _capture_weight_hash(weights):
    """Match R4-A canonical_hash for JSON-decoded RouletteWheel arrays.

    R4-H hashes the original weights with r4a_core.canonical_hash; the
    saved R4-G snapshots are JSON-native lists, so this is byte-for-byte
    equivalent to canonical_hash(jsonable(weights)). Do NOT cast the list
    values to float before hashing, since that changes the JSON bytes.
    """
    need(isinstance(weights, list),'R4-G capture weights must be JSON lists')
    need(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
             for v in weights),'Non-finite or invalid capture weights')
    return hashlib.sha256(json.dumps(weights,sort_keys=True,separators=(',',':'),
            allow_nan=False).encode()).hexdigest()


def physics(s):
    """Same exact search-state identity as R4-H report_metrics/horizon_stability.

    The R4-H report uses full-precision anneal temperature and SHA256
    canonical RouletteWheel weight arrays. R4-I v1 erroneously rounded the
    temperature to 9 decimal places, collapsing 51 source states into 36.
    """
    temp=float(s['anneal_temperature_before_accept'])
    need(math.isfinite(temp),'Non-finite anneal temperature')
    return (str(s['task_id']),int(s['seed']),int(s['stop_count']),
            str(s['current_route_sha256']),str(s['best_route_sha256']),str(s['rng_state_sha256']),
            _capture_weight_hash(s['roulette_destroy_weights']),
            _capture_weight_hash(s['roulette_repair_weights']),temp)

def source_audit(hdir,cdir,pdir):
    hdir,cdir,pdir=map(Path,(hdir,cdir,pdir))
    hr,hm=read(hdir/'report.json'),read(hdir/'manifest.json')
    pr=read(pdir/'report.json');cm=read(cdir/'manifest.json')
    need(hr.get('stage')=='R4H-TRAIN-SECTION-COUNTERFACTUAL-v1' and hr.get('status')=='COMPLETE' and hr.get('data_role')=='TRAIN_ONLY','Incomplete/nontrain R4H')
    need(hm.get('stage')==hr['stage'] and hm.get('data_role')=='TRAIN_ONLY' and hm.get('status') in ('CONFIGURED','COMPLETE'),'Nontrain R4H manifest')
    need(cm.get('stage')=='R4G-TRAIN-SECTION-SNAPSHOT-v1' and cm.get('split')=='train' and cm.get('arm')=='capture','Nontrain R4G capture')
    need(pr.get('stage')=='R4G-TRAIN-PREFIX-EXACT-v1' and pr.get('status')=='COMPLETE' and pr.get('data_role')=='TRAIN_ONLY' and pr['n_comparisons']==pr['exact_matches'],'Exact proof required')
    need(hr['source_prefix_exact_comparisons']==pr['exact_matches']==298,'Proof aggregate mismatch')
    need(hm.get('horizons')==[50,200,500] and hm.get('require_all_k') is True,'Different horizon protocol')
    need(hm['source_sha256']['prefix_report']==sha(pdir/'report.json') and
         hm['source_sha256']['prefix_manifest']==sha(pdir/'manifest.json') and
         hm['source_sha256']['prefix_paired_rows']==sha(pdir/'paired_rows.jsonl') and
         hm['source_sha256']['prefix_excluded_windows']==sha(pdir/'excluded_windows.json'),'Prefix proof changed')
    for n in ('manifest.json','snapshots.jsonl','metrics.jsonl','provenance.json'):
        field={'manifest.json':'capture_manifest','snapshots.jsonl':'capture_snapshots',
               'metrics.jsonl':'capture_metrics','provenance.json':'capture_provenance'}[n]
        need(hm['source_sha256'][field]==sha(cdir/n),'Capture provenance differs '+n)
    labs,branches,plans,snaps=(readl(p) for p in
       (hdir/'labels.jsonl',hdir/'branch_summary.jsonl',hdir/'plans.jsonl',cdir/'snapshots.jsonl'))
    need(len(labs)==hr['n_labels'] and len(branches)==hr['n_branches'] and len(plans)==hr['n_scenario_snapshots'],'R4H row count mismatch')
    need(hr['n_sham_checks']==hr['n_scenario_snapshots']*3 and hr['n_cross_k_prefix_checks']>0,'R4H SHAM/prefix controls missing')
    return hr,hm,labs,branches,plans,snaps,{
        name:sha(path) for name,path in {
          'r4h_report':hdir/'report.json','r4h_manifest':hdir/'manifest.json',
          'r4h_labels':hdir/'labels.jsonl','r4h_branches':hdir/'branch_summary.jsonl',
          'r4h_plans':hdir/'plans.jsonl','r4g_snapshots':cdir/'snapshots.jsonl',
          'prefix_proof':pdir/'paired_rows.jsonl'}.items()}

def validate_dataset(hr,hm,labs,branches,plans,snaps):
    snapmap={key(s):s for s in snaps}
    need(len(snapmap)==len(snaps),'Duplicate R4G capture')
    selected={key(p['snapshot_id']):p for p in plans}
    need(len(selected)==len(plans) and set(selected)=={key(i) for i in hm['selected_snapshot_ids']},'Snapshot manifest/plan mismatch')
    need(set(selected)<=set(snapmap),'R4H snapshot missing from capture')
    bmap={armkey(b):b for b in branches}
    lmap={armkey(x):x for x in labs}
    need(len(bmap)==len(branches) and len(lmap)==len(labs),'Duplicate action horizon')
    features={};label_by_snapshot=defaultdict(dict)
    for k,p in selected.items():
        s=snapmap[k]
        need(s['snapshot_is_actual_section_decision'] is True and s['real_selected_destroy_index']==3,'Non-Section source')
        need(s['scenario_id'].startswith('2025-07-02_') or s['scenario_id'].startswith('2025-07-09_'),'Unapproved TRAIN scenario')
        need(s['task_id'] in cm_tasks(hr,selected),'Task mismatch')
        for h in H:
            c=bmap.get(k+(h,'C3'));sh=bmap.get(k+(h,'SHAM_C3'))
            need(c is not None and sh is not None and c['best_route_sha256']==sh['best_route_sha256'] and
                 c['best_road_sec']==sh['best_road_sec'],'Baseline/SHAM source mismatch')
        for x in [z for z in labs if key(z['unit_spec'])==k]:
            i=armkey(x);h=i[-2];arm=i[-1]
            need(h in H and arm not in ('C3','SHAM_C3') and i in bmap,'Invalid action label')
            c=bmap[k+(h,'C3')];alt=bmap[i]
            need(abs((c['best_road_sec']-alt['best_road_sec'])-x['road_best_gain_sec'])<EPS,'Reward/branch mismatch')
            need(x['same_best_routes']==(c['best_route_sha256']==alt['best_route_sha256']),'Best SHA mismatched')
            need(tuple(x['original_c3_action_id'])==tuple(p['c3_identity']),'C3 action differs from plan')
            label_by_snapshot[k].setdefault(arm,{})[h]=x
        need(len(label_by_snapshot[k])==p['n_alternatives'] and 1<=p['n_alternatives']<=8,'Action sample mismatch')
        for arm,vs in label_by_snapshot[k].items():
            need(set(vs)==set(H),'Unpaired horizon action')
            action=vs[50]['first_action_id'];need(all(vs[h]['first_action_id']==action for h in H),'Changing action across horizon')
            features[k+(arm,)]=predecision_features(s,p,action)
    need(len(features)*3==len(labs),'Orphan labels')
    need(len({physics(snapmap[k]) for k in selected})==hr['n_distinct_physical_states'],'Physical state count disagrees with source')
    return selected,snapmap,label_by_snapshot,features

def cm_tasks(hr,selected):
    need(all(k[0] in {'task_0000','task_0001','task_0002','task_0003'} for k in selected),'Unexpected TRAIN task')
    return {'task_0000','task_0001','task_0002','task_0003'}

def predecision_features(s,p,action):
    routes=s['current_routes'];route=int(action[0]);pos=int(action[1]);size=int(action[2])
    need(0<=route<len(routes) and 0<=pos<=len(routes[route])-size and size in (2,3),'Invalid section geometry')
    c3=p['c3_identity'];gap=max(0.,float(s['current_surrogate_cost'])-float(s['best_surrogate_cost']))
    length=max(1,len(routes[route]));anneal=max(0.,float(s['anneal_temperature_before_accept']))
    wall=max(0.,float(s['wall_remaining_before_select_sec']))
    f=[1.,min(1.,int(s['stop_count'])/20000.),int(s['stop_count'])<=2000,
       min(1.,wall/30.),min(2.,gap/max(1.,abs(float(s['best_surrogate_cost'])))),
       min(1.,anneal/max(1.,float(s['anneal_start_temperature']))),
       min(1.,length/60.),min(1.,len(routes)/10.),route/max(1,len(routes)-1),
       pos/max(1,length-size),int(size==3),int(route==int(c3[0])),
       min(1.,(abs(route-int(c3[0]))+abs(pos-int(c3[1])))/15.),
       min(1.,int(p['candidate_count'])/150.),int(s['real_selected_repair_index']==1)]
    need(len(f)==len(FEATURES) and all(math.isfinite(v) for v in f),'Nonfinite predecision features')
    return [float(v) for v in f]

def group_report(labels,selected,snapmap,labmap):
    out={};state_groups=defaultdict(list)
    for k in selected:
        s=snapmap[k];state_groups[physics(s)].append(k)
    for h in H:
        tasks={};phase={};items=[]
        for k,arms in labmap.items():
            s=snapmap[k];grp='early_<=2000' if s['stop_count']<=2000 else 'later_>2000'
            for arm,ks in arms.items():
                g=float(ks[h]['road_best_gain_sec']);items.append((k,g))
                for table,nam in [(tasks,k[0]),(phase,grp)]:
                    x=table.setdefault(nam,[]);x.append(g)
        def summ(table):
            return {name:{'n':len(v),'positive':sum(x>EPS for x in v),
                     'negative':sum(x< -EPS for x in v),'zero':sum(abs(x)<=EPS for x in v),
                     'avg_gain_sec':sum(v)/len(v)} for name,v in sorted(table.items())}
        state_positive={physics(snapmap[k]) for k in selected
              if any(float(z[h]['road_best_gain_sec'])>EPS for z in labmap[k].values())}
        out[str(h)]={'by_task':summ(tasks),'by_phase':summ(phase),
                   'positive_physical_states':len(state_positive)}
    return {'per_horizon':out,'n_snapshots':len(selected),'n_physical_states':len(state_groups),
            'n_repeated_state_groups':sum(len(v)>1 for v in state_groups.values())}

def _linear_system(mat,vec):
    a=[list(row)+[vec[i]] for i,row in enumerate(mat)];n=len(a)
    for i in range(n):
        j=max(range(i,n),key=lambda t:abs(a[t][i]))
        need(abs(a[j][i])>1e-12,'Ill-conditioned model')
        a[i],a[j]=a[j],a[i];d=a[i][i]
        for c in range(i,n+1):a[i][c]/=d
        for r in range(n):
            if r==i:continue
            d=a[r][i]
            for c in range(i,n+1):a[r][c]-=d*a[i][c]
    return [a[i][-1] for i in range(n)]

def ridge_predictors(records,target='k500',strength=100.0,seed=0,shuffle_labels=False):
    # Weight equal across TRAIN physical search states, not repeated scenarios/actions.
    need(records,'Empty training records')
    ys=[min(250.,max(-250.,r['targets'][target])) for r in records]
    if shuffle_labels:
        rng=random.Random(seed);rng.shuffle(ys)
    freq=Counter(tuple(r['physical']) for r in records)
    n=len(FEATURES);mat=[[0.]*n for _ in range(n)];vec=[0.]*n
    for r,y in zip(records,ys):
        f=r['features'];w=1./freq[tuple(r['physical'])]
        for i in range(n):
            vec[i]+=w*f[i]*y
            for j in range(n):mat[i][j]+=w*f[i]*f[j]
    for i in range(1,n):mat[i][i]+=strength
    mat[0][0]+=1e-6
    return _linear_system(mat,vec)

def predict(coef,features):return sum(a*b for a,b in zip(coef,features))

def records_from_dataset(selected,snapmap,labmap,features):
    out=[]
    for k,arms in labmap.items():
        for arm,vals in arms.items():
            targets={f'k{h}':float(vals[h]['road_best_gain_sec']) for h in H}
            targets['robust']=min(targets['k200'],targets['k500'])
            out.append({'snap':k,'arm':arm,'task':k[0],'physical':physics(snapmap[k]),
                'features':features[k+(arm,)],'targets':targets})
    return out

def decision_audit(records):
    need(records,'Empty decision records')
    grouped=defaultdict(list)
    for r in records:grouped[r['snap']].append(r)
    folds=[];detail=[]
    for task in sorted({r['task'] for r in records}):
        train=[r for r in records if r['task']!=task]
        held=[r for r in records if r['task']==task]
        need(train and held,'Need multiple original TRAIN tasks')
        for target in ('k500','robust'):
            for null in (False,True):
                weights=ridge_predictors(train,target,seed=20261025+int(task[-1]),shuffle_labels=null)
                for threshold in (0.,5.,15.,30.):
                    picks=[]
                    for k,alternatives in sorted(grouped.items()):
                        if k[0]!=task:continue
                        scored=[(predict(weights,r['features']),r['arm'],r) for r in alternatives]
                        score,arm,r=max(scored,key=lambda v:(v[0],v[1]))
                        chosen=r if score>threshold else None
                        gain=chosen['targets']['k500'] if chosen else 0.
                        picks.append({'task':task,'snap':k,'chosen_arm':chosen['arm'] if chosen else 'C3',
                                    'predicted_margin_sec':score,'actual_k500_gain_sec':gain,
                                    'physical':alternatives[0]['physical'],'null':null,
                                    'target':target,'threshold_sec':threshold,
                                    'oracle_k500_gain_sec':max([0.]+[z['targets']['k500'] for z in alternatives]),
                                    'random_expected_k500_sec':sum(z['targets']['k500'] for z in alternatives)/(len(alternatives)+1)})
                    gain=[p['actual_k500_gain_sec'] for p in picks]
                    stage_random=[p['random_expected_k500_sec'] if p['snap'][3]<=2000 else 0. for p in picks]
                    # This is descriptive TRAIN-task-held-out CV, not blind validation.
                    folds.append({'heldout_train_task':task,'training_tasks':sorted({r['task'] for r in train}),
                                  'target':target,'negative_control_shuffled_train_labels':null,'threshold_sec':threshold,
                                  'n_snapshots':len(picks),'delegations':sum(p['chosen_arm']!='C3' for p in picks),
                                  'positive_delegations':sum(x>EPS for x in gain),
                                  'harmful_delegations':sum(x< -EPS for x in gain),
                                  'mean_k500_gain_sec':sum(gain)/len(gain),
                                  'sampled_oracle_mean_sec':sum(p['oracle_k500_gain_sec'] for p in picks)/len(picks),
                                  'random_expected_mean_sec':sum(p['random_expected_k500_sec'] for p in picks)/len(picks),
                                  'stage_early_only_random_expected_mean_sec':sum(stage_random)/len(picks)})
                    detail.extend(picks)
    return folds,detail

def to_markdown(rep):
    o=['# R4-I Train状态收益可学习性与保守委托诊断','',
       f"数据角色：**TRAIN ONLY**；验证类型：**Leave-One-Train-Task-Out，仅探索性**。",
       f"快照：{rep['signal']['n_snapshots']}；物理状态：{rep['signal']['n_physical_states']}。",'',
       '| K | 搜索阶段 | 候选数 | 正/负/零收益 |','|---:|---|---:|---:|']
    for k,v in rep['signal']['per_horizon'].items():
        for stage,c in v['by_phase'].items():
            o.append(f"| {k} | {stage} | {c['n']} | {c['positive']}/{c['negative']}/{c['zero']} |")
    o+=['','## 任务级留一交叉评估（注意：仍为Train内部探索）','',
         '| 留出任务 | 目标 | Shuffle负对照 | 门槛(s) | 委托数 | 正/负 | 实际K500平均收益(s) | 事后Oracle均值(s) |',
         '|---|---|---|---:|---:|---|---:|---:|']
    for x in rep['folds']:
        if x['threshold_sec'] in (0.,15.):
            o.append(f"| {x['heldout_train_task']} | {x['target']} | {x['negative_control_shuffled_train_labels']} | {x['threshold_sec']:.0f} | {x['delegations']} | {x['positive_delegations']}/{x['harmful_delegations']} | {x['mean_k500_gain_sec']:+.3f} | {x['sampled_oracle_mean_sec']:+.3f} |")
    o+=['','**禁止解释：** 所有结果来自Train标签的离线重放；不存在额外在线ALNS部署、独立Validation/Test或真实30秒改善。',
        '事后Oracle使用未来标签，是不可部署的上界。Shuffle与任务留一为负对照/稳健性诊断，不能将试过的门槛选择结果宣称为外部泛化。','']
    return '\n'.join(o)
