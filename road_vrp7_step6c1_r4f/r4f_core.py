"""R4-F pure, audited longitudinal diagnostics for ALNS Section counterfactuals.

No model fitting, no Road.replay recomputation and no claim of 30s value.
"""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

STAGE='R4F-HORIZON-REVERSAL-AUDIT-v1'
LONG='R4E-LONG-SECTION-CF-v1'
SHORT='R4C-CONTINUOUS-SECTION-COUNTERFACTUAL-v1'
KS=(50,100,200,500)
EPS=1e-8


def require(cond,msg):
    if not cond:raise ValueError(msg)


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(path)


def ident(spec):
    if isinstance(spec,(list,tuple)):
        require(len(spec)==4,'Malformed snapshot tuple/list')
        return (str(spec[0]),str(spec[1]),int(spec[2]),int(spec[3]))
    return (str(spec['task_id']),str(spec['scenario_id']),int(spec['seed']),int(spec['stop_count']))


def action_key(z):return ident(z['unit_spec'])+(str(z['unit_spec']['arm']),)


def horizon_key(z):return action_key(z)+(int(z['unit_spec']['horizon']),)


def physical(x):
    require(isinstance(x,list) and len(x)==6,'Wrong physical search ID shape')
    return tuple(x)


def sign(value):
    x=float(value)
    return '+' if x>EPS else '-' if x< -EPS else '0'


def index_unique(items,k,tag):
    out={}
    for z in items:
        key=k(z)
        require(key not in out,f'Duplicate {tag}: {key}')
        out[key]=z
    return out


def audit_sources(long_dir,short_dir):
    ld,sd=Path(long_dir),Path(short_dir)
    lr,lm=read(ld/'report.json'),read(ld/'manifest.json')
    sr,sm=read(sd/'report.json'),read(sd/'manifest.json')
    require(lr.get('stage')==LONG and lr.get('status')=='COMPLETE','R4E is not a completed counterfactual study')
    require(sr.get('stage')==SHORT and sr.get('status')=='COMPLETE','R4C is not a completed counterfactual study')
    require(lm.get('stage')==LONG and lm.get('status')=='COMPLETE','R4E manifest incomplete')
    require(sm.get('stage')==SHORT and sm.get('evaluation')=='FIXED_ITERATION_K_NOT_WALL_CLOCK_30SEC','R4C role mismatch')
    require(lm.get('r4c_manifest_sha256')==sha(sd/'manifest.json'),'R4E/R4C manifest SHA differs')
    require(lm.get('r4c_report_sha256')==sha(sd/'report.json'),'R4E/R4C report SHA differs')
    require(set(map(int,lr['horizons']))=={100,200,500} and set(map(int,lm['horizons']))=={100,200,500},'Require K100/200/500')
    require(lm.get('data_role')=='INSPECTED_JULY_VALIDATION_DEVELOPMENT_ONLY','Unexpected R4E data role')
    ls,lb,ss,sb=[rows(p) for p in [ld/'labels.jsonl',ld/'branches.jsonl',sd/'counterfactual_labels.jsonl',sd/'branch_summary.jsonl']]
    require(len(ls)==lr['n_candidate_horizon_labels'] and len(lb)==lr['n_branch_results'],'R4E row count mismatch')
    require(len(ss)==sr['n_counterfactual_labels'] and len(sb)==sr['n_branch_results'],'R4C row count mismatch')
    require(lr['n_snapshots']==len(lm['selected_snapshots']),'Selected snapshot count mismatch')
    require(lr['n_sham_verified']==lr['n_snapshots']*3 and lr['n_prior_k50_prefix_verified']==len(lb),
            'Missing SHAM or K50 prefix audit')
    require(lr['n_branch_results']==lr['n_candidate_horizon_labels']+lr['n_snapshots']*3*2,
            'Unexpected branch/alternative cardinality')
    return (lr,lm,ls,lb,ss,sb), {name:sha(path) for name,path in {
        'r4e_report':ld/'report.json','r4e_manifest':ld/'manifest.json',
        'r4e_labels':ld/'labels.jsonl','r4e_branches':ld/'branches.jsonl',
        'r4c_report':sd/'report.json','r4c_manifest':sd/'manifest.json',
        'r4c_labels':sd/'counterfactual_labels.jsonl','r4c_branches':sd/'branch_summary.jsonl'
    }.items()}


def audit_data(lr,lm,long_labels,long_branches,short_labels,short_branches):
    L=index_unique(long_labels,horizon_key,'R4E label')
    B=index_unique(long_branches,horizon_key,'R4E branch')
    S=index_unique(short_labels,horizon_key,'R4C label')
    SB=index_unique(short_branches,horizon_key,'R4C branch')
    starts=set(ident(z) for z in lm['selected_snapshots'])
    require(len(starts)==lr['n_snapshots'],'Duplicate selected start')
    require(set(ident(x['unit_spec']) for x in long_labels)==starts,'R4E label start coverage incomplete')
    require(set(ident(x['unit_spec']) for x in long_branches)==starts,'R4E branch start coverage incomplete')
    require(len(L)==len(long_labels) and len(B)==len(long_branches),'Duplicate horizon rows')
    require({int(k[-1]) for k in L}=={100,200,500},'Unexpected horizon keys')
    unique_physical={};per_key={};by_start=defaultdict(set)
    for key,z in L.items():
        start,arm,h=key[:4],key[4],key[5]
        require(arm not in ('C3','SHAM_C3'),'C3/SHAM in candidate labels')
        require(B.get(start+('C3',h)) and B.get(start+('SHAM_C3',h)) and B.get(key),'Missing paired R4E branch')
        base,alt,sham=B[start+('C3',h)],B[key],B[start+('SHAM_C3',h)]
        require(base['best_road_sec']==sham['best_road_sec'] and base['best_route_sha256']==sham['best_route_sha256'],
                'SHAM best route or road cost mismatch')
        require(math.isclose(float(z['best_road_gain_sec']),float(base['best_road_sec'])-float(alt['best_road_sec']),abs_tol=EPS,rel_tol=0),
                'Road label does not equal paired branch cost difference')
        require(bool(z['same_best_routes'])==(base['best_route_sha256']==alt['best_route_sha256']),
                'Best-route same flag mismatch')
        require(int(h) in (100,200,500),'Bad horizon')
        sid=physical(z['physical_id'])
        require(start[:1]+(start[2],start[3])==sid[:3],'Physical ID task/seed/iteration inconsistent')
        require(start not in unique_physical or unique_physical[start]==sid,'Same snapshot has conflicting physical IDs')
        unique_physical[start]=sid
        per_key.setdefault(key[:5],{})[h]=z
        by_start[start].add((arm,h))
    for start in starts:
        n=len({a for a,h in by_start[start] if h==100})
        require(1<=n<=8,'No valid alternatives at start')
        require(len(by_start[start])==n*3,'Inconsistent alternatives across long horizons')
        for h in (100,200,500):
            arms={a for a,hh in by_start[start] if hh==h}
            require(('C3',h) in {k[-2:] for k in B if k[:4]==start},'Missing C3 branch')
            require(arms=={a for a,hh in by_start[start] if hh==100},'Inconsistent action list across K')
    for ak,hmap in per_key.items():
        require(set(hmap)=={100,200,500},'Not all three horizons evaluated for one action')
        old=S.get(ak+(50,))
        require(old is not None,'Long action not present in original K50 audited dataset')
        old_val=float(old['road_best_gain_sec'])
        short_base=SB.get(ak[:4]+('C3',50))
        short_sham=SB.get(ak[:4]+('SHAM_C3',50))
        short_alt=SB.get(ak+(50,))
        require(short_base is not None and short_sham is not None and short_alt is not None,
                'R4C K50 branch triple incomplete')
        require(short_base['sha']==short_sham['sha'] and short_base['best_road']==short_sham['best_road'],
                'Original R4C K50 SHAM mismatched')
        require(math.isclose(old_val, float(short_base['best_road'])-float(short_alt['best_road']),
                             abs_tol=EPS,rel_tol=0),'R4C K50 cost/label mismatch')
        require(bool(old['same_best_routes'])==(short_base['sha']==short_alt['sha']),
                'R4C K50 best routes label mismatch')
        hmap[50]={'best_road_gain_sec':old_val,'best_static_gain_sec':float(old['static_best_gain_sec']),
                  'same_best_routes':bool(old['same_best_routes'])}
        for h in (100,200,500):
            z=hmap[h]
            require(z['first_action_id']==hmap[100]['first_action_id'] and
                    z['original_c3_action_id']==hmap[100]['original_c3_action_id'],
                    'First action changed between horizons')
            require(z['first_accepted']==hmap[100]['first_accepted'] and
                    z['first_outcome']==hmap[100]['first_outcome'],
                    'First-step acceptance/outcome changed across horizons')
        require(tuple(old['first_action'])==tuple(hmap[100]['first_action_id']),
                'K50 first action differs from R4E')
        require(tuple(old['c3_action'])==tuple(hmap[100]['original_c3_action_id']),
                'K50 actual C3 action differs from R4E')
    require(len(per_key)*3==len(long_labels),'Not all action horizons complete')
    require(len(set(unique_physical.values()))==lr['n_physical_states'],'Physical state count mismatch')
    require(len(starts)==lr['n_snapshots'],'Snapshot count mismatch')
    # Inspect longitudinal *best-road* baselines and candidate branches, including K50.
    def road(start,arm,h):
        src=SB if h==50 else B
        row=src.get(start+(arm,h))
        require(row is not None,f'Missing branch Road cost for {start}, {arm}, K{h}')
        return float(row['best_road'] if h==50 else row['best_road_sec'])
    return per_key,unique_physical,road


def analyze(per_key,unique_physical,road):
    grouped=defaultdict(list)
    for ak,hmap in per_key.items():grouped[ak[:4]].append((ak,hmap))
    horizon_summary={};transitions={};oracle_rows=[];action_rows=[]
    for h in KS:
        gains=[float(v[h]['best_road_gain_sec']) for vv in per_key.values() for v in [vv]]
        per_state={s:max([0.]+[float(m[h]['best_road_gain_sec']) for _,m in seq]) for s,seq in grouped.items()}
        horizon_summary[str(h)]={'n_actions':len(gains),'positive':sum(sign(v)=='+' for v in gains),
              'negative':sum(sign(v)=='-' for v in gains),'zero':sum(sign(v)=='0' for v in gains),
              'mean_road_gain_sec':mean(gains),'oracle_positive_scenario_snapshots':sum(v>EPS for v in per_state.values()),
              'mean_sampled_oracle_gain_sec':mean(per_state.values())}
        for s,g in sorted(per_state.items()):
            oracle_rows.append({'task_id':s[0],'scenario_id':s[1],'seed':s[2],
                'stop_count':s[3],'physical_id':list(unique_physical[s]),
                'horizon':h,'sampled_oracle_sec':g})
    for a,b in zip(KS[:-1],KS[1:]):
        counter=Counter();reversed_rows=[];sample=[]
        for ak,hmap in per_key.items():
            g0=float(hmap[a]['best_road_gain_sec']);g1=float(hmap[b]['best_road_gain_sec'])
            category=sign(g0)+'→'+sign(g1);counter[category]+=1
            s=ak[:4];arm=ak[4]
            base_progress=road(s,'C3',a)-road(s,'C3',b)
            alt_progress=road(s,arm,a)-road(s,arm,b)
            # Identity: delta later - delta earlier = alt progress - C3 progress.
            require(math.isclose(g1-g0,alt_progress-base_progress,abs_tol=1e-7,rel_tol=0),
                    'Longitudinal cost decomposition failed')
            rec={'task_id':s[0],'scenario_id':s[1],'seed':s[2],'stop_count':s[3],
                 'arm':arm,'physical_id':list(unique_physical[s]),'from_K':a,'to_K':b,
                 'gain_before_sec':g0,'gain_after_sec':g1,'gain_change_sec':g1-g0,
                 'C3_road_progress_sec':base_progress,'alternative_road_progress_sec':alt_progress,
                 'sign_transition':category,'first_accepted':bool(hmap[100]['first_accepted']),
                 'static_best_gain_before_sec':float(hmap[a]['best_static_gain_sec']),
                 'static_best_gain_after_sec':float(hmap[b]['best_static_gain_sec'])}
            sample.append(rec)
            if category=='+→-':reversed_rows.append(rec)
        transitions[f'{a}_to_{b}']={'sign_counts':dict(sorted(counter.items())),
            'positive_to_negative':len(reversed_rows),
            'positive_to_negative_distinct_physical':len({tuple(r['physical_id']) for r in reversed_rows}),
            'avg_C3_progress_in_flips_sec':mean([r['C3_road_progress_sec'] for r in reversed_rows]) if reversed_rows else 0.,
            'avg_alt_progress_in_flips_sec':mean([r['alternative_road_progress_sec'] for r in reversed_rows]) if reversed_rows else 0.}
        action_rows.extend(sample)
    # A deliberately NON-DEPLOYABLE hindsight experiment: if an oracle picked
    # the best K=200 candidate, would that candidate still win at K=500?
    horizon_oracle_transfer=[]
    for snap,options in sorted(grouped.items()):
        ak,vec=max(options,key=lambda it:(float(it[1][200]['best_road_gain_sec']),it[0][4]))
        early=float(vec[200]['best_road_gain_sec'])
        later=float(vec[500]['best_road_gain_sec']) if early>EPS else 0.
        chosen=ak[4] if early>EPS else 'C3'
        horizon_oracle_transfer.append({'task_id':snap[0],'scenario_id':snap[1],
          'seed':snap[2],'stop_count':snap[3],'physical_id':list(unique_physical[snap]),
          'K200_hindsight_action':chosen,'K200_hindsight_gain_sec':max(0.,early),
          'K500_gain_of_K200_hindsight_action_sec':later,
          'K500_best_possible_among_sampled_sec':max([0.]+[float(m[500]['best_road_gain_sec']) for _,m in options])})
    # How much of the long-horizon value is a property of a particular
    # physical search state? Cases/scenarios/actions cannot be treated as IID.
    cluster={}
    for h in KS:
        physical_gains=defaultdict(list)
        for ak,vs in per_key.items():physical_gains[unique_physical[ak[:4]]].append(float(vs[h]['best_road_gain_sec']))
        centers={str(p):mean(v) for p,v in physical_gains.items()}
        # Leave-one-physical-out is only a sensitivity diagnostic, never a CI.
        leave_one={str(p):mean([y for q,v in physical_gains.items() if q!=p for y in v]) if len(physical_gains)>1 else None for p in physical_gains}
        cluster[str(h)]={'physical_state_count':len(centers),
             'equal_physical_state_mean_sec':mean(centers.values()),
             'per_physical_mean_sec':centers,'leave_one_physical_out_action_mean_sec':leave_one}
    transfer_summary={'scenario_snapshots':len(horizon_oracle_transfer),
       'K200_hindsight_positive_selected':sum(x['K200_hindsight_action']!='C3' for x in horizon_oracle_transfer),
       'K500_positive_after_K200_hindsight':sum(x['K500_gain_of_K200_hindsight_action_sec']>EPS for x in horizon_oracle_transfer),
       'K500_negative_after_K200_hindsight':sum(x['K500_gain_of_K200_hindsight_action_sec']< -EPS for x in horizon_oracle_transfer),
       'mean_K500_gain_of_K200_hindsight_sec':mean(x['K500_gain_of_K200_hindsight_action_sec'] for x in horizon_oracle_transfer),
       'mean_K500_sampled_oracle_sec':mean(x['K500_best_possible_among_sampled_sec'] for x in horizon_oracle_transfer),
       'note':'Uses future K200 labels to select action; hindsight stress test only, never policy performance.'}
    return {'per_horizon':horizon_summary,'transitions':transitions,
            'hindsight_horizon_transfer':transfer_summary,
            'physical_state_sensitivity':cluster,'n_actions':len(per_key),
            'n_scenario_snapshots':len(grouped),'n_unique_physical_states':len(set(unique_physical.values()))},action_rows,oracle_rows


def make_markdown(report):
    core=report['analysis'];out=['# R4-F ALNS连续搜索价值反转审计','',
      f"状态：**COMPLETE**；动作：{core['n_actions']}；场景快照：{core['n_scenario_snapshots']}；不同物理状态：{core['n_unique_physical_states']}。",'',
      '| K | 正/负/零 | 平均Road收益(s) | 正收益场景快照 |', '|---:|---:|---:|---:|']
    for h in KS:
        v=core['per_horizon'][str(h)]
        out.append(f"| {h} | {v['positive']}/{v['negative']}/{v['zero']} | {v['mean_road_gain_sec']:+.3f} | {v['oracle_positive_scenario_snapshots']}/{core['n_scenario_snapshots']} |")
    out.extend(['','## 同一动作的收益符号转移','', '| 区间 | 正→负 | 发生反转的物理状态数 | 反转动作对应C3平均Road改善(s) | 替代动作平均Road改善(s) |',
                '|---|---:|---:|---:|---:|'])
    for interval,v in core['transitions'].items():
        out.append(f"| {interval.replace('_to_',' → ')} | {v['positive_to_negative']} | {v['positive_to_negative_distinct_physical']} | {v['avg_C3_progress_in_flips_sec']:+.3f} | {v['avg_alt_progress_in_flips_sec']:+.3f} |")
    out.extend(['','## 统计边界','',
       '- 相同物理状态的不同交通场景不是独立的ALNS搜索轨迹，动作和K也不是独立样本。',
       '- 只做已查看July Validation开发数据的**描述性审计**；不得用这些标签训练或宣称独立验证。',
       '- Oracle是已采样候选的事后上界，不代表可部署策略；K是迭代数而非30秒剩余墙钟。',
       '- 物理状态留一分析仅检查异常值影响；不计算伪独立的p值或置信区间。',''])
    return '\n'.join(out)
