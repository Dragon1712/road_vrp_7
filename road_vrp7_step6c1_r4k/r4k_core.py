"""R4-K: rigorous read-only partial/full Train counterfactual checkpoint audit.

No implicit missing-as-zero; snapshot is usable only with ALL planned arms × K.
This is a retrospective fixed-K replay, not an online 30s policy estimate.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math

STAGE = 'R4K-TRAIN-COUNTERFACTUAL-COVERAGE-v1'
CF_STAGE = 'R4H-TRAIN-SECTION-COUNTERFACTUAL-v1'
PREFIX_STAGE = 'R4G-TRAIN-PREFIX-EXACT-v1'
CAPTURE_STAGE = 'R4G-TRAIN-SECTION-SNAPSHOT-v1'
KS = (50, 200, 500)
EPS = 1e-8


def need(p, message):
    if not p:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def readl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20), b''):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def sid(o):
    if isinstance(o,(list,tuple)):
        need(len(o)==4,'Bad snapshot list')
        return (str(o[0]),str(o[1]),int(o[2]),int(o[3]))
    return (str(o['task_id']),str(o['scenario_id']),int(o['seed']),int(o['stop_count']))


def spec_key(o):
    return (*sid(o),int(o['horizon']),str(o['arm']))


def official_unit_filename(o):
    """Identical to frozen r4c_core.unit_name/branch_spec, without importing ALNS."""
    b=json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    return hashlib.sha256(b).hexdigest()[:32]+'.json'


def state_phys(s):
    """Exact R4-H state tuple and coarser fixed-stochastic-start tuple."""
    def digest(v):
        return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    t=float(s['anneal_temperature_before_accept'])
    need(math.isfinite(t),'Nonfinite SA temperature')
    shared=(s['task_id'],int(s['seed']),int(s['stop_count']),s['current_route_sha256'],s['best_route_sha256'],s['rng_state_sha256'],digest(s['roulette_destroy_weights']),digest(s['roulette_repair_weights']))
    return (*shared,t),shared


def preflight(cf_dir,prefix_dir,capture_dir):
    cf=Path(cf_dir);pf=Path(prefix_dir);cp=Path(capture_dir)
    m=read(cf/'manifest.json');pr=read(pf/'report.json');pm=read(pf/'manifest.json');cm=read(cp/'manifest.json')
    need(m.get('stage')==CF_STAGE and m.get('data_role')=='TRAIN_ONLY','Wrong CF stage/role')
    need(m.get('eval_semantics')=='FIXED_K_NOT_REMAINING_30S','CF is not exact fixed-K schema')
    need(m.get('horizons')==list(KS) and m.get('require_all_k') is True,'Expected K50/200/500 complete-action protocol')
    need(pm.get('stage')==pr.get('stage')==PREFIX_STAGE and pr.get('status')=='COMPLETE' and pr.get('data_role')=='TRAIN_ONLY','Prefix not completed Train proof')
    need(pr.get('exact_matches')==pr.get('n_comparisons') and pr['n_comparisons']>0,'Prefix exact count mismatch')
    need(cm.get('stage')==CAPTURE_STAGE and cm.get('arm')=='capture' and cm.get('split')=='train','Capture role mismatch')
    need(set(cm['task_ids'])<=set(f'task_{i:04d}' for i in range(4)),'Not original TRAIN tasks')
    need(set(cm['seeds']).isdisjoint({60,61}),'Not a new-seed R4-J capture')
    paths={
       'prefix_report':pf/'report.json','prefix_manifest':pf/'manifest.json',
       'prefix_paired_rows':pf/'paired_rows.jsonl','prefix_excluded_windows':pf/'excluded_windows.json',
       'capture_manifest':cp/'manifest.json','capture_provenance':cp/'provenance.json',
       'capture_snapshots':cp/'snapshots.jsonl','capture_metrics':cp/'metrics.jsonl'
    }
    pins={}
    for k,p in paths.items():
        pins[k]=hash_file(p)
        need(m['source_sha256'][k]==pins[k],f'CF source SHA pin disagrees: {k}')
    # Audit the actual frozen production files, when pinned by the source study.
    # In synthetic fixtures these keys are absent; official R4-H manifests
    # include them and any mismatch aborts before checkpoint aggregation.
    repo=Path(__file__).resolve().parents[1]
    code_paths={'frozen_solver':'road_vrp7_step5b_v3/run_experiment.py',
        'frozen_section':'road_vrp7_step5b_v3/section_v2.py',
        'frozen_r4a':'road_vrp7_step6c1_r4a/r4a_core.py',
        'frozen_r4b':'road_vrp7_step6c1_r4b/r4b_core.py',
        'r4c_run_one':'road_vrp7_step6c1_r4c/counterfactual_r4c.py',
        'r4c_core':'road_vrp7_step6c1_r4c/r4c_core.py',
        'r4h_collector':'road_vrp7_step6c1_r4h/collect_train_cf.py',
        'r4h_core':'road_vrp7_step6c1_r4h/r4h_core.py'}
    for key,path in code_paths.items():
        if key in m['source_sha256']:
            actual=repo/path
            need(actual.is_file() and hash_file(actual)==m['source_sha256'][key],
                 f'Frozen source file SHA changed: {key}')
    need(pr['original_capture_snapshots_sha256']==pins['capture_snapshots'],'Prefix/capture source mismatch')
    cr=read(cp/'report.json'); need(cr.get('status')=='COMPLETE','Source capture incomplete')
    source=readl(cp/'snapshots.jsonl');need(len(source)==cr['snapshots'],'Capture row count wrong')
    src={sid(s):s for s in source};need(len(src)==len(source),'Duplicate source snapshots')
    planned=[sid(x) for x in m['selected_snapshot_ids']]
    need(len(set(planned))==len(planned),'Duplicate selected snapshots')
    need(set(planned)<=set(src),'CF requested snapshots outside capture')
    for s in planned:
        need(s[0] in cm['task_ids'] and s[1] in cm['scenario_ids'] and s[2] in cm['seeds'],'CF not in authorized Train grid')
    rows=readl(pf/'paired_rows.jsonl')
    need(len(rows)==pr['n_comparisons'],'Prefix proof row count differs')
    verified=set()
    for x in rows:
        key=(str(x['task_id']),str(x['scenario_id']),int(x['seed']),int(x['snapshot_stop_count']),int(x['horizon']))
        need(key not in verified and x.get('exact') is True and int(x['verified_steps'])==key[4], 'Incorrect proof row')
        verified.add(key)
    for s in planned:
        need(all((*s,k) in verified for k in KS),'CF selected an unverified K')
    return m,src,planned,{'hashes':pins,'prefix_rows':len(rows),'source_snapshots':len(source)}


def parse_units(cf_dir,m,src,planned):
    """Validate ALL existing units; do not silently ignore orphan/invalid units."""
    files=sorted((Path(cf_dir)/'units').glob('*.json'))
    allowed=set(planned)
    by_snap=defaultdict(dict); plans={}
    for p in files:
        z=read(p);s=z['unit_spec'];key=spec_key(s);sp=key[:4];k=key[4];arm=key[5];plan=z['plan'];res=z['result']
        need(sp in allowed and k in KS,'Out-of-selection CF unit')
        need(p.name==official_unit_filename(s),'Unit filename differs from frozen R4C digest')
        need(z.get('stage')==CF_STAGE and z.get('source_manifest_sha256')==m['source_sha256']['prefix_manifest'] and z.get('source_snapshots_sha256')==m['source_sha256']['capture_snapshots'],'Changed unit provenance')
        need(key not in by_snap[sp],'Duplicate existing unit')
        alts=plan['alternatives'];arms=['C3','SHAM_C3']+[x['name'] for x in alts]
        need(arm in arms and len(set(arms))==len(arms),'Illegal/duplicate arm')
        need(len(alts)<=m['max_alternatives'] and len(alts)>0,'Bad alternative cap')
        need(plan.get('sampling_seed')==m['sampler_seed'],'Sampling seed changed')
        need(sp not in plans or plans[sp]==plan,'Plan changed across K/arms')
        plans[sp]=plan
        need(res.get('arm')==arm and int(res.get('horizon'))==k and sid(res['snapshot_spec'])==sp,'Result identity mismatch')
        events=res['events'];need(len(events)==k,'Truncated event trace')
        first=events[0];orig=src[sp]
        for attr in ('current_route_sha256','best_route_sha256','rng_state_sha256'):
            need(first[attr]==orig[attr],f'First event mismatch: {attr}')
        need(first['stop_count']==sp[3] and first['destroy_index']==3 and first['repair_index']==orig['real_selected_repair_index'],'Wrong checkpoint/operator boundary')
        for i,e in enumerate(events):need(e['stop_count']==sp[3]+i,'Nonconsecutive trace event')
        need(res['first_original_c3_action_id']==plan['c3_identity'],'C3 action identity mismatch')
        audit=res['first_section_audit'];need(audit.get('original_destroy_draws_preserved') is True,'C3 RNG not preserved')
        if arm=='C3':
            need(audit['substitution_calls']==0 and res['first_action_id']==plan['c3_identity'],'C3 branch altered')
        elif arm=='SHAM_C3':
            need(audit['substitution_calls']==1 and audit.get('sham_partial_equal') is True and res['first_action_id']==plan['c3_identity'],'SHAM not equivalently constructed')
        else:
            chosen=next(x for x in alts if x['name']==arm)
            need(audit['substitution_calls']==1 and res['first_action_id']==[chosen['route'],chosen['pos'],chosen['size']],'Alternative action changed')
        # Keep only paths, NOT the full event traces of thousands of branches.
        # The full study may otherwise exhaust RAM (>1 GB of JSON events).
        by_snap[sp][(k,arm)]=p
    return files,by_snap,plans


def sign(v):return '+' if v>EPS else '-' if v< -EPS else '0'


def analyze(planned,source,units,by_snap,plans):
    complete=[];incomplete=[];rewards=[];oracle=[];progress=[]
    horizon_coverage=Counter();arm_coverage=Counter();complete_per_task=Counter()
    for s in planned:
        actual=by_snap.get(s,{})
        if s not in plans:
            incomplete.append({'snapshot_id':list(s),'available_units':0,'expected_units':None,'missing':'NO_PLAN_YET'})
            continue
        plan=plans[s];arms=['C3','SHAM_C3']+[x['name'] for x in plan['alternatives']]
        expected={(k,arm) for k in KS for arm in arms}
        need(set(actual)<=expected,'Unplanned arm found')
        # Any available C3/SHAM pair must already be noninterfering, even when
        # the rest of a snapshot has not been computed yet.
        for h in KS:
            if (h,'C3') in actual and (h,'SHAM_C3') in actual:
                left=read(actual[(h,'C3')])['result'];right=read(actual[(h,'SHAM_C3')])['result']
                for field in ('events','end_signature','best_route_sha256','current_route_sha256','road_best_sec','road_current_sec','static_best_sec','static_current_sec'):
                    need(left[field]==right[field],f'C3/SHAM mismatch at {s} K{h}: {field}')
        missing=sorted(expected-set(actual))
        if missing:
            incomplete.append({'snapshot_id':list(s),'available_units':len(actual),'expected_units':len(expected),
                'missing':[[k,a] for k,a in missing]})
            continue
        # Parse only one complete snapshot's units into memory at a time.
        actual={(k,arm):read(path)['result'] for (k,arm),path in actual.items()}
        # Snapshot-level eligibility is all arms * all K, no censoring bias disguised as zero.
        for k in KS:
            base=actual[(k,'C3')];sham=actual[(k,'SHAM_C3')]
            for field in ('events','end_signature','best_route_sha256','current_route_sha256','road_best_sec','road_current_sec','static_best_sec','static_current_sec'):
                need(base[field]==sham[field],f'C3/SHAM mismatch at {s} K{k}: {field}')
            horizon_coverage[k]+=1
            arm_coverage[k]+=len(arms)-2
            best=0.0
            for arm in arms[2:]:
                alt=actual[(k,arm)]
                gain=float(base['road_best_sec'])-float(alt['road_best_sec'])
                rew={'snapshot_id':list(s),'task_id':s[0],'scenario_id':s[1],'seed':s[2],
                     'stop_count':s[3],'arm':arm,'horizon':k,'road_best_gain_sec':gain,
                     'static_best_gain_sec':float(base['static_best_sec'])-float(alt['static_best_sec']),
                     'road_current_gain_sec':float(base['road_current_sec'])-float(alt['road_current_sec'])}
                rewards.append(rew);best=max(best,gain)
            oracle.append({'snapshot_id':list(s),'horizon':k,'sampled_oracle_sec':best})
        for arm in arms:
            last=[]
            for k in KS:
                now=actual[(k,arm)]['events']
                need(not last or now[:len(last)]==last,'Cross-K prefix inconsistency')
                last=now
        for arm in arms[2:]:
            for k0,k1 in [(50,200),(200,500)]:
                b0,b1=actual[(k0,'C3')],actual[(k1,'C3')]
                a0,a1=actual[(k0,arm)],actual[(k1,arm)]
                c3_progress=b0['road_best_sec']-b1['road_best_sec']
                alt_progress=a0['road_best_sec']-a1['road_best_sec']
                g0=b0['road_best_sec']-a0['road_best_sec'];g1=b1['road_best_sec']-a1['road_best_sec']
                need(math.isclose(g1-g0,alt_progress-c3_progress,abs_tol=1e-7,rel_tol=0),'Road progress decomposition wrong')
                progress.append({'snapshot_id':list(s),'arm':arm,'from_k':k0,'to_k':k1,
                    'gain_before':g0,'gain_after':g1,'C3_progress_sec':c3_progress,
                    'alternative_progress_sec':alt_progress,'transition':sign(g0)+'->'+sign(g1)})
        complete.append(s);complete_per_task[s[0]]+=1
    phys_exact={state_phys(source[s])[0] for s in complete}
    phys_coarse={state_phys(source[s])[1] for s in complete}
    by_h={}
    for h in KS:
        a=[z for z in rewards if z['horizon']==h]
        by_h[str(h)]={'fully_evaluated_snapshots':horizon_coverage[h],
            'action_labels':len(a),'positive':sum(z['road_best_gain_sec']>EPS for z in a),
            'negative':sum(z['road_best_gain_sec']< -EPS for z in a),'zero':sum(abs(z['road_best_gain_sec'])<=EPS for z in a),
            'mean_action_gain_sec':sum(z['road_best_gain_sec'] for z in a)/len(a) if a else None,
            'positive_oracle_snapshots':sum(z['sampled_oracle_sec']>EPS for z in oracle if z['horizon']==h),
            'per_task':{t:{'n':len(v),'positive':sum(z['road_best_gain_sec']>EPS for z in v),
                       'negative':sum(z['road_best_gain_sec']< -EPS for z in v)}
               for t in sorted({z['task_id'] for z in a}) if (v:=[z for z in a if z['task_id']==t])}}
    flips={}
    for pair in [(50,200),(200,500)]:
        subset=[z for z in progress if (z['from_k'],z['to_k'])==pair]
        bad=[z for z in subset if z['transition']=='+->-']
        flips[str(pair[0])+'_to_'+str(pair[1])]={'n_pairs':len(subset),'sign_counts':dict(Counter(z['transition'] for z in subset)),
            'positive_to_negative':len(bad),
            'flipped_scenario_snapshots':len({tuple(z['snapshot_id']) for z in bad}),
            'mean_C3_progress_in_flips':sum(z['C3_progress_sec'] for z in bad)/len(bad) if bad else 0,
            'mean_alt_progress_in_flips':sum(z['alternative_progress_sec'] for z in bad)/len(bad) if bad else 0}
    return {'planned_scenario_snapshots':len(planned),'completed_scenario_snapshots':len(complete),
      'incomplete_scenario_snapshots':len(incomplete),'existing_unit_files':len(units),
      'complete_exact_physical_states':len(phys_exact),'complete_coarse_search_starts':len(phys_coarse),
      'complete_per_task':dict(complete_per_task),'per_horizon':by_h,'horizon_transitions':flips,
      'limitations':['PARTIAL sampling is execution-order selected, not representative or IID.',
         'Only complete snapshot/arm/K triples enter reward metrics; no missing-as-zero.',
         'Posthoc fixed-K Road.replay, not 30-second online policy performance.',
         'Exact physical SA temperatures can differ slightly across scenarios; group correlations remain.',
         'No model fitted, no Validation/Test used.']},incomplete,rewards,oracle,progress


def render(report):
    x=report['analysis'];ls=['# R4-K Train反事实checkpoint完整性与阶段价值审计','',
       f"状态：**{report['status']}**；已有unit：**{x['existing_unit_files']}**；可全K审计快照：**{x['completed_scenario_snapshots']}/{x['planned_scenario_snapshots']}**；粗粒度搜索起点：**{x['complete_coarse_search_starts']}**。",
       f"完整快照任务覆盖：{json.dumps(x['complete_per_task'],ensure_ascii=False)}。",'',
       '| K | 完整快照 | 标签数 | 正/负/零 | 平均Road收益(s) | 含正收益Oracle快照 |',
       '|---:|---:|---:|---:|---:|---:|']
    for k,v in x['per_horizon'].items():
        av=v['mean_action_gain_sec'];val=f'{av:+.3f}' if av is not None else 'N/A'
        ls.append(f"| {k} | {v['fully_evaluated_snapshots']} | {v['action_labels']} | {v['positive']}/{v['negative']}/{v['zero']} | {val} | {v['positive_oracle_snapshots']} |")
    ls+=['','## K收益符号反转（仅完整快照）','']
    for k,v in x['horizon_transitions'].items():
        ls.append(f"- {k}: 正→负 {v['positive_to_negative']}，C3进步均值 {v['mean_C3_progress_in_flips']:+.2f}s，替代进步均值 {v['mean_alt_progress_in_flips']:+.2f}s。")
    ls+=['','## 数据边界','',
         '- 部分结果是执行顺序的前缀样本，禁止据此推断全部128快照的正例率。',
         '- 缺失动作/时域不会记为零，也不会进入学习训练集。',
         '- 精确物理状态、粗粒度搜索起点均不是独立配送任务数。',
         '- 新种子仍只有4个Train任务；本报告不是独立Validation或30秒在线改善。','']
    return '\n'.join(ls)
