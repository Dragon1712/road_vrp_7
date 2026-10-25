"""Step6-C1-R4J TRAIN-only early-Section acquisition diagnostics.

Consumes existing frozen R4G, R4G-Prefix and R4H evidence; never edits these
algorithms, never fits a policy, and never claims fixed-K == 30s online gain.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib,json,math
from pathlib import Path

STAGE='R4J-TRAIN-EARLY-ACQUISITION-AUDIT-v1'
HORIZONS=(50,200,500)
CAP_STAGE='R4G-TRAIN-SECTION-SNAPSHOT-v1'
PREFIX_STAGE='R4G-TRAIN-PREFIX-EXACT-v1'
CF_STAGE='R4H-TRAIN-SECTION-COUNTERFACTUAL-v1'
EPS=1e-8


def ensure(ok,message):
    if not ok:raise ValueError(message)


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def readl(p):
    with Path(p).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,obj):
    p=Path(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    temp.replace(p)


def skey(x):
    if isinstance(x,(list,tuple)):
        ensure(len(x)==4,'Malformed snapshot identity')
        return str(x[0]),str(x[1]),int(x[2]),int(x[3])
    return str(x['task_id']),str(x['scenario_id']),int(x['seed']),int(x['stop_count'])


def ak(x):
    k=x['unit_spec']
    return (*skey(k),int(k['horizon']),str(k['arm']))


def physical(s):
    """Exact R4-H physical identity, no temperature rounding."""
    temp=float(s['anneal_temperature_before_accept'])
    ensure(math.isfinite(temp),'Nonfinite anneal temperature')
    def h(w):
        ensure(isinstance(w,list) and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in w),'Invalid weight list')
        return hashlib.sha256(json.dumps(w,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return (str(s['task_id']),int(s['seed']),int(s['stop_count']),
            str(s['current_route_sha256']),str(s['best_route_sha256']),str(s['rng_state_sha256']),
            h(s['roulette_destroy_weights']),h(s['roulette_repair_weights']),temp)


def verify_capture(capture,plain,audit,forbid_seeds=(60,61)):
    c,p,a=map(Path,(capture,plain,audit))
    cm,cr,cp=read(c/'manifest.json'),read(c/'report.json'),read(c/'provenance.json')
    pm,pr,pp=read(p/'manifest.json'),read(p/'report.json'),read(p/'provenance.json')
    ar=read(a/'report.json')
    ensure(cr.get('status')=='COMPLETE' and pr.get('status')=='COMPLETE','Capture/plain incomplete')
    ensure(cm.get('stage')==pm.get('stage')==CAP_STAGE,'Wrong collector stage')
    ensure(cm.get('split')==pm.get('split')=='train' and cm.get('arm')=='capture' and pm.get('arm')=='plain','Non-TRAIN source')
    ensure(set(cm['task_ids'])<=set(f'task_{i:04d}' for i in range(4)),'Task whitelist failed')
    ensure(cm['task_ids']==pm['task_ids'] and cm['scenario_ids']==pm['scenario_ids'] and cm['seeds']==pm['seeds'],'Capture/plain grid differs')
    ensure(set(cm['seeds']).isdisjoint(set(forbid_seeds)),'New acquisition reused original seed; use new seeds')
    ensure(cm['max_iterations']==pm['max_iterations']==20000 and cm['search_wall_sec']==pm['search_wall_sec']==30,'20k/30s capture required')
    ensure(cm['section_only'] is True,'Not real Section checkpoints')
    ensure(all(str(x).startswith('2025-07-') for x in cm['scenario_ids']),'Non-July scenario')
    ensure(ar.get('stage')=='R4G-TRAIN-NONINTERFERENCE-AUDIT-v1' and ar.get('status')=='COMPLETE','R4G audit missing')
    ensure(ar['pairs']==len(cm['task_ids'])*len(cm['scenario_ids'])*len(cm['seeds']) and ar['route_road_equal']==ar['pairs'],'Final C3 route/Road mismatch; abort')
    ensure(ar['failure_reason_counts'].get('FINAL_ROUTE_MISMATCH',0)==0 and ar['failure_reason_counts'].get('ROAD_COST_MISMATCH',0)==0,'Uncontrolled terminal divergence')
    for folder,prov in ((c,cp),(p,pp)):
        for file,key in (('manifest.json','manifest_sha256'),('metrics.jsonl','metrics_sha256'),('snapshots.jsonl','snapshots_sha256')):
            ensure(digest(folder/file)==prov[key],'Collector source SHA differs '+str(folder/file))
    snaps=readl(c/'snapshots.jsonl')
    ensure(len(snaps)==cr['snapshots'] and len(snaps)>0,'Capture snapshot count mismatch')
    ensure(all(s['boundary']=='BEFORE_SELECT_AND_DESTROY' and s['snapshot_is_actual_section_decision'] and int(s['real_selected_destroy_index'])==3 for s in snaps),'Non-Section snapshot')
    ensure(len({skey(s) for s in snaps})==len(snaps),'Duplicate snapshot')
    ps=readl(p/'metrics.jsonl');cs=readl(c/'metrics.jsonl')
    ensure(len(ps)==len(cs)==ar['pairs'],'Metric record count differs')
    return {'manifest':cm,'snapshots':snaps,'audit':ar,
            'pins':{'capture_manifest':digest(c/'manifest.json'),'capture_snapshots':digest(c/'snapshots.jsonl'),
                    'capture_metrics':digest(c/'metrics.jsonl'),'plain_manifest':digest(p/'manifest.json'),
                    'audit_report':digest(a/'report.json')}}


def verify_prefix(prefix,capture_bundle,require_complete=True):
    p=Path(prefix)
    pr,pm=read(p/'report.json'),read(p/'manifest.json')
    ensure(pr.get('stage')==pm.get('stage')==PREFIX_STAGE and pr.get('status')=='COMPLETE' and pr.get('data_role')==pm.get('data_role')=='TRAIN_ONLY','Incomplete / non-TRAIN prefix proof')
    ensure(pr['exact_matches']==pr['n_comparisons'] and pr['n_comparisons']>0,'Not all reference events exact')
    ensure(pr['original_capture_snapshots_sha256']==capture_bundle['pins']['capture_snapshots'],'Proof source SHA mismatch')
    ensure(pm['source_sha256']['capture_snapshots']==capture_bundle['pins']['capture_snapshots'],'Proof manifest SHA mismatch')
    expect=len(capture_bundle['manifest']['task_ids'])*len(capture_bundle['manifest']['scenario_ids'])*len(capture_bundle['manifest']['seeds'])
    if require_complete:ensure(pr['n_cases']==expect,'All cases not proved; run prefix pilot')
    rows=readl(p/'paired_rows.jsonl')
    ensure(len(rows)==pr['n_comparisons'] and all(x['exact'] is True and int(x['verified_steps'])==int(x['horizon']) for x in rows),'Incomplete per-event evidence')
    proof={(str(x['task_id']),str(x['scenario_id']),int(x['seed']),int(x['snapshot_stop_count']),int(x['horizon'])) for x in rows}
    ensure(len(proof)==len(rows),'Duplicate proof horizon')
    ensure({x[4] for x in proof}<=set((1,50,200,500)) and {50,200,500}<={x[4] for x in proof},'Wrong horizons')
    return {'report':pr,'rows':rows,'proof_keys':proof,
           'pins':{'prefix_report':digest(p/'report.json'),'prefix_manifest':digest(p/'manifest.json'),
                   'prefix_rows':digest(p/'paired_rows.jsonl'),'prefix_excluded':digest(p/'excluded_windows.json')}}


def verify_cf(cf,bundle,prefix):
    p=Path(cf);report=read(p/'report.json');manifest=read(p/'manifest.json')
    ensure(report.get('status')=='COMPLETE' and manifest.get('stage')==report.get('stage')==CF_STAGE,'CF result incomplete')
    ensure(report.get('data_role')==manifest.get('data_role')=='TRAIN_ONLY','CF is not Train')
    ensure(manifest['horizons']==[50,200,500] and manifest['require_all_k'] is True,'Expected three K')
    ensure(manifest['source_sha256']['capture_snapshots']==bundle['pins']['capture_snapshots'] and
           manifest['source_sha256']['prefix_report']==prefix['pins']['prefix_report'] and
           manifest['source_sha256']['prefix_paired_rows']==prefix['pins']['prefix_rows'],'CF not based on proof')
    labels=readl(p/'labels.jsonl');branches=readl(p/'branch_summary.jsonl');plans=readl(p/'plans.jsonl')
    ensure(len(labels)==report['n_labels'] and len(branches)==report['n_branches'] and len(plans)==report['n_scenario_snapshots'],'CF row count mismatch')
    require_snaps={skey(x) for x in bundle['snapshots']}
    allowed={tuple(x) for x in manifest['selected_snapshot_ids']}
    ensure(all(k in require_snaps for k in allowed) and len(allowed)==len(plans),'CF selected snapshot mismatch')
    ensure(len(set(ak(x) for x in branches))==len(branches),'Duplicate CF branch')
    bm={ak(x):x for x in branches}
    action_keys=set()
    for x in labels:
        k=ak(x);sid=k[:4];h=k[4]
        ensure(sid in allowed and (*sid,h) in prefix['proof_keys'],'Unproved CF window')
        ensure(k not in action_keys and k[5] not in ('C3','SHAM_C3'),'Duplicate / control in actions')
        action_keys.add(k)
        base=bm.get((*sid,h,'C3'));sham=bm.get((*sid,h,'SHAM_C3'));alt=bm.get(k)
        ensure(base is not None and sham is not None and alt is not None,'Missing C3/SHAM/ALT')
        ensure(base['best_route_sha256']==sham['best_route_sha256'] and base['best_road_sec']==sham['best_road_sec'],'SHAM != C3')
        ensure(math.isclose(float(x['road_best_gain_sec']),float(base['best_road_sec'])-float(alt['best_road_sec']),abs_tol=1e-8,rel_tol=0),'Wrong gain or branch match')
    ensure(report['n_sham_checks']==3*len(plans),'Not all SHAM checked')
    ensure(len(labels)%3==0,'Actions missing horizons')
    by=defaultdict(set)
    for k in action_keys:by[k[:4]+(k[5],)].add(k[4])
    ensure(all(v==set(HORIZONS) for v in by.values()),'Candidate labels missing K')
    return {'report':report,'manifest':manifest,'labels':labels,'branches':branches,'plans':plans,
            'pins':{'cf_report':digest(p/'report.json'),'cf_manifest':digest(p/'manifest.json'),
                    'cf_labels':digest(p/'labels.jsonl'),'cf_branches':digest(p/'branch_summary.jsonl')}}


def label_stats(labels,physical_map,cutoff=2000):
    stats={};bytask={};active_groups={}
    for horizon in HORIZONS:
        items=[x for x in labels if x['unit_spec']['horizon']==horizon]
        def c(a):return {'actions':len(a),'positive':sum(x['road_best_gain_sec']>EPS for x in a),
            'negative':sum(x['road_best_gain_sec']< -EPS for x in a),
            'zero':sum(abs(x['road_best_gain_sec'])<=EPS for x in a),
            'mean_gain_sec':sum(float(x['road_best_gain_sec']) for x in a)/len(a) if a else 0.,
            'positive_physical':len({physical_map[skey(x['unit_spec'])] for x in a if x['road_best_gain_sec']>EPS})}
        stats[str(horizon)]={'all':c(items),'early':c([x for x in items if x['unit_spec']['stop_count']<=cutoff]),
                              'later':c([x for x in items if x['unit_spec']['stop_count']>cutoff])}
        bytask[str(horizon)]={tid:c([x for x in items if x['unit_spec']['task_id']==tid]) for tid in sorted({x['unit_spec']['task_id'] for x in items})}
    return stats,bytask


def analyze(cf,bundle,old_cf=None):
    sm={skey(s):s for s in bundle['snapshots']}
    selected={skey(x):sm[skey(x)] for x in cf['manifest']['selected_snapshot_ids']}
    pmap={k:physical(s) for k,s in selected.items()}
    stats,bytask=label_stats(cf['labels'],pmap)
    report={'stage':STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY','cutoff_iteration':2000,
            'n_source_snapshots':len(sm),'n_cf_snapshots':len(selected),
            'n_exact_physical_states':len(set(pmap.values())),
            'n_coarse_independent_search_starts':len({(s['task_id'],s['seed'],s['stop_count']) for s in selected.values()}),
            'source_strict_pairs':bundle['audit']['strict_pass'],
            'source_all_pairs':bundle['audit']['pairs'],
            'source_route_road_equal':bundle['audit']['route_road_equal'],
            'per_horizon':stats,'per_task':bytask,
            'limitations':['Train-only new-seed exploration, not new tasks/independent validation.',
               'Source complete-20k noninterference may fail when 30s wall is reached; precise K proof remains mandatory.',
               'Road.replay outcomes are ex-post fixed-K; no remaining 30sec deployment guarantee.',
               'Same task and near-identical states across scenarios are correlated.',
               'No Validation/Test fitting, no task selection based on heldout outcome.']}
    if old_cf is not None:
        old_meta=read(Path(old_cf)/'report.json')
        old_man=read(Path(old_cf)/'manifest.json')
        ensure(old_meta.get('stage')==old_man.get('stage')==CF_STAGE and old_meta.get('status')=='COMPLETE',
               'Previous reference CF report not COMPLETE')
        ensure(old_meta.get('data_role')==old_man.get('data_role')=='TRAIN_ONLY',
               'Previous comparison is not TRAIN data')
        old=readl(Path(old_cf)/'labels.jsonl')
        ensure(len(old)==old_meta.get('n_labels'),'Previous CF label count mismatch')
        old_st=set(skey(x['unit_spec']) for x in old)
        ensure(not (old_st & set(selected)),'New/old snapshot overlap')
        report['old_vs_new']={'old_labels':len(old),'new_labels':len(cf['labels']),
             'old_positive_k500':sum(x['unit_spec']['horizon']==500 and x['road_best_gain_sec']>EPS for x in old),
             'new_positive_k500':stats['500']['all']['positive'],
             'old_positive_rate_k500':sum(x['unit_spec']['horizon']==500 and x['road_best_gain_sec']>EPS for x in old)/max(1,sum(x['unit_spec']['horizon']==500 for x in old)),
             'new_positive_rate_k500':stats['500']['all']['positive']/max(1,stats['500']['all']['actions'])}
    return report


def markdown(r):
    lines=['# R4-J TRAIN-only早期Section监督信号扩容审计','',
        f"状态：**{r['status']}**；来源快照：{r['n_source_snapshots']}；反事实快照：{r['n_cf_snapshots']}；精确物理状态：{r['n_exact_physical_states']}。",
        f"源30秒固定20k强一致：{r['source_strict_pairs']}/{r['source_all_pairs']}；源终态路线+Road相同：{r['source_route_road_equal']}/{r['source_all_pairs']}。",'',
        '| K | 子集 | 候选数 | 正/负/零 | 平均Road收益(s) | 正收益精确状态 |','|---:|---|---:|---:|---:|---:|']
    for k in ('50','200','500'):
        for phase in ('all','early','later'):
            x=r['per_horizon'][k][phase]
            lines.append(f"| {k} | {phase} | {x['actions']} | {x['positive']}/{x['negative']}/{x['zero']} | {x['mean_gain_sec']:+.3f} | {x['positive_physical']} |")
    if 'old_vs_new' in r:
        v=r['old_vs_new']
        lines+=['','原R4-H与新增R4-J都只是Train内开发标签，不能以正例率提升替代独立验证。',
                f"旧/新K500正标签：{v['old_positive_k500']} / {v['new_positive_k500']}；旧/新正例比例：{v['old_positive_rate_k500']:.3%} / {v['new_positive_rate_k500']:.3%}。"]
    lines+=['','**限制：** 新种子与新早期状态仍属于原4个Train任务；固定K的反事实收益并非30秒在线ALNS收益。','']
    return '\n'.join(lines)
