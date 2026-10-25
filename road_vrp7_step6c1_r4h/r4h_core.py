"""R4-H: provable TRAIN-only fixed-K Section counterfactual experiment.

Pure audit/selection functions; no ALNS, no road simulation, no model fit.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

STAGE='R4H-TRAIN-SECTION-COUNTERFACTUAL-v1'
PREFIX_STAGE='R4G-TRAIN-PREFIX-EXACT-v1'
CAPTURE_STAGE='R4G-TRAIN-SECTION-SNAPSHOT-v1'
HORIZONS=(50,200,500)

def require(flag, message):
    if not flag: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda: f.read(1<<20),b''):h.update(buf)
    return h.hexdigest()

def load(path):return json.loads(Path(path).read_text(encoding='utf8'))

def jsonl(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(s) for s in f if s.strip()]

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    t.replace(path)

def snap_id(x):
    c=x.get('stop_count',x.get('snapshot_stop_count'))
    require(c is not None,'Missing snapshot stop_count')
    if 'stop_count' in x and 'snapshot_stop_count' in x:
        require(int(x['stop_count'])==int(x['snapshot_stop_count']),'Conflicting snapshot counters')
    return (str(x['task_id']),str(x['scenario_id']),int(x['seed']),int(c))

def arm_id(x):
    z=x['unit_spec']
    return (*snap_id(z),int(z['horizon']),str(z['arm']))

def parse_ks(s):
    ks=sorted({int(v.strip()) for v in s.split(',') if v.strip()})
    require(ks and set(ks)<=set(HORIZONS),'H must be subset of (50,200,500)')
    return ks

def verified_allowlist(prefix_dir,capture_dir,requested):
    """Check the actual R4-G prefix evidence, not only report aggregate counts."""
    prefix_dir=Path(prefix_dir);capture_dir=Path(capture_dir)
    rep=load(prefix_dir/'report.json');man=load(prefix_dir/'manifest.json')
    cap=load(capture_dir/'manifest.json');cp=load(capture_dir/'provenance.json')
    require(rep.get('stage')==PREFIX_STAGE and rep.get('status')=='COMPLETE' and rep.get('data_role')=='TRAIN_ONLY',
            'Prefix audit incomplete or wrong data role')
    # R4G-Prefix completed studies retain CONFIGURED manifest status after publishing
    # a COMPLETE report. This is its *actual* immutable output schema.
    require(man.get('stage')==PREFIX_STAGE and man.get('status') in ('CONFIGURED','COMPLETE') and
            man.get('data_role')=='TRAIN_ONLY', 'Prefix manifest incompatible')
    require(cap.get('stage')==CAPTURE_STAGE and cap.get('split')=='train' and cap.get('arm')=='capture',
            'Capture not strictly TRAIN')
    require(man['source_sha256']['capture_snapshots']==sha(capture_dir/'snapshots.jsonl'),
            'Prefix proof uses a different capture')
    require(rep['original_capture_snapshots_sha256']==sha(capture_dir/'snapshots.jsonl'),
            'Prefix report uses a different capture')
    for field,name in [('capture_manifest','manifest.json'),('capture_provenance','provenance.json'),('capture_metrics','metrics.jsonl')]:
        require(man['source_sha256'][field]==sha(capture_dir/name),'Prefix/capture SHA mismatch: '+name)
    for name,field in [('manifest.json','manifest_sha256'),('metrics.jsonl','metrics_sha256'),('snapshots.jsonl','snapshots_sha256')]:
        require(cp[field]==sha(capture_dir/name),'Capture provenance mismatch: '+name)
    require(cap['max_iterations']==man['source_max_iterations']==20000 and
            cap['search_wall_sec']==man['source_wall_sec']==30,'TRAIN source budget changed')
    require(len(set(cap['task_ids']))==len(cap['task_ids']) and
            set(cap['task_ids'])<=set(f'task_{i:04d}' for i in range(4)), 'Not authorized TRAIN tasks')
    require(rep['n_cases']==len(man['case_plan']) and rep['exact_matches']==rep['n_comparisons'],
            'Prefix proof not full exact')
    rows=jsonl(prefix_dir/'paired_rows.jsonl')
    require(len(rows)==rep['n_comparisons'] and len(rows)>0,'Prefix rows incomplete')
    allowed={}
    for r in rows:
        key=(*snap_id(r),int(r['horizon']))
        require(key not in allowed,'Repeated proof row')
        require(r.get('exact') is True and int(r['verified_steps'])==key[-1],
                'Unverified proof row')
        allowed[key]=r
    manifest_set=set()
    for tid,sid,seed,windows in man['case_plan']:
        require(tid in cap['task_ids'] and sid in cap['scenario_ids'] and seed in cap['seeds'],
                'Proof contains non-TRAIN case')
        for stop,ks in windows:
            for k in ks:
                key=(str(tid),str(sid),int(seed),int(stop),int(k))
                require(key not in manifest_set,'Duplicate proof plan entry')
                manifest_set.add(key)
    require(set(allowed)==manifest_set,'Proof paired_rows differs from exact manifest selection')
    require(len(allowed)==sum(int(v['exact']) for v in rep['per_horizon'].values()),
            'Proof report counts do not agree with rows')
    excluded=load(prefix_dir/'excluded_windows.json')
    excluded_keys={(*snap_id(r),int(r['horizon'])) for r in excluded}
    require(len(excluded_keys)==len(excluded),'Duplicate excluded K window')
    require(not (excluded_keys&set(allowed)),'Excluded window was verified')
    require(rep['excluded_horizon_count']==len(excluded),'Excluded count mismatch')
    snapshots=jsonl(capture_dir/'snapshots.jsonl')
    snapmap={}
    for s in snapshots:
        key=snap_id(s)
        require(key not in snapmap,'Duplicate source snapshot')
        snapmap[key]=s
    require(len(snapmap)==rep['eligible_snapshots'],'Wrong source snapshot count')
    require(all(key[:4] in snapmap for key in allowed),'Proof references absent snapshot')
    for s in snapshots:
        require(s['scenario_id'] in cap['scenario_ids'] and s['task_id'] in cap['task_ids'],
                'Capture has out-of-role snapshot')
    return allowed,snapmap,rep,man,cap

def case_round_robin(keys):
    """Interleave tasks so a small pilot does not consist of only task_0000."""
    by=defaultdict(list)
    for k in sorted(keys):by[k[0]].append(k)
    ordered=[]
    while by:
        for task in sorted(list(by)):
            ordered.append(by[task].pop(0))
            if not by[task]:del by[task]
    return ordered

def select_train_snapshots(verified,snapmap,ks,max_cases,max_snaps,require_all=True):
    require(max_cases>=0 and max_snaps>=0,'Invalid study size')
    viable={}
    for key,s in snapmap.items():
        present=[k for k in ks if (*key,k) in verified]
        if (set(present)==set(ks) if require_all else bool(present)):
            viable[key]=present
    bycase=defaultdict(list)
    for key in sorted(viable):bycase[key[:3]].append(key)
    cases=case_round_robin(bycase)
    if max_cases:cases=cases[:max_cases]
    chosen=[]
    for case in cases:
        candidates=bycase[case]
        if max_snaps and len(candidates)>max_snaps:
            # Balanced early/mid/late selections for a small pilot.
            if max_snaps==1: idx=[(len(candidates)-1)//2]
            else:idx=[round(i*(len(candidates)-1)/(max_snaps-1)) for i in range(max_snaps)]
            require(len(set(idx))==len(idx),'Selection collision')
            candidates=[candidates[i] for i in idx]
        chosen.extend((key,snapmap[key],viable[key]) for key in candidates)
    require(chosen,'No TRAIN snapshots covered by exact proof for requested K')
    return chosen

def paired_reward(base,arm):
    require(base['arm']=='C3' and arm['arm'] not in ('C3','SHAM_C3'),'Not a C3/alternative pair')
    require(base['snapshot_spec']==arm['snapshot_spec'] and base['horizon']==arm['horizon'],'Different paired state/horizon')
    return {'road_best_gain_sec':float(base['road_best_sec']-arm['road_best_sec']),
            'road_current_gain_sec':float(base['road_current_sec']-arm['road_current_sec']),
            'static_best_gain_sec':float(base['static_best_sec']-arm['static_best_sec']),
            'static_current_gain_sec':float(base['static_current_sec']-arm['static_current_sec']),
            'same_best_routes':base['best_route_sha256']==arm['best_route_sha256'],
            'same_current_routes':base['current_route_sha256']==arm['current_route_sha256']}

def report_metrics(labels,snapshots,ks):
    by=defaultdict(list)
    for z in labels:by[int(z['unit_spec']['horizon'])].append(z)
    physical={}
    for key,s,_ in snapshots:
        from r4a_core import canonical_hash
        phys=(s['task_id'],int(s['seed']),int(s['stop_count']),s['current_route_sha256'],s['best_route_sha256'],
              s['rng_state_sha256'],canonical_hash(s['roulette_destroy_weights']),
              canonical_hash(s['roulette_repair_weights']),float(s['anneal_temperature_before_accept']))
        physical[key]=phys
    metrics={}
    for k in ks:
        rs=by[k]
        groups=defaultdict(list)
        for r in rs:groups[snap_id(r['unit_spec'])].append(r)
        gains=[float(r['road_best_gain_sec']) for r in rs]
        oracles=[max([0.]+[float(q['road_best_gain_sec']) for q in gg]) for gg in groups.values()]
        metrics[str(k)]={'n_actions':len(rs),'n_scenario_snapshots':len(groups),
                         'positive':sum(g>1e-8 for g in gains),'negative':sum(g<-1e-8 for g in gains),
                         'zero':sum(abs(g)<=1e-8 for g in gains),
                         'mean_best_road_gain_sec':sum(gains)/len(gains) if gains else 0.,
                         'positive_oracle_scenario_snapshots':sum(x>1e-8 for x in oracles),
                         'positive_oracle_distinct_physical_states':len({physical[key] for key,g in groups.items()
                                  if max([0.]+[float(q['road_best_gain_sec']) for q in g])>1e-8})}
    return {'n_scenario_snapshots':len(snapshots),
            'n_distinct_physical_states':len(set(physical.values())),'per_horizon':metrics}

def horizon_stability(labels, branches, snapshots, ks):
    """Track same sampled Section arm at multiple K, with C3 catch-up decomposition.

    Descriptive TRAIN mechanism audit only. Static-best/Road-best is an ex-post
    route evaluation and may not be monotone in the frozen static objective.
    """
    def sg(x):return '+' if x>1e-8 else '-' if x < -1e-8 else '0'
    labelmap={}
    for z in labels:
        key=arm_id(z)
        require(key not in labelmap,'Duplicate action label')
        labelmap[key]=z
    branchmap={}
    for z in branches:
        key=arm_id(z)
        require(key not in branchmap,'Duplicate branch')
        branchmap[key]=z
    byarm=defaultdict(dict)
    for key,z in labelmap.items():
        byarm[(*key[:4],key[5])][key[4]]=z
    phys={}
    from r4a_core import canonical_hash
    for key,s,_ in snapshots:
        phys[key]=(s['task_id'],int(s['seed']),int(s['stop_count']),s['current_route_sha256'],
              s['best_route_sha256'],s['rng_state_sha256'],canonical_hash(s['roulette_destroy_weights']),
              canonical_hash(s['roulette_repair_weights']),float(s['anneal_temperature_before_accept']))
    results={}
    for lo,hi in zip(ks,ks[1:]):
        transitions=Counter();reversal_states=set();src_prog=[];alt_prog=[]
        for (tid,sid,seed,stop,arm),vals in byarm.items():
            require(set(vals)==set(ks),'Action was not evaluated at every K')
            a,b=vals[lo],vals[hi]
            prev=float(a['road_best_gain_sec']);later=float(b['road_best_gain_sec'])
            status=f'{sg(prev)}->{sg(later)}'
            transitions[status]+=1
            base0=branchmap[(tid,sid,seed,stop,lo,'C3')]['best_road_sec']
            base1=branchmap[(tid,sid,seed,stop,hi,'C3')]['best_road_sec']
            arm0=branchmap[(tid,sid,seed,stop,lo,arm)]['best_road_sec']
            arm1=branchmap[(tid,sid,seed,stop,hi,arm)]['best_road_sec']
            cp=float(base0-base1);ap=float(arm0-arm1)
            require(abs((later-prev)-(ap-cp))<1e-7,
                    'Incorrect longitudinal gain decomposition')
            if status=='+->-':
                reversal_states.add(phys[(tid,sid,seed,stop)])
                src_prog.append(cp);alt_prog.append(ap)
        results[f'{lo}_to_{hi}']={'sign_counts':dict(sorted(transitions.items())),
          'positive_to_negative':transitions.get('+->-',0),
          'positive_to_negative_physical_states':len(reversal_states),
          'mean_C3_progress_for_flips_sec':sum(src_prog)/len(src_prog) if src_prog else 0.,
          'mean_alt_progress_for_flips_sec':sum(alt_prog)/len(alt_prog) if alt_prog else 0.}
    return results
