"""R4-E: pure, testable controls for long-horizon exact C3 continuation and CF diagnostics.

All search returns are FIXED-K, not wall-clock 30s effects. No training labels
from previously inspected July validation are allowed for model fitting.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path

STAGE_VERIFY = 'R4E-LONG-EXACT-CONTINUATION-v1'
STAGE_CF = 'R4E-LONG-SECTION-CF-v1'
HORIZONS = (100, 200, 500)
BASE_R4B = 'R4B-OFFICIAL-EXACT-CONTINUATION-AUDIT-v1'


def require(x, msg):
    if not x: raise ValueError(msg)


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1 << 20), b''): d.update(b)
    return d.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temp.replace(p)


def read_lines(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


def horizons(s):
    ks=tuple(sorted(set(int(x.strip()) for x in s.split(',') if x.strip())))
    require(ks and all(x in HORIZONS for x in ks),'Long horizons must be nonempty subset of 100,200,500')
    return ks


def snapshot_id(s):
    """Identity for R4-A snapshots, R4-B proof rows, and R4-E rows.

    R4-B's immutable paired_rows.jsonl schema uses ``snapshot_stop_count``;
    R4-A and R4-E use ``stop_count``.  These are different source formats,
    not different time counters.  Reject absent or disagreeing counters.
    """
    found = [int(s[k]) for k in ('stop_count', 'snapshot_stop_count') if k in s]
    require(found and len(set(found)) == 1,
            'Missing or inconsistent stop_count/snapshot_stop_count')
    return (str(s['task_id']), str(s['scenario_id']), int(s['seed']), found[0])


def physical_id(s):
    return (str(s['task_id']),int(s['seed']),int(s['stop_count']),
            str(s['current_route_sha256']),str(s['best_route_sha256']),str(s['rng_state_sha256']))


def verify_r4b_evidence(directory, capture_dir):
    path=Path(directory)
    report=read(path/'report.json');manifest=read(path/'manifest.json')
    require(report.get('stage')==BASE_R4B and report.get('status')=='COMPLETE','R4B report not complete')
    require(report.get('n_comparisons')==64 and report.get('exact_matches')==64,'R4B 64/64 not verified')
    require(report.get('source_snapshot_sha256')==sha(Path(capture_dir)/'snapshots.jsonl'),'R4B capture SHA changed')
    require(manifest.get('source_capture_snapshot_sha256')==report['source_snapshot_sha256'],'R4B manifest/source mismatch')
    rows=read_lines(path/'paired_rows.jsonl');require(len(rows)==64,'R4B paired evidence incomplete')
    vals=defaultdict(set)
    for r in rows:
        require(r.get('exact') is True and r['verified_steps']==r['horizon'],'R4B step not verified')
        vals[snapshot_id(r)].add(int(r['horizon']))
    require(len(vals)==16 and all(v=={1,5,20,50} for v in vals.values()),'R4B only partially checked source checkpoints')
    return vals


def verified_long_evidence(directory, capture_dir, r4b_dir, expected_horizons):
    root=Path(directory);rep=read(root/'report.json');man=read(root/'manifest.json')
    require(rep.get('stage')==STAGE_VERIFY and rep.get('status')=='COMPLETE','R4E exact proof incomplete')
    require(man.get('stage')==STAGE_VERIFY and man.get('status')=='COMPLETE','R4E proof manifest incomplete')
    require(man.get('source_capture_sha256')==sha(Path(capture_dir)/'snapshots.jsonl'),'Capture mismatch')
    require(man.get('source_r4b_report_sha256')==sha(Path(r4b_dir)/'report.json'),'R4B proof changed')
    require(man.get('source_r4b_rows_sha256')==sha(Path(r4b_dir)/'paired_rows.jsonl'),'R4B rows changed')
    rows=read_lines(root/'paired_rows.jsonl')
    require(rep['n_comparisons']==rep['exact_matches']==len(rows),'Long proof row count mismatch')
    expected=set(expected_horizons);groups=defaultdict(set)
    for r in rows:
        require(r.get('exact') is True and r['verified_steps']==r['horizon'],'Non-exact long proof entry')
        require(int(r['horizon']) in HORIZONS,'Unexpected horizon in proof')
        groups[snapshot_id(r)].add(int(r['horizon']))
    require(groups and all(expected.issubset(v) for v in groups.values()),'Incomplete long proof K coverage')
    require(set(map(int,rep['horizons'])).issuperset(expected),'Long proof horizons inconsistent')
    return groups


def first_divergence(base,alt,field):
    """1-based number of completed iterations after which state hash differs.

    Snapshots/events store PRE-iteration states. State after the last iteration
    comes from end_signature, not an event at position K.
    """
    n=len(base['events'])
    require(n==len(alt['events']) and n>=1,'Unequal trace lengths')
    for j in range(1,n):
        if base['events'][j][field]!=alt['events'][j][field]:return j
    if base['end_signature'][field]!=alt['end_signature'][field]:return n
    return None


def diagnose(base,alt):
    require(base['arm']=='C3' and alt['arm'] not in ('C3','SHAM_C3'),'Incorrect paired arms')
    require(base['snapshot_spec']==alt['snapshot_spec'] and base['horizon']==alt['horizon'],'Not same source/K')
    from math import isclose
    eb=base['events'][0];ea=alt['events'][0]
    require(eb['rng_state_sha256']==ea['rng_state_sha256'] and eb['destroy_index']==ea['destroy_index']==3,
            'Unpaired RNG or wrong operator')
    require(eb['repair_index']==ea['repair_index'],'Repair policy mismatch')
    o=int(ea['outcome'])
    require(o in (0,1,2,3),'Unexpected official ALNS outcome index')
    r={
        'first_candidate_changed':eb['candidate_route_sha256']!=ea['candidate_route_sha256'],
        'first_accepted':o != 3,
        'first_outcome':o,
        'first_best_divergence_step':first_divergence(base,alt,'best_route_sha256'),
        'first_current_divergence_step':first_divergence(base,alt,'current_route_sha256'),
        'best_road_gain_sec':float(base['road_best_sec'])-float(alt['road_best_sec']),
        'current_road_gain_sec':float(base['road_current_sec'])-float(alt['road_current_sec']),
        'best_static_gain_sec':float(base['static_best_sec'])-float(alt['static_best_sec']),
        'current_static_gain_sec':float(base['static_current_sec'])-float(alt['static_current_sec']),
        'same_best_routes':base['best_route_sha256']==alt['best_route_sha256'],
        'same_current_routes':base['current_route_sha256']==alt['current_route_sha256'],
    }
    require((r['best_road_gain_sec']==0.0) if r['same_best_routes'] else True,
            'Same best route cannot have different Road.replay under paired scenario')
    require((r['current_road_gain_sec']==0.0) if r['same_current_routes'] else True,
            'Same current route cannot have different Road.replay under paired scenario')
    return r


def aggregate(records,ks):
    from statistics import mean
    results={}
    for k in ks:
        arr=[r for r in records if int(r['unit_spec']['horizon'])==k]
        if not arr:continue
        groups=defaultdict(list)
        for r in arr:groups[snapshot_id(r['unit_spec'])].append(r)
        phys=defaultdict(set)
        for sid,rs in groups.items():phys[tuple(rs[0]['physical_id'])].add(sid[1])
        gains=[r['best_road_gain_sec'] for r in arr]
        cap=[max([0.]+[r['best_road_gain_sec'] for r in rs]) for rs in groups.values()]
        results[str(k)]={
            'n_actions':len(arr),'n_scenario_snapshots':len(groups),
            'n_distinct_physical_search_states':len(phys),
            'first_accepted':sum(r['first_accepted'] for r in arr),
            'candidate_changed':sum(r['first_candidate_changed'] for r in arr),
            'terminal_best_changed':sum(not r['same_best_routes'] for r in arr),
            'terminal_current_changed':sum(not r['same_current_routes'] for r in arr),
            'positive_best':sum(x>1e-8 for x in gains),
            'negative_best':sum(x<-1e-8 for x in gains),
            'zero_best':sum(abs(x)<=1e-8 for x in gains),
            'mean_road_best_gain_sec':mean(gains),
            'oracle_positive_scenario_snapshots':sum(x>1e-8 for x in cap),
            'sampled_oracle_gain_mean_sec':mean(cap),
            'first_best_divergence_steps':sorted(r['first_best_divergence_step'] for r in arr if r['first_best_divergence_step'] is not None),
        }
    return results
