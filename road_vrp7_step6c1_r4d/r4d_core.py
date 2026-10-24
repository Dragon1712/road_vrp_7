"""R4-D: immutable R4-C fixed-K counterfactual funnel / label audit.

No ALNS, Road.replay or model training is performed here. Every value is
computed from source-pinned R4-C branch units; evaluation stays development.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

STAGE = 'R4D-CONTINUOUS-CF-SIGNAL-AUDIT-v1'


def need(cond, msg):
    if not cond:
        raise ValueError(msg)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump_json(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')


def spec_key(z):
    return (str(z['task_id']), str(z['scenario_id']), int(z['seed']),
            int(z['stop_count']), int(z['horizon']), str(z['arm']))


def snap_key(z):
    return (str(z['task_id']), str(z['scenario_id']), int(z['seed']), int(z['stop_count']))


def spec_from_tuple(z):
    return dict(zip(('task_id', 'scenario_id', 'seed', 'stop_count', 'horizon', 'arm'), z))


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.


def _fraction(n, d):
    return float(n / d) if d else 0.


def approx(a, b, tol=1e-8):
    return math.isclose(float(a), float(b), abs_tol=tol, rel_tol=0.)


def validate_inputs(folder, *, source_dir=None, capture_dir=None, r4b_dir=None):
    """Fail closed on mixed R4-C results. Optional upstream file checks are strict."""
    folder = Path(folder)
    manifest, report = read_json(folder/'manifest.json'), read_json(folder/'report.json')
    need(manifest.get('stage') == 'R4C-CONTINUOUS-SECTION-COUNTERFACTUAL-v1', 'Not R4-C collection')
    need(report.get('stage') == manifest['stage'] and report.get('status') == 'COMPLETE', 'Incomplete R4-C run')
    need(manifest.get('evaluation') == 'FIXED_ITERATION_K_NOT_WALL_CLOCK_30SEC', 'Unexpected experiment semantics')
    need(manifest.get('data_role') == 'INSPECTED_JULY_VALIDATION_DEVELOPMENT_ONLY', 'Unexpected data role')
    need(manifest.get('horizons') == [1, 5, 20, 50], 'Require same K windows as published R4-C')
    need(manifest.get('max_alternatives') == 8, 'Require frozen 8-action candidate design')
    need('counterfactual_r4c.py' in manifest.get('code_sha256',{}) and
         'r4c_core.py' in manifest.get('code_sha256',{}), 'Missing R4-C source identity')
    if source_dir is not None:
        src=Path(source_dir)
        for name, digest in manifest['code_sha256'].items():
            need(name in {'counterfactual_r4c.py','r4c_core.py'}, 'Unexpected R4-C source')
            need(sha256(src/name) == digest, 'Frozen R4-C code SHA mismatch: '+name)
    if capture_dir is not None:
        need(sha256(Path(capture_dir)/'snapshots.jsonl') == manifest['r4a_snapshots_sha256'],
             'R4-A snapshot content changed')
    if r4b_dir is not None:
        p=Path(r4b_dir)
        for name,key in [('report.json','r4b_report_sha256'),('manifest.json','r4b_manifest_sha256'),
                         ('paired_rows.jsonl','r4b_paired_rows_sha256')]:
            need(sha256(p/name)==manifest[key],'R4-B verified audit changed: '+name)
    labels, branches, plans = (read_jsonl(folder/name) for name in
       ('counterfactual_labels.jsonl','branch_summary.jsonl','plans.jsonl'))
    need(len(labels)==report['n_counterfactual_labels'], 'Label row count mismatch')
    need(len(branches)==report['n_branch_results'], 'Branch row count mismatch')
    need(len(plans)==report['n_snapshot_start_points'], 'Snapshot plan count mismatch')
    need(len({spec_key(x['unit_spec']) for x in labels})==len(labels), 'Duplicate candidate label')
    need(len({spec_key(x['unit_spec']) for x in branches})==len(branches), 'Duplicate branch result')
    need(len({snap_key(x) for x in plans})==len(plans), 'Duplicate snapshot plan')
    need(len([x for x in branches if x['unit_spec']['arm']=='SHAM_C3'])==report['n_sham_control_checks'],
         'SHAM row count mismatch')
    return manifest,report,labels,branches,plans


def analyze_records(label_rows, branch_rows, unit_results, *, horizons=(1,5,20,50)):
    """Pure analysis; `unit_results` maps the exact six-field key to R4-C result."""
    from r4c_core import compare_sham, normalize_reward
    labelmap={spec_key(x['unit_spec']):x for x in label_rows}
    branches={spec_key(x['unit_spec']):x for x in branch_rows}
    need(len(labelmap)==len(label_rows) and len(branches)==len(branch_rows),'Repeated keys')
    need(set(branches)==set(unit_results),'Missing or extra full event units')
    by_start=defaultdict(dict)
    for key,result in unit_results.items():
        tid,sid,seed,count,k,arm=key
        need(int(k) in horizons,'Unexpected K')
        need(result['arm']==arm and result['horizon']==k,'Branch identity mismatch')
        need(snap_key(result['snapshot_spec'])==(tid,sid,seed,count), 'Checkpoint start mismatch')
        events=result['events'];need(len(events)==k,'Incomplete event series')
        for j,e in enumerate(events):
            need(e['stop_count']==count+j,'Nonconsecutive ALNS iterations')
        need(events[0]['destroy_index']==3, 'First operator is not true Section')
        need(events[0]['repair_index']==result['first_frozen_operator_pair'][1], 'First Repair inconsistent')
        need(result['first_frozen_operator_pair'][0]==3,'First operator pair not Section')
        need(result['best_route_sha256']==result['end_signature']['best_route_sha256'], 'Terminal best SHA mismatch')
        need(result['current_route_sha256']==result['end_signature']['current_route_sha256'], 'Terminal current SHA mismatch')
        b=branches[key]
        need(b['first_action']==result['first_action_id'] and b['sha']==result['best_route_sha256'],
             'Branch summary differs from unit')
        need(approx(b['best_static'],result['static_best_sec']) and
             approx(b['best_road'],result['road_best_sec']),'Branch costs differ from unit')
        by_start[(tid,sid,seed,count,k)][arm]=result
    for start,items in by_start.items():
        need('C3' in items and 'SHAM_C3' in items,'Missing C3/SHAM baseline')
        compare_sham(items['C3'],items['SHAM_C3'])
        for arm,v in items.items():
            if arm in ('C3','SHAM_C3'):continue
            key=(*start,arm)
            need(key in labelmap,'Missing action label')
            r=normalize_reward(items['C3'],v)
            original=labelmap[key]
            need(original['first_action']==v['first_action_id'] and
                 original['c3_action']==v['first_original_c3_action_id'],'First Section identity changed')
            for name,val in r.items():
                if isinstance(val,bool): need(original[name] is val, 'Changed truth label '+name)
                else:need(approx(original[name],val),'Changed reward label '+name)
    # Every longer-K arm trajectory must have the identical prefix to shorter K.
    by_arm=defaultdict(list)
    for key,result in unit_results.items():
        by_arm[(key[0],key[1],key[2],key[3],key[5])].append((key[4],result['events']))
    for group,seq in by_arm.items():
        seq.sort(key=lambda x:x[0]);need([k for k,_ in seq]==list(horizons),'Missing K for '+str(group))
        for (smaller,a),(larger,b) in zip(seq,seq[1:]):
            need(a==b[:smaller],f'Cross-K event prefix mismatch at {group}: {smaller}->{larger}')
    rows=[]
    for start,items in sorted(by_start.items()):
        base=items['C3'];first=base['events'][0];tid,sid,seed,count,k=start
        for arm,result in sorted(items.items()):
            if arm in ('C3','SHAM_C3'):continue
            e=result['events'][0]
            lbl=labelmap[(*start,arm)]
            first_div=next((i+1 for i,(x,y) in enumerate(zip(base['events'],result['events'])) if x!=y),None)
            rows.append({
                'task_id':tid,'scenario_id':sid,'seed':seed,'stop_count':count,'horizon':k,'arm':arm,
                'first_outcome':int(e['outcome']),'c3_first_outcome':int(first['outcome']),
                'first_accepted':int(e['outcome'])!=3,'c3_first_accepted':int(first['outcome'])!=3,
                'first_candidate_nontrivial':e['candidate_route_sha256']!=e['current_route_sha256'],
                'first_candidate_diff_from_c3':e['candidate_route_sha256']!=first['candidate_route_sha256'],
                'first_event_diff_from_c3':e!=first,
                'first_divergence_step':first_div,
                'same_best_routes':lbl['same_best_routes'],'same_current_routes':lbl['same_current_routes'],
                'road_best_gain_sec':float(lbl['road_best_gain_sec']),
                'road_current_gain_sec':float(lbl['road_current_gain_sec']),
                'static_best_gain_sec':float(lbl['static_best_gain_sec']),
                'static_current_gain_sec':float(lbl['static_current_gain_sec']),
                'base_static_best_sec':float(base['static_best_sec']),
                'base_static_current_sec':float(base['static_current_sec']),
                'base_road_best_sec':float(base['road_best_sec']),
                'base_road_current_sec':float(base['road_current_sec']),
            })
    need(len(rows)==len(label_rows),'Missing candidate records')
    return rows,by_start


def summarize(rows, snapshots, horizons=(1,5,20,50)):
    per={};snaprows=[]
    for h in horizons:
        arr=[x for x in rows if x['horizon']==h]
        need(arr,'Missing per-K labels: '+str(h))
        state=defaultdict(list)
        for x in arr: state[(x['task_id'],x['scenario_id'],x['seed'],x['stop_count'])].append(x)
        for key,cands in state.items():
            best=max([0.]+[x['road_best_gain_sec'] for x in cands]);current=max([0.]+[x['road_current_gain_sec'] for x in cands])
            snaprows.append({'horizon':h,'task_id':key[0],'scenario_id':key[1],
                             'seed':key[2],'stop_count':key[3],'n_actions':len(cands),
                             'oracle_positive_best':best>1e-8,'oracle_best_gain_sec':best,
                             'oracle_current_gain_sec':current,'any_terminal_current_change':any(not x['same_current_routes'] for x in cands),
                             'any_terminal_best_change':any(not x['same_best_routes'] for x in cands)})
        best=[x['road_best_gain_sec'] for x in arr];current=[x['road_current_gain_sec'] for x in arr]
        events=[x['first_divergence_step'] for x in arr if x['first_divergence_step'] is not None]
        group_s=[x for x in snaprows if x['horizon']==h]
        per[str(h)]={
            'n_actions':len(arr),'n_snapshots':len(state),
            'first_candidate_diff_from_c3':sum(x['first_candidate_diff_from_c3'] for x in arr),
            'first_candidate_nontrivial':sum(x['first_candidate_nontrivial'] for x in arr),
            'first_accepted':sum(x['first_accepted'] for x in arr),
            'first_event_diff_from_c3':sum(x['first_event_diff_from_c3'] for x in arr),
            'terminal_current_changed':sum(not x['same_current_routes'] for x in arr),
            'terminal_best_changed':sum(not x['same_best_routes'] for x in arr),
            'road_current_nonzero':sum(abs(x)>1e-8 for x in current),
            'road_best_nonzero':sum(abs(x)>1e-8 for x in best),
            'road_best_positive':sum(x>1e-8 for x in best),
            'road_best_negative':sum(x< -1e-8 for x in best),
            'road_best_mean_sec':_mean(best),'road_current_mean_sec':_mean(current),
            'static_best_mean_sec':_mean([x['static_best_gain_sec'] for x in arr]),
            'static_current_mean_sec':_mean([x['static_current_gain_sec'] for x in arr]),
            'oracle_positive_snapshots':sum(x['oracle_positive_best'] for x in group_s),
            'mean_snapshot_oracle_best_gain_sec':_mean([x['oracle_best_gain_sec'] for x in group_s]),
            'median_first_divergence_step':sorted(events)[len(events)//2] if events else None,
            'first_candidate_accepted_and_terminal_best_changed':sum(x['first_accepted'] and not x['same_best_routes'] for x in arr),
            'first_candidate_accepted_and_current_changed':sum(x['first_accepted'] and not x['same_current_routes'] for x in arr),
        }
    # Independently distinguish physically identical preselection states from different ex-post traffic scenarios.
    physical=defaultdict(set)
    for s in snapshots:
        physical[(s['task_id'],int(s['seed']),int(s['stop_count']),
                  s['current_route_sha256'],s['best_route_sha256'],s['rng_state_sha256'])].add(s['scenario_id'])
    duplicates={"|".join(map(str,key)):sorted(scenarios) for key,scenarios in physical.items() if len(scenarios)>1}
    # Deduplicating physical trajectories is descriptive only; roads remain scenario dependent.
    report={'stage':STAGE,'status':'COMPLETE','horizons':list(horizons),
       'n_candidate_horizon_rows':len(rows),'n_snapshots':len(snapshots),
       'n_distinct_preselection_search_states':len(physical),
       'n_physical_states_repeated_across_scenarios':len(duplicates),
       'duplicate_physical_states':duplicates,'per_horizon':per,
       'limitations':['Data only from inspected July VALIDATION cases; NOT allowed as blind model test or supervised TRAIN labels.',
         'K-step ALNS continuation is a fixed-iteration counterfactual, not 30-second remaining-wall-clock performance.',
         'Road.replay best-route gain is sparse partly because the incumbent best rarely changes within K<=50.',
         'Current-route gain is a process diagnostic, NOT interchangeable with output solution quality.',
         'Same physical search trajectories under different traffic scenarios are correlated; do not treat raw actions as independent samples.',
         'One RNG trajectory per snapshot: no confidence intervals for effects and no guarantee of same RNG draws after intervention.']}
    return report,snaprows


def render_md(rep):
    rows=['# R4-D 连续搜索反事实信号漏斗审计','',
       f"状态：**{rep['status']}**。候选×K记录：{rep['n_candidate_horizon_rows']}；真实起点：{rep['n_snapshots']}；不同物理搜索起点：{rep['n_distinct_preselection_search_states']}。",'',
       '| K | 候选与C3首步结果不同 | 首步被接受 | 最终当前解不同 | 最终最优解不同 | 最优解Road非零收益 | 有正收益的起点 |',
       '|---:|---:|---:|---:|---:|---:|---:|']
    for h,v in rep['per_horizon'].items():
        rows.append(f"| {h} | {v['first_candidate_diff_from_c3']}/{v['n_actions']} | {v['first_accepted']}/{v['n_actions']} | {v['terminal_current_changed']}/{v['n_actions']} | {v['terminal_best_changed']}/{v['n_actions']} | {v['road_best_nonzero']}/{v['n_actions']} | {v['oracle_positive_snapshots']}/{v['n_snapshots']} |")
    rows+=['','| K | 最优解Road收益均值(s) | 当前解Road收益均值(s) | 状态Oracle最优可改善均值(s) |',
           '|---:|---:|---:|---:|']
    for h,v in rep['per_horizon'].items():
        rows.append(f"| {h} | {v['road_best_mean_sec']:+.3f} | {v['road_current_mean_sec']:+.3f} | {v['mean_snapshot_oracle_best_gain_sec']:+.3f} |")
    rows+=['','## 解释边界','',
      '当前解变化不等价于历史最优解改善；零收益不必然表示Section未影响搜索。重复的静态搜索轨迹不能算独立实验单位。',
      '本实验不执行ALNS或Road.replay；所有诊断是对固定K反事实原始事件的事后分析，不能作为在线30秒收益。','']
    return '\n'.join(rows)
