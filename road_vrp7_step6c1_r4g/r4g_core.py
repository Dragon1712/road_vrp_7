"""R4-G pure TRAIN role, provenance, paired equality, physical-state accounting."""
from __future__ import annotations
import json
from collections import Counter,defaultdict
from pathlib import Path
from r4a_core import sha256,canonical_hash

STAGE='R4G-TRAIN-SECTION-SNAPSHOT-v1'

def require(ok,message):
    if not ok: raise ValueError(message)

def read_json(path): return json.loads(Path(path).read_text(encoding='utf8'))
def read_jsonl(path): return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]

def validate_scenario_role(task,scenario_ids,role='train'):
    found={s['scenario_id']:s for s in task['scenarios']}
    require(set(scenario_ids)<=set(found),'Unknown scenario ID')
    for sid in scenario_ids:
        require(found[sid].get('split')==role and found[sid].get('eligible') is True,
                f'Forbidden role/ineligible: {sid}: {found[sid].get("split")}')
    return True

def load_capture(folder,expected_arm=None):
    p=Path(folder)
    mf=read_json(p/'manifest.json');rep=read_json(p/'report.json');prov=read_json(p/'provenance.json')
    require(mf.get('stage')==STAGE and mf.get('split')=='train','Wrong stage or non-TRAIN capture')
    require(rep.get('stage')==STAGE and rep.get('status')=='COMPLETE','Incomplete capture')
    if expected_arm:require(mf.get('arm')==expected_arm,'Wrong arm')
    for k,filename in [('manifest_sha256','manifest.json'),('metrics_sha256','metrics.jsonl'),('snapshots_sha256','snapshots.jsonl')]:
        require(prov.get(k)==sha256(p/filename),f'Source SHA mismatch: {filename}')
    metrics=read_jsonl(p/'metrics.jsonl');snaps=read_jsonl(p/'snapshots.jsonl')
    require(len(metrics)==rep.get('records') and len(snaps)==rep.get('snapshots'),'Row counts differ')
    require(len(metrics)>0,'Empty capture')
    keys=[(r['task_id'],r['scenario_id'],int(r['seed'])) for r in metrics]
    require(len(set(keys))==len(keys),'Duplicate metric key')
    require(set(keys)=={(t,s,int(seed)) for t in mf['task_ids'] for s in mf['scenario_ids'] for seed in mf['seeds']},'Incomplete task/scenario/seed grid')
    require(set(mf['task_ids'])<={'task_0000','task_0001','task_0002','task_0003'},'Only 4 original TRAIN tasks authorized in R4G v1')
    if expected_arm=='capture':
        from r4b_core import validate_checkpoint
        require(mf.get('section_only') is True and bool(snaps),'Empty/non-Section snapshot capture')
        for s in snaps:
            require((s['task_id'],s['scenario_id'],int(s['seed'])) in set(keys),'Snapshot run key not found')
            validate_checkpoint(s,int(mf['max_iterations']))
    elif expected_arm=='plain':require(not snaps,'Plain must have zero snapshots')
    return mf,metrics,snaps

def physical_id(s):
    return (str(s['task_id']),int(s['seed']),int(s['stop_count']),
            s['current_route_sha256'],s['best_route_sha256'],s['rng_state_sha256'],
            canonical_hash(s['roulette_destroy_weights']),canonical_hash(s['roulette_repair_weights']),
            float(s['anneal_temperature_before_accept']))

def paired_audit(plain,capture):
    mp,rp,sp=plain;mc,rc,sc=capture
    for k in ('task_ids','scenario_ids','seeds','split','search_wall_sec','max_iterations',
              'anneal_iterations','top_k','source_sha256','road_sha256','edge_sha256',
              'solver_sha256','section_sha256'):
        require(mp.get(k)==mc.get(k),'Pair source/config differs in '+k)
    def index(rows):return {(r['task_id'],r['scenario_id'],int(r['seed'])):r for r in rows}
    P=index(rp);C=index(rc)
    require(set(P)==set(C),'Pair keys mismatch')
    details=[]
    for key in sorted(P):
        a=P[key];b=C[key]
        iter_limited=a['iterations']==b['iterations']==mp['max_iterations']
        route_equal=a['best_route_sha256']==b['best_route_sha256']
        road_equal=a['road_cost_sec']==b['road_cost_sec']
        match=route_equal and road_equal
        reasons=[]
        if not iter_limited:reasons.append('ITERATION_CAP_NOT_MET')
        if a['iterations']!=b['iterations']:reasons.append('ITERATION_COUNT_DIFFERS')
        if not route_equal:reasons.append('FINAL_ROUTE_MISMATCH')
        if not road_equal:reasons.append('ROAD_COST_MISMATCH')
        details.append({'task_id':key[0],'scenario_id':key[1],'seed':key[2],
                        'plain_iterations':a['iterations'],'capture_iterations':b['iterations'],
                        'plain_termination_reason':a.get('termination_reason'),
                        'capture_termination_reason':b.get('termination_reason'),
                        'plain_wall_sec':float(a['search_wall_sec']),
                        'capture_wall_sec':float(b['search_wall_sec']),
                        'road_delta_capture_minus_plain_sec':float(b['road_cost_sec']-a['road_cost_sec']),
                        'route_equal':route_equal,'road_equal':road_equal,
                        'iteration_limited_both':iter_limited,'route_and_road_equal':match,
                        'strong_pass':bool(iter_limited and match),
                        'failure_reasons':reasons,
                        'capture_minus_plain_wall_sec':round(b['search_wall_sec']-a['search_wall_sec'],6),
                        'n_snapshots':b['captured']})
    physical=defaultdict(set)
    for s in sc:physical[physical_id(s)].add(str(s['scenario_id']))
    failures=Counter(reason for d in details for reason in d['failure_reasons'])
    strict_all=all(d['strong_pass'] for d in details)
    return {'stage':'R4G-TRAIN-NONINTERFERENCE-AUDIT-v1','status':'COMPLETE',
            'strict_verdict':'VERIFIED' if strict_all else 'NOT_VERIFIED',
            'can_claim_fixed_iteration_noninterference':strict_all,
            'n_failed_pairs':sum(not d['strong_pass'] for d in details),
            'failure_reason_counts':dict(sorted(failures.items())),
            'pairs':len(details),'strict_pass':sum(x['strong_pass'] for x in details),
            'route_road_equal':sum(x['route_and_road_equal'] for x in details),
            'wall_limited_pairs':sum(not x['iteration_limited_both'] for x in details),
            'n_snapshots':len(sc),'n_distinct_physical_states':len(physical),
            'multi_scenario_physical_states':sum(len(v)>1 for v in physical.values()),
            'n_task_groups':len(set(x['task_id'] for x in details)),
            'details':details,
            'limitations':['Only TRAIN role, not model performance.','Fixed-iteration equality only when both reach cap.',
                           'Same physical states in several scenarios are not independent.',
                           'No restored ALNS equivalence until R4G proof passes.']}

def select_phase_snapshots(all_snapshots,max_states=3):
    """Evenly cover EARLY/MIDDLE/LATE captured states, not just earliest prefixes."""
    require(max_states>0,'max_states must be >0')
    ss=sorted(all_snapshots,key=lambda s:s['stop_count'])
    require(len({x['stop_count'] for x in ss})==len(ss),'Duplicate snapshot iteration')
    if len(ss)<=max_states:return ss
    # Index quantiles, with explicit non-colliding groups.
    indices=[round(j*(len(ss)-1)/(max_states-1)) for j in range(max_states)] if max_states>1 else [len(ss)//2]
    require(len(set(indices))==len(indices),'Nonunique selected phases')
    return [ss[i] for i in indices]
