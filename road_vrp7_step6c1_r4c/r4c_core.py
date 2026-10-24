"""R4-C reproducible candidate/action plumbing and pure counterfactual diagnostics.

Only the FIRST Section destroy partial may be substituted. C3's original
Section.draw is always called first, consuming identical RNG draws. No model.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path

STAGE = 'R4C-CONTINUOUS-SECTION-COUNTERFACTUAL-v1'
HARD_CUTOFF = -40.0


def require(x, msg):
    if not x: raise ValueError(msg)


def canonical_json(o):
    return json.dumps(o, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(o):
    return hashlib.sha256(canonical_json(o).encode('utf8')).hexdigest()


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    os.replace(t,p)


def state_identifier(task_id,scenario_id,routes):
    # Identical to the frozen R2/Step6-B2-R state_identifier implementation.
    obj=[task_id,scenario_id,tuple(tuple(r) for r in routes)]
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()[:24]


def candidate_identity(candidate):
    return (int(candidate['route']),int(candidate['pos']),int(candidate['size']))


def partial_identity(part):
    require(len(part.blocks)==1,'Expected exactly one C3 Section block')
    b=part.blocks[0]
    return (int(b.route),int(b.pos),int(len(b.ids)))


def make_partial(routes,selected,obj):
    """Exact v3.SectionSelector static Section *construction* after selection.

    Does not bypass original RNG draw: caller has already invoked original
    Selector.destroy on the SAME generator at the same state.
    """
    from section_v2 import frozen,mutable,Block,Partial
    rt,pos,length=candidate_identity(selected)
    old=frozen(routes)
    require(0<=rt<len(old) and 0<=pos and length in (2,3) and pos+length<len(old[rt])+1,
            'Invalid selected section boundary')
    require(len(old[rt])-length>=1,'Section would empty route')
    block=Block(tuple(old[rt][pos:pos+length]),rt,pos)
    new=mutable(old)
    del new[rt][pos:pos+length]
    return Partial(frozen(new),obj,(block,),old)


def validate_sham(original,reconstructed):
    from section_v2 import frozen
    require(frozen(original.routes)==frozen(reconstructed.routes),'SHAM partial routes differ')
    require(frozen(original.original)==frozen(reconstructed.original),'SHAM original routes differ')
    require(len(original.blocks)==len(reconstructed.blocks)==1,'SHAM block count mismatch')
    require(original.blocks[0]==reconstructed.blocks[0],'SHAM block mismatch')
    require(original.obj is reconstructed.obj,'SHAM objective ref mismatch')
    return True


def select_alternatives(candidates,sampled,c3,maximum):
    """Keep sampler order, unique physical boundaries, excluding real C3."""
    require(maximum>=1,'Maximum alternatives must be positive')
    keep=[];seen={tuple(c3)}
    for j,c in sampled:
        j=int(j)
        require(0<=j<len(candidates),'Sampled action index invalid')
        require(candidate_identity(candidates[j])==candidate_identity(c),'Sampler candidate mismatch')
        ident=candidate_identity(c)
        if ident in seen:continue
        seen.add(ident);keep.append({'name':f'A{j:03d}','action_index':j,
             'route':ident[0],'pos':ident[1],'size':ident[2]})
        if len(keep)>=maximum:break
    return keep


def branch_spec(snapshot,k,arm):
    return {'task_id':str(snapshot['task_id']),'scenario_id':str(snapshot['scenario_id']),
            'seed':int(snapshot['seed']),'stop_count':int(snapshot['stop_count']),
            'horizon':int(k),'arm':str(arm)}


def unit_name(spec):
    return digest(spec)[:32]+'.json'


def inspect_events(events,snapshot,k):
    require(len(events)==k,'Incomplete continuation events')
    first=events[0]
    require(int(first['stop_count'])==int(snapshot['stop_count']),'Start counter differs')
    for field in ('current_route_sha256','best_route_sha256','rng_state_sha256'):
        require(first[field]==snapshot[field],f'Start {field} mismatch')
    require(first['destroy_index']==3,'First selected operator is not Section')
    require(first['repair_index']==snapshot['real_selected_repair_index'],'Repair choice mismatch')
    for j,event in enumerate(events):
        require(event['stop_count']==snapshot['stop_count']+j,'Non-consecutive continuation event')
    return True


def compare_sham(base,sham):
    require(base['events']==sham['events'],'SHAM changed ALNS event trace')
    require(base['end_signature']==sham['end_signature'],'SHAM changed final ALNS state')
    for k in ('best_route_sha256','current_route_sha256','road_best_sec','road_current_sec','static_best_sec','static_current_sec'):
        require(base[k]==sham[k],f'SHAM differed on {k}')
    require(base['first_action_id']==sham['first_action_id'],'SHAM first action mismatch')
    return True


def normalize_reward(base,arm):
    require(base['snapshot_spec']==arm['snapshot_spec'],'Compared different search starts')
    require(base['horizon']==arm['horizon'],'Compared different K')
    require(base['arm']=='C3','Baseline not C3')
    require(arm['arm'] not in ('C3','SHAM_C3'),'Expected alternative action')
    return {'road_best_gain_sec':float(base['road_best_sec']-arm['road_best_sec']),
        'road_current_gain_sec':float(base['road_current_sec']-arm['road_current_sec']),
        'static_best_gain_sec':float(base['static_best_sec']-arm['static_best_sec']),
        'static_current_gain_sec':float(base['static_current_sec']-arm['static_current_sec']),
        'is_severe_harm':bool(base['road_best_sec']-arm['road_best_sec']<HARD_CUTOFF),
        'same_best_routes':bool(base['best_route_sha256']==arm['best_route_sha256']),
        'same_current_routes':bool(base['current_route_sha256']==arm['current_route_sha256'])}


def summarize(items):
    require(bool(items),'Empty labels')
    labels=[float(x['road_best_gain_sec']) for x in items]
    return {'count':len(items),'mean_road_best_gain_sec':sum(labels)/len(labels),
       'positive_frac':sum(x>1e-8 for x in labels)/len(labels),
       'negative_frac':sum(x< -1e-8 for x in labels)/len(labels),
       'zero_frac':sum(abs(x)<=1e-8 for x in labels)/len(labels),
       'severe_harm_fraction':sum(x< HARD_CUTOFF for x in labels)/len(labels),
       'min_road_best_gain_sec':min(labels),'max_road_best_gain_sec':max(labels)}
