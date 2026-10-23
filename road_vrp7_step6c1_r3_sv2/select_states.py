#!/usr/bin/env python3
"""Pre-register outcome-blind state sampling for frozen R3-SV restart evaluation.

Never uses gain, oracle labels, regrets or model predictions for selection.
Creates a self-contained SV-compatible one-step view; leaves source untouched.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_ARMS={'C3','M0_PROXY','R2_F2','T2_HUBER'}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def stable_key(seed, identifier):
    return hashlib.sha256(f'{seed}:{identifier}'.encode()).hexdigest()

def load_metadata(one_step_dir,role):
    source=Path(one_step_dir)
    m=json.loads((source/'manifest.json').read_text())
    report=json.loads((source/'report.json').read_text())
    if m.get('stage')!='R3-SV-one-step-v1' or report.get('status')!='COMPLETE':
        raise ValueError('Need frozen COMPLETE R3-SV one-step study')
    group={}
    with (source/'decisions.jsonl').open() as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            if row.get('role')!=role:continue
            k=row['decision_id'];arm=row['arm']
            if arm not in REQUIRED_ARMS:raise ValueError('Unexpected arm '+str(arm))
            rec=group.setdefault(k,{'task_id':row['task_id'],'scenario_id':row['scenario_id'],
                                    'date':row['date'],'seed':row['seed'],'arms':set()})
            if (rec['task_id'],rec['scenario_id'],rec['seed'])!=(row['task_id'],row['scenario_id'],row['seed']):
                raise ValueError('Inconsistent metadata in paired decision: '+k)
            if arm in rec['arms']:raise ValueError('Duplicate decision/arm: '+k+'/'+arm)
            rec['arms'].add(arm)
    for k,rec in group.items():
        if rec['arms']!=REQUIRED_ARMS:raise ValueError('Incomplete decision: '+k)
    return group,m

def choose(groups,n,seed,scheme='uniform'):
    """Uniform: deterministic random hash order over states, not model outcomes.

    Balanced: round-robin by task/scenario. Descriptive equal-strata cohort ONLY,
    never call it an unbiased population sample if strata sizes differ.
    """
    if n<1 or n>len(groups):raise ValueError(f'States must be 1..{len(groups)}')
    if scheme=='uniform':
        return sorted(groups,key=lambda k:(stable_key(seed,k),k))[:n]
    if scheme!='balanced':raise ValueError('scheme must be uniform or balanced')
    bins=defaultdict(list)
    for k,r in groups.items():bins[(r['task_id'],r['scenario_id'])].append(k)
    for grp in bins:bins[grp].sort(key=lambda k:(stable_key(seed,k),k))
    # Deterministic round-robin; not equal-probability sampling over all states.
    selected=[]
    while len(selected)<n:
        found=False
        for grp in sorted(bins):
            if bins[grp] and len(selected)<n:
                selected.append(bins[grp].pop(0));found=True
        if not found:raise RuntimeError('Cannot fill selection')
    return selected

def create(source_dir,out_dir,states=48,seed=20261024,role='internal_test',scheme='uniform'):
    source=Path(source_dir).resolve();out=Path(out_dir)
    if out.exists():raise FileExistsError('Output exists; use a NEW selection directory')
    groups,manifest=load_metadata(source,role)
    chosen=choose(groups,states,seed,scheme)
    out.mkdir(parents=True,exist_ok=False)
    # Copy rather than symlink, so existing SV strict_inputs and SHA checks work
    # and the frozen source directory can later be moved independently.
    for fname in ('manifest.json','decisions.jsonl','report.json'):
        shutil.copyfile(source/fname,out/fname)
    plan={'role':role,'seed':seed,'max_states':states,'selected_decision_ids':chosen,
          'sampling':f'outcome_blind_{scheme}',
          'warning':'Uniform is a sample of the captured labeled states, NOT an independent task/date test.'}
    (out/'pilot_states.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    strata=Counter((groups[k]['task_id'],groups[k]['scenario_id']) for k in chosen)
    provenance={'stage':'R3-SV2-outcome-blind-selection-v1','source_dir':str(source),
        'source_one_step_sha256':sha256(source/'decisions.jsonl'),
        'source_manifest_sha256':sha256(source/'manifest.json'),
        'source_pilot_sha256':sha256(source/'pilot_states.json'),
        'selection_sha256':sha256(out/'pilot_states.json'),
        'sampler_seed':seed,'scheme':scheme,'role':role,
        'n_eligible':len(groups),'n_selected':len(chosen),
        'strata_selected':{'|'.join(k):v for k,v in sorted(strata.items())},
        'leakage_rule':'Only role, task_id, scenario_id, decision_id and stable hash used to select',
        'nonblind':'This internal_test split has already been examined in earlier studies'}
    (out/'sv2_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    return provenance

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--one-step-dir',default='runs/chicago_july_v1/step6c1_r3_sv/sv_same_state_v1')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--states',type=int,default=48)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--role',choices=['train','validation','internal_test'],default='internal_test')
    p.add_argument('--scheme',choices=['uniform','balanced'],default='uniform')
    a=p.parse_args();print(json.dumps(create(a.one_step_dir,a.output_dir,a.states,a.seed,a.role,a.scheme),indent=2))
