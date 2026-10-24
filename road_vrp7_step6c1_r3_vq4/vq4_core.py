"""VQ4 checkpoint reuse and nested, outcome-blind learning-curve views.

Not an online algorithm. All road data are retrospective learning/evaluation labels.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

VQ_STAGE = 'R3-VQ-train-only-restart-v1'
VQ_REPORT = 'R3-VQ-labels-v1'
SIZES = (48, 96, 144, 192)
EXEMPT_MANIFEST_KEYS = {'selected_decision_ids'}


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,p)


def comparable_manifest(m):
    return {k:v for k,v in m.items() if k not in EXEMPT_MANIFEST_KEYS}


def verify_nested(source_manifest,target_manifest,old_n=48,new_n=192):
    if source_manifest.get('stage')!=VQ_STAGE or target_manifest.get('stage')!=VQ_STAGE:
        raise ValueError('Wrong collection stage')
    if comparable_manifest(source_manifest)!=comparable_manifest(target_manifest):
        raise ValueError('Different restart inputs/codes/data/H/seed: cannot reuse checkpoints')
    old=source_manifest['selected_decision_ids']; new=target_manifest['selected_decision_ids']
    if len(old)!=old_n or len(new)!=new_n or len(set(new))!=len(new):
        raise ValueError('Unexpected selection sizes or duplicate decision IDs')
    if new[:len(old)]!=old:
        raise ValueError('Selection not prefix nested; do not reuse old checkpoints')
    if new_n % 4 or old_n % 4:
        raise ValueError('Need evenly balanced four-task sizes')
    return True


def validate_unit(data,di,h,rep,reps):
    if data.get('decision_id')!=di or int(data.get('horizon',-1))!=h or int(data.get('replicate',-1))!=rep:
        raise ValueError('Checkpoint filename/payload mismatch')
    rows=data.get('rows')
    if not isinstance(rows,list) or not rows:
        raise ValueError('No checkpoint action rows')
    seen=set()
    for r in rows:
        if r.get('decision_id')!=di or int(r.get('horizon',-1))!=h or int(r.get('replicate',-1))!=rep:
            raise ValueError('Bad row identity')
        a=int(r['action_index'])
        if a in seen: raise ValueError('Duplicate candidate')
        seen.add(a)
        for field in ('final_road','final_td_od','one_step_gain_td_od'):
            if field not in r or not math.isfinite(float(r[field])):
                raise ValueError('Missing/nonfinite '+field)
    if -1 not in seen: raise ValueError('No actual C3 baseline arm')
    if not 0<=rep<reps:raise ValueError('Invalid replicate')
    return len(rows)


def reuse_checkpoints(source,target):
    """Copy old files into an already initialized target with exact matching config.

    The target must first have been initialized with `MAX_NEW_UNITS=1` so its
    manifest is authored by original collect_vq.py, not synthesized here.
    """
    source=Path(source);target=Path(target)
    sm=read_json(source/'manifest.json');tm=read_json(target/'manifest.json')
    sr=read_json(source/'report.json')
    if sr.get('stage')!=VQ_REPORT or sr.get('status')!='COMPLETE':
        raise ValueError('Source collection is incomplete')
    verify_nested(sm,tm,len(sm['selected_decision_ids']),len(tm['selected_decision_ids']))
    if (target/'report.json').exists():raise ValueError('Target already finalized; refusing to mutate')
    if not (target/'units').is_dir():raise ValueError('Initialize target by collecting one unit first')
    horizons=[int(x) for x in sm['horizons']];reps=int(sm['replicates']);old=sm['selected_decision_ids']
    expected=len(old)*len(horizons)*reps
    if sr.get('n_selected_states')!=len(old):raise ValueError('Old report count mismatch')
    audit={'stage':'R3-VQ4-reused-checkpoints-v1','source':str(source),'target':str(target),
           'source_manifest_sha256':sha256(source/'manifest.json'),
           'target_manifest_sha256':sha256(target/'manifest.json'),
           'source_rollouts_sha256':sha256(source/'rollouts.jsonl'),
           'expected_units':expected,'copied_units':0,'already_present_units':0,'copies':{},
           'existing_conflicts_checked':0}
    for di in old:
        for h in horizons:
            for rep in range(reps):
                basename=f'{di}__H{h}__R{rep}.json'
                inp=source/'units'/basename;dst=target/'units'/basename
                if not inp.is_file():raise ValueError('Old checkpoint missing: '+str(inp))
                data=read_json(inp);validate_unit(data,di,h,rep,reps)
                if dst.exists():
                    existing=read_json(dst)
                    validate_unit(existing,di,h,rep,reps)
                    # Identical initialization is not guaranteed to preserve wall time.
                    # Compare deterministic true-road and surrogate outcome values.
                    left={int(z['action_index']):z for z in data['rows']}
                    right={int(z['action_index']):z for z in existing['rows']}
                    if left.keys()!=right.keys() or any(abs(float(left[k][metric])-float(right[k][metric]))>1e-5
                        for k in left for metric in ('final_road','final_td_od','one_step_gain_td_od')):
                        raise ValueError('Existing target unit yields different objective values: '+basename)
                    audit['already_present_units']+=1;audit['existing_conflicts_checked']+=1
                else:
                    # Preserve the old validated row bytes; no recomputation or rewriting.
                    shutil.copy2(inp,dst)
                    audit['copied_units']+=1
                audit['copies'][basename]=sha256(dst)
    # This audit is intentionally outside collection's manifest and unit paths.
    write_json(target/'vq4_reuse_audit.json',audit)
    return audit


def _rows_by_state(source,ids,horizons,reps):
    rows=[];keys=set();seen_by_unit=defaultdict(set)
    selected=set(ids)
    with (source/'rollouts.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);di=r['decision_id']
            if di not in selected:continue
            h=int(r['horizon']);rep=int(r['replicate']);a=int(r['action_index'])
            if h not in horizons or not 0<=rep<reps:raise ValueError('Bad horizon/replicate in restart rows')
            key=(di,h,rep,a)
            if key in keys: raise ValueError('Duplicate rollout action')
            keys.add(key);seen_by_unit[(di,h,rep)].add(a);rows.append(r)
    for di in ids:
        for h in horizons:
            expected=None
            for rep in range(reps):
                ss=seen_by_unit.get((di,h,rep))
                if not ss or -1 not in ss:raise ValueError('Missing C3/replicate in view')
                if expected is None:expected=ss
                if expected!=ss:raise ValueError('Candidate mismatch across replicates')
    return rows


def create_views(source,out_root,old48=None,sizes=SIZES):
    source=Path(source);out_root=Path(out_root)
    sm=read_json(source/'manifest.json');sr=read_json(source/'report.json')
    if sm.get('stage')!=VQ_STAGE or sr.get('stage')!=VQ_REPORT or sr.get('status')!='COMPLETE':
        raise ValueError('Need COMPLETE original VQ collection')
    all_ids=sm['selected_decision_ids'];n=len(all_ids)
    if len(set(all_ids))!=n:raise ValueError('Repeated collected decision')
    if sr.get('n_selected_states')!=n:raise ValueError('Collection/report size mismatch')
    if old48:
        old=read_json(Path(old48)/'manifest.json')
        verify_nested(old,sm,len(old['selected_decision_ids']),n)
    horizons=[int(h) for h in sm['horizons']]; reps=int(sm['replicates'])
    results=[]
    for size in sizes:
        if not (0<size<=n) or size%4:raise ValueError('Invalid curve size')
        view=out_root/f'vq4_view{size}_v1'
        if view.exists():raise FileExistsError('Learning-curve view already exists: '+str(view))
        ids=all_ids[:size];rows=_rows_by_state(source,ids,horizons,reps)
        report=copy.deepcopy(sr);report['n_selected_states']=size
        report['n_restart_action_rows']=len(rows)
        idslabels={(r['decision_id'],int(r['horizon']),int(r['action_index'])) for r in rows}
        report['n_state_horizon_action_labels']=len(idslabels)
        report['per_horizon_action_labels']={str(h):len([v for v in idslabels if v[1]==h]) for h in horizons}
        manifest=copy.deepcopy(sm);manifest['selected_decision_ids']=ids
        view.mkdir(parents=True)
        write_json(view/'manifest.json',manifest)
        write_json(view/'report.json',report)
        with (view/'rollouts.jsonl').open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n')
        write_json(view/'vq4_provenance.json',{
            'stage':'R3-VQ4-nested-learning-curve-view-v1',
            'original_collection_manifest_sha256':sha256(source/'manifest.json'),
            'original_collection_rollouts_sha256':sha256(source/'rollouts.jsonl'),
            'view_manifest_sha256':sha256(view/'manifest.json'),
            'view_rollouts_sha256':sha256(view/'rollouts.jsonl'),
            'n_states':size,'selection_outcome_blind':True,
            'limits':['No new independent tasks; all prefixes share the same 4 TRAIN tasks',
                      'This is a view of exactly the existing measured restarts.']})
        results.append({'size':size,'view':str(view),'restart_rows':len(rows),'labels':len(idslabels)})
    return results
