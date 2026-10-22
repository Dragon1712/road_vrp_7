#!/usr/bin/env python3
"""Offline label actual C3 action and 8 learned alternatives using SAME Static Repair.

Oracle TD costs are retrospective *labels only*. The saved X/B features use
static topology and STRICTLY PREDEPARTURE historical observations exclusively.
"""
from __future__ import annotations
import argparse, json, sys, time, os
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
for f in ('road_vrp7_step6c1_r2','road_vrp7_step6a','road_vrp7_step6b2','road_vrp7_step6b1_v2','road_vrp7_stage3','road_vrp7_step5b_v3'):
    sys.path.insert(0,str(ROOT/f))
from dataset import sha256
from section_dataset import select_actions,repair_for,HistoricalCorridorFeatures,read_raw_cost
from road_vrp import Road,OD,Objective,legal,read_json
from section_v2 import Selector
from features import FEATURE_MODES
from r2_core import paired_features,matches,state_key


def score_one(routes,action,td,static,top_k,timeout,enable_swap,orig_td):
    """Frozen static Section Repair; TD oracle used only for ex post value."""
    result,reason,elapsed,ok,diag=repair_for(static,routes,action,top_k,timeout,enable_swap)
    if not ok:return False,0.,reason,elapsed
    return True,float(orig_td-td.cost(result)),reason,elapsed


def run(a):
    out=Path(a.output_dir)
    if out.exists() and not a.resume:raise FileExistsError('New output directory required; --resume allows identical input')
    captured=Path(a.capture_dir);capmanifest=read_json(captured/'manifest.json')
    summary=read_json(captured/'summary.json')
    if capmanifest['stage']!='step6c1r2-real-C3-capture-v1' or summary['status']!='COMPLETE' or summary['state_sha256']!=sha256(captured/'states.jsonl'):
        raise ValueError('Invalid capture dataset or changed states')
    if a.action_count!=8:raise ValueError('Frozen 8-candidate treatment required')
    if capmanifest['section_sha256']!=sha256(ROOT/'road_vrp7_step5b_v3/section_v2.py'):
        raise ValueError('Frozen C3/Static Repair source changed since capture')
    road=Road(a.cost_dir)
    if capmanifest['road_sha256']!=road.cost_file_hash or capmanifest['edge_sha256']!=road.edge_file_hash:
        raise ValueError('Road source differs from C3 capture')
    task_manifest=read_json(Path(a.task_dir)/'manifest.json')
    if task_manifest['cost_edge_index_sha256']!=road.edge_file_hash or task_manifest['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Task/cost mismatch')
    raw=read_raw_cost(a.cost_dir)
    hist=HistoricalCorridorFeatures(road,raw,6)
    states=[json.loads(line) for line in (captured/'states.jsonl').read_text().splitlines() if line.strip()]
    if len(states)!=summary['captured'] or len({r['decision_id'] for r in states})!=len(states):
        raise ValueError('Missing or duplicated C3 states')
    if a.max_states:states=states[:a.max_states]
    checksum={'stage':'step6c1r2-label-checkpoint-v1','capture_state_sha256':summary['state_sha256'],
      'capture_manifest_sha256':sha256(captured/'manifest.json'),
      'task_manifest_sha256':sha256(Path(a.task_dir)/'manifest.json'),
      'road_sha256':road.cost_file_hash,'edge_sha256':road.edge_file_hash,
      'action_count':a.action_count,'sampler_seed':a.sampler_seed,'top_k':a.top_k,
      'deadline_sec':a.deadline_sec,'max_states':a.max_states,'enable_swap':a.enable_swap,
      'od_step_min':a.od_step_min,'od_hours':a.od_hours,
      'label_code_sha256':sha256(Path(__file__))}
    if out.exists():
        if not (out/'checkpoint_meta.json').exists() or read_json(out/'checkpoint_meta.json')!=checksum:
            raise ValueError('Resume refused: labeler inputs/config/code hash changed')
        if (out/'summary.json').exists() and read_json(out/'summary.json').get('status')=='COMPLETE':
            print('Existing COMPLETE label output left untouched');return
    else:
        out.mkdir(parents=True)
        (out/'checkpoint_meta.json').write_text(json.dumps(checksum,indent=2)+'\n')
    checkpoint=out/'checkpoint.jsonl';done={}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if not line.strip():continue
            z=json.loads(line)
            if z['decision_id'] in done:raise ValueError('Duplicate label checkpoint')
            done[z['decision_id']]=z
    if not set(done).issubset({z['decision_id'] for z in states}):
        raise ValueError('Label checkpoint contains unknown state')
    def save_checkpoint(record):
        with checkpoint.open('a') as f:
            f.write(json.dumps(record,separators=(',',':'))+'\n')
            f.flush();os.fsync(f.fileno())
        done[record['decision_id']]=record
    tasks={}; od_cache={}; observations=[];Xs=[]; Bs=[];Ys=[];BY=[];Vs=[];Sames=[]
    skipped=[]
    for count,s in enumerate(states,1):
        old=done.get(s['decision_id'])
        if old is not None:
            if old['status']=='skipped':
                skipped.append({'decision_id':s['decision_id'],'reason':old['reason']});continue
            if old['status']!='ok':raise ValueError('Bad checkpoint row status')
            Xs.append(np.asarray(old['X'],np.float32));Bs.append(np.asarray(old['B'],np.float32))
            Ys.append(np.asarray(old['Y'],np.float32));BY.append(old['BY'])
            Vs.append(np.asarray(old['valid'],bool));Sames.append(np.asarray(old['same'],bool))
            observations.append(old['group']);continue
        tid=s['task_id'];scid=s['scenario_id'];key=(tid,scid)
        if tid not in tasks:tasks[tid]=read_json(Path(a.task_dir)/(tid+'.json'))
        task=tasks[tid];sc=next((sc for sc in task['scenarios'] if sc['scenario_id']==scid),None)
        if sc is None or sc['departure_local']!=s['departure_local']:raise ValueError('Capture scenario mismatch')
        routes=s['routes'];legal(task,routes)
        if state_key(tid,scid,routes)!=s['state_id']:raise ValueError('Capture route/state mismatch')
        if key not in od_cache:
            od=OD(road,task,sc,step_min=a.od_step_min,hours=a.od_hours,cache_dir=a.od_cache_dir)
            od_cache[key]=od
        od=od_cache[key];td=Objective(task,od,'oracle');static=Objective(task,od,'static')
        hist.enter(tid,od,sc['departure_local']);timestamp=pd.Timestamp(sc['departure_local'])
        hour=timestamp.hour+timestamp.minute/60.;weekday=timestamp.weekday()
        selector=Selector(static)
        pool=selector.candidates(routes,traffic=False)
        choices=select_actions(pool,a.action_count,s['state_id'],a.sampler_seed)
        if len(choices)!=a.action_count:
            skipped.append({'decision_id':s['decision_id'],'reason':'too_few_candidates'});save_checkpoint({'decision_id':s['decision_id'],'status':'skipped','reason':'too_few_candidates'});continue
        orig_td=float(td.cost(routes));cost=float(static.cost(routes))
        c3=s['c3_action'];selected=next((c for c in pool if matches(c,c3)),None)
        if selected is None or routes[c3['route']][c3['pos']:c3['pos']+c3['size']]!=c3['block_ids']:
            raise ValueError('Stored actual C3 action is not an enumerated Section')
        bx=paired_features(routes,selected,static,hist,hour,weekday,'F4',cost)
        if bx is None:
            skipped.append({'decision_id':s['decision_id'],'reason':'C3_no_proxy'});save_checkpoint({'decision_id':s['decision_id'],'status':'skipped','reason':'C3_no_proxy'});continue
        ok,br,why,elapsed=score_one(routes,selected,td,static,a.top_k,a.deadline_sec,a.enable_swap,orig_td)
        if not ok:
            skipped.append({'decision_id':s['decision_id'],'reason':'C3_'+why});save_checkpoint({'decision_id':s['decision_id'],'status':'skipped','reason':'C3_'+why});continue
        x=np.zeros((a.action_count,44),np.float32)
        y=np.zeros(a.action_count,np.float32);valid=np.zeros(a.action_count,bool);same=np.zeros(a.action_count,bool)
        for j,(idx,c) in enumerate(choices):
            fx=paired_features(routes,c,static,hist,hour,weekday,'F4',cost)
            if fx is None:continue
            x[j]=fx
            same[j]=matches(c,c3)
            if same[j]:
                y[j]=br;valid[j]=True;continue
            good,gain,reason,_=score_one(routes,c,td,static,a.top_k,a.deadline_sec,a.enable_swap,orig_td)
            if good:y[j]=gain;valid[j]=True
        if not valid.any():
            skipped.append({'decision_id':s['decision_id'],'reason':'all_candidates_invalid'});save_checkpoint({'decision_id':s['decision_id'],'status':'skipped','reason':'all_candidates_invalid'});continue
        Xs.append(x);Bs.append(bx);Ys.append(y);BY.append(br);Vs.append(valid);Sames.append(same)
        group={k:s[k] for k in ('decision_id','state_id','task_id','scenario_id','seed','date','section_call')}
        observations.append(group)
        save_checkpoint({'decision_id':s['decision_id'],'status':'ok','group':group,
            'X':x.tolist(),'B':bx.tolist(),'Y':y.tolist(),'BY':float(br),
            'valid':valid.tolist(),'same':same.tolist()})
        if count%20==0:print('C3LABEL',count,'/',len(states),'valid',len(Xs),'skips',len(skipped),flush=True)
    if not Xs:raise RuntimeError('No valid C3-relative training states; increase repair deadline or capture')
    np.savez_compressed(out/'paired.npz',X=np.stack(Xs),B=np.stack(Bs),Y=np.stack(Ys),BY=np.asarray(BY,np.float32),
                valid=np.stack(Vs),same=np.stack(Sames))
    with (out/'groups.jsonl').open('w') as f:
        for z in observations:f.write(json.dumps(z,ensure_ascii=False)+'\n')
    (out/'skipped.json').write_text(json.dumps(skipped,ensure_ascii=False,indent=2)+'\n')
    mf={'stage':'step6c1r2-real-C3-labelled-v1','capture_manifest_sha256':sha256(captured/'manifest.json'),
        'capture_state_sha256':sha256(captured/'states.jsonl'),'pair_sha256':sha256(out/'paired.npz'),
        'groups_sha256':sha256(out/'groups.jsonl'),'road_sha256':road.cost_file_hash,'edge_sha256':road.edge_file_hash,
        'label_code_sha256':sha256(Path(__file__)),'source_dir':str(captured),
        'label_role':'RETROSPECTIVE oracle TD objective ONLY, never online feature',
        'feature_role':'F4 historical observations strictly before departure',
        'action_count':a.action_count,'sampler_seed':a.sampler_seed,'top_k':a.top_k,
        'deadline_sec':a.deadline_sec,'enable_swap':a.enable_swap,
        'notes':['B is the ACTUAL C3 Section observed during frozen C3 training trajectory.',
            'Both C3 baseline and learner actions use frozen Static Repair and TD ex-post scoring.',
            'One-step static-repair TD advantage is NOT full 30-second ALNS return.',
            'States are observationally collected; ALNS runtime and RNG dependence remain.']}
    (out/'manifest.json').write_text(json.dumps(mf,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps({'status':'COMPLETE','states':len(Xs),'skipped_states':len(skipped),
       'candidate_valid_fraction':float(np.stack(Vs).mean()),'candidate_including_c3_rate':float(np.stack(Sames).any(1).mean()),
       'baseline_gain_mean':float(np.mean(BY))},indent=2)+'\n')
    print('C3LABEL COMPLETE states',len(Xs),'skipped',len(skipped),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--action-count',type=int,default=8);p.add_argument('--sampler-seed',type=int,default=20261019)
    p.add_argument('--top-k',type=int,default=12);p.add_argument('--deadline-sec',type=float,default=3.)
    p.add_argument('--max-states',type=int);p.add_argument('--resume',action='store_true')
    p.add_argument('--od-step-min',type=int,default=20)
    p.add_argument('--od-hours',type=float,default=5.);p.add_argument('--enable-swap',action='store_true',default=True)
    run(p.parse_args())
