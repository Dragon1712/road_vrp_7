#!/usr/bin/env python3
"""R4-E: audited FIRST Section counterfactuals at K=100/200/500.

Runs from verified R4-A checkpoints, via unchanged R4-C intervention and R4-B
restoration. C3/SHAM must match. Prior R4-C K=50 prefix must match each arm.
Development ONLY: inspected July Validation, no training or 30s gain claims.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r4e','road_vrp7_step6c1_r4c','road_vrp7_step6c1_r4b',
               'road_vrp7_step6c1_r4a','road_vrp7_step5b_v3','road_vrp7_step6b2_r',
               'road_vrp7_step6a','road_vrp7_stage3'):
    p=str(ROOT/folder)
    if p not in sys.path:sys.path.insert(0,p)
from r4e_core import (STAGE_CF,STAGE_VERIFY,require,sha,read,write,read_lines,
                      horizons,snapshot_id,physical_id,verified_long_evidence,
                      diagnose,aggregate,verify_r4b_evidence)
import counterfactual_r4c as r4c
from r4c_core import (STAGE as R4C_STAGE,branch_spec,unit_name,compare_sham,normalize_reward)


def preflight(a, ks):
    proof=verified_long_evidence(a.long_verify_dir,a.capture_dir,a.r4b_dir,ks)
    v3,OD,Objective,legal,taskmap,cfg,road,mf,snapshots,pins,verified=r4c.preflight(a)
    require(set(proof).issubset(set(verified)),'Long proof uses R4B-unverified snapshot')
    proof_manifest=read(Path(a.long_verify_dir)/'manifest.json')
    require(proof_manifest.get('code_sha256',{}).get('verify_long.py')==sha(Path(__file__).with_name('verify_long.py')),
            'Long proof verifier code changed since audit')
    require(proof_manifest.get('code_sha256',{}).get('r4e_core.py')==sha(Path(__file__).with_name('r4e_core.py')),
            'Long proof core changed since audit')
    r4c_dir=Path(a.r4c_dir)
    orig=read(r4c_dir/'report.json');mm=read(r4c_dir/'manifest.json')
    require(orig.get('stage')==R4C_STAGE and orig.get('status')=='COMPLETE','Original R4C study incomplete')
    require(mm.get('r4a_snapshots_sha256')==pins['r4a_snapshots_sha256'],'R4C source snapshot differs')
    require(mm.get('code_sha256',{}).get('counterfactual_r4c.py')==sha(ROOT/'road_vrp7_step6c1_r4c/counterfactual_r4c.py') and
            mm.get('code_sha256',{}).get('r4c_core.py')==sha(ROOT/'road_vrp7_step6c1_r4c/r4c_core.py'),
            'Frozen R4C intervention implementation changed')
    require(mm.get('r4b_report_sha256')==pins['r4b_report_sha256'],'R4C R4B proof differs')
    require(mm.get('action_count')==a.action_count and mm.get('sampler_seed')==a.sampler_seed,
            'R4C action sampler differs from long experiment')
    require(mm.get('max_alternatives')>=a.max_alternatives,'R4C did not evaluate all requested arms at K50')
    require(mm.get('frozen_solver_sha256')==mf['solver_sha256'] and mm.get('frozen_section_sha256')==mf['section_sha256'],
            'R4C uses a different frozen solver')
    return (proof,(v3,OD,Objective,legal,taskmap,cfg,road,mf,snapshots,pins,verified),
            {'r4c_manifest_sha256':sha(r4c_dir/'manifest.json'),
             'r4c_report_sha256':sha(r4c_dir/'report.json'),
             'long_proof_report_sha256':sha(Path(a.long_verify_dir)/'report.json'),
             'long_proof_rows_sha256':sha(Path(a.long_verify_dir)/'paired_rows.jsonl')})


def verify_prior_k50(old_dir,snapshot,arm,result):
    spec=branch_spec(snapshot,50,arm);unit=Path(old_dir)/'units'/unit_name(spec)
    require(unit.exists(),f'K50 frozen reference missing for {spec}')
    saved=read(unit)
    require(saved.get('stage')==R4C_STAGE and saved.get('unit_spec')==spec,'Wrong original R4C K50 unit')
    old=saved['result'];require(old['arm']==arm,'Old arm mismatch')
    require(len(old['events'])==50 and result['events'][:50]==old['events'],
            f'Long-K arm {arm} changed original R4C first 50 steps')
    require(result['first_action_id']==old['first_action_id'],'First Section identity differs from R4C')
    return True


def run(a):
    ks=horizons(a.horizons)
    require(a.max_cases>=0 and a.max_snapshots_per_run>=1 and
        1<=a.max_alternatives<=a.action_count<=8 and a.max_new_units>=0,'Invalid experiment size')
    proof,payload,pins=preflight(a,ks)
    v3,OD,Objective,legal,taskmap,cfg,road,mf,snapshots,r4c_pins,verified=payload
    chosen=r4c._selected_snapshots(snapshots,a.max_cases,a.max_snapshots_per_run)
    for _,s in chosen:
        sid=snapshot_id(s)
        require(sid in proof and set(ks).issubset(proof[sid]),'Requested R4-E checkpoint/K is not precisely validated')
    out=Path(a.output_dir)
    manifest={'stage':STAGE_CF,'status':'RUNNING',
        'source_capture_sha256':r4c_pins['r4a_snapshots_sha256'],
        'source_r4b_sha256':r4c_pins['r4b_report_sha256'],
        **pins,'horizons':list(ks),'max_cases':a.max_cases,
        'max_snapshots_per_run':a.max_snapshots_per_run,
        'action_count':a.action_count,'max_alternatives':a.max_alternatives,'sampler_seed':a.sampler_seed,
        'selected_snapshots':[list(snapshot_id(s)) for _,s in chosen],
        'code_sha256':{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('r4e_core.py'))},
        'evaluation':'FIXED_ITERATION_K_NOT_30S_WALL_CLOCK',
        'data_role':'INSPECTED_JULY_VALIDATION_DEVELOPMENT_ONLY'}
    if out.exists():
        require(a.resume,'Independent output directory required; set RESUME=1 to restart')
        old=read(out/'manifest.json');require({k:v for k,v in old.items() if k!='status'}=={k:v for k,v in manifest.items() if k!='status'},'Resume SHA/config mismatch')
    else:
        require(not a.resume,'Cannot resume a new study')
        out.mkdir(parents=True,exist_ok=False);write(out/'manifest.json',manifest)
    (out/'units').mkdir(exist_ok=True)
    all_rows=[];all_branch=[];all_oracles=[];reused=created=0;expected_branches=0
    for index,((tid,sid,seed),snapshot) in enumerate(chosen,1):
        task=taskmap[tid];sc=next(x for x in task['scenarios'] if x['scenario_id']==sid)
        od=OD(road,task,sc,step_min=int(cfg['od_step_min']),hours=float(cfg['od_hours']),cache_dir=a.od_cache_dir)
        obj=Objective(task,od,'static')
        plan=r4c.shadow_action(snapshot,obj,a.action_count,a.sampler_seed,a.max_alternatives)
        arms=[('C3',None),('SHAM_C3',None)]+[(z['name'],z) for z in plan['alternatives']]
        expected_branches+=len(arms)*len(ks)
        by_arm={};physical=list(physical_id(snapshot))
        for k in ks:
            runs={}
            for arm,choice in arms:
                spec=branch_spec(snapshot,k,arm)
                unit=out/'units'/unit_name(spec)
                if unit.exists():
                    pack=read(unit)
                    require(pack.get('stage')==STAGE_CF and pack.get('unit_spec')==spec and pack.get('plan')==plan,
                            'Resumed unit config/plan mismatch')
                    require(pack.get('r4c_manifest_sha256')==pins['r4c_manifest_sha256'] and
                            pack.get('long_proof_rows_sha256')==pins['long_proof_rows_sha256'],
                            'Resumed unit source proof SHA mismatch')
                    result=pack['result'];reused+=1
                else:
                    if a.max_new_units and created>=a.max_new_units:
                        write(out/'partial.json',{'status':'PARTIAL','completed_new_units':created,
                                                  'reused_units':reused,'next_unit':spec})
                        print('R4E collection capped. Re-run with RESUME=1 and larger MAX_NEW_UNITS',flush=True)
                        return
                    result=r4c.run_one(v3,task,sc,int(seed),obj,snapshot,k,arm,choice,
                                      plan['c3_identity'],road,legal,mf,
                                      plan['first_rng_after_original_C3_destroy_sha256'])
                    verify_prior_k50(a.r4c_dir,snapshot,arm,result)
                    write(unit,{'stage':STAGE_CF,'unit_spec':spec,'plan':plan,'result':result,
                                'r4c_manifest_sha256':pins['r4c_manifest_sha256'],
                                'long_proof_rows_sha256':pins['long_proof_rows_sha256']})
                    created+=1
                # Re-check saved checkpoint to detect post-hoc corruptions.
                verify_prior_k50(a.r4c_dir,snapshot,arm,result)
                key=(snapshot_id(snapshot),arm)
                if key in by_arm:
                    earlier=by_arm[key]
                    require(result['events'][:len(earlier)]==earlier,'Cross-K event prefix diverged')
                by_arm[key]=result['events']
                runs[arm]=result
                if arm=='SHAM_C3':compare_sham(runs['C3'],result)
                if arm not in ('C3','SHAM_C3'):
                    diagnostics=diagnose(runs['C3'],result)
                    all_rows.append({'unit_spec':spec,'physical_id':physical,
                                     'first_action_id':result['first_action_id'],
                                     'original_c3_action_id':result['first_original_c3_action_id'],
                                     **diagnostics})
                all_branch.append({'unit_spec':spec,'best_road_sec':result['road_best_sec'],
                    'current_road_sec':result['road_current_sec'],'best_route_sha256':result['best_route_sha256']})
            gain=max([0.0]+[float(r['best_road_gain_sec']) for r in all_rows
                if snapshot_id(r['unit_spec'])==snapshot_id(snapshot) and r['unit_spec']['horizon']==k])
            all_oracles.append({'snapshot_id':list(snapshot_id(snapshot)),'physical_id':physical,
                                'horizon':k,'sampled_oracle_best_gain_sec':gain,
                                'positive':gain>1e-8,'evaluated_arms':len(arms)-2})
            print('R4E LONG',index,'/',len(chosen),tid,sid,seed,'stop',snapshot['stop_count'],
                  'K',k,'alts',len(arms)-2,'oracle_gain',round(gain,3),flush=True)
    require(len(all_branch)==expected_branches,'Branch count mismatch')
    from r4e_core import aggregate
    report={'stage':STAGE_CF,'status':'COMPLETE','horizons':list(ks),
            'n_branch_results':len(all_branch),'n_candidate_horizon_labels':len(all_rows),
            'n_snapshots':len(chosen),'n_physical_states':len(set(physical_id(s) for _,s in chosen)),
            'n_sham_verified':len(chosen)*len(ks),'n_prior_k50_prefix_verified':len(all_branch),
            'per_horizon':aggregate(all_rows,ks),
            'limitations':['Previously inspected July Validation ONLY; not train/independent test.',
                           'Only fixed K not the remaining 30-second online wall budget.',
                           'Actions share a state and scenarios share the same static search trajectory.',
                           'One trajectory seed per physical start; no confidence interval.']}
    with (out/'labels.jsonl').open('w',encoding='utf8') as f:
        for r in all_rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'branches.jsonl').open('w',encoding='utf8') as f:
        for r in all_branch:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'snapshot_oracle.jsonl').open('w',encoding='utf8') as f:
        for r in all_oracles:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    write(out/'report.json',report)
    lines=['# R4-E 更长固定K连续搜索反事实','',
        f"状态：**COMPLETE**；快照：{len(chosen)}；不同物理起点：{report['n_physical_states']}；SHAM：{report['n_sham_verified']}；K50前缀：{report['n_prior_k50_prefix_verified']}。",'',
        '| K | 动作数 | 正/负/零收益 | 当前解改变 | 最优解改变 | 采样Oracle正收益快照 | 最优解Road均值(s) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for k,x in report['per_horizon'].items():
        lines.append(f"| {k} | {x['n_actions']} | {x['positive_best']}/{x['negative_best']}/{x['zero_best']} | {x['terminal_current_changed']} | {x['terminal_best_changed']} | {x['oracle_positive_scenario_snapshots']}/{x['n_scenario_snapshots']} | {x['mean_road_best_gain_sec']:+.3f} |")
    lines.extend(['','仅固定迭代、既有July Validation开发场景；不能推断30秒在线收益。',''])
    (out/'report.md').write_text('\n'.join(lines),encoding='utf8')
    manifest['status']='COMPLETE';write(out/'manifest.json',manifest)
    print('R4E COMPLETE',out,'new',created,'reused',reused,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for s in ('capture-dir','r4b-dir','r4c-dir','long-verify-dir','model-source','cost-dir','task-dir','od-cache-dir','output-dir'):
        p.add_argument('--'+s,required=True)
    p.add_argument('--horizons',default='100,200,500')
    p.add_argument('--max-cases',type=int,default=0)
    p.add_argument('--max-snapshots-per-run',type=int,default=2)
    p.add_argument('--action-count',type=int,default=8)
    p.add_argument('--max-alternatives',type=int,default=8)
    p.add_argument('--sampler-seed',type=int,default=20261019)
    p.add_argument('--max-new-units',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
