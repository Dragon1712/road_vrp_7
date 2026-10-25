#!/usr/bin/env python3
"""R4-H TRAIN-only exact-gated Section first-step counterfactuals (fixed K).

Reuses R4-C's untouched shadow_action/run_one and frozen v3 Section/ALNS
instead of forking solver logic. Stops closed if R4G-Prefix proof unavailable.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r4h','road_vrp7_step6c1_r4g_prefix',
               'road_vrp7_step6c1_r4g','road_vrp7_step6c1_r4c',
               'road_vrp7_step6c1_r4b','road_vrp7_step6c1_r4a',
               'road_vrp7_step5b_v3','road_vrp7_step6b2_r',
               'road_vrp7_step6a','road_vrp7_stage3'):
    path=str(ROOT/folder)
    if path not in sys.path:sys.path.insert(0,path)
from r4h_core import (STAGE,require,sha,load,save,snap_id,parse_ks,
                      verified_allowlist,select_train_snapshots,paired_reward,report_metrics,horizon_stability)


def preflight(a):
    # This TRAIN-specific preflight pins v3/Section/R4A/R4G collector, road/OD
    # data hashes and validates per-task TRAIN/eligible scenario membership.
    import verify_train_r4g as trainer
    from r4g_core import validate_scenario_role
    import importlib.metadata
    require(importlib.metadata.version('alns')=='7.0.0','Require frozen official ALNS v7.0.0')
    mf,source_snaps,taskmap,road,OD,Objective,legal,v3,cfg=trainer.preflight(a)
    require(mf['split']=='train' and set(mf['task_ids'])<=set(f'task_{i:04d}' for i in range(4)),
            'R4-H only accepts original TRAIN tasks')
    for tid in mf['task_ids']:
        validate_scenario_role(taskmap[tid],mf['scenario_ids'])
    # The R4-C library itself may not be changed without an explicit new audit.
    import counterfactual_r4c as r4c
    import r4c_core
    for path in (ROOT/'road_vrp7_step6c1_r4c/counterfactual_r4c.py',
                 ROOT/'road_vrp7_step6c1_r4c/r4c_core.py',
                 ROOT/'road_vrp7_step6c1_r4b/r4b_core.py'):
        require(path.is_file(),'Missing frozen solver/replay library: '+str(path))
    return mf,source_snaps,taskmap,road,OD,Objective,legal,v3,cfg,r4c,r4c_core


def build(a,ks,selected,prefix_report,prefix_manifest,cap_manifest):
    p=Path(a.prefix_dir);c=Path(a.capture_dir)
    src={
        'prefix_report':sha(p/'report.json'),
        'prefix_manifest':sha(p/'manifest.json'),
        'prefix_paired_rows':sha(p/'paired_rows.jsonl'),
        'prefix_excluded_windows':sha(p/'excluded_windows.json'),
        'capture_manifest':sha(c/'manifest.json'),
        'capture_provenance':sha(c/'provenance.json'),
        'capture_snapshots':sha(c/'snapshots.jsonl'),
        'capture_metrics':sha(c/'metrics.jsonl')}
    for key,path in {'frozen_solver':'road_vrp7_step5b_v3/run_experiment.py',
       'frozen_section':'road_vrp7_step5b_v3/section_v2.py',
       'frozen_r4a':'road_vrp7_step6c1_r4a/r4a_core.py',
       'frozen_r4b':'road_vrp7_step6c1_r4b/r4b_core.py',
       'r4c_run_one':'road_vrp7_step6c1_r4c/counterfactual_r4c.py',
       'r4c_core':'road_vrp7_step6c1_r4c/r4c_core.py',
       'r4h_collector':'road_vrp7_step6c1_r4h/collect_train_cf.py',
       'r4h_core':'road_vrp7_step6c1_r4h/r4h_core.py'}.items():
        src[key]=sha(ROOT/path)
    # Require actual prefix SHA pins for frozen ALNS source and exact R4B library.
    for current,source in [('frozen_solver','original_c3_solver'),('frozen_section','original_section'),
                           ('frozen_r4a','original_r4a_core'),('frozen_r4b','code_r4b_core')]:
        require(src[current]==prefix_manifest['source_sha256'][source],
                'Frozen code hash differs from precise R4G-Prefix proof: '+current)
    require(src['frozen_solver']==cap_manifest['solver_sha256'] and
            src['frozen_section']==cap_manifest['section_sha256'], 'Frozen train C3/Section modified')
    return {'stage':STAGE,'status':'CONFIGURED','data_role':'TRAIN_ONLY',
        'experiment':'FIRST_SECTION_ALT_THEN_OFFICIAL_C3_FIXED_K',
        'eval_semantics':'FIXED_K_NOT_REMAINING_30S',
        'horizons':list(ks),'action_count':a.action_count,
        'max_alternatives':a.max_alternatives,'sampler_seed':a.sampler_seed,
        'max_cases':a.max_cases,'max_snapshots_per_case':a.max_snapshots_per_case,
        'require_all_k':True,
        'selected_snapshot_ids':[list(key) for key,_,_ in selected],
        'source_sha256':src,'source_exact_comparisons':prefix_report['exact_matches'],
        'warnings':['Never fit a model on Validation/Test labels.',
          'SHAM and every K prefix must remain exact; alternatives do not preserve RNG after branching.',
          'Static Section candidates, posthoc Road.replay labels; no road-label leakage into action.',
          'K is fixed iterations not remaining 30-second wall time.',
          'Physical search states repeat across traffic scenarios; group related samples.']}


def case_choose(a,verified,snapmap,ks):
    return select_train_snapshots(verified,snapmap,ks,a.max_cases,a.max_snapshots_per_case,True)


def _validate_pack(pack,spec,plan,manifest,required_len):
    require(pack.get('stage')==STAGE and pack.get('unit_spec')==spec and pack.get('plan')==plan,
            'Unit identity/plan mismatch')
    require(pack.get('source_manifest_sha256')==manifest['source_sha256']['prefix_manifest'] and
            pack.get('source_snapshots_sha256')==manifest['source_sha256']['capture_snapshots'],
            'Unit provenance has changed')
    result=pack.get('result',{})
    require(result.get('arm')==spec['arm'] and result.get('horizon')==spec['horizon'] and
            snap_id(result.get('snapshot_spec',{}))==snap_id(spec), 'Unit result identity changed')
    require(len(result.get('events',[]))==required_len,'Unit is missing fixed-K events')
    require(result['first_original_c3_action_id']==plan['c3_identity'],'First C3 action differs from plan')
    return result


def execute(a):
    require(a.max_cases>=0 and a.max_snapshots_per_case>=0 and
            1<=a.max_alternatives<=a.action_count<=8 and a.max_new_units>=0,'Bad study parameters')
    ks=parse_ks(a.horizons)
    allow,smap,proof,proof_manifest,cap_manifest=verified_allowlist(a.prefix_dir,a.capture_dir,ks)
    selected=case_choose(a,allow,smap,ks)
    settings=build(a,ks,selected,proof,proof_manifest,cap_manifest)
    if a.mode=='plan':
        plans={'status':'ELIGIBLE_NOT_COUNTERFACTUAL_PROOF','selected_scenario_snapshots':len(selected),
               'selected_by_task':dict(sorted(__import__('collections').Counter(k[0] for k,_,_ in selected).items())),
               'requested_ks':ks,'number_of_arms_per_k':'2 controls + up to max_alternatives',
               'number_of_max_search_branches':len(selected)*(2+a.max_alternatives)*len(ks),
               'source_verified_comparisons':proof['exact_matches'],
               'source_capture_snapshots':proof['eligible_snapshots'],
               'source_distinct_physical_states':proof['source_audit']['source_distinct_physical_states'],
               'selected':[[*key,valid] for key,s,valid in selected]}
        print(json.dumps(plans,ensure_ascii=False,indent=2));return
    mf,source_snaps,taskmap,road,OD,Objective,legal,v3,cfg,r4c,core=preflight(a)
    require(mf['solver_sha256']==settings['source_sha256']['frozen_solver'] and
            mf['section_sha256']==settings['source_sha256']['frozen_section'], 'TRAIN preflight solver differs')
    require(sha(Path(a.capture_dir)/'snapshots.jsonl')==settings['source_sha256']['capture_snapshots'],
            'Snapshot source changed during preflight')
    from r4b_core import validate_checkpoint
    for key,s,_ in selected:
        validate_checkpoint(s,expected_max_iter=20000)
        require(all((*key,k) in allow for k in ks),'Unverified K requested')
    out=Path(a.output_dir)
    if out.exists():
        require(a.resume,'Use a NEW independent output directory; or RESUME=1')
        require(load(out/'manifest.json')==settings,'Resume source/plan/code hashes differ')
    else:
        require(not a.resume,'Cannot resume missing study')
        out.mkdir(parents=True,exist_ok=False);save(out/'manifest.json',settings)
    (out/'units').mkdir(exist_ok=True)
    new=0;reused=0;branches=[];labels=[];plans=[]
    for index,(key,s,_) in enumerate(selected,1):
        tid,sid,seed,stop=key
        task=taskmap[tid]
        sc=next(z for z in task['scenarios'] if z['scenario_id']==sid)
        od=OD(road,task,sc,step_min=int(cfg['od_step_min']),hours=float(cfg['od_hours']),cache_dir=a.od_cache_dir)
        obj=Objective(task,od,'static')
        plan=r4c.shadow_action(s,obj,a.action_count,a.sampler_seed,a.max_alternatives)
        c3=plan['c3_identity']
        options=[('C3',None),('SHAM_C3',None)]+[(z['name'],z) for z in plan['alternatives']]
        plans.append({'snapshot_id':list(key),'candidate_count':plan['candidate_count'],
                      'n_alternatives':len(plan['alternatives']),'c3_identity':c3})
        prev={}
        for k in ks:
            results={}
            for arm,opt in options:
                spec=core.branch_spec(s,k,arm)
                file=out/'units'/core.unit_name(spec)
                if file.exists():
                    pack=load(file);result=_validate_pack(pack,spec,plan,settings,k);reused+=1
                else:
                    if a.max_new_units and new>=a.max_new_units:
                        save(out/'partial.json',{'status':'PARTIAL','created':new,'reused':reused,
                              'next_unit_spec':spec,'source_manifest_sha256':settings['source_sha256']['prefix_manifest']})
                        print('R4H PARTIAL',new,'created, resume with same study config and RESUME=1',flush=True)
                        return
                    result=r4c.run_one(v3,task,sc,seed,obj,s,k,arm,opt,c3,road,legal,mf,
                                       plan['first_rng_after_original_C3_destroy_sha256'])
                    pack={'stage':STAGE,'unit_spec':spec,'plan':plan,'result':result,
                          'source_manifest_sha256':settings['source_sha256']['prefix_manifest'],
                          'source_snapshots_sha256':settings['source_sha256']['capture_snapshots']}
                    _validate_pack(pack,spec,plan,settings,k)
                    save(file,pack);new+=1
                if arm in prev:
                    require(result['events'][:len(prev[arm])]==prev[arm],
                            'Different K for same intervention does not have exact event prefix')
                prev[arm]=result['events']
                require(result['first_original_c3_action_id']==c3,'First real C3 action differs from plan')
                results[arm]=result
                if arm=='SHAM_C3':core.compare_sham(results['C3'],result)
                if arm not in ('C3','SHAM_C3'):
                    labels.append({'unit_spec':spec,'first_action_id':result['first_action_id'],
                                   'original_c3_action_id':c3,**paired_reward(results['C3'],result)})
                branches.append({'unit_spec':spec,'best_road_sec':result['road_best_sec'],
                                 'current_road_sec':result['road_current_sec'],
                                 'best_static_sec':result['static_best_sec'],
                                 'first_action_id':result['first_action_id'],
                                 'best_route_sha256':result['best_route_sha256']})
            print('R4H TRAIN',index,'/',len(selected),tid,sid,seed,'stop',stop,'K',k,
                  'alternatives',len(options)-2,'C3 Road',round(results['C3']['road_best_sec'],3),flush=True)
    s=report_metrics(labels,selected,ks)
    stability=horizon_stability(labels,branches,selected,ks)
    report={'stage':STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
            'n_branches':len(branches),'n_labels':len(labels),'n_sham_checks':len(selected)*len(ks),
            'n_cross_k_prefix_checks':sum((len(ks)-1)*(2+p['n_alternatives']) for p in plans),
            'new_units':new,'reused_units':reused,'horizon_stability':stability,**s,
            'source_prefix_exact_comparisons':proof['exact_matches'],
            'limitations':['Only TRAIN data, not independent model validation.',
                'One RNG trajectory per physical state; correlated across K/actions/scenarios.',
                'C3/SHAM is strict, K-step not remaining-wall-clock performance.',
                'No model was fitted; benefits refer only to evaluated alternatives.']}
    import math
    for name,arr in [('labels.jsonl',labels),('branch_summary.jsonl',branches),('plans.jsonl',plans)]:
        with (out/name).open('w',encoding='utf8') as f:
            for x in arr:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
    save(out/'report.json',report)
    lines=['# R4-H TRAIN首步Section连续搜索反事实','',
           f"状态：**COMPLETE**；来源精确证明：{proof['exact_matches']} / {proof['n_comparisons']}；",
           f"快照：{s['n_scenario_snapshots']}；不同物理搜索状态：{s['n_distinct_physical_states']}；",
           f"SHAM等价：{report['n_sham_checks']}；训练动作×K标签：{len(labels)}。",'',
           '| K | 动作标签 | 正/负/零 | Road均值(s) | 含正收益候选的物理状态 |',
           '|---:|---:|---:|---:|---:|']
    for k,v in s['per_horizon'].items():
        lines.append(f"| {k} | {v['n_actions']} | {v['positive']}/{v['negative']}/{v['zero']} | {v['mean_best_road_gain_sec']:+.3f} | {v['positive_oracle_distinct_physical_states']} |")
    lines+=['','## 长时域收益稳定性','',
        '| 区间 | 正转负动作 | 涉及物理状态数 | C3平均进步(s) | 替代分支平均进步(s) |',
        '|---|---:|---:|---:|---:|']
    for interval,v in stability.items():
        lines.append(f"| {interval.replace('_to_','→')} | {v['positive_to_negative']} | {v['positive_to_negative_physical_states']} | {v['mean_C3_progress_for_flips_sec']:+.2f} | {v['mean_alt_progress_for_flips_sec']:+.2f} |")
    lines+=['','仅作TRAIN标签采集与机制诊断；未训练模型、未进行独立Validation或30秒在线比较。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf8')
    if (out/'partial.json').exists(): (out/'partial.json').unlink()
    print('R4H COMPLETE',out,'new',new,'reused',reused,'labels',len(labels),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix-dir',required=True)
    p.add_argument('--capture-dir',required=True)
    p.add_argument('--output-dir')
    p.add_argument('--mode',choices=['plan','collect'],default='plan')
    p.add_argument('--model-source',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--horizons',default='50,200,500')
    p.add_argument('--max-cases',type=int,default=0)
    p.add_argument('--max-snapshots-per-case',type=int,default=0)
    p.add_argument('--action-count',type=int,default=8)
    p.add_argument('--max-alternatives',type=int,default=8)
    p.add_argument('--sampler-seed',type=int,default=20261019)
    p.add_argument('--max-new-units',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.mode=='collect':require(args.output_dir,'Output directory required')
    execute(args)
