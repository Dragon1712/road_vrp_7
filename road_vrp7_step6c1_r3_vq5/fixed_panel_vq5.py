#!/usr/bin/env python3
"""R3-VQ5: fixed 192-state evaluation, varying only outer-fold training prefixes.

Calls unchanged VQ3 build_fold, train/OOF/risk implementations; original TRAIN tasks only.
The H-step road value is an ex-post restart surrogate, not full online ALNS Q.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for f in ('road_vrp7_step6c1_r3_vq5','road_vrp7_step6c1_r3_vq3',
          'road_vrp7_step6c1_r3_vq2','road_vrp7_step6c1_r3_vq','road_vrp7_step6c1_r2'):
    p=str(ROOT/f)
    if p not in sys.path:sys.path.insert(0,p)
import train_vq,train_vq3
from vq_core import paired_features
from r2_policy import group_split
from vq2_core import one_step_rows, rows_for_selected, metrics_for_decisions
from vq3_core import POLICIES,STAGE as VQ3_STAGE
from vq5_core import (STAGE,DEFAULT_SIZES,sha256,read_json,write_json,check_balanced_prefix,
    hybrid_outer_selection,aggregate_metrics,anchor_filter,same_decision_rows)


def run(a):
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('Use an independent NEW VQ5 study directory')
    argsizes=tuple(map(int,a.sizes.split(',')))
    if a.trees<10 or a.seed<0:raise ValueError('Bad estimator config')
    config,_,X,B,Y,BY,valid,same,groups,lookup,panel,measured=train_vq.load(a)
    _,roles=group_split(groups,'task')
    tasks=check_balanced_prefix(panel,groups,lookup,roles,argsizes)
    dataset=Path(a.dataset_dir);collection=Path(a.collection_dir)
    all_one=one_step_rows(X,B,Y,BY,valid,same,groups,roles,set(tasks),paired_features)
    manifest={'stage':STAGE,'seed':a.seed,'trees':a.trees,'sizes':list(argsizes),
       'panel_n':len(panel),'horizons':list(config['horizons']),
       'collection_manifest_sha256':sha256(collection/'manifest.json'),
       'collection_rollouts_sha256':sha256(collection/'rollouts.jsonl'),
       'dataset_paired_sha256':sha256(dataset/'paired.npz'),
       'dataset_groups_sha256':sha256(dataset/'groups.jsonl'),
       'frozen_vq3_code_sha256':{p.name:sha256(p) for p in (
             Path(train_vq3.__file__),ROOT/'road_vrp7_step6c1_r3_vq3/vq3_core.py',
             ROOT/'road_vrp7_step6c1_r3_vq2/vq2_core.py',Path(train_vq.__file__))},
       'code_sha256':{p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('vq5_core.py'))},
       'source_role':'ORIGINAL_TRAIN_ONLY','evaluation':'FIXED FULL PANEL (all states of heldout task)',
       'limits':['No original validation/internal_test tasks used.','Four correlated original TRAIN task groups only.',
         'H-step label is C3 restart surrogate, not original ALNS continuation.']}
    folds={};macros={};anchor_macros={};all_decisions=[];all_candidates=[];anchor_info={}
    for n in argsizes:
        prefix=panel[:n];keyn=str(n);folds[keyn]={};macros[keyn]={};anchor_macros[keyn]={}
        for h in config['horizons']:
            kh=str(h);folds[keyn][kh]={}; rows_h=[];counts=[];anchor_by_task={}
            for task in tasks:
                selected,train_ids,eval_ids=hybrid_outer_selection(panel,prefix,groups,lookup,task)
                m=rows_for_selected(selected,groups,lookup,X,B,Y,BY,valid,same,measured,h,paired_features)
                dec,pred,details,diag,ranks=train_vq3.build_fold(
                    X=X,B=B,Y=Y,BY=BY,valid=valid,same=same,groups=groups,lookup=lookup,
                    selection=selected,measured=measured,h=h,holdout=task,roles=roles,
                    all_one=all_one,all_multi=m,seed=a.seed,trees=a.trees)
                if len(eval_ids)!=len(panel)//4 or len(train_ids)!=(n//4)*3:
                    raise AssertionError('Train/eval size or task isolation mismatch')
                per={name:metrics_for_decisions([d for d in dec if d['policy']==name]) for name in POLICIES}
                details.update({'fixed_panel_eval_n':len(eval_ids),'varying_training_n':len(train_ids),
                                'prefix_n':n,'prefix_per_task':n//4})
                folds[keyn][kh][task]={'counts':details,'metrics':per,'candidate_risk_diagnostics':diag,
                                       'pairwise_accuracy_including_C3':ranks}
                anchor_ids=set(panel[:argsizes[0]])
                anchor_by_task[task]={'metrics':{method:metrics_for_decisions(
                    [z for z in dec if z['policy']==method and z['decision_id'] in anchor_ids])
                    for method in POLICIES}}
                for d in dec:all_decisions.append({'train_size':n,**d})
                for p in pred:all_candidates.append({'train_size':n,**p})
                rows_h.extend(dec);counts.extend(eval_ids)
            if len(set(counts))!=len(panel) or len(rows_h)!=len(panel)*len(POLICIES):
                raise AssertionError('Fixed-panel heldout state coverage incomplete')
            macros[keyn][kh]=aggregate_metrics(folds[keyn][kh],POLICIES)
            anchor_macros[keyn][kh]=aggregate_metrics(anchor_by_task,POLICIES)
    expected=len(argsizes)*len(config['horizons'])*len(panel)*len(POLICIES)
    if len(all_decisions)!=expected:raise AssertionError('Unexpected decision count')
    if len({(x['train_size'],x['decision_id'],x['horizon'],x['policy']) for x in all_decisions})!=expected:
        raise AssertionError('Duplicate fixed-panel decision')
    anchor=panel[:argsizes[0]]
    if a.anchor_dir:
        baseline_dir=Path(a.anchor_dir)
        original=read_json(baseline_dir/'report.json')
        if original['stage']!=VQ3_STAGE or original['seed']!=a.seed or original['n_selected_states']!=argsizes[0]:
            raise ValueError('Wrong frozen VQ4-n48 identity')
        old=[json.loads(line) for line in (baseline_dir/'fold_decisions.jsonl').read_text().splitlines() if line.strip()]
        new=anchor_filter([d for d in all_decisions if d['train_size']==argsizes[0]],anchor)
        count=same_decision_rows(new,old)
        anchor_info={'status':'MATCH','n_exact_policy_decisions':count,
                     'original_report_sha256':sha256(baseline_dir/'report.json'),
                     'original_fold_decisions_sha256':sha256(baseline_dir/'fold_decisions.jsonl')}
    else:
        anchor_info={'status':'SKIPPED','note':'Set --anchor-dir to check strict n48 VQ4 reproducibility'}
    out.mkdir(parents=True,exist_ok=False)
    for name,rows in [('fold_decisions.jsonl',all_decisions),('candidate_predictions.jsonl',all_candidates)]:
        with (out/name).open('w',encoding='utf8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    write_json(out/'report.json',{'stage':STAGE,'status':'COMPLETE','seed':a.seed,'trees':a.trees,
       'sizes':list(argsizes),'panel_n':len(panel),'n_original_train_tasks':4,
       'horizons':config['horizons'],'policies':list(POLICIES),'macro_task_metrics':macros,
       'folds':folds,'anchor_panel_48_macro_task_metrics':anchor_macros,
       'anchor_48_verification':anchor_info,
       'design':'Identical heldout task state panel at every train size; only source training prefix changes',
       'warning':'Same 4 TRAIN tasks; not blind test; no exact ALNS continuation or 30-second claims'})
    write_json(out/'provenance.json',manifest)
    print('VQ5 COMPLETE',out,'decisions',len(all_decisions),'anchor',anchor_info['status'],flush=True)
    for n in argsizes:
        for h in config['horizons']:
            c=macros[str(n)][str(h)]
            print(f'n={n} H={h}',{x:round(c[x]['mean_gain_sec'],3) for x in
                ['E0_MATCHED','VQ_MEAN','RESID_OOF50','PAIR_RANK','E0_RISK35']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',default='runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1')
    p.add_argument('--collection-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trees',type=int,default=160)
    p.add_argument('--sizes',default='48,96,144,192')
    p.add_argument('--anchor-dir',help='VQ4 n48 seed-specific baseline for strict action-level reproduction')
    run(p.parse_args())
