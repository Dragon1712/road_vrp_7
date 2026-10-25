#!/usr/bin/env python3
"""Audit original/new frozen Train counterfactuals and run grouped ridge baselines."""
from __future__ import annotations
import argparse
from pathlib import Path
from r4l_core import (STAGE,read_old,read_new,need,make_splits,evaluate,make_summary,
                      markdown,sha,write,dump_rows)
import r4l_core as core

def execute(a):
    out=Path(a.output_dir)
    need(not out.exists(),'Use a NEW independent output directory; never overwrite old results')
    old,op=read_old(a.old_cf_dir,a.old_capture_dir,a.old_prefix_dir)
    new,np=read_new(a.new_r4k_dir,a.new_capture_dir,a.new_cf_dir,a.new_prefix_dir)
    need({r['snap'] for r in old}.isdisjoint({r['snap'] for r in new}),
         'Old/new snapshots overlap! No leakage allowed')
    records=old+new
    splits=make_splits(records)
    need(len(splits)==6,'Expected four LOTO and two seed-pair transfers')
    folds,details=evaluate(records)
    result={'stage':STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
        'eval_semantics':'OFFLINE_FIXED_K_COUNTERFACTUAL_NOT_30S_ONLINE',
        'source_git_commit_reviewed':'4f70819387a00a2721c9b84ba8dc70fb94d18a3f',
        'summary':make_summary(old,new,folds),
        'protocol':{'splits':[{'name':name,'type':typ,'training_actions':len(tr),
                     'heldout_actions':len(te)} for name,tr,te,typ in splits],
           'ridge_regularization':100.,'thresholds_sec':[0.,5.,15.],
           'targets':['k500','robust_min_k200_k500'],
           'features':['full_R4I','geometry_only_R4I'],
           'gates':['none','early2000'],'shuffled_label_negative_control':True,
           'score_selection':'one maximum predicted alternative per snapshot; otherwise C3'},
        'sources':{'original':op,'new':np},'folds':folds}
    out.mkdir(parents=True,exist_ok=False)
    write(out/'report.json',result)
    (out/'report.md').write_text(markdown(result,folds),encoding='utf8')
    dump_rows(out/'fold_predictions.jsonl',details)
    write(out/'provenance.json',{'stage':STAGE,
            'report_sha256':sha(out/'report.json'),
            'fold_predictions_sha256':sha(out/'fold_predictions.jsonl'),
            'code_sha256':{'r4l_core.py':sha(Path(__file__).with_name('r4l_core.py')),
                           'probe_r4l.py':sha(__file__),
                           'r4i_core.py':sha(core.ROOT/'road_vrp7_step6c1_r4i/r4i_core.py'),
                           'r4k_core.py':sha(core.ROOT/'road_vrp7_step6c1_r4k/r4k_core.py')},
            'source_sha256':{'old':op['source_sha256'],'new':np['source_sha256']}})
    print(markdown(result,folds),flush=True)
    print('R4-L result:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-cf-dir',default='runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1')
    p.add_argument('--old-capture-dir',default='runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture')
    p.add_argument('--old-prefix-dir',default='runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1')
    p.add_argument('--new-r4k-dir',default='runs/chicago_july_v1/step6c1_r4k/r4k_full_128audited_v1')
    p.add_argument('--new-cf-dir',default='runs/chicago_july_v1/step6c1_r4h/r4j_train_early_s62s63_v1_cf_full')
    p.add_argument('--new-capture-dir',default='runs/chicago_july_v1/step6c1_r4g/r4j_train_early_s62s63_v1_capture')
    p.add_argument('--new-prefix-dir',default='runs/chicago_july_v1/step6c1_r4g_prefix/r4j_train_early_s62s63_v1_prefix_full')
    p.add_argument('--output-dir',required=True)
    execute(p.parse_args())
