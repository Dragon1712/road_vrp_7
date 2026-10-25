#!/usr/bin/env python3
"""TRAIN-only posthoc source-pinned diagnostic; no ALNS/road reruns."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4i'))
from r4i_core import (STAGE,sha,save,source_audit,validate_dataset,group_report,
                       records_from_dataset,decision_audit,to_markdown)

def run(a):
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('Must use independent new output directory: '+str(out))
    hr,hm,labs,branches,plans,snaps,pins=source_audit(a.r4h_dir,a.capture_dir,a.prefix_dir)
    selected,smap,labmap,features=validate_dataset(hr,hm,labs,branches,plans,snaps)
    signal=group_report(labs,selected,smap,labmap)
    records=records_from_dataset(selected,smap,labmap,features)
    folds,details=decision_audit(records)
    report={'stage':STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
       'source_provenance_sha256':pins,'source_git_commit':'ef4de64795fe9b3e404552a431757a7d0d2b5a21',
       'signal':signal,'n_action_groups':len(records),
       'models':'regularized linear utility: K500 and min(K200,K500), fixed 4 abstain margins',
       'training_group_weighting':'equal weight per distinct physical state',
       'heldout_split':'LEAVE_ONE_TRAIN_TASK_OUT; EXPLORATORY ONLY',
       'folds':folds,
       'limitations':['Only Train label diagnostics; NOT fresh independent validation/test.',
       'K-step Road.replay counterfactuals not remaining 30s wall-clock performance.',
       'Posthoc oracle depends on future search, never available at decision time.',
       'Small number of positive physical states means no robust deployment guarantee.',
       'Threshold sweep is pre-registered diagnostic; do not pick a winner on heldout task results.',
       'Physical and scenario repeats, actions and K are correlated.']}
    out.mkdir(parents=True,exist_ok=False)
    save(out/'report.json',report)
    (out/'report.md').write_text(to_markdown(report),encoding='utf8')
    with (out/'fold_predictions.jsonl').open('w',encoding='utf8') as f:
        for row in details:
            # No raw source snapshots or raw RNG states written; only selected
            # action and ground truth gain plus physical group identity.
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    save(out/'provenance.json',{'source_sha256':pins,'report_sha256':sha(out/'report.json'),
         'fold_predictions_sha256':sha(out/'fold_predictions.jsonl'),
         'code_sha256':{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('r4i_core.py'))}})
    print(to_markdown(report),flush=True)
    print('R4I COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r4h-dir',default='runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1')
    p.add_argument('--capture-dir',default='runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture')
    p.add_argument('--prefix-dir',default='runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1')
    p.add_argument('--output-dir',required=True)
    run(p.parse_args())
