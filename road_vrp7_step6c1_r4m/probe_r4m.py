#!/usr/bin/env python3
"""Frozen-source R4-M Train-only nested risk calibration; no ALNS execution."""
import argparse
from pathlib import Path
from r4m_core import STAGE, ROOT, need, sha, save, source_preflight, evaluate, report_md
import r4m_core as core


def run(a):
    paths={'r4l_result':a.r4l_result,
       'old_cf':a.old_cf,'old_capture':a.old_capture,'old_prefix':a.old_prefix,
       'new_r4k':a.new_r4k,'new_cf':a.new_cf,'new_capture':a.new_capture,
       'new_prefix':a.new_prefix}
    if a.mode=='preflight':
        rec,pins=source_preflight(paths)
        print('R4M SOURCE PASS: candidates',len(rec),'snapshots',len({r['snap'] for r in rec}),
              'LOTO+SEED_TRANSFER=6; all original R4-L provenance checked')
        return
    out=Path(a.output_dir)
    need(not out.exists(),'Choose a new independent STUDY_NAME; outputs are immutable')
    rec,pins=source_preflight(paths)
    folds,decisions=evaluate(rec)
    result={'stage':STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
       'eval_semantics':'OFFLINE_FIXED_K_NOT_30S',
       'n_candidates':len(rec),'n_snapshots':len({r['snap'] for r in rec}),
       'n_configs':len(folds),'calibration':'inner leave-one-TRAIN-task-out; upper empirical overprediction quantiles',
       'pre_registered_policies':list(core.POLICIES),
       'fail_closed_min_inner_samples':5,'fail_closed_min_inner_tasks':2,
       'reference_report_sha256':pins['r4l_report_sha256'],'results':folds,
       'limitations':['Train-internal exploratory task/seed transfers only',
          'Not conformal or distribution-free guarantee','No fresh Validation/Test labels',
          'No inference timing, no 30-second ALNS online integration']}
    out.mkdir(parents=True,exist_ok=False)
    save(out/'report.json',result)
    (out/'report.md').write_text(report_md(result),encoding='utf8')
    with (out/'fold_predictions.jsonl').open('w',encoding='utf8') as f:
        import json
        for x in decisions:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
    save(out/'provenance.json',{'stage':STAGE,
       'report_sha256':sha(out/'report.json'),'predictions_sha256':sha(out/'fold_predictions.jsonl'),
       'code_sha256':{'r4m_core.py':sha(ROOT/'road_vrp7_step6c1_r4m/r4m_core.py'),
                      'probe_r4m.py':sha(__file__)},'source_sha256':pins})
    print('R4M COMPLETE',out,'configs',len(folds),'heldout records',len(decisions),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','probe'),default='probe')
    p.add_argument('--r4l-result',default='runs/chicago_july_v1/step6c1_r4l/r4l_seed_transfer_loto_train_v1')
    p.add_argument('--old-cf',default='runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1')
    p.add_argument('--old-capture',default='runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture')
    p.add_argument('--old-prefix',default='runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1')
    p.add_argument('--new-r4k',default='runs/chicago_july_v1/step6c1_r4k/r4k_full_128audited_v1')
    p.add_argument('--new-cf',default='runs/chicago_july_v1/step6c1_r4h/r4j_train_early_s62s63_v1_cf_full')
    p.add_argument('--new-capture',default='runs/chicago_july_v1/step6c1_r4g/r4j_train_early_s62s63_v1_capture')
    p.add_argument('--new-prefix',default='runs/chicago_july_v1/step6c1_r4g_prefix/r4j_train_early_s62s63_v1_prefix_full')
    p.add_argument('--output-dir')
    a=p.parse_args()
    if a.mode=='probe' and not a.output_dir:p.error('--output-dir required for probe')
    run(a)
