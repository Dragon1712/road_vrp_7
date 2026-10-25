#!/usr/bin/env python3
"""R4-N frozen Train label study; strictly read-only input and fail-closed output."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import r4n_core as c


def run(a):
    paths={name:getattr(a,name) for name in ('r4l_result','old_cf','old_capture','old_prefix',
          'new_r4k','new_cf','new_capture','new_prefix')}
    records,pins=c.r4m.source_preflight(paths)
    source=Path(a.r4m_result)
    report=c.read(source/'report.json');prov=c.read(source/'provenance.json')
    c.need(report.get('stage')==c.r4m.STAGE and report.get('status')=='COMPLETE' and
           report.get('data_role')=='TRAIN_ONLY' and report.get('n_configs')==120,
           'R4-M report not complete/train/expected 120')
    c.need(prov.get('stage')==c.r4m.STAGE and prov.get('report_sha256')==c.sha(source/'report.json')
           and prov.get('predictions_sha256')==c.sha(source/'fold_predictions.jsonl'),
           'R4-M file SHA mismatch')
    c.need(report.get('reference_report_sha256')==pins['r4l_report_sha256'] and
           prov['source_sha256']==pins,'R4-M and R4-L upstream SHA pins disagree')
    for file,h in prov['code_sha256'].items():
        orig=c.ROOT/'road_vrp7_step6c1_r4m'/file
        c.need(orig.is_file() and c.sha(orig)==h,'Changed frozen R4-M code '+file)
    c.need(len(records)==1572 and len(c.group(records))==200,'Unexpected Train grid')
    print('R4N PREFLIGHT PASS: 1572 candidate actions, 200 scenario snapshots, 6 heldout folds, all inputs pinned',flush=True)
    if a.mode=='preflight':return
    out=Path(a.output_dir)
    c.need(not out.exists(),'Use a NEW independent STUDY_NAME, existing results are immutable')
    folds,details=c.evaluate(records)
    coarse=len({tuple(r['physical'][:-1]) for r in records})
    result={'stage':c.STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
      'eval_semantics':'OFFLINE_FIXED_K_ONLY_NOT_30_SECOND_ONLINE',
      'data_summary':{'actions':len(records),'snapshots':len(c.group(records)),'coarse_starts':coarse},
      'protocol':{'splits':'4 TRAIN task-LOTO + 2 seed-pair transfer',
          'features':['R4I_full','R4I_geometry'],
          'shuffle_only_outer_training':True,
          'pairwise_target':'within-snapshot K500 gain differences, no absolute score calibration',
          'policies':list(c.POLICIES),'oracle_conditioned_policies':list(c.ORACLE_POLICIES),
          'rank_fit_regularization':1.0,'gate_model':'frozen R4L Ridge K500, threshold 0; Section <=2000 only'},
      'sources':{**pins,'r4m_report_sha256':c.sha(source/'report.json'),
                 'r4m_provenance_sha256':c.sha(source/'provenance.json')},
      'folds':folds,'limitations':['TRAIN only, 4 tasks: exploratory','Oracle-gated rows are not deployable',
             'No real 30-second online ALNS and no Validation/Test',
             'Near-repeated traffic scenarios share coarse starts',
             'No heldout threshold tuning']}
    out.mkdir(parents=True,exist_ok=False)
    c.save(out/'report.json',result)
    (out/'report.md').write_text(c.markdown(result),encoding='utf8')
    with (out/'fold_predictions.jsonl').open('w',encoding='utf8') as f:
        for z in details:f.write(json.dumps(z,ensure_ascii=False,allow_nan=False)+'\n')
    c.save(out/'provenance.json',{'stage':c.STAGE,'report_sha256':c.sha(out/'report.json'),
          'fold_predictions_sha256':c.sha(out/'fold_predictions.jsonl'),
          'source_sha256':result['sources'],
          'code_sha256':{'r4n_core.py':c.sha(Path(__file__).with_name('r4n_core.py')),
                         'probe_r4n.py':c.sha(__file__)}})
    print('R4N COMPLETE',out,'folds',len(folds),'decisions',len(details),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['preflight','probe'],default='probe')
    ap.add_argument('--r4m-result',default='runs/chicago_july_v1/step6c1_r4m/r4m_nested_risk_train_v1')
    ap.add_argument('--r4l-result',default='runs/chicago_july_v1/step6c1_r4l/r4l_seed_transfer_loto_train_v1')
    ap.add_argument('--old-cf',default='runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1')
    ap.add_argument('--old-capture',default='runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture')
    ap.add_argument('--old-prefix',default='runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1')
    ap.add_argument('--new-r4k',default='runs/chicago_july_v1/step6c1_r4k/r4k_full_128audited_v1')
    ap.add_argument('--new-cf',default='runs/chicago_july_v1/step6c1_r4h/r4j_train_early_s62s63_v1_cf_full')
    ap.add_argument('--new-capture',default='runs/chicago_july_v1/step6c1_r4g/r4j_train_early_s62s63_v1_capture')
    ap.add_argument('--new-prefix',default='runs/chicago_july_v1/step6c1_r4g_prefix/r4j_train_early_s62s63_v1_prefix_full')
    ap.add_argument('--output-dir')
    a=ap.parse_args()
    if a.mode=='probe' and not a.output_dir:ap.error('--output-dir required')
    run(a)

if __name__=='__main__':main()
