#!/usr/bin/env python3
"""Audit R4-J partial/full R4-H units, without running ALNS or recomputing labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from r4k_core import (STAGE,CF_STAGE,KS,need,read,readl,hash_file,save,preflight,
                      parse_units,analyze,render,sid)


def run(a):
    output=Path(a.output_dir)
    need(not output.exists(),'Use an independent NEW output directory; old audit must remain immutable')
    m,src,selected,pins=preflight(a.cf_dir,a.prefix_dir,a.capture_dir)
    files,units,plans=parse_units(a.cf_dir,m,src,selected)
    result,incomplete,rewards,oracles,progress=analyze(selected,src,files,units,plans)
    report_path=Path(a.cf_dir)/'report.json'
    finished=report_path.is_file()
    if finished:
        original=read(report_path)
        need(original.get('stage')==CF_STAGE and original.get('status')=='COMPLETE' and
             original.get('data_role')=='TRAIN_ONLY','Original CF report not fully COMPLETE')
        need(result['incomplete_scenario_snapshots']==0,'Original CF report claims COMPLETE but checkpoints incomplete')
        need(int(original['n_branches'])==len(files),'CF branch count differs from units')
        need(int(original['n_labels'])==len(rewards),'CF label count differs from reconstructable full set')
        need(int(original['n_scenario_snapshots'])==len(selected),'CF snapshot count differs')
        original_labels=readl(Path(a.cf_dir)/'labels.jsonl')
        need(len(original_labels)==len(rewards),'Labels.jsonl count differs')
        label_by_key={(sid(z['unit_spec']),int(z['unit_spec']['horizon']),z['unit_spec']['arm']):z for z in original_labels}
        need(len(label_by_key)==len(original_labels),'Duplicated original CF labels')
        for z in rewards:
            k=(tuple(z['snapshot_id']),int(z['horizon']),z['arm'])
            need(k in label_by_key,'Original CF report omitted a reconstructed action')
            for f in ('road_best_gain_sec','road_current_gain_sec','static_best_gain_sec'):
                need(abs(z[f]-float(label_by_key[k][f]))<=1e-8,'Original CF label disagrees: '+f)
        need(int(original['n_sham_checks'])==3*len(selected),'Reported SHAM controls missing')
    if a.require_complete:
        need(finished and result['incomplete_scenario_snapshots']==0,'CF is PARTIAL: finish existing study first, do not fabricate full stats')
    report={'stage':STAGE,'status':'SOURCE_COMPLETE' if finished else 'PARTIAL_COVERAGE_ONLY',
           'data_role':'TRAIN_ONLY','source_commit_reviewed':'95b3fbf78166e8a9db2d6985b26e3ae3abc4f951',
           'analysis':result,
           'source_sha256':{**pins['hashes'], 'cf_manifest':hash_file(Path(a.cf_dir)/'manifest.json'),
                **({'cf_report':hash_file(report_path)} if finished else {})},
           'no_model_fit':True,'no_cf_execution':True}
    output.mkdir(parents=True,exist_ok=False)
    save(output/'report.json',report)
    (output/'report.md').write_text(render(report),encoding='utf-8')
    save(output/'missing_units.json',incomplete)
    for name,values in [('audited_labels.jsonl',rewards),('sampled_oracles.jsonl',oracles),('K_transitions.jsonl',progress)]:
        with (output/name).open('w',encoding='utf-8') as f:
            for z in values:f.write(json.dumps(z,ensure_ascii=False,allow_nan=False)+'\n')
    save(output/'provenance.json',{'stage':STAGE,'report_sha256':hash_file(output/'report.json'),
          'code_sha256':{'r4k_core.py':hash_file(Path(__file__).with_name('r4k_core.py')),
                          'audit_r4k.py':hash_file(__file__)},
          'source_sha256':report['source_sha256']})
    print(render(report),flush=True)
    print('R4K OUTPUT',output,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cf-dir','prefix-dir','capture-dir','output-dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--require-complete',action='store_true')
    run(parser.parse_args())
