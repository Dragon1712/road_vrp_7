#!/usr/bin/env python3
"""R4-F source-pinned long-horizon reversal / baseline-catch-up audit."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4f'))
from r4f_core import STAGE,require,sha,dump,audit_sources,audit_data,analyze,make_markdown


def run(args):
    out=Path(args.output_dir)
    require(not out.exists(),'Independent NEW output directory required')
    content,pins=audit_sources(args.r4e_dir,args.r4c_dir)
    lr,lm,ls,lb,ss,sb=content
    if args.require_full:
        require(lr['n_snapshots']==16 and lr['n_physical_states']==8,
                'Full study must cover 16 scenario snapshots and 8 physical starts')
    maps,unique,road=audit_data(lr,lm,ls,lb,ss,sb)
    results,transitions,oracles=analyze(maps,unique,road)
    require(results['n_scenario_snapshots']==lr['n_snapshots'],'Analyzed snapshot count does not match R4E')
    require(results['n_unique_physical_states']==lr['n_physical_states'],'Physical count mismatch')
    report={'status':'COMPLETE','stage':STAGE,'analysis':results,
        'source_shas':pins,'source_long_dir':str(args.r4e_dir),'source_short_dir':str(args.r4c_dir),
        'frozen_code_sha256':{name:sha(ROOT/'road_vrp7_step6c1_r4f'/name)
             for name in ('audit_r4f.py','r4f_core.py')},
        'role':'INSPECTED_JULY_VALIDATION_DEVELOPMENT_ONLY',
        'limitations':['Not a model training set, independent test or 30-second online result.',
          'Actions, K windows and repeated scenarios are correlated.',
          'One first-step RNG trajectory and tiny physical-state count.',
          'Oracle and horizon-best arms are HINDSIGHT controls, not deployable policies.']}
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'report.json',report)
    (out/'report.md').write_text(make_markdown(report),encoding='utf8')
    import json
    for f,data in [('horizon_transition_rows.jsonl',transitions),('snapshot_oracle.jsonl',oracles)]:
        with (out/f).open('w',encoding='utf8') as h:
            for x in data:h.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
    dump(out/'provenance.json',{'source_sha256':pins,'report_sha256':sha(out/'report.json'),
        'transition_rows_sha256':sha(out/'horizon_transition_rows.jsonl'),
        'snapshot_oracle_sha256':sha(out/'snapshot_oracle.jsonl')})
    print(make_markdown(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r4e-dir',required=True);p.add_argument('--r4c-dir',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--require-full',action='store_true')
    run(p.parse_args())
