#!/usr/bin/env python3
"""Read-only R4-J evidence stages and label density comparison."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4j'))
from r4j_core import (verify_capture,verify_prefix,verify_cf,analyze,markdown,
                       digest,write,ensure,STAGE)

def run(a):
    c=verify_capture(a.capture_dir,a.plain_dir,a.audit_dir,tuple(map(int,a.forbid_seeds.split(','))))
    print('R4J CAPTURE',len(c['snapshots']),'snapshots',c['audit']['strict_pass'],'/',c['audit']['pairs'],
          '20k strong pairs; route+Road:',c['audit']['route_road_equal'],'/',c['audit']['pairs'],flush=True)
    if a.phase=='capture':return
    p=verify_prefix(a.prefix_dir,c,require_complete=(a.phase!='prefix-smoke'))
    print('R4J PREFIX',p['report']['exact_matches'],'/',p['report']['n_comparisons'],
          'cases',p['report']['n_cases'],flush=True)
    if a.phase in ('prefix','prefix-smoke'):return
    cf=verify_cf(a.cf_dir,c,p)
    report=analyze(cf,c,a.old_cf_dir)
    report['source_sha256']={**c['pins'],**p['pins'],**cf['pins']}
    report['code_sha256']={f:digest(Path(__file__).with_name(f)) for f in ('audit_r4j.py','r4j_core.py')}
    out=Path(a.output_dir)
    ensure(not out.exists(),'Audit output must be a NEW independent directory')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'report.json',report)
    (out/'report.md').write_text(markdown(report),encoding='utf8')
    write(out/'provenance.json',{'stage':STAGE,'source_sha256':report['source_sha256'],
                                  'report_sha256':digest(out/'report.json')})
    print(markdown(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['capture','prefix-smoke','prefix','cf'],default='cf')
    ap.add_argument('--capture-dir',required=True)
    ap.add_argument('--plain-dir',required=True)
    ap.add_argument('--audit-dir',required=True)
    ap.add_argument('--prefix-dir')
    ap.add_argument('--cf-dir')
    ap.add_argument('--old-cf-dir')
    ap.add_argument('--output-dir')
    ap.add_argument('--forbid-seeds',default='60,61')
    a=ap.parse_args()
    if a.phase in ('prefix','prefix-smoke','cf'):ap.error('Use --prefix-dir') if not a.prefix_dir else None
    if a.phase=='cf':
        if not (a.cf_dir and a.output_dir):ap.error('Require --cf-dir and --output-dir for cf')
    run(a)
