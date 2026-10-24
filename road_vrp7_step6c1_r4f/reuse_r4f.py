#!/usr/bin/env python3
"""Safely reuse existing R4-E 8-snapshot unit checkpoints in a full 16-snapshot study.

Requires target R4-E manifest already initialized by collect-pilot MAX_NEW_UNITS=1,
and source/target to have identical official solver/proofs/policy configurations.
The R4-E collector itself revalidates plan, provenance, SHAM, K50 and prefix.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4f'))
sys.path.insert(0,str(ROOT/'road_vrp7_step6c1_r4c'))
from r4f_core import require,sha,read,dump
from r4c_core import unit_name

PINNED=('stage','source_capture_sha256','source_r4b_sha256','r4c_manifest_sha256',
        'r4c_report_sha256','long_proof_report_sha256','long_proof_rows_sha256',
        'horizons','max_snapshots_per_run','action_count','max_alternatives',
        'sampler_seed','code_sha256','evaluation','data_role')


def run(args):
    src,dst=Path(args.source_dir),Path(args.target_dir)
    require(src.resolve()!=dst.resolve(),'Cannot reuse into the same directory')
    sm,tm=read(src/'manifest.json'),read(dst/'manifest.json')
    require(sm.get('status')=='COMPLETE' and tm.get('status')=='RUNNING','Source must complete; target initialized and running')
    for k in PINNED:require(k in sm and sm[k]==tm.get(k),'Reuse pin/config differs: '+k)
    require(sm.get('max_cases')==4 and tm.get('max_cases')==0,'Expect 8-case source / 16-case target')
    require(len(sm['selected_snapshots'])==8 and len(tm['selected_snapshots'])==16,'Unexpected sample counts')
    require(tm['selected_snapshots'][:8]==sm['selected_snapshots'],'Full target does not embed exact same 8 source starts')
    source_units=sorted((src/'units').glob('*.json'))
    require(len(source_units)==read(src/'report.json')['n_branch_results'],'Missing source units')
    dest_units=dst/'units';require(dest_units.is_dir(),'Initialize target with MAX_NEW_UNITS=1 first')
    copied=present=0;specs=set()
    for f in source_units:
        pack=read(f)
        spec=pack['unit_spec']
        require(f.name==unit_name(spec),'Source filename and unit_spec digest inconsistent')
        key=(spec['task_id'],spec['scenario_id'],int(spec['seed']),int(spec['stop_count']))
        require(list(key) in sm['selected_snapshots'],'Source has out-of-scope checkpoint')
        require(pack.get('stage')==sm['stage'] and pack.get('result',{}).get('arm')==spec['arm'],
                'Source result identity mismatch')
        require(pack.get('r4c_manifest_sha256')==sm['r4c_manifest_sha256'] and
                pack.get('long_proof_rows_sha256')==sm['long_proof_rows_sha256'],'Source unit provenance mismatch')
        require(spec['horizon'] in sm['horizons'],'Source unit horizon is unsupported')
        require(f.name not in specs,'Duplicate source unit digest')
        specs.add(f.name)
        target=dest_units/f.name
        if target.exists():
            require(sha(f)==sha(target),'Existing target checkpoint differs from source')
            present+=1
        else:
            shutil.copyfile(f,target)
            require(sha(f)==sha(target),'Copied checkpoint digest mismatch')
            copied+=1
    # Only metadata; R4-E will still validate full SHAM, original 50-step prefix,
    # complete plan equality and inter-horizon traces as it resumes.
    dump(dst/'r4f_checkpoint_reuse.json',{'source_dir':str(src),'source_manifest_sha256':sha(src/'manifest.json'),
         'target_manifest_sha256':sha(dst/'manifest.json'),'reused_copied':copied,'already_present':present,
         'source_unit_count':len(source_units),'note':'All reused units must pass the frozen R4-E resume revalidation.'})
    print('R4F REUSE VERIFIED',copied,'copied and',present,'already existed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',required=True);p.add_argument('--target-dir',required=True)
    run(p.parse_args())
