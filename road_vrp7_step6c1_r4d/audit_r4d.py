#!/usr/bin/env python3
"""Read-only R4-C FULL-UNIT diagnostics; never runs ALNS, replay or fits a model."""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for pkg in ('road_vrp7_step6c1_r4d','road_vrp7_step6c1_r4c'):
    folder=str(ROOT/pkg)
    if folder not in sys.path:sys.path.insert(0,folder)
from r4d_core import (STAGE,need,read_json,read_jsonl,sha256,dump_json,spec_key,snap_key,
    validate_inputs,analyze_records,summarize,render_md)
from r4c_core import unit_name


def audit(args):
    p=Path(args.r4c_dir);out=Path(args.output_dir)
    need(not out.exists(),'Use a brand-new independent R4-D output directory')
    manifest,original,labels,branches,plans=validate_inputs(p,
        source_dir=args.r4c_source_dir,capture_dir=args.capture_dir,r4b_dir=args.r4b_dir)
    keys={spec_key(z['unit_spec']) for z in branches}
    need(len(keys)==len(branches),'Duplicate branch spec')
    full={};nbytes=0
    for n,key in enumerate(sorted(keys),1):
        unitfile=p/'units'/unit_name(dict(zip(('task_id','scenario_id','seed','stop_count','horizon','arm'),key)))
        need(unitfile.is_file(),'Missing R4-C full branch checkpoint: '+str(unitfile))
        wrapper=read_json(unitfile)
        need(wrapper.get('stage')==manifest['stage'],'Wrong unit stage')
        need(spec_key(wrapper['unit_spec'])==key,'Wrong checkpoint identity')
        need(wrapper['checkpoint_source_sha256']==manifest['r4a_snapshots_sha256'],'Mixed snapshot SHA')
        result=wrapper['result']
        need(wrapper['plan']['c3_identity']==result['first_original_c3_action_id'], 'Source plan C3 differs')
        arm=key[-1]
        if arm not in ('C3','SHAM_C3'):
            opts={x['name']:x for x in wrapper['plan']['alternatives']}
            need(arm in opts,'Candidate arm not in source plan')
            opt=opts[arm]
            need(result['first_action_id']==[opt['route'],opt['pos'],opt['size']],
                 'Candidate identity changed from plan')
        full[key]=result;nbytes+=unitfile.stat().st_size
        if n%150==0:print(f'Validated {n}/{len(keys)} full checkpoints',flush=True)
    rows,by_start=analyze_records(labels,branches,full)
    # Scope physical duplication analysis only to the R4-C selected 16 starts.
    selected={snap_key(z) for z in plans}
    snapshots=read_jsonl(Path(args.capture_dir)/'snapshots.jsonl') if args.capture_dir else []
    relevant=[s for s in snapshots if snap_key(s) in selected]
    if args.capture_dir:need(len(relevant)==len(selected),'Missing R4-A physical start snapshot')
    else:
        # No duplicate evidence without source capture. Still allow synthetic/offline report,
        # but label physical counts as unknown, not independent samples.
        relevant=[{'task_id':k[0],'scenario_id':k[1],'seed':k[2],'stop_count':k[3],
          'current_route_sha256':f'unverified:{k[1]}','best_route_sha256':f'unverified:{k[1]}',
          'rng_state_sha256':f'unverified:{k[1]}'} for k in sorted(selected)]
    report,snapshot_rows=summarize(rows,relevant)
    # The original R4-C report is a derived summary; independently recompute
    # and compare it instead of trusting a potentially stale or mixed report.
    if original.get('per_horizon'):
        for h, old in original['per_horizon'].items():
            need(h in report['per_horizon'], 'Unexpected horizon in source report')
            new=report['per_horizon'][h]
            for field in ('count','mean_road_best_gain_sec','positive_frac','negative_frac','zero_frac'):
                if field=='count':
                    need(int(old[field])==new['n_actions'], 'Original R4-C count mismatch K='+h)
                else:
                    values=[x['road_best_gain_sec'] for x in rows if str(x['horizon'])==h]
                    fresh={'mean_road_best_gain_sec':sum(values)/len(values),
                           'positive_frac':sum(v>1e-8 for v in values)/len(values),
                           'negative_frac':sum(v< -1e-8 for v in values)/len(values),
                           'zero_frac':sum(abs(v)<=1e-8 for v in values)/len(values)}[field]
                    need(abs(float(old[field])-fresh)<1e-8, 'Original R4-C report inconsistent: '+h+'/'+field)
    report['source_report_sha256']=sha256(p/'report.json')
    report['source_manifest_sha256']=sha256(p/'manifest.json')
    report['source_labels_sha256']=sha256(p/'counterfactual_labels.jsonl')
    report['source_branches_sha256']=sha256(p/'branch_summary.jsonl')
    report['source_plans_sha256']=sha256(p/'plans.jsonl')
    report['n_validated_full_units']=len(full)
    report['total_unit_bytes']=nbytes
    report['original_sham_checks']=original['n_sham_control_checks']
    report['data_role']=manifest['data_role']
    report['capture_physical_state_verified']=bool(args.capture_dir)
    out.mkdir(parents=True,exist_ok=False)
    for filename,data in [('candidate_funnel.jsonl',rows),('snapshot_oracle.jsonl',snapshot_rows)]:
        with (out/filename).open('w',encoding='utf8') as f:
            for r in data:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    dump_json(out/'report.json',report)
    (out/'report.md').write_text(render_md(report),encoding='utf8')
    dump_json(out/'provenance.json',{'stage':STAGE,'source_r4c_dir':str(p),
       'source_r4b_dir':str(args.r4b_dir) if args.r4b_dir else None,
       'source_r4a_capture_dir':str(args.capture_dir) if args.capture_dir else None,
       'source_file_sha256':{name:sha256(p/name) for name in ('report.json','manifest.json',
            'counterfactual_labels.jsonl','branch_summary.jsonl','plans.jsonl')},
       'code_sha256':{x.name:sha256(x) for x in [Path(__file__),Path(__file__).with_name('r4d_core.py')]},
       'no_new_search':True,'no_training':True})
    print(render_md(report),flush=True)
    print(f'R4D COMPLETE: {out} | full units={len(full)} | R4C SHAM verified',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r4c-dir',default='runs/chicago_july_v1/step6c1_r4c/r4c_pilot_16snaps_v1')
    p.add_argument('--r4c-source-dir',default='road_vrp7_step6c1_r4c')
    p.add_argument('--capture-dir',default='runs/chicago_july_v1/step6c1_r4a/r4a_pilot_20k_fix1_capture')
    p.add_argument('--r4b-dir',default='runs/chicago_july_v1/step6c1_r4b/r4b_exact_8cases_2snap_v1')
    p.add_argument('--output-dir',required=True)
    audit(p.parse_args())
