#!/usr/bin/env python3
"""Create an entirely independent Step5-C from frozen Step5-B v3 code.

The installer refuses ambiguous anchors and never modifies v3 itself.
"""
from __future__ import annotations
import argparse
import hashlib
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DEST=ROOT/'road_vrp7_step5c'
SRC=ROOT/'road_vrp7_step5b_v3'


def replace_one(src,old,new):
    num=src.count(old)
    if num!=1:raise ValueError(f'Expected exactly one anchored match, found {num}: {old[:110]}')
    return src.replace(old,new,1)


def build_runner(s):
    # Every new variant retains the v3 identical time-dependent OD and Top-K Section operators.
    s=replace_one(s,"from snapshots_v3 import BestTrace,parse_times", "from snapshots_v3 import BestTrace,parse_times\nfrom selectors_step5c import make_step5c_selector")
    s=replace_one(s,"'U0_gain','U1_walltime','U2_calibrated')\nKEY=", "'U0_gain','U1_walltime','U2_calibrated',\n          'C7_alpha_ucb','C8_thompson','C9_lrw')\nKEY=")
    s=replace_one(s,"ucb_c=.35,ucb_epsilon=.08,anneal_iterations=None,cost_profile=None):", "ucb_c=.35,ucb_epsilon=.08,anneal_iterations=None,cost_profile=None,lrw_profile=None):")
    s=replace_one(s,"elif name in ('C3_topk','C4_time_ucb','C5_gated','C6_shuffled','U0_gain','U1_walltime','U2_calibrated'):",
                    "elif name in ('C3_topk','C4_time_ucb','C5_gated','C6_shuffled','U0_gain','U1_walltime','U2_calibrated','C7_alpha_ucb','C8_thompson','C9_lrw'):")
    s=replace_one(s,"'U0_gain':'static','U1_walltime':'static','U2_calibrated':'static'}[name]",
                    "'U0_gain':'static','U1_walltime':'static','U2_calibrated':'static',\n                      'C7_alpha_ucb':'static','C8_thompson':'static','C9_lrw':'static'}[name]")
    s=replace_one(s,"'U0_gain','U1_walltime','U2_calibrated'))\n                if reason!=", "'U0_gain','U1_walltime','U2_calibrated','C7_alpha_ucb','C8_thompson','C9_lrw'))\n                if reason!=")
    s=replace_one(s,"awareness=name in ('C4_time_ucb','C5_gated','C6_shuffled','U0_gain','U1_walltime','U2_calibrated')\n    if name in ('U0_gain'",
                    "awareness=name in ('C4_time_ucb','C5_gated','C6_shuffled','U0_gain','U1_walltime','U2_calibrated','C7_alpha_ucb','C8_thompson','C9_lrw')\n    if name in ('C7_alpha_ucb','C8_thompson','C9_lrw'):\n        selector=make_step5c_selector(name,seed,lrw_profile,len(ds),2)\n    elif name in ('U0_gain'")
    s=replace_one(s,"    road=Road(args.cost_dir)",
                    "    needs_lrw='C9_lrw' in variants\n    lrw_path=Path(args.lrw_profile) if args.lrw_profile else None\n    if needs_lrw and (lrw_path is None or not lrw_path.is_file()):\n        raise ValueError('C9_lrw requires frozen --lrw-profile. Run learn_lrw.py on prior development logs.')\n    lrw_profile=json.loads(lrw_path.read_text(encoding='utf-8')) if needs_lrw else None\n    lrw_hash=sha256(lrw_path) if needs_lrw else None\n    road=Road(args.cost_dir)")
    s=replace_one(s,"Path(__file__).with_name('calibrate_cost.py'),Path(__file__).with_name('analyze_v3.py'),",
                    "Path(__file__).with_name('calibrate_cost.py'),Path(__file__).with_name('analyze_step5c.py'),\n            Path(__file__).with_name('selectors_step5c.py'),Path(__file__).with_name('learn_lrw.py'),")
    s=replace_one(s,"cfg={'stage':'step5b-v3'", "cfg={'stage':'step5c-v1'")
    s=replace_one(s,"'calibration_sha256':profile_hash,'od_step_min':args.od_step_min,",
                    "'calibration_sha256':profile_hash,'lrw_profile_file':str(lrw_path) if needs_lrw else None,\n         'lrw_profile_sha256':lrw_hash,'od_step_min':args.od_step_min,")
    s=replace_one(s,"anneal_iterations=args.anneal_iterations,cost_profile=cost_profile)",
                    "anneal_iterations=args.anneal_iterations,cost_profile=cost_profile,lrw_profile=lrw_profile)")
    s=replace_one(s,"    p.add_argument('--calibration-file',default='')",
                    "    p.add_argument('--calibration-file',default='')\n    p.add_argument('--lrw-profile',default='')")
    return s


def build_analyzer(s):
    s=replace_one(s,"cfg.get('stage')!='step5b-v3'", "cfg.get('stage')!='step5c-v1'")
    s=replace_one(s,"Not a completed step5b-v3 run", "Not a completed step5c-v1 run")
    old="for new,base_solver in [('C3_topk','C0_baseline'),('U0_gain','C3_topk'),\n                                    ('U1_walltime','C3_topk'),('U2_calibrated','C3_topk'),\n                                    ('U1_walltime','U0_gain'),('U2_calibrated','U0_gain'),\n                                    ('U2_calibrated','U1_walltime')]:"
    new="for new,base_solver in [('U2_calibrated','C3_topk'),('C7_alpha_ucb','C3_topk'),\n                                    ('C8_thompson','C3_topk'),('C9_lrw','C3_topk'),\n                                    ('C9_lrw','C7_alpha_ucb'),('C9_lrw','C8_thompson')]:"
    s=replace_one(s,old,new)
    s=replace_one(s,"'data_role':'July development',", "'data_role':'July development, not an independent test',")
    return s


def install(root=ROOT):
    root=Path(root);src=root/'road_vrp7_step5b_v3'; dest=root/'road_vrp7_step5c'
    if dest.exists():raise FileExistsError(f'{dest} exists. Refusing overwrite.')
    if not (src/'run_experiment.py').is_file():raise FileNotFoundError('v3 runner missing; checkout and verify main branch')
    # Verify the exact v3 build that produced the user's frozen July Pilot.
    expected_sha={
      'run_experiment.py':'1d77b5266bd0481d2a5231311efdc83f6dd21a84c40a23e1386f2e7e90086ea1',
      'analyze_v3.py':'52a2383fe5457207c513b6a85c0b87f240af4bf3cc14990e96269d4aa334274b',
    }
    for filename, expected in expected_sha.items():
        got=hashlib.sha256((src/filename).read_bytes()).hexdigest()
        if got!=expected:raise ValueError(f'Frozen v3 SHA mismatch {filename}: {got} != {expected}')
    # All transformations BEFORE writing any destination files.
    runner=build_runner((src/'run_experiment.py').read_text(encoding='utf-8'))
    analyzer=build_analyzer((src/'analyze_v3.py').read_text(encoding='utf-8'))
    keep=['section_v2.py','time_selector.py','selectors_v3.py','snapshots_v3.py','calibrate_cost.py']
    missing=[x for x in keep if not (src/x).is_file()]
    if missing:raise FileNotFoundError(f'v3 files missing: {missing}')
    dest.mkdir()
    for name in keep:shutil.copy2(src/name,dest/name)
    for name in ('selectors_step5c.py','learn_lrw.py','install_step5c.py','run_step5c.sh'):
        shutil.copy2(Path(__file__).with_name(name),dest/name)
    (dest/'tests').mkdir()
    shutil.copy2(Path(__file__).parent/'tests'/'test_selectors_step5c.py',dest/'tests'/'test_selectors_step5c.py')
    (dest/'run_experiment.py').write_text(runner,encoding='utf-8')
    (dest/'analyze_step5c.py').write_text(analyzer,encoding='utf-8')
    print('PASS Step5-C isolated code created at',dest)
    return dest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT))
    a=p.parse_args();install(a.root)
