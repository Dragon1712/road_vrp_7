#!/usr/bin/env python3
"""Verify TRAIN-only ALNS snapshot continuation within true 30s source prefix.

Original and restored references use frozen official ALNS v7.0.0/R4B.
The *new* uninterrupted official C3 reference is allowed longer wall time,
not used as an online performance claim. Each captured checkpoint must match
that fresh reference at its exact BEFORE_SELECT_AND_DESTROY boundary.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for module in ('road_vrp7_step6c1_r4g_prefix', 'road_vrp7_step6c1_r4g',
               'road_vrp7_step6c1_r4b', 'road_vrp7_step6c1_r4a',
               'road_vrp7_step5b_v3', 'road_vrp7_step6b2_r', 'road_vrp7_stage3'):
    path = str(ROOT / module)
    if path not in sys.path:
        sys.path.insert(0, path)
from prefix_core import (STAGE, require, sha256, read_json, write_json, parse_horizons,
                         plan_cases, selected_manifest, validate_unit, compare_prefix_events,
                         summary, expected_identity_rows)


def preflight(a):
    # The existing R4G verifier already guards the frozen C3 source hash,
    # Section, R4-A observer, TRAIN role, raw road files, and provenance.
    import verify_train_r4g as r4g
    from r4g_core import load_capture, paired_audit
    mf, snapshots, taskmap, road, OD, Objective, legal, v3, cfg = r4g.preflight(a)
    p_mf, plain, no_snapshots = load_capture(a.plain_dir, 'plain')
    c_mf, capture, _ = load_capture(a.capture_dir, 'capture')
    audit = paired_audit((p_mf, plain, no_snapshots), (c_mf, capture, snapshots))
    require(audit['route_road_equal'] == audit['pairs'] and audit['pairs'] == len(capture),
            'Plain/Capture final routes/costs differ; cannot claim captured stable source')
    require(audit['n_failed_pairs'] == audit['wall_limited_pairs'],
            'Unexpected non-wall-capped source noninterference differences')
    return (mf, plain, capture, snapshots, taskmap, road, OD, Objective, v3, cfg, audit)


def build_settings(a, mf, plan, audit, horizons):
    base = Path(a.capture_dir)
    plain = Path(a.plain_dir)
    source = {
        'capture_manifest': sha256(base / 'manifest.json'),
        'capture_provenance': sha256(base / 'provenance.json'),
        'capture_metrics': sha256(base / 'metrics.jsonl'),
        'capture_snapshots': sha256(base / 'snapshots.jsonl'),
        'plain_manifest': sha256(plain / 'manifest.json'),
        'plain_provenance': sha256(plain / 'provenance.json'),
        'plain_metrics': sha256(plain / 'metrics.jsonl'),
        'original_c3_solver': mf['solver_sha256'],
        'original_section': mf['section_sha256'],
        'original_r4a_core': mf['core_sha256'],
        'original_r4g_collector': mf['collector_sha256'],
    }
    for name,relative in {
        'verifier': 'road_vrp7_step6c1_r4g_prefix/verify_prefix.py',
        'core': 'road_vrp7_step6c1_r4g_prefix/prefix_core.py',
        'r4b_core': 'road_vrp7_step6c1_r4b/r4b_core.py',
        'r4b_reference': 'road_vrp7_step6c1_r4b/verify_r4b.py',
        'r4g_preflight': 'road_vrp7_step6c1_r4g/verify_train_r4g.py',
        'r4g_core': 'road_vrp7_step6c1_r4g/r4g_core.py',
    }.items():
        source['code_' + name] = sha256(ROOT / relative)
    return {'stage': STAGE, 'status': 'CONFIGURED',
            'source_sha256': source, 'data_role': 'TRAIN_ONLY',
            'horizons': horizons, 'max_cases': a.max_cases,
            'max_snapshots_per_case': a.max_snapshots_per_case,
            'reference_wall_sec': a.reference_wall_sec,
            'source_wall_sec': mf['search_wall_sec'],
            'source_max_iterations': mf['max_iterations'],
            'case_plan': selected_manifest(plan),
            'source_audit_strict_pass': audit['strict_pass'],
            'source_audit_pair_count': audit['pairs'],
            'source_audit_route_road_equal': audit['route_road_equal'],
            'reference_protocol': 'FRESH_OFFICIAL_C3_WITH_LONGER_REFERENCE_WALL_THEN_R4B_RESTORE',
            'limitations': ['The 90s reference is ONLY a deterministic trajectory verification control.',
                            'The original 30s wall cannot be exactly replayed.',
                            'No R4-G past results or frozen sources are overwritten.']}


def case_filename(case):
    d = json.dumps(list(case), separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(d.encode('utf-8')).hexdigest()[:24] + '.json'


def verify_case(a, case, windows, ctx, settings_hash):
    from verify_r4b import reference_recording, exact_resume
    from r4b_core import require as r4b_require, check_start, verify_events
    mf, plain, capture, snapshots, taskmap, road, OD, Objective, v3, cfg, audit = ctx
    tid, sid, seed = case
    task = taskmap[tid]
    sc = next(x for x in task['scenarios'] if x['scenario_id'] == sid)
    from r4g_core import validate_scenario_role
    validate_scenario_role(task, [sid])
    od = OD(road, task, sc, step_min=int(cfg['od_step_min']), hours=float(cfg['od_hours']),
            cache_dir=a.od_cache_dir)
    obj = Objective(task, od, 'static')
    selections = [(int(w['snapshot']['stop_count']), int(w['max_horizon'])) for w in windows]
    required_end = max(start + horizon - 1 for start, horizon in selections)
    with reference_recording(selections) as instances:
        answer = v3.solve('C3_topk', task, obj, seed,
                          a.reference_wall_sec, mf['max_iterations'],
                          top_k=mf['top_k'], anneal_iterations=mf['anneal_iterations'])
    require(len(instances) == 1, 'Expected exactly one frozen original ALNS.iterate')
    require(int(answer['iterations']) >= required_end,
            'Even longer-wall original ALNS did not cover requested reference windows; '
            f'actual={answer["iterations"]}, required={required_end}')
    ref_events = instances[0].events
    results = []
    for w in windows:
        snapshot = w['snapshot']
        start = int(snapshot['stop_count'])
        kh = int(w['max_horizon'])
        require(all(step in ref_events for step in range(start, start + kh)),
                f'Missing official reference events: {case} / {start}')
        check_start(snapshot, ref_events[start])
        # One longest restored continuation per checkpoint; K=1/50/200/500
        # all reuse genuine prefixes from that SAME proven continuous run.
        container = []
        with exact_resume(snapshot, kh, container):
            v3.solve('C3_topk', task, obj, seed,
                     mf['search_wall_sec'], mf['max_iterations'],
                     top_k=mf['top_k'], anneal_iterations=mf['anneal_iterations'])
        require(len(container) == 1, 'Expected exactly one R4B continuation trace')
        ref = [ref_events[i] for i in range(start, start + kh)]
        compare_prefix_events(ref, container[0], w['horizons'], verify_events)
        for k in w['horizons']:
            results.append({'task_id': tid, 'scenario_id': sid, 'seed': seed,
                            'snapshot_stop_count': start,
                            'horizon': k, 'verified_steps': k, 'exact': True,
                            'starting_RNG_sha256': snapshot['rng_state_sha256'],
                            'source_prefix_min_iterations': w['source_min_iterations'],
                            'reference_iterations': int(answer['iterations'])})
    return {'stage': STAGE, 'case': list(case), 'rows': results,
            'source_snapshots_sha256': sha256(Path(a.capture_dir) / 'snapshots.jsonl'),
            'settings_sha256': settings_hash,
            'original_reference_iterations': int(answer['iterations']),
            'original_reference_wall_sec': a.reference_wall_sec}


def run(a):
    require(a.reference_wall_sec >= 30 and a.reference_wall_sec <= 300,
            'Reference wall must be between 30s and 300s')
    require(a.max_cases >= 0 and a.max_snapshots_per_case >= 0,
            'Negative caps are invalid')
    ks = parse_horizons(a.horizons)
    ctx = preflight(a)
    mf, plain, capture, snapshots, taskmap, road, OD, Objective, v3, cfg, audit = ctx
    plan, excluded = plan_cases(plain, capture, snapshots, ks,
                                max_cases=a.max_cases, max_snapshots=a.max_snapshots_per_case)
    settings = build_settings(a, mf, plan, audit, ks)
    settings_hash = hashlib.sha256(json.dumps(settings, ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    out = Path(a.output_dir)
    if out.exists():
        require(a.resume, 'Use independent NEW output dir or RESUME=1')
        require(read_json(out / 'manifest.json') == settings,
                'Provenance/code/options/selected snapshots differ from resume manifest')
    else:
        require(not a.resume, 'Cannot resume non-existent proof directory')
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'manifest.json', settings)
    (out / 'units').mkdir(exist_ok=True)
    write_json(out / 'excluded_windows.json', excluded)
    all_rows = []
    for position, (case, windows) in enumerate(plan.items(), 1):
        unit_file = out / 'units' / case_filename(case)
        if unit_file.exists():
            saved = read_json(unit_file)
            source = settings['source_sha256']['capture_snapshots']
            rows = validate_unit(saved, case, windows, source, settings_hash)
            print(f'R4G PREFIX REUSE {position}/{len(plan)} {case} {len(rows)} matched windows', flush=True)
        elif not windows:
            rows = []
            print(f'R4G PREFIX SKIP {case}: zero source-bounded windows', flush=True)
        else:
            unit = verify_case(a, case, windows, ctx, settings_hash)
            rows = validate_unit(unit, case, windows,
                                 settings['source_sha256']['capture_snapshots'], settings_hash)
            write_json(unit_file, unit)
            print(f'R4G PREFIX EXACT {position}/{len(plan)} {case}, '
                  f'{len(windows)} snapshots, {len(rows)} K checks', flush=True)
        all_rows.extend(rows)
    report = summary(plan, excluded, all_rows, {
        'strict_pass': audit['strict_pass'], 'pairs': audit['pairs'],
        'route_road_equal': audit['route_road_equal'],
        'wall_limited_pairs': audit['wall_limited_pairs'],
        'source_snapshots': audit['n_snapshots'],
        'source_distinct_physical_states': audit['n_distinct_physical_states']})
    report['original_capture_snapshots_sha256'] = settings['source_sha256']['capture_snapshots']
    write_json(out / 'report.json', report)
    with (out / 'paired_rows.jsonl').open('w', encoding='utf-8') as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    lines = ['# R4G-Prefix Train固定K精确恢复', '',
             f"**{report['exact_matches']}/{report['n_comparisons']}**：逐步精确恢复。",
             f"核验案例 {report['n_cases']}；覆盖快照 {report['eligible_snapshots']}；"
             f"源配对终态一致 {audit['route_road_equal']}/{audit['pairs']}；"
             f"源固定20k严格通过 {audit['strict_pass']}/{audit['pairs']}。", '',
             '| K | 可验证快照 | 逐步一致 | 原30秒源前缀不足 |',
             '|---:|---:|---:|---:|']
    for k in ks:
        x = report['per_horizon'].get(str(k), {'eligible': 0, 'exact': 0, 'excluded': 0})
        lines.append(f"| {k} | {x['eligible']} | {x['exact']} | {x['excluded']} |")
    lines.extend(['', '## 重要界限', '',
                  '- 参考使用更长墙钟重新运行**冻结C3**；每个起点和后续事件均严格匹配。',
                  '- 这**不是**原30秒剩余墙钟的精确续接，也不是模型的在线收益。',
                  '- 原始30秒Plain/Capture只用于起点来源和迭代长度界限。',
                  '- 后续任何反事实标签需另行构造C3/SHAM负对照。', ''])
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('R4G PREFIX COMPLETE', report['exact_matches'], '/', report['n_comparisons'],
          'directory', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir', required=True)
    p.add_argument('--plain-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model-source', default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--cost-dir', default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir', default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir', default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--reference-wall-sec', type=float, default=90.0)
    p.add_argument('--horizons', default='1,50,200,500')
    p.add_argument('--max-cases', type=int, default=0)
    p.add_argument('--max-snapshots-per-case', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())
