"""Strict TRAIN checkpoint K-prefix planning and provenance utilities.

No ALNS import: planning is independently unit-testable and does not assert a
state is valid until original-reference vs restored trace actually matches.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path

STAGE = 'R4G-TRAIN-PREFIX-EXACT-v1'
DEFAULT_HORIZONS = (1, 50, 200, 500)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def parse_horizons(value):
    values = sorted({int(t.strip()) for t in value.split(',') if t.strip()})
    require(bool(values) and min(values) >= 1 and max(values) <= 500,
            'Expected nonempty K in [1,500]')
    return values


def identity(record):
    return (str(record['task_id']), str(record['scenario_id']), int(record['seed']))


def snapshot_identity(record):
    return (*identity(record), int(record['stop_count']))


def metric_map(metrics):
    out = {}
    for m in metrics:
        key = identity(m)
        require(key not in out, 'Duplicate metrics case')
        out[key] = m
    return out


def plan_cases(plain, capture, snapshots, horizons, *, max_cases=0, max_snapshots=0):
    """Plan usable windows against *both* original 30s iteration endpoints.

    Returns a dictionary keyed by (task, scenario, seed) with snapshots and
    horizons; all exclusions are recorded rather than silently dropped.
    """
    require(max_cases >= 0 and max_snapshots >= 0, 'Invalid case/snapshot caps')
    require(horizons == sorted(set(horizons)) and horizons[0] >= 1,
            'Horizon set must be unique and sorted')
    pm, cm = metric_map(plain), metric_map(capture)
    require(set(pm) == set(cm) and bool(pm), 'Plain/Capture task grid differs or empty')
    by = defaultdict(list)
    for s in snapshots:
        key = identity(s)
        require(key in cm, 'Snapshot does not belong to captured case')
        by[key].append(s)
    require(set(by) == set(cm), 'At least one capture case has no Section snapshot')
    result = {}
    excluded = []
    keys = sorted(by)
    if max_cases:
        keys = keys[:max_cases]
    for case in keys:
        p, c = pm[case], cm[case]
        require(p['best_route_sha256'] == c['best_route_sha256'] and
                float(p['road_cost_sec']) == float(c['road_cost_sec']),
                f'Original C3 Plain/Capture TERMINAL mismatch: {case}')
        cap = min(int(p['iterations']), int(c['iterations']))
        require(cap > 0, 'No iterations in paired reference')
        source = sorted(by[case], key=lambda s: int(s['stop_count']))
        require(len({int(s['stop_count']) for s in source}) == len(source),
                'Duplicate checkpoint stop_count')
        if max_snapshots:
            source = source[:max_snapshots]
        planned = []
        for s in source:
            stop = int(s['stop_count'])
            applicable = [k for k in horizons if stop + k - 1 <= cap]
            for k in horizons:
                if k not in applicable:
                    excluded.append({'task_id': case[0], 'scenario_id': case[1], 'seed': case[2],
                                     'stop_count': stop, 'horizon': k,
                                     'reason': 'SOURCE_30S_PREFIX_TOO_SHORT',
                                     'plain_iterations': int(p['iterations']),
                                     'capture_iterations': int(c['iterations'])})
            if applicable:
                planned.append({'snapshot': s, 'horizons': applicable, 'max_horizon': max(applicable),
                                'source_min_iterations': cap})
        result[case] = planned
    require(bool(result) and any(result.values()), 'No source-bounded horizon is available')
    return result, excluded


def selected_manifest(plan):
    return [[*case, [[x['snapshot']['stop_count'], x['horizons']] for x in items]]
            for case, items in plan.items()]


def expected_identity_rows(case, windows):
    return {(case[0], case[1], case[2], int(w['snapshot']['stop_count']), k)
            for w in windows for k in w['horizons']}


def validate_unit(unit, case, windows, source_sha, required_settings):
    require(unit.get('case') == list(case), 'Reused unit has wrong case')
    require(unit.get('source_snapshots_sha256') == source_sha, 'Reused unit source SHA differs')
    require(unit.get('settings_sha256') == required_settings, 'Reused unit config SHA differs')
    rows = unit.get('rows', [])
    require(isinstance(rows, list) and len(rows) == len(expected_identity_rows(case, windows)),
            'Reused unit is incomplete')
    keys = set()
    for r in rows:
        key = (r['task_id'], r['scenario_id'], int(r['seed']),
               int(r['snapshot_stop_count']), int(r['horizon']))
        require(key not in keys and r.get('exact') is True and
                int(r.get('verified_steps', -1)) == int(r['horizon']),
                'Duplicate, incorrect or unverified resumed row')
        keys.add(key)
    require(keys == expected_identity_rows(case, windows), 'Reused unit horizon/identity differs')
    return rows


def compare_prefix_events(original, restored, horizons, compare_fn):
    require(len(original) == len(restored) == max(horizons),
            'Reference or restored longest trace incomplete')
    for k in horizons:
        compare_fn(original[:k], restored[:k])
    return len(horizons)


def summary(plan, excluded, rows, source_summary):
    expected = sum(len(w['horizons']) for windows in plan.values() for w in windows)
    require(len(rows) == expected and all(z.get('exact') is True for z in rows),
            'Not all planned prefixes verified')
    eligible_states = sum(len(windows) for windows in plan.values())
    by_k = {}
    for k in sorted({h for wins in plan.values() for w in wins for h in w['horizons']}):
        actual = sum(int(z['horizon']) == k for z in rows)
        by_k[str(k)] = {'eligible': actual, 'exact': actual,
                        'excluded': sum(int(z['horizon']) == k for z in excluded)}
    return {'stage': STAGE, 'status': 'COMPLETE', 'data_role': 'TRAIN_ONLY',
            'n_cases': len(plan), 'eligible_snapshots': eligible_states,
            'n_comparisons': expected, 'exact_matches': expected,
            'per_horizon': by_k, 'excluded_horizon_count': len(excluded),
            'source_audit': source_summary,
            'limits': ['Source 30s windows used ONLY for eligibility, not event trace evidence.',
                       'Reference re-run may use longer wall for exact fixed-K verification.',
                       'Origin check + all K events must agree before marked exact.',
                       'Not 30s online performance or a learned strategy label.']}
