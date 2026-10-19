#!/usr/bin/env python3
"""Step6-A, audited OFFLINE Section action utility dataset on frozen ALNS snapshots.

This collector creates counterfactual labels, NOT online ALNS decisions. The labels
may use retrospective oracle traffic; history features use strictly prior RAW
observations and a real observation mask. This is July DEVELOPMENT data only.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'road_vrp7_stage3'))
sys.path.insert(0, str(ROOT / 'road_vrp7_step5b_v3'))
from road_vrp import Road, OD, Objective, legal, read_json, sha256, write_json
from section_v2 import Block, Partial, Selector, Stats, frozen, mutable, repair

VERSION = 'step6a-offline-v1'
DEFAULT_SOURCE = ('runs/chicago_july_v1/step5b_v3/full30s_1200_v1/'
                  '30s_anneal5000_cap20000')
DEFAULT_TASKS = 'datasets/processed/chicago/july_v1/tdcvrp30/tasks_10'
DEFAULT_COSTS = 'datasets/processed/chicago/july_v1/dynamic_costs'
DEFAULT_CACHE = 'runs/chicago_july_v1/stage3/od_cache'


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(',', ':')).encode('utf-8')).hexdigest()


def code_hashes():
    paths = [Path(__file__), Path(__file__).with_name('analyze_step6a.py'),
             Path(__file__).with_name('export_step6a.py'),
             Path(__file__).with_name('feature_schema.json'),
             ROOT / 'road_vrp7_step5b_v3/section_v2.py',
             ROOT / 'road_vrp7_stage3/road_vrp.py']
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p) for p in paths}


def load_snapshot_states(source_dir: Path, source_solvers, max_tasks=None,
                         max_scenarios=None, target_times=None, allowed_seeds=None):
    """Select deduplicated feasible-state references, never randomize across reruns."""
    snap_path = source_dir / 'snapshot_records.jsonl'
    conf = read_json(source_dir / 'run_config.json')
    summary = read_json(source_dir / 'benchmark_summary.json')
    if conf.get('stage') != 'step5b-v3' or summary.get('status') != 'COMPLETE':
        raise ValueError('Source must be a completed frozen Step5-B v3 dataset')
    if conf.get('data_role', '').lower().find('july development') < 0:
        raise ValueError('Unexpected source data role')
    if summary.get('actual_count') != summary.get('expected_count'):
        raise ValueError('Incomplete source experiment even though status is COMPLETE')
    task_keep = set(sorted(p.removesuffix('.json') for p in conf['tasks'])[:max_tasks]) if max_tasks else None
    scene_keep = set(conf['scenarios'][:max_scenarios]) if max_scenarios else None
    allowed = set(source_solvers)
    times = set(target_times) if target_times is not None else None
    seeds = set(allowed_seeds) if allowed_seeds is not None else None
    states = {}
    with snap_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            if d['solver'] not in allowed or d['planner'] != 'oracle':
                continue
            if task_keep is not None and d['task_id'] not in task_keep:
                continue
            if scene_keep is not None and d['scenario_id'] not in scene_keep:
                continue
            if seeds is not None and int(d['seed']) not in seeds:
                continue
            for sp in d['snapshots']:
                if sp['status'] != 'reached' or sp['routes'] is None:
                    continue
                if times is not None and float(sp['target_sec']) not in times:
                    continue
                routes = frozen(sp['routes'])
                ident = stable_digest([d['task_id'], d['scenario_id'], routes])[:24]
                state = {'state_id': ident, 'task_id': d['task_id'],
                         'scenario_id': d['scenario_id'], 'routes': [list(r) for r in routes],
                         'snapshot_sec': float(sp['target_sec']),
                         'source_solver': d['solver'], 'seed': int(d['seed']),
                         'source_line': line_no}
                # Prefer C3 and earlier snapshots; all alternatives are equivalent frozen routes.
                if ident not in states or (allowed_list_rank(d['solver']), float(sp['target_sec']), int(d['seed'])) < (
                    allowed_list_rank(states[ident]['source_solver']), states[ident]['snapshot_sec'], states[ident]['seed']):
                    states[ident] = state
    if not states:
        raise ValueError('No reached Oracle snapshots for filters')
    return list(states.values()), conf, snap_path


def allowed_list_rank(name):
    return {'C3_topk': 0, 'C0_baseline': 1}.get(name, 2)


def pick_balanced_states(states, limit, seed):
    by_pair = defaultdict(list)
    for r in states:
        by_pair[(r['task_id'], r['scenario_id'])].append(r)
    for pair, group in by_pair.items():
        group.sort(key=lambda s: stable_digest([seed, s['state_id']]))
    pairs = sorted(by_pair, key=lambda p: stable_digest([seed, p]))
    indices = {p: 0 for p in pairs}
    out = []
    quota = len(states) if limit is None else min(limit, len(states))
    while len(out) < quota:
        progressed = False
        for pair in pairs:
            i = indices[pair]
            if i < len(by_pair[pair]):
                out.append(by_pair[pair][i]); indices[pair] += 1; progressed = True
                if len(out) >= quota:
                    break
        if not progressed:
            raise AssertionError('Balanced sampler exhausted early')
    return out


def select_actions(cs, k, state_id, seed):
    """Stratified sampling over the STATIC Section score; no Oracle traffic seen."""
    if not cs:
        return []
    rng = np.random.default_rng(int(stable_digest([seed, state_id])[:16], 16))
    ordered = sorted(range(len(cs)), key=lambda i: (float(cs[i]['static']), i))
    bins = [list(x) for x in np.array_split(np.asarray(ordered, dtype=int), 3)]
    selected = []
    while len(selected) < min(k, len(cs)) and any(bins):
        for bin_ in bins:
            if not bin_ or len(selected) >= min(k, len(cs)):
                continue
            j = int(rng.integers(len(bin_)))
            selected.append(int(bin_.pop(j)))
    return [(i, cs[i]) for i in selected]


def make_partial(routes, c, obj):
    original = frozen(routes)
    r = int(c['route']); p = int(c['pos']); size = int(c['size'])
    ids = original[r][p:p+size]
    if len(ids) != size or len(original[r]) <= size:
        raise ValueError('Invalid Section that empties an exact-K route')
    stripped = mutable(original)
    del stripped[r][p:p+size]
    return Partial(frozen(stripped), obj, (Block(ids, r, p),), original)


def repair_for(obj, routes, candidate, top_k, deadline_sec, enable_swap):
    stats = Stats()
    tic = time.perf_counter()
    deadline = tic + deadline_sec
    partial = make_partial(routes, candidate, obj)
    found, reason = repair(partial, obj, deadline,
                           stats, top_k=top_k, nontrivial=True, enable_swap=enable_swap)
    elapsed = time.perf_counter() - tic
    obj_ok = reason == 'moved' and stats.d['repair_deadline'] == 0 and elapsed < deadline_sec-1e-4
    if obj_ok:
        legal(obj.task, found)
    elif reason not in ('deadline_fallback', 'no_feasible_nontrivial', 'moved'):
        raise RuntimeError(f'Unexpected Section repair status {reason}')
    return found, reason, elapsed, obj_ok, stats.summary()


def evaluate_section(routes, c, obj_td, obj_static, top_k, deadline_sec, enable_swap):
    """Same frozen S and block, compare two repair policies and cross-evaluate.

    TD labels are never returned as features. A legal but deteriorating move is
    still a valid supervised label; don't clamp negative gains to zero.
    """
    orig_td = float(obj_td.cost(routes))
    orig_static = float(obj_static.cost(routes))
    td_routes, td_reason, td_seconds, td_ok, td_stats = repair_for(
        obj_td, routes, c, top_k, deadline_sec, enable_swap)
    st_routes, st_reason, st_seconds, st_ok, st_stats = repair_for(
        obj_static, routes, c, top_k, deadline_sec, enable_swap)
    valid = td_ok and st_ok
    result = {'label_valid': bool(valid), 'td_repair_reason': td_reason,
              'static_repair_reason': st_reason, 'td_repair_wall_sec': td_seconds,
              'static_repair_wall_sec': st_seconds,
              'td_exact_evaluations': int(td_stats['repair_candidates_exact']),
              'static_exact_evaluations': int(st_stats['repair_candidates_exact']),
              'td_timed_out': bool(td_stats['repair_deadline'] > 0 or td_seconds >= deadline_sec-1e-4),
              'static_timed_out': bool(st_stats['repair_deadline'] > 0 or st_seconds >= deadline_sec-1e-4)}
    if valid:
        after_td = float(obj_td.cost(td_routes))
        after_static_same_move = float(obj_static.cost(td_routes))
        after_static = float(obj_static.cost(st_routes))
        after_td_static_repair = float(obj_td.cost(st_routes))
        result.update(td_gain_sec=orig_td-after_td,
                      static_gain_same_td_move_sec=orig_static-after_static_same_move,
                      static_repair_static_gain_sec=orig_static-after_static,
                      static_repair_td_gain_sec=orig_td-after_td_static_repair,
                      td_advantage_over_static_repair_sec=after_td_static_repair-after_td,
                      td_repaired_routes=td_routes,
                      static_repaired_routes=st_routes)
    else:
        result.update(td_gain_sec=None, static_gain_same_td_move_sec=None,
                      static_repair_static_gain_sec=None, static_repair_td_gain_sec=None,
                      td_advantage_over_static_repair_sec=None,
                      td_repaired_routes=None, static_repaired_routes=None)
    return result


class HistoricalCorridorFeatures:
    """Road RAW observed travel times STRICTLY before departure; static shortest paths.

    Raw observations are never reconstructed using future FIFO adjustment.
    Features are explanatory development covariates, not causal proofs.
    """
    def __init__(self, road, raw_travel_times, n_slots):
        self.road = road
        self.raw = raw_travel_times
        self.n_slots = n_slots
        if self.raw.shape != road.observed.shape:
            raise ValueError('Raw time/road array mismatch')
        self.paths = {}
        self.feature_cache = {}
        self.index_anchor = None
        self.key = None

    def enter(self, task_id, od, departure_local):
        depart = self.road.seconds(departure_local)
        # strict < departure, even if departure aligns exactly with a grid knot.
        anchor = int(np.searchsorted(np.arange(len(self.road.times)) * self.road.dt,
                                          depart, side='left')) - 1
        if anchor - self.n_slots + 1 < 0:
            raise ValueError('Insufficient true historical observations before departure')
        self.index_anchor = anchor
        self.key = (task_id, departure_local)
        self.od = od
        self.feature_cache.clear()

    def path_features(self, i, j):
        pkey = (self.od.task['task_id'], i, j)
        if pkey not in self.paths:
            _, path = self.road.path(self.od.nodes[i], self.od.nodes[j], self.od.start, mode='static')
            self.paths[pkey] = np.asarray(path, dtype=int)
        edges = self.paths[pkey]
        cache_key = (i, j)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        if not len(edges):
            ans = {'edge_count': 0, 'covered_fraction': [0.]*self.n_slots,
                   'observed_rel_cost': [0.]*self.n_slots, 'has_any_observation': False}
        else:
            indices = np.arange(self.index_anchor-self.n_slots+1, self.index_anchor+1)
            mask = self.road.observed[np.ix_(indices, edges)]
            vals = self.raw[np.ix_(indices, edges)]
            stat = self.road.static[edges][None, :]
            ratio = np.divide(vals, stat, out=np.zeros_like(vals, dtype=float), where=stat>0)
            coverage = mask.mean(axis=1)
            sums = (ratio * mask).sum(axis=1)
            denomin = np.maximum(1, mask.sum(axis=1))
            observed_rel = sums / denomin
            ans = {'edge_count': int(len(edges)),
                   'covered_fraction': coverage.tolist(),
                   'observed_rel_cost': observed_rel.tolist(),
                   'has_any_observation': bool(mask.any())}
        self.feature_cache[cache_key] = ans
        return ans

    def section_history(self, original_routes, candidate, obj):
        route, pos, size = (int(candidate[k]) for k in ('route','pos','size'))
        customers = original_routes[route]
        before = 0 if pos == 0 else obj.index[customers[pos-1]]
        after = 0 if pos+size == len(customers) else obj.index[customers[pos+size]]
        seq = [before]+[obj.index[c] for c in customers[pos:pos+size]]+[after]
        feats = [self.path_features(i, j) for i, j in zip(seq[:-1], seq[1:])]
        counts = np.array([f['edge_count'] for f in feats], dtype=float)
        if not counts.sum():
            return {'history_observed_share': 0., 'history_observed_rel_cost': 0.,
                    'history_slots': self.n_slots, 'history_coverage_by_slot': [0.]*self.n_slots,
                    'history_rel_cost_by_slot': [0.]*self.n_slots, 'history_has_observation': False}
        coverage = np.average(np.array([f['covered_fraction'] for f in feats]), axis=0, weights=counts)
        # Weight observed ratio means by the number of REAL observed traversals.
        weights = np.array([np.array(f['covered_fraction'])*f['edge_count'] for f in feats])
        numerator = (weights*np.array([f['observed_rel_cost'] for f in feats])).sum(axis=0)
        denominator = weights.sum(axis=0)
        ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator>0)
        weighted_mean = float(numerator.sum()/denominator.sum()) if denominator.sum()>0 else 0.
        return {'history_observed_share': float(coverage.mean()),
                'history_observed_rel_cost': weighted_mean,
                'history_slots': self.n_slots,
                'history_coverage_by_slot': coverage.tolist(),
                'history_rel_cost_by_slot': ratio.tolist(),
                'history_has_observation': bool(denominator.sum() > 0)}


def read_raw_cost(cost_dir):
    with np.load(Path(cost_dir)/'edge_costs.npz', allow_pickle=False) as z:
        if 'travel_sec_raw' not in z:
            raise ValueError('Raw historical road travel times missing')
        return z['travel_sec_raw']


def parse_csv_values(s, typ=str):
    return tuple(typ(x.strip()) for x in s.split(',') if x.strip())


def make_config(args, source_config, snap_path, task_files):
    return {'stage': VERSION, 'data_role': 'July development only; no blind holdout',
            'source_experiment': str(Path(args.source_dir)),
            'source_snapshot_sha256': sha256(snap_path),
            'source_config_sha256': sha256(Path(args.source_dir)/'run_config.json'),
            'source_cost_hash': source_config['cost_hash'],
            'source_edge_hash': source_config['edge_hash'],
            'task_manifest_sha256': sha256(Path(args.task_dir)/'manifest.json'),
            'task_files': {p.name: sha256(p) for p in task_files},
            'code_sha256': code_hashes(), 'source_solvers': parse_csv_values(args.source_solvers),
            'target_times': parse_csv_values(args.target_times, float),
            'max_states': args.max_states, 'max_tasks': args.max_tasks,
            'max_scenarios': args.max_scenarios, 'states_seed': args.states_seed,
            'actions_per_state': args.actions_per_state, 'top_k': args.top_k,
            'candidate_deadline_sec': args.candidate_deadline_sec,
            'enable_swap': bool(args.enable_swap), 'lookback_slots': args.lookback_slots,
            'od_step_min': args.od_step_min, 'od_hours': args.od_hours,
            'label_role': 'retrospective oracle objective; NOT deployable features',
            'feature_role': 'pre-departure raw observed traffic only; static topology'}


def write_line(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
        f.flush(); os.fsync(f.fileno())


def resume_states(output_dir, config, resume):
    out = Path(output_dir)
    cf = out/'run_config.json'
    file = out/'state_records.jsonl'
    if cf.exists():
        if read_json(cf) != json.loads(json.dumps(config)):
            raise ValueError('Manifest/source/code changed; choose NEW output directory')
        if not resume:
            raise FileExistsError('Existing output directory: rerun with --resume or use NEW study')
    else:
        if file.exists() or (out/'benchmark_summary.json').exists():
            raise ValueError('Unmanifested partial output; inspect manually')
        out.mkdir(parents=True, exist_ok=True)
        write_json(cf, config)
        file.touch()
    ids = []
    with file.open(encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.endswith('\n'):
                raise ValueError('Truncated checkpoint line, fail closed')
            row = json.loads(line)
            ids.append(row['state_id'])
            if len(row['candidates']) > config['actions_per_state']:
                raise ValueError('Corrupt checkpoint: too many actions')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate state in checkpoint')
    return set(ids)


def compute_task_roles(task_ids):
    ids = sorted(task_ids)
    n = len(ids)
    a = max(1, int(n*.6))
    b = max(a+1, int(n*.8)) if n>=3 else n
    return {k: ('train' if i<a else 'val' if i<b else 'internal_test') for i,k in enumerate(ids)}


def run(args):
    if args.actions_per_state < 2 or args.top_k < 1 or args.max_states < 1:
        raise ValueError('Require >=2 actions/state, >=1 topK, >=1 state')
    if args.candidate_deadline_sec <= 0 or args.lookback_slots < 1:
        raise ValueError('Invalid candidate deadline / history window')
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError('Invalid max-tasks')
    if args.max_scenarios is not None and args.max_scenarios < 1:
        raise ValueError('Invalid max-scenarios')
    source_dir = Path(args.source_dir)
    task_dir = Path(args.task_dir)
    tasks_paths = sorted(task_dir.glob('task_*.json'))
    manifest = read_json(task_dir/'manifest.json')
    if not tasks_paths or [p.stem for p in tasks_paths] != manifest['task_ids']:
        raise ValueError('Task files/manifest inconsistent')
    tasks = {p.stem: read_json(p) for p in tasks_paths}
    for t in tasks.values(): legal(t, t['seed_routes'])
    times = parse_csv_values(args.target_times, float)
    states, src_config, snap_path = load_snapshot_states(
        source_dir, parse_csv_values(args.source_solvers), args.max_tasks,
        args.max_scenarios, target_times=times)
    # Pin the original v3 Section implementation and TD objective used for snapshots.
    for relative in ('road_vrp7_step5b_v3/section_v2.py', 'road_vrp7_stage3/road_vrp.py'):
        historical_hash = src_config.get('source', {}).get(relative)
        current_hash = sha256(ROOT/relative)
        if historical_hash is None or historical_hash != current_hash:
            raise ValueError(f'Source implementation differs from Step5B v3 frozen manifest: {relative}')
    states = pick_balanced_states(states, args.max_states, args.states_seed)
    if any(s['task_id'] not in tasks for s in states):
        raise ValueError('Source references unknown task')
    for st in states:
        scenario = next((x for x in tasks[st['task_id']]['scenarios'] if x['scenario_id']==st['scenario_id']), None)
        if scenario is None or not scenario['eligible']:
            raise ValueError('Unknown/ineligible source scenario')
        if not scenario['departure_local']:
            raise ValueError('Missing departure time')
        legal(tasks[st['task_id']], st['routes'])
    config = make_config(args, src_config, snap_path,
                         tasks_paths[:args.max_tasks] if args.max_tasks else tasks_paths)
    print(f'Selected {len(states)} unique frozen states from {snap_path}', flush=True)
    print(f'Candidate evaluations requested: up to {len(states)*args.actions_per_state}', flush=True)
    if args.dry_run:
        print('DRY RUN: no road tensor loaded or output created')
        return
    completed = resume_states(args.output_dir, config, args.resume)
    wanted = {s['state_id'] for s in states}
    if not completed.issubset(wanted):
        raise ValueError('Checkpoint contains states outside current selection')
    if len(completed) == len(states):
        print('All states already evaluated, re-running final audit')
    road = Road(args.cost_dir)
    if src_config['cost_hash']!=road.cost_file_hash or src_config['edge_hash']!=road.edge_file_hash:
        raise ValueError('Road tensor differs from source Step5-B v3')
    if manifest['cost_edge_index_sha256']!=road.edge_file_hash or manifest['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Task manifest and road costs inconsistent')
    if read_json(Path(args.cost_dir).parent/'stage2_validation.json').get('validation')!='PASS':
        raise ValueError('Stage2 cost validation not PASS')
    raw = read_raw_cost(args.cost_dir)
    hist = HistoricalCorridorFeatures(road, raw, args.lookback_slots)
    by_scenario = defaultdict(list)
    for state in states:
        if state['state_id'] not in completed:
            by_scenario[(state['task_id'], state['scenario_id'])].append(state)
    all_roles = compute_task_roles(tasks)
    output = Path(args.output_dir)
    tic = time.time()
    count = len(completed)
    for (tid, scid), group in sorted(by_scenario.items()):
        task = tasks[tid]
        sc = next(x for x in task['scenarios'] if x['scenario_id']==scid)
        od = OD(road, task, sc, step_min=args.od_step_min, hours=args.od_hours,
                cache_dir=args.od_cache_dir)
        td = Objective(task, od, 'oracle')
        static = Objective(task, od, 'static')
        hist.enter(tid, od, sc['departure_local'])
        import pandas as pd
        timestamp = pd.Timestamp(sc['departure_local'])
        selector = Selector(static)
        for state in group:
            routes = state['routes']
            candidate_infos = selector.candidates(routes, traffic=False)
            chosen = select_actions(candidate_infos, args.actions_per_state,
                                    state['state_id'], args.states_seed)
            t0 = time.perf_counter()
            base_td = float(td.cost(routes))
            base_static = float(static.cost(routes))
            rr = dict(state)
            rr.update(departure_local=sc['departure_local'], date=sc['date'],
                      task_role=all_roles[tid],
                      original_td_cost_sec=base_td, original_static_cost_sec=base_static,
                      departure_hour_local=float(timestamp.hour + timestamp.minute/60.),
                      departure_weekday_local=int(timestamp.weekday()),
                      candidate_pool_size=len(candidate_infos), candidates=[],
                      scenario_role='July development only',
                      traffic_feature_cutoff='STRICTLY_BEFORE_DEPARTURE')
            for j, c in chosen:
                label = evaluate_section(routes, c, td, static, args.top_k,
                                         args.candidate_deadline_sec, args.enable_swap)
                features = hist.section_history(routes, c, td)
                block = routes[c['route']][c['pos']:c['pos']+c['size']]
                demands = sum(td.dem[int(v)] for v in block)
                row = {'candidate_index': j, 'candidate_id': stable_digest([state['state_id'], j])[:24],
                       'route_index': int(c['route']), 'position': int(c['pos']),
                       'size': int(c['size']), 'block_ids': list(map(int, block)),
                       'block_demand': int(demands), 'route_load': int(sum(td.dem[v] for v in routes[c['route']])),
                       'vehicle_capacity': int(td.capacity), 'static_section_cost_sec': float(c['static']),
                       'route_length': len(routes[c['route']]),
                       'relative_position': float(c['pos']/max(1,len(routes[c['route']])-1)),
                       **features, **label}
                rr['candidates'].append(row)
            rr['state_evaluation_wall_sec'] = time.perf_counter()-t0
            write_line(output/'state_records.jsonl', rr)
            count += 1
            if count <= 3 or count % 25 == 0:
                n_valid = sum(int(x['label_valid']) for x in rr['candidates'])
                print(f'State {count}/{len(states)} {tid}/{scid} {n_valid}/{len(rr["candidates"])} valid, elapsed={time.time()-tic:.1f}s',flush=True)
    if count != len(states):
        raise AssertionError('Incomplete collector internal state')
    done = []
    for l in (output/'state_records.jsonl').read_text(encoding='utf-8').splitlines():
        if l.strip(): done.append(json.loads(l)['state_id'])
    if len(done)!=len(wanted) or set(done)!=wanted:
        raise ValueError('Final checkpoint audit failure')
    write_json(output/'benchmark_summary.json',
               {'status':'COMPLETE','expected_states':len(states),'actual_states':len(done),
                'expected_max_actions':len(states)*args.actions_per_state,
                'code_role':'JULY_DEVELOPMENT','label':'oracle TD repair, static TD counterfactuals'})
    print(f'PASS: collected {count} states in {output}',flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',default=DEFAULT_SOURCE)
    p.add_argument('--task-dir',default=DEFAULT_TASKS)
    p.add_argument('--cost-dir',default=DEFAULT_COSTS)
    p.add_argument('--od-cache-dir',default=DEFAULT_CACHE)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--source-solvers',default='C3_topk,C0_baseline')
    p.add_argument('--target-times',default='5,15,30')
    p.add_argument('--max-tasks',type=int)
    p.add_argument('--max-scenarios',type=int)
    p.add_argument('--max-states',type=int,default=1000)
    p.add_argument('--states-seed',type=int,default=20261019)
    p.add_argument('--actions-per-state',type=int,default=8)
    p.add_argument('--top-k',type=int,default=12)
    p.add_argument('--candidate-deadline-sec',type=float,default=3.)
    p.add_argument('--enable-swap',action='store_true',default=True)
    p.add_argument('--no-swap',action='store_false',dest='enable_swap')
    p.add_argument('--lookback-slots',type=int,default=6)
    p.add_argument('--od-step-min',type=int,default=20)
    p.add_argument('--od-hours',type=float,default=5.)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    return p.parse_args()

if __name__=='__main__':run(cli())
