#!/usr/bin/env python3
"""Step6-A no-leakage, paired-within-state utility label audit and ranking diagnostics.

This does NOT claim predictive performance: oracle ranking is a label-only upper bound.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np


def mean(rows):
    return float(np.mean(rows)) if rows else None


def ranks(values):
    # Ascending ranks, midranks for ties; use no scipy dependency.
    idx=sorted(range(len(values)),key=lambda i:(float(values[i]),i))
    out=np.empty(len(values),float)
    i=0
    while i<len(idx):
        j=i+1
        while j<len(idx) and values[idx[j]]==values[idx[i]]:
            j+=1
        for k in idx[i:j]:out[k]=(i+j-1)/2+1
        i=j
    return out


def spearman(x,y):
    if len(x)<3:return None
    a,b=ranks(x),ranks(y)
    if np.std(a)<=1e-12 or np.std(b)<=1e-12:return None
    return float(np.corrcoef(a,b)[0,1])


def state_diagnostics(state):
    cs=[c for c in state['candidates'] if c['label_valid']]
    if len(cs)<2:
        return None
    td=[float(c['td_gain_sec']) for c in cs]
    static=[float(c['static_repair_static_gain_sec']) for c in cs]
    same=[float(c['static_gain_same_td_move_sec']) for c in cs]
    if not all(math.isfinite(v) for seq in (td,static,same) for v in seq):
        raise ValueError('Nonfinite utility label')
    td_best=int(np.argmax(td));st_best=int(np.argmax(static))
    regret=float(td[td_best]-td[st_best])
    if regret< -1e-7:raise AssertionError('Negative ranking regret')
    return {'state_id':state['state_id'],'task_id':state['task_id'],
            'scenario_id':state['scenario_id'],'source_solver':state['source_solver'],
            'date':state['date'],'task_role':state['task_role'],
            'n_actions':len(cs),'rho_static_td_same_move':spearman(same,td),
            'rho_static_repair_td':spearman(static,td),
            'top1_changed':int(td_best!=st_best),
            'oracle_top1_gain_sec':td[td_best],
            'static_selected_td_gain_sec':td[st_best],
            'oracle_vs_static_selection_regret_sec':regret,
            'has_useful_oracle_action':int(td[td_best]>1e-7),
            'mean_history_support':float(np.mean([float(c['history_observed_share']) for c in cs]))}


def audit_one_state(r, config):
    if len(r['candidates']) > config['actions_per_state']:
        raise ValueError('Excess candidate rows')
    ids=[c['candidate_id'] for c in r['candidates']]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate candidate identifiers')
    if len(r['routes'])==0:
        raise ValueError('Empty state routes')
    orig_flat=[i for rr in r['routes'] for i in rr]
    if len(orig_flat)!=len(set(orig_flat)):
        raise ValueError('Duplicate customers in state')
    if not math.isfinite(float(r['original_td_cost_sec'])):
        raise ValueError('Bad state objective')
    for c in r['candidates']:
        block=r['routes'][c['route_index']][c['position']:c['position']+c['size']]
        if block!=c['block_ids']:
            raise ValueError('Section block not contained in original solution')
        if len(c['history_coverage_by_slot'])!=config['lookback_slots']:
            raise ValueError('History coverage window mismatch')
        if len(c['history_rel_cost_by_slot'])!=config['lookback_slots']:
            raise ValueError('History ratio window mismatch')
        if c['history_observed_share']<0 or c['history_observed_share']>1:
            raise ValueError('Observation share outside [0,1]')
        for frac in c['history_coverage_by_slot']:
            if not (0<=frac<=1):raise ValueError('Bad history mask fraction')
        if c['label_valid']:
            td=c['td_repaired_routes'];st=c['static_repaired_routes']
            for routes in (td,st):
                if len(routes)!=len(r['routes']):raise ValueError('Route count changed')
                flat=[int(i) for rr in routes for i in rr]
                if sorted(flat)!=sorted(orig_flat):
                    raise ValueError('Candidate lost/duplicated customers')
                if any(not rr for rr in routes):raise ValueError('Candidate left an empty route')
            if abs(c['td_advantage_over_static_repair_sec']-(
                c['td_gain_sec']-c['static_repair_td_gain_sec']))>1e-5:
                raise ValueError('Cross-evaluation identity violated')
            if c['td_advantage_over_static_repair_sec'] < -1e-5:
                raise ValueError('TD repair cannot be worse than static repair in identical candidate pool')
        else:
            if c['td_gain_sec'] is not None or c['static_gain_same_td_move_sec'] is not None:
                raise ValueError('Invalid labels must be null, not filled')


def save_csv(file, rows, columns):
    with Path(file).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def analyze(experiment_dir):
    folder=Path(experiment_dir)
    config=json.loads((folder/'run_config.json').read_text(encoding='utf-8'))
    summary=json.loads((folder/'benchmark_summary.json').read_text(encoding='utf-8'))
    if config.get('stage')!='step6a-offline-v1' or summary['status']!='COMPLETE':
        raise ValueError('Step6-A must have a COMPLETE collector manifest')
    ids=set();diagnostics=[];rows=[]
    tasks=set();days=set();seeds=set();solvers=set();candidate_ids=set()
    positive=negative=tied=valid=invalid=0
    support_stats=defaultdict(lambda:{'n':0,'pos':0,'td_gain':[],'repair_adv':[]})
    by_task=defaultdict(lambda:{'states':0,'actions':0,'valid':0,'positive':0,'gain':[]})
    with (folder/'state_records.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line)
            if r['state_id'] in ids:raise ValueError('Duplicate state ID')
            ids.add(r['state_id']);tasks.add(r['task_id']);days.add(r['date'])
            seeds.add(int(r['seed']));solvers.add(r['source_solver'])
            audit_one_state(r,config)
            group=by_task[r['task_id']]
            group['states']+=1;group['actions']+=len(r['candidates'])
            d=state_diagnostics(r)
            if d is not None:diagnostics.append(d)
            for c in r['candidates']:
                if c['candidate_id'] in candidate_ids:raise ValueError('Duplicate candidate globally')
                candidate_ids.add(c['candidate_id'])
                if c['label_valid']:
                    valid+=1;group['valid']+=1
                    gain=float(c['td_gain_sec']);group['gain'].append(gain)
                    pos=(gain>1e-7)
                    if pos: positive+=1;group['positive']+=1
                    elif gain < -1e-7:negative+=1
                    else:tied+=1
                    frac=float(c['history_observed_share'])
                    level='zero' if frac<1e-10 else 'low_0_10pct' if frac<=.10 else 'medium_10_30pct' if frac<=.30 else 'high_gt30pct'
                    s=support_stats[level]
                    s['n']+=1;s['pos']+=int(pos);s['td_gain'].append(gain)
                    s['repair_adv'].append(c['td_advantage_over_static_repair_sec'])
                else:invalid+=1
    if len(ids)!=summary['expected_states'] or len(ids)!=summary['actual_states']:
        raise ValueError('Incomplete state count')
    out=folder/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    diag_cols=['state_id','task_id','scenario_id','source_solver','date','task_role','n_actions',
               'rho_static_td_same_move','rho_static_repair_td','top1_changed',
               'oracle_top1_gain_sec','static_selected_td_gain_sec',
               'oracle_vs_static_selection_regret_sec','has_useful_oracle_action',
               'mean_history_support']
    save_csv(out/'state_ranking_diagnostics.csv',diagnostics,diag_cols)
    support_rows=[]
    for k in ('zero','low_0_10pct','medium_10_30pct','high_gt30pct'):
        s=support_stats[k]
        support_rows.append({'support_bin':k,'n_candidates':s['n'],
                             'positive_rate':s['pos']/s['n'] if s['n'] else None,
                             'mean_td_gain_sec':mean(s['td_gain']),
                             'mean_td_repair_adv_sec':mean(s['repair_adv'])})
    save_csv(out/'support_strata.csv',support_rows,
             ['support_bin','n_candidates','positive_rate','mean_td_gain_sec','mean_td_repair_adv_sec'])
    task_rows=[]
    for t in sorted(by_task):
        s=by_task[t]
        task_rows.append({'task_id':t,'states':s['states'],'candidates':s['actions'],
                          'valid':s['valid'],'positive':s['positive'],
                          'positive_rate':s['positive']/s['valid'] if s['valid'] else None,
                          'mean_td_gain_sec':mean(s['gain'])})
    save_csv(out/'task_summary.csv',task_rows,
             ['task_id','states','candidates','valid','positive','positive_rate','mean_td_gain_sec'])
    rho=[x['rho_static_repair_td'] for x in diagnostics if x['rho_static_repair_td'] is not None]
    same_rho=[x['rho_static_td_same_move'] for x in diagnostics if x['rho_static_td_same_move'] is not None]
    counterfactual_regret=[x['oracle_vs_static_selection_regret_sec'] for x in diagnostics]
    overall={'stage':'step6a-offline-v1','data_role':'July development ONLY; not blind test',
             'states':len(ids),'candidates':len(candidate_ids),'valid_labels':valid,
             'invalid_labels':invalid,'positive_actions':positive,'negative_actions':negative,
             'tied_actions':tied,'positive_rate':positive/valid if valid else None,
             'states_with_two_plus_valid_actions':len(diagnostics),
             'mean_within_state_static_repair_vs_td_spearman':mean(rho),
             'mean_within_state_same_move_static_vs_td_spearman':mean(same_rho),
             'states_static_top1_differs_from_td_top1':sum(x['top1_changed'] for x in diagnostics),
             'top1_disagreement_rate':mean([x['top1_changed'] for x in diagnostics]),
             'mean_oracle_vs_static_selection_regret_sec':mean(counterfactual_regret),
             'mean_oracle_best_gain_sec':mean([x['oracle_top1_gain_sec'] for x in diagnostics]),
             'tasks':sorted(tasks),'dates':sorted(days),'seeds':sorted(seeds),
             'source_solvers':sorted(solvers),
             'warnings':['These are hindsight labels from frozen BEST snapshots, not ALNS arbitrary current-state trajectories.',
                         'Oracle-generated source routes may encode privileged future traffic indirectly; not a deployment evaluation.',
                         'Oracle ranking and regret are diagnostic upper bounds, NOT results from a trained model.',
                         'No observed-road ground truth exists for permanently unmonitored roads.',
                         'Task split uses groups, but current July data were previously inspected; no blind-test claim.',
                         'Raw historical slots strictly precede departure; never use Oracle future costs as model features.',
                         'Timed-out/censored repairs are excluded from supervised statistics.']}
    (out/'summary.json').write_text(json.dumps(overall,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:overall[k] for k in ['states','candidates','valid_labels','invalid_labels',
              'positive_rate','mean_within_state_static_repair_vs_td_spearman',
              'top1_disagreement_rate','mean_oracle_vs_static_selection_regret_sec']},indent=2))
    return overall


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir',required=True)
    analyze(p.parse_args().experiment_dir)

if __name__=='__main__':main()
