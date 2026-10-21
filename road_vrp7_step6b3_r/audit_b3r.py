#!/usr/bin/env python3
"""Strict six-seed B3 offline audit. Requires only numpy and stdlib.

Never uses test labels to tune model, threshold, or alpha. Read-only.
"""
from __future__ import annotations
import argparse,csv,json,hashlib,math
from pathlib import Path
from collections import defaultdict
import numpy as np

DEFAULT_ROOT=Path('runs/chicago_july_v1/step6b3')
METHODS=('M0','M1','M2','M3','M4')
PROTOCOLS=('task','date')
EPS=1e-7


def csvrows(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError(f'Empty CSV {path}')
    return rows


def index_states(rows):
    indexed={}
    for r in rows:
        k=r['state_id']
        if k in indexed:raise ValueError(f'Duplicate state_id {k}')
        indexed[k]=r
    return indexed


def effective_action(r):
    return str(r['selected_candidate_id']) if int(r['execute']) else '__NO_OP__'


def score_row(row):
    return float(row['realized_signed_gain_sec'])


def compare_states(baserows,treatrows,protocol='',seed=None):
    base=index_states(baserows);treat=index_states(treatrows)
    if base.keys()!=treat.keys():raise ValueError('Unmatched state ID sets')
    result=[]
    for sid in sorted(base):
        a,b=base[sid],treat[sid]
        for key in ('state_id','task_id','date','scenario_id'):
            if a.get(key)!=b.get(key):raise ValueError(f'State context mismatch {sid} {key}')
        # Realized gain must be zero when execute=0, else equal chosen counterfactual gain.
        for rr in (a,b):
            raw=float(rr['chosen_static_repair_td_gain_sec'])
            calc=raw if int(rr['execute']) else 0.
            if not math.isclose(score_row(rr),calc,rel_tol=0,abs_tol=3e-5):
                raise ValueError(f'Incorrect signed realized reward at {sid}')
        av=score_row(a);bv=score_row(b)
        changed_rank=a['selected_candidate_id']!=b['selected_candidate_id']
        changed_exec=a['execute']!=b['execute']
        changed_real=effective_action(a)!=effective_action(b)
        result.append({'seed':seed,'protocol':protocol,'state_id':sid,
            'task_id':a['task_id'],'date':a['date'],'scenario_id':a['scenario_id'],
            'base_action':effective_action(a),'treatment_action':effective_action(b),
            'base_execute':int(a['execute']),'treatment_execute':int(b['execute']),
            'base_gain':av,'treatment_gain':bv,'delta_gain_sec':bv-av,
            'changed_top1_candidate':int(changed_rank),'changed_execute':int(changed_exec),
            'changed_effective_action':int(changed_real),
            'fake_change_both_noop':int(changed_rank and not changed_real),
            'base_score':float(a['score']),'treatment_score':float(b['score'])})
    return result


def stats(rows):
    d=np.asarray([r['delta_gain_sec'] for r in rows],float)
    changed=[r for r in rows if r['changed_effective_action']]
    eff=np.asarray([r['delta_gain_sec'] for r in changed],float)
    return dict(states=len(rows),mean_gain_diff_sec_per_state=float(d.mean()),
        median_gain_diff_sec_per_state=float(np.median(d)),
        win=int(np.sum(d>EPS)),loss=int(np.sum(d< -EPS)),tie=int(np.sum(abs(d)<=EPS)),
        top1_changed=int(sum(r['changed_top1_candidate'] for r in rows)),
        execute_changed=int(sum(r['changed_execute'] for r in rows)),
        effective_action_changed=len(changed),
        effective_action_change_rate=float(len(changed)/len(rows)),
        both_noop_top1_changed=int(sum(r['fake_change_both_noop'] for r in rows)),
        mean_gain_diff_when_effective_action_changes=float(eff.mean()) if len(eff) else None,
        worst_loss_sec=float(d.min()),best_gain_sec=float(d.max()),
        by_task={k:float(np.mean([v['delta_gain_sec'] for v in vals]))
                 for k,vals in sorted(group_by(rows,'task_id').items())},
        by_date={k:float(np.mean([v['delta_gain_sec'] for v in vals]))
                 for k,vals in sorted(group_by(rows,'date').items())})


def group_by(rows,key):
    d=defaultdict(list)
    for r in rows:d[r[key]].append(r)
    return d


def readjson(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def load_data(root,seed,protocol,methods):
    study=root/f'b3_anchor_film_seed{seed}_v1'
    summ=readjson(study/'summary.json')
    manifest=readjson(study/'manifest.json')
    if summ.get('status')!='COMPLETE' or protocol not in summ.get('protocols',[]):
        raise ValueError(f'Incomplete study {study} / {protocol}')
    if manifest['stage']!='step6b3-anchored-film-v1' or manifest['seed']!=seed:
        raise ValueError(f'Wrong model study seed/stage: {study}')
    if not (study/protocol/'report.json').exists():raise FileNotFoundError(study/protocol/'report.json')
    report=readjson(study/protocol/'report.json')
    data={}
    for method in methods:
        if method not in report:raise ValueError(f'Unreported method {method} in {study}')
        data[method]=csvrows(study/protocol/f'{method}_test_states.csv')
        if len(data[method])!=int(report[method]['test_policy']['states']):
            raise ValueError('State count differs from report')
        mean=float(np.mean([score_row(r) for r in data[method]]))
        if not math.isclose(mean,report[method]['test_policy']['mean_signed_gain_sec'],abs_tol=1e-4):
            raise ValueError('CSV/report signed gain mismatch')
    return manifest,report,data


def candidate_diagnostics(study,protocol,method):
    p=study/protocol/f'{method}_test_candidates.csv'
    if not p.exists():return None
    rows=csvrows(p)
    delta=np.asarray([float(x['delta_model_vs_anchor_sec']) for x in rows])
    y=np.asarray([float(x['target_static_repair_td_gain_sec']) for x in rows])
    model=np.asarray([float(x['model_score_sec']) for x in rows])
    anchor=np.asarray([float(x['static_anchor_score_sec']) for x in rows])
    if not np.allclose(model-anchor,delta,rtol=0,atol=1e-5):
        raise ValueError('Predicted residual not equal scores difference')
    return {'n_candidates':len(rows),'mae_sec':float(np.mean(np.abs(model-y))),
            'anchor_mae_sec':float(np.mean(np.abs(anchor-y))),
            'mean_predicted_correction_sec':float(delta.mean()),
            'median_abs_correction_sec':float(np.median(abs(delta))),
            'p95_abs_correction_sec':float(np.percentile(abs(delta),95)),
            'max_abs_correction_sec':float(abs(delta).max())}


def audit(root,out,seeds,methods=METHODS,protocols=PROTOCOLS):
    root,out=Path(root),Path(out)
    if out.exists():raise FileExistsError(out)
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Nonunique/empty seeds')
    per_seed=[];changes=[];diagnostics=[];byproto=defaultdict(list)
    hashes=None
    for seed in seeds:
        for p in protocols:
            mf,report,data=load_data(root,seed,p,methods)
            fingerprint=(mf['source_sha256'],mf['feature_npz_sha256'])
            if hashes is None:hashes=fingerprint
            if hashes!=fingerprint:raise ValueError('Training source changed across seeds')
            study=root/f'b3_anchor_film_seed{seed}_v1'
            for method in methods:
                rows=compare_states(data['M0'],data[method],p,seed)
                summary=stats(rows)
                if method!='M0':changes.extend([{**r,'method':method} for r in rows if r['changed_effective_action'] or r['fake_change_both_noop']])
                byproto[(p,method)].append(summary['mean_gain_diff_sec_per_state'])
                per_seed.append({'seed':seed,'protocol':p,'method':method,
                    'selected_alpha':report[method]['alpha_selected_on_validation'],
                    'val_gain_sec_per_state':report[method]['validation_policy']['mean_signed_gain_sec'],
                    'test_gain_sec_per_state':report[method]['test_policy']['mean_signed_gain_sec'],
                    **summary})
                diag=candidate_diagnostics(study,p,method)
                if diag:diagnostics.append({'seed':seed,'protocol':p,'method':method,**diag})
    results=[]
    for (p,m),v in sorted(byproto.items()):
        z=np.asarray(v,dtype=float)
        results.append({'protocol':p,'method':m,'train_seeds':len(v),
           'mean_gain_diff_sec_per_state':float(z.mean()),
           'seeds_positive':int(np.sum(z>EPS)),
           'seeds_negative':int(np.sum(z< -EPS)),
           'seeds_tied':int(np.sum(abs(z)<=EPS)),
           'seedwise_gain_diffs_sec':list(map(float,z))})
    out.mkdir(parents=True,exist_ok=False)
    payload={'stage':'step6b3r-audit-v1','root':str(root),'seeds':list(seeds),
        'notes':['July development internal_test; not independent blind test.',
                 'Six train seeds reuse exactly the same held-out task/date states; do NOT treat seeds as independent observations.',
                 'Effective action maps non-executed chosen candidates to a single NOOP action.',
                 'Signed reward uses Static Repair TD retrospective labels, not Road.replay.',
                 'Task/date folds have few independent groups; no statistical significance claimed.'],
        'summary':results,'per_seed':per_seed,'candidate_diagnostics':diagnostics}
    (out/'audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if changes:
        with (out/'changed_states.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(changes[0]));writer.writeheader();writer.writerows(changes)
    print(json.dumps(results,ensure_ascii=False,indent=2))
    return payload


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--seeds',default='42,43,44,45,46,47')
    ap.add_argument('--methods',default=','.join(METHODS))
    a=ap.parse_args()
    seeds=[int(x) for x in a.seeds.split(',')]
    ms=tuple(a.methods.split(','))
    if 'M0' not in ms or any(m not in METHODS for m in ms):ap.error('Methods must include M0 and known methods')
    audit(a.root,a.output_dir,seeds,ms)
if __name__=='__main__':main()
