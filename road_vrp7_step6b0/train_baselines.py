#!/usr/bin/env python3
"""Leakage-audited supervised baselines on frozen Step6-A datasets.

Three model policies: static-only RF, static+predeparture-history RF, and
static score heuristic. Train/val/internal-test are GROUPED by task OR date.
Threshold to execute/no-op is tuned ONLY on validation; no hindsight gate.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

BASE = Path('runs/chicago_july_v1/step6a/section_utility_1000_v1')
THRESHOLDS = (0., 0.01, .02, .04, .06, .08, .1, .15, .2, .3, .5, .7, .9, .99, 1.01)


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_dataset(src: Path):
    adir = src / 'analysis'
    files = {'X': adir/'features_X.npz', 'y': adir/'labels_y.npz',
             'groups': adir/'groups.csv', 'schema': adir/'dataset_schema.json'}
    for p in files.values():
        if not p.is_file(): raise FileNotFoundError(p)
    with np.load(files['X'], allow_pickle=False) as f: X = f['X'].astype(np.float64)
    with np.load(files['y'], allow_pickle=False) as f: Y = f['y'].astype(np.float64)
    with files['groups'].open(newline='', encoding='utf-8') as f:
        groups = list(csv.DictReader(f))
    schema = json.loads(files['schema'].read_text(encoding='utf-8'))
    names = list(schema['feature_names']); labels = list(schema['label_names'])
    assert X.ndim == 2 and Y.ndim == 2
    if X.shape[0] != Y.shape[0] or X.shape[0] != len(groups): raise ValueError('Feature-label-group row mismatch')
    if X.shape[1] != len(names) or Y.shape[1] != len(labels): raise ValueError('Schema mismatch')
    if not np.isfinite(X).all() or not np.isfinite(Y).all(): raise ValueError('NaN / inf detected')
    if len({r['sample_id'] for r in groups}) != len(groups): raise ValueError('Repeated sample identifier')
    if 'td_gain_sec' not in labels: raise ValueError('Missing signed TD gain')
    blacklist = ('td_gain', 'td_advantage', 'static_repair_', 'original_td_', 'repaired_routes')
    if any(any(x in name for x in blacklist) for name in names):
        raise ValueError('Leakage in feature schema')
    static_idx = [j for j, name in enumerate(names) if not name.startswith('history_')]
    history_idx = [j for j, name in enumerate(names) if name.startswith('history_')]
    if not history_idx or len(static_idx) + len(history_idx) != len(names): raise ValueError('No traffic ablation')
    return X, Y[:, labels.index('td_gain_sec')], groups, names, static_idx, history_idx, files


def verify_group_split(groups, protocol):
    role_name = {'task': 'task_role', 'date': 'date_role'}[protocol]
    roles = np.asarray([r[role_name] for r in groups])
    if set(roles) != {'train', 'val', 'internal_test'}: raise ValueError(f'Incomplete roles: {set(roles)}')
    grouping = {'task': 'task_id', 'date': 'date'}[protocol]
    # No task or date, and no current state, may span splits.
    seen = {}
    for r in groups:
        for key in (grouping, 'state_id'):
            k = (key, r[key])
            if k in seen and seen[k] != r[role_name]:
                raise ValueError(f'Leakage: {key}={r[key]} in {seen[k]} and {r[role_name]}')
            seen[k] = r[role_name]
    return roles


def fit_model(X, y, seed):
    cls = (y > 1e-7).astype(int)
    if len(np.unique(cls)) < 2: return None, float(np.mean(cls))
    model = RandomForestClassifier(
        n_estimators=220, min_samples_leaf=8, max_features='sqrt',
        class_weight='balanced_subsample', n_jobs=1, random_state=seed)
    model.fit(X, cls)
    return model, None


def scores(model, constant, X):
    return (np.full(len(X), constant, dtype=float)
            if model is None else model.predict_proba(X)[:, list(model.classes_).index(1)])


def state_policy(y, prob, groups, threshold):
    """Select highest scoring action in each state. Execute iff score>=threshold.

    No hindsight clipping. Signed gain may be negative when an action executes.
    """
    items = defaultdict(list)
    for i, r in enumerate(groups): items[r['state_id']].append(i)
    gained = []; oracle = []; picked = []; selected_positive = []
    for state, indices in items.items():
        ids = np.asarray(indices)
        chosen = int(ids[int(np.argmax(prob[ids]))])
        execute = bool(prob[chosen] >= threshold)
        gained.append(float(y[chosen]) if execute else 0.)
        oracle.append(max(0., float(np.max(y[ids]))))
        picked.append(int(execute))
        selected_positive.append(int(y[chosen] > 1e-7) if execute else 0)
    return {'states': len(items), 'executed_states': sum(picked),
            'execution_rate': float(np.mean(picked)),
            'mean_actual_gain_sec': float(np.mean(gained)),
            'total_actual_gain_sec': float(np.sum(gained)),
            'mean_oracle_noop_upper_bound_sec': float(np.mean(oracle)),
            'positive_executions': int(sum(selected_positive)),
            'negative_or_zero_executions': sum(picked) - int(sum(selected_positive))}


def tune_threshold(y, prob, groups):
    results = [(t, state_policy(y, prob, groups, t)) for t in THRESHOLDS]
    # Conservative tie-break: prefer larger threshold (fewer executed moves).
    return max(results, key=lambda x: (x[1]['mean_actual_gain_sec'], x[0]))[0]


def safe_pr(y, p):
    cls = (y > 1e-7).astype(int)
    if len(np.unique(cls)) < 2: return None, None
    return float(average_precision_score(cls,p)), float(roc_auc_score(cls,p))


def report_one(X, y, groups, names, static_idx, history_idx, protocol, seed):
    roles = verify_group_split(groups, protocol)
    masks = {name: roles == name for name in ('train', 'val', 'internal_test')}
    out = {'protocol': protocol, 'counts': {}, 'methods': {},
           'positive_labels_total': int((y > 1e-7).sum())}
    for k, v in masks.items():
        out['counts'][k] = {'candidates': int(v.sum()),
                            'states': len({r['state_id'] for i,r in enumerate(groups) if v[i]}),
                            'positives': int((y[v] > 1e-7).sum())}
    if out['counts']['train']['positives'] < 2:
        raise ValueError('Too few positive training samples; do not fit classifier')
    candidates = {
        'B1_static_RF': static_idx,
        'B2_static_plus_history_RF': static_idx + history_idx,
    }
    for modelname, indices in candidates.items():
        model, constant = fit_model(X[masks['train']][:, indices], y[masks['train']], seed)
        pp = {part:scores(model, constant, X[mask][:, indices]) for part, mask in masks.items()}
        val_y = y[masks['val']]; test_y = y[masks['internal_test']]
        threshold = tune_threshold(val_y, pp['val'], [r for i,r in enumerate(groups) if masks['val'][i]])
        test_groups = [r for i,r in enumerate(groups) if masks['internal_test'][i]]
        pr, roc = safe_pr(test_y, pp['internal_test'])
        out['methods'][modelname] = {
            'static_features': [names[j] for j in static_idx],
            'history_feature_count': len(history_idx) if history_idx[0] in indices else 0,
            'threshold_chosen_on_validation': threshold,
            'test_PR_AUC': pr, 'test_ROC_AUC': roc,
            'test_positive_base_rate': float(np.mean(test_y > 1e-7)),
            'test_ungated_optimistic_relabelled_gain_sec': float(np.mean(np.maximum(0,test_y))),
            'test_policy': state_policy(test_y, pp['internal_test'], test_groups, threshold),
            'validation_policy': state_policy(val_y, pp['val'], [r for i,r in enumerate(groups) if masks['val'][i]], threshold),
        }
    # Non-learning static section score; arbitrary large ranks mean more important in original C3.
    col = names.index('static_section_cost_sec')
    gs = [r for i,r in enumerate(groups) if masks['internal_test'][i]]
    py = y[masks['internal_test']]
    baseline = state_policy(py, X[masks['internal_test'], col], gs, threshold=-float('inf'))
    out['methods']['B0_static_score_unconditional'] = {'test_policy': baseline,
        'note': 'Unconditional heuristic executes even negative moves; diagnostic only, not ALNS'}
    out['methods']['B0_noop'] = {'test_policy': state_policy(py, np.zeros(len(py)), gs, threshold=1.01)}
    return out


def run(source: Path, output: Path, seed: int, protocols):
    if output.exists(): raise FileExistsError(f'Use NEW output directory: {output}')
    X, y, groups, names, stat, hist, files = load_dataset(source)
    records = [report_one(X,y,groups,names,stat,hist,p,seed) for p in protocols]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'stage':'step6b0-supervised-v1','source':str(source),
        'source_sha256':{k:digest(v) for k,v in files.items()},
        'experiment_sha256':digest(Path(__file__)),
        'seed':seed,'protocols':list(protocols),'n':len(y),
        'feature_count':len(names),'static_features':len(stat),'traffic_features':len(hist),
        'warning':('July development ONLY; Oracle-derived source route states can contain future'
                   ' information. No prospective deployment or unseen-month claims.')}
    (output/'config.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (output/'results.json').write_text(json.dumps(records,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'status':'COMPLETE','output':str(output),'reports':[{'protocol':r['protocol'], 'counts':r['counts'],
        'test': {k: {'PR_AUC':v.get('test_PR_AUC'),'threshold':v.get('threshold_chosen_on_validation'),
                    'policy_gain_sec':v['test_policy']['mean_actual_gain_sec']}
                for k,v in r['methods'].items()}} for r in records]},indent=2,ensure_ascii=False))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=BASE)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--protocol',choices=['both','task','date'],default='both')
    a=p.parse_args();run(a.source_dir,a.output_dir,a.seed,('task','date') if a.protocol=='both' else (a.protocol,))
if __name__=='__main__':main()
