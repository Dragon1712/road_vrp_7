"""R4-M Train-only, nested task-held-out risk calibration for Section delegation.

This module NEVER runs ALNS, uses Validation/Test data, or asserts 30-second gain.
It loads original R4-L STRICT source checks; all calibrations are fit using
only outer-training tasks, with inner leave-one-task-out out-of-fold predictions.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import json, hashlib, math, sys

ROOT = Path(__file__).resolve().parents[1]
for d in ('road_vrp7_step6c1_r4l', 'road_vrp7_step6c1_r4i', 'road_vrp7_step6c1_r4k'):
    q = str(ROOT / d)
    if q not in sys.path: sys.path.insert(0, q)
import r4l_core as r4l
import r4i_core as r4i

STAGE = 'R4M-TRAIN-NESTED-RISK-CALIBRATED-SECTION-v1'
EPS = 1e-8
# Policies are pre-registered; the outer heldout results are NEVER used to tune them.
POLICIES = ('ridge_ref', 'lcb_q50', 'lcb_q75', 'lcb_q90', 'dual_lcb_q75')
CAL_Q = {'lcb_q50': 0.50, 'lcb_q75': 0.75, 'lcb_q90': 0.90,
         'dual_lcb_q75': 0.75}


def need(flag, msg):
    if not flag: raise ValueError(msg)


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf8')
    tmp.replace(path)


def source_preflight(paths):
    """Fail closed if code/data provenance differs from completed R4-L."""
    p = Path(paths['r4l_result'])
    report, prov = read(p/'report.json'), read(p/'provenance.json')
    need(report.get('stage') == r4l.STAGE and report.get('status') == 'COMPLETE'
         and report.get('data_role') == 'TRAIN_ONLY', 'R4-L report incomplete/non-TRAIN')
    need(prov.get('stage') == r4l.STAGE and sha(p/'report.json') == prov.get('report_sha256')
         and sha(p/'fold_predictions.jsonl') == prov.get('fold_predictions_sha256'),
         'R4-L report/OOF provenance SHA differs')
    pins = {'r4l_core.py': ROOT/'road_vrp7_step6c1_r4l/r4l_core.py',
            'probe_r4l.py': ROOT/'road_vrp7_step6c1_r4l/probe_r4l.py',
            'r4i_core.py': ROOT/'road_vrp7_step6c1_r4i/r4i_core.py',
            'r4k_core.py': ROOT/'road_vrp7_step6c1_r4k/r4k_core.py'}
    for name, file in pins.items():
        need(file.is_file() and sha(file) == prov['code_sha256'][name],
             f'Frozen upstream code SHA changed: {name}')
    old, old_prov = r4l.read_old(paths['old_cf'], paths['old_capture'], paths['old_prefix'])
    new, new_prov = r4l.read_new(paths['new_r4k'], paths['new_capture'],
                                  paths['new_cf'], paths['new_prefix'])
    need(old_prov == report['sources']['original'] and new_prov == report['sources']['new'],
         'Underlying R4-H/R4-K source differs from R4-L report')
    need(len(old) == 565 and len(new) == 1007, 'Expected frozen 565/1007 candidate actions')
    records = old + new
    need({r['snap'] for r in old}.isdisjoint({r['snap'] for r in new}),
         'Overlapping old/new snapshot identities')
    need(len(r4l.group_records(records)) == 200 and
         len({(r['task'],r['snap'][2],r['snap'][3]) for r in records}) == 100,
         'Not original R4-L set of 200 scenario states / 100 coarse starts')
    need(len(r4l.make_splits(records)) == 6, 'Unexpected data split protocol')
    # The original R4-L source report contains the full role and split provenance.
    return records, {'r4l_report_sha256':sha(p/'report.json'),
                     'r4l_fold_predictions_sha256':sha(p/'fold_predictions.jsonl'),
                     'r4l_provenance_sha256':sha(p/'provenance.json'),
                     'original':old_prov['source_sha256'],
                     'new':new_prov['source_sha256']}


def feature_records(records, feature_mode):
    """Exactly reuse R4-L's predecision feature mask and coarse group weights.

    R4-L's with_mask strips the trailing full-precision SA temperature from
    the weight-group identity to avoid near-duplicated scenario overcounting.
    We keep that *unchanged* for reference and calibrated comparisons.
    """
    need(feature_mode in ('full','geometry'), 'Unknown feature setting')
    return r4l.with_mask(records, feature_mode)


def fit(records, target, *, shuffle=False, seed=0):
    need(target in ('k500','k200'), 'Invalid fit target')
    need(records, 'Empty fitting partition')
    return r4i.ridge_predictors(records, target=target,
                                 strength=100., seed=seed, shuffle_labels=shuffle)


def quantile(values, q):
    """Deterministic conservative empirical upper quantile; not a coverage guarantee."""
    need(0 < q < 1 and values, 'Invalid quantile/calibration sample')
    need(all(math.isfinite(float(v)) for v in values), 'Nonfinite calibration error')
    a = sorted(float(x) for x in values)
    return max(0., a[min(len(a)-1, math.ceil(q*len(a))-1)])


def group_actions(records):
    grp=defaultdict(list)
    for r in records:grp[tuple(r['snap'])].append(r)
    need(all(1<=len(v)<=8 for v in grp.values()), 'Unexpected alternative count')
    return dict(sorted(grp.items()))


def scores(actions, predictor500, predictor200=None):
    for r in actions:
        p500=float(r4i.predict(predictor500,r['features']))
        p200=float(r4i.predict(predictor200,r['features'])) if predictor200 is not None else p500
        need(math.isfinite(p500) and math.isfinite(p200), 'Nonfinite predictor output')
        yield (r,p500,p200)


def calibrate_inner(outer_train, *, policy, shuffle, seed):
    """Nested task-out calibration: never read outer-heldout actions or labels."""
    need(policy in CAL_Q,'Cannot calibrate plain reference baseline')
    tasks=sorted({r['task'] for r in outer_train})
    need(len(tasks) >= 3,'At least 3 TRAIN tasks needed for inner task-out CV')
    is_dual = policy.startswith('dual_')
    errors500=[];errors200=[];support_tasks=set();potential=0
    for i, heldtask in enumerate(tasks):
        inner_train=[r for r in outer_train if r['task']!=heldtask]
        inner_valid=[r for r in outer_train if r['task']==heldtask]
        # Always fit on independent task groups. Fixed seed from training partition only.
        w500=fit(inner_train,'k500',shuffle=shuffle,seed=seed+11*i)
        w200=fit(inner_train,'k200',shuffle=shuffle,seed=seed+11*i+1) if is_dual else None
        for snap,acts in group_actions(inner_valid).items():
            if snap[3] > 2000:continue
            ranked=list(scores(acts,w500,w200))
            if is_dual:
                r,p5,p2=max(ranked,key=lambda v:(min(v[1],v[2]),v[0]['arm']))
                check=min(p5,p2)
            else:
                r,p5,p2=max(ranked,key=lambda v:(v[1],v[0]['arm']))
                check=p5
            if check <= 0:continue
            potential+=1;support_tasks.add(heldtask)
            errors500.append(max(0.,p5-float(r['targets']['k500'])))
            if is_dual: errors200.append(max(0.,p2-float(r['targets']['k200'])))
    # With very little genuinely task-independent support, fail closed and
    # avoid treating an arbitrary zero quantile as a calibrated safety bound.
    valid=len(errors500)>=5 and len(support_tasks)>=2
    return {'valid':valid,'n_selected_for_calibration':len(errors500),
            'n_inner_tasks_with_positive_score':len(support_tasks),
            'q500':quantile(errors500,CAL_Q[policy]) if valid else None,
            'q200':quantile(errors200,CAL_Q[policy]) if valid and is_dual else None,
            'q_level':CAL_Q[policy], 'inner_positive_score_snapshots':potential}


def select_fold(held, w500, w200, policy, calibration):
    picks=[]
    for snap, acts in group_actions(held).items():
        allscores=list(scores(acts,w500,w200))
        if policy == 'ridge_ref':
            a,p5,p2=max(allscores,key=lambda v:(v[1],v[0]['arm']))
            score=p5
        elif calibration['valid']:
            if policy.startswith('dual_'):
                a,p5,p2=max(allscores,key=lambda v:(min(v[1]-calibration['q500'],v[2]-calibration['q200']),v[0]['arm']))
                score=min(p5-calibration['q500'],p2-calibration['q200'])
            else:
                a,p5,p2=max(allscores,key=lambda v:(v[1],v[0]['arm']))
                score=p5-calibration['q500']
        else:
            a,p5,p2=max(allscores,key=lambda v:(v[1],v[0]['arm']))
            score=None
        chosen=a if score is not None and score>0 and snap[3]<=2000 else None
        gain=float(chosen['targets']['k500']) if chosen else 0.
        oracle=max([0.]+[float(x['targets']['k500']) for x in acts])
        picks.append({'snapshot_id':list(snap),'selected_arm':chosen['arm'] if chosen else 'C3',
              'gain_k500_sec':gain,'oracle_sec':oracle,
              'random_expected_sec':sum(float(x['targets']['k500']) for x in acts)/(len(acts)+1),
              'predicted_margin_sec':score,'n_alternatives':len(acts),
              'eligible_early':bool(snap[3]<=2000)})
    return picks


def summarize_picks(picks):
    need(picks,'Empty heldout decisions')
    gain=[x['gain_k500_sec'] for x in picks]; n=len(picks)
    return {'n_scenario_snapshots':n,
            'n_delegations':sum(x['selected_arm']!='C3' for x in picks),
            'n_positive':sum(x>EPS for x in gain),
            'n_negative':sum(x < -EPS for x in gain),
            'n_zero':sum(abs(x)<=EPS for x in gain),
            'total_gain_k500_sec':sum(gain),
            'mean_gain_k500_sec':sum(gain)/n,
            'sum_harm_sec':sum(min(0.,x) for x in gain),
            'worst_harm_sec':min(gain),
            'oracle_mean_gain_sec':sum(x['oracle_sec'] for x in picks)/n,
            'random_mean_expected_sec':sum(x['random_expected_sec'] for x in picks)/n}


def evaluate(records):
    """All outer heldout labels used ONLY for final evaluation, not calibration."""
    results=[];details=[]
    for i,(split,train,held,typ) in enumerate(r4l.make_splits(records)):
        need(not ({r['snap'] for r in train} & {r['snap'] for r in held}), 'Snapshot leakage')
        for feature in ('full','geometry'):
            # Reference exactly matches original R4-L weighting/feature mask.
            fit_train_ref=feature_records(train,feature)
            eval_held=feature_records(held,feature)
            fit_train=fit_train_ref
            for shuffle in (False,True):
                seed=20261025+13*i+(1 if shuffle else 0)
                ref500=fit(fit_train_ref,'k500',shuffle=shuffle,seed=seed)
                w500=fit(fit_train,'k500',shuffle=shuffle,seed=seed)
                w200=fit(fit_train,'k200',shuffle=shuffle,seed=seed+1)
                for policy in POLICIES:
                    cal=calibrate_inner(fit_train,policy=policy,shuffle=shuffle,seed=seed+100) if policy!='ridge_ref' else None
                    picks=select_fold(eval_held,ref500 if policy=='ridge_ref' else w500,
                                      w200 if policy.startswith('dual_') else None,
                                      policy,cal)
                    row={'split':split,'split_type':typ,'feature_set':feature,
                         'shuffle_train_labels':shuffle,'policy':policy,
                         'calibration':cal,
                         'train_tasks':sorted({r['task'] for r in train}),
                         'heldout_tasks':sorted({r['task'] for r in held}),
                         'train_seeds':sorted({int(r['snap'][2]) for r in train}),
                         'heldout_seeds':sorted({int(r['snap'][2]) for r in held}),
                         **summarize_picks(picks)}
                    results.append(row)
                    for p in picks:details.append({'evaluation':{'split':split,'feature_set':feature,
                          'shuffle_train_labels':shuffle,'policy':policy},**p})
    return results,details


def report_md(report):
    ss=report['results'];out=['# R4-M Train内部嵌套风险校准 Section选择器','',
     '数据：原R4-H + R4-K的 **1,572候选 / 200场景快照 / 100粗粒度搜索起点**。',
     '所有阈值和经验分位数只由外层Train中的任务留一OOF预测估计；外层留组仅用于报告。',
     '所有结果是固定K500离线Road.replay收益，**不是独立Validation/Test或30秒在线结果**。','',
     '## 预注册主比较：全特征、未打乱标签','',
     '| 留组 | 策略 | 校准有效 | 校准样本 | 委托 | 正/负 | K500均值(s) | 伤害总和(s) |',
     '|---|---|---|---:|---:|---|---:|---:|']
    for x in ss:
        if x['feature_set']!='full' or x['shuffle_train_labels']:continue
        c=x['calibration']
        out.append(f"| {x['split']} | {x['policy']} | {('N/A' if c is None else c['valid'])} | {('—' if c is None else c['n_selected_for_calibration'])} | {x['n_delegations']} | {x['n_positive']}/{x['n_negative']} | {x['mean_gain_k500_sec']:+.3f} | {x['sum_harm_sec']:+.3f} |")
    out+=['','## 报告解释边界','',
      '- 阈值0与q50/q75/q90均为事先定义的Train-only诊断，禁止根据本轮留出收益挑选最优参数并宣称外部泛化。',
      '- 经验任务留一残差分位数**不提供形式化分布外覆盖保证**；校准失败时策略严格回退C3。',
      '- 单独报告基准Ridge、Shuffle、几何特征消融以及任务和种子迁移；跨场景的重复起点不视为独立任务。',
      '- 不会修改官方ALNS、源数据或任何R4-I/L产物；没有模型被部署到30秒优化器。','']
    return '\n'.join(out)
