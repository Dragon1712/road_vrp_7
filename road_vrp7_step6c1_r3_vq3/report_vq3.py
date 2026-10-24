"""Compact VQ3 results: never claim online performance from restart labels."""
from __future__ import annotations

def markdown(d):
    t=['# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）','',
       '这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。',
       '外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。','']
    for h in d['horizons']:
        h=str(h);t.extend([f'## H={h}','',
                          '| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |',
                          '|---|---:|---:|---:|---:|---:|'])
        for name,r in d['macro_task_metrics'][h].items():
            t.append(f"| {name} | {r['mean_gain_sec']:+.3f} | {r['mean_regret_sec']:.3f} | {r['override_rate']:.1%} | {r['harm_gt40_per_state']:.1%} | {r['harm_gt40_per_override']:.1%} |")
        t.extend(['','### 按留出任务的风险分类与排序诊断','',
            '| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |',
            '|---|---:|---:|---:|---:|---:|'])
        for task,z in d['folds'][h].items():
            v=z['candidate_risk_diagnostics'];k=z['pairwise_accuracy_including_C3'];fmt=lambda x: '—' if x is None else f'{x:.3f}'
            t.append(f"| {task} | {v['brier']:.3f} | {fmt(v['average_precision'])} | {v['ece5']:.3f} | {fmt(k['PAIR_RANK'])} | {fmt(k['PAIR_REGRET'])} |")
        t.extend(['','### 覆盖率匹配负对照','',
            'E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。',''])
    t.extend(['## 必须注意的边界','',*(f'- {s}' for s in d['limits']), ''])
    return '\n'.join(t)
