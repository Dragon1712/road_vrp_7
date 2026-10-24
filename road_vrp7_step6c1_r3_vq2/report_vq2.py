"""Plain Markdown report generated from VQ2 JSON; presentation-only."""
from __future__ import annotations

def markdown_report(r):
    lines=['# R3-VQ2 等样本 / 多步残差 / 风险校准',
           '', '**解释边界**：仅原Train的四任务LOTO；多步标签为C3-restart代理，不是在线30秒真实收益。',
           '固定风险系数，不根据留出任务结果优化。', '']
    for h in r['horizons']:
        h=str(h);lines.append('## H='+h)
        lines.extend(['','| 方法 | 多步Road平均收益↑ | Regret↓ | 覆盖率 | 严重大损失/状态 |',
                '|---|---:|---:|---:|---:|'])
        for m,v in r['macro_task_metrics'][h].items():
            lines.append(f"| {m} | {v['mean_gain_sec']:+.3f} | {v['mean_regret_sec']:.3f} | {v['override_rate']*100:.2f}% | {v['harm_gt40_per_state']*100:.2f}% |")
        def gain(m):return r['macro_task_metrics'][h][m]['mean_gain_sec']
        lines += ['',f"**目标差异（同样本）**：VQ_MEAN − E0_MATCHED = {gain('VQ_MEAN')-gain('E0_MATCHED'):+.3f} 秒/状态。",
                  f"**数据量效应**：E0_FULL − E0_MATCHED = {gain('E0_FULL')-gain('E0_MATCHED'):+.3f} 秒/状态。",
                  f"**残差贡献**：RESID_OOF50 − E0_FULL = {gain('RESID_OOF50')-gain('E0_FULL'):+.3f} 秒/状态。",'']
        lines.extend(['| 留出训练任务 | E0_FULL | E0_MATCHED | VQ_MEAN | RESID_OOF50 | SD预测MAE |',
             '|---|---:|---:|---:|---:|---:|'])
        for t,z in r['folds'][h].items():
            ps=z['policies'];d=z['prediction_diagnostics']
            lines.append(f"| {t} | {ps['E0_FULL']['mean_gain_sec']:+.2f} | {ps['E0_MATCHED']['mean_gain_sec']:+.2f} | {ps['VQ_MEAN']['mean_gain_sec']:+.2f} | {ps['RESID_OOF50']['mean_gain_sec']:+.2f} | {d['sd_mae_sec']:.2f} |")
        lines += ['', '**注意**：`VQ_LCB`的SD描述后续搜索种子的随机性，不是模型参数不确定性；此处不得声称形式化安全保证。','']
    return '\n'.join(lines)+'\n'
