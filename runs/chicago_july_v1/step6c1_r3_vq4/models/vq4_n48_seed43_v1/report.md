# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 78.035 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +5.237 | 72.798 | 85.4% | 27.1% | 30.8% |
| E0_MATCHED | +8.617 | 69.418 | 81.2% | 25.0% | 30.1% |
| VQ_MEAN | -4.107 | 82.142 | 54.2% | 12.5% | 20.7% |
| RESID_OOF25 | +6.342 | 71.692 | 83.3% | 27.1% | 31.4% |
| RESID_OOF50 | +4.218 | 73.817 | 77.1% | 22.9% | 29.0% |
| PAIR_RANK | +5.545 | 72.490 | 79.2% | 18.7% | 24.4% |
| PAIR_REGRET | +6.242 | 71.793 | 79.2% | 20.8% | 27.4% |
| E0_RISK20 | +0.000 | 78.035 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +3.846 | 74.189 | 56.2% | 16.7% | 21.8% |
| RESID25_RISK35 | -9.384 | 87.419 | 62.5% | 22.9% | 26.4% |
| PAIR_REGRET_RISK35 | -0.730 | 78.765 | 56.2% | 18.8% | 25.2% |
| E0_SCORE_COVERAGE35 | +3.846 | 74.189 | 56.2% | 16.7% | 21.8% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.194 | 0.247 | 0.131 | 0.512 | 0.485 |
| task_0001 | 0.223 | 0.283 | 0.009 | 0.549 | 0.553 |
| task_0002 | 0.265 | 0.505 | 0.139 | 0.545 | 0.544 |
| task_0003 | 0.215 | 0.327 | 0.021 | 0.535 | 0.529 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 75.967 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +3.803 | 72.164 | 85.4% | 25.0% | 29.0% |
| E0_MATCHED | +8.824 | 67.143 | 81.2% | 18.8% | 23.2% |
| VQ_MEAN | -6.128 | 82.095 | 64.6% | 18.7% | 29.2% |
| RESID_OOF25 | +3.679 | 72.288 | 85.4% | 22.9% | 26.5% |
| RESID_OOF50 | +6.029 | 69.938 | 81.2% | 20.8% | 25.0% |
| PAIR_RANK | +5.756 | 70.211 | 75.0% | 16.7% | 21.2% |
| PAIR_REGRET | +1.245 | 74.722 | 83.3% | 20.8% | 24.3% |
| E0_RISK20 | +6.546 | 69.421 | 6.2% | 0.0% | 0.0% |
| E0_RISK35 | +8.824 | 67.143 | 81.2% | 18.8% | 23.2% |
| RESID25_RISK35 | +3.679 | 72.288 | 85.4% | 22.9% | 26.5% |
| PAIR_REGRET_RISK35 | +1.245 | 74.722 | 83.3% | 20.8% | 24.3% |
| E0_SCORE_COVERAGE35 | +8.824 | 67.143 | 81.2% | 18.8% | 23.2% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.173 | 0.222 | 0.087 | 0.463 | 0.431 |
| task_0001 | 0.165 | 0.294 | 0.087 | 0.502 | 0.487 |
| task_0002 | 0.244 | 0.558 | 0.140 | 0.542 | 0.550 |
| task_0003 | 0.203 | 0.297 | 0.027 | 0.503 | 0.524 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## 必须注意的边界

- Only original TRAIN tasks; outer task leave-one-out. No R2 validation or internal_test labels.
- LOTO evaluation contains only 4 correlated tasks x 12 states, NOT independent blind validation.
- Search restart surrogate is NOT continuous 30s ALNS action value.
- Three random restarts create noisy mean-harm labels; probability has no formal safety guarantee.
- Platt calibration trained on only 3 outer train tasks; even task-OOF Platt can be unstable.
- Online latency not measured; no 30-second improvement claim.
- All alpha and risk cutoffs were fixed BEFORE LOTO evaluation, never tuned on heldout outcomes.
