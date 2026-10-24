# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 78.035 | 0.0% | 0.0% | 0.0% |
| E0_FULL | -5.142 | 83.176 | 87.5% | 29.2% | 32.7% |
| E0_MATCHED | +11.556 | 66.479 | 79.2% | 22.9% | 28.3% |
| VQ_MEAN | +2.407 | 75.628 | 45.8% | 10.4% | 15.3% |
| RESID_OOF25 | -3.694 | 81.729 | 87.5% | 27.1% | 30.4% |
| RESID_OOF50 | +0.337 | 77.697 | 68.8% | 18.8% | 26.4% |
| PAIR_RANK | +5.545 | 72.490 | 79.2% | 18.7% | 24.4% |
| PAIR_REGRET | +6.242 | 71.793 | 79.2% | 20.8% | 27.4% |
| E0_RISK20 | +0.000 | 78.035 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +5.676 | 72.358 | 54.2% | 14.6% | 20.0% |
| RESID25_RISK35 | -10.042 | 88.076 | 66.7% | 22.9% | 25.4% |
| PAIR_REGRET_RISK35 | -0.730 | 78.765 | 56.2% | 18.8% | 25.2% |
| E0_SCORE_COVERAGE35 | +5.676 | 72.358 | 54.2% | 14.6% | 20.0% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.194 | 0.242 | 0.130 | 0.512 | 0.485 |
| task_0001 | 0.222 | 0.299 | 0.008 | 0.549 | 0.553 |
| task_0002 | 0.265 | 0.507 | 0.139 | 0.545 | 0.544 |
| task_0003 | 0.215 | 0.320 | 0.019 | 0.535 | 0.529 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 75.967 | 0.0% | 0.0% | 0.0% |
| E0_FULL | -0.662 | 76.629 | 87.5% | 25.0% | 28.6% |
| E0_MATCHED | +9.251 | 66.716 | 79.2% | 18.8% | 23.5% |
| VQ_MEAN | +1.174 | 74.793 | 68.8% | 20.8% | 30.1% |
| RESID_OOF25 | +1.741 | 74.226 | 85.4% | 25.0% | 29.3% |
| RESID_OOF50 | +1.266 | 74.701 | 81.2% | 22.9% | 27.8% |
| PAIR_RANK | +5.756 | 70.211 | 75.0% | 16.7% | 21.2% |
| PAIR_REGRET | +1.245 | 74.722 | 83.3% | 20.8% | 24.3% |
| E0_RISK20 | +3.946 | 72.021 | 2.1% | 0.0% | 0.0% |
| E0_RISK35 | +9.251 | 66.716 | 79.2% | 18.8% | 23.5% |
| RESID25_RISK35 | +1.741 | 74.226 | 85.4% | 25.0% | 29.3% |
| PAIR_REGRET_RISK35 | +1.245 | 74.722 | 83.3% | 20.8% | 24.3% |
| E0_SCORE_COVERAGE35 | +9.251 | 66.716 | 79.2% | 18.8% | 23.5% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.173 | 0.208 | 0.088 | 0.463 | 0.431 |
| task_0001 | 0.165 | 0.451 | 0.087 | 0.502 | 0.487 |
| task_0002 | 0.245 | 0.528 | 0.138 | 0.542 | 0.550 |
| task_0003 | 0.203 | 0.293 | 0.026 | 0.503 | 0.524 |

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
