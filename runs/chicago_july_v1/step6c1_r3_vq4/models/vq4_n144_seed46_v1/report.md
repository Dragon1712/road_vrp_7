# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +9.838 | 67.445 | 76.4% | 21.5% | 27.5% |
| E0_MATCHED | +8.114 | 69.169 | 68.7% | 21.5% | 31.8% |
| VQ_MEAN | +3.155 | 74.128 | 49.3% | 13.2% | 26.6% |
| RESID_OOF25 | +5.380 | 71.903 | 70.1% | 20.1% | 27.9% |
| RESID_OOF50 | +2.468 | 74.815 | 66.0% | 19.4% | 29.1% |
| PAIR_RANK | -0.271 | 77.554 | 83.3% | 26.4% | 31.4% |
| PAIR_REGRET | +4.509 | 72.774 | 82.6% | 24.3% | 29.0% |
| E0_RISK20 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +4.070 | 73.213 | 33.3% | 11.8% | 25.9% |
| RESID25_RISK35 | +0.655 | 76.628 | 35.4% | 11.8% | 22.0% |
| PAIR_REGRET_RISK35 | -2.715 | 79.998 | 41.7% | 13.9% | 25.5% |
| E0_SCORE_COVERAGE35 | +0.987 | 76.296 | 34.7% | 13.9% | 26.9% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.219 | 0.326 | 0.043 | 0.555 | 0.549 |
| task_0001 | 0.209 | 0.332 | 0.105 | 0.534 | 0.526 |
| task_0002 | 0.252 | 0.361 | 0.078 | 0.558 | 0.556 |
| task_0003 | 0.246 | 0.370 | 0.069 | 0.510 | 0.508 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +1.508 | 68.260 | 76.4% | 21.5% | 27.4% |
| E0_MATCHED | +4.576 | 65.191 | 68.7% | 17.4% | 25.3% |
| VQ_MEAN | +2.650 | 67.118 | 63.9% | 17.4% | 25.7% |
| RESID_OOF25 | +5.196 | 64.572 | 74.3% | 19.4% | 25.5% |
| RESID_OOF50 | +6.925 | 62.842 | 72.2% | 18.8% | 26.5% |
| PAIR_RANK | -12.730 | 82.497 | 82.6% | 27.1% | 32.7% |
| PAIR_REGRET | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_RISK20 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +4.576 | 65.191 | 68.7% | 17.4% | 25.3% |
| RESID25_RISK35 | +5.196 | 64.572 | 74.3% | 19.4% | 25.5% |
| PAIR_REGRET_RISK35 | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_SCORE_COVERAGE35 | +4.576 | 65.191 | 68.7% | 17.4% | 25.3% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.199 | 0.261 | 0.021 | 0.508 | 0.497 |
| task_0001 | 0.180 | 0.244 | 0.087 | 0.473 | 0.475 |
| task_0002 | 0.218 | 0.346 | 0.034 | 0.441 | 0.445 |
| task_0003 | 0.231 | 0.425 | 0.072 | 0.521 | 0.509 |

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
