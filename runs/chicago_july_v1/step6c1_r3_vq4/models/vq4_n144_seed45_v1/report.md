# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +6.215 | 71.068 | 79.9% | 22.2% | 27.7% |
| E0_MATCHED | +6.562 | 70.721 | 72.2% | 22.2% | 30.0% |
| VQ_MEAN | -5.187 | 82.470 | 48.6% | 18.1% | 37.1% |
| RESID_OOF25 | +7.027 | 70.257 | 75.7% | 22.2% | 29.2% |
| RESID_OOF50 | +8.466 | 68.817 | 70.8% | 19.4% | 26.4% |
| PAIR_RANK | -0.271 | 77.554 | 83.3% | 26.4% | 31.4% |
| PAIR_REGRET | +4.509 | 72.774 | 82.6% | 24.3% | 29.0% |
| E0_RISK20 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +2.399 | 74.884 | 38.9% | 13.2% | 23.7% |
| RESID25_RISK35 | +0.397 | 76.886 | 38.9% | 13.2% | 24.1% |
| PAIR_REGRET_RISK35 | -2.201 | 79.484 | 45.1% | 15.3% | 24.0% |
| E0_SCORE_COVERAGE35 | +5.581 | 71.702 | 34.0% | 11.8% | 23.3% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.219 | 0.324 | 0.041 | 0.555 | 0.549 |
| task_0001 | 0.209 | 0.338 | 0.105 | 0.534 | 0.526 |
| task_0002 | 0.252 | 0.360 | 0.079 | 0.558 | 0.556 |
| task_0003 | 0.246 | 0.373 | 0.068 | 0.510 | 0.508 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +1.101 | 68.666 | 79.9% | 21.5% | 26.7% |
| E0_MATCHED | +1.691 | 68.077 | 72.2% | 20.1% | 26.3% |
| VQ_MEAN | -3.011 | 72.778 | 58.3% | 17.4% | 29.1% |
| RESID_OOF25 | +2.645 | 67.123 | 77.8% | 20.1% | 25.6% |
| RESID_OOF50 | +5.427 | 64.341 | 76.4% | 18.8% | 24.0% |
| PAIR_RANK | -12.730 | 82.497 | 82.6% | 27.1% | 32.7% |
| PAIR_REGRET | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_RISK20 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +1.691 | 68.077 | 72.2% | 20.1% | 26.3% |
| RESID25_RISK35 | +2.645 | 67.123 | 77.8% | 20.1% | 25.6% |
| PAIR_REGRET_RISK35 | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_SCORE_COVERAGE35 | +1.691 | 68.077 | 72.2% | 20.1% | 26.3% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.197 | 0.382 | 0.021 | 0.508 | 0.497 |
| task_0001 | 0.180 | 0.234 | 0.087 | 0.473 | 0.475 |
| task_0002 | 0.218 | 0.345 | 0.033 | 0.441 | 0.445 |
| task_0003 | 0.231 | 0.385 | 0.072 | 0.521 | 0.509 |

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
