# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_FULL | -1.595 | 78.878 | 76.4% | 25.7% | 33.6% |
| E0_MATCHED | +7.006 | 70.277 | 73.6% | 21.5% | 29.8% |
| VQ_MEAN | +0.745 | 76.538 | 52.1% | 16.7% | 32.4% |
| RESID_OOF25 | +0.794 | 76.490 | 74.3% | 24.3% | 32.8% |
| RESID_OOF50 | +2.817 | 74.466 | 67.4% | 20.8% | 31.4% |
| PAIR_RANK | -0.271 | 77.554 | 83.3% | 26.4% | 31.4% |
| PAIR_REGRET | +4.509 | 72.774 | 82.6% | 24.3% | 29.0% |
| E0_RISK20 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +1.336 | 75.947 | 36.8% | 13.9% | 27.1% |
| RESID25_RISK35 | +1.409 | 75.874 | 37.5% | 13.2% | 23.6% |
| PAIR_REGRET_RISK35 | -2.926 | 80.209 | 42.4% | 13.9% | 24.2% |
| E0_SCORE_COVERAGE35 | +3.084 | 74.199 | 31.9% | 11.1% | 24.4% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.219 | 0.327 | 0.042 | 0.555 | 0.549 |
| task_0001 | 0.208 | 0.330 | 0.104 | 0.534 | 0.526 |
| task_0002 | 0.252 | 0.361 | 0.077 | 0.558 | 0.556 |
| task_0003 | 0.246 | 0.381 | 0.068 | 0.510 | 0.508 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_FULL | -0.689 | 70.456 | 76.4% | 24.3% | 31.4% |
| E0_MATCHED | +2.905 | 66.862 | 73.6% | 19.4% | 26.4% |
| VQ_MEAN | -1.525 | 71.293 | 63.9% | 19.4% | 30.9% |
| RESID_OOF25 | -0.880 | 70.648 | 75.0% | 25.7% | 33.9% |
| RESID_OOF50 | -2.099 | 71.866 | 72.9% | 23.6% | 32.4% |
| PAIR_RANK | -12.730 | 82.497 | 82.6% | 27.1% | 32.7% |
| PAIR_REGRET | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_RISK20 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +2.905 | 66.862 | 73.6% | 19.4% | 26.4% |
| RESID25_RISK35 | -0.880 | 70.648 | 75.0% | 25.7% | 33.9% |
| PAIR_REGRET_RISK35 | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_SCORE_COVERAGE35 | +2.905 | 66.862 | 73.6% | 19.4% | 26.4% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.199 | 0.278 | 0.021 | 0.508 | 0.497 |
| task_0001 | 0.180 | 0.234 | 0.086 | 0.473 | 0.475 |
| task_0002 | 0.218 | 0.328 | 0.035 | 0.441 | 0.445 |
| task_0003 | 0.231 | 0.421 | 0.072 | 0.521 | 0.509 |

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
