# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +9.784 | 67.499 | 75.0% | 21.5% | 28.5% |
| E0_MATCHED | +4.471 | 72.812 | 70.8% | 23.6% | 33.9% |
| VQ_MEAN | -3.376 | 80.659 | 48.6% | 15.3% | 31.9% |
| RESID_OOF25 | +9.921 | 67.362 | 72.2% | 20.1% | 27.7% |
| RESID_OOF50 | +11.261 | 66.022 | 66.0% | 18.1% | 27.5% |
| PAIR_RANK | -0.271 | 77.554 | 83.3% | 26.4% | 31.4% |
| PAIR_REGRET | +4.509 | 72.774 | 82.6% | 24.3% | 29.0% |
| E0_RISK20 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +2.183 | 75.100 | 38.2% | 13.9% | 25.1% |
| RESID25_RISK35 | +2.563 | 74.720 | 39.6% | 12.5% | 21.7% |
| PAIR_REGRET_RISK35 | -1.819 | 79.103 | 43.8% | 14.6% | 25.9% |
| E0_SCORE_COVERAGE35 | +2.189 | 75.094 | 31.2% | 11.8% | 26.1% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.219 | 0.324 | 0.042 | 0.555 | 0.549 |
| task_0001 | 0.209 | 0.327 | 0.105 | 0.534 | 0.526 |
| task_0002 | 0.251 | 0.371 | 0.079 | 0.558 | 0.556 |
| task_0003 | 0.246 | 0.387 | 0.070 | 0.510 | 0.508 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +4.103 | 65.664 | 75.0% | 20.8% | 27.7% |
| E0_MATCHED | +1.083 | 68.685 | 70.8% | 20.8% | 29.5% |
| VQ_MEAN | -0.045 | 69.813 | 61.8% | 20.1% | 31.5% |
| RESID_OOF25 | +6.254 | 63.513 | 72.2% | 20.1% | 28.1% |
| RESID_OOF50 | +2.964 | 66.803 | 68.8% | 19.4% | 28.5% |
| PAIR_RANK | -12.730 | 82.497 | 82.6% | 27.1% | 32.7% |
| PAIR_REGRET | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_RISK20 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +1.083 | 68.685 | 70.8% | 20.8% | 29.5% |
| RESID25_RISK35 | +6.254 | 63.513 | 72.2% | 20.1% | 28.1% |
| PAIR_REGRET_RISK35 | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_SCORE_COVERAGE35 | +1.083 | 68.685 | 70.8% | 20.8% | 29.5% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.198 | 0.355 | 0.023 | 0.508 | 0.497 |
| task_0001 | 0.180 | 0.254 | 0.086 | 0.473 | 0.475 |
| task_0002 | 0.218 | 0.332 | 0.035 | 0.441 | 0.445 |
| task_0003 | 0.230 | 0.435 | 0.072 | 0.521 | 0.509 |

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
