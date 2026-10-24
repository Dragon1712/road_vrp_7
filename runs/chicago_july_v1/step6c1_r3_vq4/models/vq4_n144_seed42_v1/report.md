# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +4.944 | 72.339 | 76.4% | 23.6% | 30.0% |
| E0_MATCHED | +7.324 | 69.959 | 70.8% | 21.5% | 30.3% |
| VQ_MEAN | -2.480 | 79.763 | 54.2% | 19.4% | 35.6% |
| RESID_OOF25 | +7.081 | 70.202 | 71.5% | 21.5% | 29.1% |
| RESID_OOF50 | +5.979 | 71.305 | 68.1% | 18.8% | 27.9% |
| PAIR_RANK | -0.271 | 77.554 | 83.3% | 26.4% | 31.4% |
| PAIR_REGRET | +4.509 | 72.774 | 82.6% | 24.3% | 29.0% |
| E0_RISK20 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | -0.021 | 77.304 | 34.0% | 11.1% | 24.5% |
| RESID25_RISK35 | -2.035 | 79.318 | 34.7% | 12.5% | 26.7% |
| PAIR_REGRET_RISK35 | -3.370 | 80.653 | 41.7% | 13.9% | 25.9% |
| E0_SCORE_COVERAGE35 | +4.158 | 73.125 | 35.4% | 12.5% | 24.4% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.219 | 0.322 | 0.043 | 0.555 | 0.549 |
| task_0001 | 0.209 | 0.313 | 0.105 | 0.534 | 0.526 |
| task_0002 | 0.252 | 0.362 | 0.078 | 0.558 | 0.556 |
| task_0003 | 0.246 | 0.378 | 0.068 | 0.510 | 0.508 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +0.675 | 69.093 | 76.4% | 24.3% | 30.8% |
| E0_MATCHED | +3.725 | 66.042 | 70.8% | 18.1% | 24.8% |
| VQ_MEAN | -1.825 | 71.593 | 64.6% | 18.1% | 27.6% |
| RESID_OOF25 | +0.847 | 68.920 | 74.3% | 23.6% | 31.0% |
| RESID_OOF50 | +3.156 | 66.612 | 74.3% | 22.9% | 30.2% |
| PAIR_RANK | -12.730 | 82.497 | 82.6% | 27.1% | 32.7% |
| PAIR_REGRET | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_RISK20 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +3.725 | 66.042 | 70.8% | 18.1% | 24.8% |
| RESID25_RISK35 | +0.847 | 68.920 | 74.3% | 23.6% | 31.0% |
| PAIR_REGRET_RISK35 | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_SCORE_COVERAGE35 | +3.725 | 66.042 | 70.8% | 18.1% | 24.8% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.198 | 0.290 | 0.021 | 0.508 | 0.497 |
| task_0001 | 0.181 | 0.245 | 0.088 | 0.473 | 0.475 |
| task_0002 | 0.218 | 0.338 | 0.035 | 0.441 | 0.445 |
| task_0003 | 0.230 | 0.422 | 0.072 | 0.521 | 0.509 |

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
