# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +5.272 | 72.011 | 78.5% | 24.3% | 30.7% |
| E0_MATCHED | +5.023 | 72.261 | 70.1% | 21.5% | 30.0% |
| VQ_MEAN | -3.104 | 80.387 | 50.0% | 16.0% | 32.1% |
| RESID_OOF25 | +5.046 | 72.238 | 70.1% | 20.8% | 28.8% |
| RESID_OOF50 | +7.325 | 69.958 | 65.3% | 18.8% | 28.4% |
| PAIR_RANK | -0.271 | 77.554 | 83.3% | 26.4% | 31.4% |
| PAIR_REGRET | +4.509 | 72.774 | 82.6% | 24.3% | 29.0% |
| E0_RISK20 | +0.000 | 77.283 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +1.830 | 75.453 | 41.0% | 14.6% | 24.3% |
| RESID25_RISK35 | +0.380 | 76.903 | 40.3% | 14.6% | 25.0% |
| PAIR_REGRET_RISK35 | -2.388 | 79.671 | 47.2% | 16.0% | 24.3% |
| E0_SCORE_COVERAGE35 | +0.108 | 77.175 | 36.8% | 14.6% | 26.6% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.219 | 0.322 | 0.042 | 0.555 | 0.549 |
| task_0001 | 0.209 | 0.352 | 0.105 | 0.534 | 0.526 |
| task_0002 | 0.251 | 0.368 | 0.079 | 0.558 | 0.556 |
| task_0003 | 0.246 | 0.377 | 0.069 | 0.510 | 0.508 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_FULL | -1.403 | 71.171 | 78.5% | 25.0% | 31.5% |
| E0_MATCHED | +1.997 | 67.771 | 70.1% | 20.1% | 27.6% |
| VQ_MEAN | -1.462 | 71.229 | 61.1% | 18.1% | 30.0% |
| RESID_OOF25 | +4.110 | 65.657 | 75.0% | 22.2% | 28.7% |
| RESID_OOF50 | -1.214 | 70.981 | 70.1% | 22.2% | 30.9% |
| PAIR_RANK | -12.730 | 82.497 | 82.6% | 27.1% | 32.7% |
| PAIR_REGRET | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_RISK20 | +0.000 | 69.767 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +1.997 | 67.771 | 70.1% | 20.1% | 27.6% |
| RESID25_RISK35 | +4.110 | 65.657 | 75.0% | 22.2% | 28.7% |
| PAIR_REGRET_RISK35 | -12.277 | 82.045 | 86.8% | 25.7% | 29.6% |
| E0_SCORE_COVERAGE35 | +1.997 | 67.771 | 70.1% | 20.1% | 27.6% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.199 | 0.307 | 0.021 | 0.508 | 0.497 |
| task_0001 | 0.180 | 0.245 | 0.088 | 0.473 | 0.475 |
| task_0002 | 0.218 | 0.340 | 0.034 | 0.441 | 0.445 |
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
