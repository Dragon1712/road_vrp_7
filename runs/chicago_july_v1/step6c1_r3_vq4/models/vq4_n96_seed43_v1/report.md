# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 79.345 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +4.757 | 74.589 | 75.0% | 21.9% | 28.2% |
| E0_MATCHED | +2.906 | 76.439 | 76.0% | 22.9% | 30.3% |
| VQ_MEAN | +2.174 | 77.171 | 61.5% | 17.7% | 27.2% |
| RESID_OOF25 | +5.979 | 73.367 | 71.9% | 20.8% | 28.2% |
| RESID_OOF50 | +2.379 | 76.966 | 68.8% | 19.8% | 27.8% |
| PAIR_RANK | +0.730 | 78.615 | 86.5% | 24.0% | 28.2% |
| PAIR_REGRET | +0.349 | 78.996 | 86.5% | 21.9% | 25.9% |
| E0_RISK20 | +0.000 | 79.345 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +0.543 | 78.802 | 69.8% | 20.8% | 28.9% |
| RESID25_RISK35 | +1.750 | 77.596 | 66.7% | 18.8% | 27.6% |
| PAIR_REGRET_RISK35 | -0.853 | 80.198 | 77.1% | 19.8% | 24.8% |
| E0_SCORE_COVERAGE35 | +0.395 | 78.950 | 61.5% | 19.8% | 30.3% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.193 | 0.299 | 0.107 | 0.525 | 0.535 |
| task_0001 | 0.221 | 0.406 | 0.007 | 0.504 | 0.498 |
| task_0002 | 0.252 | 0.501 | 0.107 | 0.490 | 0.494 |
| task_0003 | 0.220 | 0.334 | 0.014 | 0.489 | 0.500 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 69.865 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +0.448 | 69.417 | 75.0% | 21.9% | 28.5% |
| E0_MATCHED | +1.292 | 68.573 | 76.0% | 19.8% | 26.7% |
| VQ_MEAN | +2.462 | 67.404 | 71.9% | 21.9% | 31.7% |
| RESID_OOF25 | +1.179 | 68.687 | 72.9% | 21.9% | 29.8% |
| RESID_OOF50 | -2.375 | 72.240 | 67.7% | 20.8% | 30.6% |
| PAIR_RANK | -1.202 | 71.067 | 83.3% | 20.8% | 24.7% |
| PAIR_REGRET | -3.551 | 73.416 | 89.6% | 20.8% | 23.1% |
| E0_RISK20 | +0.000 | 69.865 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +1.292 | 68.573 | 76.0% | 19.8% | 26.7% |
| RESID25_RISK35 | +1.179 | 68.687 | 72.9% | 21.9% | 29.8% |
| PAIR_REGRET_RISK35 | -3.551 | 73.416 | 89.6% | 20.8% | 23.1% |
| E0_SCORE_COVERAGE35 | +1.292 | 68.573 | 76.0% | 19.8% | 26.7% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.170 | 0.257 | 0.071 | 0.482 | 0.486 |
| task_0001 | 0.179 | 0.310 | 0.036 | 0.491 | 0.483 |
| task_0002 | 0.219 | 0.383 | 0.062 | 0.458 | 0.454 |
| task_0003 | 0.215 | 0.347 | 0.052 | 0.513 | 0.488 |

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
