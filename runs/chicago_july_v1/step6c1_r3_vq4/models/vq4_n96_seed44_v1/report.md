# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 79.345 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +8.104 | 71.241 | 72.9% | 19.8% | 26.7% |
| E0_MATCHED | -1.758 | 81.103 | 71.9% | 22.9% | 30.9% |
| VQ_MEAN | +8.832 | 70.513 | 63.5% | 12.5% | 17.9% |
| RESID_OOF25 | +5.774 | 73.571 | 69.8% | 17.7% | 24.1% |
| RESID_OOF50 | +6.947 | 72.398 | 62.5% | 14.6% | 21.3% |
| PAIR_RANK | +0.730 | 78.615 | 86.5% | 24.0% | 28.2% |
| PAIR_REGRET | +0.349 | 78.996 | 86.5% | 21.9% | 25.9% |
| E0_RISK20 | +0.000 | 79.345 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | -3.285 | 82.630 | 66.7% | 21.9% | 30.9% |
| RESID25_RISK35 | +0.197 | 79.148 | 63.5% | 16.7% | 25.0% |
| PAIR_REGRET_RISK35 | -0.938 | 80.283 | 78.1% | 20.8% | 26.4% |
| E0_SCORE_COVERAGE35 | +0.428 | 78.917 | 55.2% | 18.7% | 27.9% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.194 | 0.286 | 0.108 | 0.525 | 0.535 |
| task_0001 | 0.220 | 0.426 | 0.007 | 0.504 | 0.498 |
| task_0002 | 0.252 | 0.506 | 0.107 | 0.490 | 0.494 |
| task_0003 | 0.220 | 0.328 | 0.012 | 0.489 | 0.500 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 69.865 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +2.523 | 67.342 | 72.9% | 19.8% | 27.1% |
| E0_MATCHED | +2.112 | 67.753 | 71.9% | 19.8% | 27.1% |
| VQ_MEAN | +0.816 | 69.049 | 68.8% | 20.8% | 30.5% |
| RESID_OOF25 | +5.014 | 64.851 | 70.8% | 19.8% | 27.5% |
| RESID_OOF50 | +6.667 | 63.198 | 64.6% | 17.7% | 26.8% |
| PAIR_RANK | -1.202 | 71.067 | 83.3% | 20.8% | 24.7% |
| PAIR_REGRET | -3.551 | 73.416 | 89.6% | 20.8% | 23.1% |
| E0_RISK20 | +0.000 | 69.865 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +2.112 | 67.753 | 71.9% | 19.8% | 27.1% |
| RESID25_RISK35 | +5.014 | 64.851 | 70.8% | 19.8% | 27.5% |
| PAIR_REGRET_RISK35 | -3.551 | 73.416 | 89.6% | 20.8% | 23.1% |
| E0_SCORE_COVERAGE35 | +2.112 | 67.753 | 71.9% | 19.8% | 27.1% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.170 | 0.254 | 0.071 | 0.482 | 0.486 |
| task_0001 | 0.179 | 0.284 | 0.034 | 0.491 | 0.483 |
| task_0002 | 0.219 | 0.365 | 0.061 | 0.458 | 0.454 |
| task_0003 | 0.216 | 0.313 | 0.052 | 0.513 | 0.488 |

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
