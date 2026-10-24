# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 79.345 | 0.0% | 0.0% | 0.0% |
| E0_FULL | -6.283 | 85.628 | 76.0% | 25.0% | 32.4% |
| E0_MATCHED | -4.292 | 83.637 | 80.2% | 26.0% | 32.5% |
| VQ_MEAN | +3.104 | 76.241 | 59.4% | 14.6% | 21.9% |
| RESID_OOF25 | -5.432 | 84.778 | 72.9% | 25.0% | 34.0% |
| RESID_OOF50 | -3.010 | 82.355 | 64.6% | 19.8% | 30.2% |
| PAIR_RANK | +0.730 | 78.615 | 86.5% | 24.0% | 28.2% |
| PAIR_REGRET | +0.349 | 78.996 | 86.5% | 21.9% | 25.9% |
| E0_RISK20 | +0.000 | 79.345 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +1.752 | 77.593 | 71.9% | 20.8% | 28.2% |
| RESID25_RISK35 | -0.500 | 79.845 | 66.7% | 20.8% | 29.9% |
| PAIR_REGRET_RISK35 | -0.400 | 79.745 | 78.1% | 19.8% | 24.7% |
| E0_SCORE_COVERAGE35 | +1.198 | 78.147 | 57.3% | 18.8% | 26.8% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.194 | 0.289 | 0.108 | 0.525 | 0.535 |
| task_0001 | 0.220 | 0.467 | 0.007 | 0.504 | 0.498 |
| task_0002 | 0.252 | 0.490 | 0.107 | 0.490 | 0.494 |
| task_0003 | 0.219 | 0.342 | 0.011 | 0.489 | 0.500 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 69.865 | 0.0% | 0.0% | 0.0% |
| E0_FULL | -0.762 | 70.628 | 76.0% | 24.0% | 31.2% |
| E0_MATCHED | -0.827 | 70.692 | 80.2% | 22.9% | 28.6% |
| VQ_MEAN | +5.703 | 64.162 | 71.9% | 21.9% | 31.8% |
| RESID_OOF25 | +1.490 | 68.375 | 72.9% | 22.9% | 31.4% |
| RESID_OOF50 | -1.036 | 70.901 | 65.6% | 21.9% | 33.3% |
| PAIR_RANK | -1.202 | 71.067 | 83.3% | 20.8% | 24.7% |
| PAIR_REGRET | -3.551 | 73.416 | 89.6% | 20.8% | 23.1% |
| E0_RISK20 | +0.000 | 69.865 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | -0.827 | 70.692 | 80.2% | 22.9% | 28.6% |
| RESID25_RISK35 | +1.490 | 68.375 | 72.9% | 22.9% | 31.4% |
| PAIR_REGRET_RISK35 | -3.551 | 73.416 | 89.6% | 20.8% | 23.1% |
| E0_SCORE_COVERAGE35 | -0.827 | 70.692 | 80.2% | 22.9% | 28.6% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.171 | 0.237 | 0.071 | 0.482 | 0.486 |
| task_0001 | 0.179 | 0.282 | 0.034 | 0.491 | 0.483 |
| task_0002 | 0.219 | 0.365 | 0.061 | 0.458 | 0.454 |
| task_0003 | 0.215 | 0.369 | 0.055 | 0.513 | 0.488 |

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
