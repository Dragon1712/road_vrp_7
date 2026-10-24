# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 79.345 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +2.934 | 76.411 | 76.0% | 22.9% | 29.4% |
| E0_MATCHED | +0.798 | 78.547 | 69.8% | 21.9% | 31.1% |
| VQ_MEAN | +7.211 | 72.135 | 66.7% | 14.6% | 19.9% |
| RESID_OOF25 | +4.629 | 74.716 | 70.8% | 20.8% | 28.1% |
| RESID_OOF50 | +4.867 | 74.478 | 64.6% | 17.7% | 24.8% |
| PAIR_RANK | +0.730 | 78.615 | 86.5% | 24.0% | 28.2% |
| PAIR_REGRET | +0.349 | 78.996 | 86.5% | 21.9% | 25.9% |
| E0_RISK20 | +0.000 | 79.345 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +1.464 | 77.881 | 66.7% | 19.8% | 28.5% |
| RESID25_RISK35 | +0.640 | 78.705 | 65.6% | 18.8% | 27.3% |
| PAIR_REGRET_RISK35 | -1.067 | 80.413 | 78.1% | 20.8% | 26.4% |
| E0_SCORE_COVERAGE35 | -1.091 | 80.436 | 59.4% | 19.8% | 31.7% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.194 | 0.293 | 0.108 | 0.525 | 0.535 |
| task_0001 | 0.220 | 0.421 | 0.007 | 0.504 | 0.498 |
| task_0002 | 0.252 | 0.503 | 0.107 | 0.490 | 0.494 |
| task_0003 | 0.220 | 0.329 | 0.012 | 0.489 | 0.500 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 69.865 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +2.255 | 67.611 | 76.0% | 22.9% | 29.9% |
| E0_MATCHED | +3.313 | 66.552 | 69.8% | 15.6% | 22.9% |
| VQ_MEAN | -0.340 | 70.205 | 68.8% | 20.8% | 31.2% |
| RESID_OOF25 | +1.523 | 68.342 | 74.0% | 22.9% | 31.2% |
| RESID_OOF50 | +1.301 | 68.565 | 74.0% | 21.9% | 30.1% |
| PAIR_RANK | -1.202 | 71.067 | 83.3% | 20.8% | 24.7% |
| PAIR_REGRET | -3.551 | 73.416 | 89.6% | 20.8% | 23.1% |
| E0_RISK20 | +0.000 | 69.865 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +3.313 | 66.552 | 69.8% | 15.6% | 22.9% |
| RESID25_RISK35 | +1.523 | 68.342 | 74.0% | 22.9% | 31.2% |
| PAIR_REGRET_RISK35 | -3.551 | 73.416 | 89.6% | 20.8% | 23.1% |
| E0_SCORE_COVERAGE35 | +3.313 | 66.552 | 69.8% | 15.6% | 22.9% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.170 | 0.260 | 0.071 | 0.482 | 0.486 |
| task_0001 | 0.178 | 0.300 | 0.037 | 0.491 | 0.483 |
| task_0002 | 0.219 | 0.355 | 0.062 | 0.458 | 0.454 |
| task_0003 | 0.215 | 0.341 | 0.050 | 0.513 | 0.488 |

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
