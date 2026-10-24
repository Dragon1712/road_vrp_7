# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +0.453 | 78.840 | 73.4% | 22.4% | 30.8% |
| E0_MATCHED | +5.222 | 74.072 | 72.9% | 20.8% | 28.5% |
| VQ_MEAN | -1.924 | 81.218 | 45.8% | 15.1% | 34.4% |
| RESID_OOF25 | +0.240 | 79.053 | 69.8% | 22.4% | 32.3% |
| RESID_OOF50 | -0.048 | 79.342 | 62.5% | 18.2% | 30.0% |
| PAIR_RANK | +1.769 | 77.524 | 79.2% | 21.9% | 27.9% |
| PAIR_REGRET | +2.721 | 76.572 | 77.6% | 20.8% | 27.1% |
| E0_RISK20 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +1.069 | 78.224 | 42.7% | 13.5% | 23.0% |
| RESID25_RISK35 | +0.991 | 78.303 | 41.1% | 13.5% | 22.8% |
| PAIR_REGRET_RISK35 | -4.334 | 83.627 | 44.8% | 14.6% | 22.5% |
| E0_SCORE_COVERAGE35 | +1.994 | 77.299 | 33.9% | 10.9% | 22.9% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.211 | 0.297 | 0.079 | 0.550 | 0.548 |
| task_0001 | 0.208 | 0.302 | 0.089 | 0.531 | 0.540 |
| task_0002 | 0.268 | 0.390 | 0.129 | 0.550 | 0.546 |
| task_0003 | 0.234 | 0.435 | 0.037 | 0.486 | 0.482 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_FULL | -0.497 | 72.971 | 73.4% | 21.9% | 30.2% |
| E0_MATCHED | +0.795 | 71.679 | 72.9% | 21.4% | 29.5% |
| VQ_MEAN | -3.520 | 75.994 | 61.5% | 18.8% | 28.8% |
| RESID_OOF25 | +2.266 | 70.208 | 72.4% | 20.8% | 29.0% |
| RESID_OOF50 | +1.332 | 71.141 | 69.8% | 19.8% | 29.0% |
| PAIR_RANK | -9.160 | 81.633 | 83.9% | 25.5% | 29.5% |
| PAIR_REGRET | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_RISK20 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +0.795 | 71.679 | 72.9% | 21.4% | 29.5% |
| RESID25_RISK35 | +2.266 | 70.208 | 72.4% | 20.8% | 29.0% |
| PAIR_REGRET_RISK35 | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_SCORE_COVERAGE35 | +0.795 | 71.679 | 72.9% | 21.4% | 29.5% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.183 | 0.251 | 0.060 | 0.511 | 0.508 |
| task_0001 | 0.176 | 0.254 | 0.074 | 0.488 | 0.484 |
| task_0002 | 0.229 | 0.323 | 0.073 | 0.465 | 0.472 |
| task_0003 | 0.226 | 0.372 | 0.068 | 0.489 | 0.480 |

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
