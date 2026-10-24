# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +9.555 | 69.738 | 71.9% | 19.3% | 26.8% |
| E0_MATCHED | +4.554 | 74.739 | 75.0% | 22.4% | 29.5% |
| VQ_MEAN | +0.065 | 79.228 | 50.0% | 15.6% | 32.1% |
| RESID_OOF25 | +10.337 | 68.957 | 68.8% | 17.2% | 25.0% |
| RESID_OOF50 | +8.985 | 70.308 | 64.1% | 16.7% | 26.3% |
| PAIR_RANK | +1.769 | 77.524 | 79.2% | 21.9% | 27.9% |
| PAIR_REGRET | +2.721 | 76.572 | 77.6% | 20.8% | 27.1% |
| E0_RISK20 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | -0.342 | 79.636 | 44.8% | 15.1% | 22.4% |
| RESID25_RISK35 | +3.241 | 76.052 | 39.6% | 10.9% | 18.7% |
| PAIR_REGRET_RISK35 | -4.291 | 83.585 | 42.7% | 14.6% | 23.9% |
| E0_SCORE_COVERAGE35 | +2.086 | 77.207 | 39.1% | 13.5% | 24.0% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.210 | 0.313 | 0.080 | 0.550 | 0.548 |
| task_0001 | 0.208 | 0.295 | 0.089 | 0.531 | 0.540 |
| task_0002 | 0.268 | 0.409 | 0.131 | 0.550 | 0.546 |
| task_0003 | 0.234 | 0.467 | 0.036 | 0.486 | 0.482 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +3.073 | 69.400 | 71.9% | 18.8% | 26.3% |
| E0_MATCHED | +2.887 | 69.586 | 75.0% | 20.3% | 26.3% |
| VQ_MEAN | -2.762 | 75.236 | 58.9% | 18.2% | 30.5% |
| RESID_OOF25 | +2.337 | 70.137 | 69.8% | 18.8% | 27.4% |
| RESID_OOF50 | +5.159 | 67.315 | 65.6% | 16.7% | 26.1% |
| PAIR_RANK | -9.160 | 81.633 | 83.9% | 25.5% | 29.5% |
| PAIR_REGRET | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_RISK20 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +2.887 | 69.586 | 75.0% | 20.3% | 26.3% |
| RESID25_RISK35 | +2.337 | 70.137 | 69.8% | 18.8% | 27.4% |
| PAIR_REGRET_RISK35 | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_SCORE_COVERAGE35 | +2.887 | 69.586 | 75.0% | 20.3% | 26.3% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.183 | 0.222 | 0.060 | 0.511 | 0.508 |
| task_0001 | 0.177 | 0.215 | 0.073 | 0.488 | 0.484 |
| task_0002 | 0.229 | 0.330 | 0.073 | 0.465 | 0.472 |
| task_0003 | 0.226 | 0.406 | 0.068 | 0.489 | 0.480 |

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
