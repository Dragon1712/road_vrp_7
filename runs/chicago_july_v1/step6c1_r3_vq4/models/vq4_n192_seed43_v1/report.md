# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +5.368 | 73.926 | 77.1% | 21.9% | 28.4% |
| E0_MATCHED | +6.263 | 73.030 | 74.5% | 21.4% | 29.2% |
| VQ_MEAN | +1.357 | 77.936 | 46.4% | 14.1% | 32.9% |
| RESID_OOF25 | +5.432 | 73.862 | 68.8% | 19.8% | 28.3% |
| RESID_OOF50 | +7.779 | 71.515 | 63.5% | 16.7% | 25.5% |
| PAIR_RANK | +1.769 | 77.524 | 79.2% | 21.9% | 27.9% |
| PAIR_REGRET | +2.721 | 76.572 | 77.6% | 20.8% | 27.1% |
| E0_RISK20 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +0.113 | 79.181 | 42.2% | 14.1% | 23.1% |
| RESID25_RISK35 | +0.989 | 78.304 | 42.2% | 13.5% | 22.0% |
| PAIR_REGRET_RISK35 | -3.854 | 83.147 | 44.8% | 14.6% | 22.5% |
| E0_SCORE_COVERAGE35 | +1.586 | 77.707 | 32.8% | 11.5% | 24.6% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.210 | 0.296 | 0.079 | 0.550 | 0.548 |
| task_0001 | 0.208 | 0.258 | 0.091 | 0.531 | 0.540 |
| task_0002 | 0.268 | 0.406 | 0.131 | 0.550 | 0.546 |
| task_0003 | 0.234 | 0.491 | 0.036 | 0.486 | 0.482 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_FULL | -2.045 | 74.518 | 77.1% | 23.4% | 30.5% |
| E0_MATCHED | +2.125 | 70.349 | 74.5% | 20.8% | 28.6% |
| VQ_MEAN | +1.556 | 70.918 | 65.1% | 18.2% | 26.7% |
| RESID_OOF25 | +0.940 | 71.534 | 73.4% | 22.4% | 30.3% |
| RESID_OOF50 | -0.310 | 72.784 | 68.2% | 20.8% | 29.5% |
| PAIR_RANK | -9.160 | 81.633 | 83.9% | 25.5% | 29.5% |
| PAIR_REGRET | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_RISK20 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +2.125 | 70.349 | 74.5% | 20.8% | 28.6% |
| RESID25_RISK35 | +0.940 | 71.534 | 73.4% | 22.4% | 30.3% |
| PAIR_REGRET_RISK35 | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_SCORE_COVERAGE35 | +2.125 | 70.349 | 74.5% | 20.8% | 28.6% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.183 | 0.246 | 0.060 | 0.511 | 0.508 |
| task_0001 | 0.177 | 0.229 | 0.073 | 0.488 | 0.484 |
| task_0002 | 0.229 | 0.326 | 0.072 | 0.465 | 0.472 |
| task_0003 | 0.227 | 0.288 | 0.069 | 0.489 | 0.480 |

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
