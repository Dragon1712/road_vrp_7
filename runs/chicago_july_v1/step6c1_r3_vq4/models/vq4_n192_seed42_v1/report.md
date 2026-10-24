# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +5.532 | 73.761 | 74.5% | 20.8% | 27.9% |
| E0_MATCHED | +4.207 | 75.086 | 72.4% | 20.8% | 29.0% |
| VQ_MEAN | +0.791 | 78.502 | 44.3% | 11.5% | 27.0% |
| RESID_OOF25 | +8.006 | 71.287 | 69.8% | 18.2% | 26.0% |
| RESID_OOF50 | +8.553 | 70.741 | 64.6% | 15.6% | 23.5% |
| PAIR_RANK | +1.769 | 77.524 | 79.2% | 21.9% | 27.9% |
| PAIR_REGRET | +2.721 | 76.572 | 77.6% | 20.8% | 27.1% |
| E0_RISK20 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +0.108 | 79.185 | 41.7% | 13.5% | 22.7% |
| RESID25_RISK35 | -0.707 | 80.001 | 40.6% | 13.5% | 23.3% |
| PAIR_REGRET_RISK35 | -4.399 | 83.692 | 42.7% | 14.6% | 23.9% |
| E0_SCORE_COVERAGE35 | -0.029 | 79.322 | 37.5% | 12.0% | 20.4% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.210 | 0.315 | 0.079 | 0.550 | 0.548 |
| task_0001 | 0.208 | 0.255 | 0.091 | 0.531 | 0.540 |
| task_0002 | 0.268 | 0.400 | 0.131 | 0.550 | 0.546 |
| task_0003 | 0.234 | 0.458 | 0.036 | 0.486 | 0.482 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_FULL | -0.613 | 73.086 | 74.5% | 22.9% | 30.8% |
| E0_MATCHED | -1.077 | 73.551 | 72.4% | 21.4% | 30.1% |
| VQ_MEAN | +0.938 | 71.536 | 60.4% | 17.7% | 27.0% |
| RESID_OOF25 | -1.316 | 73.789 | 73.4% | 21.9% | 29.9% |
| RESID_OOF50 | +0.254 | 72.220 | 71.4% | 19.8% | 27.9% |
| PAIR_RANK | -9.160 | 81.633 | 83.9% | 25.5% | 29.5% |
| PAIR_REGRET | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_RISK20 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | -1.077 | 73.551 | 72.4% | 21.4% | 30.1% |
| RESID25_RISK35 | -1.316 | 73.789 | 73.4% | 21.9% | 29.9% |
| PAIR_REGRET_RISK35 | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_SCORE_COVERAGE35 | -1.077 | 73.551 | 72.4% | 21.4% | 30.1% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.183 | 0.237 | 0.060 | 0.511 | 0.508 |
| task_0001 | 0.177 | 0.240 | 0.076 | 0.488 | 0.484 |
| task_0002 | 0.229 | 0.324 | 0.072 | 0.465 | 0.472 |
| task_0003 | 0.226 | 0.371 | 0.068 | 0.489 | 0.480 |

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
