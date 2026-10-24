# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +7.156 | 72.137 | 75.0% | 20.3% | 27.0% |
| E0_MATCHED | +4.901 | 74.392 | 72.9% | 20.8% | 28.6% |
| VQ_MEAN | +4.544 | 74.749 | 46.9% | 9.9% | 22.1% |
| RESID_OOF25 | +4.789 | 74.505 | 68.2% | 18.8% | 27.7% |
| RESID_OOF50 | +5.844 | 73.449 | 64.6% | 18.2% | 28.4% |
| PAIR_RANK | +1.769 | 77.524 | 79.2% | 21.9% | 27.9% |
| PAIR_REGRET | +2.721 | 76.572 | 77.6% | 20.8% | 27.1% |
| E0_RISK20 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +0.067 | 79.226 | 44.3% | 14.6% | 23.7% |
| RESID25_RISK35 | +1.281 | 78.012 | 42.2% | 13.0% | 21.5% |
| PAIR_REGRET_RISK35 | -4.721 | 84.014 | 44.3% | 15.1% | 23.8% |
| E0_SCORE_COVERAGE35 | +0.527 | 78.766 | 39.6% | 13.0% | 23.1% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.210 | 0.303 | 0.079 | 0.550 | 0.548 |
| task_0001 | 0.208 | 0.305 | 0.089 | 0.531 | 0.540 |
| task_0002 | 0.268 | 0.388 | 0.132 | 0.550 | 0.546 |
| task_0003 | 0.234 | 0.443 | 0.036 | 0.486 | 0.482 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_FULL | -0.333 | 72.807 | 75.0% | 20.8% | 27.6% |
| E0_MATCHED | +0.804 | 71.670 | 72.9% | 19.8% | 26.8% |
| VQ_MEAN | +2.776 | 69.697 | 62.5% | 18.8% | 29.6% |
| RESID_OOF25 | +2.102 | 70.372 | 72.9% | 19.8% | 27.3% |
| RESID_OOF50 | +4.188 | 68.286 | 67.7% | 17.2% | 25.1% |
| PAIR_RANK | -9.160 | 81.633 | 83.9% | 25.5% | 29.5% |
| PAIR_REGRET | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_RISK20 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +0.804 | 71.670 | 72.9% | 19.8% | 26.8% |
| RESID25_RISK35 | +2.102 | 70.372 | 72.9% | 19.8% | 27.3% |
| PAIR_REGRET_RISK35 | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_SCORE_COVERAGE35 | +0.804 | 71.670 | 72.9% | 19.8% | 26.8% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.182 | 0.317 | 0.060 | 0.511 | 0.508 |
| task_0001 | 0.177 | 0.229 | 0.074 | 0.488 | 0.484 |
| task_0002 | 0.229 | 0.340 | 0.074 | 0.465 | 0.472 |
| task_0003 | 0.226 | 0.374 | 0.068 | 0.489 | 0.480 |

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
