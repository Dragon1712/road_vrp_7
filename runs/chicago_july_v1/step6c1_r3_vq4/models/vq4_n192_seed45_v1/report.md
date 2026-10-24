# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +5.399 | 73.894 | 77.6% | 20.8% | 27.0% |
| E0_MATCHED | +6.082 | 73.211 | 73.4% | 19.3% | 25.9% |
| VQ_MEAN | +3.498 | 75.795 | 50.5% | 13.0% | 27.3% |
| RESID_OOF25 | +6.632 | 72.662 | 71.9% | 18.2% | 25.5% |
| RESID_OOF50 | +9.584 | 69.709 | 67.7% | 17.2% | 25.2% |
| PAIR_RANK | +1.769 | 77.524 | 79.2% | 21.9% | 27.9% |
| PAIR_REGRET | +2.721 | 76.572 | 77.6% | 20.8% | 27.1% |
| E0_RISK20 | +0.000 | 79.293 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +1.121 | 78.172 | 44.8% | 13.5% | 21.7% |
| RESID25_RISK35 | +1.365 | 77.928 | 42.7% | 12.5% | 20.4% |
| PAIR_REGRET_RISK35 | -4.022 | 83.315 | 45.3% | 15.1% | 23.0% |
| E0_SCORE_COVERAGE35 | +1.341 | 77.952 | 39.1% | 12.0% | 21.7% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.210 | 0.302 | 0.080 | 0.550 | 0.548 |
| task_0001 | 0.209 | 0.271 | 0.092 | 0.531 | 0.540 |
| task_0002 | 0.269 | 0.388 | 0.131 | 0.550 | 0.546 |
| task_0003 | 0.234 | 0.430 | 0.037 | 0.486 | 0.482 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +0.335 | 72.138 | 77.6% | 19.8% | 25.8% |
| E0_MATCHED | +2.856 | 69.618 | 73.4% | 19.3% | 25.6% |
| VQ_MEAN | -2.308 | 74.782 | 65.6% | 19.8% | 28.8% |
| RESID_OOF25 | +2.672 | 69.802 | 75.0% | 18.2% | 24.8% |
| RESID_OOF50 | +1.545 | 70.928 | 72.4% | 19.8% | 27.4% |
| PAIR_RANK | -9.160 | 81.633 | 83.9% | 25.5% | 29.5% |
| PAIR_REGRET | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_RISK20 | +0.000 | 72.474 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +2.856 | 69.618 | 73.4% | 19.3% | 25.6% |
| RESID25_RISK35 | +2.672 | 69.802 | 75.0% | 18.2% | 24.8% |
| PAIR_REGRET_RISK35 | -5.911 | 78.385 | 85.9% | 25.5% | 29.1% |
| E0_SCORE_COVERAGE35 | +2.856 | 69.618 | 73.4% | 19.3% | 25.6% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.183 | 0.236 | 0.060 | 0.511 | 0.508 |
| task_0001 | 0.176 | 0.236 | 0.074 | 0.488 | 0.484 |
| task_0002 | 0.229 | 0.333 | 0.074 | 0.465 | 0.472 |
| task_0003 | 0.226 | 0.360 | 0.068 | 0.489 | 0.480 |

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
