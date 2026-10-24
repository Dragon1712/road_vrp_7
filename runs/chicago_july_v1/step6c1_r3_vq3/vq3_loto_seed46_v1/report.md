# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 78.035 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +7.504 | 70.531 | 87.5% | 25.0% | 27.9% |
| E0_MATCHED | +8.927 | 69.108 | 81.2% | 22.9% | 27.6% |
| VQ_MEAN | -7.857 | 85.891 | 50.0% | 8.3% | 14.8% |
| RESID_OOF25 | +0.643 | 77.392 | 83.3% | 22.9% | 27.2% |
| RESID_OOF50 | +7.475 | 70.560 | 68.8% | 16.7% | 22.6% |
| PAIR_RANK | +5.545 | 72.490 | 79.2% | 18.7% | 24.4% |
| PAIR_REGRET | +6.242 | 71.793 | 79.2% | 20.8% | 27.4% |
| E0_RISK20 | +0.000 | 78.035 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +2.743 | 75.291 | 56.2% | 14.6% | 19.3% |
| RESID25_RISK35 | -5.705 | 83.740 | 62.5% | 18.8% | 22.2% |
| PAIR_REGRET_RISK35 | -0.730 | 78.765 | 56.2% | 18.8% | 25.2% |
| E0_SCORE_COVERAGE35 | +2.743 | 75.291 | 56.2% | 14.6% | 19.3% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.194 | 0.228 | 0.132 | 0.512 | 0.485 |
| task_0001 | 0.223 | 0.301 | 0.008 | 0.549 | 0.553 |
| task_0002 | 0.265 | 0.454 | 0.139 | 0.545 | 0.544 |
| task_0003 | 0.216 | 0.288 | 0.021 | 0.535 | 0.529 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 75.967 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +6.702 | 69.265 | 87.5% | 22.9% | 26.3% |
| E0_MATCHED | +5.543 | 70.424 | 81.2% | 18.8% | 22.8% |
| VQ_MEAN | -7.273 | 83.240 | 62.5% | 20.8% | 34.2% |
| RESID_OOF25 | +7.192 | 68.775 | 81.2% | 20.8% | 25.3% |
| RESID_OOF50 | +9.099 | 66.868 | 79.2% | 16.7% | 20.2% |
| PAIR_RANK | +5.756 | 70.211 | 75.0% | 16.7% | 21.2% |
| PAIR_REGRET | +1.245 | 74.722 | 83.3% | 20.8% | 24.3% |
| E0_RISK20 | +5.022 | 70.945 | 4.2% | 0.0% | 0.0% |
| E0_RISK35 | +5.543 | 70.424 | 81.2% | 18.8% | 22.8% |
| RESID25_RISK35 | +7.192 | 68.775 | 81.2% | 20.8% | 25.3% |
| PAIR_REGRET_RISK35 | +1.245 | 74.722 | 83.3% | 20.8% | 24.3% |
| E0_SCORE_COVERAGE35 | +5.543 | 70.424 | 81.2% | 18.8% | 22.8% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.174 | 0.214 | 0.094 | 0.463 | 0.431 |
| task_0001 | 0.165 | 0.433 | 0.089 | 0.502 | 0.487 |
| task_0002 | 0.245 | 0.548 | 0.139 | 0.542 | 0.550 |
| task_0003 | 0.204 | 0.288 | 0.025 | 0.503 | 0.524 |

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
