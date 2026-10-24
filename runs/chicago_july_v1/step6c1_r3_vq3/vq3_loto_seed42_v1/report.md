# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 78.035 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +5.685 | 72.350 | 87.5% | 27.1% | 30.4% |
| E0_MATCHED | +6.339 | 71.696 | 83.3% | 25.0% | 30.0% |
| VQ_MEAN | -4.709 | 82.744 | 47.9% | 10.4% | 17.9% |
| RESID_OOF25 | +6.962 | 71.073 | 85.4% | 25.0% | 28.3% |
| RESID_OOF50 | +4.678 | 73.357 | 72.9% | 18.8% | 23.6% |
| PAIR_RANK | +5.545 | 72.490 | 79.2% | 18.7% | 24.4% |
| PAIR_REGRET | +6.242 | 71.793 | 79.2% | 20.8% | 27.4% |
| E0_RISK20 | +0.000 | 78.035 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | -0.865 | 78.900 | 58.3% | 18.8% | 23.8% |
| RESID25_RISK35 | -0.268 | 78.303 | 66.7% | 20.8% | 23.3% |
| PAIR_REGRET_RISK35 | -0.730 | 78.765 | 56.2% | 18.8% | 25.2% |
| E0_SCORE_COVERAGE35 | +0.009 | 78.026 | 60.4% | 18.8% | 23.8% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.194 | 0.244 | 0.134 | 0.512 | 0.485 |
| task_0001 | 0.223 | 0.280 | 0.008 | 0.549 | 0.553 |
| task_0002 | 0.265 | 0.482 | 0.139 | 0.545 | 0.544 |
| task_0003 | 0.216 | 0.328 | 0.021 | 0.535 | 0.529 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 75.967 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +3.124 | 72.843 | 87.5% | 25.0% | 28.8% |
| E0_MATCHED | +6.920 | 69.047 | 83.3% | 18.8% | 23.3% |
| VQ_MEAN | -8.252 | 84.219 | 64.6% | 25.0% | 40.7% |
| RESID_OOF25 | +6.669 | 69.298 | 85.4% | 20.8% | 24.4% |
| RESID_OOF50 | +1.875 | 74.093 | 83.3% | 22.9% | 27.5% |
| PAIR_RANK | +5.756 | 70.211 | 75.0% | 16.7% | 21.2% |
| PAIR_REGRET | +1.245 | 74.722 | 83.3% | 20.8% | 24.3% |
| E0_RISK20 | +4.919 | 71.048 | 6.2% | 0.0% | 0.0% |
| E0_RISK35 | +6.920 | 69.047 | 83.3% | 18.8% | 23.3% |
| RESID25_RISK35 | +2.457 | 73.510 | 83.3% | 20.8% | 25.0% |
| PAIR_REGRET_RISK35 | +1.245 | 74.722 | 83.3% | 20.8% | 24.3% |
| E0_SCORE_COVERAGE35 | +6.920 | 69.047 | 83.3% | 18.8% | 23.3% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.173 | 0.225 | 0.092 | 0.463 | 0.431 |
| task_0001 | 0.165 | 0.368 | 0.091 | 0.502 | 0.487 |
| task_0002 | 0.245 | 0.525 | 0.141 | 0.542 | 0.550 |
| task_0003 | 0.203 | 0.280 | 0.026 | 0.503 | 0.524 |

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
