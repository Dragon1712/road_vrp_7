# R3-VQ3：决策排序 + 平均大额损失概率校准（仅开发集）

这里的收益是 **Road.replay 的C3短程重启代理收益**，不是完整30秒在线ALNS。
外层任务留一；内层训练任务留一风险概率校准；阈值固定，未接触原Validation/Internal Test。

## H=20

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 78.035 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +11.764 | 66.271 | 87.5% | 22.9% | 25.1% |
| E0_MATCHED | +7.087 | 70.948 | 79.2% | 25.0% | 30.8% |
| VQ_MEAN | -9.625 | 87.660 | 54.2% | 14.6% | 25.7% |
| RESID_OOF25 | +16.762 | 61.273 | 83.3% | 18.8% | 22.1% |
| RESID_OOF50 | +12.322 | 65.712 | 70.8% | 14.6% | 19.4% |
| PAIR_RANK | +5.545 | 72.490 | 79.2% | 18.7% | 24.4% |
| PAIR_REGRET | +6.242 | 71.793 | 79.2% | 20.8% | 27.4% |
| E0_RISK20 | +0.000 | 78.035 | 0.0% | 0.0% | 0.0% |
| E0_RISK35 | +0.207 | 77.828 | 54.2% | 16.7% | 22.5% |
| RESID25_RISK35 | +1.035 | 77.000 | 62.5% | 14.6% | 17.1% |
| PAIR_REGRET_RISK35 | -0.730 | 78.765 | 56.2% | 18.8% | 25.2% |
| E0_SCORE_COVERAGE35 | +0.207 | 77.828 | 54.2% | 16.7% | 22.5% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.194 | 0.226 | 0.130 | 0.512 | 0.485 |
| task_0001 | 0.223 | 0.282 | 0.008 | 0.549 | 0.553 |
| task_0002 | 0.265 | 0.477 | 0.140 | 0.545 | 0.544 |
| task_0003 | 0.215 | 0.327 | 0.020 | 0.535 | 0.529 |

### 覆盖率匹配负对照

E0_RISK35 与 E0_SCORE_COVERAGE35 应比较平均收益和大额损失/覆盖，不要只比较损失/状态。

## H=50

| 方法 | 相对C3收益 (秒/状态) ↑ | Regret ↓ | 覆盖率 | 大额损失/状态 | 大额损失/覆盖 |
|---|---:|---:|---:|---:|---:|
| C3 | +0.000 | 75.967 | 0.0% | 0.0% | 0.0% |
| E0_FULL | +10.726 | 65.241 | 87.5% | 20.8% | 23.3% |
| E0_MATCHED | +5.525 | 70.442 | 79.2% | 20.8% | 26.7% |
| VQ_MEAN | -3.533 | 79.500 | 62.5% | 22.9% | 36.9% |
| RESID_OOF25 | +16.173 | 59.794 | 83.3% | 16.7% | 19.7% |
| RESID_OOF50 | +13.242 | 62.725 | 75.0% | 16.7% | 22.2% |
| PAIR_RANK | +5.756 | 70.211 | 75.0% | 16.7% | 21.2% |
| PAIR_REGRET | +1.245 | 74.722 | 83.3% | 20.8% | 24.3% |
| E0_RISK20 | +3.946 | 72.021 | 2.1% | 0.0% | 0.0% |
| E0_RISK35 | +5.525 | 70.442 | 79.2% | 20.8% | 26.7% |
| RESID25_RISK35 | +16.173 | 59.794 | 83.3% | 16.7% | 19.7% |
| PAIR_REGRET_RISK35 | +1.245 | 74.722 | 83.3% | 20.8% | 24.3% |
| E0_SCORE_COVERAGE35 | +5.525 | 70.442 | 79.2% | 20.8% | 26.7% |

### 按留出任务的风险分类与排序诊断

| 留出任务 | 风险Brier ↓ | 风险AUPRC ↑ | 风险ECE5 ↓ | PAIR_RANK准确率 ↑ | PAIR_REGRET准确率 ↑ |
|---|---:|---:|---:|---:|---:|
| task_0000 | 0.172 | 0.228 | 0.090 | 0.463 | 0.431 |
| task_0001 | 0.165 | 0.393 | 0.087 | 0.502 | 0.487 |
| task_0002 | 0.246 | 0.505 | 0.137 | 0.542 | 0.550 |
| task_0003 | 0.203 | 0.305 | 0.026 | 0.503 | 0.524 |

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
