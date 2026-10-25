# R4-N Train-only Section动作排序 vs 状态委托门控解耦诊断

数据：1572动作，200场景快照，100个粗粒度搜索起点；Train-only，无新的ALNS调用。

## 主比较：全特征、非Shuffle；外层仅评分

| 外层留组 | 策略 | 委托 | 正/负 | K500均值(s) | 正例状态Top-1命中/机会 |
|---|---|---:|---:|---:|---:|
| hold_task_task_0000 | C3 | 0 | 0/0 | +0.000 | 8/22 |
| hold_task_task_0000 | ridge_ref | 18 | 5/2 | +0.281 | 8/22 |
| hold_task_task_0000 | pairwise_with_ridge_gate | 18 | 4/2 | +1.831 | 8/22 |
| hold_task_task_0000 | oracle_gate_pairwise_rank (*) | 22 | 8/4 | +5.577 | 8/22 |
| hold_task_task_0000 | sampled_oracle_early (*) | 22 | 22/0 | +20.937 | 8/22 |
| hold_task_task_0001 | C3 | 0 | 0/0 | +0.000 | 0/1 |
| hold_task_task_0001 | ridge_ref | 36 | 0/7 | -5.525 | 0/1 |
| hold_task_task_0001 | pairwise_with_ridge_gate | 36 | 0/7 | -5.525 | 0/1 |
| hold_task_task_0001 | oracle_gate_pairwise_rank (*) | 1 | 0/1 | -0.490 | 0/1 |
| hold_task_task_0001 | sampled_oracle_early (*) | 1 | 1/0 | +0.099 | 0/1 |
| hold_task_task_0002 | C3 | 0 | 0/0 | +0.000 | 10/20 |
| hold_task_task_0002 | ridge_ref | 0 | 0/0 | +0.000 | 10/20 |
| hold_task_task_0002 | pairwise_with_ridge_gate | 0 | 0/0 | +0.000 | 10/20 |
| hold_task_task_0002 | oracle_gate_pairwise_rank (*) | 20 | 10/8 | +3.018 | 10/20 |
| hold_task_task_0002 | sampled_oracle_early (*) | 20 | 20/0 | +36.563 | 10/20 |
| hold_task_task_0003 | C3 | 0 | 0/0 | +0.000 | 6/18 |
| hold_task_task_0003 | ridge_ref | 27 | 4/5 | +1.287 | 6/18 |
| hold_task_task_0003 | pairwise_with_ridge_gate | 27 | 3/7 | -1.566 | 6/18 |
| hold_task_task_0003 | oracle_gate_pairwise_rank (*) | 18 | 6/9 | -2.118 | 6/18 |
| hold_task_task_0003 | sampled_oracle_early (*) | 18 | 18/0 | +15.195 | 6/18 |
| old_to_new | C3 | 0 | 0/0 | +0.000 | 26/54 |
| old_to_new | ridge_ref | 0 | 0/0 | +0.000 | 26/54 |
| old_to_new | pairwise_with_ridge_gate | 0 | 0/0 | +0.000 | 26/54 |
| old_to_new | oracle_gate_pairwise_rank (*) | 54 | 26/18 | +0.201 | 26/54 |
| old_to_new | sampled_oracle_early (*) | 54 | 54/0 | +25.674 | 26/54 |
| new_to_old | C3 | 0 | 0/0 | +0.000 | 2/7 |
| new_to_old | ridge_ref | 16 | 1/5 | -2.489 | 2/7 |
| new_to_old | pairwise_with_ridge_gate | 16 | 2/6 | -1.285 | 2/7 |
| new_to_old | oracle_gate_pairwise_rank (*) | 7 | 2/4 | -0.419 | 2/7 |
| new_to_old | sampled_oracle_early (*) | 7 | 7/0 | +5.921 | 2/7 |

(*) **Oracle gate / oracle policy使用留出数据的未来收益标签，绝不可上线；仅用于分解错误来源。**

## 分析约束

- Ridge门控与Pairwise排序的组合也仅是Train内部固定K离线评分，尚未包含在线推理成本。
- Pairwise强制委托是有风险的诊断策略，不能当作推荐部署方案。
- 所有外层划分依原R4-L任务留一及种子组隔离；同一状态候选不跨组。
- Shuffle仅在外层训练任务内打乱标签；特征不含未来Road收益。
- 样本仍来自4个Train任务；不能将此结果当作独立Validation/Test或30秒在线增益。
