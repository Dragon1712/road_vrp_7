# R4-O Train-only Section路线结构排序与独立委托门控

结果仅为冻结Train反事实下的固定K=500离线收益；非30秒在线或独立Validation/Test。

场景快照：200；候选：1572；粗粒度搜索起点：100。

| 留组 | 特征 | 策略 | 委托 | 正/负 | 平均K500收益(s) | 正收益Top1/机会 | 同状态随机Top1期望 |
|---|---|---|---:|---:|---:|---:|
| hold_task_task_0000 | baseline15 | C3 | 0 | 0/0 | +0.000 | 8/22 | 8.25 |
| hold_task_task_0000 | baseline15 | ridge_ref | 18 | 5/2 | +0.281 | 8/22 | 8.25 |
| hold_task_task_0000 | baseline15 | pairwise_ridge_gate | 18 | 4/2 | +1.831 | 8/22 | 8.25 |
| hold_task_task_0000 | baseline15 | learned_gate_pairwise | 18 | 5/3 | +3.460 | 8/22 | 8.25 |
| hold_task_task_0000 | baseline15 | oracle_gate_pairwise (*) | 22 | 8/4 | +5.577 | 8/22 | 8.25 |
| hold_task_task_0000 | baseline15 | sampled_oracle (*) | 22 | 22/0 | +20.937 | 8/22 | 8.25 |
| hold_task_task_0000 | route_context | pairwise_ridge_gate | 18 | 3/1 | +0.524 | 4/22 | 8.25 |
| hold_task_task_0000 | route_context | learned_gate_pairwise | 18 | 3/5 | -3.800 | 4/22 | 8.25 |
| hold_task_task_0000 | route_context | oracle_gate_pairwise (*) | 22 | 4/6 | -3.947 | 4/22 | 8.25 |
| hold_task_task_0000 | route_context | sampled_oracle (*) | 22 | 22/0 | +20.937 | 4/22 | 8.25 |
| hold_task_task_0000 | route_context_interact | pairwise_ridge_gate | 18 | 2/2 | -0.785 | 3/22 | 8.25 |
| hold_task_task_0000 | route_context_interact | learned_gate_pairwise | 18 | 2/6 | -4.845 | 3/22 | 8.25 |
| hold_task_task_0000 | route_context_interact | oracle_gate_pairwise (*) | 22 | 3/7 | -5.256 | 3/22 | 8.25 |
| hold_task_task_0000 | route_context_interact | sampled_oracle (*) | 22 | 22/0 | +20.937 | 3/22 | 8.25 |
| hold_task_task_0001 | baseline15 | C3 | 0 | 0/0 | +0.000 | 0/1 | 0.38 |
| hold_task_task_0001 | baseline15 | ridge_ref | 36 | 0/7 | -5.525 | 0/1 | 0.38 |
| hold_task_task_0001 | baseline15 | pairwise_ridge_gate | 36 | 0/7 | -5.525 | 0/1 | 0.38 |
| hold_task_task_0001 | baseline15 | learned_gate_pairwise | 20 | 0/7 | -5.525 | 0/1 | 0.38 |
| hold_task_task_0001 | baseline15 | oracle_gate_pairwise (*) | 1 | 0/1 | -0.490 | 0/1 | 0.38 |
| hold_task_task_0001 | baseline15 | sampled_oracle (*) | 1 | 1/0 | +0.099 | 0/1 | 0.38 |
| hold_task_task_0001 | route_context | pairwise_ridge_gate | 36 | 0/7 | -5.525 | 0/1 | 0.38 |
| hold_task_task_0001 | route_context | learned_gate_pairwise | 20 | 0/7 | -5.525 | 0/1 | 0.38 |
| hold_task_task_0001 | route_context | oracle_gate_pairwise (*) | 1 | 0/1 | -0.490 | 0/1 | 0.38 |
| hold_task_task_0001 | route_context | sampled_oracle (*) | 1 | 1/0 | +0.099 | 0/1 | 0.38 |
| hold_task_task_0001 | route_context_interact | pairwise_ridge_gate | 36 | 0/7 | -5.525 | 0/1 | 0.38 |
| hold_task_task_0001 | route_context_interact | learned_gate_pairwise | 20 | 0/7 | -5.525 | 0/1 | 0.38 |
| hold_task_task_0001 | route_context_interact | oracle_gate_pairwise (*) | 1 | 0/1 | -0.490 | 0/1 | 0.38 |
| hold_task_task_0001 | route_context_interact | sampled_oracle (*) | 1 | 1/0 | +0.099 | 0/1 | 0.38 |
| hold_task_task_0002 | baseline15 | C3 | 0 | 0/0 | +0.000 | 10/20 | 11.02 |
| hold_task_task_0002 | baseline15 | ridge_ref | 0 | 0/0 | +0.000 | 10/20 | 11.02 |
| hold_task_task_0002 | baseline15 | pairwise_ridge_gate | 0 | 0/0 | +0.000 | 10/20 | 11.02 |
| hold_task_task_0002 | baseline15 | learned_gate_pairwise | 12 | 5/4 | +3.664 | 10/20 | 11.02 |
| hold_task_task_0002 | baseline15 | oracle_gate_pairwise (*) | 20 | 10/8 | +3.018 | 10/20 | 11.02 |
| hold_task_task_0002 | baseline15 | sampled_oracle (*) | 20 | 20/0 | +36.563 | 10/20 | 11.02 |
| hold_task_task_0002 | route_context | pairwise_ridge_gate | 0 | 0/0 | +0.000 | 11/20 | 11.02 |
| hold_task_task_0002 | route_context | learned_gate_pairwise | 12 | 6/3 | +4.542 | 11/20 | 11.02 |
| hold_task_task_0002 | route_context | oracle_gate_pairwise (*) | 20 | 11/5 | +2.850 | 11/20 | 11.02 |
| hold_task_task_0002 | route_context | sampled_oracle (*) | 20 | 20/0 | +36.563 | 11/20 | 11.02 |
| hold_task_task_0002 | route_context_interact | pairwise_ridge_gate | 0 | 0/0 | +0.000 | 12/20 | 11.02 |
| hold_task_task_0002 | route_context_interact | learned_gate_pairwise | 12 | 6/3 | +4.542 | 12/20 | 11.02 |
| hold_task_task_0002 | route_context_interact | oracle_gate_pairwise (*) | 20 | 12/5 | +4.459 | 12/20 | 11.02 |
| hold_task_task_0002 | route_context_interact | sampled_oracle (*) | 20 | 20/0 | +36.563 | 12/20 | 11.02 |
| hold_task_task_0003 | baseline15 | C3 | 0 | 0/0 | +0.000 | 6/18 | 6.75 |
| hold_task_task_0003 | baseline15 | ridge_ref | 25 | 3/4 | +2.051 | 6/18 | 6.75 |
| hold_task_task_0003 | baseline15 | pairwise_ridge_gate | 25 | 3/6 | -0.223 | 6/18 | 6.75 |
| hold_task_task_0003 | baseline15 | learned_gate_pairwise | 14 | 3/5 | -1.690 | 6/18 | 6.75 |
| hold_task_task_0003 | baseline15 | oracle_gate_pairwise (*) | 18 | 6/9 | -2.118 | 6/18 | 6.75 |
| hold_task_task_0003 | baseline15 | sampled_oracle (*) | 18 | 18/0 | +15.195 | 6/18 | 6.75 |
| hold_task_task_0003 | route_context | pairwise_ridge_gate | 25 | 1/4 | -0.866 | 5/18 | 6.75 |
| hold_task_task_0003 | route_context | learned_gate_pairwise | 14 | 4/4 | -2.019 | 5/18 | 6.75 |
| hold_task_task_0003 | route_context | oracle_gate_pairwise (*) | 18 | 5/7 | -3.345 | 5/18 | 6.75 |
| hold_task_task_0003 | route_context | sampled_oracle (*) | 18 | 18/0 | +15.195 | 5/18 | 6.75 |
| hold_task_task_0003 | route_context_interact | pairwise_ridge_gate | 25 | 1/4 | -1.225 | 3/18 | 6.75 |
| hold_task_task_0003 | route_context_interact | learned_gate_pairwise | 14 | 2/4 | -3.691 | 3/18 | 6.75 |
| hold_task_task_0003 | route_context_interact | oracle_gate_pairwise (*) | 18 | 3/7 | -5.377 | 3/18 | 6.75 |
| hold_task_task_0003 | route_context_interact | sampled_oracle (*) | 18 | 18/0 | +15.195 | 3/18 | 6.75 |
| old_to_new | baseline15 | C3 | 0 | 0/0 | +0.000 | 26/54 | 24.14 |
| old_to_new | baseline15 | ridge_ref | 0 | 0/0 | +0.000 | 26/54 | 24.14 |
| old_to_new | baseline15 | pairwise_ridge_gate | 0 | 0/0 | +0.000 | 26/54 | 24.14 |
| old_to_new | baseline15 | learned_gate_pairwise | 98 | 21/21 | -2.249 | 26/54 | 24.14 |
| old_to_new | baseline15 | oracle_gate_pairwise (*) | 54 | 26/18 | +0.201 | 26/54 | 24.14 |
| old_to_new | baseline15 | sampled_oracle (*) | 54 | 54/0 | +25.674 | 26/54 | 24.14 |
| old_to_new | route_context | pairwise_ridge_gate | 0 | 0/0 | +0.000 | 20/54 | 24.14 |
| old_to_new | route_context | learned_gate_pairwise | 98 | 17/19 | +3.239 | 20/54 | 24.14 |
| old_to_new | route_context | oracle_gate_pairwise (*) | 54 | 20/17 | +3.995 | 20/54 | 24.14 |
| old_to_new | route_context | sampled_oracle (*) | 54 | 54/0 | +25.674 | 20/54 | 24.14 |
| old_to_new | route_context_interact | pairwise_ridge_gate | 0 | 0/0 | +0.000 | 19/54 | 24.14 |
| old_to_new | route_context_interact | learned_gate_pairwise | 98 | 16/20 | +1.654 | 19/54 | 24.14 |
| old_to_new | route_context_interact | oracle_gate_pairwise (*) | 54 | 19/19 | +2.441 | 19/54 | 24.14 |
| old_to_new | route_context_interact | sampled_oracle (*) | 54 | 54/0 | +25.674 | 19/54 | 24.14 |
| new_to_old | baseline15 | C3 | 0 | 0/0 | +0.000 | 2/7 | 2.25 |
| new_to_old | baseline15 | ridge_ref | 16 | 1/5 | -2.489 | 2/7 | 2.25 |
| new_to_old | baseline15 | pairwise_ridge_gate | 16 | 2/6 | -1.285 | 2/7 | 2.25 |
| new_to_old | baseline15 | learned_gate_pairwise | 0 | 0/0 | +0.000 | 2/7 | 2.25 |
| new_to_old | baseline15 | oracle_gate_pairwise (*) | 7 | 2/4 | -0.419 | 2/7 | 2.25 |
| new_to_old | baseline15 | sampled_oracle (*) | 7 | 7/0 | +5.921 | 2/7 | 2.25 |
| new_to_old | route_context | pairwise_ridge_gate | 16 | 1/5 | -1.923 | 1/7 | 2.25 |
| new_to_old | route_context | learned_gate_pairwise | 0 | 0/0 | +0.000 | 1/7 | 2.25 |
| new_to_old | route_context | oracle_gate_pairwise (*) | 7 | 1/4 | -1.296 | 1/7 | 2.25 |
| new_to_old | route_context | sampled_oracle (*) | 7 | 7/0 | +5.921 | 1/7 | 2.25 |
| new_to_old | route_context_interact | pairwise_ridge_gate | 16 | 1/5 | -1.923 | 1/7 | 2.25 |
| new_to_old | route_context_interact | learned_gate_pairwise | 0 | 0/0 | +0.000 | 1/7 | 2.25 |
| new_to_old | route_context_interact | oracle_gate_pairwise (*) | 7 | 1/4 | -1.296 | 1/7 | 2.25 |
| new_to_old | route_context_interact | sampled_oracle (*) | 7 | 7/0 | +5.921 | 1/7 | 2.25 |

(*) 事后Oracle策略读取留出未来收益，**不可部署**。

没有使用事后Road.replay成本作为模型特征；Route Context由原始当前/历史最优路线和候选Section边界构成。
Shuffle训练对照仅更改训练数据；不同状态/动作不是独立样本。

解释重点：真实状态门控是否比原Ridge门控提高正收益覆盖且不增加损失；Route Context与相同留组/相同策略的base15相比是否改善Top1正收益命中。
