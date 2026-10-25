# R4-Q Train-only 24h滞后交通模态消融

**仅使用出发前24小时已观测的Stage2原始历史旅行时间；不是实时交通预测或神经FiLM。**

| 留组 | 特征 | learned_gate+pairwise K500收益(秒/快照) | 委托 | 正/负 |
|---|---|---:|---:|---|
| hold_task_task_0000 | route_context | -3.800 | 18 | 3/5 |
| hold_task_task_0000 | static_concat | -1.851 | 18 | 3/5 |
| hold_task_task_0000 | lag24_concat | +0.063 | 18 | 4/4 |
| hold_task_task_0000 | lag24_conditioned | +0.063 | 18 | 4/4 |
| hold_task_task_0001 | route_context | -5.525 | 20 | 0/7 |
| hold_task_task_0001 | static_concat | -5.525 | 20 | 0/7 |
| hold_task_task_0001 | lag24_concat | -5.525 | 20 | 0/7 |
| hold_task_task_0001 | lag24_conditioned | -5.525 | 20 | 0/7 |
| hold_task_task_0002 | route_context | +4.542 | 12 | 6/3 |
| hold_task_task_0002 | static_concat | +3.824 | 12 | 6/3 |
| hold_task_task_0002 | lag24_concat | +4.167 | 12 | 6/3 |
| hold_task_task_0002 | lag24_conditioned | +4.167 | 12 | 6/3 |
| hold_task_task_0003 | route_context | -2.019 | 14 | 4/4 |
| hold_task_task_0003 | static_concat | -2.642 | 14 | 3/4 |
| hold_task_task_0003 | lag24_concat | -3.495 | 14 | 3/4 |
| hold_task_task_0003 | lag24_conditioned | -3.495 | 14 | 3/4 |
| old_to_new | route_context | +3.239 | 98 | 17/19 |
| old_to_new | static_concat | +2.045 | 98 | 17/22 |
| old_to_new | lag24_concat | +2.725 | 98 | 17/22 |
| old_to_new | lag24_conditioned | +2.725 | 98 | 17/22 |
| new_to_old | route_context | +0.000 | 0 | 0/0 |
| new_to_old | static_concat | +0.000 | 0 | 0/0 |
| new_to_old | lag24_concat | +0.000 | 0 | 0/0 |
| new_to_old | lag24_conditioned | +0.000 | 0 | 0/0 |

## 历史观测支持度

| Train场景 | 出发时刻 | 请求的24h时槽 | 真实时槽可用 | 实际使用时槽 | 全路网观测比例 | 已映射边观测比例 |
|---|---|---|---|---|---:|---:|
| 2025-07-02_0800 | 2025-07-02 08:00:00 | 2025-07-01 08:00:00 | False | 无（静态回退） | 0.000% | 0.000% |
| 2025-07-09_1700 | 2025-07-09 17:00:00 | 2025-07-08 17:00:00 | True | 2025-07-08 17:00:00 | 8.268% | 89.216% |

## 按实际历史可用性分层的模型结果（非Shuffle learned_gate_pairwise）

| 留组 | 特征 | 场景 | 历史可用 | 快照 | 委托 | 正/负 | 平均K500收益(s) |
|---|---|---|---|---:|---:|---|---:|
| hold_task_task_0000 | lag24_concat | 2025-07-02_0800 | False | 25 | 9 | 2/3 | -0.054 |
| hold_task_task_0000 | lag24_concat | 2025-07-09_1700 | True | 25 | 9 | 2/1 | +0.180 |
| hold_task_task_0000 | lag24_conditioned | 2025-07-02_0800 | False | 25 | 9 | 2/3 | -0.054 |
| hold_task_task_0000 | lag24_conditioned | 2025-07-09_1700 | True | 25 | 9 | 2/1 | +0.180 |
| hold_task_task_0000 | route_context | 2025-07-02_0800 | False | 25 | 9 | 1/3 | -6.640 |
| hold_task_task_0000 | route_context | 2025-07-09_1700 | True | 25 | 9 | 2/2 | -0.960 |
| hold_task_task_0000 | static_concat | 2025-07-02_0800 | False | 25 | 9 | 1/4 | -3.882 |
| hold_task_task_0000 | static_concat | 2025-07-09_1700 | True | 25 | 9 | 2/1 | +0.180 |
| hold_task_task_0001 | lag24_concat | 2025-07-02_0800 | False | 24 | 10 | 0/4 | -7.383 |
| hold_task_task_0001 | lag24_concat | 2025-07-09_1700 | True | 24 | 10 | 0/3 | -3.667 |
| hold_task_task_0001 | lag24_conditioned | 2025-07-02_0800 | False | 24 | 10 | 0/4 | -7.383 |
| hold_task_task_0001 | lag24_conditioned | 2025-07-09_1700 | True | 24 | 10 | 0/3 | -3.667 |
| hold_task_task_0001 | route_context | 2025-07-02_0800 | False | 24 | 10 | 0/4 | -7.383 |
| hold_task_task_0001 | route_context | 2025-07-09_1700 | True | 24 | 10 | 0/3 | -3.667 |
| hold_task_task_0001 | static_concat | 2025-07-02_0800 | False | 24 | 10 | 0/4 | -7.383 |
| hold_task_task_0001 | static_concat | 2025-07-09_1700 | True | 24 | 10 | 0/3 | -3.667 |
| hold_task_task_0002 | lag24_concat | 2025-07-02_0800 | False | 26 | 6 | 4/1 | +2.433 |
| hold_task_task_0002 | lag24_concat | 2025-07-09_1700 | True | 26 | 6 | 2/2 | +5.901 |
| hold_task_task_0002 | lag24_conditioned | 2025-07-02_0800 | False | 26 | 6 | 4/1 | +2.433 |
| hold_task_task_0002 | lag24_conditioned | 2025-07-09_1700 | True | 26 | 6 | 2/2 | +5.901 |
| hold_task_task_0002 | route_context | 2025-07-02_0800 | False | 26 | 6 | 4/1 | +2.433 |
| hold_task_task_0002 | route_context | 2025-07-09_1700 | True | 26 | 6 | 2/2 | +6.651 |
| hold_task_task_0002 | static_concat | 2025-07-02_0800 | False | 26 | 6 | 4/1 | +2.433 |
| hold_task_task_0002 | static_concat | 2025-07-09_1700 | True | 26 | 6 | 2/2 | +5.214 |
| hold_task_task_0003 | lag24_concat | 2025-07-02_0800 | False | 25 | 7 | 1/2 | -7.623 |
| hold_task_task_0003 | lag24_concat | 2025-07-09_1700 | True | 25 | 7 | 2/2 | +0.632 |
| hold_task_task_0003 | lag24_conditioned | 2025-07-02_0800 | False | 25 | 7 | 1/2 | -7.623 |
| hold_task_task_0003 | lag24_conditioned | 2025-07-09_1700 | True | 25 | 7 | 2/2 | +0.632 |
| hold_task_task_0003 | route_context | 2025-07-02_0800 | False | 25 | 7 | 2/2 | -6.378 |
| hold_task_task_0003 | route_context | 2025-07-09_1700 | True | 25 | 7 | 2/2 | +2.340 |
| hold_task_task_0003 | static_concat | 2025-07-02_0800 | False | 25 | 7 | 1/2 | -7.623 |
| hold_task_task_0003 | static_concat | 2025-07-09_1700 | True | 25 | 7 | 2/2 | +2.340 |
| new_to_old | lag24_concat | 2025-07-02_0800 | False | 36 | 0 | 0/0 | +0.000 |
| new_to_old | lag24_concat | 2025-07-09_1700 | True | 36 | 0 | 0/0 | +0.000 |
| new_to_old | lag24_conditioned | 2025-07-02_0800 | False | 36 | 0 | 0/0 | +0.000 |
| new_to_old | lag24_conditioned | 2025-07-09_1700 | True | 36 | 0 | 0/0 | +0.000 |
| new_to_old | route_context | 2025-07-02_0800 | False | 36 | 0 | 0/0 | +0.000 |
| new_to_old | route_context | 2025-07-09_1700 | True | 36 | 0 | 0/0 | +0.000 |
| new_to_old | static_concat | 2025-07-02_0800 | False | 36 | 0 | 0/0 | +0.000 |
| new_to_old | static_concat | 2025-07-09_1700 | True | 36 | 0 | 0/0 | +0.000 |
| old_to_new | lag24_concat | 2025-07-02_0800 | False | 64 | 49 | 7/12 | +1.983 |
| old_to_new | lag24_concat | 2025-07-09_1700 | True | 64 | 49 | 10/10 | +3.466 |
| old_to_new | lag24_conditioned | 2025-07-02_0800 | False | 64 | 49 | 7/12 | +1.983 |
| old_to_new | lag24_conditioned | 2025-07-09_1700 | True | 64 | 49 | 10/10 | +3.466 |
| old_to_new | route_context | 2025-07-02_0800 | False | 64 | 49 | 7/11 | +2.586 |
| old_to_new | route_context | 2025-07-09_1700 | True | 64 | 49 | 10/8 | +3.892 |
| old_to_new | static_concat | 2025-07-02_0800 | False | 64 | 49 | 7/12 | +1.385 |
| old_to_new | static_concat | 2025-07-09_1700 | True | 64 | 49 | 10/10 | +2.705 |

**关键限制：** 7月2日缺少7月1日24h历史，完整使用静态回退；只有7月9日真实使用24h历史。因此整体200快照的差异不得全部归因于动态交通。模型仍是R4-O加权Pairwise Ridge与状态门控；K500离线Road收益不是30秒在线改善。
