# R4-M Train内部嵌套风险校准 Section选择器

数据：原R4-H + R4-K的 **1,572候选 / 200场景快照 / 100粗粒度搜索起点**。
所有阈值和经验分位数只由外层Train中的任务留一OOF预测估计；外层留组仅用于报告。
所有结果是固定K500离线Road.replay收益，**不是独立Validation/Test或30秒在线结果**。

## 预注册主比较：全特征、未打乱标签

| 留组 | 策略 | 校准有效 | 校准样本 | 委托 | 正/负 | K500均值(s) | 伤害总和(s) |
|---|---|---|---:|---:|---|---:|---:|
| hold_task_task_0000 | ridge_ref | N/A | — | 18 | 5/2 | +0.281 | -27.693 |
| hold_task_task_0000 | lcb_q50 | True | 68 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0000 | lcb_q75 | True | 68 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0000 | lcb_q90 | True | 68 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0000 | dual_lcb_q75 | True | 68 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0001 | ridge_ref | N/A | — | 36 | 0/7 | -5.525 | -265.186 |
| hold_task_task_0001 | lcb_q50 | True | 72 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0001 | lcb_q75 | True | 72 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0001 | lcb_q90 | True | 72 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0001 | dual_lcb_q75 | True | 72 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0002 | ridge_ref | N/A | — | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0002 | lcb_q50 | False | 0 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0002 | lcb_q75 | False | 0 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0002 | lcb_q90 | False | 0 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0002 | dual_lcb_q75 | False | 0 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0003 | ridge_ref | N/A | — | 27 | 4/5 | +1.287 | -147.949 |
| hold_task_task_0003 | lcb_q50 | True | 68 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0003 | lcb_q75 | True | 68 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0003 | lcb_q90 | True | 68 | 0 | 0/0 | +0.000 | +0.000 |
| hold_task_task_0003 | dual_lcb_q75 | True | 68 | 0 | 0/0 | +0.000 | +0.000 |
| old_to_new | ridge_ref | N/A | — | 0 | 0/0 | +0.000 | +0.000 |
| old_to_new | lcb_q50 | False | 2 | 0 | 0/0 | +0.000 | +0.000 |
| old_to_new | lcb_q75 | False | 2 | 0 | 0/0 | +0.000 | +0.000 |
| old_to_new | lcb_q90 | False | 2 | 0 | 0/0 | +0.000 | +0.000 |
| old_to_new | dual_lcb_q75 | False | 2 | 0 | 0/0 | +0.000 | +0.000 |
| new_to_old | ridge_ref | N/A | — | 16 | 1/5 | -2.489 | -205.525 |
| new_to_old | lcb_q50 | True | 88 | 0 | 0/0 | +0.000 | +0.000 |
| new_to_old | lcb_q75 | True | 88 | 0 | 0/0 | +0.000 | +0.000 |
| new_to_old | lcb_q90 | True | 88 | 0 | 0/0 | +0.000 | +0.000 |
| new_to_old | dual_lcb_q75 | True | 88 | 0 | 0/0 | +0.000 | +0.000 |

## 报告解释边界

- 阈值0与q50/q75/q90均为事先定义的Train-only诊断，禁止根据本轮留出收益挑选最优参数并宣称外部泛化。
- 经验任务留一残差分位数**不提供形式化分布外覆盖保证**；校准失败时策略严格回退C3。
- 单独报告基准Ridge、Shuffle、几何特征消融以及任务和种子迁移；跨场景的重复起点不视为独立任务。
- 不会修改官方ALNS、源数据或任何R4-I/L产物；没有模型被部署到30秒优化器。
