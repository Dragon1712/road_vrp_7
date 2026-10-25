# R4-G TRAIN-only 非干预审计

严格结论：**NOT_VERIFIED**；固定迭代通过：**7/16**；路线和道路成本一致：**16/16**。
快照：**128**；去重物理搜索状态：**128**；跨场景重复状态：**0**。

## 失败原因统计

失败配对：9；原因计数：{"ITERATION_CAP_NOT_MET": 9, "ITERATION_COUNT_DIFFERS": 9}。

| Task | Scenario | Seed | Plain/Capture迭代 | Plain/Capture停止原因 | 路线相同 | Road差(Capture−Plain) | 强一致 | 失败原因 |
|---|---|---:|---|---|---|---:|---|---|
| task_0000 | 2025-07-02_0800 | 62 | 19638/18142 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0000 | 2025-07-02_0800 | 63 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0000 | 2025-07-09_1700 | 62 | 19927/18030 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0000 | 2025-07-09_1700 | 63 | 20000/19976 | iteration_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0001 | 2025-07-02_0800 | 62 | 17919/16377 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0001 | 2025-07-02_0800 | 63 | 18885/17217 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0001 | 2025-07-09_1700 | 62 | 17804/16262 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0001 | 2025-07-09_1700 | 63 | 18801/17290 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0002 | 2025-07-02_0800 | 62 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0002 | 2025-07-02_0800 | 63 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0002 | 2025-07-09_1700 | 62 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0002 | 2025-07-09_1700 | 63 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0003 | 2025-07-02_0800 | 62 | 19319/17577 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0003 | 2025-07-02_0800 | 63 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0003 | 2025-07-09_1700 | 62 | 19444/17610 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0003 | 2025-07-09_1700 | 63 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |

## 判读边界

- ITERATION_CAP_NOT_MET：至少一个分支未跑满固定迭代上限，不能证明固定迭代无干预，不自动等于算法逻辑已改变。
- FINAL_ROUTE_MISMATCH：末端路线不同；如都达到迭代上限，需要进一步检查算子/Repair的时间相关分支。
- ROAD_COST_MISMATCH：Road.replay成本不同，不能忽略。
- 这份报告只诊断Train捕获；不是恢复证明，也不是30秒性能改善。
- 失败配对的状态不得作为已严格验证的无干预捕获。
