# R4-G TRAIN-only 非干预审计

严格结论：**NOT_VERIFIED**；固定迭代通过：**9/16**；路线和道路成本一致：**16/16**。
快照：**76**；去重物理搜索状态：**53**；跨场景重复状态：**23**。

## 失败原因统计

失败配对：7；原因计数：{"ITERATION_CAP_NOT_MET": 7, "ITERATION_COUNT_DIFFERS": 7}。

| Task | Scenario | Seed | Plain/Capture迭代 | Plain/Capture停止原因 | 路线相同 | Road差(Capture−Plain) | 强一致 | 失败原因 |
|---|---|---:|---|---|---|---:|---|---|
| task_0000 | 2025-07-02_0800 | 60 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0000 | 2025-07-02_0800 | 61 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0000 | 2025-07-09_1700 | 60 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0000 | 2025-07-09_1700 | 61 | 19980/20000 | wall_budget/iteration_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0001 | 2025-07-02_0800 | 60 | 15812/16018 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0001 | 2025-07-02_0800 | 61 | 18123/17974 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0001 | 2025-07-09_1700 | 60 | 15825/15930 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0001 | 2025-07-09_1700 | 61 | 18194/17936 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0002 | 2025-07-02_0800 | 60 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0002 | 2025-07-02_0800 | 61 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0002 | 2025-07-09_1700 | 60 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0002 | 2025-07-09_1700 | 61 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0003 | 2025-07-02_0800 | 60 | 19625/19547 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0003 | 2025-07-02_0800 | 61 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |
| task_0003 | 2025-07-09_1700 | 60 | 19569/19388 | wall_budget/wall_budget | True | +0.000 | False | ITERATION_CAP_NOT_MET,ITERATION_COUNT_DIFFERS |
| task_0003 | 2025-07-09_1700 | 61 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |

## 判读边界

- ITERATION_CAP_NOT_MET：至少一个分支未跑满固定迭代上限，不能证明固定迭代无干预，不自动等于算法逻辑已改变。
- FINAL_ROUTE_MISMATCH：末端路线不同；如都达到迭代上限，需要进一步检查算子/Repair的时间相关分支。
- ROAD_COST_MISMATCH：Road.replay成本不同，不能忽略。
- 这份报告只诊断Train捕获；不是恢复证明，也不是30秒性能改善。
- 失败配对的状态不得作为已严格验证的无干预捕获。
