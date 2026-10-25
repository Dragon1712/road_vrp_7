# R4-G TRAIN-only 非干预审计

严格结论：**VERIFIED**；固定迭代通过：**1/1**；路线和道路成本一致：**1/1**。
快照：**8**；去重物理搜索状态：**8**；跨场景重复状态：**0**。

## 失败原因统计

失败配对：0；原因计数：{}。

| Task | Scenario | Seed | Plain/Capture迭代 | Plain/Capture停止原因 | 路线相同 | Road差(Capture−Plain) | 强一致 | 失败原因 |
|---|---|---:|---|---|---|---:|---|---|
| task_0002 | 2025-07-02_0800 | 62 | 20000/20000 | iteration_budget/iteration_budget | True | +0.000 | True | — |

## 判读边界

- ITERATION_CAP_NOT_MET：至少一个分支未跑满固定迭代上限，不能证明固定迭代无干预，不自动等于算法逻辑已改变。
- FINAL_ROUTE_MISMATCH：末端路线不同；如都达到迭代上限，需要进一步检查算子/Repair的时间相关分支。
- ROAD_COST_MISMATCH：Road.replay成本不同，不能忽略。
- 这份报告只诊断Train捕获；不是恢复证明，也不是30秒性能改善。
- 失败配对的状态不得作为已严格验证的无干预捕获。
