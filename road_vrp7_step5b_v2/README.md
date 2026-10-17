# Step5-B v2 — 时间敏感 + Section 精确候选 + 可靠性门控（2026-10-17）

目标仓库：[Dragon1712/road_vrp_7](https://github.com/Dragon1712/road_vrp_7)，本次审计基准主分支提交 `80245979a624095fc963bca45d0a79aa93218be6`（审计时读取）。

> 这是一套**已经写好的独立实验实现**，**不是已有论文源代码的逐字复制**。借鉴2024 Dragomir-Müller Section思想与Johnn的ALNS算子选择框架，预算感知指标与缺失门控为针对Chicago-TD-CVRP30设计的待验证假设。**本地只有合成和模拟ALNS API测试，不含真实Chicago完整运行结果**。

## 一、代码结构

```
road_vrp7_step5b_v2/
  section_v2.py          Section选择、观测门控、负对照、整段移动/反转/跨车交换
  time_selector.py       官方ALNS 7.0.0兼容的时间敏感UCB算子对选择器
  run_experiment.py      C0-C6官方ALNS运行器，独立日志与哈希/续跑检查
  analyze_step5b_v2.py   结果完整性校验、任务级Bootstrap、配对分析
  run_stage5b_v2.sh      test/smoke/pilot/validation/july_recent/低覆盖/全月开发集
  tests/                 3个独立测试文件
  AUDIT.md               详细审计：不变式、对照边界、已知风险
```

该代码不修改旧的 `road_vrp7_stage3`、`road_vrp7_step5b` 或历史实验结果，运行时依赖它们的读取接口：

- `road_vrp7_stage3/road_vrp.py`: `Road,OD,Objective,legal,read_json,write_json,sha256`。
- `road_vrp7_stage3/solvers.py`: `VRPState,BothBudget,greedy,choose_destroy,insert_removed`。
- `road_vrp7_stage3/benchmark.py`: `load_tasks`。
- `road_vrp7_step5b/section_operators.py`: `SectionSelector,SectionState,repair_section,PathSupportProxy`。
- 官方 Python 库：`alns==7.0.0`（并非自写ALNS框架）。

## 二、实验定义

| 组别 | Destroy | Section修复 | 算子选择 | 主要变化 |
|---|---|---|---|---|
| C0_baseline | Stage3 Random/Worst/Related | 无Section | ALNS RouletteWheel | 冻结基线 |
| C1_original_section | 原Step5-B静态选择器（**强制单Section**） | 原Step5-B整段插入修复 | 同C0 | 基线Section效果；不等于旧Step5-B双Section完整版 |
| C2_noop_guard | 同C1静态选择 | 排除恢复原解，所有可行移动/反转位置用精确时变成本评价 | 同C0 | 删除原位无效候选 |
| C3_topk | v2静态Section选择 | 静态代理预筛Top-K，再用TD-OD精确计算，增加跨车Section交换 | 同C0 | 计算/邻域同时变化，**与C2不是严格单因素对照** |
| C4_time_ucb | 与C3完全相同 | 与C3完全相同 | 时间敏感UCB | 算子选择独立价值 |
| C5_gated | v2支持度门控Section评分 | 与C4完全相同 | 与C4完全相同 | 观测交通评分增益 |
| C6_shuffled | v2门控下把已观测候选交通评分错位 | 与C4完全相同 | 与C4完全相同 | 空间交通对齐负对照 |

对照重点：`C4 − C3`（仅改变选择器），`C5 − C4`（相同Section邻域和UCB，只引入动态信号），`C5 − C6`（交通对应关系负对照）。`C3 − C2`不能被称作Top-K的“单独因果效应”，因为额外加入了跨车交换且选择器实现从旧静态选择器变为新静态选择器。正式论文若要单独论证Top-K或Swap，需要再增加 `C3_no_swap` 和 `C3_full_eval_with_swap` 组。

`C1`单独禁用旧版本的双Section机制，是为了在C1-C2之间固定单Section修复对象，而不是为了声称与旧Step5-B B2算法逐字一致。历史 `B2` 应作为**外部历史参照**，不可直接与新实验作因果消融。

## 三、搜索预算和交通信息

- 所有算法固定Stage3 `5s` 搜索时钟与 `max_iterations=5000`；Greedy构造、OD预计算、道路回放在搜索之外，分别记录费用。**不能对外声称端到端5秒。**
- Section拆除和修复都在ALNS搜索时钟以内，所有候选按Stage3 `obj.cost()`完整传播到达时间；不使用静态代理作为最终接受/拒绝依据。
- C2精确穷举本Section全部可行整段插入/反转；C3-C6预筛静态代价排名，精确评价默认12个候选。若候选未能在截止前计算，保留此前已取得的最佳完整可行候选；无则回退原解。
- 为避免静态规划读未来数据，`planner=static`不读取动态交通矩阵和历史观测支持。C5/C6此时退化为相同静态选择方法，**不应期待Static模式上C5与C4出现交通机制性差异**，但随机路径可因算子选择而变化。
- Oracle使用历史完整速度轨迹做hindsight规划；不是面向在线部署的交通预测器。
- 交通支持采用旧Step5-B `PathSupportProxy`（静态最短路径的出发时刻观测掩码快照），不是动态最短路径的真实覆盖率。每个Section支持度 `g∈[0,1]`，评分公式：

  `S = Static_normalized + traffic_weight * g * (Traffic_normalized - Static_normalized)`。

  因此 `g=0` 时 `S=Static_normalized` **严格成立**；C6只在支持度>0的候选中打乱交通特征，并仍使用该候选自身的g。
- 交通特征是正向拥堵增量（相对静态）与20分钟历史趋势幅度的候选汇总，是**待证实的启发式**；不保证选择更好的Section。
- 时间UCB根据 `100*max(0,C(current)-C(candidate))/C(current)/elapsed_seconds` 估计即时候选改进效率，使用探索常数和随机探索。它**不是**在估计整个未来ALNS轨迹价值。探索和候选费用都计入官方ALNS搜索时钟。

## 四、Ubuntu运行命令（依次执行）

解压到项目根目录：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step5b_v2_代码与详细审计.zip
python -m pip show alns
# 必须是7.0.0，先检查现有环境，不要随意修改其他依赖
```

**第一步：运行不需要Chicago的测试。**

```bash
bash road_vrp7_step5b_v2/run_stage5b_v2.sh test
```

**第二步：真实官方ALNS+Chicago冒烟，7组×1任务×1场景×1种子=7条。**

```bash
bash road_vrp7_step5b_v2/run_stage5b_v2.sh smoke
bash road_vrp7_step5b_v2/run_stage5b_v2.sh analyze smoke
```

**第三步：Pilot，7组×3任务×2场景×2种子×Oracle=84条。**

```bash
bash road_vrp7_step5b_v2/run_stage5b_v2.sh pilot
bash road_vrp7_step5b_v2/run_stage5b_v2.sh analyze pilot
```

**第四步：完整验证，7组×10任务×8情景×3种子×2规划模式=3360条。**

```bash
bash road_vrp7_step5b_v2/run_stage5b_v2.sh validation
bash road_vrp7_step5b_v2/run_stage5b_v2.sh analyze validation
```

**第五步（可选）：此前使用过的7月26～31日作为开发扩展集。**

```bash
bash road_vrp7_step5b_v2/run_stage5b_v2.sh july_recent
bash road_vrp7_step5b_v2/run_stage5b_v2.sh analyze july_recent_development
```

还支持 `july_low_coverage`（QC不合格的7月27日压力测试）、`july_all_development`（7月所有符合QC的场景）。使用 `RESUME=1 bash ... pilot` 只允许在代码SHA/任务哈希/预算/参数不变、且CSV与JSONL主键完全对齐时续跑。半行写入不自动恢复，出错应人工核对并修复一致性。

输出在 `runs/chicago_july_v1/step5b_v2/<阶段>/` 下，包含 `run_config.json`,`metrics.csv`,`solutions.jsonl`,`benchmark_summary.json` 以及分析结果：`solver_summary.csv`,`paired_gains.csv`,`operator_pair_events.csv`,`analysis_summary.json`。

## 五、最重要的正式验收标准

1. `smoke`7/7 + `pilot`84/84，完全符合容量、客户唯一性、每车至少一名客户、所有候选目标值有限，记录完整。
2. `section_changed`, `section_unchanged`, `section_improving`, `section_new_best` 构成有意义的趋势；`C2`相对`C1`应该更少原位重插（不保证解质量更好）。
3. C3的 `section_candidate_exact` 和 `section_swap_attempts`必须>0；大量交换被筛掉可以正常，但最终接受的新解必须可行。
4. `C5`相对`C4` 的交通选择变化率>0才表示门控参与搜索；缺少观测的场景下应接近0。
5. 配对主指标始终是 ex-post TDSP道路回放行驶时间，按**任务聚类**Bootstrap，不把同一任务的多日期、多种子当独立样本。报告95%探索性区间、目标均值、求解时间/迭代、算子调用数。
6. 只有 `C5-C4` 和 `C5-C6` 同时体现可靠正收益，才继续考虑多模态编码；否则回到Section邻域和评价机制，**不要仅因本版本更复杂就假设性能提升**。

**注意**：目前文件内容已经过静态审计、运行了合成单元、7组模拟ALNS协议和分析器负向测试；**尚未**对你服务器上Chicago成本/官方ALNS安装执行真实 `smoke`。如果安装环境的官方API与文档不一致，请保留报错与版本信息再修复，不应继续扩大实验。
