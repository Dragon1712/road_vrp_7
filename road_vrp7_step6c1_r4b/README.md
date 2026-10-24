# Step6-C1-R4-B v1：官方ALNS 7.0.0 精确续接验证

## 核心目标

R4-A 的固定 2,000 迭代 Smoke 和固定 20,000 迭代 Pilot 均为 **8/8 原C3最终路线与 Road.replay 成本完全一致**。20k Pilot 包含 42 个实际 Section 选择前快照。R4-B 不训练新模型、不修改原 ALNS/Section/退火/轮盘实现，直接验证这些快照能否在后续 **K=1、5、20、50** 步形成与未中断的**官方 C3**相同的每一步轨迹。

该验证不等价于「精确恢复剩余30秒」。墙钟剩余时间和执行硬件状态不能以离线快照严格恢复，本实验仅验证 **固定 K 次迭代** 的连续搜索语义。

## 已确认的原始数据

- GitHub `Dragon1712/road_vrp_7` commit `22de792fe92dba513cb8303df5def6edd3ee0af0`
- 源快照：`runs/chicago_july_v1/step6c1_r4a/r4a_pilot_20k_fix1_capture/snapshots.jsonl`
- 20k固定迭代：8组、42条真实Section决策边界状态；全部达到迭代上限，源R4-A一致性审计8/8
- 任务：`task_0008,task_0009`，场景 `2025-07-22_0800,2025-07-22_1700`，种子60、61
- 只是先前看过的July Validation开发数据，**不是新盲测**

## 文件

- `r4b_core.py`：R4-A快照格式与SHA验证、原迭代观察器、源代码等价的恢复K步搜索和逐步严格比较
- `verify_r4b.py`：加载同一OD/任务后，调用未修改`v3.solve()`获得官方连续参考，再以**另一条冻结的`v3.solve()`**构建原算子闭包，将快照注入迭代入口
- `preflight_r4b.py`：检查源R4-A数据/快照SHA和schema
- `run_r4b.sh`：完整入口
- `tests/test_r4b_core.py`：负对照与虚拟ALNS端到端20步恢复验证

## 为什么这才是连续续接，而不是旧的重启代理？

R4-A保存的边界是 `BEFORE_SELECT_AND_DESTROY`，其中包含 **当前接受路线、历史最优路线、RouletteWheel两组权重、SimulatedAnnealing温度、NumPy RNG bit-generator状态，以及已调用停止器的迭代计数**。

原始连续路径：直接调用官方ALNS 7.0.0 的 `ALNS.iterate`，仅由轻量的透明观察器记录关键窗口的轨迹。

恢复路径：原本的 `v3.solve('C3_topk')`构造未修改的 destroy/repair/roulette/acceptance/统计闭包，进入 `ALNS.iterate` 时恢复快照。根据官方 v7.0.0 源码模拟余下固定K步的 `selector -> destroy -> repair -> _eval_cand -> selector.update`；每步使用真实冻结的算子和真实 SA/轮盘，未调用模型、不使用未来道路成本、不改变C3动作。

严格比较项目：

1. 迭代索引、当前/最优路线 SHA、当前/最优搜索目标值；
2. 随机数状态 SHA、Destroy/Repair权重、接受温度；
3. 实际选择的Destroy/Repair索引；
4. Repair后候选路线SHA、ALNS Outcome、更新后的算子权重和退火温度。

**只要任意字段、任意一步不一致就失败，不以最后道路成本一致代替逐步一致。**

## 环境要求

在你的现有 WZLroadvrp2 环境中运行，必须有 `alns==7.0.0`、NumPy、Pandas和Chicago已生成的Stage3/Stage5/Stage6A数据。要保留原仓库目录、尤其：

- `road_vrp7_step6c1_r4a/r4a_core.py`（SHA固定）
- `road_vrp7_step5b_v3/run_experiment.py`（SHA固定）
- `road_vrp7_step5b_v3/section_v2.py`（SHA固定）
- 原 R4-A Capture实验目录及其`manifest.json / snapshots.jsonl / provenance.json`

## 命令

### 1. 解压和本地自检

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r4b_v1.zip
bash road_vrp7_step6c1_r4b/run_r4b.sh test
bash road_vrp7_step6c1_r4b/run_r4b.sh preflight
```

### 2. 最小官方 ALNS Smoke（1个真实案例、1个快照、K=1/5/20/50）

```bash
STUDY_NAME=r4b_exact_smoke_1case_v1 \
  bash road_vrp7_step6c1_r4b/run_r4b.sh smoke
```

**首个Smoke首先重新运行原始C3完整20,000次迭代以得到官方参考轨迹**，之后对快照分别恢复K=1/5/20/50步。验证报告：

```
runs/chicago_july_v1/step6c1_r4b/r4b_exact_smoke_1case_v1/report.md
```

### 3. 完整Pilot（8个真实案例、每例取前2个Section快照）

```bash
STUDY_NAME=r4b_exact_8cases_2snap_v1 \
  MAX_SNAPSHOTS_PER_RUN=2 \
  HORIZONS=1,5,20,50 \
  bash road_vrp7_step6c1_r4b/run_r4b.sh pilot
```

预计：8次C3连续参考轨迹 + 最多64次很短的固定K步续接，最多64个`快照×H`逐步一致性比较；实际结果由服务器报告为准。这里8×2×4=64，基于每例至少2个快照。**实际运行可能因CPU/道路环境有所不同，不提供运行时间保证。**

输出：

```
runs/chicago_july_v1/step6c1_r4b/r4b_exact_8cases_2snap_v1/
  manifest.json
  report.json
  report.md
  paired_rows.jsonl
```

如果中途发生数据、代码、参考轨迹或恢复结果不一致，写入`status=FAILED`与第一个错误，不能继续使用该目录作为成功的实验结果。重新试验请使用新的`STUDY_NAME`，**不要覆盖结果**。

## 通过标准

- `test`的8项本地测试全部通过。
- `preflight`核对原R4-A全部源文件SHA、Section真实快照。完整`smoke`还核对冻结求解器、Section、道路和模型源SHA。
- `smoke`报告`status: COMPLETE`且`exact_matches == n_comparisons`。
- `pilot`报告K=1、5、20、50各自精确一致，没有任一首步或中途差异。

如果出现`Original R4A snapshot does not match fresh frozen C3`，应先审计环境/数据/是否墙钟触发，不得强制覆盖SHA。如果K=1通过而K>1失败，检查该步的候选/接受/权重/退火/RNG和修复期限。时间依赖的deadline是主要候选原因，但不能未经查证就断言。

## 通过后做什么？

只有R4-B逐步一致性验证通过，R4-C才能在相同的真实ALNS当前解、最优解、随机数、温度和轮盘状态下，更改首次Section动作并测量K步收益。R4-C中需谨慎设计首步RNG耦合、后续算子权重更新和固定预算，严格区分固定K研究与30秒在线性能。

## 科研解释限制

1. 完整验证是**原C3原样续接**，不包含新的学习策略，不能声称任何道路成本改善。
2. 当前只验证固定K步、原July Validation开发任务的可恢复性。
3. 模拟精确恢复依赖官方`alns==7.0.0`的内部实现，不可任意升级依赖。
4. 恢复时固定K窗口不使用已消耗的墙钟Deadline（`stop.deadline=inf`），避免伪造30秒续接保证；真实墙钟超时路径需要未来单独处理。
5. 路线和RNG的原始快照仅用于受控开发，不应误当作未来特征。未来部署的策略只能使用当前可观测交通与当前搜索状态。
