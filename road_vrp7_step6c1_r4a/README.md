## v1.1 补丁说明（Smoke 的 Plain 分支 UnboundLocalError）

原始 v1 的 `collect_r4a.py` 只在 `capture` 分支给 `ans` 赋值；
`plain` 分支却直接读取 `ans['iterations']`，导致 Smoke 立即报错。
本补丁修复为 `plain` 分支直接调用一次**原始冻结**的 `v3.solve('C3_topk', ...)`；
`capture` 分支原调用保持不变。增加真实collector控制流的合成端到端回归测试：
要求两种arm都运行C3、完成 `metrics.jsonl` / `report.json`，并通过固定迭代配对审计。

**已失败的原结果目录不会被复用、覆盖或删除。** 请解压v1.1并使用新的研究名称：

```bash
cd ~/code/WZL/road_vrp_7
unzip -o road_vrp7_step6c1_r4a_v1_1.zip
bash road_vrp7_step6c1_r4a/run_r4a.sh test
STUDY_PREFIX=r4a_smoke_2k_fix1 bash road_vrp7_step6c1_r4a/run_r4a.sh smoke
```

v1.1的 `run_r4a.sh` 会提前检测Plain/Capture/Audit结果目录是否存在，给出明确错误信息。

---

# Step6-C1-R4-A v1 — 官方 ALNS 7.0.0 连续搜索状态捕获 / 无干预审计

## 0. 为什么现在优先做这个实验

GitHub 提交 `be46ed5c8ff5b3dd657cb65a16f2c44131a3f55b` 的 VQ6 Pilot 是两个 task (`task_0008/0009`) × 两个 7月22日 Validation 场景 × 搜索 seed 60/61 = 8 个配对案例：

| 策略 | C3 相对的平均道路收益（秒/例，正为好） | 胜/负/平 | 每例有效 Section 覆盖 | 搜索评分耗时（秒/例） |
|---|---:|---:|---:|---:|
| 强制推理后回退 C3 (`DEFER_RESID`) | 0.000 | 0/0/8 | 0 | 0.502 |
| 一步 `E0_MATCHED` | **-12.984** | 2/5/1 | 37.25 | 0.383 |
| 多步残差 `RESID_OOF50` | **-21.203** | 3/5/0 | 35.88 | 0.632 |
| RESID vs E0 | -8.219 | 2/4/2 | — | — |

**研究判定**：仅有 8 个相关验证配对，不能证明普遍退化；但这一 Pilot 完全不支持推广 RESID。下一步研究重心从“调更多多步模型”转到检查离线从修复路线**重启**的监督目标是否与官方 ALNS 真实的**连续状态**错位。

官方 `alns==7.0.0` 每个迭代包含 `[调用 stop] → [选择 destroy/repair] → [执行算子] → [候选评估并更新当前解和最优解] → [更新算子权重]`。仅保存当前 routes 远远不够。R4A 在 **选择算子之前** read-only 捕获当前路线、最优路线、轮盘算子权重、退火温度、NumPy RNG state、停止计数等；不修改官方选算子、修复和接受规则。

R4A 还**不提供精确续接或反事实多步标签**。续接需要后续 R4B 在同一个迭代边界完整重建接受准则、最佳解和随机数，并通过原轨迹回放一致性校验后才能宣称成功。

## 1. 安装

要求服务器原 Conda 环境 `alns==7.0.0`、Chicago July 数据、Step5-B V3、Stage3源码和OD缓存均在仓库原位置；解压后：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r4a_v1.zip
bash road_vrp7_step6c1_r4a/run_r4a.sh test
python -c "import importlib.metadata; print(importlib.metadata.version('alns'))"
```

测试无外部数据依赖；包括纯函数快照单元测试和本地合成 ALNS 调用。一旦真实服务器上的 `alns` / `v3.solve` 接口变化，collector 会 fail closed，不会伪造温度、轮盘权重。

## 2. 第一步：**强制确定性 smoke**（推荐第一步必须跑）

```bash
bash road_vrp7_step6c1_r4a/run_r4a.sh smoke
```

默认：`task_0008`、`2025-07-22_0800`、seed60，先原始 C3 plain（**无 hook**），再官方C3 capture（**只用代理观测 selector**）；双方都是原来的 `--wall-sec=30`，但最高仅 2000 iterations（不是完整30秒性能比较）。Capture 在迭代 100、500、1000、1500 之前抓取状态（Smoke不启用Section过滤），并在搜索结束后写盘。

结果：

- `runs/chicago_july_v1/step6c1_r4a/r4a_smoke_2k_v1_plain/`
- `runs/chicago_july_v1/step6c1_r4a/r4a_smoke_2k_v1_capture/`
- `runs/chicago_july_v1/step6c1_r4a/r4a_smoke_2k_v1_audit/report.md`

**硬性判断标准**：Plain/Capture 都因2000次迭代停止、迭代数相同、最终路线Hash相同、Road.replay成本相同，才能通过 `--require-exact`。一旦一方由于30秒wall先停，程序**判定 smoke 未通过**，应减少迭代到例如1000并使用对应采样点，而不是忽略失败。

例如一个更短的确定性 Smoke（不同独立新结果目录）：

```bash
STUDY_PREFIX=r4a_smoke_1000_v1 MAX_ITERATIONS=1000 SNAPSHOTS=100,300,600,900 \
  bash road_vrp7_step6c1_r4a/run_r4a.sh smoke
```

同名目录绝不覆盖。**必须先通过确定性 Smoke，再运行Pilot。**

## 3. 第二步：原 20,000 次 / 30秒上限 Pilot

```bash
bash road_vrp7_step6c1_r4a/run_r4a.sh pilot
```

默认：`task_0008/0009` × July 22日 08:00/17:00 × seed 60/61；分别运行 **plain 和 capture**，共16次固定30秒上限搜索（不是16个独立任务）。每例在第500、2500、5000、10000、15000、19000步**及其后第一个真实被选中的Section算子**开始前捕获内部状态（并非每个里程碑恰好有Section操作）；到墙钟上限后不可能继续收集更晚快照，这是正常情况。

三个结果目录：`r4a_pilot_20k_v1_{plain,capture,audit}`，位于 `runs/chicago_july_v1/step6c1_r4a/`。

**注意**：此阶段保留官方原 stop `BothBudget(20000,30)`。快照计算（取routes/obj.cost/复制RNG状态）发生在30秒内，写入JSON发生在搜索完成之后。因计算开销和执行时间抖动，墙钟提前停止的Plain/Capture可能获得不同路线；不能从这类差异断言“捕获改变了算法”，只能报告运行时误差。只有**双方都严格跑满迭代上限**的相同路线才构成强无干预证据。

如果只打算做小范围Pilot，可通过环境变量覆盖：

```bash
STUDY_PREFIX=r4a_pilot_small_v1 TASK_IDS=task_0008 SEEDS=60,61 \
  SCENARIO_IDS=2025-07-22_0800 SNAPSHOTS=250,1000,2500,5000 \
  bash road_vrp7_step6c1_r4a/run_r4a.sh pilot
```

## 4. 输出结构、状态边界和审计

`capture/snapshots.jsonl` 每条包括以下字段：

| 字段 | 含义 |
|---|---|
| `boundary=BEFORE_SELECT_AND_DESTROY` | 已完成上一个候选决策及算子更新，未抽取下一个算子 |
| `next_iteration_one_based`, `stop_count` | 当前迭代标号、停止规则进度 |
| `current_routes`, `best_routes` | 当前接受解 / 历史最优解，不混用 |
| `current_surrogate_cost`, `best_surrogate_cost` | 当前和最优静态OD代理成本 |
| `roulette_destroy_weights`, `roulette_repair_weights` | 真实官方 RouletteWheel 的4组destroy、2组repair权重 |
| `anneal_temperature_before_accept`, `anneal_{start,end,step,method}` | 真实 SimulatedAnnealing 对象；不是重启后重新autofit的温度 |
| `rng_bit_generator_state`, `rng_state_sha256` | 当前 NumPy Generator 的完整 bit-generator 状态，用于后续再现 |
| `wall_elapsed_before_select_sec`, `wall_remaining_before_select_sec` | 原30秒预算中的进度，采样时间包含于搜索预算 |
| `completed_updates_before_boundary` | 应等于 `next_iteration_one_based - 1` |
| `requested_milestone`, `real_selected_destroy_index`, `snapshot_is_actual_section_decision` | 20k Pilot中：该快照确实紧邻一次真实选择的Section操作；快照中的RNG为该次选择**之前**的状态 |

`metrics.jsonl`: 原 C3道路成本、搜索迭代、30秒墙钟、路线Hash、终止原因。`audit/report.json` 包含强确定性通过数、道路成本一致数、快照数、wall开销差和每例配对信息。`manifest.json`与`provenance.json`针对此次调用记录源码/输入/输出SHA。

**不对真实未来道路标签作模型输入**；道路Replay仅在搜索完成后。数据选择限制为**已使用的July Validation开发样本**。不要把本轮结果当作独立Blind Test。

## 5. 后续R4B/R4C（需R4A通过后再实施）

1. **R4B Exact Replay/Resume**：把捕获的 `rng_bit_generator_state`、`current_routes`、`best_routes`、RouletteWheel权重、真实SA温度和迭代进度恢复为续接对象；先不改变任何Section，要求从同一状态继续跑K步所得算子、接受、当前解与历史最优解逐步同轨迹一致。
2. **R4C One-action Counterfactual**：在同一捕获状态的第一个 Section 时刻，将候选动作替换为一个不同Section；之后使用一致的停止/退火/权重设置，执行K步以上的 paired continuation，计算真实的连续搜索相对C3效果。
3. **R4D On-policy Aggregation**：只在R4C因果对照可信且收益可稳定估计后，使用DAgger-style状态访问聚合；随后再测试FiLM/SPCLT交通表示能否进一步解释“在哪些状态适宜学习式委托”。

## 6. 与官方库源码的关系

此工具**不修改** `road_vrp7_step5b_v3/run_experiment.py`、`section_v2.py`及官方ALNS库。它仅在 `v3.solve` 调用期间临时拦截 `ALNS.iterate` 的 selector 入参，并在 `finally` 中恢复原方法。代理对原 `selector(rng,best,curr)` 和 `selector.update(cand,...)` 各恰好转发一次。R4A不触碰原随机数抽样逻辑。该策略依据官方 ALNS 7.0.0 迭代顺序设计。

本机缺少 `alns` 安装及Chicago原始运行数据；已做**合成及接口级**测试，真实服务器集成必须先过 Smoke，尚未宣称完整Chicago端到端通过。
