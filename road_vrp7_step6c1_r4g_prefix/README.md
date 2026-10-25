# R4-G-Prefix v1：原Train状态的墙钟截断前缀精确续接审计

**源码/结果参照**：[`Dragon1712/road_vrp_7` commit `3934546e17c03fe0544ab0fefab68581cf2f2a9e`](https://github.com/Dragon1712/road_vrp_7/commit/3934546e17c03fe0544ab0fefab68581cf2f2a9e)。该包是**独立新增目录**，复用服务器已有的冻结 `road_vrp7_step6c1_r4g`、`r4b`、`r4a`、`step5b_v3`、`road_vrp7_stage3` 和官方 `alns==7.0.0`。**不要覆盖官方ALNS、原v3.solve、R4-A/B、历史R4-G数据或原 `verify_train_r4g.py`。**

## 为什么需要

GitHub最新真实记录：R4-G 16组Plain/Capture的终态SHA和 Road.replay **16/16相同**，但由于30秒墙钟，仅 **9/16** 组两者都完成20,000次迭代，严格全文证明`NOT_VERIFIED`，原数据不可宣称“16/16非干预”。获取76个真实Train Section快照，53个不同物理ALNS状态（另外23个是在不同交通场景重用的物理状态）。第一例 R4-G Train续接已经 **9/9**（3快照×K=1/50/500）逐事件完全一致；其他快照尚未证明。

R4-G-Prefix不修改或洗白上述7组墙钟截断的严格失败，而是建立独立证据：

> 一个既有30秒训练轨迹产生的**真实Section起点**，能否在官方ALNS的**固定K步**中逐事件精确恢复？

根据原30秒Plain和Capture各自结束时的迭代数，事先约束：

`start_stop_count + K - 1 <= min(plain_iterations, capture_iterations)`。

超出范围的窗口只在`excluded_windows.json`标注`SOURCE_30S_PREFIX_TOO_SHORT`，**绝不创建假的exact标签**。

从GitHub原始76个快照逐条计算，理论可核验窗口数量如下（这不是已通过的实测结果）：

| K | 源30秒共同迭代范围允许 | 排除 |
|---:|---:|---:|
| 1 | 76 | 0 |
| 50 | 76 | 0 |
| 200 | 74 | 2 |
| 500 | 72 | 4 |
| 合计 | **298** | **6** |

## 严格验证的三层证据

1. **原始30秒证据**：R4-G Train-only source manifest、快照、Plain/Capture metrics/provenance的 SHA 全部重新读取。只允许原Train task/scenario、原冻结成本图与C3，要求16组原终态Route SHA/Road值一样；旧9/16完整20k的状态**照旧，不做篡改**。
2. **新的未中断原生参考路径**：用相同的冻结 `v3.solve('C3_topk',...)` 和**官方ALNS 7.0.0原生`ALNS.iterate`**重新跑原任务、日期、seed，引用 `R4-B reference_recording` 只读采集所需迭代的事件。这个参考运行的墙钟上限默认 **90秒**，唯一目的是避免原30秒截断导致没有足够的参考事件。任务、目标、20k最大迭代数、退火跨度、随机种子与所有算子保持不变。若超过90秒仍未覆盖请求窗口，就失败；不偷偷跳过。
3. **原R4-B真实快照恢复**：每个Section快照首先用 `check_start()` 严格匹配新原生参考的路线SHA、RNG、RouletteWheel、SA温度、代理成本、真实算子对。随后通过R4-B `exact_resume` 从保存状态恢复**最长**允许的K，和新原始参考轨迹使用 `verify_events()`逐字段比对。对K较短的项目，复用同一个最长轨迹的真正前缀，逐K验证。

这说明新的证明是**固定K连续搜索**，**不是**重新得到原30秒求解器截止时的决策，也不是30秒后还有90秒可用。初始checkpoint必须在原30秒实际产生；参考只用于算法路径证据。

## 第一步：安装与本地测试

在服务器仓库根目录：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r4g_prefix_v1.zip
bash road_vrp7_step6c1_r4g_prefix/run_prefix.sh test
```

测试涵盖状态/长度有效性、原Plain/Capture终态不一致拒绝、重复快照拒绝、SHA和checkpoint身份、完整与破坏的逐事件比较、模拟真实“最长恢复→多K证明”工作流；无需额外第三方包。本地模拟测试不能代替真正Chicago服务器Proof。

## 第二步：只读规划审核——不运行ALNS

```bash
SOURCE_STUDY=r4g_train16_fix2 \
  bash road_vrp7_step6c1_r4g_prefix/run_prefix.sh plan
```

用原R4-G数据角色/完整SHA审计，输出实际可验证K数量。预期16个案例、76快照、53个物理状态、16/16终态一致，以及K=1/50/200/500分别为76/76/74/72的候选**覆盖**。若不一致，先检查`SOURCE_STUDY`和代码/原始来源，不运行求解器。

## 第三步：真实1案例Smoke

```bash
SOURCE_STUDY=r4g_train16_fix2 \
STUDY_NAME=r4g_prefix_onecase_v1 \
  bash road_vrp7_step6c1_r4g_prefix/run_prefix.sh smoke

cat runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_onecase_v1/report.md
```

默认对第一个case选3个Section快照，K=1/50/200/500，理论12个窗口。第一个案例旧版1/50/500已9/9通过，但K200及新版起点/事件检查仍必须在真实服务器重新执行。全部通过再扩大。

## 第四步：所有16案例的完整前缀证明

```bash
SOURCE_STUDY=r4g_train16_fix2 \
STUDY_NAME=r4g_prefix_all16_v1 \
REFERENCE_WALL_SEC=90 \
  bash road_vrp7_step6c1_r4g_prefix/run_prefix.sh pilot

cat runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1/report.md
```

默认全部Section快照、不裁剪每个任务，预计允许298个快照×K组合。每次只为每个快照恢复一次最长K，不重复跑四次。全部通过才可报告“298/298精确续接”，实际服务端可能在某个case失败，严禁提前宣布。即使全部通过，原R4-G严格全文审计仍是**9/16**。

如果仅为定位问题希望首轮固定少数状态，可通过Smoke覆盖更小的上限：

```bash
MAX_SNAPSHOTS_PER_CASE=2 STUDY_NAME=r4g_prefix_two_states_v1 \
SOURCE_STUDY=r4g_train16_fix2 \
  bash road_vrp7_step6c1_r4g_prefix/run_prefix.sh smoke
```

## 中断续跑

Proof独立新目录，每个 `(task, scenario, seed)` 校验成功后原子保存到 `units/*.json`。运行中断时，**保留相同的全部配置**并在命令前加`RESUME=1`：

```bash
SOURCE_STUDY=r4g_train16_fix2 \
STUDY_NAME=r4g_prefix_all16_v1 \
REFERENCE_WALL_SEC=90 RESUME=1 \
  bash road_vrp7_step6c1_r4g_prefix/run_prefix.sh pilot
```

复用时会验证原始快照SHA、代码/配置哈希、每个case全部预期K和`exact=true`，拒绝重复/缺失/污染的单元。结果目录存在但未设置RESUME时失败；不会覆盖R4-G既有数据。

## 输出及统计意义

`runs/chicago_july_v1/step6c1_r4g_prefix/<STUDY_NAME>/`：

- `manifest.json`：Source SHA、GitHub原冻结C3/Section/R4-A/B/G代码SHA、样本计划、原30秒和新参考墙钟预算。
- `excluded_windows.json`：超过原始30秒共同迭代长度的每个窗口及具体证据。
- `units/*.json`：已通过全轨迹逐字段比较的case级证明，支持断点续跑。
- `paired_rows.jsonl`：逐Section×K真正验证通过的证据行。
- `report.json/report.md`：按K分组的`exact/eligible/excluded`和原9/16完整迭代审计限制。

原始状态的`task_id`、日期、种子和序号**不是**独立训练案例；不同交通场景复用的物理搜索状态应去重，源R4-G记录为53个。

## 然后再做什么：R4-H

**只有本轮所有目标窗口成功证明后**，才能开发新的TRAIN首步Section反事实标签收集：保持旧R4-C `C3`、`SHAM_C3`、替代Section配对和最初C3 RNG消耗语义；每个Train快照采用已验证K（K=50/200/500，若K500被排除就不能硬生成K500标签）。输出当前解和历史最优解Road收益及符号翻转。训练必须按照独立物理状态/任务分组，保留未使用的验证/最终盲测场景；此包没有训练模型，也没有生成收益标签。
