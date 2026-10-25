# Step6-C1-R4-K v1：R4-J部分checkpoint可信审计、断点续采与完整Train标签核验

## 一、为什么必须先做R4-K

冻结GitHub提交：`Dragon1712/road_vrp_7@95b3fbf78166e8a9db2d6985b26e3ae3abc4f951`（2026-10-25）。

1. R4-J `r4j_train_early_s62s63_v1`：4个原Train任务×2场景×新seed62,63，128个Section快照，Plain/Capture最终路线与Road replay 16/16相同，20k迭代严格配对7/16（其余9/16墙限）；同一128快照K1/50/200/500逐步精确512/512。
2. R4-H格式的R4-J新seed反事实**8场景快照Pilot**已经完成：K500 62动作中2正、28负、32零，平均Road收益−19.435s；K200→500有9个正→负。Pilot仅8场景快照，不代表128完整快照。
3. 真实 `.../step6c1_r4h/r4j_train_early_s62s63_v1_cf_full` 当前**只有480个unit**、`partial.json`，**没有COMPLETE报告**。因此不能训练、不可以把缺失K/动作记成零收益，也不能把执行顺序前缀当作IID样本。

**R4-K的任务：** 完整保留已计算units，先审核部分完整快照及缺失清单，按原冻结R4-J/R4-H代码继续采集，再对全128快照生成完整、可审计Train标签集。**R4-K本身不训练模型，不修改ALNS，也不声称30s部署性能。**

## 二、文件

- `run_r4k.sh`：`test`、`partial`、`resume`、`complete`。
- `r4k_core.py`：源清单SHA、角色Train、逐unit来源、C3/SHAM、动作身份、首事件状态、跨K连续事件核验、完整快照筛选、收益分解。为全量CF按快照装载事件，避免数千branch同时常驻RAM。
- `audit_r4k.py`：只读审计、报告和JSONL标签重建；`complete`模式强制要求源Full有COMPLETE报告，并与源标签逐行Road/Static收益一致。
- `tests/`：部分数据不可误记零、SHAM篡改拒绝、源SHA破坏拒绝、跨K收益翻转、完整证明、精确温度区分等负对照。

没有修改源 `road_vrp7_step6c1_r4a`...`r4j` 或任何旧结果目录。

## 三、服务器运行

在原conda `(WZLroadvrp2)` 和仓库根目录：

```bash
cd ~/code/WZL/road_vrp_7
unzip -o road_vrp7_step6c1_r4k_v1.zip
bash road_vrp7_step6c1_r4k/run_r4k.sh test
```

### Step 1：审计GitHub中已完成的480个unit（不运行ALNS）

```bash
R4K_SOURCE_STUDY=r4j_train_early_s62s63_v1 \
STUDY_NAME=r4k_partial_480units_v1 \
  bash road_vrp7_step6c1_r4k/run_r4k.sh partial

cat runs/chicago_july_v1/step6c1_r4k/r4k_partial_480units_v1/report.md
```

输出为新独立目录：

- `report.md`, `report.json`：实际完整快照数、任务覆盖、K50/200/500分布、收益反转。
- `missing_units.json`：逐快照仍未完成的(`K`,arm)项；完全未开始的快照为`NO_PLAN_YET`。
- `audited_labels.jsonl`：**仅完整快照**的经校验反事实标签。
- `sampled_oracles.jsonl`：已采样候选事后Oracle，仅为参考上界。
- `K_transitions.jsonl`：每个同一动作K50→200、K200→500的收益、C3/ALT进步分解。
- `provenance.json`：采集源、证明源、冻结求解器源码与输出SHA。

**重要：** 部分报告没有统计独立性保证，任务覆盖可能严重偏斜；此阶段不要据此选择训练超参。当前实际units由源仓库决定，480是已查看提交时的数量，之后若服务器又运行过请用新的`STUDY_NAME`记录新的审计时间点。

### Step 2：沿用**既有**Full研究目录继续采集

无需再运行Plain、Capture、Prefix，也无需重跑有效R4-H checkpoints。

```bash
R4K_SOURCE_STUDY=r4j_train_early_s62s63_v1 \
R4K_MAX_NEW_UNITS=240 \
  bash road_vrp7_step6c1_r4k/run_r4k.sh resume
```

这一命令调用冻结的：

`R4J_SOURCE_STUDY=... R4J_MAX_NEW_UNITS=240 RESUME=1 bash road_vrp7_step6c1_r4j/run_r4j.sh cf-full`

可多次重复上述`resume`，直到原 `cf_full/report.json` 写出`status=COMPLETE`。**仅当报告已完成时，才允许`complete`审计**。若算力允许，将`R4K_MAX_NEW_UNITS=0`设置为不限本轮新unit，由源R4-H自行完成；可能较耗时，建议先240验证。

R4-H保存同一来源清单、原抽样计划及其SHA。不要改 `R4J_MAX_ALTERNATIVES`、原源码、采样种子或现有cf_full manifest；不同配置将拒绝恢复。

### Step 3：Full完成后审计全部标签（不运行ALNS）

```bash
R4K_SOURCE_STUDY=r4j_train_early_s62s63_v1 \
STUDY_NAME=r4k_full_128audited_v1 \
  bash road_vrp7_step6c1_r4k/run_r4k.sh complete

cat runs/chicago_july_v1/step6c1_r4k/r4k_full_128audited_v1/report.md
```

本命令要求128/128计划场景快照均具备全部计划候选 × K50/200/500，同时校验R4-H完整源标签和SHAM。如果FULL未完成，**报错且不写出伪完整报告**。

## 四、统计边界与下一阶段决策

- 完整K500正收益覆盖是核心：按任务/seed/搜索阶段数不同的**粗粒度搜索起点**，而不是只数重复场景和动作。
- 验证C3后续追赶：对于同一动作，
  `gain(K2)-gain(K1) = progress_alt(K1→K2) - progress_C3(K1→K2)`。本包逐动作核对。
- 新种子增加的是**同4个Train任务的新随机轨迹**，不是新任务泛化证据。
- 若Full扩容增加了跨任务、seed独立的长K正收益，可以开展 R4-L：将seed60/61与62/63源标签分组，固定特征工程与门槛，以seed/任务为组做Train内部保守委托诊断。
- 若Full正例仍集中少数task/迭代阶段，应优先**真正新增独立Train任务**，而非继续在4个任务上增加模型复杂度。
- 任何后续FiLM/对比学习必须以道路交通特征不泄露ex-post Road.replay标签、实际推理耗时计入30秒预算、外部Validation（原已查看过的7月Validation不应称为盲测）为前提。

R4-K只读审核不代表源服务器端运行完成；真实Full结果以最后`complete`报告为准。
