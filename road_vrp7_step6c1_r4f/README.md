# Step6-C1-R4-F v1 — 长K收益反转归因 + 完整16快照复核

## 背景与证据

基于已完成的 R4-E GitHub 提交 `5593b4de16c3c6e465be06a4ca0d75646c9f604d`：

- R4-E 原C3精准续接 K=100/200/500：**48/48**；16个快照。
- R4-E 已收集反事实：**8个场景快照（仅4个独立物理起点）**，63个替代动作×3个K。
- K=100：12/1/50正/负/零收益，均值 +15.675 秒；K=200：17/0/46，均值 +29.794 秒；K=500：2/17/44，均值 −22.778 秒。
- 同一63动作中 **15个从 K200 正收益变 K500 负收益**。该结论仅来自现有开发样本，不能独立泛化。
- K500相对K200，原C3参考轨迹后续改善是重要来源，而非简单的“Road.replay统计错误”：静态收益也反转。

**研究问题：** 是真正的长期收益反转、原C3逐步追赶，还是局部样本（task_0008）导致的假象？

## 设计

本包只新增独立的分析和复用脚本，**不修改任何 ALNS、Section、R4-A/B/C/E 文件**。

1. `audit-preview`：直接审计已有8场景快照的R4-E报告、R4-C K50结果，合并同一候选在K=50/100/200/500的收益；检查源SHA、C3/SHAM、道路收益代数等式，生成完整转移矩阵、每个物理状态的敏感性。
2. `init-full`：在**独立新目录**中启动原始R4-E完整16快照采集，先运行1个unit，写入目标manifest与checkpoint。
3. `reuse`：核对source/target完整来源及算子代码SHA，要求新16快照严格包含旧8快照，安全复制已有的237个unit（不修改旧结果）。任何不一致则失败。
4. `collect-full`：交由**原R4-E代码**再次验证每个checkpoint plan、K50前缀、SHAM一致性、跨K轨迹前缀，并补跑余下的task_0009场景快照。
5. `audit-full`：在16场景快照、8物理起点的完整集合上做同样的收益反转归因；强制要求样本数量满足预期。

### 收益分解

采用Road.replay车辆道路旅行总秒数，越小越好。对同一动作：

`gain(K) = RoadBest_C3(K) - RoadBest_Alternative(K)`。

令 `progress_C3(a→b) = RoadBest_C3(a) - RoadBest_C3(b)`，以及替代分支的同名进步量：

`gain(b) - gain(a) = progress_Alternative(a→b) - progress_C3(a→b)`。

代码要求该等式逐动作成立。该 RoadBest 是静态搜索中所维护 best route 的**事后道路评价**，不保证随K单调下降。不要把它当成实际30秒在线目标。

还计算：

- K50→100、K100→200、K200→500，同一动作 `+→-` 等收益转移；
- 反转时C3与替代分支分别改善了多少Road成本；
- 相同物理搜索起点的场景嵌套、各物理状态均值与留一敏感性，**不把多个K/动作/场景视为IID样本**；
- 已采样动作的事后Oracle收益（不能当部署策略）；
- K200事后最佳动作到K500的收益（hindsight horizon-overfit stress test，**不是可部署模型指标**）。

## 环境

在现有 `/home/xys/code/WZL/road_vrp_7` 根目录执行，和R4-E使用相同的Python环境。没有新增Python第三方依赖；正式补采阶段仍需要现有ALNS 7.0.0、Chicago道路与任务文件、原R4-E/R4-C。

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r4f_v1.zip
bash road_vrp7_step6c1_r4f/run_r4f.sh test
```

### 第1步：立刻审计现有8场景快照（无需新ALNS调用）

```bash
STUDY_NAME=r4f_preview_8snaps_v1 \
  bash road_vrp7_step6c1_r4f/run_r4f.sh audit-preview
cat runs/chicago_july_v1/step6c1_r4f/r4f_preview_8snaps_v1/report.md
```

原8快照应重现K200→500 **15个正转负动作**；若不一致，停止并核查SHA/字段。

### 第2步：初始化完整16场景快照的独立新采集

```bash
FULL_STUDY=r4f_long_cf_16snaps_v1 \
  bash road_vrp7_step6c1_r4f/run_r4f.sh init-full
```

`init-full`预计会打印`R4E collection capped`，**这是预期的部分采集断点，不是错误**。目标结果的manifest应为RUNNING。

### 第3步：安全复用原已验证的237个units

```bash
PREVIEW_STUDY=r4e_long_cf_8snaps_v1 \
FULL_STUDY=r4f_long_cf_16snaps_v1 \
  bash road_vrp7_step6c1_r4f/run_r4f.sh reuse
```

要求source为COMPLETE、target为RUNNING、选定快照严格前缀、代码SHA/道路/证明/采样器等配置一致。它不会绕过R4-E自己的checkpoint复验。

### 第4步：由冻结的R4-E采集器补跑其余快照

```bash
FULL_STUDY=r4f_long_cf_16snaps_v1 \
  bash road_vrp7_step6c1_r4f/run_r4f.sh collect-full
```

如果执行中断，**直接重复相同的`collect-full`命令**，自动RESUME并跳过有效单位。不要把结果追加到先前的8快照目录。

### 第5步：完整16快照、8个物理起点审计

```bash
FULL_STUDY=r4f_long_cf_16snaps_v1 \
STUDY_NAME=r4f_reversal_16snaps_v1 \
  bash road_vrp7_step6c1_r4f/run_r4f.sh audit-full
cat runs/chicago_july_v1/step6c1_r4f/r4f_reversal_16snaps_v1/report.md
```

## 输出

`runs/chicago_july_v1/step6c1_r4f/<STUDY_NAME>/`：

- `report.md`：四个K的收益、收益翻转/C3追赶比较和统计局限；
- `report.json`：逐物理起点的宏平均与敏感性、K200事后选最优后K500损失；
- `horizon_transition_rows.jsonl`：每个动作在K50→100、100→200、200→500的变化与C3/替代改善分解；
- `snapshot_oracle.jsonl`：每场景快照已采样动作的收益上界；
- `provenance.json`：源文件SHA及输出哈希。

## 后续科研决策

- 如果 task_0009也出现K200→K500的收益反转：首要问题是**视野敏感性与C3追赶**，优先研究多时域/稳健价值，不应拿K200单独做监督标签。
- 如果反转只集中在task_0008：当前样本不足以证明普遍性，优先在原TRAIN任务中采集**不同物理状态**，避免用反复评测过的Validation训练。
- 不论哪种，当前数据都**不用于训练或宣称Blind Test改善**；真实30秒墙钟预算仍需单独验证。

## 审计边界

- 当前原Train中4个任务的状态采样与R4A/B/C在Validation使用的4条独立路线不能混为一谈。
- 这里的"不同物理起点"依据任务、随机种子、迭代、当前/最优路线SHA和RNG状态去重；不同交通场景的道路成本可异，但ALNS静态搜索相同。
- `MAX_NEW_UNITS`和wall时间只影响计算完成速度，不构成在线政策。
- 安全复用依赖源目标配置及文件身份严格一致，若已有实验配置不同将直接拒绝，而不会自动改写过去结果。
