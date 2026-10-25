# Step6-C1-R4-H v1 | TRAIN-only真实ALNS Section首步反事实价值采集

来源仓库：`Dragon1712/road_vrp_7`，提交 `1cdb19541b9d1c11211ef03930b93d2f5e945349`（2026-10-25）。

**本包不训练模型，不修改冻结v3求解器/ALNS 7.0.0/R4-A/B/C/G/Prefix，也不产生30秒在线收益结论。** 只为下一轮学习准备Train监督信号，所有可用状态必须通过现有R4-G-Prefix逐步精确续接证据。

## 为什么现在开发R4-H？

仓库`r4g_prefix_all16_v1`报告：全部16个Train场景-任务-种子搜索实例；原30秒采集共76个真实Section前选择快照、53个不同物理搜索状态；逐步精确恢复 **298/298**：K1=76/76、K50=76/76、K200=74/74、K500=72/72。缺少K200或K500原30秒搜索前缀的6个“状态×K”窗口被显式排除，绝不可转成正/零监督标签。

早期R4-F在**Validation开发集**的8个不同物理状态发现K200→500收益符号反转、强C3追赶。这只是方法机制假设，不是可用于训练的独立标签。

## 方法和防泄漏边界

1. `verified_allowlist`读取且验证**真实**R4-G-Prefix `report.json`、`manifest.json`、`paired_rows.jsonl`、`excluded_windows.json`与R4-G捕获SHA。前缀证明中status可以是已知的`CONFIGURED`，但报告必须`COMPLETE`且全部逐步exact。
2. 仅接受原Train `task_0000..0003`，源场景须已经在R4-G采集时被标记TRAIN、eligible，预训练环境/成本/Section SHA与当时相同。`verify_train_r4g.preflight`复核原始场景角色与全部源SHA。
3. 默认K=50、200、500；**只有三个K都已经通过精确恢复验证的起点才入选**（源76快照中理论上72个全K候选起点）。无“补齐K”的伪标签。`plan`阶段允许本地只读检查，不运行ALNS。
4. 对每个源状态调用原R4-C `shadow_action`计算真实C3首步Section、原有静态Section候选/抽样。对C3/SHAM_C3/最多8个替代Section分别运行原R4-C `run_one`，**首步原始C3随机数消耗严格保留**，后续ALNS/Repair/SA均不修改。
5. 分别在K=50/200/500固定迭代结束后对历史最优路线和当前路线做Post-hoc Road.replay。主标签：`RoadBest(C3)-RoadBest(Alternative)`，正数表示替代收益。Road.replay仅用于输出监督值，**不是在线决策特征**。
6. SHAM_C3必须与C3全事件、当前/最佳终点、Road成本完全一致；每个动作的不同K事件必须严格具有相同前缀；错误直接失败，不写完成报告。
7. 集合报告分别显示“快照数”“不同物理搜索状态数”“动作数”“正/负/零收益”和K50→200、K200→500收益反转及C3后续追赶；不将不同交通场景当作独立物理轨迹。
8. 每个分支独立`units/*.json`带证据来源SHA/动作plan；`RESUME=1`时先校验配置、代码SHA、plan、已保存事件，绝不默默追加到其他配置。`R4H_MAX_NEW_UNITS`可以按单元限额做可恢复的部分实验。

## 安装与测试

将ZIP放至仓库根目录后：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r4h_v1.zip
bash road_vrp7_step6c1_r4h/run_r4h.sh test
```

本地13项测试包括：真实R4-G-Prefix源**模式**下的SHA/身份校验、排除窗不得入选、非TRAIN拒绝、控制组收益公式、完整模拟分支采集、断点续跑及SHAM故意不等价必须停止。**不等于已在服务器真Chicago/ALNS环境完成验证**。

## 第一阶段：只读计划（不运行ALNS）

```bash
SOURCE_STUDY=r4g_train16_fix2 \
  bash road_vrp7_step6c1_r4h/run_r4h.sh plan
```

预期约72个具有全K证据的场景快照，但最终以本地SHA检查和报告为准。全部72个不是独立样本；物理状态数须重新统计。

## 第二阶段：单快照 + 2个Section候选 Smoke

```bash
STUDY_NAME=r4h_train_smoke_v1 \
  bash road_vrp7_step6c1_r4h/run_r4h.sh smoke
cat runs/chicago_july_v1/step6c1_r4h/r4h_train_smoke_v1/report.md
```

通过门槛：所有三个K下C3/SHAM完全一致，同一Section在长K上保留短K全事件前缀，所有标签带源精确验证身份，报告COMPLETE。

## 第三阶段：跨4个Train任务的8快照Pilot

```bash
STUDY_NAME=r4h_train_pilot_8states_v1 \
R4H_MAX_ALTERNATIVES=8 \
  bash road_vrp7_step6c1_r4h/run_r4h.sh pilot
```

默认任务round-robin，每个任务优先选择1个场景案例、每案例2个处于早/晚阶段的状态，共最多8个快照，覆盖4个Train任务；预计最多8×(2+8)×3=240个分支（实际候选可能少于8）。

**如需先测一部分并能断点重启：**

```bash
STUDY_NAME=r4h_train_pilot_chunked_v1 R4H_MAX_NEW_UNITS=15 \
  bash road_vrp7_step6c1_r4h/run_r4h.sh pilot

STUDY_NAME=r4h_train_pilot_chunked_v1 RESUME=1 R4H_MAX_NEW_UNITS=0 \
  bash road_vrp7_step6c1_r4h/run_r4h.sh pilot
```

每个输出目录必须独立，勿与前次study叠加。`MAX_NEW_UNITS=0`表示不设置新分支数量上限，并不代表0次。

## 第四阶段：完整Train采集（仅在Pilot全部通过后）

```bash
STUDY_NAME=r4h_train_full_v1 R4H_MAX_ALTERNATIVES=8 \
  bash road_vrp7_step6c1_r4h/run_r4h.sh full
```

理想情况下源72个全K起点，每个至多10臂×3K即2160个固定K分支。此实验可能明显耗时；用独立全量Study、必要时分块和断点续跑，不覆盖任何旧结果。**不承诺全量会有多少正收益样本。**

## 输出

`runs/chicago_july_v1/step6c1_r4h/<STUDY_NAME>/`：

- `manifest.json`：固定求解器/R4C/精确前缀证明/Train快照来源SHA、实验配置及全部已选状态；
- `units/*.json`：每个状态×K×臂逐事件结果及后验Road评价，可独立检查SHAM与跨K一致性；
- `report.json`与`report.md`：正负零收益、不同物理状态、采样Oracle正状态数、多时域收益符号与C3追赶；
- `labels.jsonl`：可用于**将来Train内部**监督学习的反事实动作标签，需进一步按物理状态分组；
- `branch_summary.jsonl`、`plans.jsonl`：来源和基线反事实比较审计；
- `partial.json`（仅部分实验）：未完成时下一单位及来源记录，`COMPLETE`后删除。

## 下一阶段决策（不是本包声称已完成的结果）

- 若Train的多时域正/负标签也高度稀疏，优先增加**Train不同物理搜索状态**、不同种子和决策阶段，而不是立即训练FiLM。
- 若Train有充足且稳定的监督信号，先训练低成本静态效用/保守回退/多时域轻量选择器，使用**未用于拟合的Validation**做严格离线和30秒在线消融。
- 多模态和对比学习作为后续独立F0～F5消融，不把已看过的Validation机制分析样本加入Train。

**解释限制：** 固定K反事实不是30秒在线算法收益；所评估的至多8个动作不等于全动作空间；一次固定RNG轨迹不是收益分布置信区间。
