# Step6-C1-R4-C v1：冻结 ALNS 连续搜索的首步 Section 反事实实验

## 为什么是 R4-C

R4-B `r4b_exact_8cases_2snap_v1` 已在 GitHub `a8d2d947e03469fb675ca32d39e8bb3b367495e7` 验证 **16 个快照 × K={1,5,20,50} = 64/64 精确恢复**。但 R4-B 尚未做任何干预。

R4-C 固定 **同一个真实 Section pre-selection checkpoint**，只改变**第一个 Section Destroy 的具体子问题**，之后让官方 ALNS 7.0.0 的 C3 迭代逻辑、Repair、RouletteWheel、SimulatedAnnealing 继续执行 K 步。第 1 次选算子仍然由快照 RNG 与权重决定（即真实 d_idx=3，r_idx 由随机轮盘重现）。在所有干预分支里，**先真实执行原 C3 Selector.destroy 一次以消耗完整相同的 RNG Draw**，再替换它返回的 Section `Partial`。从第 2 步开始不干预，全部使用原 C3。

**此次只验证“固定 K 步的连续搜索 Section 决策代理效用”**。官方 `BothBudget` 的剩余墙钟未精确恢复，因此这**不是**30 秒在线收益或最终实时VRP性能的证明。所有道路 replay 在 K 步求解后离线执行，只用于标注。

## 三条实验臂

- `C3`：R4-B 完全相同的快照恢复 C3 基线，**不作任何补丁**。
- `SHAM_C3`：真实消费 C3 Destroy RNG 后，用独立的 `Block/Partial` 构建一个**语义完全相同**的 C3 动作。逐迭代 `events`、RNG、最优解和当前解、道路成本须与未打补丁 C3 精确相同。负对照若失败则**直接停止**。
- `Axxx`：从同一状态的静态 `SectionSelector.candidates` 经冻结 Step6 的 `select_actions` 确定性采样，取最多8个与真实C3不同的Section，替换第一个 Destroy 的 `Partial`，Repair与接受决策维持原算子。之后C3正常搜索K-1步。每次都重新恢复快照、重新构造冻结求解器，不继承其他干预臂的状态。

`Axxx` 是**动作位置标识**，而非预测成绩排名。候选不读取未来流量、道路事后标签、原 VQ 模型预测。候选数依实际样本决定，若8个样本含C3则替代数可能少于8。

## 结果标签及解释

主要标签：

`road_best_gain_sec = Road.replay(best_solution_of_C3_after_K) - Road.replay(best_solution_of_alt_after_K)`

正值意味着备选动作后K步**历史最优路线**的事后道路成本更低；负值意味着损失。另有 `road_current_gain_sec`（当前接受解）、`static_best_gain_sec`、`static_current_gain_sec`、相同路径比例、严重损失(`gain < -40s`)等。

相同初始RNG并不保证干预后所有随机数序列仍相同。首步 C3 Section 选择对RNG的消耗可配对；后续因接受结果、权重和路径分叉而形成的随机轨迹差异是策略效应的一部分，**不应该通过强制拷贝后续轨迹掩盖**。

## 运行环境

保持 `road_vrp7_step6c1_r4a`、`road_vrp7_step6c1_r4b`、`road_vrp7_step5b_v3`、`road_vrp7_stage3`、`road_vrp7_step6a`、`road_vrp7_step6b2_r` 等原目录**不改动**。已有官方 `alns==7.0.0` + NumPy + 原 Chicago 路网数据与 OD cache。GitHub 提交的 R4-A Capture 42个快照、R4-B 64/64审核记录存在服务器对应默认路径。

**最关键约束：** R4-B仅审计了每条轨迹的前2个Section快照。R4-C会从R4-B `paired_rows.jsonl` 逐项建立16快照白名单，拒绝第3个及以后的未经审计快照。

## 第一步：代码检查及预审

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r4c_v1.zip
bash road_vrp7_step6c1_r4c/run_r4c.sh test
bash road_vrp7_step6c1_r4c/run_r4c.sh preflight
```

Preflight必须通过冻结 v3/Section SHA、来源道路及任务的 SHA、完整 R4-A Capture SHA、R4-B **64/64** Report + 16组快照×K 的`paired_rows.jsonl`校验。

## 第二步：单快照 Smoke

```bash
STUDY_NAME=r4c_smoke_1snap_v1 \
HORIZONS=1,5,20 MAX_ALTERNATIVES=3 \
bash road_vrp7_step6c1_r4c/run_r4c.sh smoke

cat runs/chicago_july_v1/step6c1_r4c/r4c_smoke_1snap_v1/report.md
```

Smoke 用 **1案例×1个已验证快照×3个K×(C3+SHAM_C3+最多3个备选)**。`SHAM_C3` 必须与C3逐事件相同；相同动作的K步长序列必须保持前缀一致。

## 第三步：16快照Pilot

```bash
STUDY_NAME=r4c_pilot_16snaps_v1 \
HORIZONS=1,5,20,50 \
MAX_SNAPSHOTS_PER_RUN=2 MAX_ALTERNATIVES=8 \
bash road_vrp7_step6c1_r4c/run_r4c.sh pilot

cat runs/chicago_july_v1/step6c1_r4c/r4c_pilot_16snaps_v1/report.md
```

16个来源快照、4种K；最多每组2个控制臂+8个替代臂。每个 `快照×K×动作` 对应独立 `units/<hash>.json`，不需要在失败后重复所有单元。Pilot不修改真正30秒官方ALNS源文件。

## 中断续跑

可先执行小批量，例如：

```bash
STUDY_NAME=r4c_resume_demo_v1 MAX_NEW_UNITS=5 \
  bash road_vrp7_step6c1_r4c/run_r4c.sh smoke

STUDY_NAME=r4c_resume_demo_v1 RESUME=1 MAX_NEW_UNITS=0 \
  bash road_vrp7_step6c1_r4c/run_r4c.sh smoke
```

输出存在时**必须**显式 `RESUME=1`。复用前比对原配置、所有代码 SHA、R4-A Capture、R4-B Report和审计明细SHA、每个checkpoint内的动作计划与规格。所有新模型研究须使用新的目录。

## 输出

`runs/chicago_july_v1/step6c1_r4c/<STUDY_NAME>/`

- `manifest.json`：配置和上游 SHA、任务/快照白名单、数据角色。
- `units/*.json`：分支完整原始 `events` 与最优/当前静态和道路成本、回溯 RNG/权重/温度记录。
- `branch_summary.jsonl`：全分支的路线、成本及动作身份摘要。
- `counterfactual_labels.jsonl`：每个替代Section相对同起点真实C3的各项收益。
- `plans.jsonl`：每条快照的候选数、C3与替代动作数、抽样配置。
- `report.json/report.md`：分搜索长度的效用摘要、正负样本比例和严重损失。

## 实验决策标准

1. C3对SHAM_C3不满足K=1/5/20/50的**全轨迹一致**，则停止，不生成可信标签。
2. 同一动作不同K的前缀轨迹不一致，停止并检查时间截止分支或冻结源码。
3. 若大量K=1/5产生零Road收益，优先分析当前解收益和K=20/50区分度；不要把零收益当成标签错误。
4. 若备选样本负收益比例很高，说明C3自身局部选择可能难超越。需要先量化oracle上界、搜索方差与Section动作的分布，再考虑模型。
5. 当前快照来自**已使用过的July Validation开发场景**。这些标签只用于算法机制诊断，**不能**直接进入跨任务监督模型训练并把它们作为独立验证。后续R4-D如需真正训练，应重新在原Train采集/验证快照，并按任务和日期独立隔离。

## 已知界限

- 不保证剩余30秒墙钟预算一致；后续官方ALNS迭代次数固定，`stop.deadline=inf` 与R4-B一致。
- 反事实分支从真实预选Section快照恢复，但选择的备选Section并不必然被原C3在未干预轨迹中访问，因而反事实标签不能当“观察到的真实在线增益”。
- 结果取决于输入随机种子与初始状态；单次paired RNG不是模型置信区间。
- 冻结C3仍使用static planner，Road.replay事后成本是监督目标，不可作为在线预测特征。
- 本地单元测试使用模拟Section与RNG；未在本地提供Chicago道路和官方ALNS环境，**真实服务端Smoke必须完成**。
