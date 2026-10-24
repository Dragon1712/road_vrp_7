# Step6-C1-R4-E v1：K=100/200/500 精确续接与更长Section反事实

## 1. 为什么需要R4-E

截至GitHub提交 `a2a915d2a3cbd4aa2eb302f5dce6ee46631cfed1`：

- R4-A：8/8 原始C3与只读捕获在20,000固定迭代下终态完全一致；42个真实Section快照。
- R4-B：16个快照×K={1,5,20,50}，64/64逐步精确续接。
- R4-C：125个候选×4H=500条短K反事实标签；所有64个C3/SHAM检查通过。
- R4-D：125个替代动作有124个改变首步候选，仅48个被接受；K50仅6个改变历史最优，5个具有正收益。仅2/16交通情境快照、实际上1/8独立物理搜索状态存在正向已采样Oracle动作。

**本阶段不训练模型、不使用验证集标签参与学习、不声称30秒收益。** 已知7月Validation只用于方法机制诊断。必须在后续原始TRAIN任务上重新采集状态与标签，才能开展模型训练。

## 2. 设计

### 2.1 长K精确续接证明（verify_long.py）

先从源R4-A快照及R4-B 64/64证据校验起点；重新运行冻结的原官方ALNS `iterate`，提取每个快照后续前500步的原轨迹；另外从快照恢复同一随机数、RouletteWheel权重、SA温度及当前/最好解，使用R4-B原续接器运行500步。

每一步匹配：所选Destroy/Repair、候选路线SHA、接受Outcome、RNG、SA温度、算子权重及当前/最好解，逐K={100,200,500}给出精确证明。

所有后续K必须满足 `start_iteration+K-1 <= 20000`。原C3参考若因30秒墙钟提前停止，程序**报错而不是跳过**。若修复中出现截止时间依赖分支，也可能无法精确恢复，这时不可生成R4-E标签。

### 2.2 更长反事实（collect_long.py）

采集必须有已完成的对应快照/K长续接证明。对每个已验证快照执行C3、SHAM_C3、最多8个Section替代动作。使用**原R4-C的 `run_one`、原R4-B的 `restored_iterator` 和原冻结的v3.solve/Section/Repair/SA**。

每次首步先执行真正C3 Destroy，并消耗同样的首步RNG抽样；替代的只有首个Section Partial。后续轨迹从相同初始化状态继续。SHAM必须逐步与C3一致；同一动作的长K轨迹必须具有相同短K前缀；而且本阶段K>=100的前**50**步必须与原R4-C既有单元文件事件**完全一致**。

主指标：历史最好解 `road_best_sec`（事后 Road.replay）。另记录当前解过程收益、首步接受率、首次当前/历史最优路线出现差异的迭代步数、采样Oracle收益上界。

### 2.3 依赖约束

安装原项目环境，包括 `alns==7.0.0`、NumPy，Chicago路网/OD缓存及既有项目源码和结果。**不要修改或覆盖**R4-A/B/C和官方v3源文件。

数据角色：原Validation开发场景，仅机制验证。16个场景快照包括8个重复物理起点在两种交通情境下的事后评价；不能当独立样本。

## 3. 服务器命令

解压（ZIP应放仓库根目录）：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r4e_v1.zip
bash road_vrp7_step6c1_r4e/run_r4e.sh test
```

第一道门槛：一个快照的K100/200/500逐步精确续接Smoke：

```bash
STUDY_NAME=r4e_long_proof_smoke_v1 \
  bash road_vrp7_step6c1_r4e/run_r4e.sh verify-smoke
cat runs/chicago_july_v1/step6c1_r4e/r4e_long_proof_smoke_v1/report.md
```

确认3/3精确恢复，再进行一个快照最多2个Section的长K反事实Smoke：

```bash
LONG_PROOF_STUDY=r4e_long_proof_smoke_v1 \
STUDY_NAME=r4e_long_cf_smoke_v1 MAX_ALTERNATIVES=2 \
  bash road_vrp7_step6c1_r4e/run_r4e.sh collect-smoke
cat runs/chicago_july_v1/step6c1_r4e/r4e_long_cf_smoke_v1/report.md
```

通过后，在全部**16个R4-B已验证快照**上执行长K精确续接验证（不是扩大到R4-A的42个快照）：

```bash
STUDY_NAME=r4e_long_proof_16snaps_v1 \
  bash road_vrp7_step6c1_r4e/run_r4e.sh verify-pilot
cat runs/chicago_july_v1/step6c1_r4e/r4e_long_proof_16snaps_v1/report.md
```

要求48/48快照×K组合均通过。然后可进行长K反事实。**建议先限4个场景案例，减少服务器计算：**

```bash
LONG_PROOF_STUDY=r4e_long_proof_16snaps_v1 \
STUDY_NAME=r4e_long_cf_8snaps_v1 \
MAX_CASES=4 MAX_SNAPSHOTS_PER_RUN=2 MAX_ALTERNATIVES=8 \
  bash road_vrp7_step6c1_r4e/run_r4e.sh collect-pilot
```

如需对全部16个场景快照采集，使用独立新目录：

```bash
LONG_PROOF_STUDY=r4e_long_proof_16snaps_v1 \
STUDY_NAME=r4e_long_cf_16snaps_v1 \
MAX_CASES=0 MAX_SNAPSHOTS_PER_RUN=2 MAX_ALTERNATIVES=8 \
  bash road_vrp7_step6c1_r4e/run_r4e.sh collect-pilot
```

断点续跑时确保**所有配置完全相同**，增加`RESUME=1`。例如：

```bash
LONG_PROOF_STUDY=r4e_long_proof_16snaps_v1 \
STUDY_NAME=r4e_long_cf_8snaps_v1 \
MAX_CASES=4 MAX_SNAPSHOTS_PER_RUN=2 MAX_ALTERNATIVES=8 RESUME=1 \
  bash road_vrp7_step6c1_r4e/run_r4e.sh collect-pilot
```

默认输出：`runs/chicago_july_v1/step6c1_r4e/<STUDY_NAME>/`。各自独立的`report.json`、`report.md`、`manifest.json`；精确续接有`paired_rows.jsonl`，反事实有`labels.jsonl`、`branches.jsonl`、`snapshot_oracle.jsonl`以及`units/*.json`原始事件。所有结果在完成后再标记`COMPLETE`。

## 4. 实验决策

- 如果K100/200/500精确续接失败：不要做对应K反事实；检查具体哪一步及哪个事件字段首先分歧，尤其是deadline依赖。
- 如果SHAM或旧K50前缀不一致：停止，不能使用新标签。
- 如果更长K后历史最优变化明显多于K50：进一步研究首次改善时间和剩余搜索预算如何影响预测/委托。
- 如果历史最优仍高度稀疏：优先回到原始TRAIN任务增加独立的可恢复状态，选择多样化搜索阶段/任务（状态层面的采样），而不是堆网络或将Validation标签混入训练。
- 如果当前解经常变化而最好解长期不变：将当前解指标保留为过程辅助监督，主要评估仍使用最终最优解Road.replay。

## 5. 审计与局限

- 旧v3最大迭代20,000、真实30s搜索预算、相同求解器不变；恢复后固定K和30秒**剩余墙钟**不是同一问题。
- 续接的同一快照RNG只在干预**之前**精确匹配；干预会改变后续搜索状态与后续RNG消耗分布，不能将其视为不变的随机路径。
- 本项目中仅4个Validation任务×场景的开发子集被研究过，这里仅两个task，不能报告外部泛化。
- 官方ALNS 7.0.0和服务器真实Chicago数据不在此代码包中。本地单元测试不能替代服务器的严格Proof Smoke。
- 每个反事实是一条配对轨迹，没有置信区间；复用同一物理搜索起点的交通场景也不增加搜索起点独立样本数。

## R4-E v1.1 修复说明（R4-B字段兼容及Resume）

R4-B 的 `paired_rows.jsonl` 使用 `snapshot_stop_count`，而R4-A快照和R4-E自产出的记录使用 `stop_count`。R4-E v1读取R4-B证据时误用统一字段导致 `KeyError: 'stop_count'`。v1.1已在 `r4e_core.snapshot_id` 兼容这两种**确切来源格式**；若两个计数同时出现而不一致，会拒绝加载。另修复 `verify_long.py` 续跑时JSON列表与Python元组比较导致的误报。

在仓库根目录用 `unzip -o road_vrp7_step6c1_r4e_v1_1.zip` 更新 **R4-E文件夹**，不会覆盖冻结的R4-A/B/C。运行 `bash road_vrp7_step6c1_r4e/run_r4e.sh test` 后，用独立名称 `STUDY_NAME=r4e_long_proof_smoke_fix1 bash road_vrp7_step6c1_r4e/run_r4e.sh verify-smoke`。

注意：v1报错发生在创建新结果目录之前；若该目录未被创建，原名称可继续用。为了避免不完整输出混淆，建议总是使用新名称。修复后**还没有**真实Chicago官方ALNS长K的成功结果；只有真实服务器成功验证K100/200/500后才能采集新反事实标签。
