# Step6-B0 + Step6-A v2：下一阶段实验代码（2026-10-19）

**适用仓库**：`https://github.com/Dragon1712/road_vrp_7`，代码审计时 `main=51f1c8a3d239b2666c7e91dff5e94aa651615613`。请在已有代码之上叠加本包两个新目录；绝不覆盖 `road_vrp7_step5b_v3/` 和 `road_vrp7_step6a/`。

背景：原 Step6-A v1 完成 948 个历史最优快照状态、7584 个合法候选，仅有 104 个正改善动作（1.37%）。其中 Oracle 与静态 Top-1 不一致比例 29.3%，但允许保留原解后真正有额外正收益的状态只有 8/948。必须将**可预测性**与**可实现增益**分开评估。原始标签仅是历史 Oracle TD 代理目标，不是独立的真实道路级优化结果。

## A. Step6-B0：先运行，无需重跑求解器

使用冻结的 `features_X.npz`、`labels_y.npz`、`groups.csv`，运行两种 RandomForestClassifier：

- **B1**：只有静态路线/Section特征；
- **B2**：B1 加上真正出发前历史交通与观测掩码统计；
- 附带不学习的 Section 静态评分、总是 no-op 两个参照。

分类标签：`td_gain_sec>1e-7`。分别执行按 **任务** 和按 **日期** 的分组 train/validation/internal-test 协议（两套独立对照，不能混为一谈）。训练时不加载完整 `state_records.jsonl`，也不使用 TD 标签作为输入。执行阈值只从 validation 选择，test 不参与调参：对每个冻结状态，找概率最高的候选，若概率达阈值执行，否则原样保留，**真实测试指标累加有符号改善值，不允许利用Oracle结果在测试时后验夹取到零**。

安装依赖（推荐沿用当前已存在的独立Step6虚拟环境）：

```bash
cd ~/code/WZL/road_vrp_7
python -m pip install 'scikit-learn>=1.2,<2' numpy
python road_vrp7_step6b0/train_baselines.py --help
```

运行：

```bash
STUDY_NAME=step6b0_static_traffic_v1 \
  bash road_vrp7_step6b0/run_b0.sh
```

结果：`runs/chicago_july_v1/step6b0/step6b0_static_traffic_v1/{config.json,results.json}`。记录两种分组下B1/B2的 PR-AUC、ROC-AUC、组内可执行有符号收益、Oracle no-op 上界、选择阈值、负收益执行次数。若内部测试某折无正样本，PR-AUC/ROC-AUC返回null而不是胡编数值。

**注意**：目前仅10任务、4个早已查看过的7月日期，这是开发集内部诊断而非泛化结论。当前训练样本来自Oracle最优快照，模型测试并不能证明部署时可用。

## B. Step6-A v2：采集真正已接受的当前搜索状态

设计与v1相比的改变：

- Source由 `C3_topk` 的 **static** 规划器生成（不使用Oracle生成源路线），30秒搜索、5000退火、20,000迭代上限；
- 用冻结的 v3 `solve()` 函数 **内存中精确两点插桩**，新回调仅捕获**accepted + changed**的实际候选，不额外执行任何候选Repair；
- 每5秒时间档，优先捕获一个比进入接受判据时的历史最优解更差的已接受状态；所有原始候选仍由官方ALNS及原求解器决定；
- 搜索结束后，使用原Step6-A `section_dataset.py` 构建完全相同的静态、Oracle TD反事实标签，静态/历史流量特征仍按原白名单导出；
- 相同任务+场景+路线的样本**全局去重**，再按任务/场景均衡抽样；每个候选静态与TD修复执行同样的Top-K；
- v1源码/标签/分析代码**不修改**，仅运行时替换来源加载函数，v2输出独立版本标志 `step6a-offline-v2`；
- 记录源数据SHA、v3源码SHA、内存插桩版本SHA、标签器SHA和所有结果配置；已完成结果禁止覆盖，`RESUME=1`要求配置完全一致。

### B1. 合成单元测试（无需路网张量）

```bash
bash road_vrp7_step6a_v2/run_v2.sh test
```

### B2. 真实Smoke（必须在服务器验证）

```bash
bash road_vrp7_step6a_v2/run_v2.sh smoke
```

Smoke 会采集1任务×1场景×1种子×3秒当前状态，再按对应1、2、3秒窗口进行离线标注（4个Section/状态）。需要 `alns==7.0.0` 和已有Road/OD/Stage3任务数据。

### B3. 小样本Pilot：8条搜索轨迹

```bash
export STUDY_NAME=current_static_pilot_v1
export MAX_TASKS=2 MAX_SCENARIOS=2 SEEDS=42,43
export WALL_SEC=30 INTERVAL_SEC=5
export MAX_STATES=100 ACTIONS_PER_STATE=8
bash road_vrp7_step6a_v2/run_v2.sh plan
bash road_vrp7_step6a_v2/run_v2.sh capture
bash road_vrp7_step6a_v2/run_v2.sh label
```

不同阶段结果目录：

```
runs/chicago_july_v1/step6a_v2/
├── current_source_current_static_pilot_v1/
│   ├── run_config.json
│   ├── runs/*.json             # 每条搜索轨迹独立安全断点
│   ├── current_states.jsonl    # 采集完成后统一生成
│   └── benchmark_summary.json
└── labeled_current_static_pilot_v1/
    ├── run_config.json
    ├── state_records.jsonl
    ├── benchmark_summary.json
    └── analysis/
        ├── summary.json
        ├── state_ranking_diagnostics.csv
        ├── features_X.npz
        ├── labels_y.npz
        ├── groups.csv
        └── current_state_diagnostics.json
```

### B4. 扩大采集（仅在Pilot可行后）

建议10任务×8交通场景×2种子=**160条30秒搜索轨迹**，每轨最多6个时间桶，**理论最多960条原始当前状态**，去重后会减少。每状态8～12候选，实际总数由日志报告，不人为凑数。来源搜索名义耗时约80分钟，另加OD预计算和离线Section反事实评价时间。

```bash
export STUDY_NAME=current_static_full_v1
export MAX_TASKS=10 MAX_SCENARIOS=8 SEEDS=42,43
export WALL_SEC=30 INTERVAL_SEC=5
export MAX_STATES=1000 ACTIONS_PER_STATE=8
bash road_vrp7_step6a_v2/run_v2.sh plan
bash road_vrp7_step6a_v2/run_v2.sh capture
bash road_vrp7_step6a_v2/run_v2.sh label
```

断点中断时，在同一参数配置下：

```bash
RESUME=1 bash road_vrp7_step6a_v2/run_v2.sh capture
RESUME=1 bash road_vrp7_step6a_v2/run_v2.sh label
```

**注意**：如果第一次`label`已经完成，直接分析用：

```bash
bash road_vrp7_step6a_v2/run_v2.sh analyze
```

### B5. 再跑B0对照（只建议完整A-v2数据集后）

```bash
SOURCE_DIR=runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1 \
STUDY_NAME=step6b0_current_static_traffic_v1 \
  bash road_vrp7_step6b0/run_b0.sh
```

B0完整任务分组需要足够任务与日期；2任务的v2 Pilot不适合做完整的task/date三折比较，应直接分析正收益和no-op指标，不要强制训练。

## C. 实验验收和下一阶段门槛

首先比较 v1 BEST-SNAPSHOT 与 v2 CURRENT-ACCEPTED 两类采样的：

1. 状态中是否包含至少一个正收益动作的比例，而不仅是动作级正收益率；
2. 真实 `source_planner=static` 的独立性、捕获的非历史最优比例、候选有效率、数据来源哈希；
3. Static/TD一致性、排名分歧，**以及允许no-op后真正可实现的决策Regret**；
4. B1静态与B2静态+历史交通在不同分组测试中的PR-AUC和**可执行有符号收益**。只有B2在未见任务/日期比B1稳定改善，才支持进一步引入SPIN与SPCLT。

不要用模型测试中的未来Oracle收益决定是否执行：阈值必须在验证集确定。暂不宣称多模态/对比学习产生了效果；本包仅提供Step6-B0简单监督和Step6-A v2数据来源修正。

## D. 真实性测试边界

本地已执行合成数据 `B0`训练/评估与 A-v2 回调/来源解析的回归，见 `test_log.txt`；本环境**没有安装官方ALNS 7.0.0，也没有完整芝加哥道路张量**，无法完成实际Smoke。服务器 `smoke` 成功之前，不要直接启动160条正式实验。如果v3求解器SHA与冻结1200条源实验不一致，采集器会主动拒绝，应该人工审计新版本后使用独立实验代码而不是关闭SHA检查。
