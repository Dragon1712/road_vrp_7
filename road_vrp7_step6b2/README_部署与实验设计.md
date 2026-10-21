# Step6-B2 v1 — 真实道路移除/插入边界特征 + F0–F5 + 30秒ALNS

**研究状态：实现、合成数据训练和适配器单元测试已通过；未在真实芝加哥Cost/Task数据上完成服务器集成测试或得到新的道路指标。** 本模块建立在仓库 `Dragon1712/road_vrp_7` 的 **Step6-A v2 + Step6-B1 v2 + Step5-B v3** 上，不修改任何原有源码；过去Step5/Step6实验不会被重跑或覆盖。

## 重要工程协议

1. 冻结输入：`runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1`、`road_vrp7_step5b_v3/section_v2.py` 和 `road_vrp7_stage3/road_vrp.py`，强制检查源代码SHA、路网SHA、Schema和Label数据SHA。
2. 训练标签仍是 **`static_repair_td_gain_sec`**：Static Repair产生候选路线，由历史TD代理规划目标离线事后打分。**不是实测路网Ground Truth**，不允许作为输入或在线决策使用。
3. 所有新交通特征来自 **出发时间严格之前的RAW道路旅行时间**，使用静态最短路径映射和真实观测mask。缺失观测保持mask，并在成本估计中静态回退，不使用未来FIFO后处理、同一天未来的路段速度或Oracle OD。
4. 插入位置：从原Section移除后，通过冻结的 `_block_options` 在所有合法非原样**Relocate**方案中，选择**最小静态代理增量**的候选作为「插入边界探针」。不使用任何Oracle-TD Repair的未来插入位置，也**不假装是最终Static Repair所选位置**（后者可能选Swap）。探索方向和可行性由原Section算子决定。
5. 30秒Online采用原始 `v3.solve('C3_topk', ...)` 和官方ALNS 7.0.0。只在一次求解期间临时替换`Selector.destroy`，其他三种Destroy、Repair、Selector权重、SA、`BothBudget`保持原样；查询结束在 `finally` 恢复原类函数和计时器。
6. 30秒搜索内计入生成Section、移除/插入代理、历史特征提取、模型推理和Repair；OD、历史矩阵载入、预计算**放在搜索预算之外**并单独记录。因此最终报告必须同时展示30秒内指标与额外前处理成本。每次道路回测在搜索后执行，不能影响搜索。
7. **输入分布不一样**：原C3从静态前20%随机挑选Section；Step6-B2 learned使用与原Step6-A相同三分位分层抽样8个Section，再用模型评分。除了C3，提供`sampled_random`单独基线检验抽样效果。切勿把C3 vs B2解释成“只增加模型”的纯净消融。

## 一、文件

- `features.py`：静态移除/插入边界、历史交通代理、可靠性掩码、在线12维/旧15维兼容特征。
- `extract.py`：从**原Step6-A v2冻结原始状态和候选ID**新增边界特征，写独立NPZ，支持原状态粒度的JSONL断点恢复。
- `train_b2.py`：P2（RF有符号收益回归）、P4（RF两阶段收益）×F0–F5，严格按任务/日期划分；阈值仅验证集确定。保存所有模型joblib和逐状态、逐Section结果。
- `analyze_ablation.py`：原状态配对分析，含**决策差异率、收益差异、按任务划分**。
- `benchmark_30s.py`：正式集成之前的静态Planner、30秒C3在线实验启动器。
- `analyze_30s.py`：同任务×场景×种子的道路成本配对对比，严格校验输入/求解器哈希与预算。
- `run_b2.sh` + `tests/test_b2.py`：运行入口和合成数据回归测试。

## 二、特征实验矩阵

| 实验 | 维度 | 特征 | 真正验证什么 |
|---|---:|---|---|
| F0 | 12 | 原结构+时间 | 冻结P2/P4无历史交通基线 |
| F1 | 27 | F0 + 老15维Section自身历史交通 | 复现Step6-B1的交通输入 |
| F2 | 26 | F0 + 14维静态移除/插入位置（含反向内部边代价） | 控制更多几何与插入信息本身的作用 |
| F3 | 36 | F2 + 10维静态边界的历史旅行成本/变化 | **F3 vs F2识别交通边际贡献** |
| F4 | 44 | F3 + 8维历史覆盖、真实观测/缺失mask | F4 vs F3识别可靠性信息作用 |
| F5 | 44 | F4，但分组内部从相同任务的其他日期借用交通+mask | **负对照；仅任务划分可做** |

注意F5在`date`协议的val/test中只有单一日期，数学上无法做“其他日期”置换，因此实现**拒绝伪造置换并明确跳过date/F5**；不是缺失实验数据也不是跑错命令。F5用同任务、不同日期且候选数量一致的状态配对，保持静态特征不变。所有train/val/test互不交换交通数据。

静态代理`proxy_static_net_gain_sec`包括Section翻转时内部定向弧的差异；历史交通对应的`proxy_history_net_gain_sec`同样保留内部弧影响。`source_arrival_static_sec`、`target_arrival_static_sec`是静态估计时间，不是未来Oracle到达时间。

## 三、安装 & 快速测试

上传ZIP至仓库根目录，然后：

```bash
cd ~/code/WZL/road_vrp_7
conda activate WZLroadvrp
unzip road_vrp7_step6b2_v1.zip
python -m pip install -r road_vrp7_step6b2/requirements.txt
bash road_vrp7_step6b2/run_b2.sh test
```

`tests` 本地已通过8项，真实服务器首次应再次运行，并确认Step6-A v2源数据和历史路网Costs已经存在。**不要以`base`环境替代原来锁定的ALNS依赖。**

## 四、先测试真实数据特征导出（独立小样本）

```bash
STUDY_NAME=b2_features_pilot5_v1 MAX_STATES=5 \
  bash road_vrp7_step6b2/run_b2.sh pilot
```

路径：`runs/chicago_july_v1/step6b2/b2_features_pilot5_v1/`；标为`PILOT_ONLY`，不能用于正式训练。目的是及时捕捉上游道路成本路径、Core在线/离线数值和插入几何协议不一致。

## 五、全量3776条特征导出（只做一次，支持断点续跑）

```bash
FEATURES_DIR=runs/chicago_july_v1/step6b2/features_boundary_v1 \
  bash road_vrp7_step6b2/run_b2.sh extract \
  2>&1 | tee b2_extract.log
```

中断后保持相同代码/配置：

```bash
FEATURES_DIR=runs/chicago_july_v1/step6b2/features_boundary_v1 \
  EXTRA_ARGS=--resume \
  bash road_vrp7_step6b2/run_b2.sh extract
```

完成状态：`summary.json.status=COMPLETE`、`boundary_X.npz`以及`features_checkpoint.jsonl`齐全；不会重新评价任何TD收益标签，不接触原B1模型。

## 六、F0–F5训练（先少量树Smoke，再完整种子）

```bash
STUDY_NAME=b2_smoke_seed42_v1 SEED=42 TREES=10 PROTOCOL=task \
  METHODS=P2,P4 MODES=F0,F2,F3,F4,F5 \
  bash road_vrp7_step6b2/run_b2.sh train

STUDY_NAME=b2_full_seed42_v1 SEED=42 TREES=120 \
  bash road_vrp7_step6b2/run_b2.sh train \
  2>&1 | tee b2_full_seed42.log
```

多种子：

```bash
for seed in 43 44 45 46 47; do
    STUDY_NAME="b2_full_seed${seed}_v1" SEED="$seed" TREES=120 \
      bash road_vrp7_step6b2/run_b2.sh train
 done
```

**已有目录一律不可覆盖。** 修改`STUDY_NAME`。正式训练结果在：

```
runs/chicago_july_v1/step6b2/b2_full_seed42_v1/
  manifest.json
  task/report.json
  task/P2_F3_model.joblib
  task/P2_F3_test_states.csv
  task/P2_F3_test_candidates.csv
  date/report.json    # F0-F4，日期分组中F5不适用
  summary.json
```

比较：

```bash
python road_vrp7_step6b2/analyze_ablation.py \
  --study runs/chicago_july_v1/step6b2/b2_full_seed42_v1 \
  --output runs/chicago_july_v1/step6b2/b2_full_seed42_paired_v1.json
```

**判据**：F3 vs F2、F4 vs F3的平均收益、按任务配对差、`different_decisions`、负收益执行率，以及F4 vs F5负对照，不以纯PR-AUC决定是否入ALNS。仅六个训练种子、两种任务测试和一天日期测试，不声称独立统计显著性。

## 七、P2/P4 × 固定30秒C3在线试验

以下 **先使用任务协议** 的B2 seed42 模型，并只挑任务`task_0008,task_0009`上验证split前两个合格场景做**开发级小样本**。道路级和模型输入属于已参与研究的7月数据，不是盲测。**先成功运行C3和抽样随机两个对照，再跑模型。**

先冻结对照：

```bash
COMMON="--split validation --task-ids task_0008,task_0009 --max-scenarios 2 --seeds 42,43 --wall-sec 30 --max-iterations 20000 --anneal-iterations 5000 --top-k 12"

python road_vrp7_step6b2/benchmark_30s.py \
  --control C3 \
  --output-dir runs/chicago_july_v1/step6b2/online30s_C3_pilot_v1 \
  $COMMON

python road_vrp7_step6b2/benchmark_30s.py \
  --control sampled_random \
  --output-dir runs/chicago_july_v1/step6b2/online30s_random8_pilot_v1 \
  $COMMON
```

P2-F0（静态强基线，必跑）：

```bash
python road_vrp7_step6b2/benchmark_30s.py \
  --control learned --model-method P2 --model-mode F0 --model-protocol task \
  --model-path runs/chicago_july_v1/step6b2/b2_full_seed42_v1/task/P2_F0_model.joblib \
  --output-dir runs/chicago_july_v1/step6b2/online30s_P2_F0_pilot_v1 \
  $COMMON
```

P4-F0（另一个静态强基线，必跑）：

```bash
python road_vrp7_step6b2/benchmark_30s.py \
  --control learned --model-method P4 --model-mode F0 --model-protocol task \
  --model-path runs/chicago_july_v1/step6b2/b2_full_seed42_v1/task/P4_F0_model.joblib \
  --output-dir runs/chicago_july_v1/step6b2/online30s_P4_F0_pilot_v1 \
  $COMMON
```

只有F3相较F2出现稳健交通增益后，才优先跑P2-F3/P4-F3；若F4再提升，可改成P2-F4/P4-F4。示例：

```bash
python road_vrp7_step6b2/benchmark_30s.py \
  --control learned --model-method P2 --model-mode F3 --model-protocol task \
  --model-path runs/chicago_july_v1/step6b2/b2_full_seed42_v1/task/P2_F3_model.joblib \
  --output-dir runs/chicago_july_v1/step6b2/online30s_P2_F3_pilot_v1 \
  $COMMON
```

每个结果包括`manifest.json`、`metrics.csv`、`solutions.json`、`summary.json`。`metrics.csv`包含道路`ex_post_road_travel_sec`、后验总车辆时长、真实观测覆盖、30秒搜索耗时、实际迭代数、模型推理/特征耗时、no-op次数。`precompute_wall_sec`单独计每个任务场景OD处理，`manifest.history_setup_sec`计历史矩阵加载初始化。**原ALNS不变，权重选择由原RouletteWheel控制；并未插入额外的Oracle修复。**

道路级配对比较：

```bash
python road_vrp7_step6b2/analyze_30s.py \
  --baseline runs/chicago_july_v1/step6b2/online30s_C3_pilot_v1 \
  --treatment runs/chicago_july_v1/step6b2/online30s_P2_F0_pilot_v1 \
  --output runs/chicago_july_v1/step6b2/paired30s_P2F0_vs_C3_pilot_v1.json
```

根据Pilot观察，逐步扩展到**同一批 ≥10个配送任务、跨日期/高低覆盖场景、各≥5随机种子**，预算分别测5s/15s/30s时应使用新目录；只有所有方法30秒预算一致，才可以做最终道路级结论。

## 八、重要限制

- 本代码在本地测试了合成路线的特征方向、F5严格分组、P2/P4所有F模型训练导出与joblib恢复、可逆ALNS Section Hook；**本地没有GitHub上的完整`Road/OD`数据，因此真实 `extract → train → online` 仍需在你的服务器执行。**
- 在线仅用**静态Planner接受和更新最佳解**，TD Planner的未来数据完全不进入求解器。即使离线标签预测收益为正，静态ALNS的最优解和真实道路回测仍可能变差；这正是实际必须测试的结论。
- F2-F4边界探针遍历所有可行Relocate候选，在线可能吞掉较多30秒预算。必须以`section_feature_wall_sec`、`iterations`判断是否值得继续；若开销过大，下一版单独测试静态边界缓存/候选剪枝，不允许不透明地将推理时间移出预算。
- 禁止将`joblib.load`用于不可信第三方模型文件；这里只加载服务器本地训练得到、配置与输入SHA完整一致的模型。
- 同一数据集多个随机训练seed不能作为独立样本做盲测置信区间；日期分组存在重复道路路线拓扑。
- F5在线不可用，并会显式拒绝；F3/F4即使提高**离线单步收益**，还需统一30秒条件下实现**Road.replay**改进才能支持后续多模态+对比学习主张。
