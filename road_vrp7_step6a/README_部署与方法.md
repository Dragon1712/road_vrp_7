# Step6-A v1：Section优化收益数据集（芝加哥真实路网TD-CVRP30）

> 本目录为独立模块，不复制或修改官方ALNS、Step5-B v3、Step5-C或任何旧实验结果。**尚未在服务器完整芝加哥数据上跑过**；先运行测试和Smoke。

## 0. 研究目标和范围

1. 从已完成的`step5b_v3/full30s_1200_v1`的**历史最优解快照**建立冻结可行路线状态，作为Step6-A第一轮开发样本。**不是**ALNS随机时刻的当前解采样，不应以此冒充完整状态分布。
2. 每个冻结状态**只用静态Section评分**从合法Section候选池中分层选择4～8个动作；选择过程与Oracle未来交通无关。
3. 对同一状态、同一Section、同一Top-K修复候选池，分别使用**时变Oracle**及**静态成本**选择修复结果，然后交叉评价，获得收益和反事实标签，保留负收益，不进行截断。
4. 由`edge_costs.npz['travel_sec_raw']`、`chicago_observed_mask`和静态路径，提取**严格早于规划出发时间**的多时段真实历史观测统计，避免从`travel_sec_fifo`的可能未来平滑中提取可部署特征。
5. 计算静态/TD候选收益的Spearman相关、静态最佳与TD最佳候选是否一致、Oracle候选上限的选择后悔值，按历史观测支持度分析。诊断Oracle价值不等于证明历史交通可预测它。
6. 导出严格白名单的`features_X.npz`、`labels_y.npz`和隔离的分组元数据`groups.csv`，服务于下一步Step6-B监督学习。

## 1. 与论文代码的关系

- **Learning to Delegate (NeurIPS 2021)**：借鉴“冻结当前解→枚举多个局部子问题→独立评价各候选→监督标签”的实验组织；没有复制其求解器或静态CVRP的LKH距离函数，也没有直接使用其Transformer权重。
- **DRHG**：保留连续片段的整体性、路线/位置/片段结构，可在后续编码器复用其概念；没有引入DRHG欧氏路线成本。
- **SPIN/STMAE/SPCLT**：为后续Step6-B/C准备历史交通时间窗口与严格观测掩码；本v1没有把GNN或对比学习训练伪装成已完成。

核心代码直接导入仓库已有：

- `road_vrp7_step5b_v3/section_v2.py`的`Selector.candidates`、`Block`、`Partial`、`repair`
- `road_vrp7_stage3/road_vrp.py`的`Road`、`OD`、`Objective`、`legal`
- 原Step5-B v3的`run_config.json`和`snapshot_records.jsonl`

## 2. 安装

把压缩包上传到`road_vrp_7`仓库根目录并解压：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6a_v1.zip
python -m pip install numpy pandas
bash road_vrp7_step6a/run_step6a.sh test
```

真实整合测试需能导入仓库`road_vrp7_stage3/`和`road_vrp7_step5b_v3/`。无需额外安装图神经网络依赖；不调用ALNS迭代器。

`tests/test_step6a.py`的真实Section合成集成测试仅会在项目目录下执行。离开项目的本地基础测试会跳过它。

## 3. 建议执行顺序

首先列出抽样计划（不会加载道路张量或创建实验目录）：

```bash
STUDY_NAME=section_utility_pilot_v1 \
MAX_STATES=1000 MAX_TASKS=10 MAX_SCENARIOS=8 ACTIONS_PER_STATE=8 \
bash road_vrp7_step6a/run_step6a.sh plan
```

再单独跑3个状态 × 4动作Smoke：

```bash
bash road_vrp7_step6a/run_step6a.sh smoke
```

若Smoke通过，先进行100状态的开发Pilot（新的独立目录）：

```bash
STUDY_NAME=section_utility_pilot100_v1 MAX_STATES=100 \
ACTIONS_PER_STATE=8 bash road_vrp7_step6a/run_step6a.sh collect \
  2>&1 | tee step6a_pilot100.log
```

确认标签、耗时和正改善率后，再采集至多1000个**不重复**的历史最优状态：

```bash
STUDY_NAME=section_utility_1000_v1 MAX_STATES=1000 \
ACTIONS_PER_STATE=8 bash road_vrp7_step6a/run_step6a.sh collect \
  2>&1 | tee step6a_1000.log
```

原始快照可能重复，经过去重后若不足1000个独立状态，会如实只输出实际可用数量，**不会通过复制状态来凑数**。

发生中断，参数和源码未变化时：

```bash
STUDY_NAME=section_utility_1000_v1 MAX_STATES=1000 \
ACTIONS_PER_STATE=8 RESUME=1 bash road_vrp7_step6a/run_step6a.sh collect \
  2>&1 | tee -a step6a_1000.log
```

只分析与导出，不重新评估候选：

```bash
STUDY_NAME=section_utility_1000_v1 bash road_vrp7_step6a/run_step6a.sh analyze
```

可调整`SOURCE_DIR`、`SOLVERS`、`SNAPSHOT_TIMES`、`TOP_K`、`LOOKBACK_SLOTS`、`CANDIDATE_DEADLINE_SEC`等环境变量，但**任何参数改变都应新建实验目录**。

## 4. 输出目录

```text
runs/chicago_july_v1/step6a/section_utility_1000_v1/
├── run_config.json              # 输入文件、源码、路网、采样器参数的SHA256及角色声明
├── state_records.jsonl          # 一行一个冻结状态，含多个候选的特征/标签/两种修复路线
├── benchmark_summary.json       # COMPLETE、实际状态数和上限动作数
└── analysis/
    ├── summary.json
    ├── state_ranking_diagnostics.csv
    ├── support_strata.csv
    ├── task_summary.csv
    ├── features_X.npz          # 仅白名单可用特征，X: [N, F]
    ├── labels_y.npz            # 真值及反事实标签，y: [N, 5]
    ├── groups.csv              # 与X/y按行对应的日期、task_id、seed、split等分组信息
    └── dataset_schema.json
```

其中单条候选记录重要字段：

- 输入属性：`route_index`,`position`,`size`,`block_demand`,`route_load`,`static_section_cost_sec`,`history_coverage_by_slot`,`history_rel_cost_by_slot`,`history_observed_share`。
- Oracle监督标签：`td_gain_sec=TD(S)-TD(repair_TD(S,section))`，可以为负数。
- 静态反事实：`static_repair_static_gain_sec=Static(S)-Static(repair_static(...))`。
- 交叉反事实：`static_repair_td_gain_sec=TD(S)-TD(repair_static(...))`、`td_advantage_over_static_repair_sec=TD(repair_static)-TD(repair_TD)`。
- 同一移动的静态收益：`static_gain_same_td_move_sec=Static(S)-Static(repair_TD(...))`。
- 若修复超时或无可行非平凡解，则`label_valid=false`且收益设为null；仅在有效标签上计算均值。不会编造时间未完成的动作标签。

**切勿将`state_records.jsonl`全字段传给学习模型**！它包含标签、Oracle修复路线、任务标识及其他敏感泄漏来源。未来训练须读取`features_X.npz`并采用`feature_schema.json`白名单；`labels_y.npz`只作为监督目标。

## 5. 时间、成本与可复现性审计

- `source_snapshot_sha256`绑定来源1200条Step5-B v3结果；验证原运行使用的Section和道路评价源码SHA与当前文件完全相同。
- 采样种子和候选ID由哈希稳定决定；相同配置重复运行应得到相同候选集合（耗时字段可能波动）。
- 每个状态单独写入JSONL并执行flush/fsync；断点续跑核对已有ID、配置哈希和完整JSON。如果文件末尾损坏，拒绝自动修复。
- 所有离线候选评价、静态路径与历史特征提取时间均**不计入旧的Step5-B搜索预算**；它们只是生成监督数据的代价。未来在线模型必须另外核算编码、预测和ALNS运行时间。
- `--candidate-deadline-sec=3`是**每个静态或TD修复子调用的最大等待时间**，并非30秒ALNS搜索预算；两个对照使用相同上限。
- `Road`加载历史FIFO及完整道路，另单独加载原始历史成本数组，可能额外占用数百MB内存；建议在资源充足的服务器运行。

## 6. 解释结果时的严格限制

1. **当前采样的是Oracle ALNS生成的历史最优路线。** 路线本身可能编码未来交通知识，即使时间序列特征截断到规划出发前，也**不能据此宣称整个Step6-A样本代表可部署策略**。未来必须另采基于静态或预测信息生成的非Oracle当前搜索状态。
2. **没有在所有状态上采ALNS任意当前解。** 历史最优状态较接近局部极小值，会导致正改善候选偏少。若正标签比例很低，下一版应在独立Stage6-A采样运行中插桩ALNS Acceptance/Stop，采集随机当前解，而不是伪造正改善标签。
3. **“静态最佳与Oracle TD最佳”的差距是有先知信息的理论可利用空间。** 历史交通可预测性必须由真正训练出的模型、未来时段/任务保留测试来验证。
4. **不可使用7月既有验证场景声称盲测泛化。** `task_role`仅是开发数据内部任务分组，`date_role`是另一个独立的按日期分组；它们不是同一套联合训练/测试拆分，不能混用。后续需在全年其他月份预先冻结最终未见测试集。
5. **道路路段真实观测很稀疏。** 历史观测覆盖统计基于静态最短路径近似走廊、仅计有效观测；不意味着未观测道路有真实速度标签。
6. **道路级事后回测不是本v1每候选的标签。** 此版本使用与C3相同`Objective.cost`的OD代理成本，后续少量样本可加入昂贵的`Road.replay()`核对代理与真实路网路径成本间的误差。

## 7. 何时推进Step6-B？

先检查`analysis/summary.json`：

- 有效候选数与修复超时率是否可接受；
- 正改善候选是否足够支持监督学习；
- `top1_disagreement_rate`是否显著大于0、Static/TD排序是否存在有意义的差异；
- `mean_oracle_vs_static_selection_regret_sec`是否不只是接近0的浮点噪声；
- 上述差异是否在不同任务/观测支持度分层中重复出现。

若静态和TD收益排序高度一致、后悔值接近零，应先重新检查观测覆盖、候选定义和TD-OD数据差异，不要直接上SPIN/对比学习。若确有排序机会，再训练不带CL的LightGBM/MLP/Section Transformer，最后引入SPIN和SPCLT式CL。

## 8. 本地测试与服务器真实集成

代码包在无完整芝加哥道路数据的环境下通过合成数据逻辑测试、冻结配置恢复检查、历史时间切片测试、分析和白名单导出测试。**完整芝加哥OD计算、官方ALNS搜索与道路回测没有在当前环境实际执行。** 服务器必须执行`test → plan → smoke → 100状态Pilot`后再扩展。
