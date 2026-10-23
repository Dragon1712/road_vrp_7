# Step6-C1-R3-SV: 同状态决策审计 + 冻结C3短程多步价值Pilot

> **真实运行状态**：本包代码及合成测试完成；**尚未**在Chicago服务器上运行真实道路数据。本包不修改原R2、R3、ALNS源码与已有结果。

## 0. 研究目的与关键边界

基于已经标注的1,024个冻结C3状态及8个候选Section，诊断“离线一步TD收益”是否与“后续H次ALNS搜索后的道路质量”一致。保留四个动作来源：

- `C3`：捕获的**真实**冻结C3动作；
- `M0_PROXY`：使用原M0 P2-F2模型，在**同一8候选动作集合**选择，并将拒绝动作归一化为真实C3回退。**不是**冻结M0独立在线轨迹的完整重演；
- `R2_F2`：冻结R2-F2选择器；
- `T2_HUBER`：已有R3-T1的CatBoost-Huber选择器。

### 阶段A：同状态一步价值（无需道路Oracle重新计算）

读取已有`paired.npz`的`X,B,Y,BY,valid,same`，统一比较上述选择器：

\[
 \Delta_1(s,a)=\operatorname{Gain}_{\rm TD-OD}(s,a)-\operatorname{Gain}_{\rm TD-OD}(s,a_{C3})=Y_a-BY
\]

C3相对优势记为0。输出所有状态的收益、Regret、Top1Hit、错误覆盖（相对C3劣化超过40秒）及分train/validation/internal_test的指标。**仅内部测试折可用于诊断展示，不得据此选参然后宣称独立泛化。**

### 阶段B：冻结C3的H步重启搜索代理

从**捕获的同一状态**重建真实C3动作与8个Section候选，按原有`section_dataset.repair_for()`进行相同Static Repair，先逐项核对原一步标签（误差容限0.12秒，float32量化）。再将四个修复后路线作为初始解，分别调用**原始**`road_vrp7_step5b_v3/run_experiment.py`里的`v3.solve('C3_topk', ...)`，后续迭代数固定为`H=5,20,50`，两次重启重复实验使用相同的状态级随机数种子。原始ALNS的destroy、repair、RouletteWheel、SimulatedAnnealing、目标函数（static）及topK保持不变。

**极其重要的限制**：R2采集文件只保存搜索路线、真实C3动作、任务和场景，**没有保存ALNS算子权重、原搜索时刻的退火温度、RNG内部状态**。所以本方法是**paired short-horizon ALNS restart surrogate**，不是从原30秒搜索精确续接的反事实模拟。每条分支会重新从新初始解`SimulatedAnnealing.autofit()`，因此SA初始温度也可能略有差别。报告必须以此命名；即使看到H步收益也不能直接解释成原30秒搜索的严格动作Q值。

每个重启分支结束后，用**事后**`Objective(...,'oracle').cost()`评估最终最佳静态路线的TD-OD成本；可选择额外启用`--road-replay`，产生`Road.replay`的完整事后路网指标。**Oracle仅用于离线标签/回测，绝不提供给静态ALNS搜索。**

固定H是迭代预算而非固定壁钟时间；另设每个分支30秒安全上限，若无法完成H次迭代将直接失败，不会错误地比较部分完成结果。

## 1. 依赖

在之前已成功运行R2 / R3-T1的`~/code/WZL/road_vrp_7`服务器环境：

- Python含 `numpy pandas joblib scikit-learn catboost torch` 与`alns==7.0.0`（原项目依赖）；
- 冻结源码目录`road_vrp7_step5b_v3`、`road_vrp7_stage3`、`road_vrp7_step6a`、`road_vrp7_step6c1_r2`、`road_vrp7_step6c1_r3_tricks`、`road_vrp7_step6b1_v2`、`road_vrp7_step6b2`；
- 已有R2采集与标签：`runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1`及其manifest引用的capture目录；
- 已有R2-F2、M0-P2/F2、R3-T2 CatBoost-Huber种子42模型及Chicago 2025年7月动态道路数据。

## 2. 安装与验证

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r3_sv_v1.zip
bash road_vrp7_step6c1_r3_sv/run_sv.sh test
bash road_vrp7_step6c1_r3_sv/run_sv.sh preflight
```

`preflight`只核对SHA、源数据/模型身份及存在性，不执行完整OD重建。

## 3. 阶段A：完整同状态审计

```bash
STUDY_NAME=sv_same_state_v1 \
PILOT_STATES=12 PILOT_ROLE=internal_test \
bash road_vrp7_step6c1_r3_sv/run_sv.sh same-state
```

生成：

- `decisions.jsonl`：全部1,024个状态×4个选择器的逐状态实际签名、1-step signed TD优势、regret与错误覆盖；
- `report.json`：Train / Validation / Internal Test的平均收益、Top1、Regret、错误覆盖；
- `pilot_states.json`：确定性抽样的12个研究状态。约一半为R2-F2和Huber动作不一致状态，另一半为一致状态；**这是为诊断而富集的样本，不代表总体无偏估计**；
- `manifest.json`：输入数据、模型与代码SHA。

### 建议先观察

1. 两模型在相同状态下产生不同动作的频率；
2. Huber对比R2-F2在相同状态下的真实一步收益差和平均Regret；
3. `M0_PROXY`是否在相同8候选中优于其他模型；
4. 损失超过40秒的错误覆盖主要集中在哪些任务/场景/搜索阶段。

## 4. 阶段B：H=5/20/50多步Pilot

先只运行**1个状态×3个H×1次随机重启×4种策略**，作为真实路网Smoke（12次短程ALNS重启）：

```bash
STUDY_NAME=sv_rollout_smoke_v1 \
ONE_STEP_STUDY=sv_same_state_v1 \
MAX_STATES=1 REPLICATES=1 HORIZONS=5,20,50 \
bash road_vrp7_step6c1_r3_sv/run_sv.sh rollout
```

如果真实路径、候选重建和原标签配对通过，再执行完整Pilot：

```bash
STUDY_NAME=sv_rollout_12states_v1 \
ONE_STEP_STUDY=sv_same_state_v1 \
MAX_STATES=12 REPLICATES=2 HORIZONS=5,20,50 \
MAX_NEW_UNITS=1 \
bash road_vrp7_step6c1_r3_sv/run_sv.sh rollout
```

首次只完成1个“状态×H×重启种子”的四策略配对单元，随后继续：

```bash
STUDY_NAME=sv_rollout_12states_v1 \
ONE_STEP_STUDY=sv_same_state_v1 \
MAX_STATES=12 REPLICATES=2 HORIZONS=5,20,50 \
RESUME=1 \
bash road_vrp7_step6c1_r3_sv/run_sv.sh rollout
```

如需直接评估真实道路回放，而非TD-OD成本，在一个**全新结果目录**启用`ROAD_REPLAY=1`：

```bash
STUDY_NAME=sv_rollout_12states_road_v1 \
ONE_STEP_STUDY=sv_same_state_v1 \
MAX_STATES=12 REPLICATES=2 HORIZONS=5,20,50 \
ROAD_REPLAY=1 \
bash road_vrp7_step6c1_r3_sv/run_sv.sh rollout
```

**严禁向不同SHA或不同`ROAD_REPLAY`设置的旧目录追加数据。**

生成：

- `units/<decision_id>__H<h>__R<r>.json`：四种策略组成的不可拆分配对单元，原子写入，支持安全续跑；
- `rollouts.jsonl`：完整后续搜索代价与路径信息；
- `report.json`：每个H的相对真实C3的多步成本改善、胜负平、以及一步与多步动作价值**正负号翻转比例**；
- `manifest.json`：Road、Section、ALNS、原标签、模型及各脚本的校验和。

每种H共有12状态×2重启×4种动作=96条分支；三个H合计288条**短程重启**（不是288条30秒ALNS求解）。

## 5. 科学解释与下一步决策

- 一步值高，多步值也高：可继续改进Section排序/可靠性委托，并考虑F2+交通F4的条件融合；
- 一步值高，多步频繁翻负：有证据表明目前一步监督目标与连续搜索目标错位，应优先设计多步价值学习/DAgger状态收集；
- H5/H20/H50不稳定：检查有效样本与退火初始温度差异，扩大新的状态和重启种子后再决定；
- 离线选择准确率变好、30秒在线依旧不改善：不能继续只优化MAE/Top1，应转向实际搜索回报对齐与独立月份评估。

目前含源文件级SHA校验和停止条件，不自动覆盖已有历史结果。所有这批7月数据仍属开发/诊断，**不是独立盲测**。


## v1.1 修复：旧 M0 模型反序列化 `ModuleNotFoundError: models`

`same_state.py` v1 遗漏 `road_vrp7_step6b1_v2` 的 import 路径，而 M0 的
`P2_F2_model.joblib` 使用 `models.Predictor`（其源码保存在
`road_vrp7_step6b1_v2/models.py`）。v1.1 已在 `joblib.load()` 前补齐该路径；
`preflight_sv.py` 原本已有这一目录，无须修改旧模型，也无须重新训练。

**旧源码的立即绕过方式**（不修改文件）：

```bash
cd ~/code/WZL/road_vrp_7
PYTHONPATH="$PWD/road_vrp7_step6b1_v2:$PWD/road_vrp7_step6c1_r3_tricks:$PWD/road_vrp7_step6c1_r2:${PYTHONPATH:-}" \
STUDY_NAME=sv_same_state_v1 PILOT_STATES=12 PILOT_ROLE=internal_test \
  bash road_vrp7_step6c1_r3_sv/run_sv.sh same-state
```

**安装 v1.1 后**沿用前述常规 `bash .../run_sv.sh same-state` 命令即可。
如果本次错误发生于 `out.mkdir()` 前，原结果目录不会被创建，可继续使用同一个
`STUDY_NAME`；否则应检查目录内容并新建结果名。

这次仅更改导入路径和测试，不更改 Section 决策、标签、搜索过程或统计逻辑。
