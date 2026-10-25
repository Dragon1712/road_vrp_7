# Step6-C1 R4-Q v1.1：严格历史24小时交通与数据起始边界静态回退

## 研究依据（GitHub已验证）

基于 `Dragon1712/road_vrp_7` commit `423d638a7a06be6db8da6fb231277c69dffa9c8e` 的 R4-P 源码和 `r4p_static_od_fusion_train_v1/report.json`。

- 4个Train任务，200个场景快照，1,572个替代Section，6个留组。
- 任务级留一、正常标签、`learned_gate_pairwise`固定K500收益：`route_context` −1.599765625秒/快照，`road_concat` −1.455009765625，`road_conditioned` −1.074111328125。3个模型均未超过 C3（0）。
- 静态道路特征有时缓解损失，但无法验证“**随时间变化的真实交通信息与路线状态**”的互补性。
- Stage2 `build_july_stage2.py` 把**对应时槽独立的**原始边费用存在`travel_sec_raw`，保留`chicago_observed_mask`；额外的 `travel_sec_fifo` 用于经过FIFO修正的道路费用。R4-Q **不读取FIFO作为模型特征**。

## R4-Q的核心问题

使用**决策时刻整整24小时前**已经发生的交通观测（例如7月9日17:00只引用7月8日17:00），能否比纯静态道路OD、路线结构特征提供更稳定的跨任务/跨Seed Section决策信号？

这只是**历史交通代理（past-only lagged proxy）**，不是在线已知的当前交通，更不是训练得到的交通预测器。目标日同一时段的实测值、原 Road.replay 以及任何未来动态OD**都不参与特征生成**。

## 特征消融与严格对照

| 模式 | 维度 | 内容 |
|---|---:|---|
| `route_context` | 38 | R4-O冻结的路线结构特征 |
| `static_concat` | 50 | 路线38 + R4-P静态有向OD 12 |
| `lag24_concat` | 63 | 路线38 + 静态OD12 + 前一日同一时刻道路OD12 + 已映射边历史观测覆盖1 |
| `lag24_conditioned` | 75 | 上述63维 + 12个预先固定的“结构×(滞后OD−静态OD)”交互 |

模型仍复用R4-O的Pairwise加权Ridge和独立状态门控，**并没有训练FiLM或对比编码器**。训练参数和分组方法保持R4-O/P一致；这样模型差异主要来自输入模态。外层四任务留一和两组种子迁移，无状态候选跨组拆分。另含Shuffle训练标签负对照。事后Oracle策略只用于诊断上界，禁止部署。

模型用的滞后OD不是“把目标日Road.replay标签往回移动”。处理如下：

1. 读取Stage2 `edge_index.csv`静态有向道路图，核对边顺序。
2. 校验 `edge_costs.npz` 与R4-G旧/新Train捕获的 `road_sha256` 完全相同；核对 `time_index.csv` 与NPZ纳秒时槽一致。
3. 对每个Train场景以当地无时区壁钟计算`departure_local − 24小时`。若目标时槽在Stage2起始时刻**之前**，只允许完整静态回退，并在报告中标记`available=false`、`fallback_reason=BEFORE_DATA_START`；若目标在索引范围内但缺失精确时槽，**仍立即报错**。绝不使用更晚或更近的交通观测代替。
4. 真实历史时槽存在时，仅对`chicago_observed_mask=True`的边使用`travel_sec_raw`；其余使用`static_travel_sec`。历史整体不可用时，所有边使用`static_travel_sec`，构造出的OD必须与静态基线逐值相同。
5. 基于上述历史边费用重新计算任务的**有向**客户OD，仅输入已知路由状态和候选Section几何。客户ID只用于OD索引，绝不直接输入模型。
6. 报告全道路边与已映射（监测）边的历史观测比例、真实时槽可用性和**按场景分别统计**的留组收益，防止将7月2日纯静态回退误当交通模态的贡献。

Stage2压缩NPZ加载时会解压历史+未来完整张量以进行轴/来源验证；**特征值只引用指定历史行**。本地单元测试主动篡改目标日及之后的原始交通矩阵，确认历史特征结果不变。该协议不是端到端实时数据获取API。

## 服务器运行命令

```bash
cd ~/code/WZL/road_vrp_7
unzip -o road_vrp7_step6c1_r4q_v1_1.zip
bash road_vrp7_step6c1_r4q/run_r4q.sh test
bash road_vrp7_step6c1_r4q/run_r4q.sh preflight
STUDY_NAME=r4q_past24_traffic_train_fix1 bash road_vrp7_step6c1_r4q/run_r4q.sh probe
cat runs/chicago_july_v1/step6c1_r4q/r4q_past24_traffic_train_fix1/report.md
```

输出文件：`report.md`、`report.json`（240个模型/负对照/留组配置）、`fold_predictions.jsonl`（固定K500策略实际收益与门控结果）、`provenance.json`（全部模型、历史数据及原始来源SHA）、`historical_od/*.npz`（8个任务×场景组合的滞后/静态回退有向OD矩阵、请求的历史时槽和真实可用性）。**输出目录必须为全新的目录，不覆盖任何R4-A~P结果。**

可通过 `R4Q_ROOT`、`STUDY_NAME` 和 `R4Q_OLD_CF_DIR` 等 `R4Q_` 前缀变量自定义输入，具体见 `run_r4q.sh`。不要覆盖其它阶段的同名泛用环境变量。

### 可能的服务器问题

- `No exact 24h-past observation slot`：目标位于Stage2覆盖范围**内部或右侧**但缺少精确时槽，属于异常，必须停止并核查数据。仅**左边界之前**允许静态回退。
- `Stage2 traffic NPZ differs from frozen Capture`：道路成本被修改或引用了错误版本，停止实验并核查SHA。
- 观测覆盖率低：意味着大量lag24最短路边费用事实上回退静态；此时即使整体有一点改善也不足以说明交通模态有效。
- `New independent STUDY_NAME required`：该结果目录已存在，使用新的实验名，绝不在旧目录追加。
- 服务器需能加载完整Stage2压缩NPZ（原始二维边成本与观测掩码），此外会建立8个有向滞后OD矩阵；CPU/内存开销比R4-P高。

## 判读优先级

1. 先确认历史时槽确实早于出发24小时、历史观测覆盖度足够且 `provenance.json` SHA匹配；缺任一项**不允许解释增益**。
2. 比较 `lag24_concat`、`lag24_conditioned` 是否在**四个Train任务留一折**和两方向种子迁移中稳定优于`static_concat`；不能只挑最佳折。
3. 区分Top-1正收益命中与随机排序基线，计算正/负收益委托及负收益总和；如果仍比C3差，不进入在线部署。
4. 比较正常训练与Shuffle负对照。两者相近说明监督与预决策特征关系弱。
5. 即使表现改善，**下一阶段也必须先冻结策略，进入真正独立Validation及严格30秒在线预算对照**，而不是把本轮固定K离线收益当作在线增益。

## 验证状态与限制

- **已完成**：17项本地合成单元和完整4任务留组流程测试；Python编译/Shell语法；封装SHA校验。
- **未完成**：真实Chicago Stage2 NPZ、路网Dijkstra性能及R4-Q收益的服务器级实验。
- 仅有4个Train任务、2个日期；历史同日/次日相关性和近重复场景限制统计独立性。
- 本轮仍不是完整神经FiLM与对比学习复现；其目标是判定真实历史交通模态是否值得进一步开发。

## v1.1变更与研究边界（必须阅读）

已通过GitHub `time_index.csv`确认Stage2首行日期为**2025-07-02 00:00:00**，故7月2日08:00的前24小时来源**2025-07-01 08:00:00**在此数据版本下缺失。原v1误将两个Train场景均要求历史存在，并因此在preflight报错。修复版行为：

| 场景 | 需要的时槽 | 源是否可用 | 特征处理 |
|---|---|---|---|
| 2025-07-02_0800 | 2025-07-01 08:00 | 否（早于索引起点） | **100%静态回退**，请求时槽保留、实际时槽为null、观测掩码全false |
| 2025-07-09_1700 | 2025-07-08 17:00 | 是（索引中） | 读取该时槽的Stage2原始观察并静态回退未观察边 |

模型的四组特征仍为38/50/63/75维，模型、超参数、样本、固定K标签及外层分组均未改变，新增分场景`history_stratified`报告表。**本次修复不是“补足7月1日历史交通”，不能把7月2日的结果当作动态交通模态有效的证据。** 若研究目标要求“两个日期都有真实的历史交通”，应另行设计仅使用7月2日之后有效时槽的新Train场景，不应假造7月1日交通。

`historical_od/`中对于无历史来源的场景，`source_slot`为空字符串，`requested_source_slot`保留被请求日期，`history_available=false`。数据覆盖范围内的缺口仍会触发错误，不启用最近邻回退，也不修改Stage2数据、ALNS或上游来源SHA。
