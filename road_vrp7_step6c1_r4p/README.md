# R4-P v1 — 冻结 Train 反事实上的**静态路网模态** Section 排序实验

## 1. 研究问题及诚实边界

R4-O 在 Train 内部 LOTO 检验中发现：38维路线结构并未稳定优于15维几何特征，甚至使 task_0000、task_0003 的 Top-1 正收益命中降低；task_0002 有小幅改善。已有学习门控在 task_0001 仍产生伤害。由此本阶段检验**新增独立路网模态的静态图距离**能否补足路线表示，绝不宣称已经使用了实时交通多模态。

**静态路网模态：** `datasets/processed/chicago/july_v1/dynamic_costs/edge_index.csv` 中有向 `osm_u,osm_v,static_travel_sec`，由道路Dijkstra计算任务客户/仓库的静态OD路径。只读取这三个列。**不读取** `edge_costs.npz`、`travel_sec_fifo`、未来交通观测或事后 `Road.replay` 成本作为特征。反事实K500 Road收益只作为**监督标签**。此特征在决策前可计算，但离线Dijkstra的预计算时间 **不等于在线免费**：未来30秒线上实验必须实际衡量预计算/推理开销。

本阶段的 `road_conditioned` 只是**显式路网×路线结构双线性交互（FiLM-style linearized proxy）**，不是原始论文中的可训练FiLM神经模块，也不是对比学习。若有可信跨任务增益，后续再考虑真实预测交通模态、完整FiLM与InfoNCE消融。

## 2. 固定消融

| 变体 | 特征维数 | 内容 |
|---|---:|---|
| `route_context` | 38 | R4-O 原结构基线（15+23） |
| `road_concat` | 50 | 加入12个静态路网OD指标 |
| `road_conditioned` | 62 | 50维+12个路线×静态距离的固定乘积；学习系数只在训练折拟合 |

**保证等预算、公平对照：** 沿用未修改的R4-O分组 Pairwise Ridge及状态Gate训练器、固定正则化，不调外层留组阈值。分别比较 `C3`、`pairwise_ridge_gate`、`learned_gate_pairwise`、`oracle_gate_pairwise` 和`sampled_oracle`。最后两项读取留出未来标签，仅用于诊断，不可部署。

6个外层留组：四个Train task留一以及两种种子组迁移。3组特征 ×2组Train全状态Shuffle对照×5种策略 = **180个评价配置**。所有训练与Shuffle只用外层训练分区；同场景同快照Section候选不跨组。报告同时保存原生R4-O和静态路网结果供配对比较。

## 3. 最小执行顺序（在已有代码仓库根目录）

```bash
cd ~/code/WZL/road_vrp_7
unzip -o road_vrp7_step6c1_r4p_v1.zip
bash road_vrp7_step6c1_r4p/run_r4p.sh test
bash road_vrp7_step6c1_r4p/run_r4p.sh preflight
STUDY_NAME=r4p_static_od_fusion_train_v1 \
  bash road_vrp7_step6c1_r4p/run_r4p.sh probe
cat runs/chicago_july_v1/step6c1_r4p/r4p_static_od_fusion_train_v1/report.md
```

`preflight`在扫描昂贵路网之前检查：冻结R4-O证据、全部R4-L/M/N旧有来源 SHA、Train任务/场景角色、道路edge_index字节SHA同原R4-G捕获记录一致；任何缺失即失败。`probe`随后逐任务从静态图计算OD，检查捕获快照中出现的客户编号与对应任务完全匹配。

可覆盖路网/任务目录（必须通过上游SHA）：

```bash
R4P_COST_DIR=datasets/processed/chicago/july_v1/dynamic_costs \
R4P_TASK_DIR=datasets/processed/chicago/july_v1/tdcvrp30/tasks_10 \
STUDY_NAME=r4p_static_od_trial2 \
bash road_vrp7_step6c1_r4p/run_r4p.sh probe
```

必须使用独立新 `STUDY_NAME`，已存在结果不允许覆盖；不写入也不修改任何R4-A~O源码与结果。`probe` 是离线特征构造+模型评估，没有ALNS调用（但静态道路最短路计算可能比较耗时）。

## 4. 输出

`runs/chicago_july_v1/step6c1_r4p/<STUDY_NAME>/`:

- `report.json`：180项固定实验，全组收益/伤害/Top-1对照及限定条款。
- `report.md`：未打乱模型下学习型门控+排序的配对摘要。
- `fold_predictions.jsonl`：每个快照的被选Section、固定K500收益、oracle对照标志。
- `provenance.json`：输入/输出和源静态图/任务清单SHA。
- `static_od/task_XXXX.npz`：仅**静态**有向OD及对应客户轴ID，必须按 `customer_ids` 轴读取。

缺失、不合法、非Train来源一律拒绝；无完整来源时不允许只读汇总JSON“重建”模型结果。

## 5. 判读规则

优先看4个Train task留一、非Shuffle的 `learned_gate_pairwise` 净收益、最差伤害及正负委托，然后比较 `route_context` vs `road_concat` vs `road_conditioned`。如果仅旧种子→新种子改善而task留一依然为负，不能声称静态道路模态解决泛化。若 `road_conditioned` 高维交互进一步恶化，说明正则化/数据规模存在风险，不应直接上更复杂FiLM。

**独立Validation和固定30秒在线搜索尚未完成。** 当前Road.replay 500步反事实收益只回答“在现有C3轨迹某一步替代Section的反事实后果”，不能直接加总成完整在线策略收益。特征虽不泄漏标签，Oracle上界仍不可部署。

## 6. 已确认的限制

1. 只有四个原始Train配送任务；两日期多种子不等价于大量独立任务。
2. R4-P使用**静态路网地理模态**，不是实时预测交通；动态历史路况未经预测时序隔离，不应输入模型。
3. 线性化交互是FiLM风格的基线，不是复现AAAI 2018官方FiLM训练源码。
4. 静态OD构造的离线CPU耗时及内存须单独核算，30秒在线预算没有覆盖。
5. 结果是离线固定K=500 Train内部外层留组，不是独立Validation/Test或线上改善证据。
