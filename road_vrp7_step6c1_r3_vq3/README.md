# Step6-C1-R3-VQ3 v1 — 决策导向Section排序与严重损失概率校准

## 0. 适用背景与目标

在GitHub `Dragon1712/road_vrp_7` 的 R3-VQ2 六种子实验中，原Train四任务留一得到：

| 策略 | H=20 Road平均多步收益 | H=50 Road平均多步收益 |
|---|---:|---:|
| E0_FULL | +4.90s/状态 | +4.28s/状态 |
| E0_MATCHED | +8.11 | +7.19 |
| VQ_MEAN | -4.48 | -6.22 |
| RESID_OOF25 | +5.72 | +7.34 |
| RESID_OOF50 | +6.75 | +7.40 |

VQ_MEAN在6个种子、两个H上均输给E0_MATCHED；VQ2 H=50 的RESID_OOF50相对E0_FULL约+3.13秒，但尚未稳定超过E0_MATCHED。多步价值回归MAE更好而正收益判别、选动作效果更差；旧SD惩罚还可能仅靠降低覆盖率降低损失。因此本次用 **相同旧标签** 验证：

1. 以真实C3为第9候选的 **within-state Pairwise Logistic Ranker**，学习排序而非精确收益数值。
2. 基于重启Road.replay的 **严重损失分类（平均多步增益<-40秒）** 与仅Train内部的跨任务OOF Platt校准。
3. 用 **相同预期覆盖率的纯分数门槛** 作为负对照，检验风险模型是否真的识别危险动作。

本实验**不会修改官方ALNS、不创建新的多步标签、不接入30秒在线搜索，不把R2原Validation或Internal Test标签用于训练/调参**。

## 1. 文件与旧版本依赖

新目录：`road_vrp7_step6c1_r3_vq3/`（与VQ2等并列），代码：

- `vq3_core.py`：真实C3锚点、轻量RankNet式线性pairwise逻辑回归、跨任务风险OOF、概率校准、覆盖率匹配
- `train_vq3.py`：Train四任务外层留一，内层OOF；结果、候选预测与模型评分审计
- `report_vq3.py`：每个训练种子的Markdown报告
- `aggregate_vq3.py`：6种子汇总，SHA一致性检查
- `run_vq3.sh`：完整运行入口；`tests/test_vq3.py`：合成/边界/对照回归测试

需要保持原代码和数据：

```
road_vrp7_step6c1_r2/r2_policy.py
road_vrp7_step6c1_r3_vq/vq_core.py
road_vrp7_step6c1_r3_vq/train_vq.py
road_vrp7_step6c1_r3_vq2/vq2_core.py
runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1/{paired.npz,groups.jsonl,manifest.json}
runs/chicago_july_v1/step6c1_r3_vq/vq_train48_H20H50_R3_v1/{rollouts.jsonl,manifest.json,report.json}
```

环境：Python 3.10+，NumPy、scikit-learn、joblib及原仓库已有的依赖。旧依赖由原VQ模块负责检查；使用可信的本地模型文件，不下载/加载未知pickle。

## 2. 实验组

全部方法 **原Train任务task_0000...0003外层LOTO**，以验证状态上H=20和50的**三重启Road.replay平均收益**评估。C3动作的相对收益定义为0。

| 方法 | 选择分数/规则 |
|---|---|
| `C3` | 永不覆盖 |
| `E0_FULL` | 原Train其他3任务的完整一步ET树（约3,000有效候选） |
| `E0_MATCHED` | 与VQ相同的36个训练状态、约282–285候选的一步ET |
| `VQ_MEAN` | 相同候选，直接回归多步重启均值 |
| `RESID_OOF25` | 全量一步 + 0.25×任务OOF残差树 |
| `RESID_OOF50` | 全量一步 + 0.50×任务OOF残差树 |
| `PAIR_RANK` | 同状态候选+C3的普通成对逻辑排序，正分数才覆盖 |
| `PAIR_REGRET` | 同上，按收益差距sqrt(|gap|/40)加权，权重归一为每状态总和1 |
| `E0_RISK20` | E0_MATCHED最优候选且正分，校准严重损失概率≤0.20，否则回退C3 |
| `E0_RISK35` | 同上，概率≤0.35 |
| `RESID25_RISK35` | RESID_OOF25首选候选 + 概率≤0.35 |
| `PAIR_REGRET_RISK35` | PAIR_REGRET首选候选 + 概率≤0.35 |
| `E0_SCORE_COVERAGE35` | 仅E0_MATCHED分数门槛；门槛由**外层训练任务OOF**设定，使其覆盖率接近`E0_RISK35`的OOF覆盖率 |

**Risk门控选择规则**：先按主模型选Top-1，如分数≤0或该Section风险概率超过阈值，就回退真实C3。不会改选第二候选；保证只改一个机制。

### Pairwise中特别重要的C3特征

候选特征为原F2的78维`[candidate26,C3_26,candidate-C3]`。真实C3的配对特征应该是`[C3,C3,0]`，**不是一个全零向量**。因此`PAIR_RANK`严格通过特征差向量学习候选相对C3的排序，避免上轮训练归一化导致错误锚点。

### 这里的“风险概率”到底是什么

监督目标是`1[三个配对重启种子下的平均Road.replay相对收益 < -40秒]`，它**不是**“下一次重启出现大额损失的概率”，也**不是**“模型准确率的置信下界”。仅在3个外层训练任务里做内层任务OOF概率+Platt校准，数据很少，不能声明可证明的安全性。

## 3. 安装与运行

仓库根目录执行：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r3_vq3_v1.zip
bash road_vrp7_step6c1_r3_vq3/run_vq3.sh test
bash road_vrp7_step6c1_r3_vq3/run_vq3.sh preflight
```

先跑Seed42（独立结果目录，**不可继续复用VQ2目录**）：

```bash
TRAIN_SEED=42 TREES=160 STUDY_NAME=vq3_loto_seed42_v1 \
  bash road_vrp7_step6c1_r3_vq3/run_vq3.sh train
```

如Seed42完成，再运行种子43–47：

```bash
for s in 43 44 45 46 47; do
  TRAIN_SEED=$s TREES=160 STUDY_NAME=vq3_loto_seed${s}_v1 \
    bash road_vrp7_step6c1_r3_vq3/run_vq3.sh train
done
```

六种子汇总：

```bash
INPUT_STUDIES=$(printf 'vq3_loto_seed%s_v1,' 42 43 44 45 46 47)
INPUT_STUDIES=${INPUT_STUDIES%,}
INPUT_STUDIES="$INPUT_STUDIES" STUDY_NAME=vq3_aggregate_seed42to47_v1 \
  bash road_vrp7_step6c1_r3_vq3/run_vq3.sh aggregate
```

结果位于`runs/chicago_july_v1/step6c1_r3_vq3/<STUDY_NAME>/`；每个种子包含`report.json`、`report.md`、`fold_decisions.jsonl`、`candidate_predictions.jsonl`、`provenance.json`，聚合包含`report.json`、`report.md`、`provenance.json`。每个目录必须是新的；代码默认拒绝覆盖已存在的目录。

若已有VQ的原数据路径不一致，可覆盖：`DATASET_DIR=... COLLECTION_ROOT=... COLLECTION_STUDY=... OUTPUT_ROOT=...`。

## 4. 必须审计的主要结果

**第一优先：** `PAIR_RANK` 与 `PAIR_REGRET` 是否在 **H=20和H=50、两种子以上、4个留出任务** 上比 `E0_MATCHED` 提升状态级Top-1/Regret/道路收益；pairwise accuracy是否同步改善。如果只有pairwise accuracy改善但最终收益不变，不可声称优化质量上升。

**第二优先：** `E0_RISK35` 与覆盖率匹配的 `E0_SCORE_COVERAGE35` 必须直接比较 `harm_gt40_per_override` 和 `mean_gain_sec`。如果风险门控没有超过同覆盖率纯门槛，那么只证明了“少做动作”，未证明“识别风险”。

**第三优先：** `RESID25_RISK35` 与 `RESID_OOF25` 比较覆盖率下降造成的机会损失。如果小样本校准使门控过于激进，直接终止Risk路线而不反复对留出任务调参。

单独报告Brier、ECE5、AUPRC、风险类别流行率、按任务的结果；跨种子统计是**重复训练稳定性**，不是多个独立任务的样本量。

## 5. 决策阈值与后续计划

- 只有当**任务级**改善有重复性、同时风险门控在相近覆盖率下仍减少每次覆盖大额损失，才考虑扩大原Train多步标注到192状态。
- 若排序或风险预测的改善不稳定，下一阶段不继续堆模型：优先扩充不同任务/新日期并获取更接近连续在线搜索的真实决策状态。
- 当前没有任何真实30秒在线测量；模型评估标签来自“修复后路线 + 冻结C3重新启动H步搜索”。**不得称为在30秒在线ALNS中提升了道路优化性能**。
- 封存的R2 Validation / Internal Test仍不能用于本轮模型训练、Platt校准或阈值选择。新方法选定以后，在从未参与设计的外部任务/月份执行固定30秒配对实验。

## 6. 安全/可复现边界

1. 严格继承原R3-VQ `train_vq.load()`的完整性与训练任务验证；检查原R2源NPZ/Group SHA。
2. 旧版VQ2五个对照`E0_FULL`、`E0_MATCHED`、`VQ_MEAN`、`RESID_OOF25`、`RESID_OOF50`在合成回归测试中逐项复现，决策/评估仍使用相同流程。
3. 概率风险预测器在外层3任务中每轮只用另外2任务拟合，输出内层OOF；随后只用这3个Train任务校准，避免把外层留出任务的标签用于门槛/校准。
4. 按状态归一化Pairwise对，不能把每状态的成对数当成独立实验样本。
5. 只将真实Road.replay数据作为**回顾性反事实训练标签**，绝不作为在线输入。
6. 本包仅通过本地合成数据测试；未访问服务器上真实Chicago数据，真实结果均待跑。
