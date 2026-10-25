# R4-I v1 — Train 多时域标签可学习性与保守Section委托试验

## 研究背景（已核查GitHub 2026-10-25 最新 R4-H）

源仓库：`Dragon1712/road_vrp_7`，提交 `ef4de64795fe9b3e404552a431757a7d0d2b5a21`。

R4-H Full：72 个交通场景下的Section快照、51 个不同物理状态、565 个候选Section动作、3 个搜索长度K={50,200,500}、1695条动作×K标签、2127个分支；216个SHAM严格匹配和1418个跨K前缀检查。来源R4-G-Prefix 298/298。

| K | 正收益 | 负收益 | 零收益 | 平均Road收益(s) |
|---|---:|---:|---:|---:|
| 50 | 2 | 1 | 562 | +0.044 |
| 200 | 17 | 16 | 532 | +1.424 |
| 500 | 18 | 30 | 517 | -0.964 |

最值得进一步证伪的发现：所有K的**非零历史最优道路收益**都只发生在每个搜索轨迹约1000次迭代的首个Section快照：K500时早期125个动作正18/负30/零77；其余后期440个动作全零。此前K200→500 6次正收益转负收益，涉及3个不同物理状态。

### 目的与边界

此版本**不再运行ALNS**，也不训练复杂神经网络，不访问Validation/Test，不宣称30秒在线收益。它验证现有Train固定K收益**是否值得学习**，先建立可复核且保守的轻量模型及C3回退基线。

特别注意：R4-H实验本身曾查看了不同候选的Train结果，R4-I的任务留一只是 **Train内部探索性的LOTO（Leave-One-Train-Task-Out）**，不是全新盲测。任务仅4个，且K500只有7/51个物理状态含可改善候选。任何模型有效结论都要在未用于研发的Validation上重新验证，之后才有资格进入FiLM/对比学习多模态训练。

## 实验设计

- 读取源文件 `r4h_train_full_v1/report.json`, `manifest.json`, `labels.jsonl`, `branch_summary.jsonl`, `plans.jsonl`，与 `r4g_train16_fix2_capture/snapshots.jsonl`、`r4g_prefix_all16_v1` 中的固定K证明互相检验。所有来源有固定SHA记录。
- 验证实验仅为 `TRAIN_ONLY`，同一Section每个K有唯一C3/SHAM/替代分支；Road收益逐行重算为`C3_best_road - alt_best_road`。一旦SHAM路线或成本不一致则停止。
- 使用在`BEFORE_SELECT_AND_DESTROY`时已经可获得的**静态低成本特征**：搜索进度、早期指示、剩余墙钟比例、当前与历史最优代理成本差、退火温度、Section边界的route/position/size、与C3区块距离、候选集合大小、原Repair类型。特征中无Road收益、未来搜索事件、未来K标签或结果。
- 两类目标：`K500道路收益` 和保守的 `min(K200道路收益, K500道路收益)`。
- 纯Python岭回归效用模型，按**不同物理搜索状态**归一权重；训练标签在固定±250秒范围截断，仅用于稳定探索性拟合，实际评价始终用**未截断的真实K500收益**。
- **每折留出整个task**：训练只用其他3个Train任务，留出任务的所有seed、交通场景、Section动作、K都不得进入训练。对每个留出状态先按预测效用找最高Section，若预测不超过预设门槛，则回退C3。
- 固定门槛0、5、15、30秒全部报告，**不能挑最好门槛然后宣称独立验证**；增加训练标签随机置乱负对照，排除只学到Train分布频率的假象。
- 参考对照：永不委托C3=0；包括C3在内的均匀随机动作的**期望**收益；仅早期才允许随机的阶段门控；已采样候选的事后Oracle最大收益（不可部署上界）。
- 生成按任务、搜索阶段的正/负/零收益分布，按物理状态去重。**不做动作级IID置信区间或显著性伪证明。**

## 运行命令

在原GitHub仓库根目录解压本包（不会修改任何原ALNS/R4-A至R4-H源码）：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r4i_v1.zip
bash road_vrp7_step6c1_r4i/run_r4i.sh test
```

执行一次只读离线评估（不需要道路文件或ALNS运行环境；仅本仓库原始JSON/JSONL结果）：

```bash
STUDY_NAME=r4i_signal_loto_train_v1 \
  bash road_vrp7_step6c1_r4i/run_r4i.sh probe

cat runs/chicago_july_v1/step6c1_r4i/r4i_signal_loto_train_v1/report.md
```

已有相同输出目录时修改 `STUDY_NAME`，不追加或覆盖旧实验。

默认来源分别是：

```text
runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1
runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture
runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1
```

如果实际目录不同，使用环境变量 `R4H_DIR`、`R4G_CAPTURE_DIR` 和 `R4G_PREFIX_DIR` 指定；输出根目录可以用`R4I_ROOT`覆盖。

## 输出

`runs/chicago_july_v1/step6c1_r4i/<STUDY_NAME>/`：

- `report.md`：分任务、分搜索阶段的标签稀疏性；LOTO固定门槛结果与负对照。
- `report.json`：所有Fold及门槛的可计算指标与解释限制。
- `fold_predictions.jsonl`：每个留出状态的委托动作或回退C3、实际收益、样本Oracle上界和随机期望，便于查错。
- `provenance.json`：输入SHA以及核心代码、报告SHA。

## 下一步分岔决策

**A. 正收益只来自极早期**：发展R4-J early-stage Train捕获协议。必须先在原Train任务采集更多不同seed、任务与早期实际Section状态，走原R4-G-Prefix精确验证，再运行R4-H反事实；不能直接以当前R4-H Train中表现好的状态重新作独立评估。

**B. LOTO学习模型不超过永不委托C3或与Shuffle结果相近**：放弃立即训练复杂多模态网络，说明样本和可辨识信号尚不充分，优先改进状态采集与动作表示。

**C. LOTO存在跨任务收益且伤害可控**：先冻结算法、门槛及模型形式，在此前未用于模型选择的Validation日期上做独立离线配对比较，然后进行真正30秒墙钟在线基准。通过后再逐步纳入道路序列、地图特征及FiLM/对比学习消融。不能将此离线LOTO当成独立的泛化声明。

## v1.1 hotfix：精确物理搜索状态身份（2026-10-25）

R4-I v1在真正训练开始前可能报：

```
ValueError: Physical state count disagrees with source
```

这不是R4-H的51个物理状态错误。R4-H用未经舍入的退火温度及
`r4a_core.canonical_hash`对RouletteWheel权重生成状态身份；R4-I v1
用 `round(temperature,9)`，在真实72个R4-H场景快照中，把51个精确状态
合并成36个（其中15组仅有微小退火浮点差异）。

v1.1中的`r4i_core.physics()`现在按R4-H原始精确口径：

- 保留退火温度的完整双精度，拒绝NaN/Inf。
- 对原始保存的JSON权重数组按R4-A规范生成SHA256，**不转换整数为浮点后再哈希**。
- 保留强校验：生成数目必须与R4-H报告的`n_distinct_physical_states=51`一致。

安装与重跑（原始R4-H、R4-G、R4-G-Prefix数据保持原样）：

```bash
cd ~/code/WZL/road_vrp_7
unzip -o road_vrp7_step6c1_r4i_v1_1.zip
bash road_vrp7_step6c1_r4i/run_r4i.sh test
STUDY_NAME=r4i_signal_loto_train_fix1 \
  bash road_vrp7_step6c1_r4i/run_r4i.sh probe
cat runs/chicago_july_v1/step6c1_r4i/r4i_signal_loto_train_fix1/report.md
```

**统计解释：** 严格与源一致的51个状态并不等于51次独立随机实验。
有些不同交通场景对应相同路线、RNG和近乎相同的退火温度；若后续对泛化
误差做推断，应另开展按任务/seed/迭代位置聚类的敏感性实验。
本补丁没有修改收益、监督信号、跨任务拆分或阈值，也没有运行在线30秒ALNS。
