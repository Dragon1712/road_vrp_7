# Step6-C1-R3-VQ v1：训练任务专用多步价值 / 风险标签Pilot

## 0. 为什么开发VQ

SV2在48个原R2 Internal Test状态上完成1728次C3重启搜索（不是原轨迹精确续接）。Road.replay评价中，R2-F2在H=5/20/50的平均改善分别为50.824/33.543/34.856秒，CatBoost-Huber为49.214/29.853/30.933秒，但H=50跨重启种子的收益方向不稳定比例分别为65.5%与63.3%。因此下一步不应继续简单拟合“一次Repair收益”，而是先在**原Train任务**生成候选级多步期望收益与噪声标签，评价是否比一步标签更有预测价值。

**本项目当前没有精确多步在线Q标签**：短程搜索调用的是原Step5-B v3的 `solve('C3_topk')`，从修复后的路线重新启动，未恢复原30秒ALNS的退火状态、算子分数和随机数内部状态。多步标签始终称为**C3-restart surrogate value（重启代理价值）**。

## 1. 冻结条件与数据隔离

- 在 `road_vrp7_step6c1_r2/r2_c3_label_train8x8_v1` 中使用此前已生成的真实C3状态、8候选与一步TD-OD标签。**不重新生成Step6-C1-R2标签，也不动原先模型。**
- 状态只从原始按任务划分的Train集合选取：默认 `task_0000`–`task_0003`。Validation `0004`–`0005` 与Internal Test `0006`–`0007` 的状态禁止用于本轮VQ训练及调参。
- 选择规则仅使用`task_id`与`decision_id`稳定哈希，按训练任务轮转抽样；不使用收益/Regret/策略分歧选样。
- 对真实C3和每个**有效且与C3不同**的候选先运行同一冻结Static Repair，和原一步TD-OD标签逐项比较（0.12s浮点容差），不一致直接失败。
- 然后对C3及8候选各自修复后的路线调用相同冻结C3重启搜索。相同状态/重复种子下所有候选使用**相同重启随机种子**；在H=20和H=50也使用相同初始随机种子。
- 每次搜索**仍只使用静态代理目标**；真实`Road.replay`仅在短程搜索完成后计算，作为离线标签，不会进入求解器或在线选择器。
- 若多个Section操作实际产生完全相同的修复后路线，只搜索一次并复用结果；这是严格的相同路线安慰剂检查。
- 严格SHA固定：原Label NPZ、groups、capture、Section、Frozen solver、SV restart、Repair、Road与VQ代码；已完成分片只允许相同配置`RESUME=1`继续。

## 2. 安装与基础检查

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r3_vq_v1.zip
python -c 'import numpy, sklearn, alns; print("Dependencies OK")'
bash road_vrp7_step6c1_r3_vq/run_vq.sh test
```

必须保留以下**已同步的完整源码目录**：
`road_vrp7_step6c1_r2/`、`road_vrp7_step6c1_r3_sv/`、`road_vrp7_step6a/`、`road_vrp7_step5b_v3/`、`road_vrp7_stage3/`，以及Chicago Road/Task/OD Cache。

## 3. 先执行1状态Smoke

```bash
STUDY_NAME=vq_smoke_1state_v1 \
STATES=1 REPLICATES=1 HORIZONS=5 \
MAX_NEW_UNITS=0 ROAD_REPLAY=1 \
bash road_vrp7_step6c1_r3_vq/run_vq.sh collect
```

Smoke不允许执行`train`：只有一个状态/一次重启，无法估计种子间标准差和跨任务泛化。

## 4. 正式Train-only候选价值采集

为了避免初始就执行很大的复评开销，先执行一个分片：

```bash
STUDY_NAME=vq_train48_H20H50_R3_v1 \
STATES=48 SELECT_SEED=20261025 \
HORIZONS=20,50 REPLICATES=3 ROAD_REPLAY=1 \
MAX_NEW_UNITS=1 \
bash road_vrp7_step6c1_r3_vq/run_vq.sh collect
```

检查已完成的分片后续跑：

```bash
STUDY_NAME=vq_train48_H20H50_R3_v1 \
STATES=48 SELECT_SEED=20261025 \
HORIZONS=20,50 REPLICATES=3 ROAD_REPLAY=1 \
RESUME=1 \
bash road_vrp7_step6c1_r3_vq/run_vq.sh collect
```

每个状态最多C3+8个有效候选，理论最多`48×9×2×3=2592`次短程重启与Road.replay，实际会因无效/同C3候选及相同修复路线去重而减少。这个阶段成本主要是事后真实道路回测；该估计是**最多次数**，不是保证的运行时长。

输出目录：

```text
runs/chicago_july_v1/step6c1_r3_vq/
  vq_train48_H20H50_R3_v1/
    manifest.json
    units/<decision_id>__H20__R0.json   # 可续跑的完整单元
    ...
    rollouts.jsonl                     # COMPLETE时才生成
    report.json
```

## 5. 按任务留一离线价值模型消融

数据采集完全完成后再运行：

```bash
STUDY_NAME=vq_loto_seed42_v1 \
COLLECTION_STUDY=vq_train48_H20H50_R3_v1 \
TRAIN_SEED=42 \
bash road_vrp7_step6c1_r3_vq/run_vq.sh train
```

这个环节固定模型超参数，在4个原训练任务之间循环留出1个任务，不接触原Validation与Internal Test任务。比较5种决策：

| 策略名 | 含义 |
|---|---|
| `C3` | 始终回退原始C3，收益=0 |
| `ONE_STEP_ET` | 仅在当折其他训练任务上学习的单步TD-OD收益ExtraTrees（不是旧R2模型工件） |
| `VQ_MEAN` | 学习3个种子Road.replay多步重启收益均值的ExtraTrees |
| `VQ_LCB05` | 预测均值减0.5×预测标准差，只有正分数才覆盖C3 |
| `VQ_LCB10` | 预测均值减1.0×预测标准差，只有正分数才覆盖C3 |

对每个H=20/50分别训练并评价，模型特征均为**仅26维静态F2的三路拼接 `(candidate, C3, candidate-C3)`**，不额外消费真实未来交通。VQ分别训练均值模型与标准差模型（两棵ExtraTrees集成族），且未来部署将需要计入双模型推理开销。本次只做离线选择诊断。

输出`report.json`：每个留出任务与方法的mean gain、state regret、错误覆盖率、覆盖率、Top1，以及任务宏平均；`fold_decisions.jsonl`记录逐状态选择与真实留出Road值。`provenance.json`固定代码、数据与随机种子SHA。

## 6. 解释准则与最重要的局限

1. 如果VQ_MEAN在大多数留出任务的**Road.replay期望收益**上超过ONE_STEP_ET，并降低Regret，可考虑扩大原Train状态的多步标签。否则不要急于部署新网络。
2. 如果LCB05/LCB10降低大额负收益率，但因拒绝更多动作而损失均值，需要报告Pareto权衡，不要只看风险单指标。
3. 若同一状态种子间SD远高于收益均值，3个重复仍可能不足。扩种子应在**Train任务**增加，并保持原Validation/Internal Test封存。
4. 4个留出任务共享原始任务构造/月份且状态相关，属于内部开发级交叉验证，**不代表独立月份/新任务的泛化证据**。
5. 现有重启模拟与完整30秒ALNS不同，代码**尚未集成真实在线策略，也不承诺改善30秒最终道路成本**。若离线模型具有正面信号，再设计新月份上的冻结在线对照。

## 7. 其他工程说明

- 本包新增目录为`road_vrp7_step6c1_r3_vq/`，不覆盖旧R2、R3-T、SV、SV2源码和结果。
- 不依赖旧M0反序列化，所以不会重复出现`No module named models`的问题。
- 若真实状态在Static Repair核对时不一致，**不要跳过一致性检查**；检查原Label创建时的OD参数、3s Repair deadline、当时的源码SHA和机器负载。
- 如需严格判断任何新算法能否优于M0，仍需在最终独立任务与月份执行固定30秒全ALNS配对实验。
