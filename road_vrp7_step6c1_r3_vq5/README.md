# Step6-C1-R3-VQ5 v1 — 固定评价面板的训练状态学习曲线

> 使用最新GitHub冻结VQ3训练器（不复制、不修改模型代码）与VQ4已经完成的192状态Road.replay多步标签。**不重新执行ALNS/Static Repair/道路回测。**

## 0. 背景与为何不能直接信任原VQ4学习曲线

VQ4已经成功采集192个原Train任务状态，生成10,158条动作×重启记录、3,386个状态×搜索长度×动作标签；4种嵌套视图（48/96/144/192）均完成六种子VQ3训练。观察到H=50的PAIR_RANK平均收益从n48的+5.756s降到n192的-9.160s，H=50的RESID_OOF50从+7.403s降到+2.028s。**这里n越大，同时改变了训练状态数和留出评价状态数**；这些曲线不是严格固定评价集合的样本效率比较。

VQ5使用完全一致的VQ4 `vq4_view192_v1/` 作为唯一标签源；仍按原Train的4个任务轮流留出：

- 固定评价面板：在每一折始终使用留出任务48个状态（四折共192状态），不随n改变。
- 可变训练集合：训练总前缀n ∈ {48,96,144,192}。在每一折只使用其余3个任务的n/4个训练状态（实际36/72/108/144个状态）。
- 额外固定锚点：对所有训练规模再评估原先最早的48个状态（每折12个），区分状态难度分布与训练规模的影响。
- 不使用原R2的Validation或Internal Test任务进行模型训练、风险概率校准、阈值学习或外层留一评价。
- 仍使用同一批候选、全部候选的Road.replay后验标签以及真实C3为零的决策语义。

**注意**：`E0_FULL`按照VQ3原定义，训练于除留出任务外的全部一步标签，因此在所有n下其训练数量恒定；它是强参照，不应被叫作随n扩展的模型。E0_MATCHED、VQ_MEAN、PAIR_*、OOF残差和风险模型随n增长。

## 1. 文件与设计约束

| 文件 | 作用 |
|---|---|
| `preflight_vq5.py` | 校验VQ4 n192完整性、原R2标签SHA、四Train任务平衡与嵌套样本前缀。无写入。 |
| `fixed_panel_vq5.py` | 复用原 `train_vq3.build_fold`，训练n48/n96/n144/n192并在相同192+48锚点上评价。 |
| `vq5_core.py` | 训练与评价划分审计、动作级锚点一致性检查、SHA固定。 |
| `summarize_vq5.py` | 六种子/四任务固定面板统计及配对差。拒绝混合数据或代码。 |
| `tests/test_vq5.py` | 5项纯函数、负控及合成真实训练器端到端测试。 |
| `run_vq5.sh` | `test/preflight/train/analyze` 命令入口。 |

VQ5不重写原有VQ3的损失、参数、OOF任务拆分、候选排序、门控逻辑、目标函数、修复器、搜索器和Road.replay。原VQ3 `E0_FULL/E0_MATCHED/VQ_MEAN/RESID_OOF/PAIR_RANK/PAIR_REGRET/E0_RISK`等13策略保持不变。

**强制对照**：在 `n=48` 的原始48状态锚点中，对每个训练种子、每个H和13个策略逐动作比较VQ5选择和原VQ4-n48的`fold_decisions.jsonl`，有任何不一致就停止。

## 2. 安装与基础检查

将ZIP放在仓库根目录，运行：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r3_vq5_v1.zip
bash road_vrp7_step6c1_r3_vq5/run_vq5.sh test
bash road_vrp7_step6c1_r3_vq5/run_vq5.sh preflight
```

需要此前已同步的 `road_vrp7_step6c1_r3_vq3/`、`road_vrp7_step6c1_r3_vq2/`、`road_vrp7_step6c1_r3_vq/`、`road_vrp7_step6c1_r2/`，以及：

- `runs/chicago_july_v1/step6c1_r3_vq4/vq4_view192_v1/`（有manifest/report/rollouts）
- `runs/chicago_july_v1/step6c1_r3_vq4/models/vq4_n48_seed42_v1/` … `seed47_v1/`（有report和fold_decisions）
- `runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1/`（原一步配对标签，不能改动）

若存在不同的根目录，用 `VQ5_ROOT / COLLECTION_DIR / DATASET_DIR / VQ4_MODEL_ROOT` 环境变量覆盖；四个模型评价规模由 `SIZES` 设定，生产实验保持默认。

## 3. 运行Seed42，然后六种子

```bash
TRAIN_SEED=42 TREES=160 \
  bash road_vrp7_step6c1_r3_vq5/run_vq5.sh train
```

输出在：

```
runs/chicago_july_v1/step6c1_r3_vq5/vq5_fixed192_seed42_v1/
  report.json                # 4规模×2H×4留出任务×13策略
  fold_decisions.jsonl       # 按任务留出、固定192状态，同一个评价状态跨n重复
  candidate_predictions.jsonl
  provenance.json            # 原始源SHA + 冻结VQ3代码SHA + VQ5源码SHA
```

确认`anchor_48_verification.status == "MATCH"`后，运行剩余种子：

```bash
for seed in 43 44 45 46 47; do
  TRAIN_SEED=$seed TREES=160 bash road_vrp7_step6c1_r3_vq5/run_vq5.sh train
done
```

## 4. 聚合分析

```bash
bash road_vrp7_step6c1_r3_vq5/run_vq5.sh analyze
```

输出：

```
runs/chicago_july_v1/step6c1_r3_vq5/vq5_fixed192_curve_seed42to47_v1/
  report.json
  report.md
  provenance.json
```

`report.md`为H=20与H=50分别给出：

1. 固定192状态评价面板上，训练n=48/96/144/192的各方法收益。
2. 固定原始48状态锚点上，同样训练规模带来的收益变化。
3. `VQ_MEAN−E0_MATCHED`、`RESID_OOF50−E0_MATCHED`、`RESID_OOF50−E0_FULL`、`PAIR_RANK−E0_MATCHED`、`E0_RISK35−E0_SCORE_COVERAGE35`的任务级配对差与方向。

六训练种子复用同一数据，不能当6次独立实验做显著性推断。只有4个独立的原Train任务，也不能凭漂亮的学习曲线宣称新任务或新月份泛化。

## 5. 如何解读

- **若原VQ4曲线大幅波动，但VQ5固定192面板随n改善**：原实验主要受评价集合组成变化影响，扩大标签可能仍有价值。
- **若固定192面板仍无显著趋势/排序继续退化**：问题不是简单的评价集合变化，建议暂停多步价值模型扩容，转入捕获ALNS内部状态、精确连续搜索/预算感知的序列数据采集。
- **若固定48锚点随训练量改善而192面板没有改善**：模型可能只适用于早期/特定轨迹状态，应检查状态分布而不是盲目扩展训练数据。
- **若E0_RISK35在覆盖率近似时不能胜过E0_SCORE_COVERAGE35**：没有证据说明目前的概率门控具备独立识别损失的优势。

本包不训练在线部署工件；所有Road.replay仍仅用于离线评价。该实验的多步标签是从Repair后路线**重新启动冻结C3**的代理价值，未恢复原30秒ALNS的温度、算子权重或完整RNG状态。

## 6. 测试边界

开发容器通过5项新增测试，包含旧VQ3的动作级锚点复现、固定留出规模审计以及汇总数据混用拒绝。本地没有你的Chicago道路/OD完整数据，**真实服务器端的preflight与训练尚未执行**。

整个研究仍应以固定30秒真实道路成本为最终目标。在VQ5厘清学习曲线后，若多步监督仍缺乏稳定信号，优先进入状态聚合/完整搜索轨迹价值的采集审计，而不是直接加入FiLM或大Transformer。
