# Step6-C1-R3-VQ2 v1：多步价值公平消融、OOF残差与风险尺度诊断

> **状态**：完成新代码与合成测试；没有在本地执行Chicago真实路网数据。需要在用户服务器运行。不会改动冻结ALNS、原R2/VQ标签、任何旧实验目录。

## 0. 本轮为何开发

R3-VQ v1 已完成Train-only 48状态、H=20/50、每状态3个重启，得到425个 `(状态,候选)` 标签/搜索长度（含C3零增益）；共2,550条候选重启记录。LOTO发现 H=20：ONE_STEP_ET +5.685，VQ_MEAN -4.709，LCB10 0；H=50：ONE_STEP_ET +3.124，VQ_MEAN -8.252，LCB10 0，均为留一任务的Road.replay多步重启代理收益（秒/状态）。

**重大混杂**：ONE_STEP_ET每折用了3014–3026个单步候选样本，VQ_MEAN只有282–285个多步候选样本；所以不能据此判定多步监督本身无效。同时三重启种子的收益标准差常在100秒左右，原`mu-0.5*sd`门控过度回退。

这轮不增加成本昂贵的短程搜索或新标签，直接复用已有VQ `rollouts.jsonl` 与R2 `paired.npz`，只重新训练轻量ExtraTrees离线模型，分开检验**样本数量影响、监督目标影响、稳定残差修正、风险惩罚尺度**。

## 1. 完整对照

固定输入：F2的26维静态Section特征，构造`[candidate, actual_C3, difference]`，共78维。所有ExtraTrees使用与旧VQ一致的超参数、默认每个候选统一样本权重；`E0_FULL`的模型随机种子与旧`ONE_STEP_ET`一致，`E0_MATCHED`和`VQ_MEAN`使用相同模型随机种子以减少额外混杂。每状态只允许原8候选中valid且非同C3的动作；第9个动作为**真实C3回退**（相对收益0）。

| 名称 | 每折训练数据 | 预测分数与目的 |
|---|---|---|
| C3 | 无 | 永远选择原C3 |
| E0_FULL | 原Train其他3任务的所有一步有效候选（约3000） | 一步ExtraTrees：现有数据量参考 |
| E0_MATCHED | 与VQ完全相同的36状态/约282–285候选 | 严格等样本一步ExtraTrees |
| VQ_MEAN | 上述相同候选 | 直接预测H步C3重启平均Road收益 |
| BLEND25 | E0_FULL + VQ_MEAN | `0.75*step_full + 0.25*vq_mean` |
| BLEND50 | E0_FULL + VQ_MEAN | `0.5*step_full + 0.5*vq_mean` |
| RESID_OOF25 | E0_FULL + 额外残差树 | `step_full + 0.25*residual_pred` |
| RESID_OOF50 | E0_FULL + 额外残差树 | `step_full + 0.5*residual_pred` |
| RESID_OOF100 | E0_FULL + 额外残差树 | `step_full + 1.0*residual_pred` |
| VQ_LCB10 | VQ_MEAN + SD树 | `vq_mean - 0.10*predicted_restart_sd` |
| VQ_LCB25 | VQ_MEAN + SD树 | `vq_mean - 0.25*predicted_restart_sd` |

所有方案对真实C3采用统一门槛 `score>0`，非正值直接回退；权重0.25、0.5、1和风险系数0.10、0.25提前固定，**绝不依据外层留出任务或原Validation/Internal Test标签挑选最优系数**。

### OOF残差为何必须做内层任务隔离

4个原Train任务：task_0000…0003。每个外层LOTO留出1任务评价，剩余3任务训练。对3个训练任务的每一个，单独从**另外2个**任务的一步数据训练内层模型，预测当前任务的已选候选；拟合残差标签：

`residual_target = H_step_mean_gain - inner_task_out_of_fold_one_step_prediction`

之后使用全部3个外层训练任务训练E0_FULL与残差树，外层留出任务得到：

`final_score = E0_FULL_pred + alpha * residual_tree_pred`

这避免了让残差树拟合基模型在自身训练样本上的过拟合残差。需注意：内层基模型只用2任务，外层基模型用3任务，其预测分布存在变化，结果仍需验证。

## 2. 输入要求

GitHub根目录下必须已经包含以下源码和数据（均沿用旧结果）：

- `road_vrp7_step6c1_r3_vq/{train_vq.py,vq_core.py}`；
- `road_vrp7_step6c1_r2/r2_policy.py`；
- `runs/chicago_july_v1/step6c1_r3_vq/vq_train48_H20H50_R3_v1/{manifest.json,report.json,rollouts.jsonl}`；
- `runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1/{manifest.json,paired.npz,groups.jsonl}`。

VQ2调用原`train_vq.load()`：原标签SHA和分组SHA必须匹配、收集必须COMPLETE、每个选中状态必须属于原Train任务、候选及每个重复种子必须完整。若缺文件/版本错误，将显式报错。

## 3. 一键运行

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r3_vq2_v1.zip

# 运行7项新增合成单元/集成测试，附语法编译
bash road_vrp7_step6c1_r3_vq2/run_vq2.sh test

# Seed42，使用独立目录，不覆盖前次VQ
TRAIN_SEED=42 TREES=160 STUDY_NAME=vq2_loto_seed42_v1 \
  bash road_vrp7_step6c1_r3_vq2/run_vq2.sh train
```

接着固定六种训练种子进行敏感性检查（仅训练随机性变化，数据和留出任务相同）：

```bash
for s in 43 44 45 46 47; do
  TRAIN_SEED=$s TREES=160 STUDY_NAME=vq2_loto_seed${s}_v1 \
    bash road_vrp7_step6c1_r3_vq2/run_vq2.sh train
done

INPUT_STUDIES=vq2_loto_seed42_v1,vq2_loto_seed43_v1,vq2_loto_seed44_v1,vq2_loto_seed45_v1,vq2_loto_seed46_v1,vq2_loto_seed47_v1 \
STUDY_NAME=vq2_aggregate_seed42to47_v1 \
  bash road_vrp7_step6c1_r3_vq2/run_vq2.sh aggregate
```

默认输出：

```
runs/chicago_july_v1/step6c1_r3_vq2/
  vq2_loto_seed42_v1/
    report.json                  # 2个H、4个留出Train任务、11种策略
    report.md                    # 人可读性能表
    fold_decisions.jsonl         # 48个状态*2 H*11策略 = 1056行
    candidate_predictions.jsonl  # 每有效候选的预测与实际mu、sd
    provenance.json              # VQ原标签、rollouts、源码SHA
  vq2_aggregate_seed42to47_v1/
    report.json                  # 6个训练种子的描述性稳定性，不是独立测试
    provenance.json
```

如需自定义数据路径，使用`DATASET_DIR=... COLLECTION_ROOT=... COLLECTION_STUDY=... OUTPUT_ROOT=...`环境变量。所有研究目录必须是新的；不支持覆盖已生成结果。

## 4. 主要报告指标和判断门槛

- **主指标**：按任务宏平均的`mean_gain_sec`（Road.replay重启代理收益，相对C3，越大越好）；其次为`mean_regret_sec`和`harm_gt40_per_state`。
- **直接检验目标是否有价值**：`VQ_MEAN − E0_MATCHED`，两者训练候选/状态/统一样本权重完全相同。
- **诊断数据量差异**：`E0_FULL − E0_MATCHED`。
- **残差是否提供增量**：`RESID_OOF25 / 50 / 100 − E0_FULL`，尤其检查4任务的增量方向是否一致。
- **风险是否有效**：`VQ_LCB10 / 25`相对于VQ_MEAN的覆盖率、每次覆盖严重损失比例和平均收益；拒绝多但完全回退不能算有效创新。
- **校准诊断**：均值预测的MAE/RMSE/偏差、预测重启SD的MAE、真实SD大于绝对收益幅度的比例、预测误差与SD比较。注意此处SD是**搜索随机性**，不是预测区间/置信上界。

### 结果解释分支

1. E0_MATCHED与VQ都明显差于E0_FULL：先考虑从原Train剩余状态增加多步标注；**不要先上复杂GNN**。
2. VQ在等样本比较中胜过E0_MATCHED但仍不如E0_FULL：多步标签有潜在增量，优先试OOF残差或固定比例融合。
3. 残差在多数留出Train任务改善且不增加严重损失：考虑再扩大原Train状态，而不是马上在原Internal Test调参。
4. LCB在低覆盖状态仍出现高误伤：后续开发风险分类或Selective Gate，但必须从Train-only交叉拟合/校准开始。
5. 全部无稳定增量：优先研究真实在线ALNS状态分布和原搜索轨迹价值，而非继续拟合C3重启代理。

## 5. 研究边界与风险

- 当前LOTO只是**4个原Train任务**：每折留出仅12状态，同任务状态相关，不能声称显著泛化。
- VQ是从Section静态修复后的路线 **重新启动冻结C3** 所得的多步Road评价，不是精确续接原30秒ALNS算子权重、退火进度及RNG的Q值。
- 模型训练和选择不使用原Validation / Internal Test标签；此前已分析的SV2 Internal Test不会再次用于训练或调参。
- 没有执行新30秒在线实验；若离线结果值得继续，应在**未参与调参的新日期和新任务**进行配对30秒ALNS验证。
- 多个训练种子复用同一批任务和状态，不能作为独立样本计算显著性。
- 不用传统Bootstrap将同一任务的8候选或3个重启种子当作独立任务。

## 6. 源码说明

- `train_vq2.py`：原标签验证、严格Train-only四折、OOF残差模型、逐状态决策；
- `vq2_core.py`：ExtraTrees、严格相同候选样本、指标和校准；
- `report_vq2.py`：生成可读Markdown消融汇总；
- `aggregate_vq2.py`：六训练种子的描述性汇总与来源一致性检查；
- `run_vq2.sh`：统一`test/train/aggregate`命令；
- `tests/test_vq2.py`：合成集成、与原VQ模型预测一致性、OOF排除自身任务、回退/掩码/错误收益与覆盖统计；
- `MANIFEST.json`：ZIP内源码的SHA-256。
