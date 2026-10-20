# Step6-B1：可执行 Static Repair 标签、收益回归、状态内排序

本目录作为 `Dragon1712/road_vrp_7` 仓库的**独立新增模块**，不改动、不会重跑 Step5、Step6-A 或 Step6-B0。已基于仓库提交 `4049ed98e12cce686e16af13e7dd8a6b2d93ca83` 的字段、输出协议适配，**但尚未在真实芝加哥服务器上运行**。本地完成合成数据端到端测试。

## 1. 先说清楚评价边界

**主标签 `static_repair_td_gain_sec`：**当前解通过 **STATIC Repair**（不读取未来交通）得到候选，然后用历史TD Oracle规划目标作*离线事后评分*。这种修复机制可在部署时调用，但**奖励数值仍是未来Oracle的历史标签**，绝不应成为模型在线输入。

**辅助标签 `td_gain_sec`：**同一Section使用**Oracle TD Repair**得到的收益，只用于上界/反事实诊断，不用于训练、阈值选择或主评价。

所有指标是**冻结状态、同一候选池中的单步离线选择收益**；不是最终 `Road.replay()` 道路回测，也不意味着整个30秒ALNS搜索已经加速。现有 `features_X.npz` 已是27维汇总特征，未来多模态道路编码还需单独采集真实路网、交通与候选插入位置的特征。

## 2. 安装

解压后目录须为仓库根目录的 `road_vrp7_step6b1/`：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6b1_v1.zip
python -m pip install -r road_vrp7_step6b1/requirements.txt
bash road_vrp7_step6b1/run_b1.sh test
```

请先检查Step6-A v2原数据存在：

```
runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1/analysis/
  features_X.npz
  labels_y.npz
  groups.csv
  dataset_schema.json
```

## 3. 先运行Static与Oracle Repair标签审计

```bash
STUDY_NAME=step6b1_label_audit_v1 \
  bash road_vrp7_step6b1/run_b1.sh audit
```

核心文件：`runs/chicago_july_v1/step6b1/step6b1_label_audit_v1/static_repair_audit.json`。

主要看：

- `static_repair_td_positive_rate`：可执行静态Repair后，事后真正改善TD规划成本的Section比例。
- `oracle_td_repair_positive_rate`：原先Oracle TD Repair的正收益率（A-v2为19.52%）。
- `mean_best_static_repair_noop_gain_sec`：每个状态8候选，允许不操作时、静态Repair的历史Oracle上界。**该上界需要评价所有候选，不是在线零成本策略**。
- `mean_best_repair_gap_sec`：替换Oracle Repair带来的候选池最优潜力差距。

## 4. Smoke与正式运行

Smoke：

```bash
STUDY_NAME=step6b1_smoke_v1 bash road_vrp7_step6b1/run_b1.sh smoke
```

正式九种方法对照（两个划分协议）：

```bash
STUDY_NAME=step6b1_current_seed42_v1 \
  bash road_vrp7_step6b1/run_b1.sh train
```

训练种子稳健性测试（不重采集道路数据）：

```bash
for seed in 43 44 45 46 47; do
  SEED="$seed" STUDY_NAME="step6b1_current_seed${seed}_v1" \
    bash road_vrp7_step6b1/run_b1.sh train
done
```

如需先跑轻量单协议：

```bash
PROTOCOL=task TREES=80 \
METHODS=P1_class_static_time,P2_reg_static_time,P3_pairwise_static_time,P8_class_history \
STUDY_NAME=step6b1_task_short_v1 \
  bash road_vrp7_step6b1/run_b1.sh train
```

**输出目录存在时一律拒绝覆盖**，请换 `STUDY_NAME`，不要向之前结果追加。

## 5. 方法定义

| 名称 | 特征 | 模型与选择 |
|---|---|---|
| P0_class_core | 10维道路/Section静态特征，无时间 | RF改善分类，分数最大+验证集执行阈值 |
| P1_class_static_time | 12维静态+小时/星期 | RF改善分类，继承Step6-B0思路但以Static Repair标签重新训练 |
| P2_reg_static_time | 12维 | RF有符号收益回归，选择预测收益最大的Section+验证集阈值 |
| P3_pairwise_static_time | 12维 | 同一状态内构造收益差异成对样本、Logistic线性排序；RF分类概率门控 |
| P4_two_part_static_time | 12维 | 分类概率×正收益幅度 + (1-p)×负收益幅度，两阶段RF |
| P5_reg_history | 27维 | P2 + 历史交通及观测掩码 |
| P6_pairwise_history | 27维 | P3 + 历史交通及观测掩码 |
| P7_two_part_history | 27维 | P4 + 历史交通及观测掩码 |
| P8_class_history | 27维 | P1 + 历史交通及观测掩码 |

其中“成对排序”的训练数据仅来自**训练状态内部**的Section对，不跨状态生成伪比较。每个动作原始TD收益可正可负；模型执行错误操作时，**负收益照计、不进行Oracle事后撤销**。输出包含 `always_noop` 与 `static_score_unconditional`（后者仅为不带风险门控的诊断，非完整ALNS）。

## 6. 划分方式、防泄漏与正式解读

- `task`：6个任务训练、2个任务验证、2个任务内部测试；同任务/状态不会跨划分。
- `date`：7月22～23日训练、24日验证、25日内部测试；同日期/状态不会跨划分；但**相同路线结构**可能跨日期重复，所以不是未见结构验证。
- 所有RF特征仅从`features_X.npz`读取，收益标签仅来自`labels_y.npz`，`groups.csv`只用于分组，ID/种子/任务不作为特征。
- 模型和执行阈值均不能使用内部测试标签，阈值使用固定备选网格、**仅验证集**选择；test保留真实有符号收益。
- 多训练种子仍复用相同的测试任务/日期，不能把它们当作独立实验单位做统计显著性。
- 所有年份/7月数据已被用于开发，无盲测结论。

## 7. 结果查看与论文研究门槛

正式输出：

```
runs/chicago_july_v1/step6b1/step6b1_current_seed42_v1/
  manifest.json
  static_repair_audit.json
  task/
    report.json
    P*_test_candidates.csv
    P*_test_states.csv
    P*_validation_states.csv
  date/
    report.json
    P*_test_candidates.csv
    P*_test_states.csv
    P*_validation_states.csv
  summary.json
```

重点报告：Static Repair正收益样本占比；在任务和日期协议下的**每状态有符号收益**、执行次数、负收益执行次数、PR-AUC、状态内Spearman和Oracle候选池上界。注意 `mean_signed_gain_sec` 是同一状态上的单步改进，不可跨状态累加后当成最终VRP路线成本节省。

学习式“多模态+对比学习”主线的下一关：**交通历史模型是否在不同任务/日期上对照同一架构的无交通版本仍有稳健的额外收益**。如无，应先增强“候选Section移入目标路线”涉及的道路走廊时序特征，再上SPIN或SPCLT，而不是直接堆叠CL。

## 8. 成对方法对比（无需重新训练）

例如：

```bash
python road_vrp7_step6b1/aggregate_b1.py \
  --studies runs/chicago_july_v1/step6b1/step6b1_current_seed{42,43,44,45,46,47}_v1 \
  --base P1_class_static_time \
  --treatment P8_class_history \
  --output runs/chicago_july_v1/step6b1/paired_P8_vs_P1_v1.json
```

将输出同一测试状态的成对差值、正/负/平局次数及按测试任务分组均值。由于`task`协议仅两组held-out task，`date`协议仅一天held-out date，脚本**不声称统计显著性**。

## 9. 独立方法和版权

代码为独立实现的轻量监督学习与成对排序方法，借鉴Learning to Delegate中“学习局部子问题收益”的思路，**不是其论文模型或原实验的复现代码**。未复制没有明确许可授权的Learning to Delegate源文件，也没有合并SPIN/SPCLT源码。验证方式：`bash road_vrp7_step6b1/run_b1.sh test`。
