# Step6-B1 v2 修复版：部署、设计和复现说明

适用仓库：`https://github.com/Dragon1712/road_vrp_7`，审计源码参考提交 `601bb448e11112fea8056df2f09c62237846a75c`。

**本压缩包新增目录 `road_vrp7_step6b1_v2/`，不修改原来的Step6-B1 v1、Step6-A v2、Step5。**

## 1. 修改了什么？

| 缺陷/风险 | 修正 |
|---|---|
| P3/P6线性差分排序无法表征状态时间与Section属性交互 | **新增**P9/P10显式状态×Section交互；保留P3/P6作为原始线性对照 |
| 原Pairwise排序分数按全部候选计算全局PR-AUC | 不再输出该误导性指标，排序看**状态内Spearman、Top1命中**，另输出门控概率PR-AUC |
| P4/P7未经校准的混合收益概率 | **新增**P11/P12：只在训练集内部采用按任务或日期分组的OOF预测完成Platt概率校准；P4/P7不变便于消融 |
| Oracle Repair收益与Static Repair收益混淆 | 主标签固定`static_repair_td_gain_sec`；Oracle Repair收益仅供离线上界；策略实现不读取真实收益决策 |
| 混淆对所有状态和仅执行状态求平均 | 分开输出`mean_oracle_repair_upgrade_all_states_sec`和`mean_oracle_repair_upgrade_executed_only_sec` |
| 训练/验证/测试元数据不够严格 | 强制Step6-A v2 `run_config.json`/`benchmark_summary.json`存在且COMPLETE，检查输入Schema、标签恒等式、状态ID与不同组划分 |
| 来源静态解在不同日期重复，容易误解为独立结构 | 报告`route_reuse`，显示跨日期重复路线数量（来自原`state_records.jsonl`，不进入模型） |
| 不保存可复用训练模型 | 每个方法单独保存joblib及字段顺序/源码/输入SHA；注意只允许加载自己生成的可信模型文件 |
| 多种子聚合无防重 | 验证来源和实现SHA、训练种子、测试状态/任务/日期/Oracle对照的一致性，拒绝重复ID |

v2中为增强验证阈值的稳健性，除了旧版固定候选阈值之外，还增加了只从**validation**分数计算出的若干分位数。故v2的P0～P8即使网络主体不变，*阈值选择协议仍与v1略有不同*，论文中应将v1与v2结果区分。对P3/P6，v2保留原始按收益差异平方根加权的线性Pairwise训练；P9/P10才引入新的状态均衡权重及显式交互。

## 2. 安装

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6b1_v2.zip
python -m pip install -r road_vrp7_step6b1_v2/requirements.txt
bash road_vrp7_step6b1_v2/run_b1_v2.sh test
```

当前合成测试共有8项，已全部通过。**在聊天环境未运行完整芝加哥数据训练，必须先进行服务器Smoke。**

## 3. 推荐服务器执行顺序

### A. 独立标签审计（不训练）

```bash
STUDY_NAME=b1v2_label_audit_v1 \
  bash road_vrp7_step6b1_v2/run_b1_v2.sh audit
```

输出 `runs/chicago_july_v1/step6b1_v2/b1v2_label_audit_v1/audit.json`，应核验：`states=472`、`candidates=3776`、Static Repair/TD Oracle标签差异及跨日期重复结构。**不要假定Static Repair的正收益率等于原A-v2 Oracle Repair的19.52%。**

### B. 真实Smoke（三种基础方法 + 交互/校准模型）

```bash
STUDY_NAME=b1v2_smoke_v1 \
  bash road_vrp7_step6b1_v2/run_b1_v2.sh smoke
```

### C. 全部13种方法/两种分组协议的正式实验

```bash
STUDY_NAME=b1v2_seed42_full_v1 \
  bash road_vrp7_step6b1_v2/run_b1_v2.sh train 2>&1 | tee b1v2_seed42.log
```

多种子（保持输入数据、特征方案、模型与验证协议相同）：

```bash
for seed in 43 44 45 46 47; do
  SEED="$seed" STUDY_NAME="b1v2_seed${seed}_full_v1" \
    bash road_vrp7_step6b1_v2/run_b1_v2.sh train
 done
```

所有结果写入 `runs/chicago_july_v1/step6b1_v2/<STUDY_NAME>/`，绝不追加历史结果；目录已存在时会拒绝覆盖。若只想先检验交互与概率校准：

```bash
PROTOCOL=task TREES=120 \
METHODS=P1_class_static_time,P3_pairwise_static_time,P9_pairwise_interaction_static,P6_pairwise_history,P10_pairwise_interaction_history,P4_two_part_static_time,P11_two_part_calibrated_static \
STUDY_NAME=b1v2_priority_task_v1 bash road_vrp7_step6b1_v2/run_b1_v2.sh train
```

### D. 多种子成对比较

```bash
python road_vrp7_step6b1_v2/aggregate.py \
  --studies runs/chicago_july_v1/step6b1_v2/b1v2_seed{42,43,44,45,46,47}_full_v1 \
  --base P1_class_static_time --treatment P8_class_history \
  --output runs/chicago_july_v1/step6b1_v2/paired_P8_vs_P1_v2.json
```

也可换为P9 vs P3、P10 vs P6、P11 vs P4、P12 vs P7。对同一测试状态计算配对差异，不把训练种子当作独立任务重复进行显著性检验。

## 4. 方法与目标

主标签：

`static_repair_td_gain_sec = TD(original_state) - TD(static_repaired_state)`。

这意味着Section修复机制本身不依赖未来道路信息，但监督标签仍由**历史Oracle TD目标事后评价**，模型不是在线读取该标签。

所有方法的模型输入均只来自原`features_X.npz`白名单；`groups.csv`仅用于分组，`labels_y.npz`仅作为监督与评价目标。

| 方法 | 特征 | 训练方法 |
|---|---|---|
| P0 | 10维纯结构 | RF分类 |
| P1 | 12维结构+时间 | RF分类 |
| P2 | 12维结构+时间 | RF有符号收益回归 |
| P3 | 12维结构+时间 | 原线性Pairwise + 分类门控 |
| P4 | 12维结构+时间 | 原未校准Two-part |
| P5 | 27维加历史交通 | RF收益回归 |
| P6 | 27维 | 原线性Pairwise + 分类门控 |
| P7 | 27维 | 原未校准Two-part |
| P8 | 27维 | RF分类 |
| **P9** | 12维+交互展开 | **Interaction Pairwise** |
| **P10** | 27维+交互展开 | **Traffic Interaction Pairwise** |
| **P11** | 12维 | **分组OOF校准Two-part** |
| **P12** | 27维 | **分组OOF校准Two-part** |

核心比较：P1 vs P8、P2 vs P5、P3 vs P6、P4 vs P7（交通增益）；P3 vs P9、P6 vs P10（状态交互增益）；P4 vs P11、P7 vs P12（概率校准增益）。

## 5. 输出与审计边界

```text
runs/chicago_july_v1/step6b1_v2/b1v2_seed42_full_v1/
├── manifest.json                     # 原数据、全部输入文件和当前源码SHA256
├── static_repair_audit.json          # Static/Oracle Repair标签诊断、跨日期路线复用
├── task/
│   ├── report.json                   # 按任务内部测试指标
│   ├── P*_model.joblib               # 完整模型+特征字段顺序；仅加载可信文件
│   ├── P*_test_states.csv            # 模型单状态动作/no-op/有符号收益
│   ├── P*_test_candidates.csv        # 逐候选预测得分和评价标签
│   └── P*_validation_states.csv
├── date/                              # 日期协议同上
└── summary.json
```

所有结果都是**冻结Section候选集合上的单步离线反事实评价**，不是30秒ALNS端到端收益、不是车辆实际道路轨迹`Road.replay()`、不是未见月份独立泛化证据。还没有处理在线模型推理、候选特征更新、Top-K Repair真实耗时；完成B1后才能开展Step6-B2上线实验。

已有4天（7月22–25日）的训练/验证/内部测试数据反复参与研发，因此不应宣称盲测；若后续发现交通历史增益不稳定，应先补充候选目标插入位置的道路动态特征，不应直接宣称SPIN/SPCLT有效。
