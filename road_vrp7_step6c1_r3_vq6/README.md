# Step6-C1-R3-VQ6 v1 — 冻结30秒ALNS在线配对验证

**核心问题**：VQ5在固定192状态Train内部留一的短程C3-restart标签上，`RESID_OOF50`仅较 `E0_MATCHED`略优（H50平均+0.630秒/状态），并没有单调学习曲线。该提升是否可以转换成**固定30秒官方ALNS最终真实Road.replay成本**的改善？

## 0. 数据来源与不可越过的边界

本包只需现有GitHub已完成的 `vq4_train192_H20H50_R3_v1`（192个原Train状态）、原始R2 1024状态标签及Step6-A模型来源；不重新运行H步反事实标签，也不修改任何原始文件。

- 训练任务严格为R2 `task` 分组所标识的4个原Train任务，原Validation和Internal Test的C3标签**不参与训练、模型选择或阈值调参**。
- 固定H=50、seed=42、ExtraTrees 160棵、0秒覆盖门槛、残差系数0.5。选择这些是预先冻结的开发方案，**不能把看过的VQ5曲线当成盲选证据**。
- `E0_MATCHED`：在192状态配对候选上训练的一步TD增益ExtraTrees（78维）；参考VQ3等样本一阶基线。
- `RESID_OOF50`：原Train所有一步标签训练基础ExtraTrees，原Train 192状态H=50搜索重启Road增益监督产生按任务OOF残差模型；在线预测为 `one_step_full + 0.5*pred_residual`。
- `DEFER_RESID`：**仍计算与RESID相同的所有特征和两模型预测**，但总是使用真实C3的克隆RNG动作及对应原RNG转移。用于评估模型评分开销对30秒最终轨迹的影响。不是形式化的算法安全对照。
- `C3`：完全冻结原官方C3_topk（无模型评分）。
- 实时控制使用已有 `road_vrp7_step6c1_r2/online_r2.py` 的 `R2Section` 与 **real-C3 shadow**，没有改动原ALNS destroy/repair/RNG/退火或路网目标。
- 在线模型输入仅有出发前可用的26维F2候选特征及真实C3对应特征。真实Road.replay用于**搜索结束后**评价，不作为在线输入。
- **固定30秒**包括C3影子动作、静态特征构造和模型评分；另外存在20000迭代上限。严格报告有效覆盖、模型调用、搜索迭代与运行时间。
- 默认评估七月22日及任务0008/0009，属于已建立的开发任务/日期，不是新月份盲测试。可后续另设新月份和新任务，对冻结策略进行正式外部检验。

## 1. 安装和测试

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r3_vq6_v1.zip
bash road_vrp7_step6c1_r3_vq6/run_vq6.sh test
```

依赖现有仓库目录：`road_vrp7_step6c1_r3_vq/`、`vq2/`、`vq3/`、`vq4/`、`step6c1_r2/`、`step6b2_r/`、`step6b2/`、`step6a/`、`step5b_v3/`、`stage3/`；以及原先已安装的 `alns==7.0.0`、`scikit-learn`、`numpy`、`joblib`、OD及道路回测数据。

## 2. 训练冻结模型（一次）

```bash
STUDY_NAME=vq6_train192_H50_seed42_v1 \
TRAIN_SEED=42 \
bash road_vrp7_step6c1_r3_vq6/run_vq6.sh train
```

生成 `runs/chicago_july_v1/step6c1_r3_vq6/vq6_train192_H50_seed42_v1/` 内三个独立 `joblib` 文件、训练日志report、provenance以及原始R2/多步采集SHA。

## 3. 开发Pilot：四个臂独立固定30秒搜索

明确指定**相同的** `TASK_IDS`、`SCENARIO_IDS`、`SEEDS`、搜索预算与采样种子。下述为2任务 × 2场景 × 2搜索种子 × 4臂，**共32次真实30秒搜索**。

```bash
export TRAIN_STUDY=vq6_train192_H50_seed42_v1
export TASK_IDS=task_0008,task_0009
export SCENARIO_IDS=2025-07-22_0800,2025-07-22_1700
export SEEDS=60,61

for arm in C3 DEFER_RESID E0_MATCHED RESID_OOF50; do
  ARM="$arm" STUDY_NAME="vq6_pilot_${arm}_v1" \
    bash road_vrp7_step6c1_r3_vq6/run_vq6.sh online
done

C3_STUDY=vq6_pilot_C3_v1 \
DEFER_STUDY=vq6_pilot_DEFER_RESID_v1 \
MATCHED_STUDY=vq6_pilot_E0_MATCHED_v1 \
RESID_STUDY=vq6_pilot_RESID_OOF50_v1 \
STUDY_NAME=vq6_pilot_compare_v1 \
  bash road_vrp7_step6c1_r3_vq6/run_vq6.sh compare
```

对照会拒绝：少运行、重复任务/场景/种子、不同模型来源/道路/求解器/搜索预算、数据不匹配、错误臂身份、没有发生真实模型评分或DEFER意外覆盖C3。

若Pilot没有错误，可使用**新的独立结果目录**进行扩大开发验证，例如：

```bash
export TASK_IDS=task_0008,task_0009
export SCENARIO_IDS=2025-07-22_0800,2025-07-22_1700,2025-07-23_0800,2025-07-23_1700,2025-07-24_0800,2025-07-24_1700
export SEEDS=60,61,62,63,64
for arm in C3 DEFER_RESID E0_MATCHED RESID_OOF50; do
  ARM="$arm" STUDY_NAME="vq6_dev6dates_${arm}_v1" \
    bash road_vrp7_step6c1_r3_vq6/run_vq6.sh online
done
C3_STUDY=vq6_dev6dates_C3_v1 DEFER_STUDY=vq6_dev6dates_DEFER_RESID_v1 \
MATCHED_STUDY=vq6_dev6dates_E0_MATCHED_v1 RESID_STUDY=vq6_dev6dates_RESID_OOF50_v1 \
STUDY_NAME=vq6_dev6dates_compare_v1 \
  bash road_vrp7_step6c1_r3_vq6/run_vq6.sh compare
```

**更严格的性能结论需要冻结选择后改用新任务/新月份。** 已查阅的七月任务配置含7月22–25四天Validation (0800/1700)；这里特意没有复用之前经常检查的7月25日场景。

## 4. 如何解读结果

主指标：最终 `ex_post_road_travel_sec`；次指标：`iterations`、模型覆盖次数、`r_scoring_seconds`、模型调用次数、ROAD成本尾部分布。每个比较严格配对 `(task_id,scenario_id,solver_seed)`。

1. `DEFER_RESID vs C3`：相同初始种子、真实C3动作但含评分开销。**若差距较大，先解决推理成本或wall-clock敏感性**。模型不采取动作并不意味着完整ALNS轨迹相同，因为墙钟停止时刻/迭代数量可能不同。
2. `RESID_OOF50 vs E0_MATCHED`：同一在线适配器/预算/输入特征，直接测试多步残差价值。
3. `RESID_OOF50 vs DEFER_RESID`：考虑计算开销的真实在线策略介入相对强制不介入对照。
4. `E0_MATCHED vs C3`：确认一步模型独立于此前R2/T2策略的在线效用。

**失败退出规则（开发建议，不是统计显著性检验）**：若RESID相比E0多任务/场景持续非正，或所有正收益都来自极少数单一场景，则暂停多步残差扩展，不要直接堆叠交通Transformer/对比学习。优先走完全ALNS状态/RNG/退火权重捕获与真正连续后续价值建模。

**不可声称**：这些2任务开发Pilot意味着泛化到未来月份；多种子是独立任务样本；多步restart标签是在线30秒真实Q值；风险模型有可证明安全性。

## 5. 文件与实现

- `train_vq6.py`：载入VQ4冻结192状态和真实R2 Train split，按原VQ3配对ExtraTrees与OOF残差训练定义拟合两模型。
- `vq6_policy.py`：严格F2 26→78维候选/C3差分特征，`R2Advantage.select`原接口及强制回退。
- `online_vq6.py`：临时只替换模型工件检查，复用原R2真实C3-shadow和30秒基准执行器。
- `compare_vq6.py`：严格配对Road结果、效果与开销及任务/场景分组输出。
- `run_vq6.sh`：统一执行入口。
- `tests/test_vq6.py`：8项单元/合成测试。

不做服务器推送；上传ZIP后需在服务器测试和跑真实Chicago数据。本包不支持继续往已有的在线目录追加，保持结果不可变与SHA追踪。
