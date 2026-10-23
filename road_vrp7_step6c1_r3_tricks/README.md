# Step6-C1-R3-T1：第一批轻量性能 Tricks（真实 C3 感知 Section 选择）

**实现状态：已完成Python源码、合成集成测试、独立在线适配、续跑及比较脚本；未在本地执行完整Chicago真实数据训练/30秒在线搜索。**

## 1. 实验目标与方法

继承冻结的Step6-C1-R2：一次C3搜索状态包含真实C3动作（baseline）、分层采样8个候选及同一Static Repair后的相对C3有符号TD收益。保持原ALNS代码、真实C3 Top20%随机选择、R2影子随机数与30s/20,000迭代限制。优化指标：

- **离线**：Regret、含C3动作的Top1Hit、Pairwise Accuracy、实际覆盖率、大损失覆盖比例、相对真实C3的每状态有符号收益；
- **在线**：`Road.replay.ex_post_road_travel_sec`（越低越好）、原C3/M0/原R2-F2相同任务×场景×求解种子配对；另外统计`search_wall_sec`、`iterations`、模型评分时间与真实C3覆盖次数。

第一批只用F2结构特征（26维），**不修改交通F4**；因为前120例没有证实F4及Shuffle正确对应的净价值。这不是放弃多模态，而是先修复Section评分及决策目标。

|ID|方法|说明|
|---|---|---|
|T0_R2_F2|冻结原R2-F2|原`road_vrp7_step6c1_r2/r2_task_seed42_v1/R2_F2.joblib`，不重训|
|T1_CB_RMSE|CatBoost RMSE|360轮、深度5、lr .04、L2 6|
|T2_CB_HUBER|CatBoost Huber|`Huber:delta=40`，其他参数同T1|
|T3_PAIRWISE|RankNet+回归辅助|相同状态8候选+C3零收益锚点，9候选成对排序|
|T4_WEIGHTED_PAIR|Regret-weighted RankNet|大收益差权重0.5–3.0，**每状态**标准化pair损失|
|T5_HUBER_MARGIN|T2+Margin|固定T2已选的验证集阈值，验证集选择Margin|
|T6_WEIGHTED_MARGIN|T4+Margin|固定T4已选阈值，验证集选择Margin|
|T7_BLEND_MARGIN|T2/T4平均融合+Margin|按0.5/0.5固定融合，验证集选择阈值和Margin，独立探索性消融|

T3/T4网络架构：`78→64→32→1`，训练时用原状态的C3候选作为第9个动作，把C3评分明确设为0；Pairwise logistic loss加小比例Huber收益回归辅助，用于保留分数与0的关系。**纯排序分数不自动等于收益秒数**，由辅助损失改善但不保证精确校准。最终仍需在线验证。

训练/验证/内部测试按R2任务划分：`0000–0003`/`0004–0005`/`0006–0007`。训练只能读取train折标签；阈值和Margin只允许读取validation折；internal_test只写入开发诊断报告，不参与选择。

### 注意：与前次实验的区别

之前R2/R2-B在2025-07-22到24日开展过120例开发测试，这些数据不再用于R3 Tricks选择。默认在线采用**2025-07-25**的08:00和17:00两场景，求解随机种子60–64（与此前42/43不同）。但是该日期此前也有8例Pilot，因此依然是**开发测试，绝非独立盲测**。最终冻结算法后应另用不参与训练/选模的任务和月份验证。

## 2. 环境及安装

服务器已有`road_vrp_7`及`road_vrp7_step6c1_r2/`冻结代码、Chicago数据、R2已标注的`paired.npz`：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r3_tricks_v1.zip
python -c 'import numpy, sklearn, catboost, torch, joblib; print("R3 dependencies OK")'
# 缺少时，在实验环境中按需安装：
# python -m pip install catboost torch scikit-learn joblib numpy
bash road_vrp7_step6c1_r3_tricks/run_r3.sh test
```

`test`包括新R3合成测试和原R2单元测试，不使用真实成本张量。不要覆盖冻结R2/R2-B历史结果。

## 3. 离线训练：固定六种子，仅作为训练随机性检查

先执行一个新的试跑目录：

```bash
TRAIN_SEED=42 EPOCHS=100 STUDY_NAME=r3_seed42_v1 \
  bash road_vrp7_step6c1_r3_tricks/run_r3.sh train
```

生成`T1..T7.joblib`、`report.json`、`manifest.json`、`summary.json`。每个`report.json`包含验证集上的门控网格，以及train/validation/internal_test的完整决策指标。**T0冻结原R2，原模型路径不替换。**

六个训练随机种子：

```bash
for s in 43 44 45 46 47; do
  TRAIN_SEED=$s EPOCHS=100 STUDY_NAME=r3_seed${s}_v1 \
    bash road_vrp7_step6c1_r3_tricks/run_r3.sh train
done
STUDY_NAME=r3_aggregate_seed42to47_v1 \
  bash road_vrp7_step6c1_r3_tricks/run_r3.sh aggregate
```

聚合以模型训练随机性为维度，不把同一内部测试数据在6种子上的重复计算伪装成独立样本。**禁止根据internal_test表现挑选上线算法。** 可以先以固定seed42作为预先指定的上线对照；其他种子用于稳定性说明。

## 4. 独立20例在线开发Pilot（30秒/条；先1个分片Smoke）

每种方法：任务`task_0008,task_0009`×2025-07-25 08:00/17:00×随机种子`60,61,62,63,64` = 20条，拆分成4个独立任务×场景分片，每片5个种子。在线调用：

- `C3/M0`调用原`road_vrp7_step6b2_r/benchmark_30s_r.py`；M0保留原cache64，C3原Top20%随机动作；
- `T0`调用冻结`road_vrp7_step6c1_r2/online_r2.py`；
- `T1..T7`调用新的`online_r3.py`，内部调用**同一个冻结R2Section**；
- C3、M0、R2、R3的Search预算/候选采样沿用各自原有协议；由于M0与R2/R3候选采样不同，该比较为**端到端基线**，不是纯推理时间隔离试验。

先对关键组各执行1个分片检查：

```bash
for arm in C3 M0 T0_R2_F2 T2_CB_HUBER T4_WEIGHTED_PAIR T5_HUBER_MARGIN T6_WEIGHTED_MARGIN T7_BLEND_MARGIN; do
  ARM=$arm STUDY_NAME=r3_online_${arm}_v1 \
  MODEL_STUDY=r3_seed42_v1 MAX_NEW_SHARDS=1 \
  bash road_vrp7_step6c1_r3_tricks/run_r3.sh online
done
```

确认无异常后续跑各组剩余分片：

```bash
for arm in C3 M0 T0_R2_F2 T2_CB_HUBER T4_WEIGHTED_PAIR T5_HUBER_MARGIN T6_WEIGHTED_MARGIN T7_BLEND_MARGIN; do
  ARM=$arm STUDY_NAME=r3_online_${arm}_v1 \
  MODEL_STUDY=r3_seed42_v1 RESUME=1 \
  bash road_vrp7_step6c1_r3_tricks/run_r3.sh online
done
```

如果离线报告显示T1或T3更有价值，可以独立增加：

```bash
for arm in T1_CB_RMSE T3_PAIRWISE; do
  ARM=$arm STUDY_NAME=r3_online_${arm}_v1 \
  MODEL_STUDY=r3_seed42_v1 \
  bash road_vrp7_step6c1_r3_tricks/run_r3.sh online
done
```

## 5. 自动比较

如果执行全部T1..T7，直接：

```bash
STUDY_NAME=r3_compare_all_v1 bash road_vrp7_step6c1_r3_tricks/run_r3.sh compare
```

如果只执行上述6个新变体：

```bash
COMPARE_ARMS=T0_R2_F2,T2_CB_HUBER,T4_WEIGHTED_PAIR,T5_HUBER_MARGIN,T6_WEIGHTED_MARGIN,T7_BLEND_MARGIN \
STUDY_NAME=r3_compare_pilot_v1 \
  bash road_vrp7_step6c1_r3_tricks/run_r3.sh compare
```

产物`report.json`和`report.md`，包含每个候选对C3及M0的平均改善、胜/负/平、每任务×场景簇Bootstrap区间。两个任务和两个场景只有4个簇，因此Bootstrap只能描述开发阶段不确定性，**不能用于宣称显著泛化**。

## 6. 断点与SHA管理

- `STUDY_NAME`必须是**新的独立名称**；每次脚本保留模型/代码SHA及评估条件。历史目录不得覆盖。
- 对已完成的分片，使用`RESUME=1`自动跳过；若分片存在但缺少`COMPLETE`则**显式报错**，请先检查日志与文件，确认后手动处理，不做自动覆盖。
- `study_r3.py`为每个分片校验模型与冻结ALNS的SHA，并在全部完成时调用本包的`r3_merge.py`严格合并。
- `online_r3.py`加载模型时验证`r3_models.py`、`decision_metrics.py`、`train_r3.py`与训练工件的SHA；修改源码后必须使用新的训练目录重新训练模型。
- 统计报告不参与阈值调节；Road.replay未来真实成本**只用于在线评价，不进入任何训练输入**。

## 7. 当前边界与预期结果解读

上述系统能检验CatBoost/RankNet/Margin在**一步C3相对优势**监督下改善离线动作选择的能力；但不能保证改善完整30秒ALNS、也不能替代多步搜索价值标注和DAgger状态聚合。若离线决策指标提高但20例在线道路成本不改善，则应优先转到R3多步搜索价值和在线轨迹偏移诊断，而不是不断增加网络宽度、深度或技巧数量。

### 文件说明

- `r3_models.py`：CatBoost与RankNet及Margin、固定比例融合；
- `decision_metrics.py`：状态regret、Top1、pairwise准确率、尾部负收益、门控网格；
- `train_r3.py`：分组不泄漏、模型保存和可复现SHA；
- `online_r3.py`：重用冻结R2在线C3-shadow选择器；
- `study_r3.py`：真实30s在线分片续跑；
- `r3_merge.py` / `r3_stats.py` / `compare_r3.py`：合并及严格配对报告；
- `aggregate_r3.py`：多训练种子离线稳定性摘要；
- `run_r3.sh`：统一入口；
- `tests/`：合成数据、在线加载兼容、配对数据及训练全流程测试；
- `MANIFEST.json`：每个文件的SHA-256。
