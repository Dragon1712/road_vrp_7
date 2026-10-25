# R4-L v1：跨任务 / 跨种子 Train-only 保守Section效用验证

> 冻结来源：GitHub `Dragon1712/road_vrp_7` commit `4f70819387a00a2721c9b84ba8dc70fb94d18a3f`，2026-10-25。  
> 实验性质：**探索性离线固定K反事实评价**。它不运行ALNS、不读Validation/Test、没有30秒在线性能结果，也不将已有Train内部留组评价声称为盲测。

## 为什么开发R4-L

* R4-H：72个Train场景快照，51个按完整退火温度区分的精确物理状态；565个备选Section动作，各有K=50/200/500标签。K500正/负/零=18/30/517。
* R4-I：在原4个Train任务留一时，轻量模型无法稳定找到正收益候选：留出task_0002时无委托，留出task_0003在阈值0时6次委托中2次伤害。不能证明在线可泛化。
* R4-J：新种子62、63的128个早期场景快照；512/512固定K精确恢复。20k/30s严格配对为7/16，另9组墙钟提前终止；这**不否定**已逐事件证明的K前缀。
* R4-K最新上传：完整审计`SOURCE_COMPLETE`，128/128场景状态、3,789分支、1,007候选×3K（3,021标签）。K500正/负/零=190/224/593；K200正→K500负动作58。新种子和早期采样增加了已观察正例，但长时域伤害仍较多。

**当前核心问题**：新增Train正例能否让轻量策略从旧种子迁移到新种子，且在*留出整个配送任务*时有正净收益，而不是用带相关性的动作级随机划分夸大效果？

## 本包文件

* `r4l_core.py`：复用原R4-I的决策前Section特征与纯Python Ridge；校验原R4-H来源；SHA绑定R4-K来源；从新增K50分支读取首步Section动作几何；任务/种子留组评估、保守回退与汇总。
* `probe_r4l.py`：完整数据审计、离线模型及独立不可覆盖输出。
* `run_r4l.sh`：`test`、`preflight`、`probe`。
* `tests/test_r4l.py`：7项测试，包括模拟128快照×1007候选完整回放、完整probe输出、分组不泄漏、消融、门控、缺失审计与SHA篡改拒绝。
* `MANIFEST.json`：包文件SHA。

## 来源目录：不得互相替代

原Train标签（种子60/61）：

```text
runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1/
runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture/
runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1/
```

新Train标签（种子62/63）：

```text
runs/chicago_july_v1/step6c1_r4k/r4k_full_128audited_v1/
runs/chicago_july_v1/step6c1_r4h/r4j_train_early_s62s63_v1_cf_full/
runs/chicago_july_v1/step6c1_r4g/r4j_train_early_s62s63_v1_capture/
runs/chicago_july_v1/step6c1_r4g_prefix/r4j_train_early_s62s63_v1_prefix_full/
```

**GitHub完整性提醒**：最新提交已经包含R4-K的`SOURCE_COMPLETE`报告和`audited_labels.jsonl`，但API读取新CF目录时，`r4j_train_early_s62s63_v1_cf_full/report.json`返回404。R4-K报告中记录了`cf_report` SHA，这说明生成报告的环境曾读取该文件；然而**不能从GitHub仓库现状独立验证文件内容**。本实验的`preflight`会要求服务器实际存在该文件，并用R4-K的来源SHA验证，不允许缺失时跳过。建议将该原始汇总报告、完整`labels.jsonl`、`plans.jsonl`和`branch_summary.jsonl`与manifest同步GitHub，以便第三方复现；不要为了通过验证而复制或重建不一致的版本。

本包仅增加`road_vrp7_step6c1_r4l/`，**不修改**既有ALNS/C3/R4-G/H/I/J/K代码及历史实验目录。

## 服务器使用（仓库根目录）

```bash
cd ~/code/WZL/road_vrp_7
unzip -o road_vrp7_step6c1_r4l_v1.zip
bash road_vrp7_step6c1_r4l/run_r4l.sh test
bash road_vrp7_step6c1_r4l/run_r4l.sh preflight
```

若preflight指出缺失`..._cf_full/report.json`，先检查服务器该阶段是否确实已完成；不要直接运行模型或伪造原始报告。

```bash
STUDY_NAME=r4l_seed_transfer_loto_train_v1 \
  bash road_vrp7_step6c1_r4l/run_r4l.sh probe

cat runs/chicago_july_v1/step6c1_r4l/r4l_seed_transfer_loto_train_v1/report.md
```

输出：`report.md`、`report.json`、`fold_predictions.jsonl`、`provenance.json`。新实验名必需，否则不覆盖已有报告。使用环境变量`R4L_OLD_CF_DIR`、`R4L_NEW_CF_DIR`、`R4L_NEW_R4K_DIR`等可指定独立来源路径；详见`run_r4l.sh`。

## 实验设计

### 数据与统计独立性

* 原数据（60/61）565个动作；新数据（62/63）1,007个动作，总共1,572个动作，每个动作有K50/200/500三维Road最佳收益。
* 旧Train共72个场景快照（按完整精度51个状态）；新Train 128个场景快照（完整精度128、按任务/种子/stop_count与已审计状态组成的粗粒度起点64）；场景相关状态不能当IID。
* 学习样本按**粗粒度搜索起点**等权加权：同一task/seed/stop_count/路线SHA/RNG/权重的跨交通场景状态共享总权重（不让近重复场景任意加权）。原R4-I的精确物理状态字段也保留以供追溯。

### 冻结特征（仅决策前）

复用R4-I 15维：迭代阶段、时钟剩余比、当前/最优代理目标差、退火水平、配送路线长度和数量、Section所在车/位置/长度、与原C3 Section的结构距离、候选数及已选的Repair槽等。仅在官方Section选中并知道Repair槽时选择替代Section，绝不能在之前使用事后数据。

对照：完整15维与几何子集（其他维度设零）；实际Road.replay、真实后续轨迹和收益**只用来监督**，不可作为输入。

### 模型与负对照

* K500价值回归：`U = Δ500`。
* 保守多时域价值：`U = min(Δ200, Δ500)`。
* Ridge正则强度100、门槛0/5/15秒预先固定；分数不超过门槛一律回退C3。
* 门控`none`或`stop_count <= 2000`；后者根据此前Train观察确定，不能当成盲测中自动发现。
* 随机打乱训练标签的负对照、始终C3（精确离线增益0）、均匀随机备选的**期望值**及样本内事后Oracle上界。
* 一次状态至多委托一个已采样Section，不允许同状态多个备选收益累加。

### 不泄漏的留组验证

* `hold_task_task_0000..0003`：留出整个Train配送任务，其余3个训练；4个任务均包含所有四个种子。
* `old_to_new`：训练60、61，预测62、63。**这是新种子 + 早期过采样联合分布迁移，不是单纯种子变化**。
* `new_to_old`：训练62、63，预测60、61；可诊断早期采样分布是否使模型对后期状态错误委托。
* 预测输出含正/负委托、平均净Road增益、总体伤害、最大单次伤害、正例Oracle覆盖和逐快照选择记录。

禁止按候选动作随机划分；禁止用任何留出折结果选择“最好”的门槛，再把其称为独立验证。

## 如何决定R4-M

1. 如果4个task留一与两个种子留组均不能超过C3，**优先扩大新的独立Train任务**，而不是直接训练FiLM/对比学习。
2. 如果new→old策略在大量中后期状态产生无效委托，优先加强状态价值门控/不确定性校准（可借鉴Selective Prediction、Learning to Defer、SPIBB的思想）。
3. 如果跨任务Train留组显示明确的收益覆盖且伤害可控，先冻结模型，再去**独立Validation**做真正30秒在线配对，并测推理时间/算子no-op/道路评价。Validation用于验证，不得回流拟合。最终Test仅一次。
4. 多模态融合、FiLM与对比学习应在可信的状态–动作监督基础上独立消融，不能将事后Road标签直接变成视觉/交通输入。

**结论标签的名称必须准确：** R4-L测量的是固定K反事实标签上的“离线决策”，不是“20,000次固定迭代整个搜索无干预证明”，也不是“30秒在线性能改善”。
