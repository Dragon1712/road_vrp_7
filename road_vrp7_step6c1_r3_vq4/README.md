# Step6-C1-R3-VQ4：扩增原Train状态、可核验的checkpoint复用与嵌套学习曲线

## 1. 目的与研究边界

VQ3六种子结果：

- H=20 `E0_MATCHED` 的C3-restart Road.replay收益 +8.112秒/状态，`VQ_MEAN` -4.480，`RESID_OOF50` +6.753，`PAIR_RANK` +5.545。
- H=50相应为 +7.195、-6.225、+7.403、+5.756。
- H=50 `E0_RISK35` 与匹配覆盖率普通分数门槛 `E0_SCORE_COVERAGE35` 的收益均为 +7.195、覆盖率均为80.56%，风险模型没有证明独立贡献。

结论：不在48状态上继续调门槛或叠模型，先验证 **增加原Train中的训练状态覆盖是否提高决策泛化**。新增样本全部从四个原始**Train任务**（0000—0003）哈希抽样。其余原始Validation/Internal Test任务的多步标签绝不进入训练。本阶段不提供30秒在线ALNS改进主张。

**VQ4完全沿用GitHub已提交的 `road_vrp7_step6c1_r3_vq/collect_vq.py` 与 `road_vrp7_step6c1_r3_vq3/train_vq3.py`。**

- 原VQ：固定真实C3及Section候选，一次Static Repair后，从修复路线调用冻结C3短程**重启**搜索，再在末尾使用Road.replay事后回测；这不是原30秒ALNS精确延续Q值。
- VQ3：任务外层LOTO、任务内部OOF风险校准，13组原有算法不变。
- 本包只增加“冻结源码核验+旧单位复用”“预注册状态数量视图”“规模×种子聚合”和回归测试。

## 2. 成本和数据

| 数据规模 | 原Train状态数 | 每任务状态 | 对照策略数 | 搜索设置 |
|---:|---:|---:|---:|---|
| 48 | 48 | 12 | 13 | H=20/50, 3重启种子 |
| 96 | 96 | 24 | 13 | 同上 |
| 144 | 144 | 36 | 13 | 同上 |
| 192 | 192 | 48 | 13 | 同上 |

最大搜索次数（如果C3+8候选都需要独立搜索）：`192×9×2×3=10,368`。旧48状态理论上最多2,592次并可复用其实际已成功完成的单位。**实际distinct搜索少于上限**，因为部分Section同C3或修复路线相同。正式采集可能显著耗费计算资源和Road.replay时间，本包不声称运行时长。

### 先核实这些目录在服务器存在

```text
road_vrp7_step6c1_r3_vq/
road_vrp7_step6c1_r3_vq2/
road_vrp7_step6c1_r3_vq3/
road_vrp7_step6c1_r3_sv/
road_vrp7_step6c1_r2/
road_vrp7_step6a/
road_vrp7_step5b_v3/
road_vrp7_stage3/
runs/chicago_july_v1/step6c1_r3_vq/vq_train48_H20H50_R3_v1/
runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1/
```

新结果全部写入`runs/chicago_july_v1/step6c1_r3_vq4/`，不改旧源码、不改旧结果、不触碰在线30秒冻结基线。

## 3. 安装与9项测试

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r3_vq4_v1.zip
bash road_vrp7_step6c1_r3_vq4/run_vq4.sh test
```

## 4. 阶段A：先采集一个192状态实验单元（生成原版manifest）

```bash
STATES=192 SELECT_SEED=20261025 HORIZONS=20,50 REPLICATES=3 ROAD_REPLAY=1 \
COLLECTION_STUDY=vq4_train192_H20H50_R3_v1 MAX_NEW_UNITS=1 \
bash road_vrp7_step6c1_r3_vq4/run_vq4.sh collect
```

必须等单元成功输出。程序自动创建新目录并写入 **原始VQ collector自己的冻结数据/代码manifest**。若已存在该目录，不要重新运行此初始化命令；直接进入下一步。

## 5. 阶段B：核验并复用旧48状态的完成单元

```bash
COLLECTION_STUDY=vq4_train192_H20H50_R3_v1 \
bash road_vrp7_step6c1_r3_vq4/run_vq4.sh preflight

COLLECTION_STUDY=vq4_train192_H20H50_R3_v1 \
bash road_vrp7_step6c1_r3_vq4/run_vq4.sh reuse
```

约288个旧单位文件（48×2×3）应被核验。其中刚创建的新实验首单元可能已存在，代码会比较**静态/真实道路目标值**后保留新的文件，允许仅计时字段不同。

强制拒绝情况：候选抽样非前缀、R2标签/道路/冻结求解器/Static Repair/重启源码SHA不同、同一Unit返回不同真实目标值、旧48采集不完整、目标192已COMPLETE等。

输出：`vq4_reuse_audit.json`，逐checkpoint附SHA。

如果因代码和输入差异被拒绝，**不要关闭校验强行拷贝**；应以原collector创建全新的192状态独立实验目录，并从头收集。

## 6. 阶段C：继续新增144状态的多步标注

```bash
STATES=192 SELECT_SEED=20261025 HORIZONS=20,50 REPLICATES=3 ROAD_REPLAY=1 \
COLLECTION_STUDY=vq4_train192_H20H50_R3_v1 RESUME=1 MAX_NEW_UNITS=1 \
bash road_vrp7_step6c1_r3_vq4/run_vq4.sh collect
```

确认续跑可行后：

```bash
STATES=192 SELECT_SEED=20261025 HORIZONS=20,50 REPLICATES=3 ROAD_REPLAY=1 \
COLLECTION_STUDY=vq4_train192_H20H50_R3_v1 RESUME=1 MAX_NEW_UNITS=0 \
bash road_vrp7_step6c1_r3_vq4/run_vq4.sh collect
```

只在最终`report.json.status == COMPLETE`时进入下一步。

## 7. 阶段D：创建四个严格嵌套的数据视图

```bash
COLLECTION_STUDY=vq4_train192_H20H50_R3_v1 \
bash road_vrp7_step6c1_r3_vq4/run_vq4.sh views
```

输出：

```text
runs/chicago_july_v1/step6c1_r3_vq4/
  vq4_train192_H20H50_R3_v1/
    manifest.json
    units/*.json
    rollouts.jsonl
    report.json
    vq4_reuse_audit.json
  vq4_view48_v1/
  vq4_view96_v1/
  vq4_view144_v1/
  vq4_view192_v1/
```

视图只过滤原始192结果中排名前N的抽样状态，不调用求解器、不修改标签；候选和重复随机种子保持一致。筛选规则为**事先固定、与真实收益无关的Train task轮转哈希选择**。

## 8. 阶段E：四个样本量×六个训练种子，不改变VQ3算法

先运行一个组合进行真实Chicago环境接口检查：

```bash
SIZE=48 TRAIN_SEED=42 TREES=160 \
bash road_vrp7_step6c1_r3_vq4/run_vq4.sh train
```

再运行剩余组合：

```bash
for n in 48 96 144 192; do
  for s in 42 43 44 45 46 47; do
    if [[ "$n" == 48 && "$s" == 42 ]]; then continue; fi
    SIZE="$n" TRAIN_SEED="$s" TREES=160 \
      bash road_vrp7_step6c1_r3_vq4/run_vq4.sh train
  done
done
```

结果全部保存在 `runs/chicago_july_v1/step6c1_r3_vq4/models/vq4_n{N}_seed{S}_v1/`。训练器只使用原Train四任务，针对每个N按任务留一，外加任务OOF风险校准。使用原始VQ3的13种策略、同一模型结构和固定阈值。

聚合每个规模：

```bash
for n in 48 96 144 192; do
  SIZE="$n" bash road_vrp7_step6c1_r3_vq4/run_vq4.sh aggregate
done
```

## 9. 阶段F：自动比较学习曲线与四任务配对差值

```bash
bash road_vrp7_step6c1_r3_vq4/run_vq4.sh analyze
```

该步骤默认自动将 **VQ4 n=48的六种子报告与GitHub已验证的VQ3 48状态报告逐一作完全一致性检查**。若同种子完全一致性失败则拒绝总结；先确认原模型代码/数据SHA和旧Unit重启回测是否真正一致。

输出：

```text
runs/chicago_july_v1/step6c1_r3_vq4/vq4_learning_curve_seed42to47_v1/
  report.json
  report.md
  provenance.json
```

重点对照：

- `VQ_MEAN − E0_MATCHED`：增加状态后直接多步监督是否从负增益转为稳定增益？
- `RESID_OOF50 − E0_MATCHED`：额外残差是否超过强的一步学习策略？
- `PAIR_RANK − E0_MATCHED`：排序是否在覆盖率与坏动作上产生实际增量？
- `E0_RISK35 − E0_SCORE_COVERAGE35`：风险分类是否优于覆盖率匹配的普通门槛？

六个模型种子使用同一任务和同一标签，不能当作24个独立样本。**必须比较4个任务的配对方向，不能只看六种子平均值。**

## 10. 决策与下一步

如果随着N=48→192增大，VQ或排序对照在多数留出**任务**均真正改善，且风险仍在覆盖率匹配下有独立效果，可考虑新月份固定30秒在线配对实验。若无改善，不应继续花钱扩大重启种子；优先尝试捕获**完整ALNS的延续状态**或DAgger-style访问状态，并先审计新状态标签分布。

本包不包含真实Chicago服务器运行结果；本地单元/合成测试通过，不宣称已获得任何新性能增益。
