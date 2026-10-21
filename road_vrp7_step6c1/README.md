# Step6-C1：真实道路边稀疏交通编码 + 状态级 Section Delegate（July 开发实验）

**基准仓库**：`Dragon1712/road_vrp_7`，锁定 `main` commit `9befbc82d5698d96340f9620bf3166fb3234d1f7`（2026-10-21）。

> 本代码是独立实现的 **SPIN-inspired mask-aware 小型有向道路图编码器**，**不是** 1:1 官方 SPIN 网络；借鉴 *Learning to Delegate* 的子问题效用监督，在这里实现 **8个Section与显式no-op的集合决策**；**没有引入SPCLT、InfoNCE或者任何对比学习损失**。SPCLT 留待 C2。不要把 C1 结果写成“SPIN、SPCLT和Learning to Delegate完整集成”。

本目录仅新增代码，完全不修改 `road_vrp7_step5b_v3/` 官方ALNS基准、`road_vrp7_step6a_v2/` 历史标签、`road_vrp7_step6b2_r/` 在线搜索，独立结果根目录 `runs/chicago_july_v1/step6c1/`。

## 文件说明

| 文件 | 作用 |
|---|---|
| `graph_features.py` | 用B2**同一个Static Repair首位Relocate插入代理**提取移除/插入六条边界OD弧；沿每条弧对应的**真实OSM有向道路路径**抽取至多4条道路边，形成24边×6时间槽的真原始值/真实mask和由OSM端点确定的有向邻接 |
| `export_graph.py` | 重放冻结的472状态/3776候选，仅抽取输入特征，严格校验样本、道路SHA与F4，可断点续跑，支持小样本Smoke |
| `c1_model.py` | 6槽观测mask-aware时间注意力、有向图邻接注意力、8候选集合注意力、冻结P2-F2静态RF锚点、显式no-op与排名/策略价值损失 |
| `train_c1.py` | task/date严格分组、训练、**仅在验证集选checkpoint**，输出D0/D1/D2T/D2与独立M0结果、模型文件与哈希 |
| `online_c1.py` | 运行时将B2-R的`OnlineSectionR`子类替换，仅在D2/D2T的Section候选特征后追加道路图张量，搜索结束恢复原始类；不动ALNS其他算子、SA、repair或目标 |
| `run_c1.sh` | Smoke、提取、训练、在线及多组对照的统一入口 |
| `tests/test_c1.py` | 6项单元测试，包括时序无未来泄漏、真实有向OSM邻接、无观测降级、Set/No-op损失、合成训练端到端 |

### 严格数据边界

- 路网：真实 `edge_index.csv` 中的有向道路边，邻接条件为 `edge_i.osm_v == edge_j.osm_u`，不能把反方向自动视为相邻。
- 时间：`HistoricalCorridorFeatures.enter()` 的最近 **6 个严格早于 departure 的 raw 历史槽位**。模型不接触Oracle候选修复、`Road.replay`或未来FIFO时变信息。
- 历史观测：只有 `chicago_observed_mask=1` 的 `travel_sec_raw/static_travel_sec` 相对值可进入编码器；缺失位置填数值0**同时保留mask=0**，不是假装已观测。
- 道路图：每条移除/插入静态OD边界弧等间距采样最多4条真实OSM道路边（上限24），按真实OSM端点连接。**不是整幅芝加哥路网，也不是未来真实动态最短路**。这适合30s试验，但表示能力受24边采样限制。
- 原始B2 F4：仍作为44维数值特征，冻结静态RF是26维F2；图输入作为sidecar额外拼接，从而保留B2-R的`--model-mode F4`与验证接口。
- 所有独立运行须使用**新目录**，不对先前实验输出追加。

## 消融表

| Arm | 样本级特征 | 集合决策监督 | 原始历史时序 | 空间邻接 | 解释 |
|---|---|---:|---:|---:|---|
| M0 | F2静态RF | 否 | 否 | 否 | 冻结强基线 |
| D0 | 冻结静态RF + F2静态结构 | 是 | 否 | 否 | 纯Learning-to-Delegate-style决策增量 |
| D1 | D0 + F4 18维汇总历史/可靠性 | 是 | 否 | 否 | 表格交通模态增量 |
| D2T | D1 + 原始有mask的道路边六槽序列 | 是 | 是 | **身份邻接** | 仅时间编码的严格结构对照 |
| D2 | D2T同样参数量和网络，但用真实有向OSM邻接 | 是 | 是 | **真实邻接** | 稀疏有向道路空间信息增量 |

其中 **D2 - D2T** 是空间邻接的最直接对照；**D1 - D0** 检验汇总历史模态，**D2T - D1** 检验道路边原始mask时序，**D0 - M0** 检验不同的状态级决策训练。D2T和D2可固定`PRETRAIN_EPOCHS=0`进行无掩码自监督预训练的对照。

## 服务器操作

所有命令从项目根目录执行。请使用与你之前B3实验同一Python环境（官方`alns==7.0.0`，`numpy, pandas, scipy, scikit-learn, torch, joblib`）。**不要在测试环境里加载不可信来源的joblib**。

### 0. 准备与测试

```bash
cd ~/code/WZL/road_vrp_7
git fetch origin && git rev-parse HEAD && git rev-parse origin/main
# 从ZIP复制 road_vrp7_step6c1/ 到仓库根目录
bash road_vrp7_step6c1/run_c1.sh test
```

### 1. 真实数据 Smoke：4个状态且逐项核对F4

```bash
MAX_STATES=4 VALIDATE_F4=1 STUDY_NAME=c1_graph_smoke4_v1 \
  bash road_vrp7_step6c1/run_c1.sh export
cat runs/chicago_july_v1/step6c1/c1_graph_smoke4_v1/summary.json
# 应标记 PILOT_ONLY；不输出可训练的全量graph_X.npz
```

### 2. 全量道路图特征（可续跑）

```bash
VALIDATE_F4=1 STUDY_NAME=c1_graph_full_v1 \
  bash road_vrp7_step6c1/run_c1.sh export
# 如中途异常，且原有脚本和所有数据哈希完全不变：
RESUME=1 VALIDATE_F4=1 STUDY_NAME=c1_graph_full_v1 \
  bash road_vrp7_step6c1/run_c1.sh export
```

最终应为 `COMPLETE`、472状态、3776候选；保存有序 `graph_X.npz`、JSON SHA与checkpoint。先审计 `observed_slot_fraction`，极低覆盖率下不要指望神经图比统计特征必然更好。

### 3. 小模型Smoke（先训练静态与汇总交通，2 epoch）

```bash
EPOCHS=2 PRETRAIN_EPOCHS=0 VARIANTS=D0,D1 \
  STUDY_NAME=c1_train_smoke_seed42_v1 \
  bash road_vrp7_step6c1/run_c1.sh train
```

### 4. 完整训练六种子（按task/date协议，D0/D1/D2T/D2）

```bash
for s in 42 43 44 45 46 47; do
  SEED=$s EPOCHS=30 PRETRAIN_EPOCHS=4 \
  STUDY_NAME=c1_delegate_seed${s}_v1 \
    bash road_vrp7_step6c1/run_c1.sh train
done
```

`run_c1.sh`会根据训练种子自动匹配已有B2 P2-F2锚点：种子42来自 `step6b2/b2_full_seed42_v1`，其他来自 `step6b2_r/b2r_ablation_seed${s}_v1`。

输出包含 `task/report.json`、`date/report.json`、每变体的 `*_history.json`、`*_test_states.csv`、`*_model.joblib` 和 manifest。**只在训练组更新参数、验证组选择epoch，测试组不得用于调参。**

### 5. 固定30秒8例在线初筛

```bash
for arm in D0 D1 D2T D2; do
  C1_VARIANT=$arm \
  MODEL_STUDY=runs/chicago_july_v1/step6c1/c1_delegate_seed42_v1 \
  STUDY_NAME=c1_online_${arm}_match8_v1 \
     bash road_vrp7_step6c1/run_c1.sh online
done
```

仍保留B2-R的官方C3修复/SA/算子轮盘、30秒墙钟和20000迭代上限，`score_every=8`，模型及特征计入搜索耗时；历史Raw读取、OD缓存初始化按原B2-R协议另计。**因为同时保留20000迭代上限，部分运行可能提前结束；这叫相同双重预算，不等同每次用满30秒。**

### 6. 固定样本严格在线对照

```bash
STUDY_NAME=c1_comparison_match8_v1 \
D0_RUN=runs/chicago_july_v1/step6c1/c1_online_D0_match8_v1 \
D1_RUN=runs/chicago_july_v1/step6c1/c1_online_D1_match8_v1 \
D2T_RUN=runs/chicago_july_v1/step6c1/c1_online_D2T_match8_v1 \
D2_RUN=runs/chicago_july_v1/step6c1/c1_online_D2_match8_v1 \
RANDOM_RUN=runs/chicago_july_v1/step6b2_r/b2r_online_random_match8_v1 \
  bash road_vrp7_step6c1/run_c1.sh compare
```

此比较复用原B3-R配对检查器：相同任务、场景、随机种子、数据/求解器SHA、TopK/动作采样器、阈值和预算配置。**与M0的性能差包含集合决策及不同执行阈值，不能全归因于交通模态；D0→D1→D2T→D2链条才是主消融。**

## 主要结果判定

1. 离线最重要的是测试组的**每状态有符号收益**，不是单纯PR-AUC/MAE；训练集和验证集收益不能作为结论。必须保留`0`执行（no-op）作为候选。
2. 在线唯一主要终点是 `Road.replay(...).ex_post_road_travel_sec`，但8条Pilot只有2个任务与1个日期，绝对不能据此宣称统计显著。
3. 30秒必须记录模型执行动作数、C3回退次数、推理+特征时间和迭代次数；D2可能因图提取过慢而触发2秒评分预算，需要诚实报告。
4. **离线no-op标签收益是0**，但在线模型拒绝时为了避免B2的no-op陷阱回退C3，二者不是同一动作；这是当前C1的已知监督/部署不一致，需要以后采集真实C3回退收益标签才能完全消除。
5. 任务/日期交叉重复，六个模型训练种子并不是独立测试样本；最终必须冻结架构，在真正未触碰的月份和任务上进行盲测、分簇bootstrap。
6. C1未实现SPCLT。只有当D2或D2T提供稳定且可解释的交通时序贡献，才值得在C2增加结构保持对比损失。

## 测试和限制

本ZIP在当前独立容器中通过**6项合成/接口单元测试和py_compile/bash -n**。没有在此容器直接访问服务器的真实Chicago NPZ，因此**不能声称真实数据的导出、训练、在线Benchmark已经成功**；必须按上面 `Smoke → full export → train → online` 的顺序运行。请不要将本包直接称为官方SPIN源码复现。

### 静态锚点的训练内泄漏审计

参考前一阶段B3，为避免在训练集中直接使用P2-F2随机森林的**训练内预测残差**，C1在每个task/date**训练折内部**重新做`GroupKFold`静态RF OOF预测，仅作为神经决策损失中的固定anchor项；**val/test/在线**继续使用未经修改的冻结P2-F2静态RF。`report.json`中的`_anchor_diagnostics`记录OOF与full RF训练预测的MAE、均值差、标准差。这减少了训练内偏乐观偏差，但不能完全消除OOF与部署时full RF的分布差，必须检查。
