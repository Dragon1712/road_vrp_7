# Step6-C1-R：可审计的锚点修复、代价敏感Delegate与保守回退

**基准**：[Dragon1712/road_vrp_7](https://github.com/Dragon1712/road_vrp_7)，2026-10-21 `main` commit `f43534fa3da8c1bc1b1dbeaa618d4f8fcda5a37d`。本目录是**新代码**，无需覆盖原C1、B3或任何冻结官方ALNS源码。

## 研究定位和不能夸大的地方

原C1（7月开发内部测试）六种子D0、D1、D2T、D2都没有稳定超过冻结P2-F2静态M0，且离线zero-noop与在线C3 fallback不一致。C1-R**首先作受控消融，定位失败来源，不声称新方法一定改善效果**。

已检查用户仓库上传的论文代码：

| 借鉴代码（仓库相对路径） | 源代码实际做法 | C1-R的独立适配范围 |
|---|---|---|
| `literature/code/learn-to-defer-master/README.md` 及各实验脚本 | cost-sensitive learning-to-defer到专家；README引导至新版human_ai_deferral | 新增**固定M0代理动作**作为可选择的defer目标、成本敏感状态级预期收益损失；不是原论文的L_CE^α精确实现 |
| `literature/code/SPIBB-master/spibb.py` (`compute_mask*`, `update_pi`) | 数据支持不足的state-action保留基线策略概率 | 使用验证集选择的最小得分差、最大修正幅度与交通覆盖阈值；拒绝回退原冻结C3；**非真正SPIBB，无理论安全保证** |
| `literature/code/learning-to-delegate-main/supervised.py` | VRP子问题监督、邻域/路径mask和学习选择 | 复用C1的8-Section集合评分/一次Static Repair标签，以“模型动作 vs M0代理动作”训练；不运行原LKH/HGS |
| `literature/code/SmartPredictThenOptimize-master` | 目标导向SPO+优化损失 | 借鉴“决策优于MAE”的目标；不是SPO+求解器或其理论保证 |
| `literature/code/spin-main/spin/models/spin.py` | 显式mask、稀疏时空图注意力 | 保持原C1 mask-aware道路图网络，不在R阶段再改表示 |
| `literature/code/STMAE-main`, `STGCL-main`, `spclt-main`, `NDS-main`, `DRHG-master` | 掩码自编码、时空对比、组合搜索/重优化 | **暂未改动训练网络**，以后单独做空间/对比损失与搜索模块消融，避免与本轮修复混淆 |

> C1-R中无逐字复制论文作者的算法源码，独立改写了参考思想。不要声称“官方SPIBB/SPIN/SPCLT算法复现”。由于原始上传代码具有各自许可证，若后续复制代码，先逐一核查许可证。

## 四级受控消融

在四种C1表征`D0,D1,D2T,D2`上，使用同一组训练划分和随机种子：

| 训练臂 | 训练RF anchor | 状态级训练目标 | 推理门控 | 唯一主要改动 |
|---|---|---|---|---|
| `OOF_ZERO` | GroupKFold OOF | 原C1 zero-noop loss | 不启用 | 复核原C1训练约定 |
| `FULL_ZERO` | 冻结P2-F2 FullRF | 原C1 zero-noop loss | 不启用 | 仅改变anchor，测训练/部署分布偏移 |
| `FULL_DEFER` | FullRF | M0代理动作作为固定额外选项的期望收益loss | 不启用 | 借鉴代价敏感Learning to Defer |
| `FULL_DEFER_SAFE` | 与FULL_DEFER**同一组训练权重** | 同上 | **验证集调参**的conservative gate | 单独测保守回退的贡献 |

`FULL_DEFER_SAFE`不会单独重新训练网络，而是拷贝已经训练的`FULL_DEFER`模型并选择验证集门控，因此模型部分严格相同。

### 训练期loss定义

令`a_0`为相同8个Section中冻结静态RF分数最高的Section，若原P2-F2阈值不通过，则代理奖励为0。

\[
L_{\rm defer} = L_{\rm reg/rank} - \lambda\sum_{a\in\{a_1,...,a_8,\text{M0-defer}\}} \mathrm{softmax}(\ell_a / 35)\, \mathrm{clip}(r_a,-300,300)/100.
\]

**注意：**“M0-defer”奖励来自**当前标签集合中的静态RF动作**，不是ALNS运行中的原C3 Section随机采样动作；二者的数据分布不一样。当前无法声称训练和在线fallback已经严格对齐。为了获得真正的C3-aware训练，下一轮必须额外采集同状态的冻结C3动作并离线标注其Static Repair收益。

### 在线保守门控

模型先选择 argmax 的Section。若它与当前集合静态RF最优Section不同，则需同时满足：

- 预测模型最佳与模型第二参考分数的优势超过margin；
- 预测模型最佳分数比冻结静态RF最优分数高出margin；
- 候选预测残差绝对值不超过cap；
- 交通模型需要达到最低历史观测覆盖率。

不满足则通过原`OnlineSectionR`的`gate`机制调用**原C3回退**。这些是**经验可靠性条件**，不是可校准的置信区间或者理论安全改进证明。验证集只选择`margin ∈ {0,5,10,20,40}`, `cap ∈ {20,40,80,160}`，覆盖率`{0,.05,.2}`（D0不用交通覆盖）。选择时对同等收益优先减少模型干预，避免因为少量异常值增加风险。

### 关键公平性修复

`D0`训练仍使用与原C1相同的44维F4 scaler接口（其网络完全不使用交通维），但部署artifact改为B2-R **F2 26维特征**，`ReliableC1.predict()`将缺失的交通维补零作为不参与网络的占位。已有合成测试验证同一静态输入的F2/F4预测相同。**D0在线不会加载历史Raw，也不会提取F4交通边界。** D1、D2T、D2仍使用F4；D2T、D2追加原C1 984维真实图sidecar。

## 运行

在装有`alns==7.0.0, torch, numpy, pandas, scipy, scikit-learn, joblib`及已有环境的服务器：

```bash
cd ~/code/WZL/road_vrp_7
# 将压缩包中的 road_vrp7_step6c1_r/ 解压到此目录
bash road_vrp7_step6c1_r/run_c1r.sh test
```

### 1. 最先做静态锚点和fallback的两任务协议离线诊断（推荐先smoke）

```bash
EPOCHS=2 PRETRAIN_EPOCHS=0 VARIANTS=D0 \
 ARMS=OOF_ZERO,FULL_ZERO,FULL_DEFER,FULL_DEFER_SAFE \
 STUDY_NAME=c1r_smoke_seed42_v1 \
 bash road_vrp7_step6c1_r/run_c1r.sh train
```

确认smoke通过之后，训练第一组正式消融：

```bash
for s in 42 43 44 45 46 47; do
 SEED=$s VARIANTS=D0,D1 EPOCHS=30 PRETRAIN_EPOCHS=0 \
  STUDY_NAME=c1r_seed${s}_v1 \
  bash road_vrp7_step6c1_r/run_c1r.sh train
done
STUDY_NAME=c1r_aggregate_D0D1_v1 \
 bash road_vrp7_step6c1_r/run_c1r.sh aggregate
```

若静态集合决策回到M0附近，再单独训练图分支（必须使用**其他输出目录**，不能覆盖刚才的研究）：

```bash
for s in 42 43 44 45 46 47; do
 SEED=$s VARIANTS=D2T,D2 EPOCHS=30 PRETRAIN_EPOCHS=4 \
  STUDY_NAME=c1r_graph_seed${s}_v1 \
  bash road_vrp7_step6c1_r/run_c1r.sh train
done
MODEL_SEEDS=42,43,44,45,46,47 \
 STUDY_NAME=c1r_aggregate_graph_v1 \
 python road_vrp7_step6c1_r/aggregate_c1r.py \
   --prefix 'c1r_graph_seed{seed}_v1' \
   --output-dir runs/chicago_july_v1/step6c1_r/c1r_aggregate_graph_v1
```

**本项目不能在`internal_test`结果上反复调门控**；初筛后应该冻结参数，使用新的任务/月份做最终回测。

### 2. 原道路图连接性及观测率审计（不训练，不读未来标签）

```bash
STUDY_NAME=c1r_graph_audit_v1 \
 bash road_vrp7_step6c1_r/run_c1r.sh graph-audit
```

生成`graph_audit.json`：无padding真实边的观测率、平均有效节点数、有向非自环连接数、孤立节点率、无历史信息的图占比。修复图采样应建立在这份诊断上，且必须重新导出、新建训练目录。

### 3. 固定30秒在线匹配Pilot（只用task模型）

建议第一轮仅比较D0，严格保留C1已记录的2个任务×2个场景×2个种子：

```bash
for arm in OOF_ZERO FULL_ZERO FULL_DEFER FULL_DEFER_SAFE; do
 C1_VARIANT=D0 C1_ARM=$arm \
 MODEL_STUDY=runs/chicago_july_v1/step6c1_r/c1r_seed42_v1 \
 STUDY_NAME=c1r_online_D0_${arm}_match8_v1 \
 bash road_vrp7_step6c1_r/run_c1r.sh online
done
```

门控本身不是零计算成本，需要汇报`r_model_actions`, `r_c3_fallbacks`, `r_scoring_seconds`, `section_feature_wall_sec`, `section_inference_wall_sec`, `iterations`, `search_wall_sec`，并与相同采样策略下的原M0 F2回测做严格配对。

```bash
R_RUN=runs/chicago_july_v1/step6c1_r/c1r_online_D0_FULL_DEFER_SAFE_match8_v1 \
 STUDY_NAME=c1r_pair_D0_safe_match8_v1 \
 bash road_vrp7_step6c1_r/run_c1r.sh compare
```

在线唯一主要终点是`Road.replay().ex_post_road_travel_sec`（越小越好）；离线两个指标`zero_noop`和`M0_proxy_fallback`**不能混同**。原实验在`max_iterations=20000`上限条件下并非每次都使用完整30秒，需与既有基线保持完全一致。本代码也没有声称在本地跑过真实道路实验。

### 4. 向新目录扩大验证

先跑8条固定配对验证，若结果稳定且优于基线，再另外用新的独立目录、增加任务/日期/种子。请不要使用内测test反复挑选冠军模型。

## 数据及学术结论边界

1. `C1-R`只是从现有数据做**受控修复/安全选择**；尚未完成**真实C3子问题回退标签**采集，因此不是严格复现的Learning-to-Delegate到C3或Learning-to-Defer-to-C3算法。
2. 此包尚未加入SPCLT、STGCL或STMAE的新对比学习目标。原C1 SPIN-inspired图编码及masked reconstruction预训练不变，这样可先确认决策端修复的独立贡献。
3. 没有修改静态ALNS solver、objective、SA接受准则、C3 fallback、随机采样、维修算子或30秒停止规则。新`online_c1r.py`只临时替换B2-R中评分选择器类型与模型加载函数，`finally`中恢复。
4. `FULL_ZERO`改变训练锚点可缓解OOF/Full部署分布偏移，但使用原FullRF的训练内预测可能存在训练内偏乐观，应与OOF对照而不是默认它更好。
5. 合成测试和`py_compile`只验证部分函数/接口；**真正的原始数据导出、训练与在线30秒实验必须在你的服务器环境运行，结果未知**。
6. GitHub本轮文献源码均通过读取验证存在，C1-R不修改那些代码。研究中应分别注明`inspired by`与`official reproduced`。
