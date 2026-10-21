# Step6-B3-A：FiLM × SPIN-inspired Reliability × Static-Anchored Residual

**适配基准GitHub提交：** `0fb025be97603f0681d783aed12cac66b29c5139`（Dragon1712/road_vrp_7）。

该目录独立新增，不修改或覆盖 `road_vrp7_step5b_v3/`、`road_vrp7_step6a_v2/`、`road_vrp7_step6b2/`、`road_vrp7_step6b2_r/`，也不修改已有训练数据与旧结果。

## 1. 官方源代码引用与许可边界

当前仓库实际存在下列源码：

| 论文及出处 | 官方文件（已有） | B3具体借鉴 |
|---|---|---|
| FiLM (Perez et al., AAAI 2018) | `literature/code/film-master/vr/models/film_gen.py`, `vr/models/filmed_net.py` | 条件输入生成逐通道 gamma/beta；使用 `h * (1+gamma) + beta`，以静态表示为基底 |
| SPIN (NeurIPS 2022) | `literature/code/spin-main/spin/models/spin.py`, `spin/layers/temporal_graph_additive_attention.py` | 真实观测 mask、无可靠观测时回退静态预测；不是原论文 GNN 重建模型 |
| Learning to Delegate (NeurIPS 2021) | `literature/code/learning-to-delegate-main/supervised.py` | 选择局部Section的收益回归（延续B1/B2），不复制其神经网络 |
| SPCLT (2025) | `literature/code/spclt-main/modules/losses.py` | 后续CL阶段预留，**本版未实现结构保持对比损失** |

FiLM官方仓库的`LICENSE`为 **CC BY-NC 4.0**。本包没有重新分发其原始视觉模型代码，`model.py`的 FiLM 数学机制系重新独立编写，保留上述论文与源码归属。若计划直接复制原源码或进行商业再分发，应另行核对许可。

## 2. 输入与监督定义

全部复用当前：

```
runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1/
    analysis/features_X.npz
    analysis/labels_y.npz
    analysis/groups.csv
runs/chicago_july_v1/step6b2/features_boundary_v1/
    boundary_X.npz
runs/chicago_july_v1/step6b2/b2_full_seed42_v1/
    task/P2_F2_model.joblib
    date/P2_F2_model.joblib
```

静态锚点：**P2-F2**（26维结构+时间+静态移除/插入边界）。
交通增量与可靠性：**B2-F4共44维输入**，拆为26维结构、10维历史交通增量、8维可靠性特征。

主监督标签为 `static_repair_td_gain_sec`：静态Repair后的路线在历史TD规划成本下**事后评分**。不在在线推理时读取真实未来成本，不使用 Oracle Repair 标签训练。结果仍是离线单步收益，不是最终Road.replay的30秒道路收益。

M1~M4使用任务/日期**训练域内部 GroupKFold交叉拟合P2-F2静态预测**形成残差目标，避免直接以训练集内静态模型的过拟合误差训练交通分支。在线基础模型始终使用完整训练折上**原始冻结P2-F2**模型，因此两者可能有分布差异；仅通过验证集选择 `alpha`，不可利用内部测试集优化。

**M0至M4共享同一个P2-F2执行阈值**（从原B2训练折的验证记录继承），以避免把门控差异错当成交通增益。`alpha=0`总是合法，能完全退回静态锚点。

## 3. 方法与消融

| 方法 | 模型 | 来源思想 |
|---|---|---|
| M0 | 原P2-F2冻结静态预测器（alpha=0） | 严格静态对照 |
| M1 | M0 + Ridge预测归一化交通增量残差 | Traffic Delta Normalization |
| M2 | M1 + `q`覆盖率门控；无任一边界可靠观测则q=0 | SPIN-inspired sparse observation |
| M3 | M0 + 小型FiLM残差神经网络 + q门控 | FiLM官方缩放/平移机制的独立表格化改造 |
| M4 | M3 + 模态Dropout + 一致性训练 | Missing-modality robustness（非SPCLT复现） |

**M3/M4不是原官方FiLMedNet**，也没有宣称已经运行SPIN或SPCLT。它们是基于论文公式的独立微型模型。

可靠性门控 `q=sqrt(min(c_rem,c_insert))` 且若任一边界真实观测占比为0，则`q=0`。当前q偏保守，需报告门控为0的候选占比；如果大多数候选无观测，M2–M4可能与M0完全相同，不应因此假装模型有增益。

模型训练在CPU即可执行，建议单线程torch，使用AdamW、SmoothL1、权重衰减与梯度裁剪，默认55轮、小型hidden24。强度`alpha`候选 `[0, .1, .25, .5, 1, 1.5]` 在原验证集上选取；若并列，选择绝对值较小者。

## 4. 安装与运行

服务器仓库根目录：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6b3_a_v1.zip
python -m pip install -r road_vrp7_step6b3/requirements.txt
bash road_vrp7_step6b3/run_b3.sh test
```

**本地测试7项通过**，涵盖合成44维特征、FiLM初始化、model pickle、GroupKFold交叉拟合、完整M0～M4×任务/日期端到端跑通。**未在真实芝加哥NPZ上集成运行**，须在服务器先跑Smoke。

```bash
STUDY_NAME=b3_smoke_seed42_v1 bash road_vrp7_step6b3/run_b3.sh smoke
```

Smoke通过后在真实472状态、3776候选上完整训练：

```bash
STUDY_NAME=b3_anchor_film_seed42_v1 SEED=42 \
  bash road_vrp7_step6b3/run_b3.sh train
```

训练输出：

```
runs/chicago_july_v1/step6b3/b3_anchor_film_seed42_v1/
    manifest.json
    task/
        report.json
        reliability_diagnostics.json
        M0_model.joblib ... M4_model.joblib
        M*_test_candidates.csv
        M*_test_states.csv
    date/
        report.json
        M*_model.joblib
        M*_test_candidates.csv
        M*_test_states.csv
    summary.json
```

如需其他已冻结B2-R静态P2-F2基线种子，训练多种子：

```bash
for seed in 43 44 45 46 47; do
  SEED="$seed" STUDY_NAME="b3_anchor_film_seed${seed}_v1" \
      bash road_vrp7_step6b3/run_b3.sh train
done
```

脚本自动使用`runs/chicago_july_v1/step6b2_r/b2r_ablation_seed{seed}_v1`作为对应静态锚点；若锚点SHA或训练种子不匹配，会拒绝运行。

## 5. 逐状态成对比较

```bash
python road_vrp7_step6b3/compare_b3.py \
 --study runs/chicago_july_v1/step6b3/b3_anchor_film_seed42_v1 \
 --base M0 --treatment M3 \
 --output runs/chicago_july_v1/step6b3/paired_M3_vs_M0_seed42_v1.json
```

该脚本不仅计算平均收益，还计算实际被修改的Section决策占比、改善/恶化/持平状态及仅在决策变化时的平均收益。任务与日期内部测试仍属多次用于开发的7月数据，绝非盲测。

## 6. 30秒在线集成——先验收离线再上线

原B2-R的`OnlineSectionR.form`天然支持`F4`，原`benchmark_30s_r.py`的`load_artifact`接受P2+F4，只需读取新的B3 model artifact，不改核心C3代码。

**重要公平性：**M0在线要直接使用原`P2-F2`模型（不加载历史交通）；M1～M4用`F4`完整特征。必须另行记录F4额外历史特征预处理的时间成本。均使用同一30秒预算、任务场景种子、抽样器、`score_every=8`、C3回退。

```bash
MODEL_STUDY=runs/chicago_july_v1/step6b3/b3_anchor_film_seed42_v1 \
B3_METHOD=M3 STUDY_NAME=b3_online_M3_match8_v1 \
  bash road_vrp7_step6b3/run_b3.sh online
```

再运行M1/M2/M4，保持任务、场景、随机种子完全一致，与已冻结`b2r_online_P2_F2_match8_v1`及Sampled-Random配对。不要只和原C3比较。

**上线前必须做真实数据特征对齐检查。** M3/M4的训练F4与在线F4特征顺序已由模型校验，但同一候选的数值一致性仍需要真实路网验证（真实Pilot）；并检查小FiLM加载、模型开销和依赖版本。

## 7. 当前研究边界与预期

- 本版没有重新训练任何SPIN、SPCLT模型，也没有真正序列图编码；只是SPIN可靠性处理和FiLM的轻量数学机制适配。
- 不要用“加FiLM使模型更复杂”作为创新理由；核心要证明M3在同一静态边界基础上稳定优于M0，且打乱交通观测后改善消失。
- 交通完全缺失：M2～M4保证数值上恢复M0（独立单元测试验证）。
- M1可能在缺失覆盖时产生非零残差，是用于对照的无门控负迁移模型。
- 虽然`alpha=0`可以保护模型不在验证集上任意退化，验证集收益并非对新任务/月份可靠性的保证。
- 最终要在新的任务与月份做严格独立泛化、30秒ALNS与道路级回测；当前7月测试数据早已用于方法设计，不可宣称显著泛化。
