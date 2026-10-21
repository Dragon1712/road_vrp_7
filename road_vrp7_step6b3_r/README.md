# Step6-B3-R：严格归因审计与冻结交通负对照

适配 `Dragon1712/road_vrp_7`，**2026-10-21 锁定 Git commit `c1519387f3b2f0262cc120d640865480c8ed4045`**。该目录独立新增，不修改旧 ALNS、B2-R、B3 文件，不覆盖既有 July 结果。

## 研究定位

- `audit_b3r.py`：6模型训练种子×task/date严格成对离线测试。纠正 `compare_b3.py` 把双方都不执行的 Top-1 改变计为“实际动作改变”的错误；同时输出完整的有效动作改动明细、MAE/预测残差幅度、胜负及每 seed 平均差。仅**诊断**，不调整超参数。
- `placebo_b3r.py`：复用原始 `road_vrp7_step6b2_r/frozen_permutation.py::shuffle_same_task_date`，**冻结B3模型、alpha和阈值**，同一个 task、不同日期按状态成块交换交通与/或可靠性特征。每一置换种子同时用于所有模型；绝不交换结构特征。`traffic_and_mask` 为主要负对照；`traffic_only` 与 `mask_only` 有意破坏交通和mask一致性，属于敏感性检查。
- `online_compare_b3r.py`：以原冻结P2-F2在线M0为主基线，按 task/scenario/seed 检查配对及manifest SHA和所有搜索配置；输出 M3/M2/M4/Random/C3 的道路成本、迭代、评分次数、C3回退等。只有对M0的差可尝试做模型贡献分析。**对C3的差异混合了Section抽样协议变化**。
- `run_b3r.sh`：独立新目录运行所有实验、允许显式设置不同运行名称。

## 安装

把整个目录复制到仓库根目录下；已具备 `road_vrp7_step6b3/` 和 `road_vrp7_step6b2_r/` 依赖环境，沿用原来 `alns==7.0.0`、numpy、scikit-learn、joblib、torch。以**可信自训的joblib**为唯一加载来源。

```bash
cd ~/code/WZL/road_vrp_7
# 解压下载包： unzip road_vrp7_step6b3_r_v1.zip
bash road_vrp7_step6b3_r/run_b3r.sh test
```

## Stage 1：6 seed 离线准确归因（无需重新训练）

```bash
STUDY_NAME=b3r_audit_s42to47_v1 bash road_vrp7_step6b3_r/run_b3r.sh audit
# outputs:
# runs/chicago_july_v1/step6b3_r/b3r_audit_s42to47_v1/audit.json
# runs/chicago_july_v1/step6b3_r/b3r_audit_s42to47_v1/changed_states.csv
```

`changed_states.csv` 的 `changed_effective_action` 使用 `NOOP` 作为实际不执行动作，剔除“两个模型不执行但换了候选Top-1”的虚假动作变化。`audit.json` 同时提供各 seed 的配对有符号改善和模型预测残差幅度分布。**同一批测试状态重复6个模型种子，不可用6倍样本量做显著性宣称。**

## Stage 2：冻结交通+可靠性置换实验（仅task协议有效）

```bash
STUDY_NAME=b3r_placebo_joint_s42to47_v1 \
  SHUFFLES=traffic_and_mask \
  bash road_vrp7_step6b3_r/run_b3r.sh placebo

# 若联合置换能检测出模型对交通时间对齐的敏感性，再补两个诊断：
STUDY_NAME=b3r_placebo_component_s42to47_v1 \
  SHUFFLES=traffic_only,mask_only \
  bash road_vrp7_step6b3_r/run_b3r.sh placebo
```

`placebo.json` 记录每 seed×模型×置换类型×20置换种子下相对原始交通的差值，负值代表错位交通使收益下降。主要负对照只在同一测试task内部跨日期置换；**date协议测试折通常只有一个日期，不能伪造date-placebo**。置换不同状态的候选尽管保持了8候选排列，却可能不匹配几何语义。因此该实验只是模型对历史对齐敏感性证据，不构成“实际交通具有独立因果收益”的证明。对比M0与负对照、端到端道路成本必须一起看。

## Stage 3：在线补齐M2/M4固定30秒对照

```bash
MODEL_STUDY=runs/chicago_july_v1/step6b3/b3_anchor_film_seed42_v1 \
B3_METHOD=M2 STUDY_NAME=b3r_online_M2_match8_v1 \
  bash road_vrp7_step6b3_r/run_b3r.sh online

MODEL_STUDY=runs/chicago_july_v1/step6b3/b3_anchor_film_seed42_v1 \
B3_METHOD=M4 STUDY_NAME=b3r_online_M4_match8_v1 \
  bash road_vrp7_step6b3_r/run_b3r.sh online

STUDY_NAME=b3r_online_comparison_match8_v1 \
M2_RUN=runs/chicago_july_v1/step6b3_r/b3r_online_M2_match8_v1 \
M4_RUN=runs/chicago_july_v1/step6b3_r/b3r_online_M4_match8_v1 \
RANDOM_RUN=runs/chicago_july_v1/step6b2_r/b2r_online_random_match8_v1 \
C3_RUN=runs/chicago_july_v1/step6b2_r/b2r_online_C3_match8_v1 \
 bash road_vrp7_step6b3_r/run_b3r.sh online-compare
```

比较器拒绝 manifest中任务、场景、种子、道路/输入哈希、官方求解器、Section源码、预算、迭代/SA、采样器和缓存/回退协议不一致的运行。M0与M3已有8例可直接先比较；M2/M4未运行时可暂不传 `M2_RUN`/`M4_RUN`。

## 质量与可复现性门槛

1. **先运行test。** 本包测试只验证纯逻辑与合成CSV。Github公开结果已用于接口审计，当前运行环境没有完整真实Chicago NPZ与torch joblib，**尚不能宣称真实数据placebo脚本已运行通过**；须在有数据的服务器上跑完 Stage1/2。
2. Placebo必须复现冻结M1~M4的 `M*_test_candidates.csv` 分数，防止偷偷切换加载模型。并严格验证离线数据、F4 44维顺序、source及边界文件SHA、训练alpha和threshold。
3. Stage 3输出独立目录。所有Road.replay指标、模型特征时间和推理时间在同一个30秒搜索预算内核算；OD和历史图加载在30秒之外单列。
4. 8例Pilot只有2个配送任务；任何平均提升均不可写成鲁棒显著收益；7月为反复用于开发的数据，真正盲测必须预留未见任务和未见月份。
5. 该实验**尚未实现SPIN图神经网络、SPCLT损失或完整Learning-to-Delegate网络**。这些应在 B3-R归因通过后另起目录开发，并保持C3、P2-F2、B3模型三层基线。
