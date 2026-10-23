# Step6-C1-R3-SV2：代表性同状态多步价值验证（v1）

## 0. 目的与先验审计

**冻结代码基线：** `Dragon1712/road_vrp_7`，提交`4b71365ae94e620fbc204f866f0aa4b72c47faf6`，实验来源：

- `runs/chicago_july_v1/step6c1_r3_sv/sv_same_state_v1`：1,024状态 × 4策略完整的同状态一步审计。
- `runs/chicago_july_v1/step6c1_r3_sv/sv_rollout_12states_v1`：12个“F2/Huber分歧富集”状态 × 4策略 × H=5,20,50 ×2随机种子，共288次C3重启；**不是原始ALNS精确续接**。
- H=50时T2_HUBER相对C3平均+1.003秒TD-OD，H5和H20则分别为−39.348、−30.352秒；同状态的一步收益方向与重启多步方向翻转率约45%–62%。这些现象发生在**有意选择的12状态**，不能代表所有256个Internal Test状态。

**下一阶段关键任务不是训练多步价值神经网络，而是先验证上述符号翻转是否普遍，以及究竟有多大的搜索随机数波动。**

此包不修改原R3-SV的`rollout_sv.py`、官方ALNS、Static Repair、特征/标签生成或任何模型。它只增加：

1. `select_states.py`：只依据role、decision ID等**不读结果标签**的元数据，对已经捕获的状态进行预注册抽样；默认对256个Internal Test状态作确定性哈希随机均匀抽取48个。
2. `run_sv2.sh`：使用原R3-SV复算逐动作Static Repair并做H=5,20,50 C3搜索重启；可断点续跑、支持事后真实Road.replay；内置原有SHA检查。
3. `analyze_values.py`：以**捕获状态为重采样单位**、先平均重复种子再进行配对Bootstrap，报告H0/H多步、符号翻转、种子波动、same-action placebo及事后Road.replay。
4. `tests/`：数据不泄漏、采样复现、严格配对、完全性、安慰剂以及端到端合成流水线测试。

**注意：** `M0_PROXY`依然是同一8候选集合上的M0代理，不是完整M0在线算法。`td_od`是事后时变OD目标，不是最终30秒Road.replay；当`ROAD_REPLAY=1`时后者会被另行报告，但同样不是精确原ALNS续接。

## 1. 安装

把压缩包放到已有的服务器仓库根目录，然后：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r3_sv2_v1.zip
bash road_vrp7_step6c1_r3_sv2/run_sv2.sh test
bash road_vrp7_step6c1_r3_sv/run_sv.sh preflight
```

`run_sv2.sh test`会运行本包新增7项及原R3-SV的7项合成测试；不需要服务器真实道路数据。

## 2. 预先固定不读收益的48个状态

```bash
STUDY_NAME=sv2_select_uniform48_v1 \
STATES=48 SELECT_SEED=20261024 SPLIT_ROLE=internal_test SCHEME=uniform \
bash road_vrp7_step6c1_r3_sv2/run_sv2.sh select
```

新目录`runs/chicago_july_v1/step6c1_r3_sv/sv2_select_uniform48_v1`下包含：

- 原始不可变`decisions.jsonl`,`manifest.json`,`report.json`副本；
- 新的`pilot_states.json`：固定48个decision ID，用于原SV `rollout_sv.py`；
- `sv2_provenance.json`：原始SHA、抽样种子、各任务/场景的入选数量。

默认`uniform`使每个已捕获、可标注的Internal Test状态在随机哈希下具有相同入选机会，**不是对全年的所有ALNS状态或所有任务/日期的无偏代表**；这批数据也早已被用于开发诊断，不能称为盲测。不要根据结果重新更换随机种子挑选好看的48状态。

另有`SCHEME=balanced`供**独立探索性**诊断，让各任务×场景入选数量趋于接近；若各分组规模不同，这不是相同状态入选概率的抽样，不应与`uniform`混合成总体无偏指标。

## 3. 真实多步Smoke（必须先做）

先执行只含一个状态的检验，不生成完整指标。`MAX_STATES=1`不会改变原48个预注册入选ID，正式实验应另用一个新结果目录：

```bash
STUDY_NAME=sv2_rollout_smoke_v1 \
ONE_STEP_STUDY=sv2_select_uniform48_v1 \
MAX_STATES=1 REPLICATES=1 HORIZONS=5,20,50 \
ROAD_REPLAY=1 \
bash road_vrp7_step6c1_r3_sv2/run_sv2.sh rollout
```

确认日志打印 `R3-SV ROLLOUT COMPLETE`，并检查Road.replay是否完成、Static Repair是否在原deadline内、一致性检查是否全部通过。

## 4. 完整48状态 × 3种子多步实验

固定H=5,20,50，3个独立C3重启随机种子（每个状态+重复下不同策略和H使用同一个初始种子）：

\[
48 \times 4 \times 3 \times 3=1728\text{次短程重启搜索}
\]

**注意：启用`ROAD_REPLAY=1`，每个短程搜索结果需要在路网上做事后最短路回测，可能比纯TD-OD实验耗时明显更多。** 它不读取任何未来成本作为在线特征，但多次Road.replay应控制服务器负载。

先一个分片：

```bash
STUDY_NAME=sv2_rollout_uniform48_v1 \
ONE_STEP_STUDY=sv2_select_uniform48_v1 \
REPLICATES=3 HORIZONS=5,20,50 ROAD_REPLAY=1 \
MAX_NEW_UNITS=1 \
bash road_vrp7_step6c1_r3_sv2/run_sv2.sh rollout
```

续跑：

```bash
STUDY_NAME=sv2_rollout_uniform48_v1 \
ONE_STEP_STUDY=sv2_select_uniform48_v1 \
REPLICATES=3 HORIZONS=5,20,50 ROAD_REPLAY=1 \
RESUME=1 \
bash road_vrp7_step6c1_r3_sv2/run_sv2.sh rollout
```

完整后包含原R3-SV格式的`rollouts.jsonl`、`report.json`和每个state×H×repeat的SHA锁定unit。若修改原代码、抽样列表或预算，旧研究目录会拒绝续跑，请使用新名字。

## 5. 独立聚类分析

```bash
STUDY_NAME=sv2_analysis_uniform48_v1 \
ROLLOUT_STUDY=sv2_rollout_uniform48_v1 \
ONE_STEP_STUDY=sv2_select_uniform48_v1 \
BOOTSTRAPS=4000 ANALYSIS_SEED=20261024 \
bash road_vrp7_step6c1_r3_sv2/run_sv2.sh analyze
```

生成：

- `report.json`及方便汇报的`report.md`：TD-OD和Road.replay双目标、H=5/20/50下的状态级配对收益与95%状态重采样区间。
- `value_labels.jsonl`：每个状态、策略和H的3随机种子均值，包含H0 TD-OD、H后收益、种子标准差及完整配对收益；**只能作为描述性诊断标签，不可将已分析的Internal Test用于下一轮训练/调参**。
- `provenance.json`：rollout原数据及代码SHA。

报告主要关注：

| 检查 | 判定 |
|---|---|
| same-action placebo | 同一Section + 相同重启seed必须有完全相同结果，否则直接报错 |
| sign-flip/state | 同状态平均H后收益与一步收益方向是否反转，不将3随机种子伪装3状态 |
| rng-sign-instability | 同一state+arm+H的3个重启种子是否出现相反符号 |
| CI(state) | 以48个状态而不是144个重复为重采样单位，且报告任务间异质性 |
| objective check | `td_od`与事后`road_replay`结论是否一致；Road.replay既不是训练目标也不是在线可见信息 |

**重要限制：** 48个捕获状态可能共享原搜索轨迹与任务；即使按状态Bootstrap，也没有消除任务级依赖。最终需要训练任务中新的多步标注、保存可恢复ALNS内部状态，及独立月份/任务的冻结测试。不要据此声称标准意义的统计显著性。

## 6. 如果多步优势可以复现，下一阶段做什么？

- 若多数state的H0和H50排序方向确实冲突且跨随机种子稳定：在**训练任务**中采集独立多步标签，建模`E[H-step advantage | state, action]`，与一步目标比较，然后用DAgger-style收集学习策略自身轨迹。
- 若主要是种子间符号不稳定：优先研究预测不确定性和有支持的安全C3回退，而不是拟合噪声较大的H=50标签。
- 若H0/H50相关较强但Road.replay和TD-OD不一致：先修正评价目标与观测覆盖率，而不是贸然引入更深神经网络。
- 若H50始终不稳定且相比C3没正收益：停止扩大多步训练集，优先改进Section采样或计算预算。

所有独立研究结果应上传GitHub并**同时上传本包源码及SHA清单**，避免只保留实验结果无法审计。
