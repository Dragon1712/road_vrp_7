# Step6-C1-R4-D v1：真实ALNS反事实标签的信号漏斗与可学习性审计

## 研究结论与设计动机

R4-A 的 20,000 次固定迭代无干预检查 8/8 通过；R4-B 原生 ALNS 7.0.0 精确恢复 64/64 通过；R4-C 16 个真实 Section 快照、125 个替代动作、四个 K={1,5,20,50} 完成了 628 个分支（含 C3/SHAM），提供 500 个动作×K 标签，64 个 C3/SHAM 完整轨迹等价验证全部通过。

目前历史最优解（`road_best_gain_sec`）K=1/5 的全部125个替代动作为零；K=20只有4/125非零（3正1负），K=50只有6/125非零（5正1负）。但当前接受解（`road_current_gain_sec`）在K=1/5/20/50的非零数量分别为48/38/39/39，明显不同于“所有动作都没有效果”的解释。

**R4-D 不是再训练模型，也不是再运行ALNS；而是用所有R4-C分支的原始每步事件来归因上述稀疏标签是发生在哪个决策环节。**

## 信号漏斗

针对每个 R4-C Section 反事实动作和每个 K：

1. 原始候选修复后的`candidate_route_sha256`是否与原 C3 不同。
2. 这个候选是否非平凡（不是原当前路线）。
3. ALNS 首步的 `outcome` 是否属于 `BEST/BETTER/ACCEPT`（0/1/2）；拒绝为 `REJECT=3`。
4. 最终当前接受解与 C3 是否不同。
5. 最终历史最优解与 C3 是否不同。
6. 当前解 Road.replay 的收益非零、正负比例。
7. 历史最优解 Road.replay 的收益非零、正负比例。
8. **每快照Oracle上界**：在这个快照已被评估的最多8个动作上，取`max(0, road_best_gain_sec)`；未抽样动作不在此Oracle的定义域内。
9. 对于同一个动作，重新检查K=1/5/20/50对应完整事件流的前缀一致性。
10. 检查相同算法快照（Task/Seed/Stop/Current/Best/RNG）在不同道路交通场景下的重复，**不要把它们误认为独立物理搜索起点**。

不把当前解Road收益直接当作最终优化收益；它仅用于解释连续搜索内部的潜在中间效应。

## 数据与实现边界

- 完全重用 `runs/chicago_july_v1/step6c1_r4c/r4c_pilot_16snaps_v1/`，含628份`units/*.json`。
- 完整逐项验证 R4-C `C3 == SHAM_C3` 轨迹、每个标签与底层分支结果的一致性、每个动作跨 K 前缀一致、样本身份/Plan身份及所有 source SHA。
- 样本来自此前已经看过的**原July Validation开发场景**，只能作为机制诊断，不能据此在Validation上调学习阈值后继续宣称独立测试效果。
- 16个快照及125个候选不是独立统计样本；两个不同日期相同的静态搜索状态可能构成重复物理轨迹，其Road.replay评价仍可能不同。
- 当前实验只验证固定K步的续接代理，没有恢复实际30秒余下墙钟预算；不主张形式化反事实识别保证。
- 没有置信区间：每快照只有一次配对随机数轨迹，且干预后随机消耗可能分叉。

## 安装与运行

服务器GitHub仓库根目录（已有R4-C及其依赖，无需安装新依赖）：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r4d_v1.zip
bash road_vrp7_step6c1_r4d/run_r4d.sh test
```

然后执行只读完整审计（扫描628个checkpoint，**不运行ALNS**）：

```bash
STUDY_NAME=r4d_funnel_pilot16_v1 \
  bash road_vrp7_step6c1_r4d/run_r4d.sh audit
```

已有同名目录请改用新名称，例如：

```bash
STUDY_NAME=r4d_funnel_pilot16_v2 \
  bash road_vrp7_step6c1_r4d/run_r4d.sh audit
```

如服务器数据根目录不同，可覆盖`R4C_DIR`、`R4B_DIR`、`R4A_CAPTURE_DIR`与`R4D_ROOT`环境变量。绝不修改或覆盖旧R4-A/B/C结果。

## 输出路径

`runs/chicago_july_v1/step6c1_r4d/<STUDY_NAME>/`

- `report.md`：按K汇总“Section候选→接受→当前解→最优解”的信号漏斗、每快照的已采样Oracle上界。
- `report.json`：机器可读的完整指标、物理搜索状态去重情况与解释限制。
- `candidate_funnel.jsonl`：500个动作×K的逐条归因记录，包括动作是否接受、候选改变、收益和起始最优-当前成本。
- `snapshot_oracle.jsonl`：16个起点×4个K的已采样Oracle最优收益，作为后续判定标签可学习性的上界诊断。
- `provenance.json`：冻结原报告/标签/摘要/Plan/source SHA及新分析代码SHA。

## 何时值得继续？

**先看漏斗的瓶颈：**

- 首步实际候选没有变化：应检查Section候选去重及修复算子 no-op。
- 多数候选不同但第一步被拒绝：应检查静态搜索目标、SA温度、接受准则/算子反馈的影响，而非直接训练多步回归。
- 当前解经常变化而历史最优长期不变：应明确它是best-so-far锁定问题，并将**前缀探索进展**或“到达更优静态解的时间”作为辅助诊断；不能直接把它替换为Road最优解收益。
- 只有极少数快照有可改善动作：可探索**选择更具决策价值的状态起点**，但新模型训练必须在原TRAIN任务重新采集这些状态，不可通过根据Validation结果挑选状态来宣称独立泛化。
- 如果要把K增至100/200/500，应先单独扩充R4-B的相应续接验证和deadline审计，再以新采集目录运行反事实。**本包不假装已经验证K>50。**

## 测试

9项测试覆盖：正常漏斗、SHAM破坏、标签篡改、跨K前缀篡改、缺失候选、跨场景物理重复、完整文件读写、源码SHA拒绝及Unit身份篡改。Python无额外第三方依赖，仅依赖仓库中已有纯Python `r4c_core.py`。
