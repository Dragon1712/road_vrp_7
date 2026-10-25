# Step6-C1-R4-O v1：Section决策前路线结构表示与独立门控诊断

## 1. 当前依据

GitHub `Dragon1712/road_vrp_7` commit `98c16e14a28a4edb16ba6145d1af7d68aad2cbec`，R4-N真实报告：

- Train 4任务、种子60/61/62/63、200场景快照、1,572候选动作。
- 四任务留一：Ridge+原排序平均K500 Road增益-0.934秒/快照；Ridge门控+Pairwise排序-1.260秒/快照（200状态合计）。
- “Oracle门控+学习Pairwise” +1.532秒/快照，但仍有22次负收益动作；事后Oracle全部最优 +18.563秒/快照。
- “Ridge门控+Oracle动作”+4.169秒/快照，但该策略也使用真实未来收益来选择动作及拒绝负收益，不能部署或当作纯排序效果。
- 在具有已采样正收益候选的61个Train留组场景状态上，Pairwise Top1真实正收益命中24个；排序仍不稳定。

**R4-O研究问题**：原始15维手工几何状态特征是否丢失了可预测Section价值的路线局部结构信息？在同一外层划分上，独立的状态门控是否优于已有Ridge门控？

## 2. 冻结部分与数据安全

不修改 `road_vrp7_step5b_v3`、官方ALNS、R4-G/H/I/J/K/L/M/N 或任何已完成的输出。输入仅为现有Train来源（全部SHA审计）；模型特征仅来自原R4-G Section**决策前**的`current_routes`、`best_routes`、迭代/退火/剩余搜索墙钟等字段，以及事先列举好的候选Section `(route,pos,size)`。客户ID只用于判断节点相等/所在路线/相邻关系，不作为数值特征。**禁止在输入中使用Road.replay、候选未来轨迹、K后状态、事后交通实测信息。**

真正的时变交通要接入未来模型时，应首先确定在决策时可获得的预测/历史可用交通源，不能将R4-H/K用于标签的历史实际道路结果当作实时路况输入。

数据规模、任务角色、上游R4-L/M/N报告与代码SHA、全部候选动作身份、输出路径均严格核查；不能用R4-N的Oracle标签做模型训练特征。

## 3. 特征消融

| 输入配置 | 含义 | 维度 |
|---|---|---:|
| `baseline15` | 原R4-I 15维Section几何+状态 | 15 |
| `route_context` | baseline+23维路线结构：Section是否保持最佳路线邻接、客户在最佳解中的归属、Section两端边界、车辆路线长度不均衡和相对位置等 | 38 |
| `route_context_interact` | 上述再添加12组预定义的结构交互项，不根据heldout选择 | 50 |

- 排序器：同状态两候选的相对K500 Road收益差分，线性Pairwise Ridge、按任务及粗粒度搜索起点加权；无法形成非零差分则标记`fitted=false`。
- 门控器：**另一套**状态级二分类模型；训练标签`是否存在至少一个正收益Section`，只读取外层训练集；任务/粗粒度起点平衡、单一类别即回退C3。
- 门控门槛预先固定为0.5，所有策略仅在原始`stop_count<=2000`早期状态考虑委托。它是诊断门槛，**不是**具备风险保证的部署阈值。
- Shuffle负对照：在外层**训练**任务内相同动作数量的状态间置换完整的动作收益向量，保持标签边际分布；留组标签原封不动。

## 4. 策略（7种）

| Policy | 门控 | 排序 | Oracle? |
|---|---|---|---|
| C3 | 永远不委托 | 无 | 否 |
| ridge_ref | 原Ridge>0 | 原Ridge | 否 |
| pairwise_ridge_gate | 原Ridge>0 | Pairwise | 否 |
| learned_gate_pairwise | 独立状态级门控>0.5 | Pairwise | 否（仅离线候选） |
| oracle_gate_pairwise | 事后真实有收益状态 | Pairwise | **是，不可部署** |
| learned_gate_oracle_rank | 独立状态门控 | 事后最佳且允许拒绝亏损 | **是，不可部署** |
| sampled_oracle | 事后在C3/候选中选最优 | 事后 | **是，不可部署** |

四个Train任务分别留一，以及旧种子(60/61)→新种子(62/63)、反向种子迁移，总计6折。3组特征×2组Shuffle×7策略×6折=252配置。所有对照同时报告每折委托次数、实际正/负收益、K500均值、真实正收益状态Top1命中数，以及基于每状态正收益动作比例计算的**随机Top1期望**。

**Oracle门控和Oracle排序上界只能诊断瓶颈，不能导出部署模型。** 所有结果仍为Train内部固定500步离线反事实；没有独立Validation/Test或真实30秒性能改进。

## 5. 服务器操作

在仓库根目录：

```bash
cd ~/code/WZL/road_vrp_7
unzip -o road_vrp7_step6c1_r4o_v1.zip
bash road_vrp7_step6c1_r4o/run_r4o.sh test
bash road_vrp7_step6c1_r4o/run_r4o.sh preflight
STUDY_NAME=r4o_struct_rank_gate_train_v1 bash road_vrp7_step6c1_r4o/run_r4o.sh probe
cat runs/chicago_july_v1/step6c1_r4o/r4o_struct_rank_gate_train_v1/report.md
```

要求环境中有`numpy`，以及已经安装的R4-I/L/M/N源代码及全部原始Train实验文件。R4-L/M/N已在服务器运行成功，正常无需新安装ALNS或重跑历史反事实。

如上游目录不同，脚本提供`R4O_R4N_DIR`、`R4O_R4M_DIR`、`R4O_R4L_DIR`、`R4O_OLD_CF_DIR`、`R4O_OLD_CAPTURE_DIR`、`R4O_OLD_PREFIX_DIR`、`R4O_NEW_R4K_DIR`、`R4O_NEW_CF_DIR`、`R4O_NEW_CAPTURE_DIR`、`R4O_NEW_PREFIX_DIR`。如文件SHA不匹配，应恢复原始文件而不是关闭校验。

实验输出：`report.json`、`report.md`、`fold_predictions.jsonl`、`provenance.json`。必须使用全新`STUDY_NAME`，严禁覆写旧实验。`preflight`不创建结果目录，不运行ALNS。

## 6. 判读、下一步

1. **先看随机Top1对照**，否则Top1命中率无法解释；即使命中率更高，仍检查伤害总和与跨任务一致性。
2. 若路线结构特征比原15维稳定改善Top1（4任务LOTO+两个seed迁移），值得继续增加**决策时可用**的静态OD或真实预测交通模态，比较直接拼接、FiLM、对比目标。
3. 若结构改善排序但独立门控仍不可靠，优先收集更多**独立Train任务**，明确状态级收益机会标签的覆盖。
4. 若结构与门控均无改善，不能通过调整阈值宣称成功；应先验证候选Action和边界交通表示质量，适当增加真正独立任务。
5. 在任何在线实验前，单独验证推理开销、无效委托率、原C3冻结一致性，并使用事先冻结的独立Validation协议。

## 7. 局限

- 200场景快照中仍只存在4个独立配送任务，随机种子/场景相关性高。
- `best_routes`在决策前存在，所以可作为特征；这不等于未来最优解。
- 时间维度K500不是30秒的剩余墙钟预算；离线收益不可直接视作在线收益。
- 当前包**没有引入真实交通模态**，故不能报告多模态或对比学习增益。其目的为证明更有信息的结构化Section表征是否值得继续融合。
- 本地测试为合成单元/流程，真实Chicago服务器probe运行前，无法知道R4-O是否优于R4-N。
