# Step6-C1-R4-N v1：Section候选排序与状态委托门控解耦实验

## 研究动机（已验证事实）

GitHub `Dragon1712/road_vrp_7` commit `b5e60b4a5b22152e0e8af96a81728641361324d5` 的R4-M报告显示：1,572个Train候选动作、200个场景快照、100个粗粒度搜索起点；外层4个Train任务留一、2个种子组迁移；全部四种嵌套LCB保守委托**零委托、零增益**。Ridge参考策略在任务留一合并后的平均K500道路收益约-0.934秒/快照。

这不能判定Section排序一定失败：单个绝对价值模型同时承担“是否委托”与“委托哪个Section”，而保守校准将两者一起关闭。R4-N用新实验分解两个机制：

1. **Action ranking**：条件相同的Section候选，学习`Δ500(a)-Δ500(b)`，是否能把较优Section排到前面？
2. **State gate**：是否能识别该搜索状态值得用替代Section而不是C3？

若Perfect Gate + Learned Rank有效、Learned Gate + Perfect Rank无效，则主要障碍是**状态门控**；反之则主要障碍是**动作排序**。两个Perfect/Oracle诊断都要读取外层留出未来标签，因此**不可部署、不得算作泛化模型收益**。

## 严格边界

- **不运行ALNS**，不更改C3、原R4-G～R4-M或Road.replay，使用冻结的R4-L原始Train数据校验（R4-M的`source_preflight`重新核对所有来源和算法SHA）。
- 使用原R4-H/R4-K 1,572个Section候选、200个快照，`K=50/200/500`源历史最优Road标签；R4-N排序当前只使用**K500**。没有独立Validation/Test。
- 所有模型输入仅使用`r4i_core.predecision_features`中的**Section决策前特征**。禁止将反事实Road收益作为输入特征。
- 外层按整个task留一，以及Seed 60/61→62/63、Seed 62/63→60/61；同一个状态的候选不跨Train/heldout。
- Pairwise训练只形成同一快照内部**不相等收益**候选的差分；同任务粗粒度搜索起点获得近似均匀权重；跨场景近重复搜索状态不是独立任务。
- 非零差分不足时，Pairwise模型停止委托，不伪造特征或标签。
- 训练标签Shuffle只发生在训练任务内部，保持单任务K500标签直方图，绝不打乱外层留出数据。
- Pairwise输出为**相对排序分数**，不是可校准的道路秒数；不能直接按`>0`委托。

## 策略与含义

| 代码策略 | 实际规则 | 未来收益参与决策？ | 部署资格 |
|---|---|---|---|
| `C3` | 永远不替换Section | 否 | 基准 |
| `ridge_ref` | R4-L Ridge最大预测收益大于0时委托 | 否 | 仅离线候选 |
| `pairwise_forced_early` | 强制早期状态选Pairwise最高分 | 否 | **危险诊断，禁止上线** |
| `pairwise_with_ridge_gate` | 原Ridge决定是否委托，Pairwise只负责候选排序 | 否 | 仅离线候选 |
| `oracle_gate_pairwise_rank` | 事后知道该状态存在正收益才开放Pairwise排序 | **是** | **不可部署** |
| `ridge_gate_oracle_rank` | Ridge门控；事后从动作中选最优并允许C3回退 | **是** | **不可部署** |
| `sampled_oracle_early` | 事后在早期已采样候选和C3中取收益最大 | **是** | 上界、不可部署 |

所有策略使用`stop_count<=2000`早期门控；训练/评估阶段固定、不做外层门槛调优。

## 服务器命令

在根目录执行：

```bash
cd ~/code/WZL/road_vrp_7
unzip -o road_vrp7_step6c1_r4n_v1.zip
bash road_vrp7_step6c1_r4n/run_r4n.sh test
bash road_vrp7_step6c1_r4n/run_r4n.sh preflight
STUDY_NAME=r4n_rank_gate_train_v1 bash road_vrp7_step6c1_r4n/run_r4n.sh probe
cat runs/chicago_july_v1/step6c1_r4n/r4n_rank_gate_train_v1/report.md
```

完整输出还包括`report.json`（168个分组/模型诊断单元）、`fold_predictions.jsonl`（各决策及是否Oracle条件）、`provenance.json`（代码和上游SHA）。仅仅运行`test`不需要Chicago数据或官方ALNS包。

若本地来源目录不是默认路径，使用`R4N_R4M_DIR`、`R4N_R4L_DIR`、`R4N_OLD_CF_DIR`、`R4N_OLD_CAPTURE_DIR`、`R4N_OLD_PREFIX_DIR`、`R4N_NEW_R4K_DIR`、`R4N_NEW_CF_DIR`、`R4N_NEW_CAPTURE_DIR`、`R4N_NEW_PREFIX_DIR`设置；所有非默认来源必须经现有SHA/Train角色检查。

**注意**：GitHub中R4-J Full曾缺少`report.json`的同步，而R4-L/M已经在真实服务器运行成功。此阶段要求现有R4-L/M源文件仍在服务器，且SHA未改变；如果缺失，应该恢复原始来源，不能绕过检验或仅使用已打印的汇总表伪造标签。

每次运行必须使用**新的`STUDY_NAME`**。`probe`如果输出目录已存在，会拒绝覆盖；不会自动续写过去的训练实验。

## 判读标准

**先查看**`oracle_gate_pairwise_rank`的Top-1真实正收益概率与`sampled_oracle_early`；再比较`ridge_gate_oracle_rank`。尤其分别观察task_0001的有害委托与task_0002的漏检。

- **Oracle Gate + Pairwise Rank较好，但Ridge Gate + Oracle Rank较差**：状态级门控是瓶颈，下一步建立显式“正收益状态/不委托”表征，并用更多真正独立Train任务测试。
- **Oracle Gate + Pairwise Rank仍差，而Ridge Gate + Oracle Rank表现较好**：当前Section几何特征不足以正确排序，应增加真实路网交通+Section边界表示、FiLM条件融合和成对对比学习消融。
- **Oracle条件两侧都差**：同时面临状态与动作表征不足；不能通过降低LCB阈值宣称算法获益。
- **Pairwise + Ridge Gate跨任务偶然为正，但存在明显伤害**：仍不得部署，需要独立Validation、墙钟/推理开销和保守选择协议。

本实验**不**生成独立任务，也**不**证明30秒在线ALNS改进。固定K标签分布与真实墙钟剩余预算不同。所有Oracle条件策略使用了留出未来标签，绝不可作为部署候选。
