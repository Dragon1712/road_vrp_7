# Step5-A v2：交通特征真实性、算子贡献与严格消融审计

项目：`Dragon1712/road_vrp_7`。源代码审阅提交：`0cf000ce4d1240b109f9160a3a3f59208825cde7`。本工具读取仓库现有Stage3及2025年7月数据，只在**冻结验证集2025-07-22～25**运行，不写入旧Step5-A、Stage3/4结果目录。

## 1. 当前Pilot证据（旧v1结果，非v2新跑结果）

完成：`112/112`，2任务×2交通情景×2随机种子×2规划模式×7变体；每组搜索5秒。

| 配置 | Static平均道路旅行秒 | Oracle平均道路旅行秒 |
|---|---:|---:|
| baseline | 4338.81 | 4365.52 |
| at_timeonly | 4331.47 | 4320.36 |
| at_rr | 4331.47 | **4291.80** |
| te_local | 4393.77 | 4321.80 |
| te_wr | 4322.95 | 4364.44 |
| both | 4331.75 | 4329.64 |
| placebo_both | 4349.73 | 4302.92 |

AT-RR在Oracle下比AT-TimeOnly低28.56秒，但仅2任务×2场景，不保证独立收益；TE-WR比TE-Local高42.64秒，需解释下游外部性排序为什么可能不利。现有日志只有选择次数，无法证明交通特征实际上改变了移除集合。

## 2. v2新增机制（保留原v1代码于另一路径）

### AT-RR：配对的影子排序与分项得分

在每次真实`at_rr`调用中，对相同当前解、同一锚点、同一q、同一随机微扰，同时生成“无拥堵水平及趋势”的时间相关排序。

- `at_shadow_compared`：配对次数；`at_shadow_changed`：最终破坏集合不同次数；`at_shadow_change_rate`：实际变更率；`at_shadow_mean_jaccard`：移除集合Jaccard。
- `at_feature_mean_parts`：距离、ETA、需求、交通水平、交通趋势五项对排序分数的平均数值贡献（单位是归一化分数，不是秒）。
- `at_space`：严格**无动态OD特征**的空间+需求对照；`at_timeonly`：空间+需求+当前规划ETA，不含拥堵水平与趋势；`at_rr`：再加入OD的拥堵水平和趋势；`at_misaligned`：将客户交通特征身份错配的负对照，保留ETA等其他部分。
- STATIC模式保留静态信息防火墙，不读取OD.dynamic。`at_rr`静态版与`at_timeonly`应行为一致。

**限制：** AT的交通特征由20分钟采样的动态OD与静态OD比值导出，其支持道路并非全部真实监测，尤其全路网观测边-时间支持约6.7%。新增`audit_traffic_support.py`独立验证客户静态近邻对在当前ETA下的动态最短路径究竟覆盖多少监测走廊分摊的OSM边，不能把其测量支持率视为全路网直接观测率。

### TE-WR：下游外部性与正/负效应

保留严格恒等式：`total_gain = local_gain + downstream_externality`（旅行时间代理成本，不是精确GPS路段回放）。

- `te_static_local`：使用不随时间变化的静态OD局部接边收益（完全不调用动态OD，亦可在Oracle实验作为信息隔离对照）；`te_local`：时变OD局部接边收益；`te_wr`：外部性幅值`abs(e)+max(0,e)`优先；`te_positive`：仅对正外部性`max(0,e)`排序。
- TE-WR与Stage3 Total-Worst影子排序结果逐调用比较：`te_shadow_compared/changed/change_rate/Jaccard`。
- 记录被移除客户中外部性正值和负值的数量、均值，避免将大负外部性误称为“节省交通成本”。
- 不移空任何车辆；修复和最终道路回放仍用Stage3共用代码。STATIC TE-WR原样保留随机回退，专门视作负对照，不与Oracle下的外部性效应混为一谈。

### Operator contribution日志

使用**官方ALNS选择/接受准则的包装器**，不替换真正的SimulatedAnnealing，所有计时纳入搜索5秒。

- `operator_contribution.per_destroy`：调用数、总Destroy耗时、Repair耗时、正候选数、接受次数、接受劣解次数、新历史最优次数、候选总收益。
- `operator_contribution.per_pair`：Destroy×Repair组合的统计，避免把修复效果完全算作Destroy本身。
- `sample_events`：每次运行保留最多前120次事件样本，完整调用汇总无采样截断。
- 输出以`solutions.jsonl`作为持久事实源，`analyze_step5a.py`转换为`operator_contribution.csv`、`operator_pair_contribution.csv`、`sample_operator_events.csv`及`feature_effectiveness.csv`。
- **注意**：这些是条件关联统计，不是算子因果贡献证明；由于增加了日志计算，v2的Stage3 baseline重测与原Stage3可能在5秒搜索预算下存在小差异；以v2内部baseline进行主要配对。所有变体采用相同日志包装器。

## 3. 新实验矩阵（11组，每组均同时运行static/oracle）

| 变体 | Destroy集合增加项 | 主要问题 |
|---|---|---|
| baseline | 原Random/Worst/Related | 本轮公平带日志参考 |
| at_space | 静态空间+需求 | 到达时间是否有独立作用 |
| at_timeonly | 空间+需求+ETA | ETA以外交通特征贡献 |
| at_rr | 空间+需求+ETA+水平/趋势 | AT-RR主机制 |
| at_misaligned | 错配客户交通签名 | 有意义交通配对 vs 任意交通数值 |
| te_static_local | 静态局部接边收益 | 动态局部成本是否有价值 |
| te_local | 时变局部接边收益 | Externality增量对照 |
| te_wr | 绝对+正值加权的下游外部性 | TE-WR主机制 |
| te_positive | 仅正外部性 | 检验负外部性探索是否有用 |
| both | AT-RR + TE-WR | 两类机制相互作用 |
| placebo_both | 原Related/Worst克隆算子 | 算子池数量安慰剂 |

方法选择固定`RouletteWheel(scores=[8,4,1,0],decay=.8)`、`SA.autofit(worse=.05,accept_prob=.75,num_iters=5000)`。Planner Oracle使用历史未来信息，必须标为**hindsight**。不把Static下退化输出解读为真实交通价值。

## 4. Ubuntu服务器上直接运行

把压缩包放到`~/code/WZL/road_vrp_7`根目录。解压后应出现`road_vrp7_step5a_v2/`，保持旧的`road_vrp7_step5a`目录不变。

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step5a_v2_增强审计代码.zip
python -m pip show alns        # 必须为7.0.0

# 0. 单元测试：数学、future-info firewall、影子排序、日志归因
bash road_vrp7_step5a_v2/run_stage5a_v2.sh test

# 1. 独立OD观测支持度审计，不会改变搜索模型参数
bash road_vrp7_step5a_v2/run_stage5a_v2.sh support

# 2. 接入真实官方ALNS的22条Smoke结果：11 × 2模式 × 1seed
bash road_vrp7_step5a_v2/run_stage5a_v2.sh smoke
bash road_vrp7_step5a_v2/run_stage5a_v2.sh analyze smoke

# 3. 176条Pilot：2任务×2场景×2种子×2模式×11配置
bash road_vrp7_step5a_v2/run_stage5a_v2.sh pilot
bash road_vrp7_step5a_v2/run_stage5a_v2.sh analyze pilot

# 4. 请先审计Pilot，再决定是否执行完整验证
bash road_vrp7_step5a_v2/run_stage5a_v2.sh validation
bash road_vrp7_step5a_v2/run_stage5a_v2.sh analyze validation
```

完整Validation理论记录`10任务×8情景×3种子×2模式×11配置=5280条`，累计搜索时间约7小时20分CPU墙钟，不含预计算和道路回放。不要直接把v1结果追加到v2输出目录。若服务器中断，在确认代码完全未改变的前提下可`RESUME=1 bash ... pilot`。

不想在Pilot中先跑全部11组，可在启动前预先固定子集，如：

```bash
METHODS=baseline,at_space,at_timeonly,at_rr,at_misaligned \
 bash road_vrp7_step5a_v2/run_stage5a_v2.sh pilot
```

不同METHODS生成不同`run_config.json`，不可以在同一输出目录续跑其它子集。所有结果需要配对统计网格完整。

## 5. 文件与验收

```
runs/chicago_july_v1/step5a_v2/
├── feature_support/{od_traffic_support.csv,support_summary.json}
├── smoke/{metrics.csv,solutions.jsonl,run_config.json,analysis/...}
├── pilot/{metrics.csv,solutions.jsonl,run_config.json,analysis/...}
└── validation/...
```

- 不应出现缺失/重复的`task×scenario×variant×planner×seed`记录；不同交通时段应严格独立。
- 必须能生成`feature_effectiveness.csv`、`operator_contribution.csv`、`operator_pair_contribution.csv`和`sample_operator_events.csv`，行键需和`metrics.csv`一致。
- Oracle AT-RR应有`at_shadow_compared>0`；Static的影子次数应为0。
- Oracle TE-WR应有`te_shadow_compared>0`，并能统计正负外部性；Static不读取未来动态OD。
- 报告平均收益与任务级Bootstrap置信区间，不凭2个任务的小Pilot均值筛选“最优参数”。
- 若OD特征路径的监测支持率极低，不能直接宣称证明真实监测交通有贡献；仅可声称`historical OD`表征的搜索信号。

## 6. 未消除的科学限制

1. 多模态/对比学习尚未训练；此阶段仅是非学习式交通感知算子机制。
2. Chicago Loop数据是监测走廊速度+大范围OSM静态回退，无法表示完整实测交通。
3. Stage3/4的7月26～31日测试集已被分析，今后同批日期不再是严格盲测，建议另找月份作为最终泛化评估。
4. 官方ALNS真实7.0.0当前容器无法安装；只在合成接口上通过，需服务器Smoke确认，不能编造真实端到端收益。
5. 相同5秒预算只保证时限比较，允许不同算子产生不同迭代数；新增日志也消耗预算，必须报告实际迭代数/耗时。
