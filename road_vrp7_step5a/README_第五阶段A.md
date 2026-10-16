# Step5-A：AT-RR + TE-WR（芝加哥2025年7月，官方ALNS 7.0.0）

**适配目标：** `https://github.com/Dragon1712/road_vrp_7`，按GitHub提交 `4c3be717e572976909f14b9987733b1f538d333c` 中的 `road_vrp7_stage3/road_vrp.py`、`solvers.py`、`benchmark.py` 接口实现。新代码全部在 `road_vrp7_step5a/` 目录；**不修改 Stage3/Stage4 代码及既有结果**。实际服务器数据必须自行运行，本包中没有声称完成Chicago实测实验。

## 1. 核心算子：精确定义（不能与旧Worst混淆）

### TE-WR：Traffic-Externality Worst Removal

Stage3现有 `worst` 计算的是整条路线删除客户的**总收益**，已隐含时变交通的一部分影响：

`total_gain(i) = C(route) - C(route without i)`。

本算子将其做精确分解：

`local_gain(i) = c(prev,i;t_prev) + c(i,next;t_depart_i) - c(prev,next;t_prev)`；

`externality(i) = total_gain(i) - local_gain(i)`。

其中路线成本 `C` 使用**原始Stage3的`Objective.route_cost`**，`t_prev`在前客户完成服务后计算，`t_depart_i` 在客户i服务结束时计算。注意：延误影响的是**下游全部客户行程的时变成本**，不是增加一次无依据的拥堵惩罚。

排序指标：`impact = abs(externality) + max(externality, 0)`，使“有利于缩短后续时间的正外部性”权重比“延误敏感、负外部性”高一倍；同时不会因为最后一位客户外部性恒为零而一直选中最后客户。这一排序指标是**有待消融验证的算法设计假设**，不是文献原公式。

* `te_wr` 仅按`externality`的带符号影响评分，不纳入`local_gain`或`total_gain`作为排名项；与Stage3 `worst`不是同一算子。
* `te_local` 只按`local_gain`排序，作为局部边收益对照。
* Static规划时，时变外部性恒为零。因此`te_wr`以不使用动态交通的随机排序退化为负对照，**不允许静态模式调用`OD.dynamic`**。
* TE计算对每个可删除客户重新评价一次route_cost。额外route_cost调用在`extra_traffic_route_cost_calls`中单独记录，Stage3的`score_calls`仅统计`Objective.cost`，不是完整求解开销。

### AT-RR：Arrival-Time Related Removal

对当前**车辆客户顺序**调用`route_trace()`，用原始`OD.travel`逐段推到客户到达时刻（含服务时间）。随机选锚点，按以下距离选出关联客户：

`R(i,a)=0.35 × normalized_static_sym_distance + 0.30 × |ETA_i - ETA_a|/1200 + 0.10 × |demand_i-demand_a|/capacity + 0.15 × |level_i-level_a| + 0.10 × |trend_i-trend_a|`

其中 `level_i(t)`是客户i到最多四个**按静态道路距离选择的近邻**的历史时变OD成本与静态OD成本比值的均值，`trend_i(t)=level_i(t+1200s)-level_i(t)`。使用原始OD时间栅格进行线性插值；测试时不能把这一交通代理特征称为真实实测路段拥堵或独立交通预测。

* `at_rr` 使用上述5项特征。
* `at_timeonly` **只删除后两项交通差异**，保留前三项相同系数，便于检验交通代理特征的增量效果。
* Static模式下`at_rr`的额外交通两项被关闭，和`at_timeonly`使用相同表达式，绝不读取未来交通。
* Oracle模式使用当日实际后续交通，严格限定为**历史回放（hindsight），不能直接部署**；实际在线算法需要历史可用信息/预测模型替代。

## 2. 为什么这一版不使用Section和GNN？

Step5-A的目的是首先验证两个**人工设计、可解释**的交通移除算子是否具有独立作用。Section是Step5-B，Graph RL/多模态+对比学习为下一阶段。先冻结基础求解器，再添加新机制，保持因果解释尽可能清楚。

## 3. 七组预注册消融

| 名称 | Destroy portfolio | 与谁比较 |
|---|---|---|
| `baseline` | random + worst + related | Stage3 `official_alns`，验证基线复现 |
| `at_timeonly` | baseline + at_timeonly | 与AT-RR配对 |
| `at_rr` | baseline + AT-RR | 交通特征增量与基线 |
| `te_local` | baseline + te_local | 与TE-WR配对 |
| `te_wr` | baseline + TE-WR | 外部性效应与基线 |
| `both` | baseline + AT-RR + TE-WR | 是否有互补收益 |
| `placebo_both` | baseline + 一份Related复制 + 一份Worst复制 | 算子池规模控制 |

所有组均沿用Stage3的 `Greedy/Regret2` Repair、官方RouletteWheel、同一接受准则`SimulatedAnnealing.autofit(init_obj, worse=.05, accept_prob=.75, num_iters=5000)`以及`BothBudget(max_iter=5000, wall=5)`；除了候选的Destroy组合外不改变路线表示或道路代价。

`placebo_both`只能匹配算子池大小，**不能匹配精确每迭代计算开销**；因此还要同时比较search_wall_sec、iterations、extra_traffic_route_cost_calls。增加一个重复算子可能改变ALNS权重分布，这是实验设计的一部分，不能宣传为严格独立的“同等计算量”试验。

## 4. 运行操作（服务器离线，只依赖已有Python环境）

把解压后的 `road_vrp7_step5a/` 放在**新7月仓库根目录**；已有Stage2道路数据、Stage3任务、OD缓存均维持原路径。

```bash
cd ~/code/WZL/road_vrp_7
python -m pip show alns
# 要求官方 alns == 7.0.0；服务器无法外网时沿用Stage3已安装环境

# 单元测试：合成OD+模拟ALNS接口（不需要真实7月数据）
bash road_vrp7_step5a/run_stage5a.sh test

# 真正官方ALNS 7.0.0冒烟：1任务×1场景×1种子×2规划模式×7变体 = 14条
bash road_vrp7_step5a/run_stage5a.sh smoke
bash road_vrp7_step5a/run_stage5a.sh analyze smoke

# Pilot：2任务×2场景×2种子×2规划模式×7变体 = 112条
bash road_vrp7_step5a/run_stage5a.sh pilot
bash road_vrp7_step5a/run_stage5a.sh analyze pilot

# 完整验证集：10任务×8场景×3种子×2规划模式×7变体 = 3360条
bash road_vrp7_step5a/run_stage5a.sh validation
bash road_vrp7_step5a/run_stage5a.sh analyze validation
```

Smoke预算0.4s（仅检测接口，**不可**用于与Stage3的5s比较）；Pilot/Validation使用5s搜索+最多5000次官方ALNS迭代上限。完整实验搜索预算上界3360×5=16800秒（约4小时40分CPU搜索时间），加上OD、道路回放等会更久；不要在同一目录同时开启多个进程。

中断后：

```bash
RESUME=1 bash road_vrp7_step5a/run_stage5a.sh validation
```

如果中断刚好发生在`solutions.jsonl`和`metrics.csv`两次落盘之间，续跑会检测到不一致并拒绝静默恢复。此时检查日志并人工修正checkpoint，或用新目录重新运行。切记不要覆盖Stage3/4结果。

### 仅做小规模或自定义算子组

```bash
python road_vrp7_step5a/run_experiment.py \
  --output-dir runs/chicago_july_v1/step5a/custom_2tasks \
  --variants baseline,at_timeonly,at_rr,te_local,te_wr,both,placebo_both \
  --planners static,oracle --seeds 42,43 \
  --max-tasks 2 --max-scenarios 2 --wall-sec 5 --max-iterations 5000
```

必须用新的`--output-dir`。分析可直接调用`python road_vrp7_step5a/analyze_step5a.py --stage5-dir ... --output-dir ... --stage3-dir runs/chicago_july_v1/stage3/validation`。

## 5. 结果结构与验收条件

输出在 `runs/chicago_july_v1/step5a/{smoke,pilot,validation}/`：

* `run_config.json`：路网、任务、Stage3源代码和Step5源代码SHA256 + 预算 + 算子集合。
* `metrics.csv`：每任务、情景、seed、规划模式的准确道路级外生回放成本；还包含搜索时间、迭代次数、原始`score_calls`、额外route_cost调用、失败修复次数。
* `solutions.jsonl`：全部路线、Destroy/Repair调用次数、TE外部性汇总。
* `benchmark_summary.json`：完整网格统计。
* `analysis/solver_summary.csv`、`paired_gains.csv`、`analysis_summary.json`：同任务/场景/seed严格配对，按任务抽样的探索性Bootstrap区间，Stage3验证配置核查。

验收：完整验证3360/3360；道路成本哈希与Stage3一致；Pilot每个算子族确有调用，TE非零外部性调用数在Oracle模式下大于零；Static下TE静态回退数大于零；`AT-RR`和`at_timeonly`的Static机制相同且绝不依赖未来OD.dynamic；所有路线满足30客户各配送一次、恰好5辆车非空、容量限制；真实行驶成本依旧通过同一`road.replay`计算。

## 6. 如何读结果，及不应宣称的内容

1. `at_rr vs at_timeonly`：直接回答交通拥堵水平/趋势是否在ETA和空间信息之外提供收益；必须查看两者的迭代次数和公平预算。
2. `te_wr vs te_local`：外部性排序vs局部收益排序；再与`baseline`比较，以确认是否胜过既有总收益Worst。
3. `both vs placebo_both`：排除只是扩大算子池导致的收益（并不排除开销差异）。
4. 在`static`条件下，AT-RR并无实时/未来交通签名，TE-WR也不会访问未来OD；这是防止不公平信息比较的控制。
5. 所有Oracle结果是使用过去真实未来状态的**回顾性试验**，没有训练学习模型，不能说已形成部署时可获得的交通感知策略。
6. 观测边比例很低：历史混合路网大量边由OSM静态成本填补。即便算法提升，也不能推断对真实交通拥堵的泛化一定成立。后续须使用新的未见月份与覆盖敏感性试验。

## 7. 本地验证的真实范围

本包已通过：

- TE-WR外部性恒等式；静态OD外部性为零；动态OD非零；合成数据上TE-WR选出的客户**不同于Stage3 Worst**；
- AT/TE在合成道路上的客户完整性与车辆非空条件；Static模式禁止访问动态OD；
- 7种变体的**模拟ALNS**协议测试、严格配对审计、Stage3配置不一致拒绝机制。

**没有在当前容器运行真正官方ALNS 7.0.0或完整7月Chicago数据。** 真正的官方库与历史道路集成必须在你的服务器执行`smoke`后再确认。这个合成测试不构成有效果证明。
