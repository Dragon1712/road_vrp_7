# Step5-B：Section / Traffic-Sensitive Section 完整实验代码

针对 `https://github.com/Dragon1712/road_vrp_7` 新仓库。**只读取 Stage3 官方 ALNS 与 Stage5-A v2 TrafficDestroy；不覆盖任何已有 Stage3、Stage4、Step5-A 文件/结果**。代码是已完成实现，不是算法伪代码；本地已运行合成回归，但尚未在用户真实Chicago + 官方 `alns==7.0.0` 环境跑成功，因此正式实验必须先 `smoke`。

## 0. 你有一整年数据，7月26～31日的使用方式

可以把7月2～31日**全部划为开发数据**：用来研究算子、观察消融、调整配置。此前7月26～31日已经在Stage3中使用和观察，已不属于新算法的未见最终测试集；但这完全不妨碍把它们用于研究/训练。**当前Step5-B为非神经网络的ALNS启发式研究，并不需要用数据集训练网络**；这里的`train`/`validation`/`test`只是任务生成时的历史分区标签；`july_recent`从历史`test`标签读取数据，但现在的角色被显式重新定义为开发集。

从其余月份预留真正没看过的1～2个月，待选好最终算法后一次性评估，并为新月份重新构建动态成本和任务，避免把7月混合路网哈希误用到其他月份。

## 1. 代码目录及依赖

在新仓库根目录解压后：

```
road_vrp7_step5b/
├── section_operators.py      # 连续Section选择、分段保留、移动/反转/双段交换
├── run_experiment.py         # 官方ALNS + B0-B6消融、哈希/断点续跑
├── analyze_step5b.py        # 完整网格、配对Bootstrap、算子贡献日志
├── run_stage5b.sh           # 命令入口
├── tests/test_section.py    # 合成交通、破坏修复、静态无泄漏、API模拟
├── tests/test_analysis.py   # 完整性和不完整结果拒绝检查
└── README_使用说明.md
```

依赖：已存在的`road_vrp7_stage3/`、`road_vrp7_step5a_v2/`以及`numpy,pandas,networkx,alns==7.0.0`。使用原Stage3 `Road`, `OD`, `Objective`, `greedy`, `insert_removed`, `VRPState`, `BothBudget` 和原Stage5A `TrafficDestroy`（仅B6 Oracle模式）。

安装：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step5b_完整代码.zip
python -m pip show alns
# 未安装需提前在联网机器下载wheel，离线装入服务器：
# python -m pip install alns==7.0.0
```

## 2. 实际实现了什么？

客户Section是**路线中的连续客户序列**，不是道路OSM边集合。每次选1段（默认70%）或来自不同路线的2段（30%），每段2～3个客户，至少给每辆车留1个客户。修复时将整个Section视作不可拆分的块，在所有车辆/插入位置寻找容量可行的较优插入；尝试正向和反转，并在双段情况下评估跨车交换。每个候选解调用Stage3 `obj.route_cost()`（或者`obj.cost()`）重算有向时变成本，不允许用对称距离近似。若找不到可行方案或耗尽预算，回退原解，不将不可行解交给官方ALNS。

ALNS仍用官方`alns==7.0.0`，保留Stage3相同Random/Worst/Related三种破坏、Greedy/Regret2修复、轮盘赌奖励`[8,4,1,0]`衰减`0.8`、`autofit(worse=.05,accept_prob=.75,num_iters=5000)`以及5秒/最多5000次迭代。各新算子加入同一destroy池，**没有改变接受准则**；Section普通和交通版本用同一个Section破坏-修复结构，仅选择片段的分数不同。Section拥有完整独立的destroy及block-aware repair路径，不会被普通修复拆散。

### 七组预先定义消融（B0～B6）

| 编号 | 程序名 | 内容 | 因果对照意义 |
|---|---|---|---|
| B0 | `baseline` | Stage3三种破坏、两种修复 | 冻结官方ALNS基线 |
| B1 | `section_random` | 加随机Section | 算子池规模/随机片段扰动对照 |
| B2 | `section_static` | 静态道路代价较高的Section优先 | 普通结构化Section |
| B2c | `section_static_probe` | 计算交通特征但仍按静态评分选择 | **匹配动态特征计算开销**，排除仅耗时带来的差异 |
| B3 | `section_traffic` | 静态+OD拥堵波动+道路监测支持度代理 | 交通评分独立增量 |
| B4 | `section_shuffled` | 打乱Section与交通特征对应关系 | 避免“随机多样性”被误当作交通收益 |
| B5 | `at_rr_section` | B3+已有Step5-A v2 AT-RR（仅Oracle） | 互补性探索；算子数更多，不能单独视作等预算同算子池对照 |

重要比较：`B2−B0`说明Section结构价值；`B3−B2`说明交通方案的**净增益**（包含动态特征额外计算成本）；`B3−B2c`更好隔离交通评分内容；`B3−B4`检验特征是否需要正确绑定Section。`B5−B3`只是初步交互探索，算子数量未匹配。所有方法等搜索墙钟时间，但每次迭代开销不同，会同步报告。

### 交通特征及观测支持

- **Static**模式不计算动态评分、不访问未来`OD.dynamic`，B3退化为B2；最终道路级历史回放可依然使用时变成本，属于**评价**而非规划泄漏。
- **Oracle**模式使用已经存在的历史未来OD构造拥堵相对变化与20分钟趋势，作为能否利用已知交通状态的可行性上界，不是可部署预测。
- 每个客户Section的交通分数由进入、内部与离开Section的有向OD边组成，按服务时间前推；先对静态与交通项分别做当前候选集的10/90分位归一化，再按默认`traffic_weight=.45`组合。
- `PathSupportProxy`在独立预计算阶段，以OSM**静态最短路径**及每个OD采样时刻的观测掩码，计算长度加权观测支持占比，并缓存到`runs/chicago_july_v1/step5b/od_support_cache`。它不是当前动态最短路径的实测比例，也不证明每条OSM边上存在真实传感器观测。只有B3/B4/B5会调用；B2c也会计算交通特征，用于匹配开销。交通分数仅被支持度调节，未监测路径不会被硬拒绝。
- `support_precompute_sec`与`od_precompute_sec`单独记录、当前5秒搜索预算只包含搜索。真实部署端到端预算必须同时计入预计算；本阶段优先保持与Stage3可比的**在线搜索时间**，不可声称端到端等预算。

## 3. 运行顺序

```bash
cd ~/code/WZL/road_vrp_7
# 1) 合成回归，无需官方库及真实城市图
bash road_vrp7_step5b/run_stage5b.sh test

# 2) 真实7月官方ALNS接口冒烟：1任务×1验证时段×1种子×7组×Oracle = 7条
bash road_vrp7_step5b/run_stage5b.sh smoke
bash road_vrp7_step5b/run_stage5b.sh analyze smoke

# 3) Pilot：3任务×2验证时段×2种子×7组×Oracle = 84条
bash road_vrp7_step5b/run_stage5b.sh pilot
bash road_vrp7_step5b/run_stage5b.sh analyze pilot

# 4) validation：10任务×8时段×3种子×7组×2规划模式 = 3360条
bash road_vrp7_step5b/run_stage5b.sh validation
bash road_vrp7_step5b/run_stage5b.sh analyze validation

# 5) 以前的7月26-31日（10个通过QC的时段）作为进一步**开发**验证：4200条
bash road_vrp7_step5b/run_stage5b.sh july_recent
bash road_vrp7_step5b/run_stage5b.sh analyze july_recent_development

# 6) 7月27日08/17低覆盖开发压力测试：840条
bash road_vrp7_step5b/run_stage5b.sh july_stress
bash road_vrp7_step5b/run_stage5b.sh analyze july_stress
```

有必要时才运行July全30天可用时段（缺损QC时段默认排除、共58时段；10任务×58时段×3种子×7组×2模式=24360条）：

```bash
bash road_vrp7_step5b/run_stage5b.sh july_development
bash road_vrp7_step5b/run_stage5b.sh analyze july_development
```

全月计划至少需要~33.8个CPU小时的搜索时间（另加OD、路径支持缓存、道路回放），不推荐跳过Pilot直接运行。完整validation的累计5秒搜索时间约4.67 CPU小时（3360×5秒）；实际墙钟需依服务器而定。

中断续跑（**相同代码/配置/任务/道路哈希**，不能同时运行相同结果目录的多个进程）：

```bash
RESUME=1 bash road_vrp7_step5b/run_stage5b.sh pilot
```

断点恢复检查metrics.csv和solutions.jsonl中所有唯一主键一致，否则拒绝。分析器也要求满足完整预期网格；如果中途漏记录，将拒绝输出可发表的配对结论。

## 4. 数据、输出与验收

默认：

```
runs/chicago_july_v1/step5b/
├── od_support_cache/                    # 单独SHA绑定，不能用错误月份复用
├── smoke/{metrics.csv,solutions.jsonl,run_config.json,benchmark_summary.json,analysis/}
├── pilot/...
├── validation/...
├── july_recent_development/...
├── july_stress/...
└── july_development/...
```

输出包括平均道路回放成本、`paired_gains.csv`任务级Bootstrap区间、算子次数与调用耗时、Section整体重排统计、平均支持度代理、时变评分是否改变了首选Section、以及每组实际迭代数。运行时重点关注：

1. `benchmark_summary.status=COMPLETE`，并且已运行的行数严格等于计划网格；所有路线合法、每车至少一名客户。
2. `section_traffic_change_rate`显著大于0，否则交通评分未真实介入；对照B2c应保持静态选择，但承担相似交通特征计算开销。
3. B3是否优于B2c和B4；如果仅优于B0却不优于B2/B4，说明收益可能来自Section结构或随机性而不是交通信号。
4. `operator_contribution.csv`里Section调用的候选改善率、接受率、新最优次数、耗时；不是只比较调用次数。
5. `support_mean`是静态路径出发时刻的支持代理，不能当成完整动态路径实测比例。关注高低支持条件的稳健性。

## 5. 阶段与论文界限

- 这阶段是**组合优化算法的算子实验**，不是训练交通预测器/GNN/对比模型；7月数据用于历史动态代价和算子开发。
- ALNS oracle规划使用未来真实速度作为离线hindsight；不能声称真实部署能够获得未来交通。
- 原有Chicago观测支持稀疏，绝大多数道路为静态回退。提升即使存在也需在其他月份及新客户分布上独立检验。
- 本地合成测试已经通过，但官方ALNS 7.0.0真实集成+5503边实际成本未在此环境运行。Smoke、Pilot必须在服务器通过后才能开始3360条完整验证。
