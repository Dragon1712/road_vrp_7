# road_vrp_7 第四阶段：预算公平性与时间感知退火消融（验证集限定）

对应GitHub `Dragon1712/road_vrp_7`、审核提交 `a35ba8f780f34490bdb4334f37cc270a75678a83`。

## 0. Stage3基线审核结果（GitHub上**已执行**，不是本工具重新跑出的数据）

三个实验网格均完成：validation=1920/1920；test_eligible=2400/2400；test_july27_stress=480/480。一致的道路成本SHA256 `3612b66ad23df15ed409c86f2585755d4393d7a9044c38dd3ee1a30e5b49dd68`，10任务、随机种子42/43/44、Static/Oracle两个规划模式。

| 场景 | Greedy Static/Oracle | SA Static/Oracle | VND Static/Oracle | ALNS Static/Oracle |
|---|---|---|---|---|
| validation：8场景 | 5511.5 / 5537.1 | 4781.7 / 4810.8 | 4667.2 / 4645.4 | **4185.3 / 4218.4** |
| test eligible：10场景 | 5608.8 / 5644.2 | 4879.7 / 4913.0 | 4771.1 / 4695.0 | **4282.2 / 4298.8** |
| test July27 stress：2场景 | 5387.3 / 5432.1 | 4651.8 / 4678.9 | 4556.1 / 4521.8 | **4064.7 / 4108.1** |

单位为秒，属于相同7月混合实测+静态回退、FIFO道路级历史回放成本。Oracle使用完整历史未来信息，仅作hindsight上界/参考，**不是可部署预测**。注意：因为近似优化及时间预算的限制，Oracle解并非全局最优，不能称它为VRP全局最优下界。

官方ALNS与VND在测试集上的任务级配对改善为Static `10.19%`、Oracle `8.35%`。压力测试为Static `10.73%`、Oracle `9.12%`。但Stage3 SA/VND通常不足0.6秒就被5000次迭代截断，ALNS搜索5秒。**不能直接把该差距归因于ALNS算子本身。**

## 1. 本次实验假设

只使用已冻结的**7月22～25日验证集**。不使用实验6/7测试结果筛选参数，不覆写既有Stage3数据。

| 变体 | 对照基线 | 共同搜索预算 | 主要变化 | 解读警告 |
|---|---|---|---|---|
| `sa_timecool` | Stage3 `sa` | 5秒 | 移除5000迭代瓶颈，并用**墙钟时间**指数退火 | 同时更改预算利用与退火，不是纯粹温度单因素消融 |
| `vnd_multistart` | Stage3 `vnd` | 5秒 | 固定原VND邻域，扰动重启充分利用预算 | 重启本身也是算法变化，不能称为纯预算控制 |
| `alns_timecool` | Stage3 `official_alns` | 5秒 | 相同破坏/修复算子和轮盘赌，**仅把退火依时间推进** | 5秒壁钟ALNS真实集成尚需服务器Smoke |

新ALNS的初始温度设为 `T0=(0.05×初始目标)/(-ln(0.75))`，使初始时比当前解差5%的候选以75%概率被接受；时间进度`p=elapsed/wall`时 `T(p)=T0×(0.005)^p`。这个温度轨迹属于**待验证假设**，不是已经证实最优的参数。为了匹配计算时间，本阶段新变体默认最大迭代上限100万次，常规情况会触发5秒停止。输出明确记录是否确实用满搜索预算。

## 2. 安装与依赖

在**新的** `road_vrp_7` 仓库根目录解压，预期得到：

```
road_vrp7_stage4/
├── budget_experiment.py
├── analyze_budget.py
├── run_stage4.sh
├── README_第四阶段.md
└── tests/test_stage4.py
```

此包直接引用新仓库中的 `road_vrp7_stage3/road_vrp.py`、`solvers.py`、`tasks.py` 和 `benchmark.py`，不依赖旧 `road_vrp` 六月数据。需要 Python 3.10+ 及 numpy、pandas、networkx 和官方 `alns==7.0.0`：

```bash
cd ~/code/WZL/road_vrp_7
python -m pip install 'alns==7.0.0'
python -c 'import alns; from alns import ALNS; print("official ALNS import PASS")'
```

**别把项目Git仓库里的旧`stage3/validation`结果覆盖掉**；本工具写入新目录 `runs/chicago_july_v1/stage4_budget/`。

## 3. 按顺序运行

```bash
cd ~/code/WZL/road_vrp_7
# 1. 合成检查（ALNS以模拟API代替；不证明真实第三方库可执行）
bash road_vrp7_stage4/run_stage4.sh test

# 2. 首先测试真正的官方ALNS 7.0.0，单任务/场景/种子 × 3变体 × 2模式 = 6条结果
bash road_vrp7_stage4/run_stage4.sh smoke
bash road_vrp7_stage4/run_stage4.sh analyze smoke

# 3. 两任务、两验证场景、两种子 × 3变体 × 2模式 = 48条结果
bash road_vrp7_stage4/run_stage4.sh pilot
bash road_vrp7_stage4/run_stage4.sh analyze pilot

# 4. 10任务、8验证场景、3种子 × 3变体 × 2模式 = 1440条结果
bash road_vrp7_stage4/run_stage4.sh validation
bash road_vrp7_stage4/run_stage4.sh analyze validation
```

*Smoke预算0.35秒、上限5000次，不是最终科学比较。Pilot/Validation使用默认5秒、最大100万迭代。*

Validation阶段总搜索预算约7200 CPU秒（1440×5秒），不含OD、解构造、道路回放与文件写入。视机器而定预计超过2小时。建议先检查Pilot的CPU耗时、实际搜索时间、内存和解可行率。**不要为了节省时间在同一目录并行运行多个进程**。如果被中断且结果中的`metrics.csv`和`solutions.jsonl`完全一致，可以设置`RESUME=1`继续。中断发生在两文件写入之间时，会拒绝静默续跑，请先人工恢复一致性。

```bash
RESUME=1 bash road_vrp7_stage4/run_stage4.sh validation
```

## 4. 输出和关键验收门槛

```
runs/chicago_july_v1/stage4_budget/
├── smoke/
├── pilot/
└── validation/
    ├── run_config.json       # 数据、算法源码、方案、任务SHA和预算
    ├── metrics.csv           # 1440条预期；费用、时间、迭代数、观测路径占比
    ├── solutions.jsonl       # 相同run key的合法路线与算子统计
    ├── benchmark_summary.json
    └── analysis/
         ├── stage4_solver_summary.csv
         ├── paired_gains.csv
         └── analysis_summary.json
```

建议的验收检查：

1. `benchmark_summary.json.status == COMPLETE` 且 `expected == actual == 1440`，`analysis_summary.verified_complete_grid=true`；
2. 三个新变体的`mean_search_sec`接近5秒，`fraction_spent_90pct_wall`尽可能接近100%（VND局部停滞或迭代耗尽要单独解释）；
3. 只读取`Stage3/validation`的1920条**已冻结基线**作配对比较，不使用测试集；
4. 对ALNS时间降温变体重点看Oracle模式能否缩小相对Static的差距，同时观察随机种子间方差、道路观测边占比；
5. 报告整个验证集和每任务配对Bootstrap区间；胜率/置信区间不能替代独立日期的泛化检验。

## 5. 怎样解释结果与下一阶段

- 若5秒`sa_timecool`追近ALNS，说明Stage3的部分ALNS优势源于预算不等；不应撤销Stage3数据，而应在论文中追加预算公平性控制。
- 若`alns_timecool`改善Oracle，同时Static变化较小，则支持「Oracle昂贵目标让按迭代降温失配」这一解释；否则应继续检查OD近似/初始化/观测稀疏，而非反复调参数。
- `vnd_multistart`改变了局部搜索范式，是一项**增强搜索基线**，不能简单称“相同VND搜索5秒”。
- **测试数据已经被审阅**：即便Stage4只在验证集上改进算法，未来直接复用同一批7月26～31日测试数据衡量新算法也应明确标注为“已有测试集上的后续探索性复核”，如果需要新的真正未见测试，必须再冻结其他日期/月份。

下一阶段才是Traffic-ALNS：保持稳定的预算与温度设置，对Related/Worst破坏算子加入交通暴露特征，与不使用交通的相同预算搜索基线做配对消融；之后再加入多模态+对比学习。

## 6. 测试范围的真实限制

代码已在本地通过合成7月FIFO道路的交通OD、CVRP求解、严格结果网格、断点续跑以及使用**假ALNS接口**的算子流程测试。

当前执行环境无法安装`alns==7.0.0`，**没有在本地执行真正官方库的`alns_timecool`**，也未对真实7月完整数据进行Stage4跑分。用户服务器必须首先运行 `smoke`，检查真正官方库兼容性后，再运行Pilot/Validation。绝不将合成测试或模拟ALNS结果误当真实Chicago结论。
