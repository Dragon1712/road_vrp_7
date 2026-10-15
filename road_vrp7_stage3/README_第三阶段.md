# road_vrp_7 · 第三阶段实验代码（2025-07 完全独立实验体系）

> **基于GitHub最新提交 `087a3a91e22469e369ffc6026ef16d8e9811cf86` 对照数据格式开发。** 不复制旧road_vrp仓库的求解器、订单或6月成本。历史oracle只用于事后诊断，不能当作未来可获得的交通预测。

## 0. 现有第二阶段结果（已见于仓库）

- `stage2_validation.json`：`validation=PASS`；4320×1047监测张量、5503有向边、510条映射边、68条监测走廊。
- 全监测单元有效：2,287,478；映射边可观测单元1,592,564；映射边平均有效率72.284%；全路网观测支持6.699%；FIFO修正0。
- 7月28日08:00的TDSP 30组OD，平均oracle路线优势10.35秒；17:00为5.90秒。两组都`PASS`；这些是样本内道路路由指标，不能推出VRP必然改善。
- 7月26~31测试期有12个08:00/17:00情景，其中**10个符合已有质量门槛**，7月27日两场景列入压力测试。
- `traffic_long.csv`不用提交到GitHub，也无需第三阶段重读，主要使用已提交的`edge_costs.npz`等道路成本文件。

## 1. 目录与依赖

将整个`road_vrp7_stage3/`复制到`~/code/WZL/road_vrp_7/`项目根目录，**所有命令在项目根目录执行**：

```bash
cd ~/code/WZL/road_vrp_7
python -c 'import numpy,pandas,networkx; print("road libs OK")'
python -m pip show alns
# 如果未安装，且服务器有可用安装源：
python -m pip install 'alns==7.0.0'
# 对于无法联网的服务器，在有网络的电脑下载wheel及依赖，复制后离线安装。
```

要求Python >= 3.10，`numpy`, `pandas`, `networkx`, **`alns==7.0.0`（正式官方ALNS实验必需）**。Python不需要使用OSMnx（使用第二阶段已经处理的CSV和NPZ）。

文件结构：

```
road_vrp7_stage3/
  road_vrp.py          # 独立有向FIFO时变最短路、精确道路回测、OD网格与CVRP目标函数
  tasks.py             # 10个以上独立订单任务生成；验证exactly-5、cap-45
  solvers.py           # Greedy, SA, VND, official_alns(N-Wouda/ALNS 7.0.0)
  benchmark.py         # 配对搜索与完整结果存档；哈希绑定防串数据
  analyze.py           # 严格实验网格检查、逐task bootstrap、日报告
  run_stage3.sh        # 各阶段一键启动
  tests/test_stage3.py
  README_第三阶段.md
```

## 2. 运行的正确顺序

### 第一步：合成测试（不用真实订单训练）

```bash
bash road_vrp7_stage3/run_stage3.sh test
```

预期最后打印`PASS stage3 synthetic July fixture ...`。若官方ALNS未安装，合成测试仅会明确SKIP官方库适配部分。安装`alns==7.0.0`后务必重跑，出现`PASS official ALNS 7.0.0 integration`才算该接口已通过测试。

### 第二步：生成10个相互独立、跨日期固定的订单任务

```bash
bash road_vrp7_stage3/run_stage3.sh generate
```

生成：

```
datasets/processed/chicago/july_v1/tdcvrp30/tasks_10/
  task_0000.json ... task_0009.json
  manifest.json
```

每个任务30客户、5辆必须使用的车、单车容量45、客户服务120秒。67%客户优先取匹配走廊附近约600米，剩余客户在同一最大强连通分量抽样。客户**位置与需求人为合成**，不是真实快递公司订单；所有算法共用这些任务。实验种子`20261015`，保存成本SHA256和原始质检CSV SHA256。

**生成过程默认拒绝覆盖非空任务目录。** 不要为了试算更换客户/任务，先冻结生成结果后再开始基线对比。

### 第三步：官方ALNS接口与基线烟雾测试

```bash
bash road_vrp7_stage3/run_stage3.sh smoke
bash road_vrp7_stage3/run_stage3.sh analyse smoke
```

单任务、一个训练期日期时段（默认2025-07-02 08:00）、seed42、两个规划模式、4个算法，共8行；最大100迭代或2秒搜索预算。全部记录都须成功，无不可行路线。**如果官方ALNS报导入或API错误，请保留日志，不要自动替换为自编ALNS。**

### 第四步：训练期探针 / Pilot（不使用测试集）

```bash
bash road_vrp7_stage3/run_stage3.sh pilot
bash road_vrp7_stage3/run_stage3.sh analyse pilot
```

3任务×2训练时段×2随机种子×4求解器×2规划模式=96行；用于确认时间预算、路线质量和计算时间。此阶段可合理修复程序bug或调整预先定义的实验参数，但调整后使用新输出目录及完整记录；不要基于测试集选参数。

### 第五步：验证集基线

```bash
bash road_vrp7_stage3/run_stage3.sh validation
bash road_vrp7_stage3/run_stage3.sh analyse validation
```

10任务×8合格情景（7月22～25）×3 seeds×4算法×2模式=**1920行**。

### 第六步：冻结设置后一次性测试集

```bash
bash road_vrp7_stage3/run_stage3.sh testset
bash road_vrp7_stage3/run_stage3.sh analyse test_eligible
```

10任务×10合格测试情景（7月26、28、29、30、31日的08/17时段）×3 seeds×4算法×2模式=**2400行**。

### 第七步：独立低覆盖压力测试（原QC拒绝）

```bash
bash road_vrp7_stage3/run_stage3.sh stress
bash road_vrp7_stage3/run_stage3.sh analyse test_july27_stress
```

7月27日08/17两情景独立结果：10×2×3×4×2=**480行**。标记为压力测试，不与10个已通过质量门槛的主测试场景混算。

对于服务器中断：原样重跑并在命令前加 `RESUME=1`：

```bash
RESUME=1 bash road_vrp7_stage3/run_stage3.sh validation
```

断点恢复严格检查`run_config.json`、`metrics.csv`和`solutions.jsonl`的主键与哈希；**不能用不同代码、任务、道路成本、求解预算混合续跑。** 如果在单条结果的双文件写入之间发生掉电，需人工修复记录一致性后续跑；脚本不会悄悄吞掉半行结果。

## 3. 实验比较定义

| 求解器 | 方法 | 训练 | 目的 |
|---|---|---|---|
| `greedy` | 可行贪婪插入 | 无 | 构造参考，单独报告构造时间 |
| `sa` | 随机扰动+模拟退火 | 无 | 固定预算元启发式基线 |
| `vnd` | relocate / swap / 2-opt首次改进 | 无 | 可解释局部搜索基线 |
| `official_alns` | N-Wouda/ALNS 7.0.0；随机/最坏/相关破坏+贪婪/Regret2修复；自适应轮盘与SA接受 | 无 | 官方ALNS基线 |

`--planners static,oracle`：
- `static`使用OSM静态道路旅行时间进行规划；实际成本**仍用相同7月混合历史成本+FIFO TDSP道路级重放**。
- `oracle`使用7月整个历史交通已知的5小时采样TDSP OD表做规划，属于离线hindsight上界参考，并非实时部署或数学最优。
- 搜索目标是5辆车**总路上行驶时间**，不包含服务120秒的加和；服务时长只推进每车到达时间。记录`ex_post_total_vehicle_elapsed_sec`作诊断。
- `surrogate_travel_sec`是规划OD插值代理目标；**主指标使用`ex_post_road_travel_sec`**精确重放，不用代理目标代替主结果。
- 前处理（例如OD预计算）时间独立列出，不混入`search_wall_sec`预算。Greedy并不会强行耗满搜索时间，不要将其视为同等计算成本。

官方ALNS自适应销毁算子权重由上游库实现；`alns==7.0.0`必须从官方库安装。本包不包含/不伪装上游库。所有搜索为CPU离线优化，**这一阶段不训练GNN、对比学习或交通预测网络**。

## 4. 输出与完整性约束

```
runs/chicago_july_v1/stage3/
  od_cache/                          # SHA绑定7月成本+task节点+日期+采样配置
  smoke/metrics.csv
  smoke/solutions.jsonl
  smoke/run_config.json
  smoke/benchmark_summary.json
  smoke/analysis/solver_summary.csv
  smoke/analysis/paired_gains.csv
  pilot/...
  validation/...
  test_eligible/...
  test_july27_stress/...
```

统计程序首先建立完整的“task×scenario×method×planner×seed”理论网格，**即使各算法同时缺失相同键，也会报错**。统计包括算法均值、配对增益、按独立任务bootstrap的95%探索区间、日期分层结果。训练/测试分区独立，且对应数据从7月QC原始`scenario_quality.csv`冻结。

## 5. 可信度限制 / 后续研究

1. `stage2_validation=PASS`支持文件形状、FIFO和观测标记的自动一致性，不表示510条OSM匹配边经过全部人工地理核验。
2. 测试期Oracle规划**读取未来实测历史交通**；未来需用仅过去窗口训练、且未见测试日的预测模型替换Oracle输入，才谈得上可部署性能。
3. 未监测的约93.3%路网单元是静态回退，评价并非完整实测真实交通；必须报告已测边覆盖率。
4. 30客户订单和需求为合成场景；不能声称真实物流公司派送结果。
5. 5秒预算控制在算法每次迭代之间检查，昂贵单次operator可能略微超时；需要报告实际`search_wall_sec`。官方ALNS的自适应温度尺度由最大迭代数而非实际限时内迭代数设定。
6. 一周内周末少；10任务、6测试日期属于初步实证，不等价多城市多季节泛化。

完成这些之后才进入Traffic-ALNS算子改造、交通补全/预测和多模态学习的下一阶段。
