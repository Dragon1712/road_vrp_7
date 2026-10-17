# Step5-B v2 独立时间预算实验补丁

适配仓库：`https://github.com/Dragon1712/road_vrp_7`。

此补丁增加独立 `budget_ablation.py`，并**只在原 `run_stage5b_v2.sh` 增加4个case入口**。不修改 `run_experiment.py`、`section_v2.py`、`time_selector.py`、Stage3、已完成的原 `pilot/validation` 或其数据。原本的求解器已经支持 `--output-dir` 和 `run_config.json` 严格锁定代码、参数和任务，因此采用它保存结果即可，无须修改算法。

## 一键安装

解压ZIP至仓库根目录，运行：

```bash
cd ~/code/WZL/road_vrp_7
unzip step5b_v2_budget_独立结果目录补丁.zip
bash step5b_v2_budget_patch/install.sh

# 非真实实验数据的独立预算运行器回归测试
python road_vrp7_step5b_v2/tests/test_budget_ablation.py
```

安装器会备份原始入口文件为 `road_vrp7_step5b_v2/run_stage5b_v2.sh.before_budget_patch.bak`，且仅第一次安装时插入命令；已修改的核心源码不会被覆盖。Python3.10+，真实实验仍需服务器的`alns==7.0.0`、Stage3道路图/OD数据和既有依赖。

## 建议首先运行 Oracle 5/15/30s（三任务、两场景、两个种子）

```bash
cd ~/code/WZL/road_vrp_7

# 仅显示计划路径，不修改任何文件
bash road_vrp7_step5b_v2/run_stage5b_v2.sh budget_dry_run

# 自动运行每档时间预算（可先放在screen/tmux里）
bash road_vrp7_step5b_v2/run_stage5b_v2.sh budget

# 各预算独立分析+跨预算总报告
bash road_vrp7_step5b_v2/run_stage5b_v2.sh analyze_budget
```

默认配置由新入口控制，不继承原始5秒`validation/`，默认：

- budgets: `5,15,30`秒；
- methods: `C0_baseline,C3_topk,C4_time_ucb,C5_gated`；
- planner: `oracle`；
- seeds: `42,43`；
- tasks: 3；scenarios: 2；
- `max_iterations=5000`（三档完全相同）；
- Split：`validation`（7月开发用途，非全新盲测）。

共`4算法×3任务×2场景×2种子×3预算=144`条；累计名义搜索预算`48×(5+15+30)=2400秒`，约40分钟，实际还含OD与回放开销。达到5000迭代上限的运行可能提前停止，务必参考`budget_summary.csv`的`hit_iteration_limit_count`。

## 独立目录结构

```
runs/chicago_july_v1/step5b_v2/
├── pilot/                  # 原实验，不碰
├── validation/             # 原3360条验证，不碰
└── budget_studies/
    └── section_budget_pilot_v1/
        ├── study_manifest.json  # 配置、源码SHA锁定
        ├── 5s_iter5000/
        │   ├── metrics.csv
        │   ├── solutions.jsonl
        │   ├── run_config.json
        │   ├── benchmark_summary.json
        │   └── analysis/
        ├── 15s_iter5000/...
        ├── 30s_iter5000/...
        └── analysis/
            ├── budget_summary.csv
            ├── budget_pairwise.csv
            ├── budget_progression.csv
            └── summary.json
```

默认**拒绝重复覆盖**已存在的研究目录；只有完全相同的源代码、任务/道路哈希、参数并使用`RESUME=1`才允许断点续跑。在一次预算的两个结果文件落盘之间中断时，核心运行器会检查不一致并拒绝静默恢复，需要人工审计该结果目录。

### 自定义独立实验名称

```bash
STUDY_NAME=budget_oracle_july_v2 \
BUDGETS=5,15,30,60 \
METHODS=C0_baseline,C3_topk,C4_time_ucb,C5_gated \
PLANNERS=oracle \
SEEDS=42,43,44 \
MAX_ITER=5000 \
MAX_TASKS=3 \
MAX_SCENARIOS=2 \
bash road_vrp7_step5b_v2/run_stage5b_v2.sh budget

STUDY_NAME=budget_oracle_july_v2 \
bash road_vrp7_step5b_v2/run_stage5b_v2.sh analyze_budget
```

改变预算、算法组、任务数量、种子、源码或迭代上限时，使用不同`STUDY_NAME`。分析入口根据保存的`study_manifest.json`读取原配置，不会错误使用当前Shell环境的默认参数。

### 中断后恢复

```bash
RESUME=1 bash road_vrp7_step5b_v2/run_stage5b_v2.sh budget
```

如果最初使用自定义`STUDY_NAME`/`BUDGETS`/`METHODS`/`SEEDS`等，续跑时也要提供**完全相同配置**，例如：

```bash
STUDY_NAME=budget_oracle_july_v2 BUDGETS=5,15,30,60 \
METHODS=C0_baseline,C3_topk,C4_time_ucb,C5_gated PLANNERS=oracle \
SEEDS=42,43,44 MAX_ITER=5000 MAX_TASKS=3 MAX_SCENARIOS=2 \
RESUME=1 bash road_vrp7_step5b_v2/run_stage5b_v2.sh budget
```

## 结果该怎么解读？

- `budget_summary.csv`：每预算、方法、规划模式的道路平均成本、规划目标、迭代次数、命中迭代上限的次数。
- `budget_pairwise.csv`：同预算内C3/C0、C4/C3、C5/C3、C5/C4等**严格配对**对比，任务级Bootstrap区间（极少任务时仅探索性）。
- `budget_progression.csv`：相同算法、任务、时段和种子从5→15、15→30的平均增益。这里是**不同预算的独立重跑**，不是同一次搜索轨迹的在线历史最优快照，因此结果不必单调改善。
- 每档预算仍各自运行原始`analyze_step5b_v2.py`，原有算子统计文件得到保留。

**注意**：预算上限为 `min(墙钟秒数, 5000次迭代)`，更长的预算有可能早早撞上5000迭代上限。`SimulatedAnnealing.autofit(num_iters=5000)`保持一致，不能把同时变更迭代上限和降温日程所得结果完全归因于墙钟时间。为了公平，当前实验先固定5000次；随后如需测试10000/20000次，另开独立研究目录。

实验完整性验证是强约束：`benchmark_summary`必须完成；每档任务、场景、算法、规划模式、种子网格完整；跨档道路/任务/源码/OD参数一致（只允许`wall_sec`不同）。不满足时拒绝合并跨预算分析。

## 本地测试范围

已使用合成**模拟runner**测试预算解析、隔离目录、不可覆盖、续跑、跨预算匹配和不一致配置拒绝。**没有运行官方ALNS 7.0.0真实Chicago搜索**；真实完整性须在服务器执行上述实验后确认。
