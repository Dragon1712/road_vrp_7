# Step5-B v3：独立退火、同轨迹快照、UCB奖励消融

适配仓库：`https://github.com/Dragon1712/road_vrp_7`，依赖既有 `road_vrp7_step5b_v2/`、`road_vrp7_stage3/`、`road_vrp7_step5b/`、Chicago July OD数据和 `alns==7.0.0`。

**不会修改任何v2代码/结果。** 本包包括 `install_v3.py`：根据仓库冻结的 v2 求解器 SHA256=`1e69a4f9e206990775016c2ad01b5845422402f570b7bed737574456e65f8d7c` 生成**完整的** v3 `run_experiment.py`，并复制 v2 的 `section_v2.py`、`time_selector.py`。其余模块已经包含在ZIP中。若v2源码不一致，安装器拒绝打补丁而非猜测接口。所有输出在 `runs/chicago_july_v1/step5b_v3/`。服务器需要Python>=3.10。

## 环境与运行顺序

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step5b_v3_完整代码.zip
python -m pip install 'alns==7.0.0' # 如原环境已安装则无需重复
bash road_vrp7_step5b_v3/run_v3.sh install
bash road_vrp7_step5b_v3/run_v3.sh test
bash road_vrp7_step5b_v3/run_v3.sh calibrate
bash road_vrp7_step5b_v3/run_v3.sh smoke
```

先检查 smoke 的 `run_config.json`、`snapshot_records.jsonl`、`metrics.csv`和`solutions.jsonl`，以及 `termination_reason` 和所有路线可行性，再执行Pilot。Smoke默认5组×单一场景×单一种子×0.35秒，目标快照是0.1/0.2/0.3秒。**本地只验证了合成测试，未在本环境安装和执行官方ALNS 7.0.0或真实Chicago成本。**

## Pilot：退火长度和搜索预算独立控制

```bash
bash road_vrp7_step5b_v3/run_v3.sh plan
bash road_vrp7_step5b_v3/run_v3.sh pilot
```

默认5组（C0基线、C3 Top-K、U0普通UCB、U1实时耗时UCB、U2固定耗时UCB）；3任务×2时段×2种子；预算30/60秒；迭代停止上限20,000；退火长度5,000/10,000。总计 `5×3×2×2×2×2=240` 条搜索，名义搜索时间约3小时，额外有初始化、OD、每条搜索的四次道路回测等耗时。若只想快速验证，可以设置：

```bash
STUDY_NAME=quick_v1 MAX_TASKS=2 MAX_SCENARIOS=1 METHODS=C0_baseline,C3_topk,U0_gain,U1_walltime,U2_calibrated \
BUDGETS=30 ANNEAL_LIST=5000 \
bash road_vrp7_step5b_v3/run_v3.sh pilot
```

全部指标和同轨迹时间快照在每个独立目录，随后分析：

```bash
bash road_vrp7_step5b_v3/run_v3.sh analyze
# 使用自定义study时也传 STUDY_NAME / BUDGETS / ANNEAL_LIST / MAX_ITER
```

每个case结果目录形如：

```
runs/chicago_july_v1/step5b_v3/decoupled_pilot_v1/
  60s_anneal5000_cap20000/
    run_config.json
    metrics.csv
    solutions.jsonl
    snapshot_records.jsonl
    benchmark_summary.json
    analysis/{solver_summary.csv,snapshot_summary.csv,snapshot_pairwise.csv,
              snapshot_progression.csv,ucb_operator_feedback.csv,analysis_summary.json}
```

对于快照，`snapshot_records.jsonl`中 `target_sec` 为5、15、30、60，`status=reached`才是真正观测到的同一次搜索历史最优状态。若运行提前达到迭代上限，`status=censored`，其成本为null并从该预算统计中剔除。**搜索结束后**才执行相应的真实道路回测，不计入`search_wall_sec`；包含回测的总时长见 `solve_and_replay_wall_sec`。同轨迹的**规划目标**历史最优非增，但实际道路回测成本不保证非增。

## 三个UCB版本的严格差异

相同Destroy/Repair (`random,worst,related,section` × `greedy,regret2`)、UCB探索常数c=.35、epsilon=.08、随机机制、Section Top-K=12、共同退火及搜索预算。

- `U0_gain`：`min(5, 100*positive_gain/current_cost / GLOBAL_FIXED_SECONDS)`，全局固定分母仅统一奖励尺度，不偏袒特定算子组合。
- `U1_walltime`：`min(5, 100*positive_gain/current_cost / MEASURED_WALL_SECONDS)`，与原C4的实时耗时奖励机制一致（因日志开销不同不保证逐条bitwise一致）。
- `U2_calibrated`：`min(5, 100*positive_gain/current_cost / FROZEN_PAIR_SECONDS)`，其中估计值来自冻结开发集的均值Destroy耗时加均值Repair耗时，**不是精确的整次算子组合耗时**。

`U0/U1/U2`的奖励都保留当前`min(5, ...)`上限，另外输出`reward_capped_counts`与`raw_reward_sum`以审计奖励饱和程度；若大多数正改善均触顶，后续需单独设计奖励缩放消融，不能将U0/U2优劣简单归因于时间分母。

`calibrate`从**已有开发验证集** `runs/chicago_july_v1/step5b_v2/validation/solutions.jsonl` 中只读C3 Top-K、Oracle记录，产生`road_vrp7_step5b_v3/calibration/section_pair_costs_dev.json`。该文件有原始结果SHA256、样本数、配对成本矩阵和来源说明，其自身SHA也写入每个v3运行配置。不能用新月份盲测结果计算成本。

## 公平性与结果解释

- `max_iterations=20000`只作为宽松**迭代停止上限**；`anneal_iterations=5000/10000`独立设置退火日程；所有预算档使用相同源码与OD。
- 不同退火日程必须用**不同输出目录**。`run_config`锁定本地源码SHA、道路哈希、任务哈希、退火长度、成本校准SHA和时间快照档。配置不一致拒绝覆盖。
- 原v2的C4/C5还可以作为可选变体使用；本研究主消融仅用C0/C3/U0/U1/U2。U2代表计算成本感知，不代表交通感知；交通独立贡献必须另设交通负对照。
- 同轨迹快照对比最适合识别预算效应，但较长预算的单次搜索中仍存在退火阶段影响；要比较降温机制应固定同一预算和其他参数。
- 与未来多模态方法相比，本阶段仍是历史Oracle交通上限研究，不能宣称真实部署性能；7月已是算法开发集，最终盲测必须使用未参与设计的其他月份。

## 可复现性对照

建议同样的参数单独执行3次，只改变目录名称，不修改代码/校准文件：

```bash
for name in repeat_a repeat_b repeat_c; do
  STUDY_NAME="$name" BUDGETS=30 ANNEAL_LIST=5000 MAX_TASKS=2 MAX_SCENARIOS=2 \
    METHODS=C0_baseline,C3_topk,U0_gain,U1_walltime,U2_calibrated \
    bash road_vrp7_step5b_v3/run_v3.sh pilot
done

python road_vrp7_step5b_v3/repeat_compare.py \
  runs/chicago_july_v1/step5b_v3/repeat_a/30s_anneal5000_cap20000 \
  runs/chicago_july_v1/step5b_v3/repeat_b/30s_anneal5000_cap20000
```

比较`mean_absolute_road_delta_sec`、`changed_road_records`、迭代差异，以及UCB算子反馈。运行时建议限制BLAS线程及CPU资源并记录机器负载，不能仅靠随机seed保证计时敏感U1复现。

## 断点续跑

与v2相同，只有三个结果文件(`metrics.csv`,`solutions.jsonl`,`snapshot_records.jsonl`)记录键完全一致才能继续，运行之前手动审计文件：

```bash
RESUME=1 bash road_vrp7_step5b_v3/run_v3.sh pilot
```

修改参数、源码、校准后必须换新`STUDY_NAME`；续跑原实验必须提供**同样的环境变量配置**。

## 实现边界

1. 安装器是源码补丁生成器，而非直接修改v2；严格检测冻结源SHA。不可在未知v2版本上硬套。
2. 本地完成合成数学/分析回归；真实官方ALNS Smoke与Chicago OD联调是正式实验前的硬门槛。
3. U2算子成本代理采用加性近似，不使用未来交通；随后可开展独立微基准校准以提高真实性。
4. `snapshot_records.jsonl`中保存到达快照的客户路径，其目的是可审计性，文件较大；未到达快照不提供伪结果。
