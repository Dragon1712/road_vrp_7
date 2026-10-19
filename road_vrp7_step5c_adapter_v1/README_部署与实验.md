# Step5-C 独立适配包：官方 ALNS Bandit + GRLOS-inspired LRW

适用项目：`Dragon1712/road_vrp_7`，源代码依赖已上传的 `road_vrp7_step5b_v3/`。**此包不修改 Step5-B v3 源码、校准或任何结果**。

## 1. 先运行1200条Step5-B v3扩大验证

```bash
cd ~/code/WZL/road_vrp_7
export STUDY_NAME=full30s_1200_v1
export METHODS=C0_baseline,C3_topk,U0_gain,U1_walltime,U2_calibrated
export MAX_TASKS=10 MAX_SCENARIOS=8 SEEDS=42,43,44
export PLANNERS=oracle BUDGETS=30 ANNEAL_LIST=5000 MAX_ITER=20000
export SNAPSHOTS=5,15,30
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
bash road_vrp7_step5b_v3/run_v3.sh plan
bash road_vrp7_step5b_v3/run_v3.sh pilot | tee step5b_v3_full30s_1200.log
bash road_vrp7_step5b_v3/run_v3.sh analyze
```

**注意**：`pilot`为v3脚本预留的统一网格运行入口；环境变量将网格扩大到 **10任务×8场景×3种子×5算法=1200**。若续跑需保持完全相同环境变量，设置`RESUME=1`。采用新`STUDY_NAME`，不能往原Pilot目录追加。

## 2. 创建独立Step5-C并安装依赖

上传 `road_vrp7_step5c_adapter_v1.zip` 至仓库根目录后：

```bash
unzip road_vrp7_step5c_adapter_v1.zip
python -m pip install 'alns[mabwiser]==7.0.0'
python road_vrp7_step5c_adapter_v1/install_step5c.py
bash road_vrp7_step5c/run_step5c.sh test
```

安装器先校验冻结v3求解器SHA与分析器SHA，再复制求解器接口、Section、快照模块到新的 `road_vrp7_step5c/`，对锚点作**精确单次替换**，从不覆盖或修改现有v3。若仓库的v3源码已有修改，将拒绝安装，请审计源码后另做新版本。

## 3. 从历史开发Pilot离线学习LRW-inspired联合算子权重

```bash
bash road_vrp7_step5c/run_step5c.sh train-lrw
```

训练数据来源：`runs/chicago_july_v1/step5b_v3/decoupled_pilot_v1/30s_anneal5000_cap20000/solutions.jsonl`中已冻结的C3 Top-K + Oracle算子日志，仅包含7月22日两个时段、3任务、2种子。生成 `road_vrp7_step5c/calibration/lrw_from_v3_pilot.json`，记录输入文件SHA、训练任务、日期、pair概率。**这是GRLOS LRW思想的简化独立实现，不是对原作者LRW的逐行复现，也不是复制其仓库源码。** C9按联合Destroy–Repair pair学离线固定轮盘权重；训练策略`P(pair)`根据平滑的历史新最优率和候选改善率设置，并保留15%的均匀探索质量。

不能把同任务或同一时间段作为“盲测”来宣称泛化；本例测试仅验证在不同7月开发交通时段的迁移。真盲测需新月份且提前冻结权重。

## 4. 官方ALNS选择器Smoke与开发Pilot

```bash
bash road_vrp7_step5c/run_step5c.sh smoke
bash road_vrp7_step5c/run_step5c.sh pilot
```

Smoke：1个任务、1个7月23日08:00情景、1个种子、5种算法、0.5秒。Pilot：3个任务×2个7月23日情景×2种子×5算法=60条，30秒预算，退火跨度5000，最大迭代上限20000，快照5/15/30秒。

5种算法：C3 Top-K、U2 Calibrated、**C7 official AlphaUCB**、**C8 official MABSelector + MABWiser ThompsonSampling**、**C9 GRLOS-inspired frozen LRW**。

ALNS原生C7/C8依据官方四种结果类型奖励`[8,4,1,0]`，C8将正奖励映射为二值成功，不使用v3的连续目标改善/耗时奖励；因此不同Bandit策略间的效果**不能仅归因于探索算法**。C9属于冻结的离线算子权重基线。比较总预算/算子/修复公平，但需补充统一奖励对照才能做纯机制归因。

默认输出：

```
runs/chicago_july_v1/step5c_v1/
  smoke_v1/{run_config.json,metrics.csv,solutions.jsonl,snapshot_records.jsonl,analysis/}
  step5c_july23_dev_v1/{run_config.json,metrics.csv,solutions.jsonl,snapshot_records.jsonl,analysis/}
```

仅分析已有Step5-C Pilot：

```bash
bash road_vrp7_step5c/run_step5c.sh analyze
```

中断续跑：

```bash
RESUME=1 bash road_vrp7_step5c/run_step5c.sh pilot
```

需同一版本源码、校准数据、LRW权重与所有参数；改动任一项都要设置新的`STUDY_NAME`。代码与配置SHA被保存。

### 建议拆分运行阶段

如果1200条实验与Step5-C都在同一台服务器，**可以提前安装、测试、训练LRW**，但不要同时运行搜索Benchmark。因为U1依赖实时墙钟耗时，Step5-C训练或搜索占CPU会干扰对照的运行环境。优先完成1200条，随后进行Step5-C Smoke/Pilot。

## 5. 完成标准与局限

- Step5-B v3：1200/1200, 5个算法每个240；以任务为统计独立簇，报告95%区间、种子波动、30秒预算、历史道路回测。
- Step5-C：Smoke应全部可行并生成有效快照；若C8在ALNS 7.0.0的MABWiser依赖或API下报错，停止Pilot，先核对具体错误再适配，不得跳过并声称成功。
- Step5-C LRW从过去的7月开发日志学习；此Pilot不证明对未见配送任务泛化。后续建议先按任务留出或新月份盲测，另外增加无学习的等权轮盘、官方AlphaUCB、静态Section对照。
- 交通的Oracle时变OD代表事后已知未来交通，不能当成可部署预测模型；最终部署需要历史可得交通输入与公平预算计时。
- 本代码在当前工作环境通过合成数学/补丁锚点单元测试，但**未执行服务器真实官方ALNS 7.0.0的Smoke**。
