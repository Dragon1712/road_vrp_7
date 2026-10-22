# Step6-C1-R2 — 真实C3对照动作监督、学习式Section优势预测与30秒保守回退

**代码基准**：[`Dragon1712/road_vrp_7`](https://github.com/Dragon1712/road_vrp_7)，冻结主分支 `c929836b3ea76518d4a1e681b1dbc729f69d4c69`（2026-10-21）。本包**独立**安装到 `road_vrp7_step6c1_r2/`，不会覆盖C1、C1-R、Step6A、官方ALNS或B2-R。

## 0. 研究动机 / 前一阶段失败原因

C1-R的8例真实道路回测，FULL_DEFER_SAFE比M0平均多43.156秒。旧训练默认动作是8个Section里的静态RF最优项（M0代理），但在线门控拒绝时回退的是**冻结C3**，两种选择机制不同；旧离线接受率也包含很多 `same_as_static`，不能当作有效覆盖。本包只测试R2决策监督是否比旧的M0代理更适合在线系统，**不引入新SPIN或SPCLT**。

## 1. 与已上传论文源码的关系

在已有 `literature/code/learning-to-delegate-main/supervised.py` 的子问题效用监督、`literature/code/learn-to-defer-master` 的专家回退、`literature/code/SPIBB-master/spibb.py` 的基线保留思想下，独立实现**C3相对收益回归 + 保守选择**。不是论文源码逐字复制或SPO+/SPIBB的精确复现，也没有继承其理论性能/安全保证。

优化目标：设在真实ALNS状态 `S` 的冻结C3选择为 `a_C3`，学习侧经过相同分层采样器得到 `a_1..a_8`：

\[
\Delta_{i}(S)=\big[C_{TD}(S)-C_{TD}(R_{static}(S,a_i))\big]
-\big[C_{TD}(S)-C_{TD}(R_{static}(S,a_{C3}))\big]
\]

也就是同一静态维修子程序和相同状态上的**有符号TD增益差**。TD oracle仅在离线标签阶段使用，不能被在线读取。R2使用固定超参数ExtraTrees回归 `Δ_i`，输入只有 `F2 / F4` 候选特征、C3基线动作特征及差值。该模型是**可审计的非深度轻量决策基线**，目的是先检验监督语义。未来D2/SPIN/SPCLT应以**完全相同的C3标签和在线选择语义**接入，以确保能证明各模态的独立作用。

在线决策：`argmax predicted Δ` 且 `predicted Δ > τ` 且 `a_i != a_C3` 时才覆盖C3；否则直接沿用真实C3的已计算Partial。阈值 τ 从验证状态选取 `{0,5,10,20,40,80,永不覆盖}`。**只有真正选中不同Section才计为有效覆盖。**

### 精确C3动作与随机数

* 冻结C3的 `Selector.destroy` 并非Top-1：它静态归一化排序、加入极小随机jitter，然后从前20%随机抽Section。
* `collect_c3.py`在**真正的冻结C3运行**中调用原 `Selector.destroy` **一次**，得到原ALNS RNG的实际选择，之后仅复制必要状态信息。捕获开销包含在30秒搜索墙钟内，因此捕获轨迹用于训练，**不作为求解性能基线**。
* `online_r2.py`使用同一冻结C3代码，在复制的 `np.random.Generator` 和隔离的 `Selector.stats` 上做**影子C3**。若学习模型回退，直接提交影子C3 Partial并同步原始rng状态及统计；若学习模型覆盖，也推进与C3等量的随机数消费以改善共同随机数对照，但不添加C3的选择统计。
* 对于未达到模型决策cadence或评分CPU预算耗尽的调用，原始C3在原始rng上照常执行。
* 不缓存RNG相关C3选择结果，因此 `cache_size=0`。C3影子与历史特征/预测耗时全部计入30秒搜索墙钟和2秒累积模型评分预算。

## 2. 文件说明

| 文件 | 主要功能 |
|---|---|
| `r2_core.py` | 与冻结B2-R一致的状态ID、克隆C3 RNG、真实C3选择、提交shadow状态、复用F2/F4特征 |
| `collect_c3.py` | 原官方ALNS30秒运行**原位记录真实C3所选Section**；不在在线求解时读取Oracle |
| `label_c3.py` | 离线给8个学习候选和独立C3候选打**同一个**Static Repair→TD成本标签，支持C3不在8候选时的独立baseline |
| `r2_policy.py` | 真实C3相对收益回归、拒绝门控、状态级保守损失诊断、严格Task/Date分组 |
| `train_c3.py` | 六种子分别训练F2/F4，checkpoint无需复杂网络；在验证集选择阈值，internal_test只报告 |
| `online_r2.py` | 克隆真实C3选择及rng，保留原B2-R求解器、SA、Repair、操作调度和双预算 |
| `compare_r2.py` | 将真实Road.replay对照原C3及M0配对审计；原C3和M0候选采样器不相同，报告明确注明 |
| `run_r2.sh` | 统一实验入口 |
| `tests/` | 10项合成与接口测试，其中包括“复制RNG无副作用”“影子C3与原C3动作一致”“训练保存artifact” |

## 3. 服务器执行顺序（从仓库根目录）

使用已有 `alns==7.0.0, torch, numpy, pandas, scipy, scikit-learn, joblib` 环境。以`~/code/WZL/road_vrp_7`为例：

### A. 安装并检查

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6c1_r2_v1.zip
bash road_vrp7_step6c1_r2/run_r2.sh test
```

### B. 真实C3轨迹SMOKE：单次30秒（仅用于接口检查，不足以训练）

```bash
MAX_RUNS=1 MAX_PER_RUN=3 STUDY_NAME=r2_c3_capture_smoke1_v1 \
 bash road_vrp7_step6c1_r2/run_r2.sh collect
CAPTURE_STUDY=r2_c3_capture_smoke1_v1 MAX_STATES=2 \
 STUDY_NAME=r2_c3_label_smoke2_v1 \
 bash road_vrp7_step6c1_r2/run_r2.sh label
cat runs/chicago_july_v1/step6c1_r2/r2_c3_label_smoke2_v1/summary.json
```

Smoke的训练组数不足3是预期情况，**不要用此目录训练R2**。若超时、无可标注C3动作或F4缺少历史输入，必须先排查接口、环境，不要跳过。

### C. 初步正式数据采集（真实C3轨迹，train时段）

```bash
MAX_TASKS=8 MAX_SCENARIOS=8 MAX_PER_RUN=8 STRIDE=12 \
 SEEDS=42,43 STUDY_NAME=r2_c3_capture_train8x8_v1 \
 bash road_vrp7_step6c1_r2/run_r2.sh collect

CAPTURE_STUDY=r2_c3_capture_train8x8_v1 \
 STUDY_NAME=r2_c3_label_train8x8_v1 \
 bash road_vrp7_step6c1_r2/run_r2.sh label
```

这相当于最多8任务×8训练场景×2随机种子=128条**各30秒**的真实C3轨迹，按每次最多8个状态采集上限；不是128条R2性能评估。标注须为每个候选运行反事实Static Repair，**额外计算可能显著大于128×30秒**。依照服务器资源，首次可将 `MAX_TASKS=6 MAX_SCENARIOS=6 MAX_PER_RUN=4` 缩小为72条。

捕获自7月**train**时段的状态，后续July validation的在线Pilot与采集日期不重合；但你此前反复使用了July进行开发，所以它仍**不是盲测**。

采集器每完成一条30秒ALNS运行，就原子写入 `runs/*.json`；标注器每标注一个状态，就追加 `checkpoint.jsonl`。如果服务器中断，**代码、源数据、参数和STUDY_NAME均不变**时，直接使用原命令再加 `RESUME=1`：

```bash
RESUME=1 MAX_TASKS=8 MAX_SCENARIOS=8 MAX_PER_RUN=8 STRIDE=12 \
 SEEDS=42,43 STUDY_NAME=r2_c3_capture_train8x8_v1 \
 bash road_vrp7_step6c1_r2/run_r2.sh collect

RESUME=1 CAPTURE_STUDY=r2_c3_capture_train8x8_v1 \
 STUDY_NAME=r2_c3_label_train8x8_v1 \
 bash road_vrp7_step6c1_r2/run_r2.sh label
```

`--resume`逐项核验manifest，包括代码SHA，**不能**以相同结果目录改采样率、修复时间或道路数据；若配置改变请建立新目录。

### D. 六种子训练（Task分组为主、Date分组为时间外推对照）

```bash
for seed in 42 43 44 45 46 47; do
  SEED=$seed PROTOCOL=task MODES=F2,F4 \
  LABEL_STUDY=r2_c3_label_train8x8_v1 \
  STUDY_NAME=r2_task_seed${seed}_v1 \
  bash road_vrp7_step6c1_r2/run_r2.sh train
done

# 若至少采集3个不同日期、且每一折非空，另做Date协议
SEED=42 PROTOCOL=date MODES=F2,F4 \
  LABEL_STUDY=r2_c3_label_train8x8_v1 \
  STUDY_NAME=r2_date_seed42_v1 \
  bash road_vrp7_step6c1_r2/run_r2.sh train
```

Task/Date由实际采集的 `groups.jsonl` 划分，不同训练种子共享相同划分，不能当作独立测试集。报告位于 `report.json`，含验证期阈值和 `internal_test` 有符号收益差、真正的有效覆盖率、正负覆盖次数与大额伤害指标。**如果验证集选择了无限阈值（永不覆盖），这也是一个合法的负结果，不能擅自调低阈值让模型看起来有收益。**

### E. 固定30秒在线Pilot，含shadow-only负对照

```bash
MODEL_STUDY=r2_task_seed42_v1 MODEL_MODE=F2 R2_FORCE_DEFER=1 \
 STUDY_NAME=r2_online_shadow_only_match8_v1 \
 bash road_vrp7_step6c1_r2/run_r2.sh online

for mode in F2 F4; do
  MODEL_STUDY=r2_task_seed42_v1 MODEL_MODE=$mode \
    STUDY_NAME=r2_online_${mode}_match8_v1 \
    bash road_vrp7_step6c1_r2/run_r2.sh online
done
```

**必须**另运行冻结C3基线（不能拿M0冒充C3）：

```bash
python road_vrp7_step6b2_r/benchmark_30s_r.py \
 --control C3 --model-method P2 --model-mode F2 \
 --model-source runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1 \
 --output-dir runs/chicago_july_v1/step6c1_r2/r2_original_C3_match8_v1 \
 --seeds 42,43 --split validation \
 --task-ids task_0008,task_0009 \
 --scenario-ids 2025-07-25_0800,2025-07-25_1700 \
 --wall-sec 30 --max-iterations 20000 --anneal-iterations 5000 \
 --top-k 12 --action-count 8 --sampler-seed 20261019 \
 --score-every 8 --max-scoring-sec 2 --cache-size 0 \
 --min-static-fraction 0 --fallback c3
```

并列比较M0、原C3、R2-shadow-only、R2-F2和R2-F4；R2和M0都按8候选分层采样，**原C3按前20%抽样，并不与它们共享该候选集**。此外既有M0采用64项cache，R2因C3选择依赖RNG而禁用cache；比较需要把这一计算差异计入解释。原C3既是实际回退动作的监督基线，也是端到端必须同时报告的求解基线。

### F. 三方道路成本审计

```bash
R2_RUN=runs/chicago_july_v1/step6c1_r2/r2_online_F2_match8_v1 \
C3_RUN=runs/chicago_july_v1/step6c1_r2/r2_original_C3_match8_v1 \
STUDY_NAME=r2_compare_F2_match8_v1 \
 bash road_vrp7_step6c1_r2/run_r2.sh compare
```

若需要另比较F4，改为**独立的新结果目录**。`report.json`同时报告相对C3、相对M0、逐样本收益以及不同采样器的限制。真实30秒在线主指标仍是 `ex_post_road_travel_sec`，且同时有迭代上限20000（不是每条都跑满30秒）。

## 4. 上线前必须核对的审计指标

1. 原C3 capture实测action是否与 `Selector.destroy` 返回的 `Partial.blocks[0]` 一致，且capture不会进行Oracle回测。
2. 标签目录的 `summary.json`：有效状态数、候选有效比例、基线C3动作位于8候选中的比率；没有足够可用状态时不应训练。
3. `r2_diagnostics.json` 的 `r2_effective_overrides`（只有真正不同于C3才计覆盖）、`r2_shadow_calls`和 `r2_score_budget_prevented`；负收益尾部与成功覆盖比例。
4. `shadow-only` vs 原C3：行为本应相同，但CPU开销不同可能使固定30秒的迭代数与最终解不同，不能误判为学习带来的增益。
5. 实验源代码和数据SHA、train/validation任务/日期是否交叉、训练标签是否混入features。
6. 性能比较要覆盖未反复使用的更多日期/任务，使用任务/日期分组置信区间而不是把重复种子当成独立样本。

## 5. 实验有效性门槛

先检查 `label/summary.json`：若成功标注状态数低于约100、可覆盖候选很少、训练组数不足3或 `candidate_valid_fraction` 明显偏低，暂停训练，检查C3捕获和Static Repair超时。即使达到了100状态，也不能把它当作充分的统计功效保证。

**Shadow-only是不可省略的对照**：如果shadow-only已经较原C3变差，先分析计时与20000次迭代上限，不能将模型差异全部归因于学习。若训练只选择`threshold=inf`，则R2在验证集没有找到可信覆盖机会，应如实记录负结果。

## 6. 已知限制（主动保留）

- 预测的是**一次Static Repair下**的TD有符号增益差，**不是**30秒ALNS长期回报差。连续搜索策略分布偏移仍可能导致离线收益无法转化为在线收益。
- C3真实训练状态来自**冻结C3搜索轨迹**，部署覆盖之后会访问新状态。建议下一步用独立轮次的DAgger-style学习策略状态采集，不得在这轮混入。
- `R2Advantage`先采用固定ExtraTrees（简化工程和推理开销），没有引入D2图或CL。只有R2决策监督被验证后，再以相同标签/分组增加D2T、D2、SPIN、STMAE和SPCLT。
- 正式原C3运行、特征标签和在线回测只能在带真实芝加哥OD、道路NPZ、ALNS源码的服务器上验证。当前代码包通过合成测试，但**未运行真实环境**。
- `.joblib`仅加载自己信任、刚训练产生的模型文件；不要加载不可信第三方模型。
- 本版捕获和离线标注均采用新结果目录，不改已有July开发结果，也不将7月internal_test称作真正盲测。
