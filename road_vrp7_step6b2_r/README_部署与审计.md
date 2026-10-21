# Step6-B2-R v1：在线Section选择器修复、F5修正与历史交通负对照

## 适用版本与真实性界限

以GitHub `Dragon1712/road_vrp_7` 于2026-10-21仓库`main`（初始核对提交 `b53440a2d694b8274ee0cfa2fd8dcc4a4a1aa716`）的Step6-B2代码为基准。本包**独立新增**`road_vrp7_step6b2_r/`；不编辑冻结的`road_vrp7_step5b_v3/`、`road_vrp7_step6a_v2/`、`road_vrp7_step6b1_v2/`、`road_vrp7_step6b2/`、已有结果。

本地已测试：合成Section特征及训练、临时Hook恢复、no-op改C3回退、冷预算、缓存命中、F5日期报告隔离、冻结模型置换、严格配对汇总。**本环境没有官方ALNS 7.0.0及全部芝加哥数据，因此必须在服务器Smoke后才扩大实际30秒实验。** 训练数据来自7月开发集，任何收益均为开发性结果，不应宣称独立盲测效果。

## 这次具体修复了什么？

- **Section no-op问题**：B2旧在线代码对几乎每次门控拒绝返回空Partial，严重改变ALNS行为。新选择器默认回退到**冻结原C3的Section Destroy**（`--fallback c3`），不是无操作。保留`--fallback noop`仅作为明确命名的诊断消融。
- **推理开销**：`--score-every 8`每8次Section调用才尝试学习模型，其余直接原C3；`--max-scoring-sec 2`给学习选择器的候选枚举/特征提取/推理（不含原C3回退成本）一个每次求解最多约2秒的**软预算**；`--cache-size 64`按场景和完整路线缓存正/负门控；超出预算自动用原C3。
- **廉价静态门控**：`--min-static-fraction`按已知最大Section静态成本/总静态规划成本筛选。默认`0`（关闭，以免无校准阈值掩盖收益）；开启时不会读取未来交通，仅使无资格的状态回退C3。
- **F5污染**：旧`train_b2.py`在外层复用`reports={}`，导致task上的F5条目残留到date报告；新`train_b2_r.py`改为**协议内独立报告**，并在`summary.json`中列举每个协议真实执行的方法。`repair_f5_report.py`可在不更改既有目录、模型或标签的前提下，生成原Seed42日期报告的修正副本。
- **更严格的交通负对照**：`frozen_permutation.py`加载**已经训练好的F3/F4模型及已确定的验证阈值**，仅在内部测试折按同一任务的其他日期交换历史交通特征，然后记录差值。分`traffic_only`、`traffic_and_mask`、`mask_only`（F3本身没有mask维度，故不做mask_only）。严格禁止日期协议单日期置换。
- **成对道路回测**：`analyze_r.py`必须严格匹配任务、场景、种子、静态规划的Road/OD数据、求解预算与求解器SHA。道路回测正改善定义为C3成本－模型成本。报告学习调用次数、推理与打分时间、C3回退次数及真正选择的学习Section次数。

**重要：**在线F3历史交通仍只读取出发前原始历史与观测掩码，Static Repair按现有C3工作；没有未来Oracle或TD标签进入模型选择。离线收益则仍是历史Oracle时变规划目标对Static Repair的事后评价，不是道路回测。

## 1. 安装并先运行测试

解压ZIP到仓库根目录：

```bash
cd ~/code/WZL/road_vrp_7
unzip road_vrp7_step6b2_r_v1.zip
python -m pip install -r road_vrp7_step6b2/requirements.txt
bash road_vrp7_step6b2_r/run_b2r.sh test
```

依赖既有代码目录：
`road_vrp7_step6b2/`、`road_vrp7_step6b1_v2/`、`road_vrp7_step6a/`、`road_vrp7_step5b_v3/`、`road_vrp7_stage3/`。不可缺少这些模块。

## 2. 先修复旧的F5结果解释（不重训练、不覆盖）

```bash
STUDY_NAME=b2_seed42_f5_report_corrected_v1 \
  bash road_vrp7_step6b2_r/run_b2r.sh repair-report
```

结果独立位于：
`runs/chicago_july_v1/step6b2_r/b2_seed42_f5_report_corrected_v1/`，包含`audit.json`、`date_report_corrected.json`。原`date/report.json`**原样保留**，可以比对源文件SHA。

## 3. 冻结模型交通置换实验（无需重新训练）

这条命令默认使用已完成的旧B2 Seed42模型：

```bash
STUDY_NAME=frozen_traffic_placebo_seed42_v1 \
MODEL_STUDY=runs/chicago_july_v1/step6b2/b2_full_seed42_v1 \
  bash road_vrp7_step6b2_r/run_b2r.sh permutation
```

执行同任务不同日期、训练/验证/测试严格隔离的**测试折置换**，固定原F3/F4模型和验证阈值。默认20个置换种子`100..119`。保存原始收益和每个置换后的有符号收益差。负对照仅适用于`task`内部测试折，不能据此声称独立月份因果贡献。

## 4. F2/F3/F4多种子离线实验

```bash
for seed in 43 44 45 46 47; do
  SEED="$seed" TREES=120 \
  MODES=F2,F3,F4 METHODS=P2,P4 \
  STUDY_NAME="b2r_ablation_seed${seed}_v1" \
  bash road_vrp7_step6b2_r/run_b2r.sh train
done
```

如果还想复现F5日期修复的真实数据路径，额外用一个**新目录**跑全配置：

```bash
SEED=42 MODES=F0,F1,F2,F3,F4,F5 \
STUDY_NAME=b2r_f5_fixed_seed42_v1 \
  bash road_vrp7_step6b2_r/run_b2r.sh train
```

检查`.../task/report.json`包含P2_F5/P4_F5而`.../date/report.json`中完全没有F5，`summary.json`的`actual_methods_by_protocol`准确列出各折。

**警告**：Seed42的F2/F3/F4已有旧B2训练结果；额外5种子只用于评估模型随机性，同一测试状态被重复利用不能当作独立研究对象。关键对比仍是P2_F3−P2_F2、P4_F3−P4_F2及冻结模型时间置换后收益是否下降。

## 5. 先执行严格匹配的8条30秒在线Pilot

以下命令都固定**task_0008/0009、2025-07-25上午和下午、种子42/43、30秒、20000迭代、5000退火、Top-K 12**。这两项任务是原`task`模型的内部测试任务（**但7月已用于开发，不是独立盲测**）。每组2×2×2=8次。若服务器`--split validation`不含这些场景，则应先核对任务清单，不要自行改一组而忘了改其他组。

```bash
# C3原始官方ALNS基线，在新的R目录下生成，确保完全匹配。
CONTROL=C3 STUDY_NAME=b2r_online_C3_match8_v1 \
  bash road_vrp7_step6b2_r/run_b2r.sh online

# 一样的时序与种子：每8次Section调用抽1次随机候选，否则原C3。
CONTROL=sampled_random STUDY_NAME=b2r_online_random_match8_v1 \
  bash road_vrp7_step6b2_r/run_b2r.sh online

# P2-F2：主要静态边界学习基线；每8次算子调用评分一次；最多2s模型侧打分预算。
CONTROL=learned METHOD=P2 FEATURE_MODE=F2 \
MODEL_STUDY=runs/chicago_july_v1/step6b2/b2_full_seed42_v1 \
STUDY_NAME=b2r_online_P2_F2_match8_v1 \
  bash road_vrp7_step6b2_r/run_b2r.sh online

# P4-F2：不同模型形式的静态边界对照。
CONTROL=learned METHOD=P4 FEATURE_MODE=F2 \
STUDY_NAME=b2r_online_P4_F2_match8_v1 \
  bash road_vrp7_step6b2_r/run_b2r.sh online
```

如果F2真实数据Pilot通过且推理开销足够低，再执行F3交通模型：

```bash
CONTROL=learned METHOD=P2 FEATURE_MODE=F3 \
STUDY_NAME=b2r_online_P2_F3_match8_v1 \
  bash road_vrp7_step6b2_r/run_b2r.sh online
```

不同对照均默认读取已有B2 Seed42`task`协议模型；你也可以通过`MODEL_STUDY`、`MODEL_PATH`明确指定最新重新训练的R模型。不要把日期协议模型放入task内部测试任务对照。

### 结果严格配对计算

```bash
BASE_RUN=runs/chicago_july_v1/step6b2_r/b2r_online_C3_match8_v1 \
TREAT_RUN=runs/chicago_july_v1/step6b2_r/b2r_online_P2_F2_match8_v1 \
STUDY_NAME=b2r_compare_P2_F2_vs_C3_match8_v1 \
  bash road_vrp7_step6b2_r/run_b2r.sh compare
```

输出`runs/chicago_july_v1/step6b2_r/b2r_compare_P2_F2_vs_C3_match8_v1.json`。还应做random−C3、P2_F3−P2_F2（此时`BASE_RUN=P2_F2`）等同键对照。

在`metrics.csv`中额外报告：`r_scoring_calls`（实际评分次数）、`r_scoring_seconds`（候选+特征+推理，排除回退C3耗时）、`r_c3_fallbacks`、`r_gate_fallbacks`、`r_cache_hits`、`r_model_actions`、`r_model_eval_calls`、`section_inference_wall_sec`、`section_feature_wall_sec`。其中`r_scoring_seconds`是**软预算**，某一次特征函数调用可能跨过2秒边界后才返回；30秒搜索墙钟仍由原ALNS终止器约束。

## 6. 建议的接受门槛和研究风险

1. 首先**算法正确性**：在线R Hook恢复、Static-Only特征一致、没有Oracle标签输入、8个严格配对Road.replay结果全有效。不同臂任务、场景、种子集合必须严格相等。
2. 然后**工程预算**：单次30秒中新增评分端总成本目标不超过2秒（软上限）；C3回退后Section不应再有旧版99%空操作循环。记录完整搜索迭代数，不能只报告模型调用次数。
3. 再看**结果**：学习F2相对C3稳定降低道路成本，而F3相对F2稳定降低道路成本；当前8个Pilot样本只够判定接口/趋势，不够统计显著结论。
4. **F5负对照**：真实交通时间对齐信息被打乱后性能应趋于下降；如未下降，暂不声称历史交通具备独立因果贡献。即使下降，也需防路线重复与时间先验的混杂。
5. 最后**研究主线**：先证明边界交通对固定时限ALNS有帮助，才投入SPIN缺失感知编码/对比学习。不要将离线一次Section的收益秒数直接加到30秒搜索的道路级收益上。

## 7. 已发现、尚未解决的限制

- **离线/在线分布偏移**：v2标签来自一批固定Static ALNS当前状态；在线搜索会不断改变状态分布，分数/阈值容易失准。R用低频打分、C3回退和预算上限缓解，而非彻底解决。
- **F3成本**：计算每个候选插入位置的历史走廊特征，可能比F2昂贵；2秒评分预算不保证每次推理都能完美抢占。
- **算法公平性**：C3仍使用原20%候选采样，sampled_random/learned在轮到模型时采用8候选分层采样；应报告random对照及“允许同样候选采样但不使用学习”的消融。最终差异是搜索算法整体效果，不只是模型预测误差。
- **训练/开发划分**：在任务分组中task_0008/0009为内部测试，但7月开发数据已多次查看；后续还需冻结未见月份和新配送任务做最终盲测。
- **模型可信度**：仅加载自己服务器训练并保存的joblib，不要加载不可信GitHub贡献者提供的pickle/joblib文件。
- **单次迭代超时**：官方ALNS原有Destroy/Repair对deadline进行检查，但Python单个模型`predict`无法强制异步终止；超过软上限即回退，后续会记录超时后返回情况。
