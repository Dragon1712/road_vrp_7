# Step5-A v2 Pilot：分析器错误修复（2026-10-16）

## 1. 根因和修复范围

已读取 GitHub `Dragon1712/road_vrp_7`，确认出错的是`road_vrp7_step5a_v2/analyze_step5a.py`。

- Stage3 `runs/chicago_july_v1/stage3/validation/run_config.json` 采用 **`tasks_manifest_sha256`**（plural）。
- Step5-A v2 `runs/chicago_july_v1/step5a_v2/pilot/run_config.json` 采用 **`task_manifest_sha256`**（singular）。
- 两者本应匹配的值相同：`520afcd810fcb8c519903c39f760109aebcb7104d7d92d3e811c8aa60d228a24`。
- v2 原分析器执行`basecfg[k]!=cfg[k]`，直接触发`KeyError: 'tasks_manifest_sha256'`。

**本补丁只修改`analyze_step5a.py`，不会修改两份运行配置，不会重算ALNS解，不会覆盖`metrics.csv`或`solutions.jsonl`。**

为了避免“修好了报错，但悄悄放松审计”，新的`validate_stage3_reference_config()`检查：

1. 相同的Stage2道路成本与edge_index SHA256；
2. 名称不同但数值必须相同的**任务清单哈希**；
3. 同一5秒预算、5000迭代上限、20分钟OD采样与5小时Horizon；
4. 每个Pilot任务文件的SHA256必须是Stage3任务文件的真实子集；
5. Pilot随机种子、情景、规划模式必须在Stage3验证集覆盖范围内；
6. Stage3参考包含official_alns；匹配记录须不存在重复、缺失或多出的主键。

另外，显式标注：即使输入与预算完全一致，Step5 v2的日志开销与Stage3不同，因此Stage3/Step5“基线复现”只能诊断可比性，不能据此宣称两者执行轨迹或二进制实现绝对相同。

## 2. 一键修复

将 `step5a_v2_analyze_fix/` 放进 `road_vrp_7/` 根目录，执行：

```bash
cd ~/code/WZL/road_vrp_7
bash step5a_v2_analyze_fix/install_patch.sh
bash road_vrp7_step5a_v2/run_stage5a_v2.sh analyze pilot
```

安装器只接受原始 v2 分析器的SHA256或者已修复版本；对自行改过的分析器拒绝覆盖。覆盖前创建：

`road_vrp7_step5a_v2/analyze_step5a.py.before_step5a_v2_fix.bak`

修复后预期完整生成/刷新：

```
runs/chicago_july_v1/step5a_v2/pilot/analysis/
    solver_summary.csv
    feature_effectiveness.csv
    operator_contribution.csv
    operator_pair_contribution.csv
    sample_operator_events.csv
    paired_gains.csv                # 原错误使这个文件未生成
    analysis_summary.json           # 原错误使这个文件未生成
```

**不需要重新执行176条Pilot。**

## 3. GitHub Pilot已有真实结果（均为实际历史道路成本、单位秒，越低越好）

### Oracle均值（每组8次）

| 配置 | 成本 | 相比Baseline的绝对变化 |
|---|---:|---:|
| baseline | 4365.52 | — |
| at_rr | 4291.80 | -73.72 |
| at_timeonly | 4320.36 | -45.16 |
| at_misaligned | 4299.89 | -65.63 |
| te_local | 4321.80 | -43.72 |
| te_wr | 4364.44 | -1.08 |
| te_positive | 4298.84 | -66.67 |
| both | 4329.64 | -35.88 |
| placebo_both | 4302.92 | -62.59 |

### 有严格配对的核心比较

| 比较（Oracle） | 平均配对改善 | 获胜次数 |
|---|---:|---:|
| AT-RR vs AT-TimeOnly | +0.624% | 4/8（另1平局） |
| AT-RR vs AT-Misaligned | +0.181% | 3/8（另2平局） |
| AT-RR vs Baseline | +1.683% | 7/8 |
| TE-WR vs TE-Local | -1.036% | 4/8 |
| TE-Positive vs TE-WR | +1.505% | 7/8 |
| Both vs Placebo-Both | -0.652% | 3/8 |

特别注意：2个任务×2个交通时段×2个种子，**独立任务只有2个，统计区间不能支撑泛化结论**。

### 交通机制与可观测性

- AT-RR：`129/1028`次影子比较选中不同集合，约**12.55%**；交通水平/趋势评分平均0.0122/0.00281，ETA评分平均0.1448。**真实改变了搜索，但额外权重偏弱**。
- TE-WR：`1024/1027`次与原Worst的移除集合不同，约**99.71%**，但均值无提升。
- 特征支持审计：480个OD，有178个至少包含一条观测支持边，约**37.08%**；平均边路径支持率**11.88%**、中位数为0，**不能把混合道路OD比值全部当作实测拥堵标签**。
- v2 Pilot所有变体平均搜索时间约5秒，因此收益不能归因于不同名义预算；实际迭代数仍可能有差异。

## 4. 下一阶段决策

1. **先只修复分析器并重跑`analyze pilot`**，确认输出`analysis_summary.json`中`experiment_complete=true`与`stage3_rechecked=true`，再核对`paired_gains.csv`及算子贡献文件。
2. **保留当前冻结的v2参数，不根据仅2个任务的Pilot增益重新调整AT-RR权重/TE-WR排序。** 若需要修改算法，另建`v2b`并重新设定验证计划，不能混合既有176条结果。
3. **准备完整验证集5280条记录**（10任务×8验证情景×3种子×11组×2规划模式），以严格配对报告AT-RR、TE-WR、TE-Positive和联合算子的收益与代价。执行时间约7.3小时累计搜索预算，实际更长。
4. 在支持率偏低的条件下，必须保留`at_misaligned`负对照、`placebo_both`算子数量对照，以及按Static/Oracle分层结果。
5. 再讨论Step5-B Section；目前Pilot不足以宣布TE-WR或联合算子机制成功。避免把历史Oracle视作可部署交通预测。

## 5. 回归测试结果与限制

新的`tests/test_analyzer_fix.py`实际导入并调用**修复后的分析器源码**，测试：字段别名、静态/动态配置哈希不匹配拒绝、预算不匹配拒绝、完整合成Pilot导出、Stage3参考子集配对、缺失数据拒绝。**本地测试通过**。

当前容器无法直接下载GitHub的4.75MB原始`solutions.jsonl`，所以无法在此对真实Pilot运行整个分析流程；已通过GitHub接口读取真实配置、指标和机制摘要进行独立审计。请在服务器运行`analyze pilot`完成真实产物验证。
