# Step6-C1-R4-M v1 — 嵌套任务留一风险校准的Section保守委托

## 1. 研究动机及实际GitHub证据

截至原始仓库 commit `62caafdac72b55053b7a932005f6e549aac00199`：

- **R4-G/R4-G-Prefix**：真实Train Section快照及恢复证明，前缀精确；不同于30秒在线改善。
- **R4-H/R4-K**：原Seed 60/61有565个候选动作；新增Seed 62/63有1007个。每个动作有K=50、200、500的Road.replay事后收益标签。
- **R4-L**：200场景快照、100粗粒度搜索起点、4个原Train任务，共1572候选。K500正/负/零：208/254/1110。
- **R4-L全特征K500、非Shuffle、早期门控、0秒预测阈值**：四个任务LOTO联合净收益−186.810546875秒（按200个场景快照平均−0.934053秒），81次委托、9正/14负；task_0001负收益多，task_0002完全弃权。旧种子→新种子全部弃权，反向迁移平均−2.489秒/快照。

结论：更多正标签≠跨任务可部署改进。下一步应严谨检查收益预测**过度乐观的风险**能否用仅Train内部的数据校准，且允许放弃委托。

## 2. 本包做什么（与R4-L的区别）

仅创建新目录 `road_vrp7_step6c1_r4m/`，**不修改**冻结ALNS、Section/Repair、Road.replay、R4-H/K/L和任何已有结果。

数据读取直接调用未修改的：

- `r4l_core.read_old()`：原R4-H + R4-G Prefix来源SHA、角色与标签核验。
- `r4l_core.read_new()`：新R4-K SOURCE_COMPLETE核验、原始K50决策几何恢复、来源SHA。
- `r4l_core.make_splits()`：4个Train任务留一+2个两组种子迁移。
- `r4l_core.with_mask()`：原始决策前特征与相关场景样本权重定义（完整精度物理状态计数仍然和粗粒度组分开）。

外层每一折只用**训练侧**标签学习Ridge，并在其内部再按任务留一获得OOF预测，形成预测高估残差：

\[
 e_{500}(s) = \max(0, \widehat U_{500}^{\mathrm{inner\ OOF}}(a_s) - \Delta_{500}(a_s)).
\]

其中 `a_s` 只依据该内折模型在一个Section快照中的预测最大值选取；不能根据未来Road标签挑动作。保守选择分数为：

\[
 \mathrm{score}(a) = \widehat U_{500}(a)-Q_q(e_{500}),\quad q\in\{0.50,0.75,0.90\}.
\]

`dual_lcb_q75`同时要求K200和K500的下界均为正，使用

\[
 \mathrm{score}_{dual}(a)=\min\big(\widehat U_{200}(a)-Q_{.75}(e_{200}),\ \widehat U_{500}(a)-Q_{.75}(e_{500})\big).
\]

只有 `stop_count <= 2000`、预测保守分数严格为正，而且内层校准支持量满足 **≥5个正分数样本且覆盖≥2个不同Train任务** 才委托；否则严格回退C3。没有新增Road标签泄漏。`ridge_ref`保持原R4-L同类K500 Ridge +早期门控与0阈值，作为参照。

**不能把经验残差分位数称为严格Conformal覆盖保证。** 四个训练任务远不足以证实分布外误差边界。本实验更接近*cross-fitted conservative deferral*诊断。

## 3. 预注册比较和结果

- 六个外层留组：留出task_0000/0001/0002/0003各1个，以及Seed 60/61→62/63、Seed 62/63→60/61。
- 完整/几何Only特征消融各1组。
- 标准标签与训练标签Shuffle负对照各1组。
- 五种固定策略：`ridge_ref`、`lcb_q50`、`lcb_q75`、`lcb_q90`、`dual_lcb_q75`；因此一共6×2×2×5=**120**个配置。
- 统一在留组上报告：委托、正/负/零收益次数、每快照K500平均收益、总伤害、最坏伤害、随机C3+动作均匀对照期望和不可部署事后Oracle。
- **不允许从这120组测试结果中挑一个冠军再宣称独立泛化**；后续独立Validation前必须明确冻结配置。

输出：`report.json`、`report.md`、`fold_predictions.jsonl`、`provenance.json`。所有产物均写**全新的**独立目录，禁止覆盖已完成实验。

## 4. 服务器安装和运行

在 `~/code/WZL/road_vrp_7` 仓库根目录执行：

```bash
unzip -o road_vrp7_step6c1_r4m_v1.zip
bash road_vrp7_step6c1_r4m/run_r4m.sh test
bash road_vrp7_step6c1_r4m/run_r4m.sh preflight
STUDY_NAME=r4m_nested_risk_train_v1 \
  bash road_vrp7_step6c1_r4m/run_r4m.sh probe
cat runs/chicago_july_v1/step6c1_r4m/r4m_nested_risk_train_v1/report.md
```

`preflight`不会训练，但会完整读取并校验Train来源数据，可能访问1007个新来源的K50 unit（无ALNS计算）。`probe`生成新结果；不重新搜索，不请求Validation/Test数据，也不计30秒推理开销。

### 上游文件缺失怎么办

GitHub提交包含R4-K审计和R4-L结果，但源 `runs/chicago_july_v1/step6c1_r4h/r4j_train_early_s62s63_v1_cf_full/report.json` 未在浏览时显示。由于R4-L已运行成功，服务器原目录可能存在完整结果。**不能**通过修改SHA或临时伪造`report.json`绕过验证；先确认原`cf_full`目录完整，再运行preflight。

如果上述源已通过R4-L但被清理，需从真实实验备份恢复同一份文件；无法校验则停止，不以GitHub汇总数据重建训练样本。

## 5. 后续决策

- 若新保守策略在预注册的**所有Train留组**中提高净收益、限制伤害、Shuffle负对照不再呈现同样现象，则**冻结该一组协议**，再单独定义Validation与真实30秒在线ALNS对照（含推理延迟和no-op）。
- 若多数策略校准支持不足而完全回退C3，说明当前跨任务样本不足以提供可置信的委托；应增加真正不同的Train配送任务并改善Section交通/道路特征，而不是将“零伤害”冒充有效策略。
- 若收益看似提升但不稳定，继续分析任务级、随机种子级相关性和正负标签的特征条件；**不要**调参直到留出得分好看再视为外部效果。
- FiLM道路交通融合、双模态对比学习应在简单风险校准与基础Section候选排序的外部泛化边界明确后，按F0～F5严格消融；不是本轮结果自动支持的结论。

## 6. 研究完整性约束

1. **Train only**，不拟合/选择任何来自Validation（7月22–25）和Test（7月26–31）的标签。
2. 内层OOF校准和外层留组严格分开；任务级随机性高，样本不是IID。
3. 经验风险界非分布无关置信下界，不能报告形式化coverage。
4. 固定K500事后收益不是30秒在线总目标改善。
5. 如果本包在服务器上运行失败，以报错和完整来源为准；本地synthetic unit tests不能代替Chicago真实运行。
