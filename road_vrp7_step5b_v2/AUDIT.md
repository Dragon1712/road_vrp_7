# Step5-B v2详细审计记录（2026-10-17）

## 审计来源与范围

- 通过GitHub连接读取 `road_vrp7_stage3/road_vrp.py`（完整220行）、`solvers.py`（完整244行）、`benchmark.py`、`road_vrp7_step5b/section_operators.py`、`run_experiment.py`、`analyze_step5b.py`（主分支当前提交`80245979a624095fc963bca45d0a79aa93218be6`）。
- 核对ALNS官方7.0.0文档：`OperatorSelectionScheme.__call__(rng,best,curr)`、`update(candidate,d_idx,r_idx,outcome)`；算子通过 `ALNS.add_destroy_operator/ add_repair_operator`，使用 `engine.iterate()`。
- 不提供对完整Chicago结果的真实性能宣称，因为实验数据未全部下载到本环境，且本地未安装 `alns==7.0.0`。下述测试是合成与模拟协议级测试，不等于真实GitHub运行验证。

## 审计检查表

| ID | 审计点 | 结果 | 证据 / 限制 |
|---|---|---|---|
| A01 | 不依赖旧仓库、使用当前Stage3原生API | PASS（静态） | 严格导入 Road/OD/Objective 和原 ALNS 数据结构；完整服务器执行待Smoke |
| A02 | 固定有向时变目标 | PASS（合成） | 修复后 `obj.cost`；服务时间、OD阶段时间传播与Stage3一致；cheap static只排序 |
| A03 | 客户不丢失/不重复 | PASS（合成） | 整段删除/插入、跨车交换均使用完整块；每个输出 `obj.feasible` 检查，最终 `legal` |
| A04 | 每车非空、容量合法 | PASS（合成） | Section仅选择移除后源车仍非空的位置，插入/交换检查容量 |
| A05 | 不会主动返回不可行部分解 | PASS | 截止、无候选或无Section时返回原始完整解，禁止部分解评分 |
| A06 | C2排除原位无效重插 | PASS（合成） | 完整路线签名与原解比较排除，记录 noop_rejected |
| A07 | C3-6 Top-K精确候选重算 | PASS（合成） | 对静态代理前K个完整候选调用 `obj.cost()`；其他候选不作为最终解 |
| A08 | C3-6跨车交换完整性 | PASS（合成） | 原始路线两个片段直接互换，合法性检查，记录尝试/选中次数 |
| A09 | C4-C6时间敏感UCB协议 | PASS（模拟） | 接口与ALNS 7文档一致，成本含选择到更新延迟；服务器集成待Smoke |
| A10 | C5支持度0严格静态回退 | PASS（合成） | 零支持对25种子静态/gated/shuffled相同，且无需动态加权 |
| A11 | Static模式不读取未来交通 | PASS（合成） | 静态模式ToyOD对动态调用直接抛错；遍历三种Section模式通过 |
| A12 | C6错位负对照有效 | PASS（逻辑） | 仅shuffle有支持片段的交通特征，保留片段自己的静态项和支持门控 |
| A13 | 固定时间预算与迭代上限 | PASS（代码） | 共享Stage3 BothBudget；预计算单列记录、不声称端到端5秒；CPU耗时可能超截止少量尾部开销 |
| A14 | 完整网格、哈希与断点恢复 | PASS（单元） | 验证任务/道路SHA、源文件SHA、CSV/JSONL主键；中断后不自动删除尾行 |
| A15 | 配对统计与10任务Bootstrap | PASS（合成） | 缺失metrics/solutions和不合理计数强制拒绝 |
| A16 | 算子统计透明性 | PASS（模拟） | 新增changed/unchanged/improving/new_best/noop/swap/exact eval/accepted；每个destroy-repair对日志 |
| A17 | 官方ALNS7.0.0真实环境 | **未验证** | 本地没有官方ALNS wheel、没有完整7月动态成本与任务；服务器Smoke必做 |
| A18 | 交通增强真实胜出 | **未验证** | 本版本不能提前宣称性能增益，等待严格配对的真实Pilot/Validation |

## 重要设计风险 / 遗留问题（不可隐藏）

### R1：不是完全正交的C0-C6实验（严重性：中）

- C1固定单块旧Section，原Step5-B B2可使用两块。两者不完全相同。
- C2使用旧Section选择器，C3换成新的静态Section选择器，增加跨车交换且使用Top-K，**C3-C2不能解释为Top-K的纯粹贡献**。
- **C3-C4**共用同一个新静态Section选择器、同一Top-K修复、同一Swap候选集，唯一区别为RouletteWheel与TimeUCB。**C4-C5**共用TimeUCB+候选修复，唯一区别为交通门控评分。**C5-C6**共用所有机制，唯一区别为交通特征在有支持候选之间错位。
- 如果C3获得收益，需要额外增加 `C3_noswap`、`C3_fullscan_swap` 使Swap与Top-K的贡献独立。

### R2：支持度 proxy 不是真实动态路径覆盖（严重性：中）

`PathSupportProxy`沿静态最短路径和出发时刻的观测掩码取长度占比，不是动态OD旅行途中真实边序列支持；Chicago OSM严格映射也是地理候选而非人工逐段验证。门控减少了“完全未观测也给予高交通评分”的问题，但不能保证局部可信度准确。

### R3：时变Oracle属于历史hindsight（严重性：高，论文表述要求）

`OD.dynamic`来自历史完整未来，因此只能用于确定上界/机制回顾，不能声称提前知道未来。正式部署需基于出发时刻可用的历史交通及预测器，并在同一道路回放上独立检验。

### R4：预算感知的目标是即时改善/秒而非长期搜索奖励（严重性：中）

`TimeUCB`直接用非负短期候选改善/耗时，不计那些帮助逃离局部最优但本次更差的解的长期价值，因此不一定优于经典RouletteWheel。采样需要足够轮次，5秒不保证探索覆盖全部组合。候选评分浮点运行开销计入迭代搜索钟表，但构造/OD/路径支持仍在钟表之外。

### R5：计算公平性与测试配置（严重性：中）

5秒是在线ALNS `iterate()`预算，`support_precompute_sec`和`od_precompute_sec`分别记录。对最终部署须评估 `(OD + support + construct + search + replay/evaluation)` 的成本，并明确缓存冷/热状态。C5可能因特征计算过慢而迭代数少，这正是等搜索秒数的真实代价而不是不公平差异。

### R6：中断原子性（严重性：低～中）

每条记录分别fsync写`solutions.jsonl`和`metrics.csv`，发生进程中止可能形成不同步的两份文件。为了避免误配，本版本`--resume`严格拒绝；用户必须人工确认与修复；不承诺自动“无损恢复”。同时禁止两个进程写相同目录。

### R7：排序比较和真实收益之间的差距（严重性：中）

统计了改变**实际抽样Section**的比例，不再用“排名第一改变”冒充实际决策变化。但由于ALNS trajectories发散，两个独立实验的相同seed不代表同一时刻有严格同一状态；实际Section影子比较只有同一候选池上的静态-vs-交通选择对照，算子收益仍需最终配对回放验证。

### R8：数据划分（严重性：中）

此前7月26—31已经被用于Stage3结果分析。可以当开发集/调参集，但不能以“第一次从未见过的测试”呈现。对全年数据，建议至少留一个不同月份作为只运行一次的终验，且每月数据先做 QC/时区核验/OSM匹配检查。

## 本地已运行的测试命令

```bash
python road_vrp7_step5b_v2/tests/test_v2_units.py
python road_vrp7_step5b_v2/tests/test_runner_mock.py
python road_vrp7_step5b_v2/tests/test_analysis.py
python -m compileall -q road_vrp7_step5b_v2
bash -n road_vrp7_step5b_v2/run_stage5b_v2.sh
```

期望输出关键字：PASS。目前本地测试全通过。这些测试只说明算法约束、控件协议与统计逻辑基本符合设计，**并不等价于Step5-B v2在芝加哥数据上性能获胜**。

## 论文启发与准确归属

- Dragomir & Müller, *Problem size reduction methods for large CVRPs*, COR 2024, DOI `10.1016/j.cor.2024.106820`：Section概念、固定内部顺序与反转、问题压缩动机；**本版本主要复用局部连续片段邻域，不实现完整超级节点缩图**。
- Johnn et al., *A Graph Reinforcement Learning Framework for Neural Adaptive Large Neighbourhood Search*, COR 2024, DOI `10.1016/j.cor.2024.106791`：算子选择依赖状态的思路；**本版本UCB无需训练，不属于论文GRLOS复现**。
- ALNS官方7文档：https://alns.readthedocs.io/en/latest/api/select.html：选择器函数接口。
- 低覆盖门控、正拥堵变化特征、UCB即时奖励公式均是本项目待验证研究提案，而非上述论文已经证明适用于芝加哥TD-CVRP的结论。
