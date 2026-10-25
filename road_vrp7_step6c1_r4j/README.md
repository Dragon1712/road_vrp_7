# Step6-C1-R4-J v1：Train 早期 Section 决策定向扩容

## 研究依据（已验证 vs 需要检验）

来源：GitHub `Dragon1712/road_vrp_7` 最新R4-I `r4i_signal_loto_train_fix1`，commit `129f4dc3cab7a46e49472391a54bc77da2cc156c`。

已验证：

- R4-G-Prefix在Train快照已完成298/298个固定K逐事件精确续接。
- R4-H完整Train中72场景快照、51个精确物理状态，565候选动作×K50/200/500。
- R4-H所有K50/200/500 **非零历史最优Road收益**都集中在16个`stop_count <= 2000`的早期场景快照；其他440个候选动作在三个K均是0。
- R4-I四个Train任务留一分析：K500正预测在任务0002上漏掉Oracle收益，在任务0003上造成两次负收益；robust/保守门槛多数情况下完全回退C3。无法证明可泛化模型的在线收益。

**研究决策：** 先增加*新随机种子*及早期Section的真实决策状态，再分别验证“固定迭代无干预”、“固定K连续恢复”、“C3/SHAM反事实”等严格协议。当前只涉及原`task_0000`—`task_0003`，仍然不是新增任务泛化，也不使用任何Validation/Test标签。

## 代码结构

- `run_r4j.sh`: **只负责阶段编排**；直接调用原始（不修改）的R4-G采集、R4G-Prefix官方ALNS恢复、R4-H反事实采集器。保持其原始代码的冻结来源SHA。
- `r4j_core.py`: 独立、只读、逐行检查Train角色、C3终态配对、SHA完整性、精确证明逐窗口记录、C3/SHAM和收益标签的代数一致性、同一替代动作各K覆盖；精确状态计数使用完整SA温度（不舍入）。
- `audit_r4j.py`: `capture`→`prefix`→`cf`门槛检查；对比新旧Train早期/后期收益分布及正收益物理状态数。每阶段都能单独停止；CF审核使用独立全新目录。
- `tests/`: 17个合成和Shell回归测试，含错误种子、Validation日期、破坏SHA、错配SHAM、错误标签、遗漏K、跨角色污染。

无改动：`road_vrp7_step5b_v3/`、`road_vrp7_step6c1_r4a`、`r4b`、`r4c`、`r4g`、`r4g_prefix`、`r4h`、`r4i`、原数据与历史报告。

## 采样协议

| 参数 | R4-J默认 |
|---|---|
| 搜索方法 | 冻结官方C3 ALNS 7.0.0，20k迭代、30秒搜索墙钟 |
| 任务 | task_0000–task_0003，必须Train |
| 交通场景 | `2025-07-02_0800`, `2025-07-09_1700`，R4-G还会核查清单角色和eligible |
| 随机种子 | **62、63**，与原R4-G的60、61隔离 |
| Section状态采样里程碑 | **100、250、500、750、1000、1250、1500、1750** |
| 每条搜索最多快照 | 8个；只捕获真实的Section pre-selection |
| Prefix验证 | K=1、50、200、500；仅使用源30秒搜索中存在的合法窗口 |
| Counterfactual | K=50、200、500；C3、SHAM_C3和至多8个替代Section |
| 后续Road评估 | 事后Road.replay；不把道路标签作为在线输入 |

注意：里程碑是**请求点**，下一次真正被RouletteWheel选中的Section才会被捕获；停止计数可能稍大于请求点。审计以实际`stop_count <= 2000`作为早期分类，而不会伪造固定位置。新种子62、63只是新的搜索轨迹；原4个任务的外部分布和信息仍有限。

## 服务器执行（仓库根目录）

```bash
cd ~/code/WZL/road_vrp_7
unzip -o road_vrp7_step6c1_r4j_v1.zip
bash road_vrp7_step6c1_r4j/run_r4j.sh test
bash road_vrp7_step6c1_r4j/run_r4j.sh plan
```

### 第一阶段：1个任务、1场景、1个新seed的端到端Smoke

单案例采集（不会读取旧通用`TASK_IDS`等环境变量）：

```bash
bash road_vrp7_step6c1_r4j/run_r4j.sh collect-smoke
```

它创建`runs/chicago_july_v1/step6c1_r4g/r4j_train_early_smoke_s62_v1_{plain,capture,audit}`。R4-G审计使用`REQUIRE_STRICT=0`仅保存诊断结果，**不能**以此称16/16/1/1的固定20k非干预已获验证；下一步精确K证明仍然必需。

在后续命令中设置这个Smoke来源：

```bash
R4J_SOURCE_STUDY=r4j_train_early_smoke_s62_v1 \
  bash road_vrp7_step6c1_r4j/run_r4j.sh proof-smoke

# 要先证明全部源快照，再允许R4-H反事实采集
R4J_SOURCE_STUDY=r4j_train_early_smoke_s62_v1 \
  bash road_vrp7_step6c1_r4j/run_r4j.sh proof-pilot

R4J_SOURCE_STUDY=r4j_train_early_smoke_s62_v1 \
  bash road_vrp7_step6c1_r4j/run_r4j.sh cf-smoke
```

只有SHAM和跨K轨迹前缀都严格通过才视为Smoke完成。

### 第二阶段：16案例×早期Section状态Pilot（主实验）

**必须使用与Smoke不同的新实验目录。**

```bash
R4J_SOURCE_STUDY=r4j_train_early_s62s63_v1 \
  bash road_vrp7_step6c1_r4j/run_r4j.sh collect-pilot

R4J_SOURCE_STUDY=r4j_train_early_s62s63_v1 \
  bash road_vrp7_step6c1_r4j/run_r4j.sh proof-pilot

# 先采4个Train任务×最多2个不同阶段快照，不直接大量分支跑满
R4J_SOURCE_STUDY=r4j_train_early_s62s63_v1 \
  bash road_vrp7_step6c1_r4j/run_r4j.sh cf-pilot
```

成功后，才按算力与Pilot正例覆盖决定是否执行完整CF：

```bash
R4J_SOURCE_STUDY=r4j_train_early_s62s63_v1 \
R4J_MAX_NEW_UNITS=240 \
  bash road_vrp7_step6c1_r4j/run_r4j.sh cf-full
```

若出现`R4H PARTIAL`，只需重复命令，加`RESUME=1`，每轮最多新增240个有效unit。**不要**删除、续写或覆盖过去R4-H full结果目录。

> `cf-full`的分块次数取决于实际捕获数、有效K、候选数。上限可能超过3,000个分支；只在Pilot经过审计并确认有价值后运行，切勿先启动全部计算。

### 第三阶段：仅当新CF完整结束，运行独立来源审计

```bash
R4J_SOURCE_STUDY=r4j_train_early_s62s63_v1 \
  bash road_vrp7_step6c1_r4j/run_r4j.sh audit

cat runs/chicago_july_v1/step6c1_r4j/\
r4j_train_early_s62s63_v1_early_audit_v1/report.md
```

报告对K=50/200/500给出：早/晚Section状态候选数、正/负/零Road收益、含正收益的精确物理状态数；同时列出原R4-H与新R4-J K500正收益数量及正例率。但这仍是Train开发分布的变化，不是独立测试。来源SHA单独存入`provenance.json`。

### 第四阶段：可选额外新种子组（只在第一批数据有效时）

```bash
R4J_SOURCE_STUDY=r4j_train_early_s64s65_v1 R4J_SEEDS=64,65 \
  bash road_vrp7_step6c1_r4j/run_r4j.sh collect-pilot
```

之后同样执行proof-pilot、cf-pilot，保持`R4J_SOURCE_STUDY`和`R4J_SEEDS`身份一致；新一组使用不同目录，不合并旧实验的checkpoint。

## 当前不能宣称什么

- `REQUIRE_STRICT=0`只准许报告墙钟截断情况，**没有**放宽R4-G-Prefix逐事件精确恢复或R4-H SHAM控制。
- “更多正例”不是“一个可部署模型取得收益”，也不是增加网络容量的充分理由。
- 在4个原Train任务内增加seed，没有创建新的独立任务；跨场景的相似搜索起点也不是IID观测。
- 新实验仍为fixed-K恢复后的离线Road.replay，不是30秒在线ALNS策略实测。

**下一步（不在本包中）：** 若新Seed明显增加了跨任务早期正收益状态，用严格的原Train任务级分组及跨seed留组重新做R4-I；在冻结配置且明确独立的Validation协议后，再开发多模态交通FiLM/对比学习消融。若正例仍集中少数任务，则应扩大真正的独立配送任务，而非继续重复同一任务的种子。
