# Chicago 2025-07 交通数据批量质量审计与ALNS场景预选

## 0. 实施范围与结果边界

- 目标仓库：`https://github.com/Dragon1712/road_vrp`（编写时核对了 `main` 提交 `e64d6f1118ba910d983ff07aa1bdc892ccdb409c`）。
- `chicago_data/` 原始文件共 **30 个**，为 `traffic_2025-07-02.csv`～`traffic_2025-07-31.csv`。本工具默认 **只检测后29天（7月3～31日）**，严格按照用户要求；7月2日另用于真实CSV烟雾验证。7月1日没有上传。
- 脚本在运行前 **不会预先认定哪天合格**。其余29个大CSV在本环境无法逐行取得，所以未声称哪些具体7月日期已经达到质量门槛。必须在存有原始CSV的服务器上运行，才能得到真实排名。
- 原数据是Chicago Traffic Tracker路段**速度估计**，不是车流量计数；`speed<=0` 和 `speed>100 mph`（可调）都不作为训练或路由真实速度。
- 目标是筛选具有充分已观测交通数据的**路线验证日期和出发时段**，而非直接训练ALNS或预测道路交通。未监测的OSM边仍需要静态回退/预测。经过严格筛选的71条监测走廊与507条有向边映射仍需要地图人工核对。

## 1. 解压与运行：从仓库根目录

```bash
cd ~/code/WZL/road_vrp
unzip chicago_july_quality_tool.zip

python chicago_july_quality_tool/batch_quality_select.py \
  --input-dir chicago_data \
  --segment-index datasets/processed/chicago/segment_index.csv \
  --mapped-segments datasets/processed/chicago/osm_match_loop/safe_mapping/selected_monitor_segments.csv \
  --edge-mapping datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv \
  --start 2025-07-03 --end 2025-07-31 \
  --departures 08:00,17:00 --horizon-hours 4 --buffer-hours 1 \
  --selection-split test \
  --max-per-departure 3 --weekend-quota 1 \
  --output-dir runs/chicago_data_qc/july_2025 \
  --fail-on-missing --fail-on-structural --fail-on-empty
```

如只想统计和列出缺失而不中止，移除末尾三个 `--fail-on-*` 开关。可用 `--hash-files` 对全部约1GB CSV做SHA256，计算稍慢但便于论文归档。脚本用单日处理方式，避免将30天原始记录全部常驻内存；只需 `numpy`、`pandas`。

检查全部指标：

```bash
python chicago_july_quality_tool/batch_quality_select.py --help
```

**默认窗口**：08:00–12:00与17:00–21:00，另验证各自末尾**1小时**缓冲数据可用性。该缓冲与已有v2 `ODCost` 的4小时OD采样+1小时边界要求相对应。示例的出发时间符合此前上午/傍晚配送设计；可按任务需要通过`--departures`修改。

## 2. 哪些数据被筛进ALNS？

每日硬性结构检查（含原因）：

- 能读CSV，且包含`time,segment_id,speed`。
- 原始时间解析、所属日期、10分钟slot完整性，144个slot及1047个已定义路段的二维网格。
- 监测段ID与已有 `segment_index.csv` 完全匹配。
- `(slot, segment_id)`不得重复；`record_id`若提供也不得重复。
- 道路静态字段（street、direction、长度、经纬度）同一路段不得在日内或跨日出现不一致（若缺元数据也会标红）。
- 不将无效速度或静态回退伪装为观测；速度`<=0`及极端不可信`>100 mph`均视为不可用，单独统计。

配送时段门槛（可调整的*预先规定筛选参数*）：

| 指标 | 默认要求 | 意义 |
|---|---:|---|
| 全天1047段观测率 | >=40% | 极差整天不进入核心验证 |
| 4小时1047段观测率 | >=50% | 全局观测质量 |
| 4小时71段平均观测率 | >=65% | 与Chicago Loop有监测映射的路段 |
| 4小时507条映射有向边加权平均观测率 | >=65% | 更贴近Road TDSP的交通覆盖 |
| 4小时逐slot映射边覆盖P10 | >=45% | 避免时段内短暂严重失联 |
| 4小时每小时加权覆盖最低值 | >=55% | 避免某一小时严重缺失 |
| 额外1小时加权覆盖率 | >=40% | 行程缓冲段不会完全失联 |
| 4小时至少有一半slot可观测的71段占比 | >=65% | 避免多数映射走廊一直缺失 |
| 4小时完全无映射边观测的slot | 0 | 关键交通窗口不应整片失联 |

上述阈值是**初步探索实验的预注册选择规则**，不属于外部文献公认标准。若没有满足要求的时段，输出拒绝原因，**不擅自放宽**。若需要放宽，应另建实验配置和输出文件夹；切勿按照ALNS回测增益反向挑日期。

有向边覆盖采用已有 `safe_osm_edge_mapping.csv` 中“每条监测走廊映射到几条OSM有向边”的权重，要求不存在一条OSM有向边由多个监测段同时标记，且这些映射与之前6月使用的是**同一份冻结路网映射**。这是代理性观测覆盖指标；真实道路网整体覆盖不能称为100%。

## 3. 时间划分与推荐的用途

默认按天严格划分：

| 角色 | 日期 | 用途 |
|---|---|---|
| `train` | 2025-07-03～07-21 | 训练交通补全、预测或搜索评分模型 |
| `validation` | 2025-07-22～07-25 | 选择阈值、算法参数和训练checkpoint |
| `test` | 2025-07-26～07-31 | 未见过的日期，用于最终ALNS交通泛化回测 |

默认 `--selection-split test` 只从测试日期中推荐场景；每个出发时间最多3个，若有达到条件的周末候选优先安排至少1个。故总推荐上限为6个日期×出发时段组合。日期在ALNS执行前、仅按交通观测质量确定，不能再按优化结果筛选。

**提醒**：原v2生成的10个配送任务是人工合成订单，继续复用这批客户集合只能验证**跨日期**，不能证明跨订单或跨城市泛化；需要另外生成独立客户任务作为最终结论。

若想对整个月的已观测交通进行数据质量评估，而不专门筛选训练/测试时段：

```bash
python chicago_july_quality_tool/batch_quality_select.py \
  --selection-split all \
  --output-dir runs/chicago_data_qc/july_2025_all
```

## 4. 输出文件说明

```text
runs/chicago_data_qc/july_2025/
  daily_quality.csv          # 29个日期的结构、缺失、静态一致性、观测情况
  scenario_quality.csv       # 29×2日期-出发时段，包含score和拒绝原因
  eligible_scenarios.csv     # 所有通过阈值的场景（可跨划分）
  rejected_scenarios.csv     # 不能用于ALNS核心验证的场景及原因
  selected_scenarios.csv     # 默认从test日期预选的最多6个场景
  selected_scenarios.json    # 可用于导出v2任务情景
  audit_manifest.json        # 输入路径/文件大小/映射哈希/阈值/版本
  质量报告.md                  # 可读总结
```

推荐优先看：`质量报告.md`、`daily_quality.csv`和`selected_scenarios.csv`。`quality_score`只是数据完整性排名，并非交通强度、ALNS搜索难度或实验增益评分。

## 5. 如何将筛选结果用于Chicago ALNS？

### 5.1 先重建7月道路成本（不能复用6月成本）

**步骤一：将7月2～31日作为连续时间段单独清洗。** 不要把6月数据与7月直接连接成连续10分钟张量，因为中间存在日期缺口。

```bash
python chicago_road_traffic/chicago_dual_modal_starter/prepare_chicago.py \
  --input-dir chicago_data \
  --pattern 'traffic_2025-07-*.csv' \
  --output-dir datasets/processed/chicago/july_2025 \
  --format csv
```

**步骤二：使用冻结映射构造7月路网时变成本。** 检查脚本输出的新coverage、时间连续性和FIFO指标：

```bash
python chicago_td_cost_starter/build_dynamic_edge_costs.py \
  --graphml datasets/raw/chicago/chicago_loop_drive.graphml \
  --mapping datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv \
  --traffic-long datasets/processed/chicago/july_2025/traffic_long.csv \
  --time-index datasets/processed/chicago/july_2025/time_index.csv \
  --output-dir datasets/processed/chicago/dynamic_costs_july_2025
```

前述成本属于“历史已观测速度+未观测位置静态估计”，对7月日期是真实历史**部分观测**的离线近似，并非完整实测交通。路段可用性在训练/预测时仍须保留掩码。

### 5.2 创建新的任务情景而不修改原有10任务

```bash
python chicago_july_quality_tool/make_validation_tasks.py \
  --selected runs/chicago_data_qc/july_2025/selected_scenarios.json \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks_10 \
  --output-dir datasets/processed/chicago/tdcvrp30/tasks_july_validation
```

然后对照原版ALNS+（示例仍需安装官方 `alns==7.0.0`）：

```bash
python chicago_tdcvrp_solver_suite_v2/benchmark_traffic_alns.py \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks_july_validation \
  --cost-dir datasets/processed/chicago/dynamic_costs_july_2025 \
  --od-cache-dir runs/chicago_solver_suite_v2/od_cache_july \
  --output-dir runs/chicago_solver_suite_v2/july_qc_alns_compare \
  --methods upstream_plus,upstream_traffic \
  --planners oracle \
  --seeds 42,43,44 \
  --max-tasks 10 --max-scenarios 0 \
  --vehicle-rule exactly --od-step-min 20 --od-hours 4 \
  --max-iterations 5000 --wall-sec 5
```

此命令用于验证**动态目标条件下的算子贡献**，不是未来交通预报；需要真正预测用途时应再用出发前可见的输入产生未来OD成本。若你要评价Traffic-ALNS v2的C/D算子，使用其独立`benchmark_traffic_alns_v2.py`入口及相同的 July 任务/成本目录，不要混合不同搜索预算或原来6月的运行结果。

## 6. 测试（无需服务器7月29天数据）

```bash
cd chicago_july_quality_tool
python test_quality_select.py
cd ..
```

合成测试覆盖正确日、重复记录、缺失网格单元、缺日期文件、冲突映射、选窗及任务文件导出。

**在本次交付环境中还使用了已上传的7月2日真实CSV做单日烟雾测试**，其有效速度数 `82401`、全路段有效率 `54.654%` 与之前检查一致。以之前冻结映射的71个监测段和507边映射计：

| 2025-07-02验证示例 | 08:00–12:00 | 17:00–21:00 |
|---|---:|---:|
| 全1047条监测段平均覆盖 | 69.45% | 66.90% |
| 71个映射走廊平均覆盖 | 88.20% | 83.74% |
| 507条映射边加权覆盖 | 90.54% | 85.12% |
| 是否满足默认阈值 | 是 | 是 |

7月2日不属于要求分析的后29天；其烟雾测试**绝不作为7月3～31日结果**。
