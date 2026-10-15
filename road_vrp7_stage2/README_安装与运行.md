# road_vrp_7 · 第二阶段：7月交通张量与FIFO道路成本（独立离线版）

适用于新仓库 `https://github.com/Dragon1712/road_vrp_7`，已核对2026-10-15的仓库输出：30天（2025-07-02～07-31）、1047监测路段、68条重新匹配的走廊、510条唯一映射OSM边。本工具**不会复制或读取6月交通数据、旧路网成本或旧OD缓存**；它只使用新仓库已有的7月CSV、7月路段索引、重新生成的映射及当前GraphML。

> **限制**：GitHub新仓库中的GraphML字节内容与此前仓库相同。如果你要求道路图也独立重新下载，需要先替换GraphML并重跑`bootstrap_july.py map`生成与新GraphML一致的匹配。新仓库中的`safe_mapping_summary.json`保存了GraphML SHA256，构建脚本会对不一致文件报错，而不是静默匹配。

## 1. 放置代码及依赖

在服务器的 `~/code/WZL/road_vrp_7` 根目录下，解压工具包，应出现：

```
road_vrp7_stage2/
  build_july_stage2.py
  run_stage2.sh
  README_安装与运行.md
  tests/test_stage2.py
```

Python 3.10+；仅依赖numpy、pandas、networkx，构建无需网络/OSMnx/官方ALNS包：

```bash
cd ~/code/WZL/road_vrp_7
python -c 'import numpy,pandas,networkx; print(numpy.__version__,pandas.__version__,networkx.__version__)'
bash road_vrp7_stage2/run_stage2.sh test
```

预期的输入路径（均存在于新仓库）：

```
chicago_data/traffic_2025-07-02.csv ... traffic_2025-07-31.csv
 datasets/processed/chicago/segment_index.csv
 datasets/processed/chicago/segments_static_july.csv
 datasets/processed/chicago/osm_match_loop/safe_mapping/selected_monitor_segments.csv
 datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv
 datasets/processed/chicago/osm_match_loop/safe_mapping/safe_mapping_summary.json
 datasets/raw/chicago/chicago_loop_drive.graphml
```

## 2. 生成7月30天专属数据

```bash
bash road_vrp7_stage2/run_stage2.sh build
```

若想逐项指定参数：

```bash
python road_vrp7_stage2/build_july_stage2.py build \
  --csv-dir chicago_data \
  --segment-index datasets/processed/chicago/segment_index.csv \
  --static-segments datasets/processed/chicago/segments_static_july.csv \
  --mapping datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv \
  --selected-segments datasets/processed/chicago/osm_match_loop/safe_mapping/selected_monitor_segments.csv \
  --mapping-summary datasets/processed/chicago/osm_match_loop/safe_mapping/safe_mapping_summary.json \
  --graphml datasets/raw/chicago/chicago_loop_drive.graphml \
  --output-dir datasets/processed/chicago/july_v1 \
  --start 2025-07-02 --end 2025-07-31 \
  --hash-files
```

该步骤要逐日读入30个~34MB CSV，处理4,523,040个路段-时间单元，并写出一份多百万行的`traffic_long.csv`；速度取决于服务器硬盘和CPU。可能占用数百MB至1GB磁盘空间及数百MB RAM。运行前建议检查空闲磁盘。**不覆盖已有非空输出目录**；如中断出现`BUILD_FAILED.txt`，检查错误后删除不完整的输出目录或改用新的版本目录，禁止与此前结果混合。

## 3. 输出说明及严格语义

```
 datasets/processed/chicago/july_v1/
 ├── traffic_tensor.npz           # speed_kph[T=4320,S=1047] + observed_mask + segment_ids + times
 ├── traffic_long.csv             # slot,segment_id,speed_kph,segment_travel_sec,observed
 ├── segment_index.csv            # 与外部7月路段索引相同的列顺序
 ├── time_index.csv               # 4320个连续、无时区本地时间片
 ├── july_traffic_daily_quality.csv
 ├── stage2_manifest.json         # 参考文件与原始文件SHA256及生成配置
 ├── stage2_validation.json       # 独立验证阶段生成
 ├── tdsp_smoke.csv / tdsp_smoke_summary.json
 └── dynamic_costs/
      ├── edge_index.csv          # stable edge_idx + directed osm_u/v/key + static travel estimates
      ├── time_index.csv          # 与交通tensor相同的日期/时间
      ├── edge_costs.npz          # travel_sec_raw,travel_sec_fifo,chicago_observed_mask,is_mapped_edge
      ├── observed_edge_coverage_by_slot.csv
      ├── observed_edge_coverage_by_day.csv
      └── dynamic_cost_summary.json
```

- 单位：原CSV `speed` 为 **mph**，段长`length_m`为**米**。监测走廊耗时为`3.6*length_m/speed_kph`秒；按当前`path_fraction`分摊到与其匹配的OSM边。
- `speed<=0`、非数值或超过100mph的位置一律视为缺失，输出NaN和`observed_mask=False`；不是零速度，更不是可监督训练标签。
- `chicago_observed_mask=True`仅说明某条OSM边受该时刻真实监测走廊速度**支持**；走廊时间按长度分配到边，非逐条OSM边直接测量。未映射OSM边不标为观测。
- 缺失或未映射的道路成本使用静态估计值；该数值不是交通真值，也不是模型补全数据。
- 生成`travel_sec_fifo`时使用采样到达函数的单调包络；**可能修改一些原观测旅行时间**，所以专门保留`travel_sec_raw`作为原始融合/回退状态，FIFO修正成本不能训练为“原实测速度”。
- `time_local`按项目当前惯例处理为**无时区的本地时间**，未来若源文件提供UTC时区声明，应先统一校准，不可混合UTC/本地时钟。
- 训练/验证/测试划分固定：07-02～07-21 / 07-22～07-25 / 07-26～07-31；这一步构建的是全部历史交通的 **ex-post评价数据**，**不是交通预测模型**。推理时不可预知未来完整CSV。

## 4. 完整验证

```bash
bash road_vrp7_stage2/run_stage2.sh validate
bash road_vrp7_stage2/run_stage2.sh tdsp
```

`validate`逐项检查：时间轴4320个等间隔时间片、1047路段ID对齐、速度与原始观测掩码一致、OSM边数量与`edge_idx`连续性、510条映射边的观测位、未映射边的静态回退、走廊旅行时间在路径边上的总和、FIFO到达时间非递减。输出 `stage2_validation.json`，必须有`"validation": "PASS"`。

`tdsp`默认对07-28 08:00和17:00各随机30对强连通有向OD比较静态路径的时变重放成本与历史Oracle TDSP成本。Oracle必须不高于同一历史道路成本上的静态路径重放，否则立即报错。该检查只是路由与数据流水线回归测试，不是VRP实验，也不是部署性能证明。

注意 `tdsp` 第二次运行会覆盖`tdsp_smoke.csv`/`tdsp_smoke_summary.json`的第一次结果；**正式复现需各时段分别存档**（见下面可复现命令）：

```bash
python road_vrp7_stage2/build_july_stage2.py tdsp-smoke --departure '2025-07-28 08:00:00'
cp datasets/processed/chicago/july_v1/tdsp_smoke_summary.json datasets/processed/chicago/july_v1/tdsp_smoke_0800_summary.json
cp datasets/processed/chicago/july_v1/tdsp_smoke.csv datasets/processed/chicago/july_v1/tdsp_smoke_0800.csv
python road_vrp7_stage2/build_july_stage2.py tdsp-smoke --departure '2025-07-28 17:00:00'
cp datasets/processed/chicago/july_v1/tdsp_smoke_summary.json datasets/processed/chicago/july_v1/tdsp_smoke_1700_summary.json
cp datasets/processed/chicago/july_v1/tdsp_smoke.csv datasets/processed/chicago/july_v1/tdsp_smoke_1700.csv
```

## 5. 验收值与后续ALNS对接

正常情况下应观察到以下尺寸（**必须以你服务器实际生成的结果验证**）：

| 参数 | 期望值/检查 | 说明 |
|---|---:|---|
| 监测路段S | 1047 | 新仓库路段索引 |
| 时间片T | 4320 | 30天，每天144个 |
| 观测有效率 | 约50.57% | 以7月质检结果为参考，若不一致检查阈值 |
| 有向边E | 以新GraphML实际结果为准 | 不能强制旧数据5503条 |
| 映射道路边 | 510 | 新7月映射摘要，若变动应排查版本 |
| 监测走廊 | 68 | 新7月映射摘要 |
| 输出质量状态 | PASS | 逐边、逐时间检查 |
| TDSP检查 | PASS | 30对OD×2时段 |

后续的官方ALNS/Traffic-ALNS应读取 **`datasets/processed/chicago/july_v1/dynamic_costs/`**，**不能继续使用旧6月的`dynamic_costs`路径或OD缓存**。新建`runs/chicago_july_v1/od_cache/`，并基于冻结的训练/测试日期生成独立CVRP任务。应先得到新官方ALNS的基准表现，再开展Traffic-ALNS v2中的AT-RR/TE-WR/Section算子消融。

7月27日两时段当前未达到“全市观测率”门槛，但目标68条映射走廊并未完全失去观测，建议单独作为低覆盖压力测试；最终测试不应只看`selected_scenarios.csv`里全部来自训练日期的6个高质量情景。

## 6. 本工具测试范围

- 合成8条有向边、3条监测走廊、2天144×2时间片，测试原始缺失处理、边成本映射、时间守恒、FIFO和时变Dijkstra。
- 真实已上传的`traffic_2025-07-02.csv`单日时间网格读取测试：150,768条记录，82,401个有效速度位置，与第一阶段质量报告一致。
- **尚未在当前环境实际运行整个7月30天以及真实GraphML的完整构建**，因为GitHub中的30个大文件没有全部挂载到当前执行环境。请在你的Ubuntu服务器执行上述命令并同步输出。
