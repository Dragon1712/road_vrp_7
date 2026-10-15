# road_vrp_7：从7月交通CSV全新生成三个质检参考文件

## 重要结论

当前 `road_vrp_7` 仓库中只有7月2–31日30个交通CSV和质检工具，**没有能用于地图匹配的OSM GraphML**。

- `datasets/processed/chicago/segment_index.csv`：能直接由**7月CSV**重新生成。
- `datasets/processed/chicago/osm_match_loop/safe_mapping/selected_monitor_segments.csv`：需要**7月训练集的速度有效率 + 独立新OSM路网**进行有向匹配和筛选。
- `datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv`：需要独立新OSM的边ID、街道名称和有向路径，不可能仅从CSV中的起终点经纬度真实恢复。禁止编造。

本工具**不读取、不复制**旧 `road_vrp` 的道路文件和任何CSV，也没有固定71条/507边的预期数量。三个目标文件的格式兼容既有 `chicago_july_quality_tool/batch_quality_select.py`。

## 1. 安装（新仓库根目录）

将本工具包中的 `bootstrap_july.py` 和 `fix_quality_provenance.py` 放在新仓库根目录。

```bash
cd ~/code/WZL/road_vrp_7
conda activate WZLroadvrp
python -c 'import pandas,numpy,networkx;print("dependencies OK")'
```

## 2. 第一步：利用全部30天生成路段索引与训练期覆盖率

```bash
python bootstrap_july.py index \
  --input-dir chicago_data \
  --out-dir datasets/processed/chicago \
  --start 2025-07-02 \
  --end 2025-07-31 \
  --train-end 2025-07-21 \
  --hash-files
```

输出：

- `datasets/processed/chicago/segment_index.csv`
- `datasets/processed/chicago/segments_static_july.csv`
- `datasets/processed/chicago/july_daily_raw_quality.csv`
- `datasets/processed/chicago/july_metadata_conflicts.csv`
- `datasets/processed/chicago/july_index_manifest.json`

`segment_index.csv` 对所有30日的监测ID取集合并升序排列。`train_valid_speed_ratio` **仅使用7月2–21日**有效观测，以避免根据测试日的速度可用性选择新映射。`valid_speed_ratio`与输入速度真值不同：仅对速度>0且<=100mph认定有效观测。

脚本检测重复的10分钟ID单元、文件缺失、跨日记录和路段静态属性漂移。有异常会提示或拒绝；不能把原始`speed=-1`当作真实速度。`--hash-files`便于复现但需要读取全部约1GB字节。

## 3. 第二步：全新获取OSM路网（需要联网，仅首次）

**不要从旧 `road_vrp` 复制GraphML。** 在一台可以访问Overpass且安装 `osmnx>=2,<3` 的机器运行：

```bash
python -m pip install 'osmnx>=2,<3'
python bootstrap_july.py download-osm \
  --graphml datasets/raw/chicago/chicago_loop_drive_july.graphml \
  --west -87.667 --south 41.851 --east -87.592 --north 41.906
```

以上矩形是为了延续当前**Chicago Loop配送区域的研究范围**，不要求新路网与旧GraphML相同。它会生成：

- `datasets/raw/chicago/chicago_loop_drive_july.graphml`
- `datasets/raw/chicago/chicago_loop_drive_july.source.json`（OSMnx版本、bbox、边/节点数量、SHA256）

如Ubuntu计算服务器不能联网，可以先在联网电脑下载，再仅把这份**全新下载的**GraphML及其来源JSON复制到新项目相同路径：

```bash
scp datasets/raw/chicago/chicago_loop_drive_july.graphml \
    xys@<server>:/home/xys/code/WZL/road_vrp_7/datasets/raw/chicago/
scp datasets/raw/chicago/chicago_loop_drive_july.source.json \
    xys@<server>:/home/xys/code/WZL/road_vrp_7/datasets/raw/chicago/
```

## 4. 第三步：使用新路网重新匹配，生成剩余两个文件

```bash
python bootstrap_july.py map \
  --processed-dir datasets/processed/chicago \
  --graphml datasets/raw/chicago/chicago_loop_drive_july.graphml \
  --min-train-coverage 0.50 \
  --snap-max-m 70 \
  --length-ratio-min 0.80 \
  --length-ratio-max 1.25 \
  --min-street-fraction 0.75
```

输出：

- `datasets/processed/chicago/osm_match_loop/safe_mapping/selected_monitor_segments.csv`
- `datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv`
- `datasets/processed/chicago/osm_match_loop/safe_mapping/excluded_monitor_segments.csv`
- `datasets/processed/chicago/osm_match_loop/safe_mapping/safe_mapping_summary.json`
- `datasets/processed/chicago/july_osm_match_candidates.csv`（逐监测路段的接纳/拒绝理由）

匹配规则：基于**有向**网络最短路；检查起终点snap距离、方向、街道名称、长度比例；优先使用7月训练期更可靠的监测走廊；排除冲突的边分配；保存`path_fraction`且总和等于1。这里的“safe”只是自动筛选，不是经人工核实的真实地理对应。**正式ALNS前需要对保留的走廊做地图人工抽查。**

路网OSM版本变化将改变监测走廊/边数，不应假设得到旧项目的71条和507条结果。零匹配时脚本报错，须查看`july_osm_match_candidates.csv`调整地理范围或审核阈值，不应造假。

## 5. 第四步：运行原来的30天质量检查

先修正当前质检工具中的硬编码旧仓库来源文字，不改变计算逻辑：

```bash
python fix_quality_provenance.py
```

```bash
python chicago_july_quality_tool/batch_quality_select.py \
  --input-dir chicago_data \
  --segment-index datasets/processed/chicago/segment_index.csv \
  --mapped-segments datasets/processed/chicago/osm_match_loop/safe_mapping/selected_monitor_segments.csv \
  --edge-mapping datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv \
  --start 2025-07-02 --end 2025-07-31 \
  --departures 08:00,17:00 \
  --horizon-hours 4 --buffer-hours 1 \
  --selection-split all --hash-files \
  --output-dir runs/chicago_july_v1/data_qc \
  --fail-on-missing --fail-on-structural
```

## 6. 运行后应检查的验收项目

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
p=Path('datasets/processed/chicago')
i=pd.read_csv(p/'segment_index.csv')
s=pd.read_csv(p/'osm_match_loop/safe_mapping/selected_monitor_segments.csv')
e=pd.read_csv(p/'osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv',dtype={'osm_u':str,'osm_v':str,'osm_key':str})
assert i.segment_id.is_unique
assert s.segment_id.is_unique
assert e.segment_id.isin(s.segment_id).all()
assert not e.duplicated(['osm_u','osm_v','osm_key']).any()
assert (e.groupby('segment_id').path_fraction.sum()-1).abs().max()<1e-8
print(f'PASS: {len(i)} unique monitor IDs; {len(s)} new safely matched segments; {len(e)} new unique directed OSM edges')
PY
```

如果新匹配得到监测路段数为零，匹配脚本会报错。**不能仅因通过自动筛选就把OSM映射称作人工核验真值。**

## 7. 包内验证情况及限制

- 已使用**真实上传的2025-07-02 CSV**独立生成过1047条路段索引（`sample_check_july02/segment_index.csv`），并核验总记录150768、有效速度82401、144时段。
- 已用**合成有向GraphML**测试新的OSM匹配、重复边冲突和方向错误拒绝。
- 当前执行环境不能访问Overpass，且新仓库没有GraphML，因此尚不能在此生成真实的另外两个映射文件；需要按第3–4步获取全新OSM并在你的服务器完成。
- 样本索引来自7月2日**单日**，只供核对，不应跳过第2步的30天重算。
- 7月2–21训练、7月22–25验证、7月26–31测试为当前建议；新建地图映射只使用训练期有效率做评分，测试日不可用于超参数调整。
