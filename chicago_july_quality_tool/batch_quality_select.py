#!/usr/bin/env python3
"""Audit Chicago Traffic Tracker CSVs and preselect July ALNS validation scenarios.

Input speed is MPH; nonpositive/implausible readings are MISSING, never zero.
All date/time values remain *source-local naive* as in the existing road_vrp pipeline.
The selected scenarios are an observational-quality shortlist, not ground truth
and not a trained solver. Existing frozen OSM mapping is used only for assessment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"time", "segment_id", "speed"}
META = ("street", "direction", "length", "start_latitude", "start_longitude",
        "end_latitude", "end_longitude")
MINUTES_PER_SLOT = 10
SLOTS_PER_DAY = 144


def set_name(name: str) -> str:
    return name.split(".")[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_ids(path: Path, col="segment_id") -> list[int]:
    f = pd.read_csv(path, usecols=[col])
    raw = pd.to_numeric(f[col], errors="coerce")
    if raw.isna().any() or (raw % 1 != 0).any():
        raise ValueError(f"Invalid segment_id in {path}")
    ids = [int(s) for s in raw]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicated {col} in {path}")
    return ids


def read_mapping(path: Path | None, mapped_ids: list[int]) -> dict[int, int]:
    if path is None:
        return {x: 1 for x in mapped_ids}
    x = pd.read_csv(path, usecols=["segment_id", "osm_u", "osm_v", "osm_key"],
                    dtype={"osm_u": str, "osm_v": str, "osm_key": str})
    x["segment_id"] = pd.to_numeric(x.segment_id, errors="raise").astype(int)
    key_cols = ["osm_u", "osm_v", "osm_key"]
    x = x.drop_duplicates(["segment_id"] + key_cols)
    if x.duplicated(key_cols).any():
        raise ValueError("OSM edge assigned to multiple monitored segments; mapping is ambiguous")
    counts = x.groupby("segment_id").size().to_dict()
    absent = set(mapped_ids) - set(counts)
    if absent:
        raise ValueError(f"Selected monitored segments absent from edge map: {sorted(absent)}")
    return {i: int(counts[i]) for i in mapped_ids}


def parse_departures(s: str) -> list[dt.time]:
    out = []
    for part in s.split(","):
        part = part.strip()
        try:
            t = dt.datetime.strptime(part, "%H:%M").time()
        except ValueError as exc:
            raise ValueError(f"Invalid departure '{part}', expected HH:MM") from exc
        if t.minute % MINUTES_PER_SLOT:
            raise ValueError("Departure time must align with a 10-minute slot")
        if t not in out:
            out.append(t)
    if not out:
        raise ValueError("At least one departure is required")
    return out


def partition(date: dt.date, train_end: dt.date, val_end: dt.date) -> str:
    if date <= train_end:
        return "train"
    if date <= val_end:
        return "validation"
    return "test"


def stretch_true(values: np.ndarray) -> int:
    best = current = 0
    for x in values:
        current = current + 1 if bool(x) else 0
        best = max(best, current)
    return best


def qtile(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if x.size else 0.0


def stats_binned(arr: np.ndarray, t0: int, t1: int) -> dict:
    x = np.asarray(arr[t0:t1], dtype=float)
    return {
        "mean": float(np.mean(x)) if x.size else 0.0,
        "p10": qtile(x, .1),
        "min": float(np.min(x)) if x.size else 0.0,
        "zero_slots": int(np.count_nonzero(x == 0)),
        "longest_zero_streak": stretch_true(x == 0),
        "longest_below_half_streak": stretch_true(x < .5),
    }


def analyze_one(path: Path, date: dt.date, expected_ids: list[int],
                mapped_ids: list[int], edge_weights: dict[int, int],
                args: argparse.Namespace, reference_meta: dict | None):
    """Returns (daily, profiles, meta_sample, optional_error).

    Profiles contain 144-slot arrays; never hold multiple days' raw frames.
    """
    day = {
        "date": date.isoformat(), "weekday": date.strftime("%A"),
        "weekend": bool(date.weekday() >= 5), "split": partition(date,args.train_end,args.val_end),
        "file": str(path), "exists": path.is_file(), "file_bytes": path.stat().st_size if path.is_file() else 0,
    }
    if not path.is_file():
        day["fatal_reasons"] = "missing_file"
        return day, None, None
    if args.hash_files:
        day["sha256"] = sha256(path)
    try:
        headers = list(pd.read_csv(path, nrows=0).columns)
        absent = REQUIRED - set(headers)
        if absent:
            raise ValueError(f"Missing required columns: {sorted(absent)}")
        fields = list(REQUIRED) + [k for k in ("bus_count", "message_count", "record_id") + META if k in headers]
        df = pd.read_csv(path, usecols=fields, low_memory=False)
    except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as exc:
        day["fatal_reasons"] = "csv_read_or_schema_error"
        day["read_error"] = str(exc)
        return day, None, None
    day["raw_rows"] = int(len(df))
    day["missing_optional_columns"] = ",".join(k for k in META if k not in headers)
    parsed = pd.to_datetime(df["time"], errors="coerce")
    ids_numeric = pd.to_numeric(df["segment_id"], errors="coerce")
    id_valid = ids_numeric.notna() & np.isfinite(ids_numeric) & (ids_numeric % 1 == 0)
    speed = pd.to_numeric(df["speed"], errors="coerce")
    day["unparseable_times"] = int(parsed.isna().sum())
    day["invalid_segment_ids"] = int((~id_valid).sum())
    day["speed_non_numeric"] = int(speed.isna().sum())
    day["speed_nonpositive"] = int((speed <= 0).sum())
    day["speed_gt_max_mph"] = int((speed > args.max_mph).sum())
    day["valid_speed_rows"] = int(((speed > 0) & (speed <= args.max_mph)).sum())
    day["duplicate_record_ids"] = 0
    if "record_id" in df:
        day["duplicate_record_ids"] = int(df["record_id"].duplicated(keep="first").sum())
    otherwise = id_valid & parsed.notna()
    df = df.loc[otherwise].copy()
    df["_time"] = parsed[otherwise]
    df["segment_id"] = ids_numeric[otherwise].astype(np.int64)
    df["speed"] = speed[otherwise].astype(float)
    df["_slot"] = df["_time"].dt.floor(f"{MINUTES_PER_SLOT}min")
    same_day = (df["_slot"].dt.date == date)
    day["out_of_day_rows"] = int((~same_day).sum())
    df = df.loc[same_day].copy()
    observed_segments = set(df.segment_id.unique().tolist())
    known = set(expected_ids)
    day["unknown_segments"] = int(len(observed_segments - known))
    day["unknown_segment_rows"] = int((~df.segment_id.isin(known)).sum())
    day["missing_reference_segments"] = int(len(known - observed_segments))
    df = df.loc[df.segment_id.isin(known)].copy()
    day["duplicate_segment_slots"] = int(df.duplicated(["_slot", "segment_id"]).sum())
    df = df.sort_values(["_slot", "segment_id", "_time"], kind="mergesort")
    df = df.drop_duplicates(["_slot", "segment_id"], keep="last")
    day["unique_segment_slot_cells"] = int(len(df))
    day["observed_slots"] = int(df["_slot"].nunique())
    day["missing_slots"] = SLOTS_PER_DAY - day["observed_slots"]
    day["missing_grid_cells"] = SLOTS_PER_DAY * len(expected_ids) - len(df)
    day["grid_cell_fraction"] = len(df) / max(SLOTS_PER_DAY * len(expected_ids), 1)
    expected_clock = pd.date_range(pd.Timestamp(date), periods=SLOTS_PER_DAY,
                                   freq=f"{MINUTES_PER_SLOT}min")
    # Preserve the supplied segment-id order for exact alignment with the repo.
    aligned = df.pivot(index="_slot", columns="segment_id", values="speed")
    aligned = aligned.reindex(index=expected_clock, columns=expected_ids)
    values = aligned.to_numpy(float)
    obs = np.isfinite(values) & (values > 0) & (values <= args.max_mph)
    values[~obs] = np.nan
    day["valid_cells"] = int(obs.sum())
    day["full_observed_fraction"] = float(obs.mean())
    indexer = {k: i for i, k in enumerate(expected_ids)}
    ids = [indexer[k] for k in mapped_ids]
    mapped = obs[:, ids]
    mapped_vals = values[:, ids]
    weights = np.asarray([edge_weights[k] for k in mapped_ids], dtype=float)
    slot_full = obs.mean(axis=1)
    slot_mapped = mapped.mean(axis=1)
    slot_edges = (mapped @ weights) / float(weights.sum())
    day["mapped_segments"] = len(mapped_ids)
    day["mapped_edges"] = int(weights.sum())
    day["mapped_day_observed_fraction"] = float(mapped.mean())
    day["mapped_edge_day_observed_fraction"] = float(slot_edges.mean())
    day["full_day_zero_slots"] = int((slot_full == 0).sum())
    day["mapped_day_zero_slots"] = int((slot_mapped == 0).sum())
    day["mapped_edge_day_zero_slots"] = int((slot_edges == 0).sum())
    # Optional metadata integrity, with same-segment within-day and across-day drift.
    meta_sample = None
    present = [k for k in META if k in df]
    day["metadata_nonconstant_segment_fields"] = 0
    day["metadata_conflicting_segment_count"] = 0
    day["metadata_crossday_conflicting_segment_count"] = 0
    if present:
        uniques = df.groupby("segment_id")[present].nunique(dropna=False)
        bad = uniques.gt(1)
        day["metadata_nonconstant_segment_fields"] = int(bad.to_numpy().sum())
        day["metadata_conflicting_segment_count"] = int(bad.any(axis=1).sum())
        meta_sample = df.groupby("segment_id")[present].first()
        if reference_meta is not None and set(present) == set(reference_meta.columns):
            common = meta_sample.index.intersection(reference_meta.index)
            old = reference_meta.loc[common]
            new = meta_sample.loc[common]
            changed = pd.DataFrame(False, index=common, columns=present)
            for key in present:
                if key in ("length", "start_latitude", "start_longitude", "end_latitude", "end_longitude"):
                    x = pd.to_numeric(old[key], errors="coerce")
                    y = pd.to_numeric(new[key], errors="coerce")
                    changed[key] = ~np.isclose(x, y, atol=1e-6, rtol=1e-6, equal_nan=True)
                else:
                    changed[key] = old[key].fillna("<NA>").astype(str) != new[key].fillna("<NA>").astype(str)
            day["metadata_crossday_conflicting_segment_count"] = int(changed.any(axis=1).sum())
    hard = []
    for name in ("unparseable_times", "invalid_segment_ids", "unknown_segment_rows",
                 "out_of_day_rows", "duplicate_segment_slots", "duplicate_record_ids", "missing_slots",
                 "missing_grid_cells", "missing_reference_segments",
                 "metadata_conflicting_segment_count", "metadata_crossday_conflicting_segment_count"):
        if day[name]: hard.append(name)
    if day["missing_optional_columns"]: hard.append("missing_metadata_columns")
    day["structural_ok"] = not hard
    day["fatal_reasons"] = ";".join(hard)
    rows = []
    duration = int(args.horizon_hours * 60 / MINUTES_PER_SLOT)
    buffer = int(args.buffer_hours * 60 / MINUTES_PER_SLOT)
    for departure in args.departures:
        start = departure.hour * (60 // MINUTES_PER_SLOT) + departure.minute // MINUTES_PER_SLOT
        end = start + duration
        safe_end = end + buffer
        row = {
            "date": date.isoformat(), "split": day["split"], "weekday": day["weekday"],
            "weekend": day["weekend"], "departure": departure.strftime("%H:%M"),
            "scenario_id": date.strftime("%Y-%m-%d") + "_" + departure.strftime("%H%M"),
            "departure_local": date.strftime("%Y-%m-%d") + " " + departure.strftime("%H:%M:%S"),
            "source_file": path.name, "expected_horizon_slots": duration,
            "expected_buffer_slots": buffer,
            "structural_ok": day["structural_ok"],
            "day_full_valid": day["full_observed_fraction"],
        }
        if safe_end > SLOTS_PER_DAY:
            row.update({"eligible": False, "rejection_reasons": "window_or_buffer_spans_next_day",
                        "quality_score": float("nan")})
            rows.append(row)
            continue
        f = stats_binned(slot_full, start, end)
        m = stats_binned(slot_mapped, start, end)
        e = stats_binned(slot_edges, start, end)
        be = stats_binned(slot_edges, end, safe_end) if buffer else {"mean":1.0,"p10":1.0,"min":1.0,"zero_slots":0,"longest_zero_streak":0,"longest_below_half_streak":0}
        for key, s in (("full", f), ("mapped", m), ("edge", e), ("buffer_edge", be)):
            for stat, value in s.items(): row[f"{key}_{stat}"] = value
        segment_fraction = mapped[start:end].mean(axis=0)
        row["mapped_segments_half_observed_fraction"] = float(np.mean(segment_fraction >= .5))
        row["mapped_segments_zero_observed_count"] = int(np.count_nonzero(segment_fraction == 0))
        row["edge_hourly_min"] = float(min(np.mean(slot_edges[i:min(i + 6,end)])
                                          for i in range(start, end, 6)))
        row["edge_low_half_slots"] = int(np.count_nonzero(slot_edges[start:end] < .5))
        # Indicative, not ground truth: variability on actually observed
        # segments. A segment with <3 observed slots is not scored.
        finite = []
        for j in range(len(mapped_ids)):
            v = mapped_vals[start:end, j]
            v = v[np.isfinite(v)]
            if len(v) >= 3:
                finite.append(float(np.std(v) / max(np.mean(v), 1.)))
        row["traffic_temporal_cv_median"] = float(np.median(finite)) if finite else float("nan")
        row["traffic_cv_scored_segments"] = len(finite)
        reasons = []
        if not day["structural_ok"]: reasons.append("structural_quality_failure")
        if day["full_observed_fraction"] < args.min_day_full:
            reasons.append("low_day_full_coverage")
        if f["mean"] < args.min_window_full: reasons.append("low_full_window_coverage")
        if m["mean"] < args.min_window_mapped: reasons.append("low_mapped_window_coverage")
        if e["mean"] < args.min_window_edge: reasons.append("low_mapped_edge_window_coverage")
        if e["p10"] < args.min_window_edge_p10: reasons.append("poor_edge_p10")
        if row["edge_hourly_min"] < args.min_hour_edge: reasons.append("poor_hour_edge_coverage")
        if be["mean"] < args.min_buffer_edge: reasons.append("poor_buffer_coverage")
        if e["zero_slots"] > args.max_edge_zero_slots: reasons.append("zero_coverage_slots")
        if row["mapped_segments_half_observed_fraction"] < args.min_segment_active:
            reasons.append("too_many_weakly_observed_mapped_segments")
        row["quality_score"] = float(.4*e["mean"]+.2*e["p10"]+.15*m["mean"]+.15*f["mean"]+.1*be["mean"])
        row["eligible"] = len(reasons) == 0
        row["rejection_reasons"] = ";".join(reasons)
        rows.append(row)
    return day, rows, meta_sample


def shortlist(scenarios: pd.DataFrame, args) -> pd.DataFrame:
    valid = scenarios.loc[scenarios.eligible.fillna(False)].copy()
    if args.selection_split != "all":
        valid = valid.loc[valid.split == args.selection_split].copy()
    choices = []
    for dep in args.departures:
        clock = dep.strftime("%H:%M")
        pool = valid.loc[valid.departure == clock].copy()
        pool = pool.sort_values(["quality_score", "date"], ascending=[False, True])
        if pool.empty: continue
        weekend = pool.loc[pool.weekend.astype(bool)].head(min(args.weekend_quota, args.max_per_departure))
        chosen = list(weekend.index)
        rest = pool.loc[~pool.index.isin(chosen)].head(args.max_per_departure-len(chosen))
        chosen.extend(rest.index.tolist())
        choices.extend(chosen)
    out = valid.loc[sorted(set(choices))].copy()
    return out.sort_values(["date", "departure"])


def make_report(args, days: pd.DataFrame, scenarios: pd.DataFrame, selected: pd.DataFrame,
                out: Path, missed: list[str]):
    n_files = int(days.exists.sum())
    tested = days.loc[days.exists.astype(bool)]
    lines = ["# Chicago July 2025 交通数据质量筛选", "",
             f"- 检查日期：{args.start} 至 {args.end}（{len(days)} 天）",
             f"- 实际找到：{n_files} 个CSV；缺失：{len(missed)} 个（{', '.join(missed) or '无'}）",
             f"- ALNS目标：{args.selection_split} 分区，{','.join(t.strftime('%H:%M') for t in args.departures)}，"
             f"每次规划{args.horizon_hours:g}小时，附加{args.buffer_hours:g}小时缓冲",
             f"- 已通过质量门槛的日期-时段：{int(scenarios.eligible.fillna(False).sum())} 个；"
             f"推荐：{len(selected)} 个。",
             "- 注：质量筛选不代表速度已补全，不代表所有OSM边有实测，亦不证明ALNS性能。",
             "- 选择结果在求解前冻结；不要根据最终ALNS性能再挑选日期。",
             "- train/validation/test 按日期切分；验证或测试日期不能用于未来交通预测器的训练。",
             "- 交通数据按原始无时区本地时间解释；跨日边界/夏令时规则需单独核对。",
             ""]
    if not tested.empty:
        lines.extend(["## 整体观测质量", "",
                      f"- 观测完整率（日均）：{tested.full_observed_fraction.mean():.2%}",
                      f"- 映射走廊覆盖（日均）：{tested.mapped_day_observed_fraction.mean():.2%}",
                      f"- OSM映射边加权覆盖（日均）：{tested.mapped_edge_day_observed_fraction.mean():.2%}",
                      f"- 结构无异常的文件：{int(tested.structural_ok.sum())}/{len(tested)}", ""])
    lines.extend(["## 推荐的日期和时段", "",
                  "| 日期 | 星期 | 划分 | 出发 | OSM映射边窗口平均覆盖 | 映射边P10覆盖 | 评分 |",
                  "|---|---|---|---|---:|---:|---:|"])
    if len(selected):
        for r in selected.itertuples():
            lines.append(f"| {r.date} | {r.weekday} | {r.split} | {r.departure} | {r.edge_mean:.1%} | {r.edge_p10:.1%} | {r.quality_score:.3f} |")
    else:
        lines.append("| 无合格候选 | | | | | | |")
    lines.extend(["", "## 主要数据输出", "",
                  "- `daily_quality.csv`：所有目标日期的结构和每日覆盖情况",
                  "- `scenario_quality.csv`：每个出发时间的四小时窗口及缓冲质量",
                  "- `eligible_scenarios.csv`：通过各项质量门槛的全部时间窗",
                  "- `rejected_scenarios.csv`：不合格时间窗与拒绝原因",
                  "- `selected_scenarios.csv` / `.json`：推荐供ALNS初步验证的时间窗",
                  "- `audit_manifest.json`：阈值、数据来源、时间划分、版本及可选SHA256",
                  "", "## 重要局限", "",
                  "1. 这是基于原始速度的观测可用性筛选，而非交通预测精度验证。",
                  "2. 71个历史映射走廊的几何/方向匹配未被独立认证。",
                  "3. 没有实测映射的OSM边仍需要静态回退或预测；两者不能称作实测真值。",
                  "4. `selected_scenarios.json` 只有日期，需重新预处理7月的动态边成本并制作7月任务情景后才可运行ALNS。",
                  "5. 脚本根据交通观测质量选择场景，不应按未来ALNS优化收益选场景以避免选择偏差。",
                  ""])
    (out/"质量报告.md").write_text("\n".join(lines),encoding="utf-8")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-dir",type=Path,default=Path("chicago_data"))
    p.add_argument("--segment-index",type=Path,default=Path("datasets/processed/chicago/segment_index.csv"))
    p.add_argument("--mapped-segments",type=Path,default=Path("datasets/processed/chicago/osm_match_loop/safe_mapping/selected_monitor_segments.csv"))
    p.add_argument("--edge-mapping",type=Path,default=Path("datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("runs/chicago_data_qc/july_2025"))
    p.add_argument("--start",type=dt.date.fromisoformat,default=dt.date(2025,7,3))
    p.add_argument("--end",type=dt.date.fromisoformat,default=dt.date(2025,7,31))
    p.add_argument("--departures",default="08:00,17:00")
    p.add_argument("--horizon-hours",type=float,default=4.)
    p.add_argument("--buffer-hours",type=float,default=1.)
    p.add_argument("--max-mph",type=float,default=100.)
    p.add_argument("--train-end",type=dt.date.fromisoformat,default=dt.date(2025,7,21))
    p.add_argument("--val-end",type=dt.date.fromisoformat,default=dt.date(2025,7,25))
    p.add_argument("--selection-split",choices=["all","train","validation","test"],default="test")
    p.add_argument("--max-per-departure",type=int,default=3)
    p.add_argument("--weekend-quota",type=int,default=1)
    p.add_argument("--min-day-full",type=float,default=.40)
    p.add_argument("--min-window-full",type=float,default=.50)
    p.add_argument("--min-window-mapped",type=float,default=.65)
    p.add_argument("--min-window-edge",type=float,default=.65)
    p.add_argument("--min-window-edge-p10",type=float,default=.45)
    p.add_argument("--min-hour-edge",type=float,default=.55)
    p.add_argument("--min-buffer-edge",type=float,default=.40)
    p.add_argument("--min-segment-active",type=float,default=.65)
    p.add_argument("--max-edge-zero-slots",type=int,default=0)
    p.add_argument("--hash-files",action="store_true",help="Hash full ~1GB inputs for stronger reproducibility")
    p.add_argument("--fail-on-missing",action="store_true",help="Exit nonzero after writing reports if any date file missing")
    p.add_argument("--fail-on-empty",action="store_true",help="Exit nonzero if selected_scenarios is empty")
    p.add_argument("--fail-on-structural",action="store_true",help="Exit nonzero after reports if any present CSV has structural quality defects")
    args=p.parse_args(argv)
    if args.start > args.end or not args.train_end < args.val_end:
        raise ValueError("Expected start <= end and train_end < val_end")
    if args.horizon_hours<=0 or args.buffer_hours < 0:
        raise ValueError("Invalid horizon/buffer")
    for h in (args.horizon_hours,args.buffer_hours):
        if round(h*60/MINUTES_PER_SLOT) != h*60/MINUTES_PER_SLOT:
            raise ValueError("Horizon and buffer must be multiples of 10 minutes")
    if args.max_per_departure<=0 or args.weekend_quota<0:
        raise ValueError("Invalid shortlist limits")
    for k in ("min_day_full","min_window_full","min_window_mapped","min_window_edge",
              "min_window_edge_p10","min_hour_edge","min_buffer_edge","min_segment_active"):
        if not 0 <= getattr(args,k) <= 1:raise ValueError(f"{k} must be in [0,1]")
    args.departures=parse_departures(args.departures)
    idx = read_ids(args.segment_index)
    mapped=read_ids(args.mapped_segments)
    if not set(mapped).issubset(idx):
        raise ValueError("Previously mapped segment IDs not a subset of repo segment_index.csv")
    # --edge-mapping may be omitted using an explicit empty string, for
    # diagnostic purposes only; real ALNS selection should use edge weights.
    weights=read_mapping(args.edge_mapping, mapped)
    all_days=pd.date_range(args.start,args.end,freq="D")
    daily=[];windows=[];missing=[];reference_meta=None
    for ix,date_ts in enumerate(all_days):
        date=date_ts.date()
        path=args.input_dir/f"traffic_{date.isoformat()}.csv"
        data,rows,meta=analyze_one(path,date,idx,mapped,weights,args,reference_meta)
        daily.append(data)
        if reference_meta is None and meta is not None:
            reference_meta=meta.copy()
        if rows:
            windows.extend(rows)
        else:
            for dep in args.departures:
                windows.append({"date":date.isoformat(),"split":partition(date,args.train_end,args.val_end),
                                "weekend":date.weekday()>=5,"weekday":date.strftime("%A"),
                                "departure":dep.strftime("%H:%M"),
                                "scenario_id":date.isoformat()+"_"+dep.strftime("%H%M"),
                                "departure_local":date.isoformat()+" "+dep.strftime("%H:%M:%S"),
                                "source_file":path.name,"eligible":False,"quality_score":float("nan"),
                                "rejection_reasons":data.get("fatal_reasons") or "no_usable_data"})
        if not data["exists"]:missing.append(date.isoformat())
        status="GOOD" if data.get("structural_ok") else (data.get("fatal_reasons") or "missing")
        print(f"[{ix+1:02d}/{len(all_days)}] {date}: "
              f"records={data.get('raw_rows',0):>6} "
              f"full_valid={data.get('full_observed_fraction',float('nan')):.1%} "
              f"mapped_edge={data.get('mapped_edge_day_observed_fraction',float('nan')):.1%} "
              f"{status}",flush=True)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    days=pd.DataFrame(daily)
    scenarios=pd.DataFrame(windows)
    if "eligible" in scenarios:
        scenarios["eligible"] = scenarios["eligible"].fillna(False).astype(bool)
    selected=shortlist(scenarios,args)
    days.to_csv(out/"daily_quality.csv",index=False)
    scenarios.to_csv(out/"scenario_quality.csv",index=False)
    scenarios.loc[scenarios.eligible].to_csv(out/"eligible_scenarios.csv",index=False)
    scenarios.loc[~scenarios.eligible].to_csv(out/"rejected_scenarios.csv",index=False)
    selected.to_csv(out/"selected_scenarios.csv",index=False)
    payload=[]
    for r in selected.itertuples():
        payload.append({"scenario_id":r.scenario_id,"departure_local":r.departure_local,
                        "date":r.date,"split":r.split,"source_file":r.source_file,
                        "quality_score":float(r.quality_score),
                        "mapped_edge_coverage":float(r.edge_mean)})
    (out/"selected_scenarios.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    config={k:((str(v) if isinstance(v,(Path,dt.date)) else [t.strftime('%H:%M') for t in v])
                if k=="departures" else (str(v) if isinstance(v,(Path,dt.date)) else v)) for k,v in vars(args).items()}
    manifest={"input_git_repo":"Dragon1712/road_vrp_7","reference_git_commit":None,"mapping_origin":"independent_July_CSVs_and_fresh_OSM",
              "script_version":"1.0.0", "configuration":config,"expected_dates":len(all_days),
              "files_found":int(days.exists.sum()),"missing_file_dates":missing,
              "selected_count":len(payload),"eligible_scenario_count":int(scenarios.eligible.sum()),
              "mapped_monitor_count":len(mapped),"mapped_edge_count":int(sum(weights.values())),
              "reference_files":{name:{"path":str(path),"sha256":sha256(path)} for name,path in
                                  (("segment_index",args.segment_index),("mapped_segments",args.mapped_segments),
                                   ("safe_osm_edge_mapping",args.edge_mapping))},
              "script_sha256":sha256(Path(__file__)),
              "inputs":[{"date":r["date"],"path":r["file"],"bytes":int(r["file_bytes"]),
                         **({"sha256":r["sha256"]} if "sha256" in r and isinstance(r.get("sha256"),str) else {})}
                        for r in daily],
              "data_labels":"Historical observed speeds only; invalid remain missing; never call imputed or predicted labels measured"}
    (out/"audit_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    make_report(args,days,scenarios,selected,out,missing)
    print("\nAUDIT SUMMARY")
    print(f"Input files: {int(days.exists.sum())}/{len(days)} | eligible windows: {int(scenarios.eligible.sum())} | selected: {len(selected)}")
    if len(selected):
        print(selected[["scenario_id","split","weekend","quality_score","edge_mean","edge_p10"]]
              .to_string(index=False,float_format=lambda v:f"{v:.3f}"))
    else:
        print("NO SELECTED WINDOWS: consult rejected_scenarios.csv; do not silently weaken thresholds")
    print("Reports:",out.resolve())
    if args.fail_on_missing and missing: return 2
    if args.fail_on_structural and bool((days.exists.astype(bool) & ~days.structural_ok.fillna(False).astype(bool)).any()): return 4
    if args.fail_on_empty and len(selected)==0: return 3
    return 0


if __name__=="__main__":
    sys.exit(main())
