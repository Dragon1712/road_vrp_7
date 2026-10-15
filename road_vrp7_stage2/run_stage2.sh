#!/usr/bin/env bash
# Offline July-2025 speed tensor and FIFO road cost build for Dragon1712/road_vrp_7
# Execute FROM the repository root. Usage: bash road_vrp7_stage2/run_stage2.sh test|build|validate|tdsp|all
set -euo pipefail
STAGE="${1:-all}"
SCRIPT="road_vrp7_stage2/build_july_stage2.py"
OUTPUT="datasets/processed/chicago/july_v1"
case "$STAGE" in
  test)
    python road_vrp7_stage2/tests/test_stage2.py
    ;;
  build)
    python "$SCRIPT" build \
      --csv-dir chicago_data \
      --segment-index datasets/processed/chicago/segment_index.csv \
      --static-segments datasets/processed/chicago/segments_static_july.csv \
      --graphml datasets/raw/chicago/chicago_loop_drive.graphml \
      --mapping datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv \
      --selected-segments datasets/processed/chicago/osm_match_loop/safe_mapping/selected_monitor_segments.csv \
      --mapping-summary datasets/processed/chicago/osm_match_loop/safe_mapping/safe_mapping_summary.json \
      --output-dir "$OUTPUT" --start 2025-07-02 --end 2025-07-31 --hash-files
    ;;
  validate)
    python "$SCRIPT" validate --output-dir "$OUTPUT"
    ;;
  tdsp)
    python "$SCRIPT" tdsp-smoke --output-dir "$OUTPUT" --departure "2025-07-28 08:00:00" --pairs 30
    cp "$OUTPUT/tdsp_smoke.csv" "$OUTPUT/tdsp_smoke_0800.csv"
    cp "$OUTPUT/tdsp_smoke_summary.json" "$OUTPUT/tdsp_smoke_0800_summary.json"
    python "$SCRIPT" tdsp-smoke --output-dir "$OUTPUT" --departure "2025-07-28 17:00:00" --pairs 30
    cp "$OUTPUT/tdsp_smoke.csv" "$OUTPUT/tdsp_smoke_1700.csv"
    cp "$OUTPUT/tdsp_smoke_summary.json" "$OUTPUT/tdsp_smoke_1700_summary.json"
    ;;
  all)
    bash "$0" test
    bash "$0" build
    bash "$0" validate
    bash "$0" tdsp
    ;;
  *) echo 'Usage: bash road_vrp7_stage2/run_stage2.sh test|build|validate|tdsp|all' >&2; exit 2 ;;
esac
