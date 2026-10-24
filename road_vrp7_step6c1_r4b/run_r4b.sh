#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
ROOT="${R4B_ROOT:-runs/chicago_july_v1/step6c1_r4b}"
CAPTURE_DIR="${CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4a/r4a_pilot_20k_fix1_capture}"
STUDY="${STUDY_NAME:-r4b_exact_k1_5_20_50_v1}"
case "${1:-help}" in
 test)
   "$PY" -m unittest discover -s road_vrp7_step6c1_r4b/tests -p 'test_r4b*.py' -v
   "$PY" -m py_compile road_vrp7_step6c1_r4b/*.py
   bash -n road_vrp7_step6c1_r4b/run_r4b.sh
   ;;
 preflight)
   "$PY" road_vrp7_step6c1_r4b/preflight_r4b.py --capture-dir "$CAPTURE_DIR"
   ;;
 smoke|pilot)
   cases=0; states="${MAX_SNAPSHOTS_PER_RUN:-2}"
   if [[ "$1" == smoke ]];then cases=1;states=1;fi
   "$PY" road_vrp7_step6c1_r4b/verify_r4b.py \
     --capture-dir "$CAPTURE_DIR" --output-dir "$ROOT/$STUDY" \
     --model-source "${MODEL_SOURCE:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}" \
     --cost-dir "${COST_DIR:-datasets/processed/chicago/july_v1/dynamic_costs}" \
     --task-dir "${TASK_DIR:-datasets/processed/chicago/july_v1/tdcvrp30/tasks_10}" \
     --od-cache-dir "${OD_CACHE_DIR:-runs/chicago_july_v1/stage3/od_cache}" \
     --horizons "${HORIZONS:-1,5,20,50}" --max-cases "$cases" \
     --max-snapshots-per-run "$states"
   ;;
 *) echo 'Usage: bash road_vrp7_step6c1_r4b/run_r4b.sh test|preflight|smoke|pilot' >&2;exit 2;;
esac
