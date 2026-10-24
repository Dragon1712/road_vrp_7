#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
BASE="${R4A_ROOT:-runs/chicago_july_v1/step6c1_r4a}"
PKG="road_vrp7_step6c1_r4a"
MODE="${1:-help}"
case "$MODE" in
 test)
   "$PY" -m unittest discover -s "$PKG/tests" -v
   "$PY" -m py_compile "$PKG"/*.py
   bash -n "$PKG/run_r4a.sh" ;;
 smoke|pilot)
   if [[ "$MODE" == smoke ]];then
     STUDY_PREFIX="${STUDY_PREFIX:-r4a_smoke_2k_v1}"
     MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
     TASK_IDS="${TASK_IDS:-task_0008}"
     SCENARIO_IDS="${SCENARIO_IDS:-2025-07-22_0800}"
     SEEDS="${SEEDS:-60}"
     SNAPSHOTS="${SNAPSHOTS:-100,500,1000,1500}"
   else
     STUDY_PREFIX="${STUDY_PREFIX:-r4a_pilot_20k_v1}"
     MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
     TASK_IDS="${TASK_IDS:-task_0008,task_0009}"
     SCENARIO_IDS="${SCENARIO_IDS:-2025-07-22_0800,2025-07-22_1700}"
     SEEDS="${SEEDS:-60,61}"
     SNAPSHOTS="${SNAPSHOTS:-500,2500,5000,10000,15000,19000}"
   fi
   # The collector creates a manifest before solving; failed v1 smoke may
   # therefore leave a partial directory. Never append to or delete it.
   for arm in plain capture audit;do
     if [[ -e "$BASE/${STUDY_PREFIX}_${arm}" ]];then
       echo "Existing R4A output: $BASE/${STUDY_PREFIX}_${arm}" >&2
       echo "Choose a NEW STUDY_PREFIX (e.g. STUDY_PREFIX=r4a_smoke_2k_fix1)." >&2
       exit 2
     fi
   done
   for arm in plain capture;do
     extras=();
     if [[ "$MODE" == pilot && "$arm" == capture ]];then extras+=(--section-only);fi
     "$PY" "$PKG/collect_r4a.py" --arm "$arm" \
       --output-dir "$BASE/${STUDY_PREFIX}_${arm}" \
       --model-source "${MODEL_SOURCE:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}" \
       --cost-dir "${COST_DIR:-datasets/processed/chicago/july_v1/dynamic_costs}" \
       --task-dir "${TASK_DIR:-datasets/processed/chicago/july_v1/tdcvrp30/tasks_10}" \
       --od-cache-dir "${OD_CACHE_DIR:-runs/chicago_july_v1/stage3/od_cache}" \
       --task-ids "$TASK_IDS" --scenario-ids "$SCENARIO_IDS" --seeds "$SEEDS" \
       --wall-sec 30 --max-iterations "$MAX_ITERATIONS" --anneal-iterations 5000 \
       --top-k 12 --snapshots "$SNAPSHOTS" --max-snapshots 16 "${extras[@]}"
   done
   opts=();[[ "$MODE" != smoke ]] || opts+=(--require-exact)
   "$PY" "$PKG/compare_r4a.py" \
     --plain "$BASE/${STUDY_PREFIX}_plain" \
     --capture "$BASE/${STUDY_PREFIX}_capture" \
     --output-dir "$BASE/${STUDY_PREFIX}_audit" "${opts[@]}" ;;
 compare)
   : "${PLAIN_STUDY:?Specify PLAIN_STUDY}";: "${CAPTURE_STUDY:?Specify CAPTURE_STUDY}";: "${STUDY_NAME:?Specify new STUDY_NAME}"
   opts=();[[ "${REQUIRE_EXACT:-0}" != 1 ]] || opts+=(--require-exact)
   "$PY" "$PKG/compare_r4a.py" --plain "$BASE/$PLAIN_STUDY" \
     --capture "$BASE/$CAPTURE_STUDY" --output-dir "$BASE/$STUDY_NAME" "${opts[@]}" ;;
 *) echo 'Usage: bash road_vrp7_step6c1_r4a/run_r4a.sh test|smoke|pilot|compare' >&2;exit 2 ;;
esac
