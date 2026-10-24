#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
OUT="${OUTPUT_ROOT:-runs/chicago_july_v1/step6c1_r4e}"
CAPTURE="${CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4a/r4a_pilot_20k_fix1_capture}"
R4B="${R4B_DIR:-runs/chicago_july_v1/step6c1_r4b/r4b_exact_8cases_2snap_v1}"
R4C="${R4C_DIR:-runs/chicago_july_v1/step6c1_r4c/r4c_pilot_16snaps_v1}"
SOURCE="${MODEL_SOURCE:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}"
COST="${COST_DIR:-datasets/processed/chicago/july_v1/dynamic_costs}"
TASKS="${TASK_DIR:-datasets/processed/chicago/july_v1/tdcvrp30/tasks_10}"
OD="${OD_CACHE_DIR:-runs/chicago_july_v1/stage3/od_cache}"
H="${HORIZONS:-100,200,500}"
PROOF="${LONG_PROOF_STUDY:-r4e_long_proof_16snaps_v1}"
common=(--capture-dir "$CAPTURE" --r4b-dir "$R4B" \
 --model-source "$SOURCE" --cost-dir "$COST" --task-dir "$TASKS" --od-cache-dir "$OD")
case "$MODE" in
 test)
   "$PY" -m unittest discover -s road_vrp7_step6c1_r4e/tests -p 'test_r4e*.py' -v
   "$PY" -m py_compile road_vrp7_step6c1_r4e/*.py
   bash -n road_vrp7_step6c1_r4e/run_r4e.sh
   ;;
 verify-smoke|verify-pilot)
   if [[ "$MODE" == verify-smoke ]];then
      STUDY="${STUDY_NAME:-r4e_long_proof_smoke_v1}"
      cases=1;states=1
   else
      STUDY="${STUDY_NAME:-r4e_long_proof_16snaps_v1}"
      cases=0;states="${MAX_SNAPSHOTS_PER_RUN:-2}"
   fi
   args=("${common[@]}" --output-dir "$OUT/$STUDY" --horizons "$H" --max-cases "$cases" --max-snapshots-per-run "$states")
   [[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
   "$PY" road_vrp7_step6c1_r4e/verify_long.py "${args[@]}"
   ;;
 collect-smoke|collect-pilot)
   if [[ "$MODE" == collect-smoke ]];then
      STUDY="${STUDY_NAME:-r4e_long_cf_smoke_v1}"
      PROOF="${LONG_PROOF_STUDY:-r4e_long_proof_smoke_v1}"
      cases=1;states=1;alts="${MAX_ALTERNATIVES:-2}"
   else
      STUDY="${STUDY_NAME:-r4e_long_cf_16snaps_v1}"
      cases="${MAX_CASES:-0}";states="${MAX_SNAPSHOTS_PER_RUN:-2}";alts="${MAX_ALTERNATIVES:-8}"
   fi
   args=("${common[@]}" --r4c-dir "$R4C" --long-verify-dir "$OUT/$PROOF" \
    --output-dir "$OUT/$STUDY" --horizons "$H" --max-cases "$cases" \
    --max-snapshots-per-run "$states" --action-count 8 \
    --max-alternatives "$alts" --sampler-seed 20261019 --max-new-units "${MAX_NEW_UNITS:-0}")
   [[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
   "$PY" road_vrp7_step6c1_r4e/collect_long.py "${args[@]}"
   ;;
 *) echo 'Usage: bash road_vrp7_step6c1_r4e/run_r4e.sh test|verify-smoke|collect-smoke|verify-pilot|collect-pilot' >&2;exit 2;;
esac
