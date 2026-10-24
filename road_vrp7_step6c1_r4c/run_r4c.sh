#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
PKG=road_vrp7_step6c1_r4c
BASE="${R4C_ROOT:-runs/chicago_july_v1/step6c1_r4c}"
CAP="${CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4a/r4a_pilot_20k_fix1_capture}"
R4B="${R4B_REPORT:-runs/chicago_july_v1/step6c1_r4b/r4b_exact_8cases_2snap_v1}"
args=(--capture-dir "$CAP" --r4b-dir "$R4B" --model-source "${MODEL_SOURCE:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}" --cost-dir "${COST_DIR:-datasets/processed/chicago/july_v1/dynamic_costs}" --task-dir "${TASK_DIR:-datasets/processed/chicago/july_v1/tdcvrp30/tasks_10}" --od-cache-dir "${OD_CACHE_DIR:-runs/chicago_july_v1/stage3/od_cache}")
case "${1:-help}" in
 test)
   "$PY" -m unittest discover -s "$PKG/tests" -p 'test_r4c*.py' -v
   "$PY" -m py_compile "$PKG"/*.py
   bash -n "$PKG/run_r4c.sh"
   ;;
 preflight)
   "$PY" "$PKG/counterfactual_r4c.py" --preflight-only "${args[@]}"
   ;;
 smoke|pilot)
   cases="${MAX_CASES:-0}";snaps="${MAX_SNAPSHOTS_PER_RUN:-2}"
   horizons="${HORIZONS:-1,5,20,50}";maximum="${MAX_ALTERNATIVES:-8}"
   if [[ "$1" == smoke ]]; then cases=1;snaps=1;horizons="${HORIZONS:-1,5,20}";maximum="${MAX_ALTERNATIVES:-3}";fi
   : "${STUDY_NAME:?Set new STUDY_NAME}"
   extra=();[[ "${RESUME:-0}" != 1 ]] || extra+=(--resume)
   "$PY" "$PKG/counterfactual_r4c.py" "${args[@]}" --output-dir "$BASE/$STUDY_NAME" \
      --max-cases "$cases" --max-snapshots-per-run "$snaps" \
      --horizons "$horizons" --max-alternatives "$maximum" \
      --sampler-seed "${SAMPLER_SEED:-20261019}" --action-count "${ACTION_COUNT:-8}" \
      --max-new-units "${MAX_NEW_UNITS:-0}" "${extra[@]}"
   ;;
 *) echo 'Usage: bash road_vrp7_step6c1_r4c/run_r4c.sh test|preflight|smoke|pilot' >&2; exit 2;;
esac
