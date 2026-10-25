#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
R4H_ROOT="${R4H_ROOT:-runs/chicago_july_v1/step6c1_r4h}"
SOURCE="${SOURCE_STUDY:-r4g_train16_fix2}"
PREFIX="${PREFIX_DIR:-runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1}"
CAPTURE="${CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4g/${SOURCE}_capture}"
HORIZONS="${R4H_HORIZONS:-50,200,500}"
N_ALTS="${R4H_MAX_ALTERNATIVES:-8}"
STUDY="${STUDY_NAME:-r4h_train_all_v1}"
common=(--prefix-dir "$PREFIX" --capture-dir "$CAPTURE" --horizons "$HORIZONS"
        --action-count 8 --max-alternatives "$N_ALTS" --sampler-seed 20261019)
case "$MODE" in
 test)
   "$PY" -m unittest discover -s road_vrp7_step6c1_r4h/tests -p 'test_r4h*.py' -v
   "$PY" -m py_compile road_vrp7_step6c1_r4h/*.py
   bash -n road_vrp7_step6c1_r4h/run_r4h.sh
   ;;
 plan)
   "$PY" road_vrp7_step6c1_r4h/collect_train_cf.py --mode plan "${common[@]}" \
      --max-cases "${R4H_MAX_CASES:-0}" --max-snapshots-per-case "${R4H_SNAPSHOTS_PER_CASE:-0}"
   ;;
 smoke|pilot|full)
   case "$MODE" in
     smoke)
       STUDY="${STUDY_NAME:-r4h_train_smoke_v1}"
       cases="${R4H_MAX_CASES:-1}"
       states="${R4H_SNAPSHOTS_PER_CASE:-1}"
       N_ALTS="${R4H_MAX_ALTERNATIVES:-2}"
       common=(--prefix-dir "$PREFIX" --capture-dir "$CAPTURE" --horizons "$HORIZONS"
              --action-count 8 --max-alternatives "$N_ALTS" --sampler-seed 20261019)
       ;;
     pilot)
       STUDY="${STUDY_NAME:-r4h_train_pilot_8states_v1}"
       cases="${R4H_MAX_CASES:-4}"
       states="${R4H_SNAPSHOTS_PER_CASE:-2}"
       ;;
     full)
       STUDY="${STUDY_NAME:-r4h_train_full_v1}"
       cases="${R4H_MAX_CASES:-0}"
       states="${R4H_SNAPSHOTS_PER_CASE:-0}"
       ;;
   esac
   args=(--mode collect "${common[@]}" --output-dir "$R4H_ROOT/$STUDY"
         --max-cases "$cases" --max-snapshots-per-case "$states"
         --max-new-units "${R4H_MAX_NEW_UNITS:-0}")
   [[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
   echo "[R4H] mode=$MODE study=$STUDY K=$HORIZONS cases=$cases snapshots_per_case=$states alternatives=$N_ALTS" >&2
   "$PY" road_vrp7_step6c1_r4h/collect_train_cf.py "${args[@]}"
   ;;
 *) echo 'Usage: run_r4h.sh test|plan|smoke|pilot|full' >&2; exit 2;;
esac
