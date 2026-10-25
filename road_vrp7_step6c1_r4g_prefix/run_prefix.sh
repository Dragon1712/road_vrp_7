#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
ROOT_OUT="${PREFIX_ROOT:-runs/chicago_july_v1/step6c1_r4g_prefix}"
SOURCE="${SOURCE_STUDY:-r4g_train16_fix2}"
SRC_ROOT="${R4G_ROOT:-runs/chicago_july_v1/step6c1_r4g}"
CAPTURE="${CAPTURE_DIR:-${SRC_ROOT}/${SOURCE}_capture}"
PLAIN="${PLAIN_DIR:-${SRC_ROOT}/${SOURCE}_plain}"
case "$MODE" in
  test)
    "$PY" -m unittest discover -s road_vrp7_step6c1_r4g_prefix/tests -p 'test_prefix*.py' -v
    "$PY" -m py_compile road_vrp7_step6c1_r4g_prefix/*.py
    bash -n road_vrp7_step6c1_r4g_prefix/run_prefix.sh
    ;;
  plan)
    "$PY" road_vrp7_step6c1_r4g_prefix/plan_prefix.py \
      --capture-dir "$CAPTURE" --plain-dir "$PLAIN" \
      --horizons "${HORIZONS:-1,50,200,500}" \
      --max-cases "${MAX_CASES:-0}" \
      --max-snapshots-per-case "${MAX_SNAPSHOTS_PER_CASE:-0}"
    ;;
  smoke|pilot)
    if [[ "$MODE" == smoke ]];then
      NAME="${STUDY_NAME:-r4g_prefix_1case_smoke_v1}"
      CASES=1
      STATES="${MAX_SNAPSHOTS_PER_CASE:-3}"
    else
      NAME="${STUDY_NAME:-r4g_prefix_16cases_v1}"
      CASES=0
      STATES="${MAX_SNAPSHOTS_PER_CASE:-0}"
    fi
    ARGS=(--capture-dir "$CAPTURE" --plain-dir "$PLAIN"
          --output-dir "$ROOT_OUT/$NAME"
          --reference-wall-sec "${REFERENCE_WALL_SEC:-90}"
          --horizons "${HORIZONS:-1,50,200,500}"
          --max-cases "$CASES" --max-snapshots-per-case "$STATES")
    [[ "${RESUME:-0}" != 1 ]] || ARGS+=(--resume)
    "$PY" road_vrp7_step6c1_r4g_prefix/verify_prefix.py "${ARGS[@]}"
    ;;
  *) echo 'Usage: run_prefix.sh test|plan|smoke|pilot' >&2;exit 2;;
esac
