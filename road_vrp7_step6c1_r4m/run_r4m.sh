#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
ROOT_OUT="${R4M_ROOT:-runs/chicago_july_v1/step6c1_r4m}"
NAME="${STUDY_NAME:-r4m_nested_risk_train_v1}"
# Paths are passed explicitly; no inherited generic TASK_IDS/SEEDS accepted.
ARGS=(
 --r4l-result "${R4M_R4L_DIR:-runs/chicago_july_v1/step6c1_r4l/r4l_seed_transfer_loto_train_v1}"
 --old-cf "${R4M_OLD_CF_DIR:-runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1}"
 --old-capture "${R4M_OLD_CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture}"
 --old-prefix "${R4M_OLD_PREFIX_DIR:-runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1}"
 --new-r4k "${R4M_NEW_R4K_DIR:-runs/chicago_july_v1/step6c1_r4k/r4k_full_128audited_v1}"
 --new-cf "${R4M_NEW_CF_DIR:-runs/chicago_july_v1/step6c1_r4h/r4j_train_early_s62s63_v1_cf_full}"
 --new-capture "${R4M_NEW_CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4g/r4j_train_early_s62s63_v1_capture}"
 --new-prefix "${R4M_NEW_PREFIX_DIR:-runs/chicago_july_v1/step6c1_r4g_prefix/r4j_train_early_s62s63_v1_prefix_full}"
)
case "$MODE" in
 test)
   "$PY" -m unittest discover -s road_vrp7_step6c1_r4m/tests -p 'test_r4m*.py' -v
   "$PY" -m py_compile road_vrp7_step6c1_r4m/*.py
   bash -n road_vrp7_step6c1_r4m/run_r4m.sh
   ;;
 preflight) "$PY" road_vrp7_step6c1_r4m/probe_r4m.py --mode preflight "${ARGS[@]}" ;;
 probe) "$PY" road_vrp7_step6c1_r4m/probe_r4m.py --mode probe "${ARGS[@]}" --output-dir "$ROOT_OUT/$NAME" ;;
 *) echo 'Usage: bash road_vrp7_step6c1_r4m/run_r4m.sh test|preflight|probe' >&2;exit 2 ;;
esac
