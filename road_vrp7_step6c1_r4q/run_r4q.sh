#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
ARGS=(
 --r4o-result "${R4Q_R4O_DIR:-runs/chicago_july_v1/step6c1_r4o/r4o_struct_rank_gate_train_v1}"
 --r4n-result "${R4Q_R4N_DIR:-runs/chicago_july_v1/step6c1_r4n/r4n_rank_gate_train_v1}"
 --r4m-result "${R4Q_R4M_DIR:-runs/chicago_july_v1/step6c1_r4m/r4m_nested_risk_train_v1}"
 --r4l-result "${R4Q_R4L_DIR:-runs/chicago_july_v1/step6c1_r4l/r4l_seed_transfer_loto_train_v1}"
 --old-cf "${R4Q_OLD_CF_DIR:-runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1}"
 --old-capture "${R4Q_OLD_CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture}"
 --old-prefix "${R4Q_OLD_PREFIX_DIR:-runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1}"
 --new-r4k "${R4Q_NEW_R4K_DIR:-runs/chicago_july_v1/step6c1_r4k/r4k_full_128audited_v1}"
 --new-cf "${R4Q_NEW_CF_DIR:-runs/chicago_july_v1/step6c1_r4h/r4j_train_early_s62s63_v1_cf_full}"
 --new-capture "${R4Q_NEW_CAPTURE_DIR:-runs/chicago_july_v1/step6c1_r4g/r4j_train_early_s62s63_v1_capture}"
 --new-prefix "${R4Q_NEW_PREFIX_DIR:-runs/chicago_july_v1/step6c1_r4g_prefix/r4j_train_early_s62s63_v1_prefix_full}"
 --cost-dir "${R4Q_COST_DIR:-datasets/processed/chicago/july_v1/dynamic_costs}"
 --task-dir "${R4Q_TASK_DIR:-datasets/processed/chicago/july_v1/tdcvrp30/tasks_10}"
)
case "$MODE" in
  test)
    "$PY" -m unittest discover -s road_vrp7_step6c1_r4q/tests -p 'test_r4q*.py' -v
    "$PY" -m py_compile road_vrp7_step6c1_r4q/*.py
    bash -n road_vrp7_step6c1_r4q/run_r4q.sh
    ;;
  preflight)
    "$PY" road_vrp7_step6c1_r4q/probe_r4q.py --mode preflight "${ARGS[@]}"
    ;;
  probe)
    "$PY" road_vrp7_step6c1_r4q/probe_r4q.py --mode probe "${ARGS[@]}" \
      --output-dir "${R4Q_ROOT:-runs/chicago_july_v1/step6c1_r4q}/${STUDY_NAME:-r4q_past24_traffic_train_v1}"
    ;;
  *) echo 'Usage: run_r4q.sh test|preflight|probe' >&2; exit 2;;
esac
