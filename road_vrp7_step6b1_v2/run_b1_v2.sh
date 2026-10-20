#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
SOURCE_DIR="${SOURCE_DIR:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}"
STUDY_NAME="${STUDY_NAME:-b1v2_seed42_v1}"
OUTPUT_DIR="runs/chicago_july_v1/step6b1_v2/${STUDY_NAME}"
case "$MODE" in
 test) "$PY" -m unittest discover -s road_vrp7_step6b1_v2/tests -v ;;
 audit) "$PY" road_vrp7_step6b1_v2/audit_static_repair.py --source-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR" ;;
 smoke)
    "$PY" road_vrp7_step6b1_v2/train.py --source-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR" \
      --protocol task --trees 10 --seed "${SEED:-42}" \
      --methods P1_class_static_time,P2_reg_static_time,P3_pairwise_static_time,P9_pairwise_interaction_static,P11_two_part_calibrated_static ;;
 train)
    "$PY" road_vrp7_step6b1_v2/train.py --source-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR" \
      --protocol "${PROTOCOL:-both}" --trees "${TREES:-120}" --seed "${SEED:-42}" \
      --methods "${METHODS:-P0_class_core,P1_class_static_time,P2_reg_static_time,P3_pairwise_static_time,P4_two_part_static_time,P5_reg_history,P6_pairwise_history,P7_two_part_history,P8_class_history,P9_pairwise_interaction_static,P10_pairwise_interaction_history,P11_two_part_calibrated_static,P12_two_part_calibrated_history}" ;;
 *) echo 'Usage: bash road_vrp7_step6b1_v2/run_b1_v2.sh test|audit|smoke|train' >&2;exit 2;;
esac
