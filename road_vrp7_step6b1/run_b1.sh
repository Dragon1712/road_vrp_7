#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
SOURCE_DIR="${SOURCE_DIR:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}"
STUDY_NAME="${STUDY_NAME:-step6b1_static_repair_v1}"
OUTPUT_DIR="runs/chicago_july_v1/step6b1/$STUDY_NAME"
case "$MODE" in
  test)
    "$PY" -m unittest discover -s road_vrp7_step6b1/tests -v ;;
  audit)
    "$PY" road_vrp7_step6b1/audit_static_repair.py \
      --source-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR" ;;
  smoke)
    "$PY" road_vrp7_step6b1/train_b1.py \
      --source-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR" \
      --protocol task --seed "${SEED:-42}" --trees 10 \
      --methods "P1_class_static_time,P2_reg_static_time,P3_pairwise_static_time" ;;
  train)
    "$PY" road_vrp7_step6b1/train_b1.py \
      --source-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR" \
      --protocol "${PROTOCOL:-both}" --seed "${SEED:-42}" \
      --trees "${TREES:-160}" --methods "${METHODS:-P0_class_core,P1_class_static_time,P2_reg_static_time,P3_pairwise_static_time,P4_two_part_static_time,P5_reg_history,P6_pairwise_history,P7_two_part_history,P8_class_history}" ;;
  *)
    echo 'Usage: bash road_vrp7_step6b1/run_b1.sh test|audit|smoke|train' >&2
    exit 2 ;;
esac
