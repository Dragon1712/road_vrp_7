#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
PKG=road_vrp7_step6c1_r3_vq3
OUT_ROOT="${OUTPUT_ROOT:-runs/chicago_july_v1/step6c1_r3_vq3}"
COLLECTION_ROOT="${COLLECTION_ROOT:-runs/chicago_july_v1/step6c1_r3_vq}"
case "${1:-help}" in
  test)
    "$PY" -m unittest discover -s "$PKG/tests" -v
    "$PY" -m py_compile "$PKG/vq3_core.py" "$PKG/train_vq3.py" "$PKG/aggregate_vq3.py" "$PKG/report_vq3.py" "$PKG/preflight_vq3.py"
    ;;
  preflight)
    "$PY" "$PKG/preflight_vq3.py" \
      --collection-dir "$COLLECTION_ROOT/${COLLECTION_STUDY:-vq_train48_H20H50_R3_v1}" \
      --dataset-dir "${DATASET_DIR:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}"
    ;;
  train)
    : "${STUDY_NAME:?Specify NEW STUDY_NAME}"
    "$PY" "$PKG/train_vq3.py" \
      --collection-dir "$COLLECTION_ROOT/${COLLECTION_STUDY:-vq_train48_H20H50_R3_v1}" \
      --dataset-dir "${DATASET_DIR:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}" \
      --output-dir "$OUT_ROOT/$STUDY_NAME" --seed "${TRAIN_SEED:-42}" --trees "${TREES:-160}"
    ;;
  aggregate)
    : "${STUDY_NAME:?Specify NEW STUDY_NAME}"
    : "${INPUT_STUDIES:?Specify INPUT_STUDIES=name1,name2,...}"
    inputs=(); IFS=',' read -ra studies <<< "$INPUT_STUDIES"
    for study in "${studies[@]}";do inputs+=("$OUT_ROOT/$study");done
    joined=$(IFS=','; echo "${inputs[*]}")
    "$PY" "$PKG/aggregate_vq3.py" --study-dirs "$joined" --output-dir "$OUT_ROOT/$STUDY_NAME"
    ;;
  *) echo 'Usage: run_vq3.sh test|preflight|train|aggregate'; exit 2;;
esac
