#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
PKG=road_vrp7_step6c1_r3_vq2
ROOT_OUT="${OUTPUT_ROOT:-runs/chicago_july_v1/step6c1_r3_vq2}"
COLLECTION_ROOT="${COLLECTION_ROOT:-runs/chicago_july_v1/step6c1_r3_vq}"
case "${1:-help}" in
  test)
    "$PY" -m unittest discover -s "$PKG/tests" -v
    "$PY" -m py_compile "$PKG/train_vq2.py" "$PKG/vq2_core.py"
    ;;
  train)
    : "${STUDY_NAME:?Set a NEW STUDY_NAME}"
    "$PY" "$PKG/train_vq2.py" \
      --collection-dir "$COLLECTION_ROOT/${COLLECTION_STUDY:-vq_train48_H20H50_R3_v1}" \
      --dataset-dir "${DATASET_DIR:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}" \
      --output-dir "$ROOT_OUT/$STUDY_NAME" \
      --seed "${TRAIN_SEED:-42}" --trees "${TREES:-160}"
    ;;
  aggregate)
    : "${STUDY_NAME:?Set a NEW aggregation STUDY_NAME}"
    : "${INPUT_STUDIES:?Set INPUT_STUDIES=study1,study2,...}"
    inputs=()
    IFS=',' read -r -a named <<< "$INPUT_STUDIES"
    for study in "${named[@]}";do inputs+=("$ROOT_OUT/$study");done
    joined=$(IFS=',';echo "${inputs[*]}")
    "$PY" "$PKG/aggregate_vq2.py" --study-dirs "$joined" --output-dir "$ROOT_OUT/$STUDY_NAME"
    ;;
  *) echo 'Usage: run_vq2.sh test|train|aggregate';exit 2;;
esac
