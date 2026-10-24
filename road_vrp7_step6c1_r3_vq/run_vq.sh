#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
PKG=road_vrp7_step6c1_r3_vq
BASE="${OUTPUT_ROOT:-runs/chicago_july_v1/step6c1_r3_vq}"
case "${1:-help}" in
  test)
    "$PY" -m unittest discover -s "$PKG/tests" -v
    ;;
  collect)
    : "${STUDY_NAME:?Provide a NEW collection STUDY_NAME}"
    options=()
    if [[ "${RESUME:-0}" == 1 ]];then options+=(--resume);fi
    if [[ "${ROAD_REPLAY:-1}" == 1 ]];then options+=(--road-replay);fi
    "$PY" "$PKG/collect_vq.py" --output-dir "$BASE/$STUDY_NAME" \
      --dataset-dir "${DATASET_DIR:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}" \
      --states "${STATES:-48}" --seed "${SELECT_SEED:-20261025}" \
      --horizons "${HORIZONS:-20,50}" --replicates "${REPLICATES:-3}" \
      --max-new-units "${MAX_NEW_UNITS:-0}" --wall-per-arm "${WALL_PER_ARM:-30}" \
      "${options[@]}"
    ;;
  train)
    : "${STUDY_NAME:?Provide a NEW model evaluation STUDY_NAME}"
    : "${COLLECTION_STUDY:?Provide completed COLLECTION_STUDY}"
    "$PY" "$PKG/train_vq.py" --collection-dir "$BASE/$COLLECTION_STUDY" \
      --dataset-dir "${DATASET_DIR:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}" \
      --output-dir "$BASE/$STUDY_NAME" --seed "${TRAIN_SEED:-42}"
    ;;
  *) echo 'Usage: run_vq.sh test|collect|train';exit 2;;
esac
