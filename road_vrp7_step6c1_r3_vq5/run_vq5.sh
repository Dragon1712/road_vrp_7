#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
SCRIPT=road_vrp7_step6c1_r3_vq5
BASE="${VQ5_ROOT:-runs/chicago_july_v1/step6c1_r3_vq5}"
COLLECTION="${COLLECTION_DIR:-runs/chicago_july_v1/step6c1_r3_vq4/vq4_view192_v1}"
DATASET="${DATASET_DIR:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}"
BASELINE="${VQ4_MODEL_ROOT:-runs/chicago_july_v1/step6c1_r3_vq4/models}"
case "${1:-help}" in
 test)
    "$PY" -m unittest discover -s "$SCRIPT/tests" -p 'test_vq5*.py' -v
    "$PY" -m py_compile "$SCRIPT"/*.py
    bash -n "$SCRIPT/run_vq5.sh"
    ;;
 preflight)
    "$PY" "$SCRIPT/preflight_vq5.py" --collection-dir "$COLLECTION" \
      --dataset-dir "$DATASET" --sizes "${SIZES:-48,96,144,192}"
    ;;
 train)
    : "${TRAIN_SEED:?Specify TRAIN_SEED=42..47}"
    ANCHOR="$BASELINE/vq4_n48_seed${TRAIN_SEED}_v1"
    "$PY" "$SCRIPT/fixed_panel_vq5.py" --collection-dir "$COLLECTION" \
      --dataset-dir "$DATASET" --anchor-dir "$ANCHOR" \
      --output-dir "$BASE/${STUDY_NAME:-vq5_fixed192_seed${TRAIN_SEED}_v1}" \
      --seed "$TRAIN_SEED" --trees "${TREES:-160}" --sizes "${SIZES:-48,96,144,192}"
    ;;
 analyze)
    studies=()
    for s in 42 43 44 45 46 47; do studies+=("$BASE/vq5_fixed192_seed${s}_v1"); done
    list=$(IFS=,; printf '%s' "${studies[*]}")
    "$PY" "$SCRIPT/summarize_vq5.py" --study-dirs "$list" \
       --output-dir "$BASE/${ANALYSIS_STUDY:-vq5_fixed192_curve_seed42to47_v1}"
    ;;
 *) echo 'Usage: run_vq5.sh test|preflight|train|analyze' >&2;exit 2;;
esac
