#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
BASE="runs/chicago_july_v1/step6c1_r3_tricks"
NAME="${STUDY_NAME:-}"
case "$MODE" in
  test)
    "$PY" -m unittest discover -s road_vrp7_step6c1_r3_tricks/tests -v
    "$PY" -m unittest discover -s road_vrp7_step6c1_r2/tests -v ;;
  train)
    : "${NAME:?Specify a NEW STUDY_NAME}"
    "$PY" road_vrp7_step6c1_r3_tricks/train_r3.py \
      --output-dir "$BASE/$NAME" \
      --dataset-dir "${LABEL_STUDY:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}" \
      --model-source "${MODEL_SOURCE:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}" \
      --baseline-artifact "${BASELINE_ARTIFACT:-runs/chicago_july_v1/step6c1_r2/r2_task_seed42_v1/R2_F2.joblib}" \
      --seed "${TRAIN_SEED:-42}" --epochs "${EPOCHS:-100}" ;;
  aggregate)
    : "${NAME:?Specify a NEW STUDY_NAME}"
    studies=()
    IFS=',' read -r -a seeds <<< "${TRAIN_SEEDS:-42,43,44,45,46,47}"
    for seed in "${seeds[@]}"; do studies+=("$BASE/r3_seed${seed}_v1"); done
    "$PY" road_vrp7_step6c1_r3_tricks/aggregate_r3.py --studies "${studies[@]}" --output-dir "$BASE/$NAME" ;;
  online)
    : "${NAME:?Specify a NEW STUDY_NAME}"; : "${ARM:?Choose ARM (C3, M0, T0_R2_F2, T1...T7)}"
    args=(--arm "$ARM" --output-dir "$BASE/$NAME" \
      --model-study "$BASE/${MODEL_STUDY:-r3_seed42_v1}" \
      --model-source "${MODEL_SOURCE:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}" \
      --tasks "${TASK_IDS:-task_0008,task_0009}" \
      --scenarios "${SCENARIO_IDS:-2025-07-25_0800,2025-07-25_1700}" \
      --seeds "${SEEDS:-60,61,62,63,64}" --split "${SPLIT:-validation}")
    [[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
    [[ "${DRY_RUN:-0}" != 1 ]] || args+=(--dry-run)
    [[ -z "${MAX_NEW_SHARDS:-}" ]] || args+=(--max-new-shards "$MAX_NEW_SHARDS")
    "$PY" road_vrp7_step6c1_r3_tricks/study_r3.py "${args[@]}" ;;
  compare)
    : "${NAME:?Specify STUDY_NAME}";
    studies=("C3=$BASE/${C3_STUDY:-r3_online_C3_v1}" "M0=$BASE/${M0_STUDY:-r3_online_M0_v1}")
    IFS=',' read -r -a arms <<< "${COMPARE_ARMS:-T0_R2_F2,T1_CB_RMSE,T2_CB_HUBER,T3_PAIRWISE,T4_WEIGHTED_PAIR,T5_HUBER_MARGIN,T6_WEIGHTED_MARGIN,T7_BLEND_MARGIN}"
    for arm in "${arms[@]}"; do studies+=("$arm=$BASE/r3_online_${arm}_v1"); done
    "$PY" road_vrp7_step6c1_r3_tricks/compare_r3.py --studies "${studies[@]}" --output-dir "$BASE/$NAME" ;;
  *) echo 'Usage: run_r3.sh test|train|aggregate|online|compare' >&2;exit 2;;
esac
