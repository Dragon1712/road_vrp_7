#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
SOURCE_DIR="${SOURCE_DIR:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}"
FEATURES_DIR="${FEATURES_DIR:-runs/chicago_july_v1/step6b2/features_boundary_v1}"
STUDY_NAME="${STUDY_NAME:-b2r_seed42_v1}"
RESULT="runs/chicago_july_v1/step6b2_r/$STUDY_NAME"
MODEL_STUDY="${MODEL_STUDY:-runs/chicago_july_v1/step6b2/b2_full_seed42_v1}"
SEEDS="${SEEDS:-42,43}"
case "$MODE" in
  test)
    "$PY" -m unittest discover -s road_vrp7_step6b2_r/tests -v ;;
  train)
    "$PY" road_vrp7_step6b2_r/train_b2_r.py \
      --source-dir "$SOURCE_DIR" --features-dir "$FEATURES_DIR" \
      --output-dir "$RESULT" --seed "${SEED:-42}" --trees "${TREES:-120}" \
      --protocol "${PROTOCOL:-both}" \
      --methods "${METHODS:-P2,P4}" --modes "${MODES:-F0,F1,F2,F3,F4,F5}" ;;
  permutation)
    "$PY" road_vrp7_step6b2_r/frozen_permutation.py \
      --source-dir "$SOURCE_DIR" --features-dir "$FEATURES_DIR" \
      --study "$MODEL_STUDY" --output "$RESULT.json" \
      --models "${METHODS:-P2,P4}" --modes "${MODES:-F3,F4}" \
      --seeds "${PERM_SEEDS:-100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119}" ;;
  online)
    CONTROL="${CONTROL:-learned}"
    METHOD="${METHOD:-P2}"
    FEATURE_MODE="${FEATURE_MODE:-F2}"
    MODEL_PROTOCOL="${MODEL_PROTOCOL:-task}"
    TASK_IDS="${TASK_IDS:-task_0008,task_0009}"
    SCENARIO_IDS="${SCENARIO_IDS:-2025-07-25_0800,2025-07-25_1700}"
    base_args=( --output-dir "$RESULT" --control "$CONTROL" \
      --model-source "$SOURCE_DIR" --model-method "$METHOD" \
      --model-mode "$FEATURE_MODE" --model-protocol "$MODEL_PROTOCOL" \
      --seeds "$SEEDS" --split "${SPLIT:-validation}" \
      --wall-sec 30 --max-iterations "${MAX_ITERATIONS:-20000}" \
      --anneal-iterations "${ANNEAL_ITERATIONS:-5000}" --top-k "${TOP_K:-12}" \
      --action-count "${ACTION_COUNT:-8}" \
      --score-every "${SCORE_EVERY:-8}" --max-scoring-sec "${MAX_SCORING_SEC:-2.0}" \
      --cache-size "${CACHE_SIZE:-64}" --min-static-fraction "${MIN_STATIC_FRACTION:-0}" \
      --fallback "${FALLBACK:-c3}" )
    [[ -z "$TASK_IDS" ]] || base_args+=(--task-ids "$TASK_IDS")
    [[ -z "$SCENARIO_IDS" ]] || base_args+=(--scenario-ids "$SCENARIO_IDS")
    if [[ "$CONTROL" == learned ]];then
      MODEL_PATH="${MODEL_PATH:-$MODEL_STUDY/$MODEL_PROTOCOL/${METHOD}_${FEATURE_MODE}_model.joblib}"
      base_args+=(--model-path "$MODEL_PATH")
    fi
    "$PY" road_vrp7_step6b2_r/benchmark_30s_r.py "${base_args[@]}" ;;
  compare)
    : "${BASE_RUN:?Set BASE_RUN to the C3 or sampled_random run folder}"
    : "${TREAT_RUN:?Set TREAT_RUN to the treatment run folder}"
    "$PY" road_vrp7_step6b2_r/analyze_r.py \
      --baseline "$BASE_RUN" --treatment "$TREAT_RUN" --output "$RESULT.json" ;;
  repair-report)
    "$PY" road_vrp7_step6b2_r/repair_f5_report.py \
      --study "${ORIGINAL_STUDY:-runs/chicago_july_v1/step6b2/b2_full_seed42_v1}" \
      --output-dir "$RESULT" ;;
  *)
    echo 'Usage: bash road_vrp7_step6b2_r/run_b2r.sh test|train|permutation|online|compare|repair-report' >&2
    exit 2 ;;
esac
