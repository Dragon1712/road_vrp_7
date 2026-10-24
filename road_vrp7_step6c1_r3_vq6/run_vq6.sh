#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
PKG=road_vrp7_step6c1_r3_vq6
BASE="${VQ6_ROOT:-runs/chicago_july_v1/step6c1_r3_vq6}"
STUDY="${STUDY_NAME:-}"
TRAIN_STUDY="${TRAIN_STUDY:-vq6_train192_H50_seed42_v1}"
TRAIN_DIR="$BASE/$TRAIN_STUDY"
SOURCE="${MODEL_SOURCE:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}"
DATASET="${DATASET_DIR:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}"
COLLECTION="${COLLECTION_DIR:-runs/chicago_july_v1/step6c1_r3_vq4/vq4_train192_H20H50_R3_v1}"
TASK_IDS="${TASK_IDS:-task_0008,task_0009}"
# DEVELOPMENT ONLY: these validation dates are not a blind independent test.
SCENARIO_IDS="${SCENARIO_IDS:-2025-07-22_0800,2025-07-22_1700}"
SEEDS="${SEEDS:-60,61}"
SCORE_EVERY="${SCORE_EVERY:-8}"
MAX_SCORING_SEC="${MAX_SCORING_SEC:-2}"
COMMON=(--model-mode F2 --model-method P2 --model-protocol task \
  --model-source "$SOURCE" --split validation --task-ids "$TASK_IDS" \
  --scenario-ids "$SCENARIO_IDS" --seeds "$SEEDS" \
  --wall-sec 30 --max-iterations 20000 --anneal-iterations 5000 \
  --top-k 12 --action-count 8 --sampler-seed 20261019 \
  --score-every "$SCORE_EVERY" --max-scoring-sec "$MAX_SCORING_SEC" \
  --cache-size 0 --min-static-fraction 0 --fallback c3)
case "${1:-}" in
  test)
    "$PY" -m unittest discover -s "$PKG/tests" -p 'test_*.py' -v
    "$PY" -m py_compile "$PKG"/*.py
    bash -n "$PKG/run_vq6.sh"
    ;;
  train)
    : "${STUDY_NAME:?Set a new STUDY_NAME for train}"
    "$PY" "$PKG/train_vq6.py" --collection-dir "$COLLECTION" --dataset-dir "$DATASET" \
      --model-source "$SOURCE" --output-dir "$BASE/$STUDY" \
      --horizon 50 --seed "${TRAIN_SEED:-42}" --trees 160
    ;;
  online)
    : "${STUDY_NAME:?Set a new STUDY_NAME for online}"
    : "${ARM:?Set ARM=C3|DEFER_RESID|E0_MATCHED|RESID_OOF50}"
    case "$ARM" in
      C3)
        "$PY" road_vrp7_step6b2_r/benchmark_30s_r.py \
          --output-dir "$BASE/$STUDY" --control C3 "${COMMON[@]}"
        ;;
      DEFER_RESID|E0_MATCHED|RESID_OOF50)
        "$PY" "$PKG/online_vq6.py" --output-dir "$BASE/$STUDY" --control learned \
          --model-path "$TRAIN_DIR/${ARM}.joblib" "${COMMON[@]}"
        ;;
      *) echo 'Unknown ARM' >&2; exit 2;;
    esac
    ;;
  compare)
    : "${STUDY_NAME:?Set a new STUDY_NAME for comparison}"
    "$PY" "$PKG/compare_vq6.py" \
      --c3 "$BASE/${C3_STUDY:?Set C3_STUDY}" \
      --defer-resid "$BASE/${DEFER_STUDY:?Set DEFER_STUDY}" \
      --e0-matched "$BASE/${MATCHED_STUDY:?Set MATCHED_STUDY}" \
      --resid-oof50 "$BASE/${RESID_STUDY:?Set RESID_STUDY}" \
      --output-dir "$BASE/$STUDY"
    ;;
  *) echo 'Usage: run_vq6.sh test|train|online|compare' >&2;exit 2;;
esac
