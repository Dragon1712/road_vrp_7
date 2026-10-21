#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
SOURCE_DIR="${SOURCE_DIR:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}"
FEATURES_DIR="${FEATURES_DIR:-runs/chicago_july_v1/step6b2/features_boundary_v1}"
SEED="${SEED:-42}"
if [[ "$SEED" == 42 ]]; then
 ANCHOR_DEFAULT="runs/chicago_july_v1/step6b2/b2_full_seed42_v1"
else
 ANCHOR_DEFAULT="runs/chicago_july_v1/step6b2_r/b2r_ablation_seed${SEED}_v1"
fi
ANCHOR_STUDY="${ANCHOR_STUDY:-$ANCHOR_DEFAULT}"
STUDY_NAME="${STUDY_NAME:-b3_anchor_film_seed${SEED}_v1}"
OUT="runs/chicago_july_v1/step6b3/$STUDY_NAME"
case "$MODE" in
 test)
  "$PY" -m unittest discover -s road_vrp7_step6b3/tests -v ;;
 train)
  "$PY" road_vrp7_step6b3/train_b3.py \
    --source-dir "$SOURCE_DIR" --features-dir "$FEATURES_DIR" \
    --anchor-study "$ANCHOR_STUDY" --output-dir "$OUT" \
    --seed "$SEED" --protocol "${PROTOCOL:-both}" \
    --methods "${METHODS:-M0,M1,M2,M3,M4}" \
    --trees "${TREES:-120}" --epochs "${EPOCHS:-55}" ;;
 smoke)
  "$PY" road_vrp7_step6b3/train_b3.py \
    --source-dir "$SOURCE_DIR" --features-dir "$FEATURES_DIR" \
    --anchor-study "$ANCHOR_STUDY" --output-dir "$OUT" \
    --seed "$SEED" --protocol task --methods 'M0,M1,M2,M3' \
    --trees 10 --epochs 3 ;;
 online)
  if [[ "${B3_METHOD:-M2}" == M0 ]]; then
    echo 'M0 should use original B2-R P2-F2 model and F2 features to avoid unnecessary F4 history overhead.' >&2
    exit 2
  fi
  MODEL_STUDY="${MODEL_STUDY:-runs/chicago_july_v1/step6b3/b3_anchor_film_seed42_v1}"
  MODEL_PATH="$MODEL_STUDY/task/${B3_METHOD:-M2}_model.joblib"
  "$PY" road_vrp7_step6b3/online_b3.py \
    --control learned --model-method P2 --model-mode F4 --model-protocol task \
    --model-path "$MODEL_PATH" --output-dir "$OUT" \
    --split validation --task-ids 'task_0008,task_0009' \
    --seeds 42,43 --wall-sec 30 --max-iterations 20000 \
    --anneal-iterations 5000 --top-k 12 \
    --scenario-ids "${SCENARIO_IDS:-2025-07-25_0800,2025-07-25_1700}" \
    --score-every "${SCORE_EVERY:-8}" --max-scoring-sec "${MAX_SCORING_SEC:-2}" ;;
 *) echo 'Usage: bash road_vrp7_step6b3/run_b3.sh test|smoke|train|online' >&2;exit 2;;
esac
