#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
RUNS="runs/chicago_july_v1/step6c1_r2"
NAME="${STUDY_NAME:-}"
SEEDS="${SEEDS:-42,43}"
SOURCE="${SOURCE_DIR:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}"
case "$MODE" in
 test)
   "$PY" -m unittest discover -s road_vrp7_step6c1_r2/tests -v ;;
 collect)
   : "${NAME:?Specify new STUDY_NAME}"
   args=(--source-dir "$SOURCE" --output-dir "$RUNS/$NAME" --split "${SPLIT:-train}"
      --seeds "$SEEDS" --max-per-run "${MAX_PER_RUN:-16}" --stride "${STRIDE:-8}"
      --max-iterations 20000 --anneal-iterations 5000 --top-k 12)
   [[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
   [[ -z "${MAX_RUNS:-}" ]] || args+=(--max-runs "$MAX_RUNS")
   [[ -z "${MAX_TASKS:-}" ]] || args+=(--max-tasks "$MAX_TASKS")
   [[ -z "${MAX_SCENARIOS:-}" ]] || args+=(--max-scenarios "$MAX_SCENARIOS")
   [[ -z "${TASK_IDS:-}" ]] || args+=(--task-ids "$TASK_IDS")
   [[ -z "${SCENARIO_IDS:-}" ]] || args+=(--scenario-ids "$SCENARIO_IDS")
   "$PY" road_vrp7_step6c1_r2/collect_c3.py "${args[@]}" ;;
 label)
   : "${NAME:?Specify new STUDY_NAME}"
   : "${CAPTURE_STUDY:?Specify CAPTURE_STUDY directory}"
   args=(--capture-dir "$RUNS/$CAPTURE_STUDY" --output-dir "$RUNS/$NAME"
      --deadline-sec "${LABEL_DEADLINE:-3.0}")
   [[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
   [[ -z "${MAX_STATES:-}" ]] || args+=(--max-states "$MAX_STATES")
   "$PY" road_vrp7_step6c1_r2/label_c3.py "${args[@]}" ;;
 train)
   : "${NAME:?Specify new STUDY_NAME}"
   : "${LABEL_STUDY:?Specify LABEL_STUDY directory}"
   "$PY" road_vrp7_step6c1_r2/train_c3.py --dataset-dir "$RUNS/$LABEL_STUDY" \
      --output-dir "$RUNS/$NAME" --model-source "$SOURCE" --seed "${SEED:-42}" \
      --protocol "${PROTOCOL:-task}" --modes "${MODES:-F2,F4}" ;;
 online)
   : "${NAME:?Specify new STUDY_NAME}"
   : "${MODEL_STUDY:?Specify MODEL_STUDY directory}"
   MODE_NAME="${MODEL_MODE:-F2}"
   [[ "$MODE_NAME" == F2 || "$MODE_NAME" == F4 ]] || { echo 'MODE_NAME must be F2/F4' >&2;exit 2;}
   additional=()
   [[ "${R2_FORCE_DEFER:-0}" != 1 ]] || additional+=(--r2-force-defer)
   "$PY" road_vrp7_step6c1_r2/online_r2.py "${additional[@]}" --control learned \
      --model-method P2 --model-mode "$MODE_NAME" --model-protocol "${PROTOCOL:-task}" \
      --model-path "$RUNS/$MODEL_STUDY/R2_${MODE_NAME}.joblib" \
      --model-source "$SOURCE" --output-dir "$RUNS/$NAME" \
      --seeds "$SEEDS" --split "${ONLINE_SPLIT:-validation}" \
      --task-ids "${TASK_IDS:-task_0008,task_0009}" \
      --scenario-ids "${SCENARIO_IDS:-2025-07-25_0800,2025-07-25_1700}" \
      --wall-sec 30 --max-iterations 20000 --anneal-iterations 5000 \
      --top-k 12 --action-count 8 --sampler-seed 20261019 \
      --score-every "${SCORE_EVERY:-8}" --max-scoring-sec "${MAX_SCORING_SEC:-2}" \
      --cache-size 0 --min-static-fraction 0 --fallback c3 ;;
 compare)
   : "${NAME:?Specify new STUDY_NAME}"
   : "${R2_RUN:?Specify R2_RUN full relative results path}"
   : "${C3_RUN:?Specify C3_RUN full relative results path}"
   "$PY" road_vrp7_step6c1_r2/compare_r2.py \
      --m0 "${M0_RUN:-runs/chicago_july_v1/step6b2_r/b2r_online_P2_F2_match8_v1}" \
      --c3 "$C3_RUN" --r2 "$R2_RUN" --output-dir "$RUNS/$NAME" ;;
 *) echo 'Usage: run_r2.sh test|collect|label|train|online|compare' >&2;exit 2;;
esac
