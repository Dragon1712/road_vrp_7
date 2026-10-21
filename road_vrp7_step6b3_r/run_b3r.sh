#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
OUT_ROOT="runs/chicago_july_v1/step6b3_r"
STUDY_NAME="${STUDY_NAME:-}"
DATA="${SOURCE_DIR:-runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1}"
FEAT="${FEATURES_DIR:-runs/chicago_july_v1/step6b2/features_boundary_v1}"
B3="${MODEL_STUDY:-runs/chicago_july_v1/step6b3/b3_anchor_film_seed42_v1}"
SEEDS="${SEEDS:-42,43}"
case "$MODE" in
 test)
  "$PY" -m unittest discover -s road_vrp7_step6b3_r/tests -v ;;
 audit)
  : "${STUDY_NAME:?Set a new STUDY_NAME, e.g. b3r_audit_s42to47_v1}"
  "$PY" road_vrp7_step6b3_r/audit_b3r.py \
     --root runs/chicago_july_v1/step6b3 --output-dir "$OUT_ROOT/$STUDY_NAME" \
     --seeds "${MODEL_SEEDS:-42,43,44,45,46,47}" ;;
 placebo)
  : "${STUDY_NAME:?Set a new STUDY_NAME, e.g. b3r_placebo_v1}"
  "$PY" road_vrp7_step6b3_r/placebo_b3r.py \
      --source-dir "$DATA" --features-dir "$FEAT" \
      --study-root runs/chicago_july_v1/step6b3 \
      --model-seeds "${MODEL_SEEDS:-42,43,44,45,46,47}" \
      --permutation-seeds "${PERM_SEEDS:-100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119}" \
      --methods "${METHODS:-M1,M2,M3,M4}" \
      --shuffles "${SHUFFLES:-traffic_and_mask,traffic_only,mask_only}" \
      --output-dir "$OUT_ROOT/$STUDY_NAME" ;;
 online)
  : "${STUDY_NAME:?Set a new STUDY_NAME, e.g. b3r_online_M4_match8_v1}"
  METHOD="${B3_METHOD:-M4}"
  case "$METHOD" in M1|M2|M3|M4) ;; *) echo "B3_METHOD must be M1/M2/M3/M4" >&2;exit 2;; esac
  OUT="$OUT_ROOT/$STUDY_NAME"
  MODEL_PATH="$B3/task/${METHOD}_model.joblib"
  "$PY" road_vrp7_step6b3/online_b3.py \
    --control learned --model-method P2 --model-mode F4 --model-protocol task \
    --model-path "$MODEL_PATH" --model-source "$DATA" --output-dir "$OUT" \
    --split "${SPLIT:-validation}" --task-ids "${TASK_IDS:-task_0008,task_0009}" \
    --scenario-ids "${SCENARIO_IDS:-2025-07-25_0800,2025-07-25_1700}" \
    --seeds "$SEEDS" --wall-sec 30 --max-iterations 20000 \
    --anneal-iterations 5000 --top-k 12 --action-count 8 \
    --sampler-seed 20261019 --score-every "${SCORE_EVERY:-8}" \
    --max-scoring-sec "${MAX_SCORING_SEC:-2}" \
    --cache-size 64 --min-static-fraction 0 --fallback c3 ;;
 online-compare)
  : "${STUDY_NAME:?Set a new STUDY_NAME}"
  # If some new arms have not yet finished, omit their --arm args below.
  args=(--base "${BASE_RUN:-runs/chicago_july_v1/step6b2_r/b2r_online_P2_F2_match8_v1}" \
        --m3 "${M3_RUN:-runs/chicago_july_v1/step6b3/b3_online_M3_match8_v1}" \
        --output-dir "$OUT_ROOT/$STUDY_NAME")
  if [[ -n "${M2_RUN:-}" ]]; then args+=(--arm "M2=$M2_RUN"); fi
  if [[ -n "${M4_RUN:-}" ]]; then args+=(--arm "M4=$M4_RUN"); fi
  if [[ -n "${RANDOM_RUN:-}" ]]; then args+=(--arm "random=$RANDOM_RUN"); fi
  if [[ -n "${C3_RUN:-}" ]]; then args+=(--arm "C3=$C3_RUN"); fi
  "$PY" road_vrp7_step6b3_r/online_compare_b3r.py "${args[@]}" ;;
 *) echo 'Usage: bash road_vrp7_step6b3_r/run_b3r.sh test|audit|placebo|online|online-compare' >&2;exit 2 ;;
esac
