#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
BASE="${OUTPUT_ROOT:-runs/chicago_july_v1/step6c1_r3_sv}"
CMD="${1:-help}"
case "$CMD" in
  test) "$PY" -m unittest discover -s road_vrp7_step6c1_r3_sv/tests -v ;;
  preflight) "$PY" road_vrp7_step6c1_r3_sv/preflight_sv.py \
     --dataset-dir "${LABEL_DIR:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}" \
     --r2-model "${R2_MODEL:-runs/chicago_july_v1/step6c1_r2/r2_task_seed42_v1/R2_F2.joblib}" \
     --huber-model "${HUBER_MODEL:-runs/chicago_july_v1/step6c1_r3_tricks/r3_seed42_v1/T2_CB_HUBER.joblib}" \
     --m0-model "${M0_MODEL:-runs/chicago_july_v1/step6b2/b2_full_seed42_v1/task/P2_F2_model.joblib}" ;;
  same-state)
    : "${STUDY_NAME:?Set a NEW STUDY_NAME}"
    "$PY" road_vrp7_step6c1_r3_sv/same_state.py \
      --output-dir "$BASE/$STUDY_NAME" \
      --dataset-dir "${LABEL_DIR:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}" \
      --r2-model "${R2_MODEL:-runs/chicago_july_v1/step6c1_r2/r2_task_seed42_v1/R2_F2.joblib}" \
      --huber-model "${HUBER_MODEL:-runs/chicago_july_v1/step6c1_r3_tricks/r3_seed42_v1/T2_CB_HUBER.joblib}" \
      --m0-model "${M0_MODEL:-runs/chicago_july_v1/step6b2/b2_full_seed42_v1/task/P2_F2_model.joblib}" \
      --max-pilot-states "${PILOT_STATES:-12}" --pilot-role "${PILOT_ROLE:-internal_test}" \
      --pilot-seed "${PILOT_SEED:-20261023}" ;;
  rollout)
    : "${STUDY_NAME:?Set a NEW STUDY_NAME}"
    args=(--one-step-dir "$BASE/${ONE_STEP_STUDY:-sv_same_state_v1}" \
      --output-dir "$BASE/$STUDY_NAME" \
      --dataset-dir "${LABEL_DIR:-runs/chicago_july_v1/step6c1_r2/r2_c3_label_train8x8_v1}" \
      --horizons "${HORIZONS:-5,20,50}" --replicates "${REPLICATES:-2}" \
      --max-states "${MAX_STATES:-0}" \
      --max-new-units "${MAX_NEW_UNITS:-0}" \
      --wall-per-arm "${WALL_PER_ARM:-30}")
    [[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
    [[ "${ROAD_REPLAY:-0}" != 1 ]] || args+=(--road-replay)
    "$PY" road_vrp7_step6c1_r3_sv/rollout_sv.py "${args[@]}" ;;
  *) echo 'Usage: run_sv.sh test|preflight|same-state|rollout';exit 2;;
esac
