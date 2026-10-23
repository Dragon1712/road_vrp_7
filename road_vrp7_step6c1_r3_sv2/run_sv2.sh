#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
SV2="road_vrp7_step6c1_r3_sv2"
SV="road_vrp7_step6c1_r3_sv"
BASE="${OUTPUT_ROOT:-runs/chicago_july_v1/step6c1_r3_sv}"
CMD="${1:-help}"
case "$CMD" in
  test)
    "$PY" -m unittest discover -s "$SV2/tests" -v
    "$PY" -m unittest discover -s "$SV/tests" -v
    ;;
  select)
    : "${STUDY_NAME:?Supply NEW selection STUDY_NAME}"
    "$PY" "$SV2/select_states.py" \
      --one-step-dir "$BASE/${ONE_STEP_STUDY:-sv_same_state_v1}" \
      --output-dir "$BASE/$STUDY_NAME" \
      --states "${STATES:-48}" --seed "${SELECT_SEED:-20261024}" \
      --role "${SPLIT_ROLE:-internal_test}" --scheme "${SCHEME:-uniform}"
    ;;
  rollout)
    : "${STUDY_NAME:?Supply NEW rollout STUDY_NAME}"
    : "${ONE_STEP_STUDY:?Supply one-step selection study name}"
    export OUTPUT_ROOT="$BASE"
    # Use frozen SV rollout implementation and ALL its original SHA guards.
    ONE_STEP_STUDY="$ONE_STEP_STUDY" \
    HORIZONS="${HORIZONS:-5,20,50}" REPLICATES="${REPLICATES:-3}" \
    ROAD_REPLAY="${ROAD_REPLAY:-1}" \
    STUDY_NAME="$STUDY_NAME" \
    MAX_STATES="${MAX_STATES:-0}" MAX_NEW_UNITS="${MAX_NEW_UNITS:-0}" \
    RESUME="${RESUME:-0}" WALL_PER_ARM="${WALL_PER_ARM:-30}" \
    bash "$SV/run_sv.sh" rollout
    ;;
  analyze)
    : "${STUDY_NAME:?Supply NEW analysis STUDY_NAME}"
    : "${ROLLOUT_STUDY:?Supply completed rollout STUDY_NAME}"
    : "${ONE_STEP_STUDY:?Supply one-step selection STUDY_NAME}"
    "$PY" "$SV2/analyze_values.py" \
      --rollout-dir "$BASE/$ROLLOUT_STUDY" \
      --one-step-dir "$BASE/$ONE_STEP_STUDY" \
      --output-dir "$BASE/$STUDY_NAME" \
      --bootstrap "${BOOTSTRAPS:-4000}" --seed "${ANALYSIS_SEED:-20261024}"
    ;;
  *) echo 'Usage: run_sv2.sh test|select|rollout|analyze';exit 2;;
esac
