#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
STUDY_NAME="${STUDY_NAME:-section_utility_pilot_v1}"
OUTPUT_DIR="runs/chicago_july_v1/step6a/$STUDY_NAME"
SOURCE_DIR="${SOURCE_DIR:-runs/chicago_july_v1/step5b_v3/full30s_1200_v1/30s_anneal5000_cap20000}"
SOLVERS="${SOLVERS:-C3_topk,C0_baseline}"
MAX_STATES="${MAX_STATES:-1000}"
ACTIONS_PER_STATE="${ACTIONS_PER_STATE:-8}"
MAX_TASKS="${MAX_TASKS:-10}"
MAX_SCENARIOS="${MAX_SCENARIOS:-8}"
SNAPSHOT_TIMES="${SNAPSHOT_TIMES:-5,15,30}"
RESUME_ARGS=(); if [[ "${RESUME:-0}" == "1" ]];then RESUME_ARGS=(--resume);fi
case "$MODE" in
  test)
    "$PY" road_vrp7_step6a/tests/test_step6a.py ;;
  plan|collect|smoke)
    if [[ "$MODE" == 'smoke' ]]; then
      STUDY_NAME="${STUDY_NAME_SMOKE:-section_utility_smoke_v1}"
      OUTPUT_DIR="runs/chicago_july_v1/step6a/$STUDY_NAME"
      MAX_STATES=3; MAX_TASKS=2; MAX_SCENARIOS=2; ACTIONS_PER_STATE=4
    fi
    cmd=( "$PY" road_vrp7_step6a/section_dataset.py
          --source-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR"
          --source-solvers "$SOLVERS" --target-times "$SNAPSHOT_TIMES"
          --max-states "$MAX_STATES" --max-tasks "$MAX_TASKS"
          --max-scenarios "$MAX_SCENARIOS" --actions-per-state "$ACTIONS_PER_STATE"
          --candidate-deadline-sec "${CANDIDATE_DEADLINE_SEC:-3}"
          --top-k "${TOP_K:-12}" --lookback-slots "${LOOKBACK_SLOTS:-6}" )
    if [[ "$MODE" == 'plan' ]];then "${cmd[@]}" --dry-run
    else
      "${cmd[@]}" "${RESUME_ARGS[@]}"
      "$PY" road_vrp7_step6a/analyze_step6a.py --experiment-dir "$OUTPUT_DIR"
      "$PY" road_vrp7_step6a/export_step6a.py --experiment-dir "$OUTPUT_DIR"
    fi ;;
  analyze)
    "$PY" road_vrp7_step6a/analyze_step6a.py --experiment-dir "$OUTPUT_DIR"
    "$PY" road_vrp7_step6a/export_step6a.py --experiment-dir "$OUTPUT_DIR" ;;
  *) echo 'Usage: bash road_vrp7_step6a/run_step6a.sh test|plan|smoke|collect|analyze' >&2;exit 2 ;;
esac
