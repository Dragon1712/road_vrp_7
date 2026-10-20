#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-help}"
PY="${PYTHON:-python}"
STUDY_NAME="${STUDY_NAME:-current_static_pilot_v1}"
SOURCE="runs/chicago_july_v1/step6a_v2/current_source_${STUDY_NAME}"
LABELED="runs/chicago_july_v1/step6a_v2/labeled_${STUDY_NAME}"
MAX_TASKS="${MAX_TASKS:-2}"
MAX_SCENARIOS="${MAX_SCENARIOS:-2}"
SEEDS="${SEEDS:-42,43}"
WALL_SEC="${WALL_SEC:-30}"
INTERVAL_SEC="${INTERVAL_SEC:-5}"
MAX_STATES="${MAX_STATES:-1000}"
ACTIONS_PER_STATE="${ACTIONS_PER_STATE:-8}"
if [[ "$MODE" == 'smoke' ]]; then
    STUDY_NAME="${SMOKE_NAME:-current_static_smoke_v1}"
    SOURCE="runs/chicago_july_v1/step6a_v2/current_source_${STUDY_NAME}"
    LABELED="runs/chicago_july_v1/step6a_v2/labeled_${STUDY_NAME}"
    MAX_TASKS=1;MAX_SCENARIOS=1;SEEDS=42;WALL_SEC=3;INTERVAL_SEC=1
    MAX_STATES=10;ACTIONS_PER_STATE=4
fi
# Same wall-clock intervals for capture and label, including short smoke runs.
TIMES="$("$PY" - "$WALL_SEC" "$INTERVAL_SEC" <<'PY'
import sys
w,d=map(float,sys.argv[1:]); n=int(__import__('math').ceil(w/d))
if not 0<n<=100:raise ValueError('Bad slot count')
print(','.join(f'{min(w,d*(k+1)):g}' for k in range(n)))
PY
)"
resume=(); if [[ "${RESUME:-0}" == 1 ]];then resume=(--resume);fi
capture=("$PY" road_vrp7_step6a_v2/capture_current.py
    --output-dir "$SOURCE" --planner static --solver C3_topk
    --max-tasks "$MAX_TASKS" --max-scenarios "$MAX_SCENARIOS" --seeds "$SEEDS"
    --wall-sec "$WALL_SEC" --interval-sec "$INTERVAL_SEC"
    --max-iterations "${MAX_ITER:-20000}" --anneal-iterations "${ANNEAL_ITER:-5000}")
label=("$PY" road_vrp7_step6a_v2/label_current.py
    --source-dir "$SOURCE" --output-dir "$LABELED"
    --source-solvers C3_topk --target-times "$TIMES" --max-states "$MAX_STATES"
    --max-tasks "$MAX_TASKS" --max-scenarios "$MAX_SCENARIOS"
    --actions-per-state "$ACTIONS_PER_STATE" --top-k "${TOP_K:-12}"
    --candidate-deadline-sec "${CANDIDATE_DEADLINE_SEC:-3}")
analysis=("$PY" road_vrp7_step6a_v2/analyze_v2.py --experiment-dir "$LABELED")
case "$MODE" in
  test)
    "$PY" road_vrp7_step6a_v2/tests/test_next_experiments.py ;;
  plan)
    "${capture[@]}" --dry-run
    printf 'label time slots: %s\nlabel source: %s\nlabel output: %s\n' "$TIMES" "$SOURCE" "$LABELED" ;;
  capture)
    "${capture[@]}" "${resume[@]}" ;;
  label)
    "${label[@]}" "${resume[@]}"
    "${analysis[@]}" ;;
  analyze)
    "${analysis[@]}" ;;
  smoke)
    "${capture[@]}" "${resume[@]}"
    "${label[@]}" "${resume[@]}"
    "${analysis[@]}" ;;
  *) echo 'Usage: bash road_vrp7_step6a_v2/run_v2.sh test|plan|smoke|capture|label|analyze' >&2; exit 2 ;;
esac
