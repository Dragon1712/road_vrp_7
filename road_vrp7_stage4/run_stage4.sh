#!/usr/bin/env bash
# Execute from road_vrp_7 repository root.
# Command: bash road_vrp7_stage4/run_stage4.sh test|smoke|pilot|validation|analyze [smoke|pilot|validation]
set -euo pipefail
STAGE=${1:-test}
PYTHON=${PYTHON:-python}
DIR=road_vrp7_stage4
OUT=runs/chicago_july_v1/stage4_budget
WALL=${WALL:-5}
MAX_ITER=${MAX_ITER:-1000000}
FINAL_RATIO=${FINAL_RATIO:-0.005}
METHODS=${METHODS:-sa_timecool,vnd_multistart,alns_timecool}
SEEDS=${SEEDS:-42,43,44}
RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then RESUME_ARGS+=(--resume); fi
mkdir -p "$OUT"
case "$STAGE" in
 test)
   "$PYTHON" "$DIR/tests/test_stage4.py" ;;
 smoke)
   "$PYTHON" "$DIR/budget_experiment.py" --output-dir "$OUT/smoke" \
     --variants "$METHODS" --planners static,oracle --seeds 42 \
     --max-tasks 1 --max-scenarios 1 --wall-sec 0.35 --max-iterations 5000 \
     --final-ratio "$FINAL_RATIO" "${RESUME_ARGS[@]}" ;;
 pilot)
   "$PYTHON" "$DIR/budget_experiment.py" --output-dir "$OUT/pilot" \
     --variants "$METHODS" --planners static,oracle --seeds 42,43 \
     --max-tasks 2 --max-scenarios 2 --wall-sec "$WALL" --max-iterations "$MAX_ITER" \
     --final-ratio "$FINAL_RATIO" "${RESUME_ARGS[@]}" ;;
 validation)
   "$PYTHON" "$DIR/budget_experiment.py" --output-dir "$OUT/validation" \
     --variants "$METHODS" --planners static,oracle --seeds "$SEEDS" \
     --wall-sec "$WALL" --max-iterations "$MAX_ITER" \
     --final-ratio "$FINAL_RATIO" "${RESUME_ARGS[@]}" ;;
 analyze)
   SUB=${2:-validation}
   "$PYTHON" "$DIR/analyze_budget.py" \
     --stage4-dir "$OUT/$SUB" \
     --stage3-dir runs/chicago_july_v1/stage3/validation \
     --output-dir "$OUT/$SUB/analysis" ;;
 *) echo 'Use test|smoke|pilot|validation|analyze [subdir]' >&2; exit 2 ;;
esac
