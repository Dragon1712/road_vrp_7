#!/usr/bin/env bash
# Run from road_vrp_7 root.  Never modifies Stage3 or Stage4 artifacts.
set -euo pipefail
MODE=${1:-test}
DIR=road_vrp7_step5a
BASE=runs/chicago_july_v1/step5a
PYTHON=${PYTHON:-python}
METHODS=${METHODS:-baseline,at_timeonly,at_rr,te_local,te_wr,both,placebo_both}
PLANNERS=${PLANNERS:-static,oracle}
SEEDS=${SEEDS:-42,43,44}
WALL=${WALL:-5}
MAX_ITER=${MAX_ITER:-5000}
RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ];then RESUME_ARGS+=(--resume);fi
case "$MODE" in
  test) "$PYTHON" "$DIR/tests/test_step5a.py" ;;
  smoke)
    "$PYTHON" "$DIR/run_experiment.py" --output-dir "$BASE/smoke" \
      --variants "$METHODS" --planners "$PLANNERS" --seeds 42 \
      --max-tasks 1 --max-scenarios 1 --wall-sec .4 --max-iterations 5000 "${RESUME_ARGS[@]}" ;;
  pilot)
    "$PYTHON" "$DIR/run_experiment.py" --output-dir "$BASE/pilot" \
      --variants "$METHODS" --planners "$PLANNERS" --seeds 42,43 \
      --max-tasks 2 --max-scenarios 2 --wall-sec "$WALL" --max-iterations "$MAX_ITER" "${RESUME_ARGS[@]}" ;;
  validation)
    "$PYTHON" "$DIR/run_experiment.py" --output-dir "$BASE/validation" \
      --variants "$METHODS" --planners "$PLANNERS" --seeds "$SEEDS" \
      --wall-sec "$WALL" --max-iterations "$MAX_ITER" "${RESUME_ARGS[@]}" ;;
  analyze)
    SUB=${2:-pilot}
    REFERENCE=runs/chicago_july_v1/stage3/validation
    if [ "$SUB" = "smoke" ]; then REFERENCE=none; fi
    "$PYTHON" "$DIR/analyze_step5a.py" --stage5-dir "$BASE/$SUB" \
      --output-dir "$BASE/$SUB/analysis" \
      --stage3-dir "$REFERENCE" ;;
  *) echo "Usage: bash $DIR/run_stage5a.sh test|smoke|pilot|validation|analyze [smoke|pilot|validation]" >&2; exit 2 ;;
esac
