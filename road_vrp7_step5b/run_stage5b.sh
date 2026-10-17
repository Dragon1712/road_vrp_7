#!/usr/bin/env bash
# From ~/code/WZL/road_vrp_7. All July dates may be used for DEVELOPMENT.
set -euo pipefail
STAGE=${1:-test}
BASE=runs/chicago_july_v1/step5b
PY=${PYTHON:-python}
METHODS=${METHODS:-baseline,section_random,section_static,section_static_probe,section_traffic,section_shuffled,at_rr_section}
SEEDS=${SEEDS:-42,43,44}
WALL=${WALL:-5}
MAX_ITER=${MAX_ITER:-5000}
RESUME_FLAGS=()
if [ "${RESUME:-0}" = "1" ]; then RESUME_FLAGS+=(--resume); fi
mkdir -p "$BASE"
case "$STAGE" in
 test)
   "$PY" road_vrp7_step5b/tests/test_section.py
   "$PY" road_vrp7_step5b/tests/test_analysis.py ;;
 smoke)
   "$PY" road_vrp7_step5b/run_experiment.py --output-dir "$BASE/smoke" \
     --variants "$METHODS" --splits validation --planners oracle --seeds 42 \
     --wall-sec 0.4 --max-iterations 250 --max-tasks 1 --max-scenarios 1 "${RESUME_FLAGS[@]}" ;;
 pilot)
   "$PY" road_vrp7_step5b/run_experiment.py --output-dir "$BASE/pilot" \
     --variants "$METHODS" --splits validation --planners oracle --seeds 42,43 \
     --wall-sec "$WALL" --max-iterations "$MAX_ITER" --max-tasks 3 --max-scenarios 2 "${RESUME_FLAGS[@]}" ;;
 validation)
   "$PY" road_vrp7_step5b/run_experiment.py --output-dir "$BASE/validation" \
     --variants "$METHODS" --splits validation --planners oracle,static --seeds "$SEEDS" \
     --wall-sec "$WALL" --max-iterations "$MAX_ITER" "${RESUME_FLAGS[@]}" ;;
 july_development)
   "$PY" road_vrp7_step5b/run_experiment.py --output-dir "$BASE/july_development" \
     --variants "$METHODS" --splits train,validation,test --planners oracle,static --seeds "$SEEDS" \
     --wall-sec "$WALL" --max-iterations "$MAX_ITER" "${RESUME_FLAGS[@]}" ;;
 july_recent)
   # July 26,28-31: formerly Stage3 test, now JULY DEVELOPMENT only.
   "$PY" road_vrp7_step5b/run_experiment.py --output-dir "$BASE/july_recent_development" \
     --variants "$METHODS" --splits test --planners oracle,static --seeds "$SEEDS" \
     --wall-sec "$WALL" --max-iterations "$MAX_ITER" "${RESUME_FLAGS[@]}" ;;
 july_stress)
   "$PY" road_vrp7_step5b/run_experiment.py --output-dir "$BASE/july_stress" \
     --variants "$METHODS" --splits test --include-ineligible --scenario-ids 2025-07-27_0800,2025-07-27_1700 --planners oracle,static \
     --seeds "$SEEDS" --wall-sec "$WALL" --max-iterations "$MAX_ITER" "${RESUME_FLAGS[@]}" ;;
 analyze)
   WHICH=${2:-pilot}
   "$PY" road_vrp7_step5b/analyze_step5b.py \
     --experiment-dir "$BASE/$WHICH" --output-dir "$BASE/$WHICH/analysis" ;;
 *)
   echo 'Commands: test|smoke|pilot|validation|july_recent|july_development|july_stress|analyze [subdir]' >&2; exit 2 ;;
esac
