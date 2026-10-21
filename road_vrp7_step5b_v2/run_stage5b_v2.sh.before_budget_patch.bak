#!/usr/bin/env bash
# July development only. Invoke from road_vrp_7 repository root.
set -euo pipefail
STAGE="${1:-test}"
BASE="runs/chicago_july_v1/step5b_v2"
PY="${PYTHON:-python}"
METHODS="${METHODS:-C0_baseline,C1_original_section,C2_noop_guard,C3_topk,C4_time_ucb,C5_gated,C6_shuffled}"
SEEDS="${SEEDS:-42,43,44}"
WALL="${WALL:-5}"
MAX_ITER="${MAX_ITER:-5000}"
ARGS=(--variants "$METHODS" --wall-sec "$WALL" --max-iterations "$MAX_ITER")
RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then RESUME_ARGS+=(--resume); fi
case "$STAGE" in
  test)
    "$PY" road_vrp7_step5b_v2/tests/test_v2_units.py
    "$PY" road_vrp7_step5b_v2/tests/test_runner_mock.py
    "$PY" road_vrp7_step5b_v2/tests/test_analysis.py >/dev/null
    echo 'ALL LOCAL SYNTHETIC TESTS PASSED (not an official ALNS integration test)' ;;
  smoke)
    "$PY" road_vrp7_step5b_v2/run_experiment.py --output-dir "$BASE/smoke" \
      --splits validation --planners oracle --seeds 42 \
      --max-tasks 1 --max-scenarios 1 --wall-sec 0.5 --max-iterations 200 \
      --variants "$METHODS" "${RESUME_ARGS[@]}" ;;
  pilot)
    "$PY" road_vrp7_step5b_v2/run_experiment.py --output-dir "$BASE/pilot" \
      --splits validation --planners oracle --seeds 42,43 \
      --max-tasks 3 --max-scenarios 2 "${ARGS[@]}" "${RESUME_ARGS[@]}" ;;
  validation)
    "$PY" road_vrp7_step5b_v2/run_experiment.py --output-dir "$BASE/validation" \
      --splits validation --planners oracle,static --seeds "$SEEDS" \
      "${ARGS[@]}" "${RESUME_ARGS[@]}" ;;
  july_recent)
    "$PY" road_vrp7_step5b_v2/run_experiment.py --output-dir "$BASE/july_recent_development" \
      --splits test --planners oracle,static --seeds "$SEEDS" \
      "${ARGS[@]}" "${RESUME_ARGS[@]}" ;;
  july_all_development)
    "$PY" road_vrp7_step5b_v2/run_experiment.py --output-dir "$BASE/july_all_development" \
      --splits train,validation,test --planners oracle,static --seeds "$SEEDS" \
      "${ARGS[@]}" "${RESUME_ARGS[@]}" ;;
  july_low_coverage)
    "$PY" road_vrp7_step5b_v2/run_experiment.py --output-dir "$BASE/july_low_coverage" \
      --splits test --include-ineligible \
      --scenario-ids 2025-07-27_0800,2025-07-27_1700 \
      --planners oracle,static --seeds "$SEEDS" \
      "${ARGS[@]}" "${RESUME_ARGS[@]}" ;;
  analyze)
    WHICH="${2:-pilot}"
    "$PY" road_vrp7_step5b_v2/analyze_step5b_v2.py \
      --experiment-dir "$BASE/$WHICH" --output-dir "$BASE/$WHICH/analysis" ;;
  *) echo 'Usage: bash run_stage5b_v2.sh test|smoke|pilot|validation|july_recent|july_all_development|july_low_coverage|analyze [name]' >&2; exit 2 ;;
esac
