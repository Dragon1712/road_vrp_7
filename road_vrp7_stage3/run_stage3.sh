#!/usr/bin/env bash
# Execute in road_vrp_7 root; uses ONLY July stage2 outputs.
set -euo pipefail
STAGE="${1:-smoke}"
ROOT=road_vrp7_stage3
COST=datasets/processed/chicago/july_v1/dynamic_costs
TASKS=datasets/processed/chicago/july_v1/tdcvrp30/tasks_10
OUTPUT=runs/chicago_july_v1/stage3
METHODS="${METHODS:-greedy,sa,vnd,official_alns}"
SEEDS="${SEEDS:-42,43,44}"
ITERATIONS="${ITERATIONS:-5000}"
WALL="${WALL:-5}"
RESUME_ARGS=()
if [ "${RESUME:-0}" = 1 ]; then RESUME_ARGS=(--resume); fi
mkdir -p "$OUTPUT"
case "$STAGE" in
 test)
   python "$ROOT/tests/test_stage3.py" ;;
 generate)
   python "$ROOT/tasks.py" --cost-dir "$COST" --output-dir "$TASKS" \
      --scenario-quality runs/chicago_july_v1/data_qc/scenario_quality.csv \
      --num-tasks 10 --num-customers 30 --vehicles 5 --capacity 45 --seed 20261015 ;;
 smoke)
   python "$ROOT/benchmark.py" --cost-dir "$COST" --task-dir "$TASKS" \
     --output-dir "$OUTPUT/smoke" --split train --max-tasks 1 --max-scenarios 1 \
     --methods "$METHODS" --planners static,oracle --seeds 42 \
     --wall-sec 2 --max-iterations 100 --od-hours 5 "${RESUME_ARGS[@]}" ;;
 pilot)
   python "$ROOT/benchmark.py" --cost-dir "$COST" --task-dir "$TASKS" \
     --output-dir "$OUTPUT/pilot" --split train --max-tasks 3 --max-scenarios 2 \
     --methods "$METHODS" --planners static,oracle --seeds 42,43 \
     --wall-sec "$WALL" --max-iterations "$ITERATIONS" --od-hours 5 "${RESUME_ARGS[@]}" ;;
 validation)
   python "$ROOT/benchmark.py" --cost-dir "$COST" --task-dir "$TASKS" \
     --output-dir "$OUTPUT/validation" --split validation \
     --methods "$METHODS" --planners static,oracle --seeds "$SEEDS" \
     --wall-sec "$WALL" --max-iterations "$ITERATIONS" --od-hours 5 "${RESUME_ARGS[@]}" ;;
 testset)
   python "$ROOT/benchmark.py" --cost-dir "$COST" --task-dir "$TASKS" \
     --output-dir "$OUTPUT/test_eligible" --split test \
     --methods "$METHODS" --planners static,oracle --seeds "$SEEDS" \
     --wall-sec "$WALL" --max-iterations "$ITERATIONS" --od-hours 5 "${RESUME_ARGS[@]}" ;;
 stress)
   python "$ROOT/benchmark.py" --cost-dir "$COST" --task-dir "$TASKS" \
     --output-dir "$OUTPUT/test_july27_stress" --split test --include-ineligible \
     --scenario-ids 2025-07-27_0800,2025-07-27_1700 \
     --methods "$METHODS" --planners static,oracle --seeds "$SEEDS" \
     --wall-sec "$WALL" --max-iterations "$ITERATIONS" --od-hours 5 "${RESUME_ARGS[@]}" ;;
 analyse)
   SUB="${2:-validation}"
   python "$ROOT/analyze.py" --metrics "$OUTPUT/$SUB/metrics.csv" \
     --output-dir "$OUTPUT/$SUB/analysis" ;;
 *)
   echo 'Usage: bash road_vrp7_stage3/run_stage3.sh test|generate|smoke|pilot|validation|testset|stress|analyse [subdir]' >&2
   exit 2 ;;
esac
