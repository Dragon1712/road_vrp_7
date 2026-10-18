#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-test}"
BASE="runs/chicago_july_v1/step5b_v3"
PY="${PYTHON:-python}"
# Each experiment arm is immutable. For any changed parameter, change STUDY_NAME.
STUDY_NAME="${STUDY_NAME:-decoupled_pilot_v1}"
METHODS="${METHODS:-C0_baseline,C3_topk,U0_gain,U1_walltime,U2_calibrated}"
SEEDS="${SEEDS:-42,43}"
TASKS="${MAX_TASKS:-3}"
SCENARIOS="${MAX_SCENARIOS:-2}"
BUDGETS="${BUDGETS:-30,60}"
ANNEAL_LIST="${ANNEAL_LIST:-5000,10000}"
MAX_ITER="${MAX_ITER:-20000}"
PLANNERS="${PLANNERS:-oracle}"
SNAPSHOTS="${SNAPSHOTS:-5,15,30,60}"
CALIBRATION="${CALIBRATION:-road_vrp7_step5b_v3/calibration/section_pair_costs_dev.json}"

case "$MODE" in
  install)
    "$PY" road_vrp7_step5b_v3/install_v3.py ;;
  test)
    "$PY" road_vrp7_step5b_v3/tests/test_v3.py ;;
  calibrate)
    "$PY" road_vrp7_step5b_v3/calibrate_cost.py --output "$CALIBRATION" ;;
  smoke)
    "$PY" road_vrp7_step5b_v3/run_experiment.py \
      --output-dir "$BASE/smoke_v1" --splits validation --planners oracle \
      --variants "$METHODS" --seeds 42 --max-tasks 1 --max-scenarios 1 \
      --wall-sec .35 --max-iterations 200 --anneal-iterations 5000 \
      --snapshot-times .1,.2,.3 --calibration-file "$CALIBRATION" ;;
  pilot|plan)
    IFS=',' read -ra times <<< "$BUDGETS"
    IFS=',' read -ra anneals <<< "$ANNEAL_LIST"
    for wall in "${times[@]}"; do
      for cool in "${anneals[@]}"; do
        path="$BASE/$STUDY_NAME/${wall}s_anneal${cool}_cap${MAX_ITER}"
        printf '\n%s  wall=%ss, anneal=%s, cap=%s\n' "$path" "$wall" "$cool" "$MAX_ITER"
        if [[ "$MODE" == "plan" ]]; then continue; fi
        resume_flag=()
        if [[ "${RESUME:-0}" == "1" ]]; then resume_flag=(--resume); fi
        "$PY" road_vrp7_step5b_v3/run_experiment.py \
          --output-dir "$path" --splits validation --planners "$PLANNERS" \
          --variants "$METHODS" --seeds "$SEEDS" --max-tasks "$TASKS" \
          --max-scenarios "$SCENARIOS" --wall-sec "$wall" \
          --max-iterations "$MAX_ITER" --anneal-iterations "$cool" \
          --snapshot-times "$SNAPSHOTS" --calibration-file "$CALIBRATION" \
          "${resume_flag[@]}"
        "$PY" road_vrp7_step5b_v3/analyze_v3.py --experiment-dir "$path"
      done
    done ;;
  analyze)
    IFS=',' read -ra times <<< "$BUDGETS"
    IFS=',' read -ra anneals <<< "$ANNEAL_LIST"
    for wall in "${times[@]}";do
      for cool in "${anneals[@]}";do
        "$PY" road_vrp7_step5b_v3/analyze_v3.py \
          --experiment-dir "$BASE/$STUDY_NAME/${wall}s_anneal${cool}_cap${MAX_ITER}"
      done
    done ;;
  *) echo "Usage: bash road_vrp7_step5b_v3/run_v3.sh install|test|calibrate|smoke|plan|pilot|analyze" >&2;exit 2;;
esac
