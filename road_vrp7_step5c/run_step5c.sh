#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-help}"
BASE="runs/chicago_july_v1/step5c_v1"
PY="${PYTHON:-python}"
LRW_PROFILE="${LRW_PROFILE:-road_vrp7_step5c/calibration/lrw_from_v3_pilot.json}"
CALIBRATION="${CALIBRATION:-road_vrp7_step5b_v3/calibration/section_pair_costs_dev.json}"
METHODS="${METHODS:-C3_topk,U2_calibrated,C7_alpha_ucb,C8_thompson,C9_lrw}"
case "$MODE" in
 test)
  "$PY" road_vrp7_step5c/tests/test_selectors_step5c.py ;;
 train-lrw)
  "$PY" road_vrp7_step5c/learn_lrw.py \
    --source "${TRAIN_SOURCE:-runs/chicago_july_v1/step5b_v3/decoupled_pilot_v1/30s_anneal5000_cap20000/solutions.jsonl}" \
    --output "$LRW_PROFILE" ;;
 smoke|pilot)
  if [[ "$MODE" == "smoke" ]];then
    study="${STUDY_NAME:-smoke_v1}"
    tasks=1; scenarios='2025-07-23_0800'; seeds=42;wall=.5; snaps=.1,.2,.4
  else
    study="${STUDY_NAME:-step5c_july23_dev_v1}"
    tasks="${MAX_TASKS:-3}"
    scenarios="${SCENARIO_IDS:-2025-07-23_0800,2025-07-23_1700}"
    seeds="${SEEDS:-42,43}"
    wall="${WALL:-30}"
    snaps="${SNAPSHOTS:-5,15,30}"
  fi
  out="$BASE/$study"
  flags=();[[ "${RESUME:-0}" == 1 ]]&&flags=(--resume)
  "$PY" road_vrp7_step5c/run_experiment.py \
    --output-dir "$out" --splits validation --scenario-ids "$scenarios" \
    --planners oracle --variants "$METHODS" --seeds "$seeds" \
    --max-tasks "$tasks" --wall-sec "$wall" \
    --max-iterations 20000 --anneal-iterations 5000 \
    --snapshot-times "$snaps" --top-k 12 \
    --calibration-file "$CALIBRATION" --lrw-profile "$LRW_PROFILE" \
    "${flags[@]}"
  "$PY" road_vrp7_step5c/analyze_step5c.py --experiment-dir "$out" ;;
 analyze)
  "$PY" road_vrp7_step5c/analyze_step5c.py --experiment-dir "$BASE/${STUDY_NAME:-step5c_july23_dev_v1}" ;;
 *) echo "Usage: bash road_vrp7_step5c/run_step5c.sh test|train-lrw|smoke|pilot|analyze" >&2;exit 2;;
esac
