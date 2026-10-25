#!/usr/bin/env bash
# R4-J: strictly new-seed TRAIN-only early-state experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
MODE="${1:-help}"
SRC="${R4J_SOURCE_STUDY:-r4j_train_early_s62s63_v1}"
TASKS="${R4J_TASK_IDS:-task_0000,task_0001,task_0002,task_0003}"
SCENARIOS="${R4J_SCENARIOS:-2025-07-02_0800,2025-07-09_1700}"
SEEDS="${R4J_SEEDS:-62,63}"
MILESTONES="${R4J_MILESTONES:-100,250,500,750,1000,1250,1500,1750}"
ROOT_R4G="${R4G_ROOT:-runs/chicago_july_v1/step6c1_r4g}"
ROOT_PREFIX="${PREFIX_ROOT:-runs/chicago_july_v1/step6c1_r4g_prefix}"
ROOT_CF="${R4H_ROOT:-runs/chicago_july_v1/step6c1_r4h}"
ROOT_J="${R4J_ROOT:-runs/chicago_july_v1/step6c1_r4j}"
PREFIX_NAME="${R4J_PREFIX_STUDY:-${SRC}_prefix_full}"
CF_NAME="${R4J_CF_STUDY:-${SRC}_cf_full}"
CAPTURE="$ROOT_R4G/${SRC}_capture"
PLAIN="$ROOT_R4G/${SRC}_plain"
AUDIT="$ROOT_R4G/${SRC}_audit"
PREFIX_DIR="$ROOT_PREFIX/$PREFIX_NAME"
CF_DIR="$ROOT_CF/$CF_NAME"
NEW_AUDIT="$ROOT_J/${R4J_AUDIT_STUDY:-${SRC}_early_audit_v1}"

fail() { echo "[R4J] ERROR $*" >&2; exit 2; }
check_ids() {
  [[ -n "$TASKS" && -n "$SCENARIOS" && -n "$SEEDS" ]] || fail 'Empty acquisition configuration'
  IFS=',' read -ra aa <<< "$TASKS"
  for a in "${aa[@]}";do
    case "$a" in task_0000|task_0001|task_0002|task_0003) ;; *) fail "TRAIN-only task whitelist: $a";;esac
  done
  IFS=',' read -ra ss <<< "$SEEDS"
  for seed in "${ss[@]}";do
    [[ "$seed" =~ ^[0-9]+$ ]] || fail 'Seed must be an integer'
    [[ "$seed" != 60 && "$seed" != 61 ]] || fail 'R4-J must use NEW seeds (not 60,61)'
  done
  IFS=',' read -ra cs <<< "$SCENARIOS"
  for sid in "${cs[@]}";do
    [[ "$sid" =~ ^2025-07-(0[2-9]|1[0-9]|20|21)_(0800|1700)$ ]] || fail "Only preselected eligible TRAIN July scenarios allowed: $sid"
  done
}
paths() {
  printf 'R4J source=%s\nTasks=%s\nScenarios=%s\nNew seeds=%s\nEarly milestones=%s\nPrefix=%s\nCounterfactual=%s\n' \
    "$SRC" "$TASKS" "$SCENARIOS" "$SEEDS" "$MILESTONES" "$PREFIX_DIR" "$CF_DIR"
}
guard() {
  "$PY" road_vrp7_step6c1_r4j/audit_r4j.py --phase "$1" \
    --capture-dir "$CAPTURE" --plain-dir "$PLAIN" --audit-dir "$AUDIT" \
    --prefix-dir "$PREFIX_DIR" --cf-dir "$CF_DIR" \
    --old-cf-dir "${R4J_OLD_CF_DIR:-runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1}" \
    --output-dir "$NEW_AUDIT"
}
case "$MODE" in
  test)
    "$PY" -m unittest discover -s road_vrp7_step6c1_r4j/tests -p 'test_r4j*.py' -v
    "$PY" -m py_compile road_vrp7_step6c1_r4j/*.py
    bash -n road_vrp7_step6c1_r4j/run_r4j.sh
    ;;
  plan)
    check_ids; paths
    echo '[R4J] Role guard: all source manifests must confirm TRAIN; milestone eligibility is checked again by R4G-Prefix.'
    echo '[R4J] Pilot case count:' "$(( ${#aa[@]} * ${#cs[@]} * ${#ss[@]} ))"
    echo '[R4J] No Validation/Test labels, no wall-clock online gains, no R4-I task score tuning.'
    ;;
  collect-smoke|collect-pilot)
    if [[ "$MODE" == collect-smoke ]];then
      : "${R4J_SOURCE_STUDY:=r4j_train_early_smoke_s62_v1}"
      [[ "$SRC" == r4j_train_early_s62s63_v1 ]] && SRC='r4j_train_early_smoke_s62_v1'
      [[ "$TASKS" == task_0000,task_0001,task_0002,task_0003 ]] && TASKS='task_0002'
      [[ "$SCENARIOS" == 2025-07-02_0800,2025-07-09_1700 ]] && SCENARIOS='2025-07-02_0800'
      [[ "$SEEDS" == 62,63 ]] && SEEDS='62'
    fi
    check_ids
    echo '[R4J] Capture uses original R4G; REQUIRE_STRICT=0 is diagnostic only. R4G-Prefix exact proof is mandatory.' >&2
    paths
    STUDY_NAME="$SRC" R4G_TASK_IDS="$TASKS" R4G_SCENARIO_IDS="$SCENARIOS" \
      R4G_SEEDS="$SEEDS" R4G_SNAPSHOTS="$MILESTONES" R4G_MAX_SNAPSHOTS=8 \
      R4G_MAX_ITERATIONS=20000 REQUIRE_STRICT=0 \
      bash road_vrp7_step6c1_r4g/run_r4g.sh pilot
    # SOURCE STUDY was chosen dynamically for smoke, pass actual paths.
    CAPTURE="$ROOT_R4G/${SRC}_capture";PLAIN="$ROOT_R4G/${SRC}_plain";AUDIT="$ROOT_R4G/${SRC}_audit"
    guard capture
    ;;
  check-capture) guard capture;;
  proof-smoke)
    guard capture
    SOURCE_STUDY="$SRC" STUDY_NAME="${R4J_PROOF_SMOKE_STUDY:-${SRC}_prefix_1case}" \
      HORIZONS=1,50,200,500 REFERENCE_WALL_SEC="${R4J_REFERENCE_WALL_SEC:-90}" \
      bash road_vrp7_step6c1_r4g_prefix/run_prefix.sh smoke
    ;;
  proof-pilot)
    guard capture
    SOURCE_STUDY="$SRC" STUDY_NAME="$PREFIX_NAME" HORIZONS=1,50,200,500 \
      REFERENCE_WALL_SEC="${R4J_REFERENCE_WALL_SEC:-90}" RESUME="${RESUME:-0}" \
      bash road_vrp7_step6c1_r4g_prefix/run_prefix.sh pilot
    guard prefix
    ;;
  check-prefix) guard prefix;;
  cf-smoke|cf-pilot|cf-full)
    guard prefix
    case "$MODE" in
      cf-smoke) cfmode=smoke; name="${R4J_CF_SMOKE_STUDY:-${SRC}_cf_smoke}";;
      cf-pilot) cfmode=pilot; name="${R4J_CF_PILOT_STUDY:-${SRC}_cf_pilot}";;
      cf-full) cfmode=full;name="$CF_NAME";;
    esac
    SOURCE_STUDY="$SRC" PREFIX_DIR="$PREFIX_DIR" STUDY_NAME="$name" \
      R4H_MAX_ALTERNATIVES="${R4J_MAX_ALTERNATIVES:-8}" \
      R4H_MAX_NEW_UNITS="${R4J_MAX_NEW_UNITS:-0}" RESUME="${RESUME:-0}" \
      bash road_vrp7_step6c1_r4h/run_r4h.sh "$cfmode"
    ;;
  audit) guard cf;;
  *) echo 'Usage: bash road_vrp7_step6c1_r4j/run_r4j.sh test|plan|collect-smoke|collect-pilot|check-capture|proof-smoke|proof-pilot|check-prefix|cf-smoke|cf-pilot|cf-full|audit' >&2;exit 2;;
esac
